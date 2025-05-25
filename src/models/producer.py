"""
Unified Hip-Hop Producer Model that can both request generators and generate Faust mixing scripts.
Removes dependency on the mixer implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import json
import re
import random

# Import existing components
from .generator import BassGenerator, DrumGenerator, OtherGenerator
from ..data_processing.stem_extraction import StemExtractor
from transformers import AutoModel, AutoTokenizer, MistralForCausalLM


class UnifiedProducerModel(nn.Module):
    """
    Unified model that can:
    1. Plan what stems to generate based on text prompts
    2. Request generators to create stems
    3. Generate Faust DSP mixing scripts
    4. Assess quality of the final output
    """
    
    def __init__(self,
                 device: str = 'cuda',
                 sample_rate: int = 44100,
                 text_model_name: str = 'bert-base-uncased',
                 decoder_model_name: str = 'mistralai/Mistral-7B-v0.1',
                 generators_config: Optional[Dict] = None):
        """
        Initialize the Unified Producer Model.
        
        Args:
            device: Device to run the model on
            sample_rate: Audio sample rate
            text_model_name: Text encoder model name
            decoder_model_name: Text generation model name
            generators_config: Configuration for generator models
        """
        super().__init__()
        
        self.device = device
        self.sample_rate = sample_rate
        
        # Initialize text encoder for understanding prompts
        print("Initializing text encoder...")
        self.text_encoder = AutoModel.from_pretrained(text_model_name).to(device)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        
        # Initialize text generator for Faust script generation
        print("Initializing text generator...")
        self.text_generator = MistralForCausalLM.from_pretrained(
            decoder_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.generator_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        if self.generator_tokenizer.pad_token is None:
            self.generator_tokenizer.pad_token = self.generator_tokenizer.eos_token
        
        # Initialize existing generators
        print("Initializing stem generators...")
        config = generators_config or {}
        self.generators = {
            'bass': BassGenerator(device=device, **config.get('bass', {})),
            'drums': DrumGenerator(device=device, **config.get('drums', {})),
            'melody': OtherGenerator(device=device, **config.get('melody', {})),
            'harmony': OtherGenerator(device=device, **config.get('harmony', {}))
        }
        
        # Initialize stem extractor
        print("Initializing stem extractor...")
        self.stem_extractor = StemExtractor(
            model_name='htdemucs',
            device=device,
            sample_rate=sample_rate
        )
        
        # Planning head - maps text embeddings to generation decisions
        text_dim = self.text_encoder.config.hidden_size
        self.planning_head = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # 4 stems: bass, drums, melody, harmony
        ).to(device)
        
        # Quality assessment head
        self.quality_head = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)
        
        # Faust script templates
        self.faust_templates = {
            'volume': 'process = _ : *(gain) with {{ gain = {value}; }};',
            'lowpass': 'import("stdfaust.lib"); process = _ : fi.lowpass(1, {cutoff});',
            'highpass': 'import("stdfaust.lib"); process = _ : fi.highpass(1, {cutoff});',
            'reverb': 'import("stdfaust.lib"); process = _ : re.mono_freeverb(0.5, 0.5, 0.5, 0.5);',
            'delay': 'import("stdfaust.lib"); process = _ : de.delay(ma.SR, {delay_samples}) : *(0.5) :> _;',
            'compressor': 'import("stdfaust.lib"); process = _ : co.compressor_mono(ratio, threshold, attack, release) with {{ ratio = {ratio}; threshold = {threshold}; attack = 0.003; release = 0.1; }};',
            'eq': 'import("stdfaust.lib"); process = _ : fi.peak_eq(gain_low, freq_low, q) : fi.peak_eq(gain_mid, freq_mid, q) : fi.peak_eq(gain_high, freq_high, q) with {{ gain_low = {low_gain}; freq_low = 200; gain_mid = {mid_gain}; freq_mid = 1000; gain_high = {high_gain}; freq_high = 5000; q = 1; }};'
        }
        
        # Enhanced style prompts for different artists
        self.style_prompts = {
            'kendrick': "Conscious rap with complex rhythms, jazz samples, and intricate vocal layering",
            'travis': "Psychedelic trap with auto-tuned vocals, heavy reverb, and atmospheric production",
            'future': "Melodic trap with atmospheric pads, rolling hi-hats, and deep 808 patterns",
            'drake': "Smooth R&B-influenced hip-hop with emotional delivery and polished production",
            'kanye': "Innovative hip-hop with soul samples, creative arrangements, and bold sonic choices",
            'j_cole': "Lyrical hip-hop with live instrumentation, warm tones, and authentic production",
            'denzel_curry': "Aggressive rap with hard-hitting drums, distorted elements, and raw energy",
            'lil_uzi': "Melodic rap with ethereal synths, modern trap production, and catchy hooks",
            'pop_smoke': "Brooklyn drill with menacing bass, hard drums, and gritty urban atmosphere"
        }
    
    def plan_production(self, 
                       text_prompt: str,
                       audio_file_path: Optional[str] = None,
                       audio_stems: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Plan music production based on text prompt and optional audio input.
        
        Args:
            text_prompt: User's description of desired output
            audio_file_path: Path to input audio file
            audio_stems: Pre-extracted audio stems
            
        Returns:
            Production plan containing generation prompts and mixing strategy
        """
        # Extract stems if needed
        if audio_stems is None and audio_file_path:
            print("Extracting stems from input audio...")
            audio_stems, _ = self.stem_extractor.extract_stems_from_file(audio_file_path)
        
        # Encode text prompt
        with torch.no_grad():
            inputs = self.text_tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_embedding = self.text_encoder(**inputs).last_hidden_state.mean(dim=1)
            
            # Get planning decisions
            planning_logits = self.planning_head(text_embedding)
            planning_probs = torch.sigmoid(planning_logits).cpu().numpy()[0]
        
        # Create generation plan based on probabilities and existing stems
        existing_stems = list(audio_stems.keys()) if audio_stems else []
        plan = {}
        
        stem_names = ['bass', 'drums', 'melody', 'harmony']
        for i, stem_name in enumerate(stem_names):
            # Generate if probability > 0.5 and stem doesn't exist or needs enhancement
            should_generate = planning_probs[i] > 0.5 or stem_name not in existing_stems
            
            if should_generate:
                plan[f'generate_{stem_name}'] = self._create_generation_prompt(text_prompt, stem_name)
        
        # Add mixing strategy
        plan['mixing_strategy'] = self._create_mixing_strategy(text_prompt, existing_stems)
        
        return plan
    
    def generate_stems(self, 
                      plan: Dict[str, Any],
                      duration: float = 5.0,
                      conditioning_stems: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        Generate stems using the generators based on the plan.
        
        Args:
            plan: Production plan from plan_production
            duration: Duration for generated stems
            conditioning_stems: Optional stems to condition on
            
        Returns:
            Dictionary of generated stems
        """
        generated_stems = {}
        
        # Generate bass if requested
        if plan.get('generate_bass'):
            print(f"Generating bass: {plan['generate_bass']}")
            bass_audio = self.generators['bass'].generate_bass_line(
                prompt=plan['generate_bass'],
                duration=duration
            )
            if bass_audio is not None:
                generated_stems['bass'] = bass_audio
        
        # Generate drums if requested  
        if plan.get('generate_drums'):
            print(f"Generating drums: {plan['generate_drums']}")
            drums_audio = self.generators['drums'].generate(
                prompt=plan['generate_drums'],
                duration=duration
            )
            if drums_audio is not None:
                generated_stems['drums'] = drums_audio
        
        # Generate melody if requested
        if plan.get('generate_melody'):
            print(f"Generating melody: {plan['generate_melody']}")
            melody_audio = self.generators['melody'].generate(
                prompt=plan['generate_melody'],
                duration=duration
            )
            if melody_audio is not None:
                generated_stems['melody'] = melody_audio
        
        # Generate harmony if requested
        if plan.get('generate_harmony'):
            print(f"Generating harmony: {plan['generate_harmony']}")
            harmony_audio = self.generators['harmony'].generate(
                prompt=plan['generate_harmony'],
                duration=duration
            )
            if harmony_audio is not None:
                generated_stems['harmony'] = harmony_audio
        
        return generated_stems
    
    def generate_faust_script(self,
                             stems: Dict[str, np.ndarray],
                             text_prompt: str,
                             mixing_strategy: Optional[str] = None) -> str:
        """
        Generate a Faust DSP script for mixing the stems.
        
        Args:
            stems: Dictionary of audio stems
            text_prompt: Original user prompt for context
            mixing_strategy: Optional mixing strategy from planning
            
        Returns:
            Faust DSP script as a string
        """
        # Create prompt for Faust script generation
        stem_list = list(stems.keys())
        faust_prompt = self._create_faust_prompt(text_prompt, stem_list, mixing_strategy)
        
        # Generate Faust script using text generator
        with torch.no_grad():
            inputs = self.generator_tokenizer(
                faust_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(self.text_generator.device) for k, v in inputs.items()}
            
            outputs = self.text_generator.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.generator_tokenizer.eos_token_id
            )
            
            generated_text = self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        # Extract Faust script from generated text
        faust_script = self._extract_faust_script(generated_text, stems)
        
        return faust_script
    
    def assess_quality(self,
                      final_audio: np.ndarray,
                      text_prompt: str) -> float:
        """
        Assess the quality of the final mixed audio.
        
        Args:
            final_audio: Final mixed audio
            text_prompt: Original user prompt for context
            
        Returns:
            Quality score between 0 and 1
        """
        # Encode text prompt
        with torch.no_grad():
            inputs = self.text_tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_embedding = self.text_encoder(**inputs).last_hidden_state.mean(dim=1)
            
            # Get quality score
            quality_score = self.quality_head(text_embedding).cpu().numpy()[0, 0]
        
        # Add some audio-based quality metrics
        audio_quality = self._compute_audio_quality_metrics(final_audio)
        
        # Combine text-based and audio-based quality
        final_quality = 0.7 * quality_score + 0.3 * audio_quality
        
        return float(final_quality)
    
    def _create_generation_prompt(self, text_prompt: str, stem_name: str) -> str:
        """Create a generation prompt for a specific stem."""
        # Extract style information from text prompt
        style_info = self._extract_style_info(text_prompt)
        
        # Create stem-specific prompts
        if stem_name == 'bass':
            return f"{style_info} bass line with deep 808s and sub-bass frequencies"
        elif stem_name == 'drums':
            return f"{style_info} drum pattern with hard-hitting kicks and crisp snares"
        elif stem_name == 'melody':
            return f"{style_info} melodic elements with catchy hooks and lead sounds"
        elif stem_name == 'harmony':
            return f"{style_info} harmonic background with pads and atmospheric textures"
        else:
            return f"{style_info} {stem_name} track"
    
    def _create_mixing_strategy(self, text_prompt: str, existing_stems: List[str]) -> str:
        """Create a mixing strategy based on the prompt and available stems."""
        style_info = self._extract_style_info(text_prompt)
        
        strategies = [
            f"Create a balanced mix emphasizing {style_info} characteristics",
            f"Apply {style_info} mixing techniques with proper stem separation",
            f"Use {style_info} production style with modern mixing standards"
        ]
        
        return random.choice(strategies)
    
    def _extract_style_info(self, text_prompt: str) -> str:
        """Extract style information from text prompt."""
        text_lower = text_prompt.lower()
        
        # Check for artist mentions
        for artist, style in self.style_prompts.items():
            if artist in text_lower:
                return style
        
        # Check for genre mentions
        if 'trap' in text_lower:
            return "trap"
        elif 'drill' in text_lower:
            return "drill"
        elif 'boom bap' in text_lower:
            return "boom bap"
        elif 'conscious' in text_lower:
            return "conscious rap"
        else:
            return "hip-hop"
    
    def _create_faust_prompt(self, text_prompt: str, stem_list: List[str], mixing_strategy: Optional[str]) -> str:
        """Create a prompt for Faust script generation."""
        prompt = f"""Generate a Faust DSP script for mixing hip-hop stems.

User request: {text_prompt}
Available stems: {', '.join(stem_list)}
Mixing strategy: {mixing_strategy or 'Standard hip-hop mixing'}

Create a Faust script that:
1. Processes each stem appropriately
2. Applies EQ, compression, and effects as needed
3. Creates a balanced final mix
4. Uses proper Faust syntax

Faust script:"""
        
        return prompt
    
    def _extract_faust_script(self, generated_text: str, stems: Dict[str, np.ndarray]) -> str:
        """Extract and validate Faust script from generated text."""
        # Look for Faust code patterns
        lines = generated_text.split('\n')
        faust_lines = []
        in_faust_block = False
        
        for line in lines:
            line = line.strip()
            if 'import(' in line or 'process =' in line or line.startswith('//'):
                in_faust_block = True
            
            if in_faust_block:
                faust_lines.append(line)
                
            # Stop if we hit a clear end
            if line.endswith('};') and in_faust_block:
                break
        
        if faust_lines:
            return '\n'.join(faust_lines)
        else:
            # Fallback to template-based script
            return self._generate_template_faust_script(stems)
    
    def _generate_template_faust_script(self, stems: Dict[str, np.ndarray]) -> str:
        """Generate a template-based Faust script as fallback."""
        script_parts = ['import("stdfaust.lib");']
        
        # Create processing for each stem
        for i, stem_name in enumerate(stems.keys()):
            if stem_name == 'bass':
                script_parts.append(f'bass_process = _ : fi.highpass(1, 40) : co.compressor_mono(4, -20, 0.003, 0.1);')
            elif stem_name == 'drums':
                script_parts.append(f'drums_process = _ : fi.peak_eq(2, 100, 1) : co.compressor_mono(3, -15, 0.001, 0.05);')
            elif stem_name in ['melody', 'harmony']:
                script_parts.append(f'{stem_name}_process = _ : fi.peak_eq(1.5, 2000, 0.7) : re.mono_freeverb(0.3, 0.5, 0.5, 0.5);')
        
        # Main process
        script_parts.append('process = _, _, _, _ : bass_process, drums_process, melody_process, harmony_process :> _;')
        
        return '\n'.join(script_parts)
    
    def _compute_audio_quality_metrics(self, audio: np.ndarray) -> float:
        """Compute basic audio quality metrics."""
        if len(audio) == 0:
            return 0.0
        
        # RMS level
        rms = np.sqrt(np.mean(audio**2))
        
        # Peak level
        peak = np.max(np.abs(audio))
        
        # Dynamic range (simplified)
        if peak > 0:
            dynamic_range = rms / peak
        else:
            dynamic_range = 0
        
        # Combine metrics (normalized to 0-1)
        quality = min(1.0, (rms * 10 + dynamic_range) / 2)
        
        return quality
    
    def create_mix(self,
                  original_stems: Dict[str, np.ndarray],
                  generated_stems: Dict[str, np.ndarray],
                  text_prompt: str) -> np.ndarray:
        """
        Create final mix by combining original and generated stems.
        
        Args:
            original_stems: Original stems from user input
            generated_stems: Newly generated stems
            text_prompt: Original user prompt for context
            
        Returns:
            Mixed audio
        """
        # Combine all stems
        all_stems = {**original_stems, **generated_stems}
        
        # Generate Faust script
        faust_script = self.generate_faust_script(all_stems, text_prompt)
        
        # For now, create a simple mix (in practice, you'd apply the Faust script)
        mixed_audio = self._simple_mix(all_stems)
        
        return mixed_audio
    
    def _simple_mix(self, stems: Dict[str, np.ndarray]) -> np.ndarray:
        """Create a simple mix of stems."""
        if not stems:
            return np.array([])
        
        # Find the maximum length
        max_length = max(len(stem) for stem in stems.values())
        
        # Mix stems with appropriate levels
        mixed = np.zeros(max_length)
        stem_weights = {'bass': 0.8, 'drums': 0.9, 'melody': 0.6, 'harmony': 0.5}
        
        for stem_name, audio in stems.items():
            # Pad or truncate to match length
            if len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)))
            else:
                audio = audio[:max_length]
            
            # Apply weight
            weight = stem_weights.get(stem_name, 0.7)
            mixed += audio * weight
        
        # Normalize
        if np.max(np.abs(mixed)) > 0:
            mixed = mixed / np.max(np.abs(mixed)) * 0.8
        
        return mixed
    
    def iterative_refinement(self,
                           original_stems: Dict[str, np.ndarray],
                           generated_stems: Dict[str, np.ndarray],
                           text_prompt: str,
                           max_iterations: int = 3,
                           quality_threshold: float = 0.8) -> Tuple[np.ndarray, float]:
        """
        Iteratively refine the mix until quality threshold is met.
        
        Args:
            original_stems: Original stems from user input
            generated_stems: Newly generated stems
            text_prompt: Original user prompt for context
            max_iterations: Maximum number of refinement iterations
            quality_threshold: Target quality score
            
        Returns:
            Tuple of (final_mix, final_quality_score)
        """
        current_mix = self.create_mix(original_stems, generated_stems, text_prompt)
        current_quality = self.assess_quality(current_mix, text_prompt)
        
        for iteration in range(max_iterations):
            print(f"Refinement iteration {iteration + 1}: Quality = {current_quality:.3f}")
            
            if current_quality >= quality_threshold:
                print(f"Quality threshold reached: {current_quality:.3f}")
                break
            
            # Apply refinements
            refined_mix = self._apply_refinements(current_mix, text_prompt)
            refined_quality = self.assess_quality(refined_mix, text_prompt)
            
            if refined_quality > current_quality:
                current_mix = refined_mix
                current_quality = refined_quality
            else:
                print("No improvement in this iteration")
        
        return current_mix, current_quality
    
    def _apply_refinements(self, audio: np.ndarray, text_prompt: str) -> np.ndarray:
        """Apply simple refinements to the audio."""
        # Simple refinements: normalization and light compression simulation
        refined = audio.copy()
        
        # Normalize
        if np.max(np.abs(refined)) > 0:
            refined = refined / np.max(np.abs(refined)) * 0.85
        
        # Simple compression simulation (reduce dynamic range)
        threshold = 0.7
        ratio = 3.0
        above_threshold = np.abs(refined) > threshold
        refined[above_threshold] = np.sign(refined[above_threshold]) * (
            threshold + (np.abs(refined[above_threshold]) - threshold) / ratio
        )
        
        return refined 