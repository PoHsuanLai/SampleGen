import os
import numpy as np
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union, Any

from music.planner.planner import generate_beat_plan, load_clap_large
from music.generator.models.bass_generator import BassGenerator
from music.generator.models.drum_generator import DrumGenerator
from music.generator.models.other_generator import OtherGenerator
from music.mixer.mixer import MixingModule
from data_processing.stem_extraction import StemExtractor


class SampleGen():
    def __init__(self, stem_model: str = 'htdemucs', device: str = None, 
                sample_rate: int = 44100, max_duration: float = 30.0,
                model_dir: str = None):
        """
        Initialize the SampleGen system that orchestrates the planner, generators, and mixer.
        
        Args:
            stem_model: Stem separation model to use
            device: Device to run models on ('cpu', 'cuda', or None for auto-detect)
            sample_rate: Audio sample rate to work with
            max_duration: Maximum duration of generated audio in seconds
            model_dir: Directory to store downloaded models
        """
        # Auto-detect device if not specified
        if device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.model_dir = model_dir
        
        # Initialize CLAP model for audio understanding
        print("Initializing audio understanding model...")
        self.clap_model, _ = load_clap_large()
        
        # Initialize stem extractor
        print(f"Initializing stem extractor with model {stem_model}...")
        self.stem_extractor = StemExtractor(
            model_name=stem_model,
            device=self.device,
            sample_rate=sample_rate
        )
        
        # Initialize generators
        print("Initializing music generators...")
        self.bass_generator = BassGenerator(device=self.device, max_duration=max_duration, model_dir=model_dir)
        self.drum_generator = DrumGenerator(device=self.device, max_duration=max_duration, model_dir=model_dir)
        self.other_generator = OtherGenerator(device=self.device, max_duration=max_duration, model_dir=model_dir)
        
        # Initialize mixer
        print("Initializing mixing module...")
        self.mixer = MixingModule(sample_rate=sample_rate)
        
        print("SampleGen initialized successfully!")
        
    def generate(self, prompt: str):
        """
        Generate a complete music track based on a text prompt.
        
        Args:
            prompt: Text description of the desired music
            
        Returns:
            Generated audio as a numpy array
        """
        # Generate a basic plan
        plan = self._generate_plan(prompt)
        
        # Generate component tracks
        tracks = self._generate_tracks(plan, prompt)
        
        # Mix the tracks
        mixed_audio = self.mixer.mix_tracks(tracks)
        
        return mixed_audio
    
    def process(self, input_audio_path: str, user_prompt: str, 
               output_audio_path: str, duration: float = 30.0) -> str:
        """
        Process an input audio file, generate a beat based on the prompt and input,
        and save the result to the output path.
        
        Args:
            input_audio_path: Path to input audio file
            user_prompt: Text description of the desired output
            output_audio_path: Path to save the generated audio
            duration: Target duration in seconds
            
        Returns:
            A summary string describing the generation process
        """
        actual_duration = min(duration, self.max_duration)
        
        # Extract stems from the input audio
        print(f"Extracting stems from {input_audio_path}...")
        stems, _ = self.stem_extractor.extract_stems_from_file(
            audio_path=input_audio_path,
            transcribe=False
        )
        
        # Generate a beat plan based on the input audio and prompt
        print("Generating beat plan based on input and prompt...")
        plan = generate_beat_plan(input_audio_path, user_prompt, self.clap_model)
        
        # Generate new tracks based on the plan and input stems
        print("Generating new music components...")
        tracks = {}
        
        # Process existing stems
        for stem_name, stem_audio in stems.items():
            if stem_name == 'drums':
                # Generate new drums based on extracted drums
                tracks['drums'] = self.drum_generator.generate_with_conditioning(
                    prompt=f"{user_prompt} drums",
                    audio=stem_audio,
                    duration=actual_duration
                )
            elif stem_name == 'bass':
                # Generate new bass based on extracted bass
                tracks['bass'] = self.bass_generator.generate_with_conditioning(
                    prompt=f"{user_prompt} bass",
                    audio=stem_audio,
                    duration=actual_duration
                )
            else:
                # Reuse other stems
                tracks[f'sampled_{stem_name}'] = stem_audio
        
        # Generate additional components
        if 'melody' not in plan.lower():
            tracks['melody'] = self.melody_generator.generate(
                prompt=f"{user_prompt} melody",
                duration=actual_duration
            )
            
        if 'harmony' not in plan.lower() and 'chord' not in plan.lower():
            tracks['harmony'] = self.harmony_generator.generate(
                prompt=f"{user_prompt} harmony",
                duration=actual_duration
            )
        
        # If no drums were extracted, create new ones
        if 'drums' not in stems and 'drums' not in tracks:
            tracks['drums'] = self.drum_generator.generate(
                prompt=f"{user_prompt} drums",
                duration=actual_duration
            )
            
        # If no bass was extracted, create new one
        if 'bass' not in stems and 'bass' not in tracks:
            tracks['bass'] = self.bass_generator.generate(
                prompt=f"{user_prompt} bass",
                duration=actual_duration
            )
            
        # Mix all tracks together
        print("Mixing tracks...")
        mixed_audio = self.mixer.mix_tracks(tracks)
        
        # Save the output
        print(f"Saving output to {output_audio_path}...")
        sf.write(output_audio_path, mixed_audio.T, self.sample_rate)
        
        # Generate a summary
        track_names = list(tracks.keys())
        stem_names = list(stems.keys())
        
        summary = f"""
Beat Generation Summary:
-----------------------
Input: {os.path.basename(input_audio_path)}
Prompt: "{user_prompt}"
Duration: {actual_duration:.1f} seconds
Sample Rate: {self.sample_rate} Hz

Extracted stems: {', '.join(stem_names)}
Generated components: {', '.join(n for n in track_names if not n.startswith('sampled_'))}
Reused components: {', '.join(n.replace('sampled_', '') for n in track_names if n.startswith('sampled_'))}

Output saved to: {output_audio_path}
        """
        
        return summary
    
    def _generate_plan(self, prompt: str) -> str:
        """
        Generate a plan for music creation based on the prompt.
        
        Args:
            prompt: Text description of the desired music
            
        Returns:
            A plan string describing the music structure
        """
        # Default plan structure
        default_plan = """
        section: intro
        duration: 4 bars
        drums: simple beat
        bass: minimal
        melody: none
        
        section: verse
        duration: 8 bars
        drums: hiphop beat
        bass: deep 808
        melody: simple
        
        section: chorus
        duration: 4 bars
        drums: full beat
        bass: heavy
        melody: catchy
        """
        
        # For now, return a default plan
        # In a real implementation, this would use a language model to create
        # a personalized plan based on the prompt
        return default_plan
    
    def _generate_tracks(self, plan: str, prompt: str) -> Dict[str, np.ndarray]:
        """
        Generate individual audio tracks based on the plan.
        
        Args:
            plan: Generated plan for the music
            prompt: Original user prompt
            
        Returns:
            Dictionary of named audio tracks
        """
        # Parse the plan (simplified for now)
        has_drums = 'drums' in plan.lower()
        has_bass = 'bass' in plan.lower()
        has_melody = 'melody' in plan.lower()
        has_harmony = 'harmony' in plan.lower() or 'chord' in plan.lower()
        
        # Generate components
        tracks = {}
        
        # Set duration based on plan sections (simplified estimate)
        sections = plan.lower().count('section:')
        # Estimate 4 seconds per section minimum
        duration = max(4.0 * sections, 8.0)
        # Cap at max duration
        actual_duration = min(duration, self.max_duration)
        
        if has_drums:
            tracks['drums'] = self.drum_generator.generate(
                prompt=f"{prompt} drums",
                duration=actual_duration
            )
        
        if has_bass:
            tracks['bass'] = self.bass_generator.generate(
                prompt=f"{prompt} bass",
                duration=actual_duration
            )
        
        if has_melody:
            tracks['melody'] = self.melody_generator.generate(
                prompt=f"{prompt} melody",
                duration=actual_duration
            )
        
        if has_harmony:
            tracks['harmony'] = self.harmony_generator.generate(
                prompt=f"{prompt} harmony",
                duration=actual_duration
            )
            
        return tracks