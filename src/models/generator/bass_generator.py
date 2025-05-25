import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from .base_generator import Generator

class BassGenerator(Generator):
    """Specialized generator for bass lines using Meta's MusicGen"""
    
    def __init__(self,
                model_name: str = "facebook/musicgen-small",
                device: str = None,
                sample_rate: int = 32000,
                max_duration: float = 30.0,
                model_dir: str = None):
        """
        Initialize the bass generator with MusicGen model.
        
        Args:
            model_name: Name of the MusicGen model to use
            device: Device to run the model on (None for auto-detection)
            sample_rate: Audio sample rate to generate at
            max_duration: Maximum duration of generated audio in seconds
        """
        # Call parent constructor
        super().__init__(model_name, device, sample_rate, max_duration, model_dir)
    
    def generate_bass_line(self,
                          prompt: str,
                          style: str = "deep",
                          bpm: int = 120,
                          duration: float = 5.0,
                          temperature: float = 0.9,
                          key: str = None,
                          reference_audio: np.ndarray = None) -> Optional[np.ndarray]:
        """
        Generate a bass line with specified style and BPM.
        
        Args:
            prompt: Text description of the bass line to generate
            style: Style of bass (deep, synth, funk, etc.)
            bpm: Beats per minute for the rhythm
            duration: Duration of the generated music in seconds
            temperature: Sampling temperature
            key: Musical key for the bass line (optional)
            reference_audio: Reference audio for style matching (optional)
            
        Returns:
            Generated audio as a numpy array, or None if generation failed
        """
        # Enhance the prompt with bass-specific terms
        enhanced_prompt = f"{style} bass line at {bpm} BPM: {prompt}, bass only, low frequency, rhythmic bass"
        
        # Add key information if provided
        if key:
            enhanced_prompt += f", in {key} key"
        
        # Use the base generation method with bass-optimized parameters
        return self.generate(
            prompt=enhanced_prompt,
            duration=duration,
            temperature=temperature,
            top_k=175,
            guidance_scale=3.5
        )
    
    def generate(self, 
                prompt: str,
                duration: float = 5.0,
                temperature: float = 1.0,
                top_k: int = 250,
                top_p: float = 0.9,
                guidance_scale: float = 3.0) -> Optional[np.ndarray]:
        """
        Generate bass content based on a text prompt.
        
        Args:
            prompt: Text description of the bass to generate
            duration: Duration of the generated music in seconds
            temperature: Sampling temperature (higher = more random)
            top_k: Number of highest probability tokens to keep
            top_p: Cumulative probability threshold for tokens
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Generated audio as a numpy array, or None if generation failed
        """
        if self.model is None or self.processor is None:
            print("Model not loaded. Cannot generate music.")
            return None
            
        # Ensure duration is within the model's limits
        actual_duration = min(duration, self.max_duration)
        
        try:
            # Process the text prompt
            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
            # Move inputs to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Calculate max_new_tokens based on duration
            # MusicGen generates 50 tokens per second at 32kHz
            max_new_tokens = int(actual_duration * 50)
            
            # Generate the audio
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                guidance_scale=guidance_scale
            )
            
            # Convert to numpy array
            audio_data = audio_values[0, 0].cpu().numpy()
            
            return audio_data
            
        except Exception as e:
            print(f"Error generating bass: {str(e)}")
            return None
    
    def generate_bassline(self,
                        prompt: str,
                        genre: str = "",
                        duration: float = 5.0,
                        temperature: float = 0.85,
                        top_p: float = 0.92) -> Optional[np.ndarray]:
        """
        Generate a bass line with optional genre specification.
        
        Args:
            prompt: Text description of the bass line to generate
            genre: Optional genre for stylistic guidance
            duration: Duration of the generated music in seconds
            temperature: Sampling temperature
            top_p: Cumulative probability threshold for tokens
            
        Returns:
            Generated audio as a numpy array, or None if generation failed
        """
        # Enhance the prompt with bass-specific terms and genre if provided
        if genre:
            enhanced_prompt = f"{genre} bass line: {prompt}, deep bass, low frequencies, strong rhythm"
        else:
            enhanced_prompt = f"Bass line: {prompt}, deep bass, low frequencies, strong rhythm"
        
        # Use the base generation method with bass-optimized parameters
        return self.generate(
            prompt=enhanced_prompt,
            duration=duration,
            temperature=temperature,
            top_p=top_p,
            guidance_scale=3.0
        )
    
    def _enhance_bass_prompt(self, prompt: str, style: str = "deep") -> str:
        """
        Enhance a prompt with bass-specific terms.
        
        Args:
            prompt: Original prompt
            style: Bass style
            
        Returns:
            Enhanced prompt
        """
        bass_terms = {
            "deep": "deep bass, sub-bass, low frequencies, powerful",
            "synth": "synthesized bass, electronic bass, analog bass",
            "funk": "funky bass, slap bass, groove bass, rhythmic",
            "rock": "rock bass, electric bass, driving bass",
            "jazz": "jazz bass, walking bass, upright bass, smooth"
        }
        
        style_terms = bass_terms.get(style, "bass line, low frequencies")
        return f"{prompt}, {style_terms}, bass only"

    def post_process_bass(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Apply bass-specific post-processing to enhance low frequencies.
        
        Args:
            audio_data: Generated audio data
            
        Returns:
            Processed audio data
        """
        try:
            # Apply a simple low-pass filter (in a real implementation, a more sophisticated
            # approach would be used with scipy, etc.)
            from scipy import signal
            
            # Design a Butterworth low-pass filter to emphasize bass frequencies
            b, a = signal.butter(4, 0.15, 'low')
            
            # Apply the filter
            filtered_audio = signal.filtfilt(b, a, audio_data)
            
            # Mix the filtered audio with the original to retain some higher harmonics
            enhanced_audio = 0.7 * filtered_audio + 0.3 * audio_data
            
            # Normalize
            enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio)) * 0.95
            
            return enhanced_audio
            
        except ImportError:
            print("Warning: scipy not available, skipping bass post-processing")
            return audio_data
        except Exception as e:
            print(f"Error in bass post-processing: {str(e)}")
            return audio_data
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            A dictionary with model information
        """
        if self.model is None:
            return {"status": "Not loaded"}
            
        return {
            "model_name": self.model_name,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "status": "Loaded",
            "parameters": sum(p.numel() for p in self.model.parameters()) / 1_000_000  # Millions
        }
        
    def save_audio(self, audio_data: np.ndarray, output_path: str) -> bool:
        """
        Save audio data to a file.
        
        Args:
            audio_data: Audio data as a numpy array
            output_path: Path to save the audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import soundfile as sf
            sf.write(output_path, audio_data, self.sample_rate)
            return True
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            return False
            
    def load_audio(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Audio data as a numpy array, or None if loading failed
        """
        try:
            import soundfile as sf
            audio_data, _ = sf.read(file_path)
            return audio_data
        except Exception as e:
            print(f"Error loading audio file: {str(e)}")
            return None