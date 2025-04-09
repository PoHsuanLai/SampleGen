import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from .base_generator import Generator

class OtherGenerator(Generator):
    """Specialized generator for harmonic content using Meta's MusicGen"""
    
    def __init__(self,
                model_name: str = "facebook/musicgen-small",
                device: str = None,
                sample_rate: int = 32000,
                max_duration: float = 30.0,
                model_dir: str = None):
        """
        Initialize the other generator with MusicGen model.
        
        Args:
            model_name: Name of the MusicGen model to use
            device: Device to run the model on (None for auto-detection)
            sample_rate: Audio sample rate to generate at
            max_duration: Maximum duration of generated audio in seconds
        """
        # Call parent constructor
        super().__init__(model_name, device, sample_rate, max_duration, model_dir)
    
    def generate_chord_progression(self,
                                prompt: str,
                                chord_type: str = "major",
                                duration: float = 5.0,
                                temperature: float = 0.9) -> Optional[np.ndarray]:
        """
        Generate a chord progression with specified chord type.
        
        Args:
            prompt: Text description of the harmony to generate
            chord_type: Type of chords (major, minor, jazz, etc.)
            duration: Duration of the generated music in seconds
            temperature: Sampling temperature
            
        Returns:
            Generated audio as a numpy array, or None if generation failed
        """
        # Enhance the prompt with harmony-specific terms
        enhanced_prompt = f"{chord_type} chord progression: {prompt}, harmonic texture, chords, accompaniment"
        
        # Use the base generation method with harmony-optimized parameters
        return self.generate(
            prompt=enhanced_prompt,
            duration=duration,
            temperature=temperature,
            top_k=200,
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
        Generate harmonic content based on a text prompt.
        
        Args:
            prompt: Text description of the harmony to generate
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
            ).to(self.device)
            
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
            print(f"Error generating harmony: {str(e)}")
            return None
    
    def generate_with_style(self,
                          prompt: str,
                          style: str,
                          duration: float = 5.0,
                          temperature: float = 0.9,
                          top_k: int = 250) -> Optional[np.ndarray]:
        """
        Generate harmonic content with a specific style.
        
        Args:
            prompt: Text description of the harmony to generate
            style: Musical style for the harmony
            duration: Duration of the generated music in seconds
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            
        Returns:
            Generated audio as a numpy array, or None if generation failed
        """
        # Enhance the prompt with style information
        enhanced_prompt = f"{style} chords and harmony: {prompt}"
        
        # Use the base generation method
        return self.generate(
            prompt=enhanced_prompt,
            duration=duration,
            temperature=temperature,
            top_k=top_k,
            guidance_scale=3.5  # Slightly higher guidance for better adherence to style
        )
    
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