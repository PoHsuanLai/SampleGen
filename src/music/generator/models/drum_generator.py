import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from .base_generator import Generator

class DrumGenerator(Generator):
    """Specialized generator for drum patterns using Meta's MusicGen"""
    
    def __init__(self,
                model_name: str = "facebook/musicgen-small",
                device: str = None,
                sample_rate: int = 32000,
                max_duration: float = 30.0,
                model_dir: str = None):
        """
        Initialize the drum generator with MusicGen model.
        
        Args:
            model_name: Name of the MusicGen model to use
            device: Device to run the model on (None for auto-detection)
            sample_rate: Audio sample rate to generate at
            max_duration: Maximum duration of generated audio in seconds
        """
        # Call parent constructor with drum-focused model
        super().__init__(model_name, device, sample_rate, max_duration, model_dir)
    
    def generate_beat(self,
                    prompt: str,
                    bpm: int = 120,
                    genre: str = "",
                    duration: float = 5.0,
                    temperature: float = 0.95) -> Optional[np.ndarray]:
        """
        Generate a drum beat or pattern with BPM and optional genre specification.
        
        Args:
            prompt: Text description of the drum pattern to generate
            bpm: Beats per minute for the rhythm
            genre: Optional genre for stylistic guidance
            duration: Duration of the generated music in seconds
            temperature: Sampling temperature
            
        Returns:
            Generated audio as a numpy array, or None if generation failed
        """
        # Enhance the prompt with drum-specific terms, BPM, and genre if provided
        if genre:
            enhanced_prompt = f"{genre} drum beat at {bpm} BPM: {prompt}, percussion, rhythm section, drums only"
        else:
            enhanced_prompt = f"Drum beat at {bpm} BPM: {prompt}, percussion, rhythm section, drums only"
        
        # Use the base generation method with drum-optimized parameters
        return self.generate(
            prompt=enhanced_prompt,
            duration=duration,
            temperature=temperature,
            top_k=150,  # Lower top_k for more focused drum patterns
            guidance_scale=4.0  # Higher guidance for better rhythm adherence
        )
    
    def loop_pattern(self, audio_data: np.ndarray, num_loops: int = 4) -> np.ndarray:
        """
        Create a seamless loop of a drum pattern.
        
        Args:
            audio_data: Generated drum pattern
            num_loops: Number of times to loop the pattern
            
        Returns:
            Looped audio data
        """
        try:
            # Simple looping (in a real implementation, beat detection and 
            # seamless crossfading would be used)
            looped_audio = np.tile(audio_data, num_loops)
            
            # Apply some crossfading between segments to avoid clicks
            segment_length = len(audio_data)
            crossfade_length = min(int(segment_length * 0.1), 8000)  # 10% or max 8000 samples
            
            for i in range(1, num_loops):
                pos = i * segment_length
                # Apply crossfade
                for j in range(crossfade_length):
                    ratio = j / crossfade_length
                    looped_audio[pos - crossfade_length + j] = (1 - ratio) * looped_audio[pos - crossfade_length + j] + ratio * looped_audio[pos + j]
            
            return looped_audio
            
        except Exception as e:
            print(f"Error in loop creation: {str(e)}")
            # Fall back to simple tiling if error occurs
            return np.tile(audio_data, num_loops)
    
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