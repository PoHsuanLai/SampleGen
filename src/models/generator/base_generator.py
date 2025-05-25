import os
import torch
import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional, Union, Any
from transformers import AutoProcessor, MusicgenForConditionalGeneration

try:
    import soundfile as sf
except ImportError:
    sf = None

class Generator:
    """Base Generator class for music generation using Meta's MusicGen"""
    
    def __init__(self,
                model_name: str = "facebook/musicgen-small",
                device: str = None,
                sample_rate: int = 32000,
                max_duration: float = 30.0,
                model_dir: str = None):
        """
        Initialize the base generator with MusicGen model.
        
        Args:
            model_name: Name of the MusicGen model to use
            device: Device to run the model on (None for auto-detection)
            sample_rate: Audio sample rate to generate at
            max_duration: Maximum duration of generated audio in seconds
            model_dir: Directory to store downloaded models (None for default 'models' directory)
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        
        # Set the models directory
        if model_dir is None:
            # Default to 'models' directory in the project root
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            self.model_dir = os.path.join(root_dir, "models")
        else:
            self.model_dir = model_dir
            
        # Ensure the models directory exists
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load the model
        self._init_model()
    
    def _init_model(self) -> None:
        """
        Initialize the MusicGen model. If model has been downloaded,
        load it directly from disk. Otherwise, download it first.
        """
        # Create a model-specific subdirectory (replace / with _ in model name)
        model_folder_name = self.model_name.replace('/', '_')
        model_dir = os.path.join(self.model_dir, model_folder_name)
        
        # Ensure model directory exists
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if the model has already been downloaded
        model_config_path = os.path.join(model_dir, "config.json")
        model_downloaded = os.path.exists(model_config_path)
        
        print(f"Loading MusicGen model '{self.model_name}' on {self.device}...")
        
        try:
            if model_downloaded:
                # Model exists locally - load it from disk
                print(f"Loading model from local directory: {model_dir}")
                
                self.processor = AutoProcessor.from_pretrained(
                    model_dir,
                    local_files_only=True
                )
                
                self.model = MusicgenForConditionalGeneration.from_pretrained(
                    model_dir,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    local_files_only=True
                )
                
                # Set generation config from local files if needed
                if not hasattr(self.model, "generation_config"):
                    from transformers import GenerationConfig
                    self.model.generation_config = GenerationConfig.from_pretrained(
                        model_dir,
                        local_files_only=True
                    )
            else:
                # Download the model from Hugging Face and save it to model_dir
                print(f"Downloading model from Hugging Face: {self.model_name}")
                print(f"This will be saved to: {model_dir}")
                
                # Download and save processor
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.processor.save_pretrained(model_dir)
                
                # Download and save model
                self.model = MusicgenForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.model.save_pretrained(model_dir)
                
                # Save generation config
                if not hasattr(self.model, "generation_config"):
                    from transformers import GenerationConfig
                    self.model.generation_config = GenerationConfig.from_pretrained(self.model_name)
                    self.model.generation_config.save_pretrained(model_dir)
                
                print(f"Model downloaded and saved to {model_dir}")
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Save model info
            self._save_model_info(model_dir)
                
            print(f"{self.__class__.__name__} model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading MusicGen model: {str(e)}")
            self.model = None
            self.processor = None
    
    def _save_model_info(self, model_dir: str) -> None:
        """
        Save model metadata to the model directory.
        
        Args:
            model_dir: Path to the model directory
        """
        try:
            metadata = {
                "model_name": self.model_name,
                "model_type": self.__class__.__name__,
                "sample_rate": self.sample_rate,
                "last_used": self._get_current_timestamp(),
                "device": self.device,
                "parameters": f"{sum(p.numel() for p in self.model.parameters()) / 1_000_000:.2f}M"
            }
            
            metadata_path = os.path.join(model_dir, "metadata.yaml")
            with open(metadata_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=False)
                
        except Exception as e:
            print(f"Error saving model metadata: {str(e)}")
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as a string"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def generate(self, 
                prompt: str,
                duration: float = 5.0,
                temperature: float = 1.0,
                top_k: int = 250,
                top_p: float = 0.9,
                guidance_scale: float = 3.0) -> Optional[np.ndarray]:
        """
        Generate music content based on a text prompt.
        
        Args:
            prompt: Text description of the music to generate
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
            print(f"Error generating content: {str(e)}")
            return None
    
    def generate_with_conditioning(self,
                                prompt: str,
                                audio: Union[np.ndarray, str],
                                duration: float = 5.0,
                                temperature: float = 1.0,
                                top_k: int = 250,
                                guidance_scale: float = 3.0) -> Optional[np.ndarray]:
        """
        Generate music conditioned on both text and an existing audio sample.
        
        Args:
            prompt: Text description of the music to generate
            audio: Either a numpy array containing audio data or a path to an audio file
            duration: Duration of the generated music in seconds
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep
            guidance_scale: Scale for classifier-free guidance
            
        Returns:
            Generated audio as a numpy array, or None if generation failed
        """
        if self.model is None or self.processor is None:
            print("Model not loaded. Cannot generate music.")
            return None
            
        # Load audio if it's a file path
        if isinstance(audio, str):
            audio = self.load_audio(audio)
            if audio is None:
                return None
                
        try:
            # Process the inputs
            inputs = self.processor(
                text=[prompt],
                audio=audio,
                sampling_rate=self.sample_rate,
                padding=True,
                return_tensors="pt"
            )
            # Move inputs to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Calculate max_new_tokens based on duration
            max_new_tokens = int(duration * 50)
            
            # Generate the audio
            audio_values = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                guidance_scale=guidance_scale
            )
            
            # Convert to numpy array
            audio_data = audio_values[0, 0].cpu().numpy()
            
            return audio_data
            
        except Exception as e:
            print(f"Error generating with conditioning: {str(e)}")
            return None
    
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
            "max_duration": self.max_duration,
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
            if sf is None:
                raise ImportError("soundfile is required for audio saving")
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
            if sf is None:
                raise ImportError("soundfile is required for audio loading")
            audio_data, _ = sf.read(file_path)
            return audio_data
        except Exception as e:
            print(f"Error loading audio file: {str(e)}")
            return None 