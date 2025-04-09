import os
import torch
import numpy as np
from typing import Dict, Optional, List, Union, Tuple
import soundfile as sf
from demucs.pretrained import get_model
from demucs.apply import apply_model
import tempfile
from transcription import StemTranscriber

class StemExtractor:
    """
    Module for separating audio tracks into individual stems using Meta's Demucs.
    """
    
    def __init__(self, model_name: str = 'htdemucs', device: str = None, 
                transcribe_stems: bool = False, sample_rate: int = 44100):
        """
        Initialize the StemExtractor with Demucs.
        
        Args:
            model_name: The Demucs model to use. Options include:
                        'htdemucs' - The latest hybrid transformer model
                        'htdemucs_ft' - Fine-tuned version
                        'htdemucs_6s' - 6-source version (vocals, drums, bass, guitar, piano, other)
                        'mdx_extra' - MDX model with more separation quality
                        'mdx_extra_q' - Quantized version of mdx_extra
            device: Device to use for inference ('cpu', 'cuda', or None for auto-detect)
            transcribe_stems: Whether to automatically transcribe stems to MIDI
            sample_rate: Audio sample rate to work with
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.transcribe_stems = transcribe_stems
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Loading Demucs model '{model_name}' on {self.device}...")
        self.model = get_model(model_name)
        self.model.to(self.device)
        
        # Get the source names from the model
        self.source_names = self.model.sources
        print(f"Model loaded with sources: {', '.join(self.source_names)}")
        
        # Initialize the transcriber if needed
        if transcribe_stems:
            self.transcriber = StemTranscriber(sample_rate=sample_rate)
        else:
            self.transcriber = None
    
    def extract_stems(self, audio_data: np.ndarray, sample_rate: int = 44100) -> Dict[str, np.ndarray]:
        """
        Separate the audio into stems using Demucs.
        
        Args:
            audio_data: Audio data as a numpy array (channels, samples)
            sample_rate: Sample rate of the audio
            
        Returns:
            stems: Dictionary containing separated audio components
        """
        # Ensure the audio is in the right format for Demucs (channels, samples)
        if len(audio_data.shape) == 1:
            # Convert mono to stereo by duplicating the channel
            audio_data = np.stack([audio_data, audio_data])
        elif audio_data.shape[0] > 2:
            # If more than 2 channels, just take the first two
            audio_data = audio_data[:2]
        
        # Convert to torch tensor
        audio_tensor = torch.tensor(audio_data, dtype=torch.float32).to(self.device)
        
        # Demucs expects the tensor shape as (batch, channels, samples)
        audio_tensor = audio_tensor.unsqueeze(0)
        
        # Resample if necessary (Demucs expects 44.1kHz)
        if sample_rate != 44100:
            print(f"Resampling from {sample_rate}Hz to 44100Hz")
            resample_ratio = 44100 / sample_rate
            new_length = int(audio_tensor.shape[-1] * resample_ratio)
            audio_tensor = torch.nn.functional.interpolate(
                audio_tensor, size=new_length, mode='linear', align_corners=False
            )
        
        # Apply the separation model
        with torch.no_grad():
            sources = apply_model(self.model, audio_tensor, shifts=1, split=True, overlap=0.25, progress=True)
        
        # Convert back to numpy
        stems = {}
        for source_idx, source_name in enumerate(self.source_names):
            source_audio = sources[0, source_idx].cpu().numpy()
            stems[source_name] = source_audio
        
        return stems
    
    def extract_stems_from_file(self, audio_path: str, output_dir: Optional[str] = None,
                               transcribe: Optional[bool] = None, midi_output_dir: Optional[str] = None) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, str]]]:
        """
        Load an audio file and separate it into stems, optionally saving each stem and transcribing to MIDI.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Optional directory to save the extracted stems
            transcribe: Whether to transcribe stems to MIDI (overrides constructor setting)
            midi_output_dir: Optional directory to save the MIDI files (defaults to output_dir/midi)
            
        Returns:
            A tuple containing:
                - Dictionary containing separated audio components
                - Dictionary containing paths to MIDI files (if transcription enabled)
        """
        # Create a temporary file directory if not specified
        if output_dir is None:
            temp_dir = tempfile.mkdtemp()
            output_dir = temp_dir
        elif not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Determine whether to transcribe
        do_transcribe = self.transcribe_stems if transcribe is None else transcribe
        
        # Set MIDI output directory
        if do_transcribe and midi_output_dir is None:
            midi_output_dir = os.path.join(output_dir, "midi")
            os.makedirs(midi_output_dir, exist_ok=True)
        
        # Load the model in inference mode
        with torch.no_grad():
            # Demucs provides a convenient way to separate audio files directly
            from demucs.separate import main as demucs_separate
            import sys
            
            # Prepare arguments for the demucs separator
            original_args = sys.argv
            sys.argv = [
                "demucs", 
                "-n", self.model_name,
                "-o", output_dir,
                "--two-stems=vocals" if len(self.source_names) == 2 else "",
                audio_path
            ]
            
            # Run the separator
            try:
                demucs_separate()
            except SystemExit:
                # Demucs calls sys.exit(), so catch it
                pass
            
            # Restore original args
            sys.argv = original_args
        
        # Load the separated stems
        stems = {}
        model_dir = os.path.join(output_dir, self.model_name)
        
        if not os.path.exists(model_dir):
            # Some demucs versions may not create a subdirectory with the model name
            model_dir = output_dir
        
        # Get the filename without extension
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Load each stem
        for source_name in self.source_names:
            stem_path = os.path.join(model_dir, filename, f"{source_name}.wav")
            if os.path.exists(stem_path):
                # Load the audio file
                stem_audio, sr = sf.read(stem_path, always_2d=True)
                # Convert to (channels, samples) format
                stem_audio = stem_audio.T
                stems[source_name] = stem_audio
            else:
                print(f"Warning: Stem file {stem_path} not found!")
        
        # Transcribe stems if requested
        midi_paths = None
        if do_transcribe and stems and hasattr(self, 'transcriber') and self.transcriber is not None:
            print(f"Transcribing stems to MIDI in {midi_output_dir}...")
            midi_paths = self.transcriber.transcribe_all_stems(stems, midi_output_dir)
        
        return stems, midi_paths
    
    def get_available_models(self) -> List[str]:
        """
        Get a list of available pre-trained Demucs models.
        
        Returns:
            A list of available model names
        """
        # This is a static list of the most common models
        # For the most up-to-date list, one would check the Demucs repository
        return [
            'htdemucs',       # Hybrid Transformer Demucs (default)
            'htdemucs_ft',    # Fine-tuned version
            'htdemucs_6s',    # 6-source version
            'mdx',            # Music Demixing model
            'mdx_extra',      # Enhanced MDX
            'mdx_extra_q',    # Quantized version
        ]
        
    def get_model_sources(self) -> List[str]:
        """
        Get the list of source names for the current model.
        
        Returns:
            A list of source names (e.g., ['vocals', 'drums', 'bass', 'other'])
        """
        return self.source_names
        
    def generate_stem(self, 
                     prompt: str, 
                     stem_type: str, 
                     duration: float = 5.0,
                     output_path: Optional[str] = None,
                     transcribe: bool = False,
                     midi_output_path: Optional[str] = None,
                     models_dir: Optional[str] = None) -> Tuple[np.ndarray, Optional[str]]:
        """
        Generate a stem using MusicGen based on a text prompt.
        
        Args:
            prompt: Text description of the stem to generate
            stem_type: Type of stem to generate (e.g., 'bass', 'drums', 'melody')
            duration: Duration of the generated stem in seconds
            output_path: Optional path to save the generated stem
            transcribe: Whether to transcribe the generated stem to MIDI
            midi_output_path: Optional path to save the MIDI file
            models_dir: Optional directory for storing models (defaults to project_root/models)
            
        Returns:
            A tuple containing:
                - Generated audio as a numpy array
                - Path to the MIDI file (if transcribed)
        """
        from src.music.generator import MelodyGenerator, BassGenerator, DrumGenerator, HarmonyGenerator
        
        # Set default models directory if not provided
        if models_dir is None:
            # Default to 'models' directory in the project root
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            models_dir = os.path.join(root_dir, "models")
            
        # Ensure the directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Choose the appropriate generator based on stem type
        if stem_type.lower() == 'bass':
            generator = BassGenerator(models_dir=models_dir)
            audio = generator.generate_bassline(prompt, duration=duration)
        elif stem_type.lower() in ['drums', 'percussion', 'beat']:
            generator = DrumGenerator(models_dir=models_dir)
            audio = generator.generate(prompt, duration=duration)
        elif stem_type.lower() in ['melody', 'lead']:
            generator = MelodyGenerator(models_dir=models_dir)
            audio = generator.generate(prompt, duration=duration)
        elif stem_type.lower() in ['harmony', 'chord', 'pad']:
            generator = HarmonyGenerator(models_dir=models_dir)
            audio = generator.generate(prompt, duration=duration)
        else:
            # Default to melody generator
            generator = MelodyGenerator(models_dir=models_dir)
            audio = generator.generate(prompt, duration=duration)
        
        # Save audio if requested
        if output_path and audio is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Convert to proper format if needed
            if len(audio.shape) == 1:
                audio = np.expand_dims(audio, axis=0)
                
            sf.write(output_path, audio.T, self.sample_rate)
            print(f"Generated {stem_type} saved to {output_path}")
        
        # Transcribe if requested
        midi_path = None
        if transcribe and audio is not None:
            if not hasattr(self, 'transcriber') or self.transcriber is None:
                self.transcriber = StemTranscriber(sample_rate=self.sample_rate)
                
            midi_dir = os.path.dirname(midi_output_path) if midi_output_path else tempfile.mkdtemp()
            os.makedirs(midi_dir, exist_ok=True)
            
            midi_path = midi_output_path or os.path.join(midi_dir, f"generated_{stem_type}.mid")
            result = self.transcriber.transcribe_audio(audio, midi_path)
            
            if result:
                print(f"Generated {stem_type} transcribed to {midi_path}")
                midi_path = result.get("path", midi_path)
            else:
                print(f"Failed to transcribe generated {stem_type}")
                midi_path = None
        
        return audio, midi_path 