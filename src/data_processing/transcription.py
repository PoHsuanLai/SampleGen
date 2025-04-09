import os
import numpy as np
import tempfile
import soundfile as sf
from typing import Dict, List, Optional, Union, Tuple
from basic_pitch import predict
from basic_pitch.inference import predict_and_save

class StemTranscriber:
    """
    Module for transcribing audio stems to MIDI using BasicPitch.
    """
    
    def __init__(self, 
                sample_rate: int = 44100,
                temp_dir: Optional[str] = None):
        """
        Initialize the StemTranscriber.
        
        Args:
            sample_rate: Audio sample rate to work with
            temp_dir: Optional temporary directory for intermediate files
        """
        self.sample_rate = sample_rate
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        os.makedirs(self.temp_dir, exist_ok=True)
        
    def transcribe_audio(self, 
                        audio_data: np.ndarray, 
                        output_midi_path: Optional[str] = None,
                        min_note_duration: float = 0.05,
                        min_frequency: float = 27.5,  # A0
                        max_frequency: float = 4186.0,  # C8
                        return_midi_data: bool = False) -> Optional[Dict]:
        """
        Transcribe audio data to MIDI.
        
        Args:
            audio_data: Audio data as a numpy array (channels, samples)
            output_midi_path: Path to save the output MIDI file (optional)
            min_note_duration: Minimum note duration in seconds
            min_frequency: Minimum frequency to transcribe (Hz)
            max_frequency: Maximum frequency to transcribe (Hz)
            return_midi_data: Whether to return the MIDI data
            
        Returns:
            MIDI data or None
        """
        # Ensure the audio is in the right format for BasicPitch
        # BasicPitch expects mono audio
        if len(audio_data.shape) > 1:
            # Convert multichannel to mono by averaging
            audio_data = np.mean(audio_data, axis=0)
        
        # Create a temporary file to save the audio data
        temp_audio_path = os.path.join(self.temp_dir, "temp_audio_for_transcription.wav")
        sf.write(temp_audio_path, audio_data, self.sample_rate)
        
        # Create output directory if it doesn't exist
        if output_midi_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_midi_path)), exist_ok=True)
        else:
            output_midi_path = os.path.join(self.temp_dir, "transcribed_output.mid")
        
        try:
            # Use BasicPitch to transcribe
            model_output, midi_data, note_events = predict.predict(
                temp_audio_path,
                onset_threshold=0.5,  # Adjust based on your needs
                frame_threshold=0.3,  # Adjust based on your needs
                min_note_length=min_note_duration,
                min_freq=min_frequency,
                max_freq=max_frequency
            )
            
            # Save MIDI file
            predict_and_save.save_midi(
                Path=output_midi_path,
                note_events=note_events,
                onsets=model_output["onsets"],
                frames=model_output["frames"],
                onset_thresh=0.5,
                frame_thresh=0.3
            )
            
            print(f"Transcribed MIDI saved to: {output_midi_path}")
            
            if return_midi_data:
                return midi_data
            else:
                return {"path": output_midi_path, "note_events": note_events}
                
        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return None
        finally:
            # Clean up temporary file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    def transcribe_stem(self, 
                       stem_audio: np.ndarray,
                       stem_name: str,
                       output_dir: str,
                       min_note_duration: float = 0.05) -> str:
        """
        Transcribe a specific stem to MIDI.
        
        Args:
            stem_audio: Stem audio data
            stem_name: Name of the stem (e.g., 'vocals', 'drums', 'bass')
            output_dir: Directory to save the output
            min_note_duration: Minimum note duration in seconds
            
        Returns:
            Path to the output MIDI file
        """
        # Adjust parameters based on stem type
        if stem_name.lower() == 'bass':
            min_freq = 30.0     # Very low for bass
            max_freq = 500.0    # Upper limit for bass frequencies
        elif stem_name.lower() == 'drums':
            min_freq = 50.0     # Catch kick drum
            max_freq = 10000.0  # Catch cymbals and high-hats
            min_note_duration = 0.02  # Shorter for percussive sounds
        elif stem_name.lower() == 'vocals':
            min_freq = 80.0     # Human vocal range
            max_freq = 1200.0   # Upper limit for vocals
        else:
            # Default values for other instruments
            min_freq = 27.5     # A0
            max_freq = 4186.0   # C8
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set output path
        output_path = os.path.join(output_dir, f"{stem_name}_transcribed.mid")
        
        # Transcribe with stem-specific settings
        result = self.transcribe_audio(
            audio_data=stem_audio,
            output_midi_path=output_path,
            min_note_duration=min_note_duration,
            min_frequency=min_freq,
            max_frequency=max_freq
        )
        
        return output_path if result else None
    
    def transcribe_all_stems(self, 
                            stems: Dict[str, np.ndarray], 
                            output_dir: str) -> Dict[str, str]:
        """
        Transcribe all stems and save as MIDI files.
        
        Args:
            stems: Dictionary of stem name to audio data
            output_dir: Directory to save the output
            
        Returns:
            Dictionary mapping stem names to MIDI file paths
        """
        midi_paths = {}
        
        for stem_name, stem_audio in stems.items():
            print(f"Transcribing {stem_name}...")
            midi_path = self.transcribe_stem(
                stem_audio=stem_audio,
                stem_name=stem_name,
                output_dir=output_dir
            )
            
            if midi_path:
                midi_paths[stem_name] = midi_path
            else:
                print(f"Failed to transcribe {stem_name}")
        
        return midi_paths 