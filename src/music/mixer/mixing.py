import numpy as np
import torch
from typing import Dict, List, Optional, Union, Tuple, Any

class MixingModule:
    """Module for mixing and processing audio tracks for professional-quality output"""
    
    def __init__(self, 
                sample_rate: int = 44100):
        """
        Initialize the mixing module.
        
        Args:
            sample_rate: Audio sample rate to work with
        """
        self.sample_rate = sample_rate
        
        # Default track levels
        self.default_levels = {
            'sampled': 0.8,    # Sampled components
            'melody': 0.9,     # Melody tracks
            'harmony': 0.7,    # Harmony tracks
            'bass': 0.85,      # Bass tracks
            'drums': 0.8       # Drum tracks
        }
        
        # Default reverb parameters
        self.reverb_params = {
            'room_size': 0.3,
            'damping': 0.5,
            'wet_level': 0.2,
            'dry_level': 0.8
        }
        
        # Default compression parameters
        self.comp_params = {
            'threshold': -20,
            'ratio': 4.0,
            'attack': 5.0,
            'release': 50.0,
            'makeup_gain': 3.0
        }
    
    def mix_tracks(self, 
                  tracks: Dict[str, Union[np.ndarray, torch.Tensor]],
                  levels: Optional[Dict[str, float]] = None,
                  apply_processing: bool = True) -> np.ndarray:
        """
        Mix multiple audio tracks into a stereo output with optional processing.
        
        Args:
            tracks: Dictionary of named audio tracks
            levels: Optional custom levels for each track (overrides defaults)
            apply_processing: Whether to apply audio processing
            
        Returns:
            Stereo mixed audio as numpy array
        """
        if not tracks:
            return np.zeros((2, 1))  # Empty stereo audio
            
        # Initialize stereo output
        sample_count = 0
        
        # Find the maximum length of all tracks
        for name, audio in tracks.items():
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            
            # Handle both mono and stereo inputs
            if len(audio.shape) == 1:
                # Mono
                length = audio.shape[0]
            else:
                # Could be (samples, channels) or (channels, samples)
                if audio.shape[0] <= 2:  # Assuming (channels, samples)
                    length = audio.shape[1]
                else:  # Assuming (samples, channels)
                    length = audio.shape[0]
            
            sample_count = max(sample_count, length)
        
        # Initialize the output mix
        mix = np.zeros((2, sample_count))
        
        # Use provided levels or defaults
        if levels is None:
            levels = self.default_levels
        
        # Add each track to the mix
        for name, audio in tracks.items():
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
                
            track_level = levels.get(name, 0.8)  # Default to 0.8 if not specified
            track_type = self._detect_track_type(name)
            
            # Add the track with appropriate processing
            self._add_track(mix, audio, track_level, track_type, apply_processing)
        
        # Apply master processing
        if apply_processing:
            mix = self._apply_master_processing(mix)
        
        return mix
    
    def _detect_track_type(self, track_name: str) -> str:
        """
        Determine the track type based on its name.
        
        Args:
            track_name: Name of the track
            
        Returns:
            Track type category
        """
        track_name = track_name.lower()
        
        if 'melody' in track_name or 'lead' in track_name:
            return 'melody'
        elif 'harm' in track_name or 'chord' in track_name or 'pad' in track_name:
            return 'harmony'
        elif 'bass' in track_name:
            return 'bass'
        elif 'drum' in track_name or 'beat' in track_name or 'percussion' in track_name:
            return 'drums'
        elif 'sample' in track_name:
            return 'sampled'
        else:
            return 'other'
    
    def _add_track(self, 
                  mix: np.ndarray,
                  audio: np.ndarray,
                  level: float,
                  track_type: str,
                  apply_processing: bool) -> None:
        """
        Add a track to the mix with appropriate processing.
        
        Args:
            mix: The stereo mix to add to
            audio: The audio track to add
            level: Volume level for the track
            track_type: Type of track for specific processing
            apply_processing: Whether to apply processing
        """
        # Convert to 2D array (stereo) if mono
        stereo_audio = self._ensure_stereo(audio)
        
        # Apply appropriate track-specific processing
        if apply_processing:
            # Apply track-specific EQ
            if track_type == 'bass':
                stereo_audio = self._apply_bass_eq(stereo_audio)
            elif track_type == 'melody':
                stereo_audio = self._apply_melody_eq(stereo_audio)
                # Add light reverb to melody
                stereo_audio = self._apply_reverb(
                    stereo_audio, 
                    room_size=0.4, 
                    wet_level=0.15
                )
            elif track_type == 'harmony':
                stereo_audio = self._apply_harmony_eq(stereo_audio)
                # Add medium reverb to harmony
                stereo_audio = self._apply_reverb(
                    stereo_audio, 
                    room_size=0.5, 
                    wet_level=0.25
                )
            elif track_type == 'drums':
                stereo_audio = self._apply_drums_eq(stereo_audio)
                # Compress drums more aggressively
                stereo_audio = self._apply_compression(
                    stereo_audio,
                    threshold=-18,
                    ratio=5.0
                )
            
            # Apply gentle compression to all tracks
            stereo_audio = self._apply_compression(
                stereo_audio,
                threshold=-20,
                ratio=2.5,
                attack=10.0,
                release=100.0
            )
        
        # Adjust levels and add to mix
        mix += stereo_audio * level
    
    def _ensure_stereo(self, audio: np.ndarray) -> np.ndarray:
        """
        Ensure audio is in stereo format (2, samples).
        
        Args:
            audio: Input audio array
            
        Returns:
            Stereo audio array
        """
        # Check if already stereo
        if len(audio.shape) == 2:
            if audio.shape[0] == 2:  # Already (2, samples)
                return audio
            elif audio.shape[1] == 2:  # Convert from (samples, 2) to (2, samples)
                return audio.T
            else:  # Mono with another dimension, use first channel
                return np.stack([audio[:, 0], audio[:, 0]])
        else:  # Mono
            return np.stack([audio, audio])
    
    def _apply_master_processing(self, mix: np.ndarray) -> np.ndarray:
        """
        Apply final processing to the master mix.
        
        Args:
            mix: Stereo mix
            
        Returns:
            Processed stereo mix
        """
        # Apply multi-band compression
        mix = self._apply_multiband_compression(mix)
        
        # Apply final EQ
        mix = self._apply_master_eq(mix)
        
        # Apply final limiting/soft clipping
        mix = self._apply_soft_clipper(mix)
        
        # Apply subtle fadeout to avoid abrupt endings
        if mix.shape[1] > 0:
            fade_samples = min(int(0.05 * self.sample_rate), mix.shape[1])
            mix = self._apply_fade(mix, fade_in_samples=fade_samples, fade_out_samples=fade_samples)
        
        return mix
    
    def _apply_soft_clipper(self, audio: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """
        Apply soft clipping to avoid harsh digital distortion.
        
        Args:
            audio: Input audio
            threshold: Threshold where soft clipping begins
            
        Returns:
            Processed audio
        """
        # Scale audio to prevent extreme peaks
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak
        
        # Apply soft clipping using tanh function
        # This smoothly limits the audio to prevent harsh clipping
        result = np.copy(audio)
        mask = np.abs(audio) > threshold
        
        # Only apply to samples above threshold
        if np.any(mask):
            soft_clip = np.tanh((np.abs(audio[mask]) - threshold) / (1 - threshold))
            result[mask] = np.sign(audio[mask]) * (threshold + (1 - threshold) * soft_clip)
        
        return result
    
    def _apply_bass_eq(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply EQ settings optimized for bass.
        
        Args:
            audio: Input audio
            
        Returns:
            EQ'd audio
        """
        # Simple frequency-domain EQ simulation
        # This is a placeholder for actual frequency-domain EQ
        # In a real implementation, this would use FFT/IFFT
        
        # Boost low frequencies, cut very low and high
        result = np.copy(audio)
        
        # Low-shelf boost around 80-150Hz (simulated)
        result *= 1.3
        
        # High cut above 5kHz (simulated)
        result *= 0.9
        
        return result
    
    def _apply_melody_eq(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply EQ settings optimized for melody.
        
        Args:
            audio: Input audio
            
        Returns:
            EQ'd audio
        """
        # Simulated EQ: Boost mids, light presence boost
        result = np.copy(audio)
        
        # Mid boost around 1-3kHz (simulated)
        result *= 1.15
        
        # High shelf boost around 8-10kHz (simulated)
        result *= 1.1
        
        return result
    
    def _apply_harmony_eq(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply EQ settings optimized for harmony tracks.
        
        Args:
            audio: Input audio
            
        Returns:
            EQ'd audio
        """
        # Simulated EQ: Scoop mids slightly, boost highs
        result = np.copy(audio)
        
        # Mid scoop around 500Hz-1kHz (simulated)
        result *= 0.9
        
        # High shelf boost around 6-8kHz (simulated)
        result *= 1.05
        
        return result
    
    def _apply_drums_eq(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply EQ settings optimized for drums.
        
        Args:
            audio: Input audio
            
        Returns:
            EQ'd audio
        """
        # Simulated EQ: Boost lows and highs
        result = np.copy(audio)
        
        # Low boost around 100-200Hz (simulated)
        result *= 1.2
        
        # High shelf boost around 8-12kHz (simulated)
        result *= 1.15
        
        return result
    
    def _apply_master_eq(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply EQ settings for the master output.
        
        Args:
            audio: Input audio
            
        Returns:
            EQ'd audio
        """
        # Simulated EQ: Slight adjustments for final polish
        result = np.copy(audio)
        
        # Very slight boost in presence region (simulated)
        result *= 1.05
        
        # Very slight sub cut below 30Hz (simulated)
        result *= 0.98
        
        return result
    
    def _apply_compression(self, 
                         audio: np.ndarray,
                         threshold: float = -20.0,
                         ratio: float = 4.0,
                         attack: float = 5.0,
                         release: float = 50.0,
                         makeup_gain: float = 3.0) -> np.ndarray:
        """
        Apply dynamic range compression.
        
        Args:
            audio: Input audio
            threshold: Threshold in dB
            ratio: Compression ratio
            attack: Attack time in ms
            release: Release time in ms
            makeup_gain: Makeup gain in dB
            
        Returns:
            Compressed audio
        """
        # Simple envelope follower-based compression
        # In a real implementation, this would use proper attack/release envelopes
        
        # Convert parameters
        threshold_linear = 10 ** (threshold / 20.0)
        makeup_linear = 10 ** (makeup_gain / 20.0)
        
        # Calculate envelope (very simplified)
        envelope = np.max(np.abs(audio), axis=0)
        
        # Calculate gain reduction
        gain_reduction = np.ones_like(envelope)
        mask = envelope > threshold_linear
        
        if np.any(mask):
            gain_reduction[mask] = (threshold_linear + (envelope[mask] - threshold_linear) / ratio) / envelope[mask]
        
        # Apply gain reduction to both channels
        compressed = np.copy(audio)
        for i in range(audio.shape[0]):
            compressed[i, :] = audio[i, :] * gain_reduction
        
        # Apply makeup gain
        compressed *= makeup_linear
        
        return compressed
    
    def _apply_multiband_compression(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply multi-band compression (simplified).
        
        Args:
            audio: Input audio
            
        Returns:
            Processed audio
        """
        # This is a very simplified simulation of multi-band compression
        # In a real implementation, this would separate into frequency bands
        # and apply different compression to each band
        
        # Simulate low-band compression (more aggressive)
        low_comp = self._apply_compression(
            audio,
            threshold=-24.0,
            ratio=4.0,
            makeup_gain=4.0
        )
        
        # Simulate high-band compression (more gentle)
        high_comp = self._apply_compression(
            audio,
            threshold=-18.0,
            ratio=2.5,
            makeup_gain=2.0
        )
        
        # Blend the results (simulating crossover)
        result = 0.6 * low_comp + 0.4 * high_comp
        
        return result
    
    def _apply_reverb(self, 
                    audio: np.ndarray,
                    room_size: float = 0.3,
                    damping: float = 0.5,
                    wet_level: float = 0.2,
                    dry_level: float = 0.8) -> np.ndarray:
        """
        Apply reverb effect (simplified simulation).
        
        Args:
            audio: Input audio
            room_size: Size of the reverb space (0-1)
            damping: Damping factor (0-1)
            wet_level: Level of reverb signal
            dry_level: Level of dry signal
            
        Returns:
            Processed audio
        """
        # This is a very simplified simulation of reverb
        # In a real implementation, this would use convolution or feedback delay networks
        
        # Create a simulated reverb tail
        reverb_length = int(room_size * self.sample_rate)
        if reverb_length == 0:
            return audio
            
        # Create a simplified reverb impulse response
        impulse = np.exp(-damping * np.arange(reverb_length) / self.sample_rate)
        
        # Simple convolution for each channel
        reverb_audio = np.zeros_like(audio)
        for i in range(audio.shape[0]):
            # Very simplified convolution (should use np.convolve in real implementation)
            reverb_tail = np.zeros(audio.shape[1] + reverb_length - 1)
            for j in range(audio.shape[1]):
                if j % 1000 == 0:  # Sparse approximation for performance
                    reverb_tail[j:j+reverb_length] += audio[i, j] * impulse
            
            reverb_audio[i, :] = reverb_tail[:audio.shape[1]]
        
        # Mix dry and wet signals
        return dry_level * audio + wet_level * reverb_audio
    
    def _apply_fade(self, 
                  audio: np.ndarray, 
                  fade_in_samples: int = 0, 
                  fade_out_samples: int = 0) -> np.ndarray:
        """
        Apply fade in/out to audio.
        
        Args:
            audio: Input audio
            fade_in_samples: Number of samples for fade in
            fade_out_samples: Number of samples for fade out
            
        Returns:
            Audio with fades applied
        """
        result = np.copy(audio)
        
        # Apply fade in
        if fade_in_samples > 0:
            fade_in = np.linspace(0, 1, fade_in_samples)
            for i in range(audio.shape[0]):
                result[i, :fade_in_samples] *= fade_in
        
        # Apply fade out
        if fade_out_samples > 0 and audio.shape[1] > fade_out_samples:
            fade_out = np.linspace(1, 0, fade_out_samples)
            for i in range(audio.shape[0]):
                result[i, -fade_out_samples:] *= fade_out
        
        return result 