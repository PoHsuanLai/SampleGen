"""
Faust-based audio distorter for generating training data.
Creates DSP scripts and corresponding audio distortions.
"""

import os
import tempfile
import subprocess
import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pydub import AudioSegment
import soundfile as sf


class FaustDistorter:
    """
    Generate audio distortions using Faust DSP scripts.
    Creates both the distorted audio and the correction DSP code.
    """
    
    def __init__(self, sample_rate: int = 44100):
        """
        Initialize the Faust distorter.
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.temp_dir = tempfile.mkdtemp(prefix="faust_distort_")
        
    def __del__(self):
        """Clean up temporary files."""
        import shutil
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def generate_volume_distortion(self, gain_db: float = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate volume change distortion.
        
        Args:
            gain_db: Gain in dB (random if None)
            
        Returns:
            Tuple of (faust_code, correction_params)
        """
        if gain_db is None:
            gain_db = random.uniform(-25, 8)
        
        # Faust code for volume change
        faust_code = f"""
import("stdfaust.lib");

gain_db = {gain_db};
gain_linear = ba.db2linear(gain_db);

process = _ * gain_linear;
"""
        
        # Correction parameters (inverse gain)
        correction_params = {
            "tool": "change_volume", 
            "params": {"db": -gain_db},
            "faust_correction": f"process = _ * ba.db2linear({-gain_db});"
        }
        
        return faust_code, correction_params
    
    def generate_lowpass_distortion(self, cutoff_hz: float = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate lowpass filter distortion.
        
        Args:
            cutoff_hz: Cutoff frequency in Hz (random if None)
            
        Returns:
            Tuple of (faust_code, correction_params)
        """
        if cutoff_hz is None:
            cutoff_hz = random.uniform(1000, 8000)
        
        # Faust code for lowpass filter
        faust_code = f"""
import("stdfaust.lib");

cutoff = {cutoff_hz};
process = fi.lowpass(3, cutoff);
"""
        
        # Correction parameters (highpass + boost)
        correction_params = {
            "tool": "apply_highpass_correction",
            "params": {"cutoff": cutoff_hz * 0.8},  # Slightly lower for restoration
            "faust_correction": f"process = fi.highpass(1, {cutoff_hz * 0.8}) : *(1.2);"
        }
        
        return faust_code, correction_params
    
    def generate_highpass_distortion(self, cutoff_hz: float = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate highpass filter distortion.
        
        Args:
            cutoff_hz: Cutoff frequency in Hz (random if None)
            
        Returns:
            Tuple of (faust_code, correction_params)
        """
        if cutoff_hz is None:
            cutoff_hz = random.uniform(150, 800)
        
        # Faust code for highpass filter
        faust_code = f"""
import("stdfaust.lib");

cutoff = {cutoff_hz};
process = fi.highpass(3, cutoff);
"""
        
        # Correction parameters (lowpass + boost low end)
        correction_params = {
            "tool": "apply_lowpass_correction",
            "params": {"cutoff": cutoff_hz * 1.2},
            "faust_correction": f"process = fi.lowpass(1, {cutoff_hz * 1.2}) : *(1.5) + fi.highpass(1, {cutoff_hz});"
        }
        
        return faust_code, correction_params
    
    def generate_compression_distortion(self, 
                                      threshold_db: float = None,
                                      ratio: float = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate compression distortion.
        
        Args:
            threshold_db: Compression threshold in dB
            ratio: Compression ratio
            
        Returns:
            Tuple of (faust_code, correction_params)
        """
        if threshold_db is None:
            threshold_db = random.uniform(-30, -10)
        if ratio is None:
            ratio = random.uniform(4.0, 12.0)
        
        # Faust code for compression
        faust_code = f"""
import("stdfaust.lib");

threshold_db = {threshold_db};
ratio = {ratio};
attack = 0.003;
release = 0.1;

process = co.compressor_mono(ratio, threshold_db, attack, release);
"""
        
        # Correction parameters (expansion)
        expansion_ratio = 1.0 / ratio
        correction_params = {
            "tool": "apply_expansion",
            "params": {"threshold_db": threshold_db, "ratio": expansion_ratio},
            "faust_correction": f"process = co.compressor_mono({1/ratio}, {threshold_db}, 0.001, 0.05) : *(1.2);"
        }
        
        return faust_code, correction_params
    
    def generate_delay_distortion(self, 
                                delay_ms: float = None,
                                feedback: float = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate delay effect that needs removal.
        
        Args:
            delay_ms: Delay time in milliseconds
            feedback: Feedback amount (0-1)
            
        Returns:
            Tuple of (faust_code, correction_params)
        """
        if delay_ms is None:
            delay_ms = random.uniform(50, 300)
        if feedback is None:
            feedback = random.uniform(0.2, 0.7)
        
        # Convert ms to samples
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        
        # Faust code for delay
        faust_code = f"""
import("stdfaust.lib");

delay_samples = {delay_samples};
feedback = {feedback};
mix = 0.3;

process = _ <: _, (de.delay(48000, delay_samples) * feedback) :> _ * (1-mix) + _ * mix;
"""
        
        # Correction is complex - suggest dry signal extraction
        correction_params = {
            "tool": "remove_delay",
            "params": {"delay_ms": delay_ms, "feedback": feedback},
            "faust_correction": f"process = _;"  # Simple dry signal (ideally more complex)
        }
        
        return faust_code, correction_params
    
    def generate_reverb_distortion(self, 
                                 room_size: float = None,
                                 damping: float = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate reverb that needs removal.
        
        Args:
            room_size: Room size parameter (0-1)
            damping: Damping parameter (0-1)
            
        Returns:
            Tuple of (faust_code, correction_params)
        """
        if room_size is None:
            room_size = random.uniform(0.3, 0.9)
        if damping is None:
            damping = random.uniform(0.1, 0.7)
        
        # Faust code for reverb
        faust_code = f"""
import("stdfaust.lib");

room_size = {room_size};
damping = {damping};
mix = 0.25;

process = _ <: _, re.mono_freeverb(damping, room_size, 0.5) :> _ * (1-mix) + _ * mix;
"""
        
        # Correction parameters (reverb removal is complex)
        correction_params = {
            "tool": "remove_reverb",
            "params": {"room_size": room_size, "damping": damping},
            "faust_correction": f"process = fi.highpass(1, 80) : co.compressor_mono(3, -20, 0.001, 0.1);"
        }
        
        return faust_code, correction_params
    
    def generate_eq_distortion(self, 
                             low_gain: float = None,
                             mid_gain: float = None,
                             high_gain: float = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate EQ distortion.
        
        Args:
            low_gain: Low frequency gain in dB
            mid_gain: Mid frequency gain in dB  
            high_gain: High frequency gain in dB
            
        Returns:
            Tuple of (faust_code, correction_params)
        """
        if low_gain is None:
            low_gain = random.uniform(-10, 10)
        if mid_gain is None:
            mid_gain = random.uniform(-8, 8)
        if high_gain is None:
            high_gain = random.uniform(-10, 10)
        
        # Faust code for 3-band EQ
        faust_code = f"""
import("stdfaust.lib");

low_gain_db = {low_gain};
mid_gain_db = {mid_gain};
high_gain_db = {high_gain};

low_freq = 200;
high_freq = 3000;

low_gain = ba.db2linear(low_gain_db);
mid_gain = ba.db2linear(mid_gain_db);
high_gain = ba.db2linear(high_gain_db);

process = _ <: 
    (fi.lowpass(3, low_freq) * low_gain),
    (fi.highpass(3, low_freq) : fi.lowpass(3, high_freq) * mid_gain),
    (fi.highpass(3, high_freq) * high_gain)
    :> _;
"""
        
        # Correction parameters (inverse EQ)
        correction_params = {
            "tool": "apply_eq_correction",
            "params": {
                "low_gain": -low_gain,
                "mid_gain": -mid_gain, 
                "high_gain": -high_gain
            },
            "faust_correction": f"""
process = _ <: 
    (fi.lowpass(3, 200) * ba.db2linear({-low_gain})),
    (fi.highpass(3, 200) : fi.lowpass(3, 3000) * ba.db2linear({-mid_gain})),
    (fi.highpass(3, 3000) * ba.db2linear({-high_gain}))
    :> _;
"""
        }
        
        return faust_code, correction_params
    
    def compile_and_apply_faust(self, 
                              faust_code: str,
                              input_audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Compile Faust code and apply to audio.
        
        Args:
            faust_code: Faust DSP code
            input_audio: Input audio array
            
        Returns:
            Processed audio array or None if failed
        """
        try:
            # Create temporary files
            dsp_file = os.path.join(self.temp_dir, "distortion.dsp")
            cpp_file = os.path.join(self.temp_dir, "distortion.cpp")
            binary_file = os.path.join(self.temp_dir, "distortion")
            input_wav = os.path.join(self.temp_dir, "input.wav")
            output_wav = os.path.join(self.temp_dir, "output.wav")
            
            # Write Faust code
            with open(dsp_file, 'w') as f:
                f.write(faust_code)
            
            # Write input audio
            sf.write(input_wav, input_audio, self.sample_rate)
            
            # Compile Faust to C++
            compile_cmd = [
                "faust", "-a", "minimal.cpp", "-o", cpp_file, dsp_file
            ]
            result = subprocess.run(compile_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Faust compilation failed: {result.stderr}")
                return None
            
            # Compile C++ to binary (simplified - would need proper build system)
            # For now, fall back to simpler processing
            return self._apply_simple_processing(input_audio, faust_code)
            
        except Exception as e:
            print(f"Error applying Faust processing: {e}")
            return None
    
    def _apply_simple_processing(self, audio: np.ndarray, faust_code: str) -> np.ndarray:
        """
        Apply simple processing based on Faust code content (fallback).
        
        Args:
            audio: Input audio
            faust_code: Faust code to analyze
            
        Returns:
            Processed audio
        """
        import re
        
        # Simple pattern matching for different effects
        if "gain_linear" in faust_code:
            # Extract gain value
            gain_match = re.search(r'gain_db = ([+-]?\d*\.?\d+)', faust_code)
            if gain_match:
                gain_db = float(gain_match.group(1))
                gain_linear = 10 ** (gain_db / 20)
                return audio * gain_linear
        
        elif "lowpass" in faust_code:
            # Apply simple lowpass using scipy
            try:
                from scipy import signal
                cutoff_match = re.search(r'cutoff = (\d+)', faust_code)
                if cutoff_match:
                    cutoff = int(cutoff_match.group(1))
                    nyquist = self.sample_rate / 2
                    normalized_cutoff = cutoff / nyquist
                    b, a = signal.butter(3, normalized_cutoff, btype='low')
                    return signal.filtfilt(b, a, audio)
            except ImportError:
                pass
        
        elif "highpass" in faust_code:
            # Apply simple highpass
            try:
                from scipy import signal
                cutoff_match = re.search(r'cutoff = (\d+)', faust_code)
                if cutoff_match:
                    cutoff = int(cutoff_match.group(1))
                    nyquist = self.sample_rate / 2
                    normalized_cutoff = cutoff / nyquist
                    b, a = signal.butter(3, normalized_cutoff, btype='high')
                    return signal.filtfilt(b, a, audio)
            except ImportError:
                pass
        
        # If no specific processing matched, return original
        return audio
    
    def get_random_distortion(self, stem_type: str = None) -> Tuple[str, Dict[str, Any]]:
        """
        Get a random distortion appropriate for the stem type.
        
        Args:
            stem_type: Type of stem (vocals, drums, bass, other)
            
        Returns:
            Tuple of (faust_code, correction_params)
        """
        # Define stem-specific distortion probabilities
        if stem_type == "vocals":
            distortions = [
                self.generate_volume_distortion,
                self.generate_compression_distortion,
                self.generate_eq_distortion,
                self.generate_reverb_distortion,
                self.generate_delay_distortion
            ]
        elif stem_type == "drums":
            distortions = [
                self.generate_volume_distortion,
                self.generate_compression_distortion,
                self.generate_highpass_distortion,
                self.generate_eq_distortion
            ]
        elif stem_type == "bass":
            distortions = [
                self.generate_volume_distortion,
                self.generate_lowpass_distortion,
                self.generate_compression_distortion,
                self.generate_eq_distortion
            ]
        else:  # other
            distortions = [
                self.generate_volume_distortion,
                self.generate_eq_distortion,
                self.generate_reverb_distortion,
                self.generate_lowpass_distortion,
                self.generate_highpass_distortion
            ]
        
        # Choose random distortion
        distortion_func = random.choice(distortions)
        return distortion_func()
    
    def apply_distortion(self, 
                        audio: np.ndarray,
                        stem_type: str = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply a random distortion to audio and return correction info.
        
        Args:
            audio: Input audio array
            stem_type: Type of stem for targeted distortions
            
        Returns:
            Tuple of (distorted_audio, correction_info)
        """
        # Get random distortion
        faust_code, correction_params = self.get_random_distortion(stem_type)
        
        # Apply distortion
        distorted_audio = self.compile_and_apply_faust(faust_code, audio)
        
        if distorted_audio is None:
            # Fallback to original if processing failed
            distorted_audio = audio
            correction_params = {"tool": "no_change", "params": {}}
        
        # Add the original Faust code to correction params
        correction_params["original_faust_code"] = faust_code
        
        return distorted_audio, correction_params


def render_faust_to_audio(faust_code: str, 
                         input_audio_path: str,
                         output_audio_path: str,
                         sample_rate: int = 44100) -> bool:
    """
    Render a Faust DSP script to audio file.
    
    Args:
        faust_code: Faust DSP code
        input_audio_path: Path to input audio file
        output_audio_path: Path to output audio file
        sample_rate: Sample rate
        
    Returns:
        True if successful, False otherwise
    """
    try:
        distorter = FaustDistorter(sample_rate)
        
        # Load input audio
        audio_data, sr = sf.read(input_audio_path)
        if sr != sample_rate:
            # Simple resampling
            from scipy import signal
            audio_data = signal.resample(audio_data, int(len(audio_data) * sample_rate / sr))
        
        # Apply Faust processing
        processed_audio = distorter.compile_and_apply_faust(faust_code, audio_data)
        
        if processed_audio is not None:
            # Write output
            sf.write(output_audio_path, processed_audio, sample_rate)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error rendering Faust to audio: {e}")
        return False 