import random
import numpy as np
from pydub import AudioSegment
from pydub.generators import WhiteNoise, Sine

class AudioDistorter:
    """Class to generate distorted versions of audio for training the mixer model."""
    
    def __init__(self, audio, stem_type=None):
        """
        Initialize with an AudioSegment object directly.
        
        Args:
            audio: AudioSegment object to distort
            stem_type: Type of stem (vocals, drums, bass, other) to customize distortions
        """
        self.audio = audio
        self.sample_rate = audio.frame_rate
        self.stem_type = stem_type
    
    def change_volume_distortion(self, min_db=-25, max_db=8):
        """Randomly change volume to create training data for volume adjustment."""
        db_change = random.uniform(min_db, max_db)
        distorted = self.audio - db_change  # Subtracting means reducing volume
        return distorted, {"tool": "change_volume", "params": {"db": db_change}}
    
    def pan_distortion(self, min_pan=-1.0, max_pan=1.0):
        """Create imbalanced stereo panning to train pan correction."""
        pan_value = random.uniform(min_pan, max_pan)
        
        # Check if audio is mono (1 channel)
        if self.audio.channels == 1:
            # For mono, convert to stereo first
            stereo_audio = AudioSegment.from_mono_audiosegments(self.audio, self.audio)
            # Create a panned version that needs correction
            distorted = stereo_audio.pan(pan_value)
        else:
            # For stereo, apply pan directly
            distorted = self.audio.pan(pan_value)
        
        return distorted, {"tool": "pan", "params": {"pan_value": -pan_value}}  # Inverse for correction
    
    def low_pass_distortion(self, min_cutoff=1000, max_cutoff=5000):
        """Add high frequency content that needs to be filtered with low pass."""
        # Check if audio is long enough for proper processing
        if len(self.audio) < 500:  # Minimum required for proper processing
            cutoff = random.randint(min_cutoff, max_cutoff)
            return self.audio, {"tool": "apply_low_pass", "params": {"cutoff": cutoff}}
            
        cutoff = random.randint(min_cutoff, max_cutoff)
        
        # Generate high frequency noise with a safe duration
        # Limit noise duration to prevent processing issues with very long audio
        noise_duration = min(len(self.audio), 10000)
        
        # Generate the noise
        noise = WhiteNoise().to_audio_segment(duration=noise_duration)
        
        # Apply high pass to noise to keep only high frequencies
        noise = noise.high_pass_filter(cutoff)
        
        # Make noise louder for more aggressive distortion
        noise = noise - 8  # -8dB (previously -15dB)
        
        # Make sure noise matches audio length properly
        if len(noise) < len(self.audio):
            # Calculate repetitions needed
            repetitions = (len(self.audio) // len(noise)) + 1
            # Limit repetitions to prevent excessive memory usage
            repetitions = min(repetitions, 10)
            extended_noise = noise * repetitions
            # Trim to match audio length
            noise = extended_noise[:len(self.audio)]
        elif len(noise) > len(self.audio):
            # Truncate if longer
            noise = noise[:len(self.audio)]
        
        # Verify dimensions match before overlay
        if len(noise) == len(self.audio):
            # Overlay noise onto original
            distorted = self.audio.overlay(noise)
        else:
            # If dimensions still don't match (which shouldn't happen), use original audio
            distorted = self.audio
        
        return distorted, {"tool": "apply_low_pass", "params": {"cutoff": cutoff}}
    
    def high_pass_distortion(self, min_cutoff=150, max_cutoff=800):
        """Add low frequency rumble that needs to be filtered with high pass."""
        # Check if the audio segment is long enough
        if len(self.audio) < 500:  # Minimum required for proper processing
            cutoff = random.randint(min_cutoff, max_cutoff)
            return self.audio, {"tool": "apply_high_pass", "params": {"cutoff": cutoff}}
            
        cutoff = random.randint(min_cutoff, max_cutoff)
        
        # Generate low frequency sine wave rumble
        freq = random.randint(20, 80)  # Very low frequency
        
        # Limit rumble duration to avoid issues with very long audio
        rumble_duration = min(len(self.audio), 10000)
        
        # Generate the rumble effect
        rumble = Sine(freq).to_audio_segment(duration=rumble_duration)
        
        # Make sure it's only affecting low frequencies
        rumble = rumble.low_pass_filter(cutoff // 2)
        
        # Make rumble louder for more aggressive distortion
        rumble = rumble - 5  # -5dB (previously -10dB)
        
        # Make sure rumble matches audio length
        if len(rumble) < len(self.audio):
            # Calculate repetitions needed
            repetitions = (len(self.audio) // len(rumble)) + 1
            # Limit repetitions to prevent excessive memory usage
            repetitions = min(repetitions, 10)
            extended_rumble = rumble * repetitions
            # Trim to match audio length
            rumble = extended_rumble[:len(self.audio)]
        elif len(rumble) > len(self.audio):
            # Truncate if longer
            rumble = rumble[:len(self.audio)]
        
        # Overlay rumble onto original
        distorted = self.audio.overlay(rumble)
        
        return distorted, {"tool": "apply_high_pass", "params": {"cutoff": cutoff}}
    
    def normalize_distortion(self, min_db=-30, max_db=-3):
        """Create audio with varied levels that needs normalization."""
        # First create a quieter or louder version
        db_change = random.uniform(min_db, max_db)
        distorted = self.audio + db_change
        
        return distorted, {"tool": "apply_normalize", "params": {}}
    
    def delay_distortion(self):
        """Create a version that needs delay effect."""
        # The mixer will add delay, so we don't need to pre-distort
        # Just return the original with the recommendation to add delay
        delay_ms = random.randint(200, 500)  # Increased delay range
        decay = random.uniform(0.4, 0.9)     # Increased decay range
        
        return self.audio, {"tool": "apply_delay", "params": {"delay_ms": delay_ms, "decay": decay}}
    
    def reverb_distortion(self):
        """Create a version that needs reverb effect."""
        # The mixer will add reverb, so we don't need to pre-distort
        # Just return the original with the recommendation to add reverb
        return self.audio, {"tool": "apply_reverb", "params": {}}
    
    def compression_distortion(self):
        """Create dynamic range issues that need compression."""
        # Check if the segment is long enough for peak addition
        segment_length = len(self.audio)
        if segment_length <= 300:  # Safe minimum length requirement
            # If too short, just apply overall gain and return
            threshold_db = random.randint(-30, -10)  # More aggressive threshold
            ratio = random.uniform(4.0, 8.0)         # More aggressive ratio
            return self.audio, {"tool": "apply_compression", "params": {"threshold_db": threshold_db, "ratio": ratio}}
        
        # Create a copy of the audio segment
        distorted = AudioSegment(
            data=self.audio._data,
            sample_width=self.audio.sample_width,
            frame_rate=self.audio.frame_rate,
            channels=self.audio.channels
        )
        
        # Add 3-6 random peaks (increased from 2-4)
        num_peaks = random.randint(3, 6)
        for _ in range(num_peaks):
            # Calculate safe positions for peak (at least 100ms from either end)
            safe_pos_max = max(0, segment_length - 200)
            if safe_pos_max <= 100:
                # If audio is too short for complex peak manipulation, just add one peak
                peak_pos = 0
                peak_length = segment_length
            else:
                # Generate a short peak
                peak_pos = random.randint(100, safe_pos_max)
                peak_length = random.randint(50, min(200, segment_length - peak_pos - 100))
            
            # Apply gain to create peak (increased range)
            peak_gain = random.uniform(8, 20)
            
            # Apply the peak
            if peak_pos > 0:
                before_peak = distorted[:peak_pos]
                peak = distorted[peak_pos:peak_pos+peak_length].apply_gain(peak_gain)
                if peak_pos + peak_length < segment_length:
                    after_peak = distorted[peak_pos+peak_length:]
                    distorted = before_peak + peak + after_peak
                else:
                    distorted = before_peak + peak
            else:
                peak = distorted[:peak_length].apply_gain(peak_gain)
                if peak_length < segment_length:
                    after_peak = distorted[peak_length:]
                    distorted = peak + after_peak
                else:
                    distorted = peak
        
        threshold_db = random.randint(-30, -10)  # More aggressive threshold
        ratio = random.uniform(4.0, 8.0)         # More aggressive ratio
        
        return distorted, {"tool": "apply_compression", "params": {"threshold_db": threshold_db, "ratio": ratio}}
    
    def eq_distortion(self):
        """Create frequency imbalances that need EQ correction."""
        # Validate audio length
        if len(self.audio) < 500:  # Minimum length required for proper filtering
            # For very short segments, just return the original with minimal EQ settings
            return self.audio, {"tool": "apply_eq", "params": {"low_gain": 0.0, "mid_gain": 0.0, "high_gain": 0.0}}
            
        # Randomly boost or cut frequency bands to create more extreme imbalance
        low_gain = random.uniform(-12, 12)   # Increased from -8,8
        mid_gain = random.uniform(-12, 12)   # Increased from -8,8
        high_gain = random.uniform(-12, 12)  # Increased from -8,8
        
        # Create frequency-imbalanced audio
        # Process each band separately with length verification
        low = self.audio.low_pass_filter(200)
        low = low.apply_gain(low_gain)
        
        mid = self.audio.high_pass_filter(200)
        mid = mid.low_pass_filter(3000)
        mid = mid.apply_gain(mid_gain)
        
        high = self.audio.high_pass_filter(3000)
        high = high.apply_gain(high_gain)
        
        # Ensure all segments are the same length before overlay
        min_length = min(len(low), len(mid), len(high))
        if min_length < len(self.audio):
            # If any processing shortened the audio, truncate all to the same length
            low = low[:min_length]
            mid = mid[:min_length]
            high = high[:min_length]
            
        # Overlay the bands
        distorted = low.overlay(mid)
        distorted = distorted.overlay(high)
        
        # The mixer would need to apply inverse EQ to correct
        return distorted, {"tool": "apply_eq", "params": {"low_gain": -low_gain, "mid_gain": -mid_gain, "high_gain": -high_gain}}
    
    def trim_distortion(self):
        """Create a version that needs trimming of silence or artifacts."""
        # For very short samples, just return original
        if len(self.audio) < 500:
            return self.audio, {"tool": "apply_trim", "params": {"start_ms": 0, "end_ms": 0}}
            
        segment_length = len(self.audio)
        
        # Add some silence/noise at beginning or end randomly
        if random.choice([True, False]):  # Beginning
            start_silence_ms = random.randint(50, 300)
            silence = AudioSegment.silent(duration=start_silence_ms)
            distorted = silence + self.audio
            trim_params = {"start_ms": start_silence_ms, "end_ms": 0}
        else:  # End
            end_silence_ms = random.randint(50, 300)
            silence = AudioSegment.silent(duration=end_silence_ms)
            distorted = self.audio + silence
            trim_params = {"start_ms": 0, "end_ms": end_silence_ms}
            
        return distorted, {"tool": "apply_trim", "params": trim_params}
    
    def speed_change_distortion(self, min_factor=0.7, max_factor=1.3):
        """Create speed/tempo variations that need to be corrected.
        
        Args:
            min_factor: Minimum speed factor (values below 1.0 slow down the audio)
            max_factor: Maximum speed factor (values above 1.0 speed up the audio)
            
        Returns:
            Tuple of (distorted_audio, action_dict)
        """
        # Generate a random speed factor
        speed_factor = random.uniform(min_factor, max_factor)
        
        # We can't modify the original audio's speed directly with pydub
        # So we'll use a simple approximation by adjusting its frame rate
        original_frame_rate = self.audio.frame_rate
        new_frame_rate = int(original_frame_rate * speed_factor)
        
        # Create a copy of the audio with the adjusted frame rate
        distorted = self.audio._spawn(
            self.audio.raw_data,
            overrides={
                "frame_rate": new_frame_rate
            }
        )
        
        # Adjust to the original frame rate to actually change the speed/tempo
        distorted = distorted.set_frame_rate(original_frame_rate)
        
        # The correction would use the inverse factor
        correction_factor = 1.0 / speed_factor
        
        return distorted, {"tool": "speed_change", "params": {"factor": correction_factor}}
    
    def loop_section_distortion(self):
        """Create a looping section effect that needs to be corrected.
        
        Returns:
            Tuple of (distorted_audio, action_dict)
        """
        # Get audio length
        audio_length_ms = len(self.audio)
        
        # For very short audio, just return it unmodified with dummy params
        if audio_length_ms < 500:
            return self.audio, {"tool": "loop_section", "params": {"start_ms": 0, "end_ms": 0, "count": 1}}
        
        # Determine section to loop
        section_length_ms = random.randint(200, min(1000, audio_length_ms - 100))
        start_ms = random.randint(0, audio_length_ms - section_length_ms)
        end_ms = start_ms + section_length_ms
        
        # How many times to loop (2-4)
        count = random.randint(2, 4)
        
        # Extract the section to loop
        section = self.audio[start_ms:end_ms]
        
        # Create the looped version
        looped = self.audio[:start_ms]
        for _ in range(count):
            looped += section
        if end_ms < audio_length_ms:
            looped += self.audio[end_ms:]
        
        # The model would need to reduce the loop count to 1
        return looped, {"tool": "loop_section", "params": {"start_ms": start_ms, "end_ms": end_ms, "count": 1}}
    
    def duplicate_and_shift_distortion(self, min_offset=50, max_offset=300):
        """Create a duplicated and shifted version of the audio.
        
        Args:
            min_offset: Minimum offset in ms
            max_offset: Maximum offset in ms
            
        Returns:
            Tuple of (distorted_audio, action_dict)
        """
        # For very short audio, skip actual processing
        if len(self.audio) < 400:
            return self.audio, {"tool": "duplicate_and_shift", "params": {"offset_ms": 0}}
        
        # Random offset
        offset_ms = random.randint(min_offset, max_offset)
        
        # Create a copy with the offset
        distorted = self.audio.overlay(
            self.audio, 
            position=offset_ms,
            gain_during_overlay=-3  # Slightly reduce volume of overlay
        )
        
        # To fix this, the model would need to remove the duplication
        return distorted, {"tool": "duplicate_and_shift", "params": {"offset_ms": 0}}
    
    def get_stem_specific_distortion_weights(self):
        """
        Get weights for different distortion methods based on stem type.
        
        These weights determine the likelihood of each distortion being selected
        for a particular stem type.
        
        Returns:
            Dictionary mapping distortion methods to their weights
        """
        # Default weights for all stems
        weights = {
            "change_volume": 1.0,
            "pan": 0.8,
            "low_pass": 0.6,
            "high_pass": 0.6,
            "normalize": 0.5,
            "delay": 0.7,
            "reverb": 0.6,
            "compression": 0.8,
            "eq": 0.8,
            "trim": 0.5,
            "loop_section": 0.3,
            "duplicate_and_shift": 0.3,
            "speed_change": 0.3
        }
        
        # Adjust weights based on stem type
        if self.stem_type == "vocals":
            # Vocals benefit from specific processing
            weights.update({
                "compression": 1.2,    # Vocals almost always need compression
                "eq": 1.1,             # EQ is important for vocal clarity
                "delay": 0.5,          # Less likely to need delay
                "trim": 0.8,           # Trimming can be important for vocals
                "low_pass": 0.4,       # Less likely to need low pass
                "high_pass": 0.8,      # High pass is useful for removing rumble
                "reverb": 0.7,         # Common effect for vocals
                "loop_section": 0.05,  # Very rarely loop vocals
                "duplicate_and_shift": 0.1,  # Rarely duplicate vocals
                "speed_change": 0.1    # Rarely change speed of vocals
            })
        elif self.stem_type == "drums":
            # Drums often need time-based processing
            weights.update({
                "compression": 1.2,    # Drums almost always need compression
                "eq": 0.7,             # Less frequent EQ changes
                "delay": 0.3,          # Rarely need delay
                "trim": 1.1,           # Trimming important for tight drums
                "low_pass": 0.3,       # Less likely to need low pass
                "high_pass": 0.6,      # Sometimes need high pass
                "reverb": 0.5,         # Sometimes need reverb
                "normalize": 0.7,      # Normalization less common
                "loop_section": 1.5,   # Very often loop drums
                "duplicate_and_shift": 1.3,  # Often duplicate drums
                "speed_change": 0.7    # May change speed of drums
            })
        elif self.stem_type == "bass":
            # Bass needs specific processing
            weights.update({
                "compression": 1.3,    # Bass almost always needs compression
                "eq": 0.6,             # Less frequent EQ changes
                "delay": 0.2,          # Rarely need delay
                "trim": 0.9,           # Sometimes need trimming
                "low_pass": 0.4,       # Less likely to need low pass
                "high_pass": 0.3,      # Rarely need high pass
                "reverb": 0.2,         # Rarely need reverb
                "normalize": 0.6,      # Normalization less common
                "loop_section": 1.5,   # Very often loop bass
                "duplicate_and_shift": 1.2,  # Often duplicate bass
                "speed_change": 0.7    # May change speed of bass
            })
        elif self.stem_type == "other":
            # Other instruments vary widely
            weights.update({
                "compression": 0.9,    # Often need compression
                "eq": 1.0,             # Frequently need EQ
                "delay": 0.8,          # Often need delay 
                "trim": 0.7,           # Sometimes need trimming
                "low_pass": 0.7,       # May need low pass
                "high_pass": 0.7,      # May need high pass
                "reverb": 0.9,         # Often need reverb
                "normalize": 0.8,      # Normalization fairly common
                "loop_section": 0.3,   # Less often loop other instruments
                "duplicate_and_shift": 0.4,  # Sometimes duplicate
                "speed_change": 0.4    # Sometimes change speed
            })
        
        return weights
    
    def get_combined_distortions(self, num_distortions=3):
        """Apply multiple distortions, potentially all of them.
        
        Args:
            num_distortions: Base number of distortions to apply
            
        Returns:
            tuple: (distorted_audio, combined_actions)
                distorted_audio: The audio with all distortions applied
                combined_actions: List of actions that were applied, to be used as targets
        """
        distortion_methods = {
            "change_volume": self.change_volume_distortion,
            "pan": self.pan_distortion,
            "low_pass": self.low_pass_distortion,
            "high_pass": self.high_pass_distortion,
            "normalize": self.normalize_distortion,
            "delay": self.delay_distortion,
            "reverb": self.reverb_distortion,
            "compression": self.compression_distortion,
            "eq": self.eq_distortion,
            "trim": self.trim_distortion,
            "loop_section": self.loop_section_distortion,
            "duplicate_and_shift": self.duplicate_and_shift_distortion,
            "speed_change": self.speed_change_distortion
        }
        
        # Validate the audio segment before processing
        if len(self.audio) < 100:  # Extremely short samples won't work well
            # For very short audio, just use simple volume change
            distorted, action = self.change_volume_distortion()
            return distorted, [action]
        
        # Get weights based on stem type
        weights = self.get_stem_specific_distortion_weights()
        
        # 10% chance to apply ALL distortions with default values
        # This ensures the model learns to handle empty/default operations
        if random.random() < 0.1:
            all_distortions = True
            num_to_select = len(distortion_methods)
        else:
            all_distortions = False
            # Ensure num_distortions is within valid range
            num_to_select = max(1, min(num_distortions, len(distortion_methods)))
        
        # Initialize list to track which distortions should be applied
        selected_methods = []
        selected_method_names = []
        
        # If not applying all distortions, select based on probabilities
        if not all_distortions:
            # Create weighted items list
            methods_list = list(distortion_methods.keys())
            weights_list = [weights[name] for name in methods_list]
            
            # Select methods based on stem-specific weights without replacement
            while len(selected_method_names) < num_to_select and methods_list:
                # Choose a method based on weights
                method_name = random.choices(methods_list, weights=weights_list, k=1)[0]
                selected_method_names.append(method_name)
                selected_methods.append(distortion_methods[method_name])
                
                # Remove the chosen method to avoid duplicates
                idx = methods_list.index(method_name)
                methods_list.pop(idx)
                weights_list.pop(idx)
        else:
            # Apply all distortions in a specific order
            method_names = list(distortion_methods.keys())
            selected_method_names = method_names
            selected_methods = [distortion_methods[name] for name in method_names]
        
        # Start with the original audio
        combined_audio = self.audio
        combined_actions = []
        
        # Apply each selected distortion
        for i, method in enumerate(selected_methods):
            method_name = selected_method_names[i]
            
            # Create a new distorter with the current audio state
            temp_distorter = AudioDistorter(combined_audio, self.stem_type)
            
            # For all-distortions mode, only actually apply the distortion if random check passes
            # This simulates "no change needed" for some operations
            if all_distortions and random.random() > weights.get(method_name, 0.5):
                # Skip actually applying this distortion but still record the action with zeroed parameters
                if method_name == "change_volume":
                    action = {"tool": method_name, "params": {"db": 0.0}}
                elif method_name == "pan":
                    action = {"tool": method_name, "params": {"pan_value": 0.0}}
                elif method_name == "low_pass":
                    action = {"tool": method_name, "params": {"cutoff": 20000}}  # No effect at 20kHz
                elif method_name == "high_pass":
                    action = {"tool": method_name, "params": {"cutoff": 20}}     # No effect at 20Hz
                elif method_name == "eq":
                    action = {"tool": method_name, "params": {"low_gain": 0.0, "mid_gain": 0.0, "high_gain": 0.0}}
                elif method_name == "trim":
                    action = {"tool": method_name, "params": {"start_ms": 0, "end_ms": 0}}
                elif method_name == "loop_section":
                    action = {"tool": method_name, "params": {"start_ms": 0, "end_ms": 0, "count": 1}}
                elif method_name == "duplicate_and_shift":
                    action = {"tool": method_name, "params": {"offset_ms": 0}}
                elif method_name == "speed_change":
                    action = {"tool": method_name, "params": {"factor": 1.0}}
                else:
                    action = {"tool": method_name, "params": {}}
                
                combined_actions.append(action)
                continue
            
            # Apply the distortion method
            try:
                distorted, action = method.__get__(temp_distorter, AudioDistorter)()
                
                # Verify the distortion worked
                if distorted is not None and len(distorted) > 0:
                    combined_audio = distorted
                    combined_actions.append(action)
            except Exception as e:
                print(f"Error applying {method_name}: {e}")
                # Skip this distortion if it fails
                continue
        
        # If all distortions failed, apply a simple volume change
        if not combined_actions:
            distorted, action = self.change_volume_distortion()
            combined_audio = distorted
            combined_actions.append(action)
        
        return combined_audio, combined_actions 