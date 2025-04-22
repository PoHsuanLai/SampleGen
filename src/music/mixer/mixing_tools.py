from pydub import AudioSegment, effects
from pydub.generators import WhiteNoise
import random

class MixingTools:
    def __init__(self, audio: AudioSegment):
        self.audio = audio

    def change_volume(self, db: float):
        self.audio += db

    def pan(self, pan_value: float):
        """Pan from -1 (left) to 1 (right)"""
        self.audio = self.audio.pan(pan_value)

    def apply_low_pass(self, cutoff: int):
        self.audio = effects.low_pass_filter(self.audio, cutoff)

    def apply_high_pass(self, cutoff: int):
        self.audio = effects.high_pass_filter(self.audio, cutoff)

    def apply_normalize(self):
        self.audio = effects.normalize(self.audio)

    def apply_delay(self, delay_ms: int = 250, decay: float = 0.5):
        """Simple delay with feedback decay"""
        echo = self.audio - 10  # reduce echo level
        delayed = echo[::]
        for i in range(1, 4):
            delayed = delayed.overlay(echo - int(10 * decay * i), position=i * delay_ms)
        self.audio = self.audio.overlay(delayed)

    def apply_reverb(self):
        """Fake reverb via echo + slight stereo widening"""
        self.apply_delay(delay_ms=100, decay=0.3)
        self.pan(random.uniform(-0.1, 0.1))

    def apply_compression(self, threshold_db=-20, ratio=4.0):
        """Simplified soft-knee compressor logic placeholder"""
        current_db = self.audio.dBFS
        if current_db > threshold_db:
            gain_reduction = (current_db - threshold_db) * (1 - 1/ratio)
            self.change_volume(-gain_reduction)

    def apply_eq(self, low_gain=0, mid_gain=0, high_gain=0):
        """3-band EQ placeholder: split with HP/LP and gain adjustment"""
        low = effects.low_pass_filter(self.audio, 200).apply_gain(low_gain)
        mid = effects.high_pass_filter(self.audio, 200)
        mid = effects.low_pass_filter(mid, 3000).apply_gain(mid_gain)
        high = effects.high_pass_filter(self.audio, 3000).apply_gain(high_gain)
        self.audio = low.overlay(mid).overlay(high)

    def loop_section(self, start_ms: int, end_ms: int, count: int):
        """Loop a section of audio for the specified count.
        
        Args:
            start_ms: Start position in milliseconds
            end_ms: End position in milliseconds
            count: Number of times to loop the section
        """
        # If parameters indicate no looping, return unchanged
        if count <= 1 or start_ms >= end_ms or start_ms < 0:
            return
        
        # Validate end_ms is within audio length
        audio_length = len(self.audio)
        if end_ms <= 0:
            end_ms = audio_length
        else:
            end_ms = min(end_ms, audio_length)
            
        # Extract the section to loop
        section = self.audio[start_ms:end_ms]
        
        # If section is too short, don't loop
        if len(section) < 100:  # 100ms minimum
            return
            
        # Create the looped version
        result = self.audio[:start_ms]
        for _ in range(count):
            result += section
        if end_ms < audio_length:
            result += self.audio[end_ms:]
            
        self.audio = result

    def trim(self, start_ms: int, end_ms: int):
        """Trim audio by removing specified portions from start and end.
        
        Args:
            start_ms: Amount to trim from start in milliseconds
            end_ms: Amount to trim from end in milliseconds
        """
        total_len = len(self.audio)
        
        # Handle empty or invalid parameters
        if start_ms <= 0 and end_ms <= 0:
            return
            
        # Validate against audio length
        if start_ms >= total_len:
            start_ms = 0
        
        # Calculate end position
        if end_ms <= 0:
            end_pos = total_len
        else:
            end_pos = max(0, total_len - end_ms)
            
        # Apply trim
        if start_ms > 0 or end_pos < total_len:
            self.audio = self.audio[start_ms:end_pos]

    def duplicate_and_shift(self, offset_ms: int):
        """Duplicate the audio and overlay with an offset.
        
        Args:
            offset_ms: Time offset in milliseconds
        """
        # Skip if no offset
        if offset_ms <= 0:
            return
            
        # Create overlaid version
        self.audio = self.audio.overlay(self.audio, position=offset_ms)

    def speed_change(self, factor: float):
        """Change the speed/tempo of the audio.
        
        Args:
            factor: Speed factor (1.0 = no change, <1 = slower, >1 = faster)
        """
        # Skip if factor is 1.0 (no change)
        if abs(factor - 1.0) < 0.01:
            return
            
        # Limit factor to a reasonable range
        factor = max(0.5, min(2.0, factor))
        
        # Get the current frame rate
        original_frame_rate = self.audio.frame_rate
        
        # Calculate the new frame rate
        new_frame_rate = int(original_frame_rate * factor)
        
        # Create a new audio segment with the adjusted frame rate
        modified = self.audio._spawn(
            self.audio.raw_data,
            overrides={"frame_rate": new_frame_rate}
        )
        
        # Convert back to the original frame rate to change the speed
        self.audio = modified.set_frame_rate(original_frame_rate)

    def overlay_with(self, other_audio: AudioSegment, offset_ms: int = 0):
        self.audio = self.audio.overlay(other_audio, position=offset_ms)

    def export(self) -> AudioSegment:
        return self.audio
