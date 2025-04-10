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
        section = self.audio[start_ms:end_ms]
        self.audio = self.audio[:start_ms] + (section * count) + self.audio[end_ms:]

    def trim(self, start_ms: int, end_ms: int):
        self.audio = self.audio[start_ms:end_ms]

    def duplicate_and_shift(self, offset_ms: int):
        self.audio = self.audio.overlay(self.audio, position=offset_ms)

    def overlay_with(self, other_audio: AudioSegment, offset_ms: int = 0):
        self.audio = self.audio.overlay(other_audio, position=offset_ms)

    def export(self) -> AudioSegment:
        return self.audio
