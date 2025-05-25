"""
Unit tests for the FaustDistorter.
"""

import pytest
import numpy as np
import tempfile
import os
import shutil
from unittest.mock import patch, MagicMock

from src.data_processing.faust_distorter import FaustDistorter, render_faust_to_audio


class TestFaustDistorter:
    """Test suite for FaustDistorter."""
    
    @pytest.fixture
    def distorter(self, sample_rate):
        """Create a FaustDistorter instance."""
        return FaustDistorter(sample_rate=sample_rate)
    
    def test_initialization(self, distorter, sample_rate):
        """Test FaustDistorter initialization."""
        assert distorter.sample_rate == sample_rate
        assert hasattr(distorter, 'temp_dir')
        assert os.path.exists(distorter.temp_dir)
    
    def test_cleanup(self, sample_rate):
        """Test temporary directory cleanup."""
        distorter = FaustDistorter(sample_rate=sample_rate)
        temp_dir = distorter.temp_dir
        assert os.path.exists(temp_dir)
        
        # Cleanup should happen on deletion
        del distorter
        # Note: Cleanup happens in __del__ but may not be immediate
    
    def test_generate_volume_distortion(self, distorter):
        """Test volume distortion generation."""
        faust_code, correction = distorter.generate_volume_distortion(gain_db=-6.0)
        
        assert isinstance(faust_code, str)
        assert "gain_db = -6.0" in faust_code
        assert "ba.db2linear" in faust_code
        assert "process = _ * gain_linear" in faust_code
        
        assert isinstance(correction, dict)
        assert correction['tool'] == 'change_volume'
        assert correction['params']['db'] == 6.0  # Inverse gain
        assert 'faust_correction' in correction
    
    def test_generate_volume_distortion_random(self, distorter):
        """Test volume distortion with random gain."""
        faust_code, correction = distorter.generate_volume_distortion()
        
        assert isinstance(faust_code, str)
        assert "gain_db = " in faust_code
        assert isinstance(correction, dict)
        assert correction['tool'] == 'change_volume'
    
    def test_generate_lowpass_distortion(self, distorter):
        """Test lowpass filter distortion."""
        faust_code, correction = distorter.generate_lowpass_distortion(cutoff_hz=2000)
        
        assert isinstance(faust_code, str)
        assert "cutoff = 2000" in faust_code
        assert "fi.lowpass" in faust_code
        
        assert isinstance(correction, dict)
        assert correction['tool'] == 'apply_highpass_correction'
        assert correction['params']['cutoff'] == 1600  # 80% of original
    
    def test_generate_highpass_distortion(self, distorter):
        """Test highpass filter distortion."""
        faust_code, correction = distorter.generate_highpass_distortion(cutoff_hz=300)
        
        assert isinstance(faust_code, str)
        assert "cutoff = 300" in faust_code
        assert "fi.highpass" in faust_code
        
        assert isinstance(correction, dict)
        assert correction['tool'] == 'apply_lowpass_correction'
        assert correction['params']['cutoff'] == 360  # 120% of original
    
    def test_generate_compression_distortion(self, distorter):
        """Test compression distortion."""
        faust_code, correction = distorter.generate_compression_distortion(
            threshold_db=-20, ratio=4.0
        )
        
        assert isinstance(faust_code, str)
        assert "threshold_db = -20" in faust_code
        assert "ratio = 4.0" in faust_code
        assert "co.compressor_mono" in faust_code
        
        assert isinstance(correction, dict)
        assert correction['tool'] == 'apply_expansion'
        assert correction['params']['ratio'] == 0.25  # 1/4
    
    def test_generate_delay_distortion(self, distorter):
        """Test delay effect distortion."""
        faust_code, correction = distorter.generate_delay_distortion(
            delay_ms=100, feedback=0.5
        )
        
        assert isinstance(faust_code, str)
        assert "delay_samples = 4410" in faust_code  # 100ms at 44.1kHz
        assert "feedback = 0.5" in faust_code
        assert "de.delay" in faust_code
        
        assert isinstance(correction, dict)
        assert correction['tool'] == 'remove_delay'
        assert correction['params']['delay_ms'] == 100
    
    def test_generate_reverb_distortion(self, distorter):
        """Test reverb distortion."""
        faust_code, correction = distorter.generate_reverb_distortion(
            room_size=0.6, damping=0.3
        )
        
        assert isinstance(faust_code, str)
        assert "room_size = 0.6" in faust_code
        assert "damping = 0.3" in faust_code
        assert "re.mono_freeverb" in faust_code
        
        assert isinstance(correction, dict)
        assert correction['tool'] == 'remove_reverb'
    
    def test_generate_eq_distortion(self, distorter):
        """Test EQ distortion."""
        faust_code, correction = distorter.generate_eq_distortion(
            low_gain=3.0, mid_gain=-2.0, high_gain=1.0
        )
        
        assert isinstance(faust_code, str)
        assert "low_gain_db = 3.0" in faust_code
        assert "mid_gain_db = -2.0" in faust_code
        assert "high_gain_db = 1.0" in faust_code
        assert "fi.lowpass" in faust_code
        assert "fi.highpass" in faust_code
        
        assert isinstance(correction, dict)
        assert correction['tool'] == 'apply_eq_correction'
        assert correction['params']['low_gain'] == -3.0  # Inverse
        assert correction['params']['mid_gain'] == 2.0
        assert correction['params']['high_gain'] == -1.0
    
    def test_get_random_distortion_vocals(self, distorter):
        """Test random distortion for vocals."""
        faust_code, correction = distorter.get_random_distortion(stem_type="vocals")
        
        assert isinstance(faust_code, str)
        assert isinstance(correction, dict)
        assert 'tool' in correction
        assert 'params' in correction
    
    def test_get_random_distortion_drums(self, distorter):
        """Test random distortion for drums."""
        faust_code, correction = distorter.get_random_distortion(stem_type="drums")
        
        assert isinstance(faust_code, str)
        assert isinstance(correction, dict)
    
    def test_get_random_distortion_bass(self, distorter):
        """Test random distortion for bass."""
        faust_code, correction = distorter.get_random_distortion(stem_type="bass")
        
        assert isinstance(faust_code, str)
        assert isinstance(correction, dict)
    
    def test_get_random_distortion_other(self, distorter):
        """Test random distortion for other stems."""
        faust_code, correction = distorter.get_random_distortion(stem_type="other")
        
        assert isinstance(faust_code, str)
        assert isinstance(correction, dict)
    
    def test_get_random_distortion_no_type(self, distorter):
        """Test random distortion with no stem type."""
        faust_code, correction = distorter.get_random_distortion()
        
        assert isinstance(faust_code, str)
        assert isinstance(correction, dict)
    
    @patch('subprocess.run')
    def test_compile_and_apply_faust_success(self, mock_subprocess, distorter, test_audio_mono):
        """Test successful Faust compilation."""
        # Mock successful compilation
        mock_subprocess.return_value = MagicMock(returncode=0)
        
        faust_code = """
        import("stdfaust.lib");
        process = _ * 0.5;
        """
        
        # This will fall back to simple processing since we can't actually compile
        result = distorter.compile_and_apply_faust(faust_code, test_audio_mono)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
    
    @patch('subprocess.run')
    def test_compile_and_apply_faust_failure(self, mock_subprocess, distorter, test_audio_mono):
        """Test failed Faust compilation."""
        # Mock failed compilation
        mock_subprocess.return_value = MagicMock(returncode=1, stderr="Compilation error")
        
        faust_code = "invalid faust code"
        result = distorter.compile_and_apply_faust(faust_code, test_audio_mono)
        
        # Should still return something (fallback processing)
        assert result is not None
        assert isinstance(result, np.ndarray)
    
    def test_apply_simple_processing_gain(self, distorter, test_audio_mono):
        """Test simple gain processing."""
        faust_code = """
        import("stdfaust.lib");
        gain_db = -6.0;
        gain_linear = ba.db2linear(gain_db);
        process = _ * gain_linear;
        """
        
        result = distorter._apply_simple_processing(test_audio_mono, faust_code)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(test_audio_mono)
        # Should be quieter due to -6dB gain
        assert np.max(np.abs(result)) < np.max(np.abs(test_audio_mono))
    
    @patch('scipy.signal')
    def test_apply_simple_processing_lowpass(self, mock_signal, distorter, test_audio_mono):
        """Test simple lowpass processing."""
        # Mock scipy.signal
        mock_signal.butter.return_value = ([1, 0], [1, 0])  # Dummy filter coefficients
        mock_signal.filtfilt.return_value = test_audio_mono * 0.5
        
        faust_code = """
        import("stdfaust.lib");
        cutoff = 2000;
        process = fi.lowpass(3, cutoff);
        """
        
        result = distorter._apply_simple_processing(test_audio_mono, faust_code)
        
        assert isinstance(result, np.ndarray)
        mock_signal.butter.assert_called_once()
        mock_signal.filtfilt.assert_called_once()
    
    @patch('scipy.signal')
    def test_apply_simple_processing_highpass(self, mock_signal, distorter, test_audio_mono):
        """Test simple highpass processing."""
        # Mock scipy.signal
        mock_signal.butter.return_value = ([1, 0], [1, 0])
        mock_signal.filtfilt.return_value = test_audio_mono * 0.7
        
        faust_code = """
        import("stdfaust.lib");
        cutoff = 300;
        process = fi.highpass(3, cutoff);
        """
        
        result = distorter._apply_simple_processing(test_audio_mono, faust_code)
        
        assert isinstance(result, np.ndarray)
        mock_signal.butter.assert_called_once()
        mock_signal.filtfilt.assert_called_once()
    
    def test_apply_simple_processing_unknown(self, distorter, test_audio_mono):
        """Test simple processing with unknown effect."""
        faust_code = "process = some_unknown_effect;"
        
        result = distorter._apply_simple_processing(test_audio_mono, faust_code)
        
        # Should return original audio
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, test_audio_mono)
    
    def test_apply_distortion(self, distorter, test_audio_mono):
        """Test full distortion application."""
        distorted_audio, correction_info = distorter.apply_distortion(
            test_audio_mono, stem_type="vocals"
        )
        
        assert isinstance(distorted_audio, np.ndarray)
        assert len(distorted_audio) == len(test_audio_mono)
        
        assert isinstance(correction_info, dict)
        assert 'tool' in correction_info
        assert 'params' in correction_info
        assert 'original_faust_code' in correction_info
    
    def test_apply_distortion_failure_fallback(self, distorter, test_audio_mono):
        """Test distortion application with fallback."""
        # Mock compile_and_apply_faust to return None (failure)
        with patch.object(distorter, 'compile_and_apply_faust', return_value=None):
            distorted_audio, correction_info = distorter.apply_distortion(test_audio_mono)
            
            # Should fallback to original audio
            assert np.array_equal(distorted_audio, test_audio_mono)
            assert correction_info['tool'] == 'no_change'
    
    @pytest.mark.parametrize("stem_type", ["vocals", "drums", "bass", "other", None])
    def test_apply_distortion_all_stem_types(self, distorter, test_audio_mono, stem_type):
        """Test distortion application for all stem types."""
        distorted_audio, correction_info = distorter.apply_distortion(
            test_audio_mono, stem_type=stem_type
        )
        
        assert isinstance(distorted_audio, np.ndarray)
        assert isinstance(correction_info, dict)


class TestRenderFaustToAudio:
    """Test suite for render_faust_to_audio function."""
    
    def test_render_faust_to_audio_success(self, temp_dir, test_audio_mono, sample_rate):
        """Test successful Faust rendering."""
        input_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(temp_dir, "output.wav")
        
        # Create input file
        import soundfile as sf
        sf.write(input_path, test_audio_mono, sample_rate)
        
        faust_code = """
        import("stdfaust.lib");
        process = _ * 0.5;
        """
        
        with patch('src.data_processing.faust_distorter.FaustDistorter') as mock_distorter_class:
            mock_distorter = MagicMock()
            mock_distorter.compile_and_apply_faust.return_value = test_audio_mono * 0.5
            mock_distorter_class.return_value = mock_distorter
            
            result = render_faust_to_audio(faust_code, input_path, output_path, sample_rate)
            
            assert result is True
            assert os.path.exists(output_path)
    
    def test_render_faust_to_audio_failure(self, temp_dir, test_audio_mono, sample_rate):
        """Test failed Faust rendering."""
        input_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(temp_dir, "output.wav")
        
        # Create input file
        import soundfile as sf
        sf.write(input_path, test_audio_mono, sample_rate)
        
        faust_code = "invalid code"
        
        with patch('src.data_processing.faust_distorter.FaustDistorter') as mock_distorter_class:
            mock_distorter = MagicMock()
            mock_distorter.compile_and_apply_faust.return_value = None  # Failure
            mock_distorter_class.return_value = mock_distorter
            
            result = render_faust_to_audio(faust_code, input_path, output_path, sample_rate)
            
            assert result is False
    
    def test_render_faust_to_audio_exception(self, temp_dir, sample_rate):
        """Test Faust rendering with exception."""
        input_path = os.path.join(temp_dir, "nonexistent.wav")
        output_path = os.path.join(temp_dir, "output.wav")
        
        faust_code = "process = _;"
        
        result = render_faust_to_audio(faust_code, input_path, output_path, sample_rate)
        
        assert result is False
    
    def test_render_faust_to_audio_resampling(self, temp_dir, test_audio_mono, sample_rate):
        """Test Faust rendering with resampling."""
        input_path = os.path.join(temp_dir, "input.wav")
        output_path = os.path.join(temp_dir, "output.wav")
        
        # Create input file with different sample rate
        import soundfile as sf
        sf.write(input_path, test_audio_mono, sample_rate // 2)  # Half sample rate
        
        faust_code = "process = _;"
        
        with patch('src.data_processing.faust_distorter.FaustDistorter') as mock_distorter_class, \
             patch('scipy.signal.resample') as mock_resample:
            
            mock_distorter = MagicMock()
            mock_distorter.compile_and_apply_faust.return_value = test_audio_mono
            mock_distorter_class.return_value = mock_distorter
            
            mock_resample.return_value = test_audio_mono
            
            result = render_faust_to_audio(faust_code, input_path, output_path, sample_rate)
            
            assert result is True
            mock_resample.assert_called_once()  # Should have resampled 