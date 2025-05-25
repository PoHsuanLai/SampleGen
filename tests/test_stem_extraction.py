"""
Unit tests for the StemExtractor.
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import json
from unittest.mock import patch, MagicMock, Mock

from src.data_processing.stem_extraction import StemExtractor


class TestStemExtractor:
    """Test suite for StemExtractor class."""
    
    @pytest.fixture
    def mock_demucs(self):
        """Mock demucs module to avoid loading actual models."""
        with patch('src.data_processing.stem_extraction.get_model') as mock_get_model, \
             patch('src.data_processing.stem_extraction.apply_model') as mock_apply_model:
            
            # Mock model
            mock_model = MagicMock()
            mock_model.sources = ['drums', 'bass', 'other', 'vocals']
            mock_get_model.return_value = mock_model
            
            # Mock apply_model to return fake stems
            def mock_apply_fn(model, waveform, device=None, shifts=1, overlap=0.25):
                # Return mock separated stems
                batch_size, channels, length = waveform.shape
                sources = 4  # drums, bass, other, vocals
                return torch.randn(batch_size, sources, channels, length)
            
            mock_apply_model.side_effect = mock_apply_fn
            
            yield {
                'apply_model': mock_apply_model,
                'get_model': mock_get_model,
                'model': mock_model
            }
    
    @pytest.fixture
    def extractor(self, mock_demucs, device, sample_rate):
        """Create StemExtractor instance with mocked dependencies."""
        return StemExtractor(device=device, sample_rate=sample_rate)
    
    def test_initialization_default(self, mock_demucs, device):
        """Test StemExtractor initialization with default parameters."""
        extractor = StemExtractor(device=device)
        
        assert extractor.device == device
        assert extractor.sample_rate == 44100  # default
        assert hasattr(extractor, 'model')
    
    def test_initialization_custom(self, mock_demucs):
        """Test StemExtractor initialization with custom parameters."""
        custom_device = 'cpu'
        custom_sample_rate = 22050
        
        extractor = StemExtractor(
            device=custom_device,
            sample_rate=custom_sample_rate
        )
        
        assert extractor.device == custom_device
        assert extractor.sample_rate == custom_sample_rate
    
    def test_extract_stems_from_file(self, extractor, test_wav_file, temp_dir):
        """Test stem extraction from audio file."""
        # Mock the demucs.separate.main function since it's complex
        with patch('src.data_processing.stem_extraction.demucs_separate') as mock_separate:
            # Create mock stem files
            model_dir = os.path.join(temp_dir, extractor.model_name)
            filename = os.path.splitext(os.path.basename(test_wav_file))[0]
            stem_dir = os.path.join(model_dir, filename)
            os.makedirs(stem_dir, exist_ok=True)
            
            # Create mock stem files
            for source in ['drums', 'bass', 'other', 'vocals']:
                stem_path = os.path.join(stem_dir, f"{source}.wav")
                # Create a simple audio file
                mock_audio = np.random.randn(2, 44100).astype(np.float32)
                sf.write(stem_path, mock_audio.T, 44100)
            
            stems, _ = extractor.extract_stems_from_file(test_wav_file, output_dir=temp_dir)
            
            assert isinstance(stems, dict)
            assert len(stems) == 4
            assert all(stem in stems for stem in ['drums', 'bass', 'other', 'vocals'])
            
            for stem_name, stem_audio in stems.items():
                assert isinstance(stem_audio, np.ndarray)
                assert len(stem_audio) > 0
    
    def test_extract_stems_from_file_with_output_dir(self, extractor, test_wav_file, temp_dir):
        """Test stem extraction with output directory."""
        output_dir = os.path.join(temp_dir, "stems_output")
        
        stems, _ = extractor.extract_stems_from_file(test_wav_file, output_dir=output_dir)
        
        assert isinstance(stems, dict)
        assert len(stems) == 4
        assert os.path.exists(output_dir)
    
    def test_extract_stems_from_file_mono_input(self, extractor, temp_dir, test_audio_mono, sample_rate):
        """Test stem extraction from mono audio file."""
        # Create mono input file
        import soundfile as sf
        mono_file = os.path.join(temp_dir, "mono_input.wav")
        sf.write(mono_file, test_audio_mono, sample_rate)
        
        stems, _ = extractor.extract_stems_from_file(mono_file)
        
        assert isinstance(stems, dict)
        assert len(stems) == 4
    
    def test_extract_stems_from_file_stereo_input(self, extractor, temp_dir, test_audio_stereo, sample_rate):
        """Test stem extraction from stereo audio file."""
        # Create stereo input file
        import soundfile as sf
        stereo_file = os.path.join(temp_dir, "stereo_input.wav")
        sf.write(stereo_file, test_audio_stereo, sample_rate)
        
        stems, _ = extractor.extract_stems_from_file(stereo_file)
        
        assert isinstance(stems, dict)
        assert len(stems) == 4
    
    def test_generate_stem_unsupported_type(self, extractor, test_prompts):
        """Test stem generation with unsupported stem type."""
        with patch('src.data_processing.stem_extraction.MelodyGenerator'):
            audio, _ = extractor.generate_stem(
                prompt=test_prompts['generation'],
                stem_type='unsupported',
                duration=2.0
            )
            
            # Should return None for unsupported types
            assert audio is None
    
    @pytest.mark.parametrize("file_format", [".wav", ".flac"])
    def test_extract_stems_different_formats(self, extractor, temp_dir, test_audio_mono, sample_rate, file_format):
        """Test stem extraction with different audio formats."""
        # Create test file in specified format
        import soundfile as sf
        test_file = os.path.join(temp_dir, f"test{file_format}")
        sf.write(test_file, test_audio_mono, sample_rate)
        
        stems, _ = extractor.extract_stems_from_file(test_file)
        
        assert isinstance(stems, dict)
        assert len(stems) == 4
    
    def test_error_handling_invalid_file(self, extractor):
        """Test error handling with invalid file path."""
        # This should not raise an exception but return empty results
        stems, _ = extractor.extract_stems_from_file("nonexistent_file.wav")
        
        # The actual implementation handles this gracefully
        assert isinstance(stems, dict)
    
    def test_device_handling(self, mock_demucs):
        """Test device handling for different devices."""
        for device in ['cpu', 'cuda']:
            if device == 'cuda' and not torch.cuda.is_available():
                continue
                
            extractor = StemExtractor(device=device)
            assert extractor.device == device
    
    def test_sample_rate_handling(self, extractor, temp_dir, test_audio_mono):
        """Test handling of different sample rates."""
        # Create file with different sample rate
        import soundfile as sf
        different_sr = 22050
        test_file = os.path.join(temp_dir, "different_sr.wav")
        sf.write(test_file, test_audio_mono[:22050], different_sr)
        
        stems, _ = extractor.extract_stems_from_file(test_file)
        
        assert isinstance(stems, dict)
        assert len(stems) == 4
    
    def test_memory_efficiency_large_file(self, extractor, temp_dir, sample_rate):
        """Test memory efficiency with large audio file."""
        # Create a larger audio file (10 seconds)
        import soundfile as sf
        long_audio = np.random.randn(sample_rate * 10).astype(np.float32)
        large_file = os.path.join(temp_dir, "large_audio.wav")
        sf.write(large_file, long_audio, sample_rate)
        
        stems, _ = extractor.extract_stems_from_file(large_file)
        
        assert isinstance(stems, dict)
        assert len(stems) == 4
        
        # Check that stems have reasonable length
        for stem_audio in stems.values():
            assert len(stem_audio) > 0
    
    def test_get_available_models(self, extractor):
        """Test getting available models."""
        models = extractor.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
    
    def test_get_model_sources(self, extractor):
        """Test getting model sources."""
        sources = extractor.get_model_sources()
        
        assert isinstance(sources, list)
        assert len(sources) == 4
        assert all(source in sources for source in ['drums', 'bass', 'other', 'vocals']) 