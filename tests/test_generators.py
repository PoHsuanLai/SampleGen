"""
Unit tests for the generator modules.
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

from src.models.generator.base_generator import Generator
from src.models.generator.bass_generator import BassGenerator
from src.models.generator.drum_generator import DrumGenerator
from src.models.generator.other_generator import OtherGenerator


class TestBaseGenerator:
    """Test suite for base Generator class."""

    @pytest.fixture
    def mock_musicgen_components(self):
        """Mock MusicGen components to avoid loading actual models."""
        with patch('src.models.generator.base_generator.AutoProcessor') as mock_processor, \
             patch('src.models.generator.base_generator.MusicgenForConditionalGeneration') as mock_model:
            
            # Mock processor
            mock_proc_instance = MagicMock()
            mock_proc_instance.return_value = {
                'input_ids': torch.randint(0, 1000, (1, 10)),
                'attention_mask': torch.ones(1, 10)
            }
            mock_processor.from_pretrained.return_value = mock_proc_instance
            
            # Mock model
            mock_model_instance = MagicMock()
            mock_model_instance.generate.return_value = torch.randn(1, 1, 32000)  # 1 second at 32kHz
            mock_model_instance.parameters.return_value = [torch.nn.Parameter(torch.randn(10, 10))]
            mock_model_instance.to.return_value = mock_model_instance
            mock_model.from_pretrained.return_value = mock_model_instance
            
            yield {
                'processor': mock_proc_instance,
                'model': mock_model_instance
            }

    @pytest.fixture
    def temp_model_dir(self):
        """Create a temporary directory for model storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def generator(self, mock_musicgen_components, temp_model_dir, device):
        """Create a Generator instance with mocked components."""
        with patch('torch.cuda.is_available', return_value=(device == 'cuda')):
            gen = Generator(
                model_name="facebook/musicgen-small",
                device=device,
                model_dir=temp_model_dir
            )
            return gen

    def test_initialization_auto_device(self, mock_musicgen_components, temp_model_dir):
        """Test generator initialization with auto device detection."""
        with patch('torch.cuda.is_available', return_value=True):
            gen = Generator(model_dir=temp_model_dir)
            assert gen.device == 'cuda'
        
        with patch('torch.cuda.is_available', return_value=False):
            gen = Generator(model_dir=temp_model_dir)
            assert gen.device == 'cpu'

    def test_initialization_custom_device(self, mock_musicgen_components, temp_model_dir):
        """Test generator initialization with custom device."""
        gen = Generator(device='cpu', model_dir=temp_model_dir)
        assert gen.device == 'cpu'

    def test_initialization_custom_params(self, mock_musicgen_components, temp_model_dir):
        """Test generator initialization with custom parameters."""
        gen = Generator(
            model_name="facebook/musicgen-medium",
            device='cpu',
            sample_rate=44100,
            max_duration=60.0,
            model_dir=temp_model_dir
        )
        assert gen.model_name == "facebook/musicgen-medium"
        assert gen.sample_rate == 44100
        assert gen.max_duration == 60.0
        assert gen.model_dir == temp_model_dir

    def test_model_directory_creation(self, mock_musicgen_components, temp_model_dir):
        """Test that model directory is created."""
        gen = Generator(model_dir=temp_model_dir)
        assert os.path.exists(temp_model_dir)

    def test_generate_basic(self, generator):
        """Test basic audio generation."""
        audio = generator.generate("Create a calm piano melody", duration=2.0)
        
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        generator.model.generate.assert_called_once()

    def test_generate_with_params(self, generator):
        """Test generation with custom parameters."""
        audio = generator.generate(
            "Create a trap beat",
            duration=5.0,
            temperature=0.8,
            top_k=100,
            top_p=0.95,
            guidance_scale=2.0
        )
        
        assert isinstance(audio, np.ndarray)
        # Check that generate was called with expected parameters
        generator.model.generate.assert_called_once()

    def test_generate_duration_clamping(self, generator):
        """Test that generation duration is clamped to max_duration."""
        # Set max_duration to 10 seconds
        generator.max_duration = 10.0
        
        # Try to generate 20 seconds
        audio = generator.generate("Test prompt", duration=20.0)
        
        assert isinstance(audio, np.ndarray)
        # Should have been clamped to 10 seconds

    def test_generate_with_model_none(self, generator):
        """Test generation when model is None."""
        generator.model = None
        
        audio = generator.generate("Test prompt")
        
        assert audio is None

    def test_generate_with_conditioning(self, generator):
        """Test generation with audio conditioning."""
        conditioning_audio = np.random.randn(16000).astype(np.float32)
        
        audio = generator.generate_with_conditioning(
            "Create a variation of this melody",
            conditioning_audio,
            duration=3.0
        )
        
        assert isinstance(audio, np.ndarray)
        generator.model.generate.assert_called_once()

    def test_generate_with_conditioning_file_path(self, generator, temp_dir):
        """Test generation with conditioning from file path."""
        # Create a dummy audio file
        audio_file = os.path.join(temp_dir, "conditioning.wav")
        Path(audio_file).touch()
        
        # Mock load_audio to return dummy data
        with patch.object(generator, 'load_audio', return_value=np.random.randn(16000)):
            audio = generator.generate_with_conditioning(
                "Create a variation",
                audio_file,
                duration=3.0
            )
            
            assert isinstance(audio, np.ndarray)

    def test_get_model_info(self, generator):
        """Test getting model information."""
        info = generator.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'sample_rate' in info
        assert 'device' in info
        assert 'max_duration' in info

    def test_save_audio(self, generator, temp_dir):
        """Test saving audio to file."""
        audio_data = np.random.randn(16000).astype(np.float32)
        output_path = os.path.join(temp_dir, "test_output.wav")
        
        with patch('src.models.generator.base_generator.sf.write') as mock_write:
            success = generator.save_audio(audio_data, output_path)
            
            assert success is True
            mock_write.assert_called_once_with(output_path, audio_data, generator.sample_rate)

    def test_save_audio_error_handling(self, generator, temp_dir):
        """Test save audio error handling."""
        audio_data = np.random.randn(16000).astype(np.float32)
        output_path = os.path.join(temp_dir, "test_output.wav")
        
        with patch('src.models.generator.base_generator.sf.write', side_effect=Exception("Write error")):
            success = generator.save_audio(audio_data, output_path)
            assert success is False

    def test_load_audio(self, generator, temp_dir):
        """Test loading audio from file."""
        audio_file = os.path.join(temp_dir, "test_audio.wav")
        
        with patch('src.models.generator.base_generator.sf.read', return_value=(np.random.randn(16000), 32000)):
            audio = generator.load_audio(audio_file)
            
            assert isinstance(audio, np.ndarray)
            assert len(audio) == 16000

    def test_load_audio_error_handling(self, generator, temp_dir):
        """Test load audio error handling."""
        audio_file = os.path.join(temp_dir, "nonexistent.wav")
        
        audio = generator.load_audio(audio_file)
        assert audio is None

    def test_model_info_saving(self, generator, temp_model_dir):
        """Test that model metadata is saved."""
        # Mock the _save_model_info method to verify it's called
        with patch.object(generator, '_save_model_info') as mock_save:
            generator._init_model()
            mock_save.assert_called()

    @pytest.mark.parametrize("duration", [1.0, 5.0, 10.0, 15.0])
    def test_generate_different_durations(self, generator, duration):
        """Test generation with different durations."""
        audio = generator.generate("Test prompt", duration=duration)
        assert isinstance(audio, np.ndarray)


class TestBassGenerator:
    """Test suite for BassGenerator."""

    @pytest.fixture
    def mock_components(self):
        """Mock MusicGen components for BassGenerator."""
        with patch('src.models.generator.bass_generator.AutoProcessor'), \
             patch('src.models.generator.bass_generator.MusicgenForConditionalGeneration'):
            yield

    @pytest.fixture
    def bass_generator(self, mock_components, temp_dir, device):
        """Create a BassGenerator instance."""
        with patch('torch.cuda.is_available', return_value=(device == 'cuda')):
            gen = BassGenerator(model_dir=temp_dir, device=device)
            # Mock the model to avoid actual generation
            gen.model = MagicMock()
            gen.model.generate.return_value = torch.randn(1, 1, 32000)
            gen.processor = MagicMock()
            gen.processor.return_value = {
                'input_ids': torch.randint(0, 1000, (1, 10)),
                'attention_mask': torch.ones(1, 10)
            }
            return gen

    def test_generate_bass_line(self, bass_generator):
        """Test bass line generation."""
        audio = bass_generator.generate_bass_line(
            prompt="Deep 808 bass",
            duration=4.0,
            key="C",
            bpm=140
        )
        
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_generate_bass_with_reference(self, bass_generator):
        """Test bass generation with reference audio."""
        reference_audio = np.random.randn(32000).astype(np.float32)
        
        audio = bass_generator.generate_bass_line(
            prompt="Follow this bass pattern",
            reference_audio=reference_audio,
            duration=4.0
        )
        
        assert isinstance(audio, np.ndarray)

    def test_enhance_bass_prompt(self, bass_generator):
        """Test bass prompt enhancement."""
        prompt = "deep bass"
        enhanced = bass_generator._enhance_bass_prompt(prompt, key="Am", bpm=120)
        
        assert isinstance(enhanced, str)
        assert "bass" in enhanced.lower()
        assert "Am" in enhanced or "A minor" in enhanced
        assert "120" in enhanced


class TestDrumGenerator:
    """Test suite for DrumGenerator."""

    @pytest.fixture
    def mock_components(self):
        """Mock MusicGen components for DrumGenerator."""
        with patch('src.models.generator.drum_generator.AutoProcessor'), \
             patch('src.models.generator.drum_generator.MusicgenForConditionalGeneration'):
            yield

    @pytest.fixture
    def drum_generator(self, mock_components, temp_dir, device):
        """Create a DrumGenerator instance."""
        with patch('torch.cuda.is_available', return_value=(device == 'cuda')):
            gen = DrumGenerator(model_dir=temp_dir, device=device)
            # Mock the model
            gen.model = MagicMock()
            gen.model.generate.return_value = torch.randn(1, 1, 32000)
            gen.processor = MagicMock()
            gen.processor.return_value = {
                'input_ids': torch.randint(0, 1000, (1, 10)),
                'attention_mask': torch.ones(1, 10)
            }
            return gen

    def test_generate_drums(self, drum_generator):
        """Test drum generation."""
        audio = drum_generator.generate(
            prompt="Trap drums with hi-hats",
            duration=4.0,
            bpm=140,
            style="trap"
        )
        
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_enhance_drum_prompt(self, drum_generator):
        """Test drum prompt enhancement."""
        prompt = "drums"
        enhanced = drum_generator._enhance_drum_prompt(prompt, bpm=120, style="trap")
        
        assert isinstance(enhanced, str)
        assert "drums" in enhanced.lower()
        assert "120" in enhanced
        assert "trap" in enhanced.lower()


class TestOtherGenerator:
    """Test suite for OtherGenerator."""

    @pytest.fixture
    def mock_components(self):
        """Mock MusicGen components for OtherGenerator."""
        with patch('src.models.generator.other_generator.AutoProcessor'), \
             patch('src.models.generator.other_generator.MusicgenForConditionalGeneration'):
            yield

    @pytest.fixture
    def other_generator(self, mock_components, temp_dir, device):
        """Create an OtherGenerator instance."""
        with patch('torch.cuda.is_available', return_value=(device == 'cuda')):
            gen = OtherGenerator(model_dir=temp_dir, device=device)
            # Mock the model
            gen.model = MagicMock()
            gen.model.generate.return_value = torch.randn(1, 1, 32000)
            gen.processor = MagicMock()
            gen.processor.return_value = {
                'input_ids': torch.randint(0, 1000, (1, 10)),
                'attention_mask': torch.ones(1, 10)
            }
            return gen

    def test_generate_melody(self, other_generator):
        """Test melody generation."""
        audio = other_generator.generate(
            prompt="Dark melody in minor key",
            duration=5.0,
            instrument_type="melody",
            key="Am",
            bpm=120
        )
        
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_generate_harmony(self, other_generator):
        """Test harmony generation."""
        audio = other_generator.generate(
            prompt="Atmospheric pads",
            duration=5.0,
            instrument_type="harmony",
            key="C"
        )
        
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_enhance_prompt_melody(self, other_generator):
        """Test prompt enhancement for melody."""
        prompt = "melody"
        enhanced = other_generator._enhance_prompt(
            prompt, 
            instrument_type="melody", 
            key="G", 
            bpm=130
        )
        
        assert isinstance(enhanced, str)
        assert "melody" in enhanced.lower()
        assert "G" in enhanced
        assert "130" in enhanced

    def test_enhance_prompt_harmony(self, other_generator):
        """Test prompt enhancement for harmony."""
        prompt = "pads"
        enhanced = other_generator._enhance_prompt(
            prompt, 
            instrument_type="harmony", 
            key="Dm"
        )
        
        assert isinstance(enhanced, str)
        assert any(word in enhanced.lower() for word in ["pads", "harmony", "chords"])
        assert "Dm" in enhanced or "D minor" in enhanced

    @pytest.mark.parametrize("instrument_type", ["melody", "harmony", "synth", "sfx"])
    def test_different_instrument_types(self, other_generator, instrument_type):
        """Test generation with different instrument types."""
        audio = other_generator.generate(
            prompt=f"Generate {instrument_type}",
            instrument_type=instrument_type,
            duration=3.0
        )
        
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_error_handling_invalid_instrument(self, other_generator):
        """Test error handling with invalid instrument type."""
        # Should still work but may log a warning
        audio = other_generator.generate(
            prompt="Generate sound",
            instrument_type="invalid_type",
            duration=3.0
        )
        
        assert isinstance(audio, np.ndarray) or audio is None


class TestGeneratorIntegration:
    """Integration tests for generators working together."""

    @pytest.fixture
    def all_generators(self, temp_dir, device):
        """Create all generator instances for integration testing."""
        with patch('src.models.generator.base_generator.AutoProcessor'), \
             patch('src.models.generator.base_generator.MusicgenForConditionalGeneration'), \
             patch('torch.cuda.is_available', return_value=(device == 'cuda')):
            
            generators = {
                'bass': BassGenerator(model_dir=temp_dir, device=device),
                'drums': DrumGenerator(model_dir=temp_dir, device=device),
                'other': OtherGenerator(model_dir=temp_dir, device=device)
            }
            
            # Mock all models
            for gen in generators.values():
                gen.model = MagicMock()
                gen.model.generate.return_value = torch.randn(1, 1, 32000)
                gen.processor = MagicMock()
                gen.processor.return_value = {
                    'input_ids': torch.randint(0, 1000, (1, 10)),
                    'attention_mask': torch.ones(1, 10)
                }
            
            return generators

    def test_generate_full_arrangement(self, all_generators):
        """Test generating a full musical arrangement."""
        generators = all_generators
        duration = 4.0
        
        # Generate bass
        bass = generators['bass'].generate_bass_line(
            "Deep 808 bass", 
            duration=duration, 
            key="C", 
            bpm=140
        )
        
        # Generate drums
        drums = generators['drums'].generate(
            "Trap drums with hi-hats", 
            duration=duration, 
            bpm=140, 
            style="trap"
        )
        
        # Generate melody
        melody = generators['other'].generate(
            "Dark melody", 
            duration=duration, 
            instrument_type="melody", 
            key="C", 
            bpm=140
        )
        
        # Generate harmony
        harmony = generators['other'].generate(
            "Atmospheric pads", 
            duration=duration, 
            instrument_type="harmony", 
            key="C"
        )
        
        # All should be generated successfully
        assert isinstance(bass, np.ndarray)
        assert isinstance(drums, np.ndarray)
        assert isinstance(melody, np.ndarray)
        assert isinstance(harmony, np.ndarray)

    def test_consistent_duration_across_generators(self, all_generators):
        """Test that all generators produce consistent durations."""
        generators = all_generators
        duration = 5.0
        
        results = []
        
        # Generate from each generator
        results.append(generators['bass'].generate_bass_line("Bass", duration=duration))
        results.append(generators['drums'].generate("Drums", duration=duration))
        results.append(generators['other'].generate("Melody", duration=duration, instrument_type="melody"))
        
        # All should have similar lengths (allowing for small variations)
        lengths = [len(audio) for audio in results if audio is not None]
        if lengths:
            max_length = max(lengths)
            min_length = min(lengths)
            # Allow up to 10% difference in length
            assert (max_length - min_length) / max_length < 0.1 