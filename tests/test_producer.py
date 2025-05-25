"""
Unit tests for the HipHopProducerModel.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import tempfile
import os

from src.models.producer import HipHopProducerModel


class TestHipHopProducerModel:
    """Test suite for HipHopProducerModel."""
    
    @pytest.fixture
    def mock_components(self):
        """Mock all the heavy components to avoid loading actual models."""
        with patch('src.models.producer.Mixer') as mock_mixer_class, \
             patch('src.models.producer.BassGenerator') as mock_bass_gen, \
             patch('src.models.producer.DrumGenerator') as mock_drum_gen, \
             patch('src.models.producer.OtherGenerator') as mock_other_gen, \
             patch('src.models.producer.StemExtractor') as mock_stem_ext:
            
            # Configure mock mixer
            mock_mixer = MagicMock()
            mock_mixer.forward.return_value = "Generated plan text"
            mock_mixer.get_embedding.return_value = (
                torch.randn(1, 4096),  # text embedding
                torch.randn(1, 4096)   # audio embedding
            )
            mock_mixer_class.return_value = mock_mixer
            
            # Configure mock generators
            mock_bass = MagicMock()
            mock_bass.generate_bass_line.return_value = np.random.randn(44100).astype(np.float32)
            mock_bass_gen.return_value = mock_bass
            
            mock_drum = MagicMock()
            mock_drum.generate.return_value = np.random.randn(44100).astype(np.float32)
            mock_drum_gen.return_value = mock_drum
            
            mock_other = MagicMock()
            mock_other.generate.return_value = np.random.randn(44100).astype(np.float32)
            mock_other_gen.return_value = mock_other
            
            # Configure mock stem extractor
            mock_stems = MagicMock()
            mock_stems.extract_stems_from_file.return_value = (
                {
                    'vocals': np.random.randn(44100).astype(np.float32),
                    'drums': np.random.randn(44100).astype(np.float32),
                    'bass': np.random.randn(44100).astype(np.float32),
                    'other': np.random.randn(44100).astype(np.float32)
                },
                None
            )
            mock_stem_ext.return_value = mock_stems
            
            yield {
                'mixer': mock_mixer,
                'bass_gen': mock_bass,
                'drum_gen': mock_drum,
                'other_gen': mock_other,
                'stem_extractor': mock_stems
            }
    
    @pytest.fixture
    def producer_model(self, mock_components, device):
        """Create a HipHopProducerModel instance with mocked components."""
        model = HipHopProducerModel(device=device)
        return model
    
    def test_initialization(self, producer_model, device):
        """Test model initialization."""
        assert producer_model.device == device
        assert producer_model.sample_rate == 44100
        assert hasattr(producer_model, 'mixer')
        assert hasattr(producer_model, 'generators')
        assert hasattr(producer_model, 'stem_extractor')
        assert hasattr(producer_model, 'quality_head')
        
        # Check generators
        assert 'bass' in producer_model.generators
        assert 'drums' in producer_model.generators
        assert 'melody' in producer_model.generators
        assert 'harmony' in producer_model.generators
    
    def test_plan_production_text_only(self, producer_model, test_prompts):
        """Test production planning with text prompt only."""
        plan = producer_model.plan_production(test_prompts['style'])
        
        assert isinstance(plan, dict)
        assert 'generate_bass' in plan
        assert 'generate_drums' in plan
        assert 'generate_melody' in plan
        assert 'generate_harmony' in plan
        assert 'mixing_strategy' in plan
    
    def test_plan_production_with_audio(self, producer_model, test_wav_file, test_prompts):
        """Test production planning with audio input."""
        plan = producer_model.plan_production(
            text_prompt=test_prompts['style'],
            audio_file_path=test_wav_file
        )
        
        assert isinstance(plan, dict)
        # Should have called stem extractor
        producer_model.stem_extractor.extract_stems_from_file.assert_called_once()
    
    def test_plan_production_with_stems(self, producer_model, test_stems, test_prompts):
        """Test production planning with pre-extracted stems."""
        plan = producer_model.plan_production(
            text_prompt=test_prompts['style'],
            audio_stems=test_stems
        )
        
        assert isinstance(plan, dict)
        # Should not call stem extractor when stems provided
        assert not producer_model.stem_extractor.extract_stems_from_file.called
    
    def test_generate_stems(self, producer_model):
        """Test stem generation."""
        plan = {
            'generate_bass': "Deep 808 bass",
            'generate_drums': "Trap drums",
            'generate_melody': "Dark melody",
            'generate_harmony': "Atmospheric pads"
        }
        
        stems = producer_model.generate_stems(plan, duration=5.0)
        
        assert isinstance(stems, dict)
        assert len(stems) == 4  # All stems should be generated
        
        # Check that generators were called
        producer_model.generators['bass'].generate_bass_line.assert_called_once()
        producer_model.generators['drums'].generate.assert_called_once()
        producer_model.generators['melody'].generate.assert_called_once()
        producer_model.generators['harmony'].generate.assert_called_once()
    
    def test_generate_stems_partial(self, producer_model):
        """Test stem generation with only some stems requested."""
        plan = {
            'generate_bass': "Deep 808 bass",
            'generate_drums': None,  # Skip drums
            'generate_melody': None,
            'generate_harmony': "Atmospheric pads"
        }
        
        stems = producer_model.generate_stems(plan, duration=5.0)
        
        assert isinstance(stems, dict)
        assert len(stems) == 2  # Only bass and harmony
        assert 'bass' in stems
        assert 'harmony' in stems
        assert 'drums' not in stems
        assert 'melody' not in stems
    
    def test_create_mix(self, producer_model, test_stems, test_prompts):
        """Test audio mixing."""
        original_stems = test_stems
        generated_stems = {
            'bass': np.random.randn(44100).astype(np.float32),
            'drums': np.random.randn(44100).astype(np.float32)
        }
        
        mixed_audio = producer_model.create_mix(
            original_stems, generated_stems, test_prompts['style']
        )
        
        assert isinstance(mixed_audio, np.ndarray)
        assert len(mixed_audio) > 0
        assert mixed_audio.dtype == np.float64  # From _simple_mix
    
    def test_assess_quality(self, producer_model, test_audio_mono, test_prompts):
        """Test quality assessment."""
        quality_score = producer_model.assess_quality(
            test_audio_mono, test_prompts['style']
        )
        
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 1.0
    
    def test_iterative_refinement(self, producer_model, test_stems, test_prompts):
        """Test iterative refinement process."""
        original_stems = test_stems
        generated_stems = {
            'bass': np.random.randn(44100).astype(np.float32)
        }
        
        final_mix, final_quality = producer_model.iterative_refinement(
            original_stems, generated_stems, test_prompts['style'],
            max_iterations=2, quality_threshold=0.7
        )
        
        assert isinstance(final_mix, np.ndarray)
        assert isinstance(final_quality, float)
        assert 0.0 <= final_quality <= 1.0
    
    def test_simple_mix_empty_stems(self, producer_model):
        """Test simple mixing with empty stems."""
        mixed = producer_model._simple_mix({})
        assert isinstance(mixed, np.ndarray)
        assert len(mixed) == 44100  # 1 second of silence
        assert np.all(mixed == 0)
    
    def test_simple_mix_different_lengths(self, producer_model):
        """Test simple mixing with stems of different lengths."""
        stems = {
            'short': np.random.randn(22050).astype(np.float32),  # 0.5 seconds
            'long': np.random.randn(88200).astype(np.float32)    # 2 seconds
        }
        
        mixed = producer_model._simple_mix(stems)
        assert len(mixed) == 88200  # Should match longest stem
    
    def test_prepare_audio_for_mixer_empty(self, producer_model):
        """Test audio preparation with empty stems."""
        audio_tensor = producer_model._prepare_audio_for_mixer({})
        assert isinstance(audio_tensor, torch.Tensor)
        assert audio_tensor.shape == (1, 1, 1, 44100)  # Dummy tensor
    
    def test_prepare_audio_for_mixer_with_stems(self, producer_model, test_stems):
        """Test audio preparation with actual stems."""
        audio_tensor = producer_model._prepare_audio_for_mixer(test_stems)
        assert isinstance(audio_tensor, torch.Tensor)
        assert audio_tensor.shape[0] == 1  # Batch dimension
        assert audio_tensor.shape[1] == len(test_stems)  # Number of stems
    
    def test_create_planning_prompt(self, producer_model):
        """Test planning prompt creation."""
        prompt = producer_model._create_planning_prompt(
            "Dark trap beat", ['vocals', 'drums']
        )
        
        assert isinstance(prompt, str)
        assert "Dark trap beat" in prompt
        assert "vocals" in prompt
        assert "drums" in prompt
        assert "BASS:" in prompt
        assert "DRUMS:" in prompt
    
    def test_parse_plan_from_tokens(self, producer_model):
        """Test plan parsing from tokens."""
        plan = producer_model._parse_plan_from_tokens("mock tokens")
        
        assert isinstance(plan, dict)
        assert 'generate_bass' in plan
        assert 'generate_drums' in plan
        assert 'mixing_strategy' in plan
    
    def test_apply_simple_refinements(self, producer_model, test_audio_mono):
        """Test simple refinements application."""
        refined = producer_model._apply_simple_refinements(test_audio_mono)
        
        # Currently just returns original - test that it doesn't crash
        assert isinstance(refined, np.ndarray)
        assert np.array_equal(refined, test_audio_mono)
    
    def test_quality_head_forward(self, producer_model):
        """Test quality head neural network forward pass."""
        # Test with proper input size
        input_tensor = torch.randn(1, 8192).to(producer_model.device)  # Combined embedding size
        output = producer_model.quality_head(input_tensor)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 1)
        assert 0.0 <= output.item() <= 1.0  # Sigmoid output
    
    @pytest.mark.parametrize("duration", [1.0, 5.0, 10.0])
    def test_generate_stems_different_durations(self, producer_model, duration):
        """Test stem generation with different durations."""
        plan = {'generate_bass': "Deep bass"}
        
        stems = producer_model.generate_stems(plan, duration=duration)
        
        assert isinstance(stems, dict)
        # Verify duration was passed to generator
        producer_model.generators['bass'].generate_bass_line.assert_called_with(
            prompt="Deep bass", duration=duration
        )
    
    def test_error_handling_in_assess_quality(self, producer_model, test_audio_mono):
        """Test error handling in quality assessment."""
        # Mock the mixer to raise an exception
        producer_model.mixer.get_embedding.side_effect = Exception("Mock error")
        
        # Should not crash but handle gracefully
        try:
            quality = producer_model.assess_quality(test_audio_mono, "test prompt")
            # If it doesn't crash, that's good
            assert isinstance(quality, (int, float))
        except Exception:
            # If it does crash, that's a test failure
            pytest.fail("assess_quality should handle exceptions gracefully") 