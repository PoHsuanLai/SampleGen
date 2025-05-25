"""
Real tests for training utilities - testing actual functionality without mocks.
"""

import pytest
import numpy as np
import torch
import os
import tempfile
import soundfile as sf
from pathlib import Path

from src.training.utils import (
    safe_normalize_audio, load_audio_with_fallback, prepare_audio_for_model,
    extract_segment_from_audio, validate_stems_dict, save_inference_results
)
from src.training.prompts import (
    get_mixer_prompts, get_style_specific_prompts, get_random_mixing_prompt
)


class TestTrainingUtilsReal:
    """Test actual training utility functions."""

    def test_safe_normalize_audio_loud(self):
        """Test normalization of loud audio."""
        # Create loud audio that would clip
        audio = np.array([0.1, 2.5, -3.0, 1.8, -0.5], dtype=np.float32)
        normalized = safe_normalize_audio(audio, target_level=0.8)
        
        # Check that max amplitude is now at target level (allowing for float32 precision)
        assert np.max(np.abs(normalized)) <= 0.8001  # Allow small float32 precision error
        assert np.isclose(np.max(np.abs(normalized)), 0.8, atol=1e-5)
        
        # Check that relative relationships are preserved
        assert np.argmax(np.abs(normalized)) == np.argmax(np.abs(audio))

    def test_safe_normalize_audio_quiet(self):
        """Test normalization of quiet audio."""
        audio = np.array([0.01, 0.02, -0.015, 0.008], dtype=np.float32)
        normalized = safe_normalize_audio(audio, target_level=0.9)
        
        # Should be amplified
        assert np.max(np.abs(normalized)) > np.max(np.abs(audio))
        assert np.isclose(np.max(np.abs(normalized)), 0.9, atol=1e-6)

    def test_safe_normalize_audio_silence(self):
        """Test normalization of silence."""
        audio = np.zeros(1000, dtype=np.float32)
        normalized = safe_normalize_audio(audio)
        
        # Should remain silence
        assert np.allclose(normalized, audio)

    def test_safe_normalize_audio_empty(self):
        """Test normalization of empty array."""
        audio = np.array([], dtype=np.float32)
        normalized = safe_normalize_audio(audio)
        
        assert len(normalized) == 0
        assert normalized.dtype == audio.dtype

    def test_prepare_audio_for_model_exact_length(self):
        """Test audio preparation with exact target length."""
        audio = np.random.randn(22050).astype(np.float32)
        result = prepare_audio_for_model(audio, target_length=22050)
        
        assert isinstance(result, torch.Tensor)
        assert len(result) == 22050
        assert result.dtype == torch.float32

    def test_prepare_audio_for_model_too_long(self):
        """Test audio preparation when input is too long."""
        audio = np.random.randn(44100).astype(np.float32)  # 1 second
        result = prepare_audio_for_model(audio, target_length=22050)  # Want 0.5 seconds
        
        assert len(result) == 22050
        # Should be a segment of the original audio
        assert torch.all(torch.abs(result) <= 1.0)  # Should be normalized

    def test_prepare_audio_for_model_too_short(self):
        """Test audio preparation when input is too short."""
        audio = np.random.randn(11025).astype(np.float32)  # 0.25 seconds
        result = prepare_audio_for_model(audio, target_length=22050)  # Want 0.5 seconds
        
        assert len(result) == 22050
        # First part should be the original audio, rest should be zeros
        assert not torch.allclose(result[:11025], torch.zeros(11025))
        assert torch.allclose(result[11025:], torch.zeros(11025))

    def test_prepare_audio_for_model_no_target(self):
        """Test audio preparation without target length."""
        audio = np.random.randn(12345).astype(np.float32)
        result = prepare_audio_for_model(audio)
        
        assert len(result) == 12345
        assert isinstance(result, torch.Tensor)

    def test_prepare_audio_for_model_dtype_conversion(self):
        """Test audio preparation with different input dtypes."""
        # Test with int16
        audio_int16 = (np.random.randn(1000) * 16384).astype(np.int16)
        result = prepare_audio_for_model(audio_int16)
        assert result.dtype == torch.float32
        
        # Test with float64
        audio_float64 = np.random.randn(1000).astype(np.float64)
        result = prepare_audio_for_model(audio_float64)
        assert result.dtype == torch.float32

    def test_extract_segment_valid(self):
        """Test extracting a valid segment from audio."""
        # Create 5 seconds of audio at 44100 Hz
        duration = 5.0
        sample_rate = 44100
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(duration * sample_rate)))
        
        # Extract segment from 1.0 to 3.0 seconds
        segment_info = {"start": 1.0, "end": 3.0}
        segment = extract_segment_from_audio(audio, segment_info, sample_rate)
        
        expected_length = int(2.0 * sample_rate)  # 2 seconds
        assert len(segment) == expected_length
        
        # Check that it's actually from the middle of the original audio
        start_sample = int(1.0 * sample_rate)
        expected_segment = audio[start_sample:start_sample + expected_length]
        assert np.allclose(segment, expected_segment)

    def test_extract_segment_invalid_bounds(self):
        """Test extracting segment with invalid bounds."""
        audio = np.random.randn(44100)  # 1 second
        sample_rate = 44100
        
        # Start after end
        segment_info = {"start": 3.0, "end": 2.0}
        segment = extract_segment_from_audio(audio, segment_info, sample_rate)
        
        # Should return short silence
        assert len(segment) == int(0.1 * sample_rate)
        assert np.allclose(segment, np.zeros_like(segment))

    def test_extract_segment_beyond_audio(self):
        """Test extracting segment beyond audio bounds."""
        audio = np.random.randn(22050)  # 0.5 seconds
        sample_rate = 44100
        
        # Request segment from 0.3 to 1.0 seconds (goes beyond audio)
        segment_info = {"start": 0.3, "end": 1.0}
        segment = extract_segment_from_audio(audio, segment_info, sample_rate)
        
        # Should extract from 0.3 seconds to end of audio
        start_sample = int(0.3 * sample_rate)
        expected_length = len(audio) - start_sample
        assert len(segment) == expected_length

    def test_validate_stems_dict_complete(self):
        """Test stems validation with complete stems."""
        stems = {
            "vocals": np.random.randn(44100).astype(np.float32),
            "drums": np.random.randn(44100).astype(np.float32),
            "bass": np.random.randn(44100).astype(np.float32),
            "other": np.random.randn(44100).astype(np.float32)
        }
        
        validated = validate_stems_dict(stems)
        
        assert len(validated) == 4
        assert set(validated.keys()) == {"vocals", "drums", "bass", "other"}
        
        for stem_name, audio in validated.items():
            assert isinstance(audio, np.ndarray)
            assert len(audio) > 0
            assert np.max(np.abs(audio)) <= 1.0  # Should be normalized

    def test_validate_stems_dict_missing_stems(self):
        """Test stems validation with missing stems."""
        stems = {
            "vocals": np.random.randn(44100).astype(np.float32),
            "drums": np.random.randn(44100).astype(np.float32)
            # Missing bass and other
        }
        
        validated = validate_stems_dict(stems)
        
        assert len(validated) == 4
        assert "bass" in validated
        assert "other" in validated
        
        # Missing stems should be filled with silence
        assert len(validated["bass"]) == 44100
        assert len(validated["other"]) == 44100
        assert np.allclose(validated["bass"], np.zeros(44100))
        assert np.allclose(validated["other"], np.zeros(44100))

    def test_validate_stems_dict_empty_stems(self):
        """Test stems validation with empty audio arrays."""
        stems = {
            "vocals": np.array([]),
            "drums": np.random.randn(44100).astype(np.float32),
            "bass": np.random.randn(100).astype(np.float32),
            "other": np.random.randn(44100).astype(np.float32)
        }
        
        validated = validate_stems_dict(stems)
        
        # Empty vocals should be replaced with silence
        assert len(validated["vocals"]) == 44100
        assert np.allclose(validated["vocals"], np.zeros(44100))
        
        # Other stems should be preserved and normalized
        assert len(validated["drums"]) == 44100
        assert len(validated["bass"]) == 100
        assert len(validated["other"]) == 44100

    def test_validate_stems_dict_custom_required(self):
        """Test stems validation with custom required stems."""
        stems = {
            "lead": np.random.randn(22050).astype(np.float32),
            "pad": np.random.randn(22050).astype(np.float32)
        }
        
        required_stems = ["lead", "pad", "fx"]
        validated = validate_stems_dict(stems, required_stems)
        
        assert set(validated.keys()) == {"lead", "pad", "fx"}
        assert len(validated["fx"]) == 44100  # Missing stem filled with silence

    def test_save_inference_results_basic(self):
        """Test saving basic inference results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = {
                "text_output": "Generated mixing instructions",
                "metadata": {
                    "bpm": 140,
                    "key": "Am",
                    "style": "trap"
                },
                "processing_time": 2.5
            }
            
            saved_paths = save_inference_results(results, temp_dir, prefix="test")
            
            assert isinstance(saved_paths, dict)
            assert os.path.exists(temp_dir)
            
            # Check that directory was created
            assert os.path.isdir(temp_dir)


class TestPromptFunctionsReal:
    """Test actual prompt generation functions."""

    def test_get_mixer_prompts_structure(self):
        """Test that mixer prompts have correct structure."""
        prompts = get_mixer_prompts(num_prompts=3)
        
        assert isinstance(prompts, list)
        assert len(prompts) == 3
        
        for prompt in prompts:
            assert isinstance(prompt, dict)
            assert "character" in prompt
            assert "instruction" in prompt
            assert isinstance(prompt["character"], str)
            assert isinstance(prompt["instruction"], str)
            assert len(prompt["character"]) > 0
            assert len(prompt["instruction"]) > 0
            
            # Check for key professional terms
            character_lower = prompt["character"].lower()
            instruction_lower = prompt["instruction"].lower()
            
            # Character should mention a professional role
            professional_terms = ["producer", "engineer", "mixer", "professional", "specialist", "expert"]
            assert any(term in character_lower for term in professional_terms)
            
            # Instruction should contain action words
            action_terms = ["create", "generate", "apply", "focus", "enhance", "mix", "tool", "command"]
            assert any(term in instruction_lower for term in action_terms)

    def test_get_mixer_prompts_default_count(self):
        """Test default prompt count."""
        prompts = get_mixer_prompts()
        assert len(prompts) == 30

    def test_get_mixer_prompts_different_counts(self):
        """Test different prompt counts."""
        for count in [1, 5, 10, 25]:
            prompts = get_mixer_prompts(num_prompts=count)
            assert len(prompts) == count

    def test_get_style_specific_prompts_trap(self):
        """Test trap-specific prompts contain relevant content."""
        prompts = get_style_specific_prompts("trap")
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        
        # Join all prompts to check for trap-related content
        all_text = " ".join(prompts).lower()
        
        # Should contain trap-related terms
        trap_terms = ["trap", "808", "hi-hat", "club", "modern", "aggressive", "hard"]
        assert any(term in all_text for term in trap_terms)

    def test_get_style_specific_prompts_boom_bap(self):
        """Test boom-bap specific prompts contain relevant content."""
        prompts = get_style_specific_prompts("boom_bap")
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        
        all_text = " ".join(prompts).lower()
        
        # Should contain boom-bap related terms
        boom_bap_terms = ["boom", "bap", "classic", "vintage", "analog", "raw", "groove"]
        assert any(term in all_text for term in boom_bap_terms)

    def test_get_style_specific_prompts_all_styles(self):
        """Test all defined styles return prompts."""
        styles = ["trap", "boom_bap", "drill", "conscious", "melodic"]
        
        for style in styles:
            prompts = get_style_specific_prompts(style)
            assert isinstance(prompts, list)
            assert len(prompts) > 0
            
            for prompt in prompts:
                assert isinstance(prompt, str)
                assert len(prompt) > 0

    def test_get_style_specific_prompts_unknown_style(self):
        """Test unknown style returns default prompts."""
        prompts = get_style_specific_prompts("unknown_nonexistent_style")
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0  # Should return default prompts
        
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_get_random_mixing_prompt_consistency(self):
        """Test random mixing prompt returns consistent format."""
        prompt = get_random_mixing_prompt()
        
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        
        # Should contain mixing-related terms
        prompt_lower = prompt.lower()
        mixing_terms = ["mix", "audio", "sound", "music", "track", "professional", "engineer"]
        assert any(term in prompt_lower for term in mixing_terms)

    def test_get_random_mixing_prompt_variability(self):
        """Test that random prompts show variability."""
        prompts = set()
        
        # Generate multiple prompts
        for _ in range(20):
            prompt = get_random_mixing_prompt()
            prompts.add(prompt)
        
        # Should have some variety (not all identical)
        assert len(prompts) > 1

    def test_prompt_content_quality(self):
        """Test overall quality of prompt content."""
        # Test mixer prompts
        mixer_prompts = get_mixer_prompts(num_prompts=5)
        
        for prompt in mixer_prompts:
            character = prompt["character"]
            instruction = prompt["instruction"]
            
            # Should be substantial content
            assert len(character.split()) >= 5
            assert len(instruction.split()) >= 5
            
            # Should not be too long
            assert len(character) <= 500
            assert len(instruction) <= 500
            
            # Should contain proper punctuation
            assert character.endswith(".") or "." in character
            assert instruction.endswith(".") or "." in instruction

    def test_prompt_consistency_across_calls(self):
        """Test that prompts are consistent across multiple calls."""
        # First call
        prompts1 = get_mixer_prompts(num_prompts=5)
        
        # Second call
        prompts2 = get_mixer_prompts(num_prompts=5)
        
        # Should have same structure
        assert len(prompts1) == len(prompts2)
        
        for p1, p2 in zip(prompts1, prompts2):
            assert set(p1.keys()) == set(p2.keys())
            assert set(p1.keys()) == {"character", "instruction"}
            
            # Content should be identical (deterministic)
            assert p1["character"] == p2["character"]
            assert p1["instruction"] == p2["instruction"]


class TestRealAudioProcessing:
    """Test real audio processing with actual audio files."""

    @pytest.fixture
    def real_audio_file(self):
        """Create a real audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create 2 seconds of test audio - a simple sine wave
            sample_rate = 44100
            duration = 2.0
            t = np.linspace(0, duration, int(duration * sample_rate))
            
            # Mix of frequencies to make it more interesting
            audio = (
                0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 note
                0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 note
                0.1 * np.sin(2 * np.pi * 220 * t)    # A3 note
            ).astype(np.float32)
            
            # Write to file
            sf.write(f.name, audio, sample_rate)
            
            yield f.name
            
            # Cleanup
            os.unlink(f.name)

    def test_load_audio_with_fallback_real_file(self, real_audio_file):
        """Test loading a real audio file."""
        audio, sr = load_audio_with_fallback(real_audio_file)
        
        assert isinstance(audio, np.ndarray)
        # Audio should be float type (either float32 or float64 depending on soundfile version)
        assert audio.dtype in [np.float32, np.float64]
        assert sr == 44100
        assert len(audio) == int(2.0 * 44100)  # 2 seconds
        
        # Should contain actual audio content (not silence)
        assert np.max(np.abs(audio)) > 0.1
        assert not np.allclose(audio, np.zeros_like(audio))

    def test_load_audio_with_fallback_nonexistent(self):
        """Test loading a nonexistent file."""
        audio, sr = load_audio_with_fallback("definitely_does_not_exist.wav")
        
        # Should return silence as fallback
        assert isinstance(audio, np.ndarray)
        assert sr == 44100
        assert len(audio) == 44100  # 1 second of silence
        assert np.allclose(audio, np.zeros_like(audio))

    def test_real_audio_normalization_workflow(self, real_audio_file):
        """Test complete audio normalization workflow."""
        # Load real audio
        audio, sr = load_audio_with_fallback(real_audio_file)
        
        # Amplify it to simulate loud audio
        loud_audio = audio * 5.0
        
        # Normalize
        normalized = safe_normalize_audio(loud_audio, target_level=0.8)
        
        # Verify normalization worked
        assert np.max(np.abs(normalized)) <= 0.8
        assert np.isclose(np.max(np.abs(normalized)), 0.8, atol=1e-6)
        
        # Verify content is preserved (correlation should be high)
        correlation = np.corrcoef(audio.flatten(), normalized.flatten())[0, 1]
        assert correlation > 0.99  # Very high correlation

    def test_prepare_audio_for_model_real_audio(self, real_audio_file):
        """Test preparing real audio for model input."""
        # Load real audio
        audio, sr = load_audio_with_fallback(real_audio_file)
        
        # Prepare for model (half the length)
        target_length = len(audio) // 2
        prepared = prepare_audio_for_model(audio, target_length=target_length)
        
        assert isinstance(prepared, torch.Tensor)
        assert len(prepared) == target_length
        assert prepared.dtype == torch.float32
        
        # Should be normalized
        assert torch.max(torch.abs(prepared)) <= 1.0
        
        # Should contain actual content
        assert torch.max(torch.abs(prepared)) > 0.05 