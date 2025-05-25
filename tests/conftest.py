"""
Pytest configuration and shared fixtures for Hip-Hop Producer AI tests.
"""

import pytest
import torch
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path
import soundfile as sf
import json
from unittest.mock import MagicMock

# Set up path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def device():
    """Return appropriate device for testing."""
    return 'cpu'  # Use CPU for consistent test results


@pytest.fixture
def sample_rate():
    """Standard sample rate for testing."""
    return 44100


@pytest.fixture
def test_audio_mono(sample_rate):
    """Generate mono test audio."""
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate a simple sine wave
    audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz tone
    return audio.astype(np.float32)


@pytest.fixture
def test_audio_stereo(sample_rate):
    """Generate stereo test audio."""
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    left = np.sin(2 * np.pi * 440 * t) * 0.5
    right = np.sin(2 * np.pi * 880 * t) * 0.5
    return np.column_stack([left, right]).astype(np.float32)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_wav_file(temp_dir, test_audio_mono, sample_rate):
    """Create a temporary WAV file for testing."""
    wav_path = os.path.join(temp_dir, "test_audio.wav")
    sf.write(wav_path, test_audio_mono, sample_rate)
    return wav_path


@pytest.fixture
def test_stems(test_audio_mono):
    """Generate test stem data."""
    stems = {}
    for stem_type in ['vocals', 'drums', 'bass', 'other']:
        # Create slightly different audio for each stem
        stems[stem_type] = test_audio_mono * np.random.uniform(0.5, 1.0)
    return stems


@pytest.fixture
def test_metadata():
    """Generate test song metadata."""
    return {
        "path": "/test/path/song.wav",
        "bpm": 120.0,
        "beats": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "downbeats": [0.5, 2.5],
        "beat_positions": [1, 2, 3, 4, 1, 2],
        "segments": [
            {"start": 0.0, "end": 2.0, "label": "verse"},
            {"start": 2.0, "end": 4.0, "label": "chorus"}
        ]
    }


@pytest.fixture
def test_json_file(temp_dir, test_metadata):
    """Create a temporary JSON metadata file."""
    json_path = os.path.join(temp_dir, "test_metadata.json")
    with open(json_path, 'w') as f:
        json.dump(test_metadata, f)
    return json_path


@pytest.fixture
def mock_dataset_dir(temp_dir, test_audio_mono, test_metadata, sample_rate):
    """Create a mock dataset directory structure."""
    # Create artist directory
    artist_dir = os.path.join(temp_dir, "test_artist")
    os.makedirs(artist_dir)
    
    # Create song directory
    song_dir = os.path.join(artist_dir, "test_song")
    os.makedirs(song_dir)
    
    # Create main audio file
    audio_path = os.path.join(song_dir, "test_song.wav")
    sf.write(audio_path, test_audio_mono, sample_rate)
    
    # Create metadata file
    json_path = os.path.join(song_dir, "test_song.json")
    with open(json_path, 'w') as f:
        json.dump(test_metadata, f)
    
    # Create stems directory
    stems_dir = os.path.join(song_dir, "stems")
    os.makedirs(stems_dir)
    
    # Create stem files
    for stem_type in ['vocals', 'drums', 'bass', 'other']:
        stem_audio = test_audio_mono * np.random.uniform(0.3, 0.8)
        stem_path = os.path.join(stems_dir, f"{stem_type}.wav")
        sf.write(stem_path, stem_audio, sample_rate)
    
    return temp_dir


@pytest.fixture
def mock_generator():
    """Create a mock generator for testing."""
    generator = MagicMock()
    generator.generate.return_value = np.random.randn(44100).astype(np.float32)
    generator.generate_bass_line.return_value = np.random.randn(44100).astype(np.float32)
    return generator


@pytest.fixture
def mock_mixer():
    """Create a mock mixer for testing."""
    mixer = MagicMock()
    mixer.forward.return_value = "mock generated text"
    mixer.get_embedding.return_value = (
        torch.randn(1, 4096),  # text embedding
        torch.randn(1, 4096)   # audio embedding
    )
    return mixer


@pytest.fixture
def test_faust_code():
    """Sample Faust DSP code for testing."""
    return """
import("stdfaust.lib");

gain_db = -6.0;
gain_linear = ba.db2linear(gain_db);

process = _ * gain_linear;
"""


@pytest.fixture
def test_prompts():
    """Sample prompts for testing."""
    return {
        'style': "Dark trap beat with heavy 808s and atmospheric pads",
        'generation': "Generate a menacing bass line with sliding notes",
        'mixing': "Apply professional hip-hop mixing with punch and clarity"
    }


# Skip tests that require GPU if CUDA is not available
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu) 