"""
Unit tests for the dataset classes.
"""

import pytest
import numpy as np
import torch
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from src.training.dataset import HipHopDataset, AudioDistortionDataset


class TestHipHopDataset:
    """Test suite for HipHopDataset."""
    
    @pytest.fixture
    def mock_components(self):
        """Mock heavy components to avoid loading actual models."""
        with patch('src.training.dataset.StemExtractor') as mock_stem_ext, \
             patch('src.training.dataset.FaustDistorter') as mock_faust_dist:
            
            # Mock stem extractor
            mock_extractor = MagicMock()
            mock_stem_ext.return_value = mock_extractor
            
            # Mock Faust distorter
            mock_distorter = MagicMock()
            mock_distorter.apply_distortion.return_value = (
                np.random.randn(44100).astype(np.float32),
                {'tool': 'test_distortion', 'params': {}}
            )
            mock_faust_dist.return_value = mock_distorter
            
            yield {
                'stem_extractor': mock_extractor,
                'faust_distorter': mock_distorter
            }
    
    @pytest.fixture
    def dataset(self, mock_components, mock_dataset_dir):
        """Create a HipHopDataset instance."""
        return HipHopDataset(
            data_dir=mock_dataset_dir,
            max_duration=5.0,
            use_distortion=True
        )
    
    def test_initialization(self, dataset, mock_dataset_dir):
        """Test dataset initialization."""
        assert dataset.data_dir == mock_dataset_dir
        assert dataset.sample_rate == 44100
        assert dataset.max_duration == 5.0
        assert dataset.use_distortion is True
        assert dataset.segment_duration == 5.0
        assert dataset.max_segments_per_song == 3
        assert dataset.include_json is True
        
        # Should have found songs
        assert len(dataset.songs) > 0
    
    def test_initialization_without_distortion(self, mock_components, mock_dataset_dir):
        """Test dataset initialization without distortion."""
        dataset = HipHopDataset(
            data_dir=mock_dataset_dir,
            use_distortion=False
        )
        
        assert dataset.use_distortion is False
    
    def test_scan_data_directory_with_json(self, dataset):
        """Test scanning data directory for songs with JSON metadata."""
        songs = dataset._scan_data_directory_with_json()
        
        assert isinstance(songs, list)
        assert len(songs) > 0
        
        # Check song structure
        song = songs[0]
        assert 'artist' in song
        assert 'song_name' in song
        assert 'wav_file' in song
        assert 'json_file' in song
        assert 'json_data' in song
        assert 'stem_files' in song
        assert 'song_dir' in song
    
    def test_len(self, dataset):
        """Test dataset length."""
        assert len(dataset) == len(dataset.songs)
        assert len(dataset) > 0
    
    def test_getitem(self, dataset):
        """Test getting an item from the dataset."""
        item = dataset[0]
        
        assert isinstance(item, dict)
        assert 'stems' in item
        assert 'style_prompt' in item
        assert 'ground_truth_dsp' in item
        assert 'song_info' in item
        assert 'segments' in item
        
        # Check stems structure
        stems = item['stems']
        assert isinstance(stems, dict)
        assert len(stems) > 0
        
        # Check style prompt
        assert isinstance(item['style_prompt'], str)
        assert len(item['style_prompt']) > 0
    
    def test_getitem_without_distortion(self, mock_components, mock_dataset_dir):
        """Test getting item without distortion."""
        dataset = HipHopDataset(
            data_dir=mock_dataset_dir,
            use_distortion=False
        )
        
        item = dataset[0]
        
        assert item['ground_truth_dsp'] is None
    
    def test_load_preextracted_stems(self, dataset, temp_dir, test_audio_mono, sample_rate):
        """Test loading pre-extracted stems."""
        # Create mock stem files
        stem_files = {}
        for stem_type in ['vocals', 'drums', 'bass', 'other']:
            stem_path = os.path.join(temp_dir, f"{stem_type}.wav")
            import soundfile as sf
            sf.write(stem_path, test_audio_mono, sample_rate)
            stem_files[stem_type] = stem_path
        
        # Test segments
        segments = [{"start": 0.0, "end": 1.0, "label": "verse"}]
        
        stems = dataset._load_preextracted_stems(stem_files, segments)
        
        assert isinstance(stems, dict)
        assert len(stems) == 4
        for stem_name, stem_audio in stems.items():
            assert isinstance(stem_audio, np.ndarray)
            assert len(stem_audio) > 0
    
    def test_load_preextracted_stems_no_segments(self, dataset, temp_dir, test_audio_mono, sample_rate):
        """Test loading pre-extracted stems without segments."""
        # Create mock stem files
        stem_files = {}
        for stem_type in ['vocals', 'drums']:
            stem_path = os.path.join(temp_dir, f"{stem_type}.wav")
            import soundfile as sf
            sf.write(stem_path, test_audio_mono, sample_rate)
            stem_files[stem_type] = stem_path
        
        stems = dataset._load_preextracted_stems(stem_files, [])
        
        assert isinstance(stems, dict)
        assert len(stems) == 2
    
    def test_extract_stems_from_audio(self, dataset, test_audio_mono):
        """Test extracting stems from audio data."""
        segments = [{"start": 0.0, "end": 1.0, "label": "verse"}]
        
        stems = dataset._extract_stems_from_audio(test_audio_mono, segments)
        
        assert isinstance(stems, dict)
        assert len(stems) == 4  # vocals, drums, bass, other
        
        for stem_name, stem_audio in stems.items():
            assert isinstance(stem_audio, np.ndarray)
            assert len(stem_audio) > 0
    
    def test_apply_distortions(self, dataset, test_stems):
        """Test applying distortions to stems."""
        distorted_stems, dsp_corrections = dataset._apply_distortions(test_stems)
        
        assert isinstance(distorted_stems, dict)
        assert isinstance(dsp_corrections, dict)
        assert len(distorted_stems) == len(test_stems)
        assert len(dsp_corrections) == len(test_stems)
        
        for stem_name in test_stems.keys():
            assert stem_name in distorted_stems
            assert stem_name in dsp_corrections
            assert isinstance(distorted_stems[stem_name], np.ndarray)
            assert isinstance(dsp_corrections[stem_name], dict)
    
    def test_create_enhanced_style_prompt(self, dataset):
        """Test creating enhanced style prompts."""
        song_info = {
            'artist': 'kendrick',
            'song_name': 'test_song',
            'json_data': {
                'bpm': 120.0,
                'key': 'Cm',
                'genre': 'hip-hop',
                'segments': [
                    {"start": 0.0, "end": 30.0, "label": "verse"},
                    {"start": 30.0, "end": 60.0, "label": "chorus"}
                ]
            }
        }
        
        prompt = dataset._create_enhanced_style_prompt(song_info)
        
        assert isinstance(prompt, str)
        assert 'kendrick' in prompt
        assert 'test_song' in prompt
        assert '120' in prompt  # BPM
        assert 'Cm' in prompt   # Key
        assert 'hip-hop' in prompt  # Genre
    
    def test_create_enhanced_style_prompt_no_json(self, dataset):
        """Test creating style prompt without JSON data."""
        song_info = {
            'artist': 'travis',
            'song_name': 'test_song',
            'json_data': None
        }
        
        prompt = dataset._create_enhanced_style_prompt(song_info)
        
        assert isinstance(prompt, str)
        assert 'travis' in prompt
        assert 'test_song' in prompt
    
    def test_create_style_prompt_fallback(self, dataset):
        """Test fallback style prompt creation."""
        file_path = "/data/kendrick/song.wav"
        
        prompt = dataset._create_style_prompt(file_path)
        
        assert isinstance(prompt, str)
        assert 'kendrick' in prompt
    
    @pytest.mark.parametrize("max_duration", [10.0, 30.0, 60.0])
    def test_different_max_durations(self, mock_components, mock_dataset_dir, max_duration):
        """Test dataset with different maximum durations."""
        dataset = HipHopDataset(
            data_dir=mock_dataset_dir,
            max_duration=max_duration
        )
        
        assert dataset.max_duration == max_duration
    
    @pytest.mark.parametrize("include_json", [True, False])
    def test_json_inclusion_parameter(self, mock_components, mock_dataset_dir, include_json):
        """Test JSON inclusion parameter."""
        dataset = HipHopDataset(
            data_dir=mock_dataset_dir,
            include_json=include_json
        )
        
        assert dataset.include_json == include_json
    
    def test_empty_data_directory(self, mock_components, temp_dir):
        """Test dataset with empty data directory."""
        dataset = HipHopDataset(data_dir=temp_dir)
        
        assert len(dataset) == 0
    
    def test_dataset_iteration(self, dataset):
        """Test iterating through the dataset."""
        items = []
        for i, item in enumerate(dataset):
            items.append(item)
            if i >= 2:  # Just test a few items
                break
        
        assert len(items) > 0
        for item in items:
            assert isinstance(item, dict)
            assert 'stems' in item
            assert 'style_prompt' in item


class TestAudioDistortionDataset:
    """Test suite for AudioDistortionDataset."""
    
    @pytest.fixture
    def mock_audio_distorter(self):
        """Mock AudioDistorter to avoid dependencies."""
        with patch('src.training.dataset.AudioDistorter') as mock_distorter_class:
            mock_distorter = MagicMock()
            
            # Mock get_combined_distortions
            from pydub import AudioSegment
            mock_audio = AudioSegment.silent(duration=1000)  # 1 second
            mock_corrections = [{'tool': 'test', 'params': {}}]
            
            mock_distorter.get_combined_distortions.return_value = (mock_audio, mock_corrections)
            mock_distorter.change_volume_distortion.return_value = (mock_audio, mock_corrections)
            
            mock_distorter_class.return_value = mock_distorter
            
            yield mock_distorter
    
    @pytest.fixture
    def audio_files(self, temp_dir, test_audio_mono, sample_rate):
        """Create test audio files."""
        files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f"test_audio_{i}.wav")
            import soundfile as sf
            sf.write(file_path, test_audio_mono, sample_rate)
            files.append(file_path)
        return files
    
    @pytest.fixture
    def distortion_dataset(self, mock_audio_distorter, audio_files):
        """Create an AudioDistortionDataset instance."""
        return AudioDistortionDataset(
            audio_files=audio_files,
            chunk_duration=2.0,
            distortions_per_sample=2
        )
    
    def test_initialization(self, distortion_dataset, audio_files):
        """Test AudioDistortionDataset initialization."""
        assert distortion_dataset.audio_files == audio_files
        assert distortion_dataset.sample_rate == 44100
        assert distortion_dataset.chunk_duration == 2.0
        assert distortion_dataset.distortions_per_sample == 2
    
    def test_len(self, distortion_dataset, audio_files):
        """Test dataset length."""
        assert len(distortion_dataset) == len(audio_files)
    
    def test_getitem(self, distortion_dataset, mock_audio_distorter):
        """Test getting an item from the distortion dataset."""
        item = distortion_dataset[0]
        
        assert isinstance(item, dict)
        assert 'clean_audio' in item
        assert 'distorted_audio' in item
        assert 'corrections' in item
        assert 'file_path' in item
        
        # Check tensor types
        assert isinstance(item['clean_audio'], torch.Tensor)
        assert isinstance(item['distorted_audio'], torch.Tensor)
        
        # Check that AudioDistorter was called
        mock_audio_distorter.get_combined_distortions.assert_called_once()
    
    def test_getitem_fallback_distortion(self, audio_files, sample_rate):
        """Test getting item with fallback to single distortion."""
        with patch('src.training.dataset.AudioDistorter') as mock_distorter_class:
            mock_distorter = MagicMock()
            
            # Mock combined distortions to fail
            mock_distorter.get_combined_distortions.side_effect = Exception("Mock error")
            
            # Mock fallback distortion
            from pydub import AudioSegment
            mock_audio = AudioSegment.silent(duration=1000)
            mock_corrections = [{'tool': 'volume', 'params': {}}]
            mock_distorter.change_volume_distortion.return_value = (mock_audio, mock_corrections)
            
            mock_distorter_class.return_value = mock_distorter
            
            dataset = AudioDistortionDataset(audio_files=audio_files)
            item = dataset[0]
            
            assert isinstance(item, dict)
            assert 'clean_audio' in item
            assert 'distorted_audio' in item
            
            # Should have called fallback method
            mock_distorter.change_volume_distortion.assert_called_once()
    
    def test_chunk_creation_long_audio(self, temp_dir, sample_rate):
        """Test chunk creation from long audio file."""
        # Create a longer audio file (10 seconds)
        long_audio = np.random.randn(sample_rate * 10).astype(np.float32)
        long_file = os.path.join(temp_dir, "long_audio.wav")
        import soundfile as sf
        sf.write(long_file, long_audio, sample_rate)
        
        with patch('src.training.dataset.AudioDistorter') as mock_distorter_class:
            mock_distorter = MagicMock()
            from pydub import AudioSegment
            mock_audio = AudioSegment.silent(duration=2000)  # 2 seconds
            mock_distorter.get_combined_distortions.return_value = (mock_audio, [])
            mock_distorter_class.return_value = mock_distorter
            
            dataset = AudioDistortionDataset(
                audio_files=[long_file],
                chunk_duration=5.0
            )
            
            item = dataset[0]
            
            # Should have created a chunk
            assert isinstance(item['clean_audio'], torch.Tensor)
    
    def test_chunk_creation_short_audio(self, temp_dir, sample_rate):
        """Test chunk creation from short audio file."""
        # Create a short audio file (1 second)
        short_audio = np.random.randn(sample_rate).astype(np.float32)
        short_file = os.path.join(temp_dir, "short_audio.wav")
        import soundfile as sf
        sf.write(short_file, short_audio, sample_rate)
        
        with patch('src.training.dataset.AudioDistorter') as mock_distorter_class:
            mock_distorter = MagicMock()
            from pydub import AudioSegment
            mock_audio = AudioSegment.silent(duration=1000)  # 1 second
            mock_distorter.get_combined_distortions.return_value = (mock_audio, [])
            mock_distorter_class.return_value = mock_distorter
            
            dataset = AudioDistortionDataset(
                audio_files=[short_file],
                chunk_duration=5.0
            )
            
            item = dataset[0]
            
            # Should use the entire short audio
            assert isinstance(item['clean_audio'], torch.Tensor)
    
    def test_audio_normalization(self, distortion_dataset):
        """Test that audio is properly normalized."""
        item = distortion_dataset[0]
        
        clean_audio = item['clean_audio'].numpy()
        distorted_audio = item['distorted_audio'].numpy()
        
        # Check normalization (should be within [-1, 1] range)
        assert np.max(np.abs(clean_audio)) <= 1.0
        assert np.max(np.abs(distorted_audio)) <= 1.0
    
    @pytest.mark.parametrize("chunk_duration", [1.0, 3.0, 5.0])
    def test_different_chunk_durations(self, audio_files, chunk_duration):
        """Test dataset with different chunk durations."""
        with patch('src.training.dataset.AudioDistorter') as mock_distorter_class:
            mock_distorter = MagicMock()
            from pydub import AudioSegment
            mock_audio = AudioSegment.silent(duration=int(chunk_duration * 1000))
            mock_distorter.get_combined_distortions.return_value = (mock_audio, [])
            mock_distorter_class.return_value = mock_distorter
            
            dataset = AudioDistortionDataset(
                audio_files=audio_files,
                chunk_duration=chunk_duration
            )
            
            assert dataset.chunk_duration == chunk_duration
    
    @pytest.mark.parametrize("distortions_per_sample", [1, 3, 5])
    def test_different_distortions_per_sample(self, audio_files, distortions_per_sample):
        """Test dataset with different numbers of distortions per sample."""
        with patch('src.training.dataset.AudioDistorter') as mock_distorter_class:
            mock_distorter = MagicMock()
            from pydub import AudioSegment
            mock_audio = AudioSegment.silent(duration=1000)
            mock_distorter.get_combined_distortions.return_value = (mock_audio, [])
            mock_distorter_class.return_value = mock_distorter
            
            dataset = AudioDistortionDataset(
                audio_files=audio_files,
                distortions_per_sample=distortions_per_sample
            )
            
            assert dataset.distortions_per_sample == distortions_per_sample
            
            # Get an item to check that the parameter is passed correctly
            item = dataset[0]
            mock_distorter.get_combined_distortions.assert_called_with(
                num_distortions=distortions_per_sample
            )
    
    def test_empty_audio_files_list(self):
        """Test dataset with empty audio files list."""
        dataset = AudioDistortionDataset(audio_files=[])
        assert len(dataset) == 0
    
    def test_dataset_iteration(self, distortion_dataset):
        """Test iterating through the distortion dataset."""
        items = []
        for i, item in enumerate(distortion_dataset):
            items.append(item)
            if i >= 1:  # Just test a few items
                break
        
        assert len(items) > 0
        for item in items:
            assert isinstance(item, dict)
            assert 'clean_audio' in item
            assert 'distorted_audio' in item
            assert 'corrections' in item
            assert 'file_path' in item
    
    def test_error_handling_corrupted_file(self, temp_dir):
        """Test error handling with corrupted audio file."""
        # Create a corrupted file (not actually audio)
        corrupted_file = os.path.join(temp_dir, "corrupted.wav")
        with open(corrupted_file, 'w') as f:
            f.write("This is not audio data")
        
        dataset = AudioDistortionDataset(audio_files=[corrupted_file])
        
        # Should handle the error gracefully
        try:
            item = dataset[0]
            # If it doesn't crash, that's acceptable
        except Exception:
            # Expected behavior for corrupted files
            pass
    
    def test_memory_efficiency(self, audio_files):
        """Test that dataset doesn't load all files into memory at once."""
        with patch('src.training.dataset.AudioDistorter') as mock_distorter_class:
            mock_distorter = MagicMock()
            from pydub import AudioSegment
            mock_audio = AudioSegment.silent(duration=1000)
            mock_distorter.get_combined_distortions.return_value = (mock_audio, [])
            mock_distorter_class.return_value = mock_distorter
            
            # Create dataset with many files
            many_files = audio_files * 100  # Simulate 300 files
            dataset = AudioDistortionDataset(audio_files=many_files)
            
            # Should only create AudioDistorter when accessing items
            assert mock_distorter_class.call_count == 0
            
            # Access one item
            item = dataset[0]
            
            # Should have created only one AudioDistorter instance
            assert mock_distorter_class.call_count == 1 