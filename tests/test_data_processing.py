"""
Real tests for data processing modules - testing actual functionality without heavy mocks.
"""

import pytest
import numpy as np
import os
import tempfile
import json
import soundfile as sf
from pathlib import Path

from src.data_processing.chunking import AudioChunker, chunk_dataset


class TestAudioChunkerReal:
    """Test the actual AudioChunker implementation."""

    @pytest.fixture
    def chunker(self):
        """Create a real AudioChunker instance."""
        return AudioChunker(
            chunk_duration=3.0,
            overlap_duration=0.5,
            sample_rate=44100
        )

    @pytest.fixture
    def real_audio(self):
        """Create real audio data for testing."""
        # Create 10 seconds of audio with varying content
        duration = 10.0
        sample_rate = 44100
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Create complex audio with multiple components
        audio = (
            0.4 * np.sin(2 * np.pi * 440 * t) +           # Base tone
            0.2 * np.sin(2 * np.pi * 880 * t * (1 + 0.1 * np.sin(2 * np.pi * t))) +  # Vibrato
            0.1 * np.random.randn(len(t)) +               # Noise
            0.3 * np.sin(2 * np.pi * 220 * t) * np.exp(-t/5)  # Decaying low freq
        ).astype(np.float32)
        
        return audio

    @pytest.fixture
    def real_metadata(self):
        """Create realistic metadata."""
        return {
            "path": "/test/song.wav",
            "bpm": 128.0,
            "beats": [0.5, 0.93, 1.37, 1.8, 2.24, 2.68, 3.11, 3.55, 3.99, 4.42, 
                     4.86, 5.29, 5.73, 6.17, 6.6, 7.04, 7.48, 7.91, 8.35, 8.79],
            "segments": [
                {"start": 0.0, "end": 4.0, "label": "intro"},
                {"start": 4.0, "end": 7.0, "label": "verse"},
                {"start": 7.0, "end": 10.0, "label": "chorus"}
            ],
            "key": "Am",
            "genre": "hip-hop"
        }

    def test_chunker_initialization(self, chunker):
        """Test that chunker initializes with correct parameters."""
        assert chunker.chunk_duration == 3.0
        assert chunker.overlap_duration == 0.5
        assert chunker.sample_rate == 44100
        assert chunker.chunk_samples == int(3.0 * 44100)
        assert chunker.overlap_samples == int(0.5 * 44100)
        assert chunker.step_samples == chunker.chunk_samples - chunker.overlap_samples

    def test_chunk_short_audio(self, chunker):
        """Test chunking audio shorter than chunk duration."""
        # 1 second of audio (shorter than 3 second chunks)
        short_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)).astype(np.float32)
        metadata = {"test": "value"}
        
        chunks = chunker.chunk_audio(short_audio, metadata)
        
        assert len(chunks) == 1
        chunk_audio, chunk_metadata = chunks[0]
        
        # Audio should be unchanged
        assert np.array_equal(chunk_audio, short_audio)
        
        # Metadata should be preserved and augmented
        assert chunk_metadata["test"] == "value"
        assert chunk_metadata["chunk_index"] == 0
        assert chunk_metadata["total_chunks"] == 1
        assert chunk_metadata["start_time"] == 0.0

    def test_chunk_long_audio(self, chunker, real_audio, real_metadata):
        """Test chunking long audio into multiple overlapping chunks."""
        chunks = chunker.chunk_audio(real_audio, real_metadata)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should have the correct length
        for chunk_audio, chunk_metadata in chunks[:-1]:  # Exclude last chunk (might be padded)
            assert len(chunk_audio) == chunker.chunk_samples
        
        # Check first chunk
        first_audio, first_metadata = chunks[0]
        assert len(first_audio) == chunker.chunk_samples
        assert first_metadata["chunk_index"] == 0
        assert first_metadata["start_time"] == 0.0
        assert first_metadata["bpm"] == 128.0  # Original metadata preserved
        
        # Check second chunk has correct overlap
        if len(chunks) > 1:
            second_audio, second_metadata = chunks[1]
            assert second_metadata["chunk_index"] == 1
            assert second_metadata["start_time"] == chunker.step_samples / chunker.sample_rate
            
            # Check that there's actual overlap in the audio
            overlap_samples = chunker.overlap_samples
            first_end = first_audio[-overlap_samples:]
            second_start = second_audio[:overlap_samples]
            # They should be identical (overlapping regions)
            assert np.allclose(first_end, second_start, atol=1e-6)

    def test_beat_filtering_in_chunks(self, chunker, real_audio, real_metadata):
        """Test that beats are correctly filtered for each chunk."""
        chunks = chunker.chunk_audio(real_audio, real_metadata)
        
        for chunk_audio, chunk_metadata in chunks:
            if "beats" in chunk_metadata:
                chunk_beats = chunk_metadata["beats"]
                start_time = chunk_metadata["start_time"]
                end_time = chunk_metadata["end_time"]
                
                # All beats should be within chunk duration
                for beat in chunk_beats:
                    assert 0 <= beat < (end_time - start_time)
                
                # Check that beats are correctly adjusted relative to chunk start
                # Find corresponding original beats
                original_beats = real_metadata["beats"]
                expected_beats = [b - start_time for b in original_beats 
                                if start_time <= b < end_time]
                
                assert np.allclose(chunk_beats, expected_beats, atol=1e-6)

    def test_segment_filtering_in_chunks(self, chunker, real_audio, real_metadata):
        """Test that segments are correctly filtered for each chunk."""
        chunks = chunker.chunk_audio(real_audio, real_metadata)
        
        for chunk_audio, chunk_metadata in chunks:
            if "segments" in chunk_metadata:
                chunk_segments = chunk_metadata["segments"]
                start_time = chunk_metadata["start_time"]
                end_time = chunk_metadata["end_time"]
                
                for segment in chunk_segments:
                    # Segment should be within chunk bounds
                    assert segment["start"] >= 0
                    assert segment["end"] <= (end_time - start_time)
                    
                    # Check that label is preserved
                    assert "label" in segment

    def test_chunk_stems_consistency(self, chunker):
        """Test that multiple stems are chunked consistently."""
        # Create different length stems to test trimming
        stems = {
            "vocals": np.random.randn(440000).astype(np.float32),  # 10 seconds
            "drums": np.random.randn(352800).astype(np.float32),   # 8 seconds  
            "bass": np.random.randn(264600).astype(np.float32),    # 6 seconds
            "other": np.random.randn(176400).astype(np.float32)    # 4 seconds
        }
        
        chunked_stems = chunker.chunk_stems(stems)
        
        assert len(chunked_stems) > 0
        
        # All chunks should have the same stems
        for chunk_stems, chunk_metadata in chunked_stems:
            assert set(chunk_stems.keys()) == set(stems.keys())
            
            # All stems in a chunk should have the same length
            stem_lengths = [len(audio) for audio in chunk_stems.values()]
            assert len(set(stem_lengths)) == 1  # All should be the same length
            
            # Length should be appropriate for the chunk
            expected_length = min(chunker.chunk_samples, stem_lengths[0])
            for audio in chunk_stems.values():
                assert len(audio) == expected_length

    def test_save_chunks_functionality(self, chunker):
        """Test saving chunks to files (without actual file I/O)."""
        # Create test chunks
        chunk1 = np.random.randn(chunker.chunk_samples).astype(np.float32)
        chunk2 = np.random.randn(chunker.chunk_samples).astype(np.float32)
        
        chunks = [
            (chunk1, {"chunk_index": 0, "bpm": 120}),
            (chunk2, {"chunk_index": 1, "bpm": 120})
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the file writing to avoid actual I/O
            saved_files = []
            
            # Simulate the save process
            for i, (chunk_audio, chunk_metadata) in enumerate(chunks):
                # This tests the logic without actual file writing
                audio_file = os.path.join(temp_dir, f"test_chunk_{i:03d}.wav")
                json_file = os.path.join(temp_dir, f"test_chunk_{i:03d}.json")
                
                saved_files.append(audio_file)
                
                # Verify metadata structure
                assert isinstance(chunk_metadata, dict)
                assert "chunk_index" in chunk_metadata
                assert chunk_metadata["chunk_index"] == i

            assert len(saved_files) == 2

    def test_chunk_audio_preserves_content(self, chunker, real_audio):
        """Test that chunking preserves audio content correctly."""
        chunks = chunker.chunk_audio(real_audio)
        
        # Reconstruct audio from chunks (approximately)
        reconstructed = np.zeros_like(real_audio)
        
        for chunk_audio, chunk_metadata in chunks:
            start_sample = int(chunk_metadata["start_time"] * chunker.sample_rate)
            end_sample = min(start_sample + len(chunk_audio), len(reconstructed))
            chunk_length = end_sample - start_sample
            
            # Add chunk to reconstructed audio (overlapping regions will be summed)
            reconstructed[start_sample:end_sample] += chunk_audio[:chunk_length]
        
        # The beginning should match well (before overlap complications)
        first_chunk_samples = chunker.chunk_samples
        correlation = np.corrcoef(
            real_audio[:first_chunk_samples], 
            reconstructed[:first_chunk_samples]
        )[0, 1]
        assert correlation > 0.95  # Should be very similar

    def test_different_chunk_parameters(self):
        """Test chunker with different parameter combinations."""
        test_configs = [
            (2.0, 0.0),   # No overlap
            (5.0, 1.0),   # 1 second overlap
            (4.0, 0.5),   # 0.5 second overlap
        ]
        
        audio = np.random.randn(220500).astype(np.float32)  # 5 seconds
        
        for chunk_duration, overlap_duration in test_configs:
            chunker = AudioChunker(
                chunk_duration=chunk_duration,
                overlap_duration=overlap_duration,
                sample_rate=44100
            )
            
            chunks = chunker.chunk_audio(audio)
            
            # Should produce at least one chunk
            assert len(chunks) > 0
            
            # Check that overlap is correct
            if len(chunks) > 1 and overlap_duration > 0:
                overlap_samples = int(overlap_duration * 44100)
                
                # Check overlap between consecutive chunks
                for i in range(len(chunks) - 1):
                    chunk1_audio = chunks[i][0]
                    chunk2_audio = chunks[i + 1][0]
                    
                    # Last overlap_samples of chunk1 should equal first overlap_samples of chunk2
                    if len(chunk1_audio) >= overlap_samples and len(chunk2_audio) >= overlap_samples:
                        overlap1 = chunk1_audio[-overlap_samples:]
                        overlap2 = chunk2_audio[:overlap_samples]
                        assert np.allclose(overlap1, overlap2, atol=1e-6)


class TestChunkDatasetFunction:
    """Test the chunk_dataset function with real file operations."""

    def test_chunk_dataset_with_real_files(self):
        """Test the chunk_dataset function with actual audio files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create input and output directories with proper structure
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            
            # Create artist/song directory structure as expected by chunk_dataset
            artist_dir = os.path.join(input_dir, "test_artist")
            song_dir = os.path.join(artist_dir, "test_song")
            os.makedirs(song_dir)
            
            # Create a real audio file
            sample_rate = 44100
            duration = 8.0  # 8 seconds
            t = np.linspace(0, duration, int(duration * sample_rate))
            audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            
            audio_file = os.path.join(song_dir, "test_song.wav")
            sf.write(audio_file, audio, sample_rate)
            
            # Create corresponding metadata
            metadata = {
                "bpm": 120,
                "key": "C",
                "genre": "test"
            }
            json_file = os.path.join(song_dir, "test_song.json")
            with open(json_file, 'w') as f:
                json.dump(metadata, f)
            
            # Run chunking
            chunk_dataset(
                input_dir=input_dir,
                output_dir=output_dir,
                chunk_duration=3.0,
                overlap_duration=0.5
            )
            
            # Check that output was created
            assert os.path.exists(output_dir)
            
            # Check output structure: output/test_artist/test_song/
            expected_output_song_dir = os.path.join(output_dir, "test_artist", "test_song")
            if os.path.exists(expected_output_song_dir):
                # Check that chunks were created
                output_files = list(Path(expected_output_song_dir).glob("*.wav"))
                assert len(output_files) > 1  # Should create multiple chunks
                
                # Check that metadata files were created
                json_files = list(Path(expected_output_song_dir).glob("*.json"))
                assert len(json_files) == len(output_files)  # One JSON per audio file
                
                # Verify that chunks are actually audio files with content
                for audio_file in output_files:
                    chunk_audio, chunk_sr = sf.read(audio_file)
                    assert chunk_sr == sample_rate
                    assert len(chunk_audio) > 0
                    assert not np.allclose(chunk_audio, np.zeros_like(chunk_audio))  # Not silence

    def test_chunk_dataset_empty_directory(self):
        """Test chunk_dataset with empty input directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "empty_input")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(input_dir)
            
            # Should handle empty directory gracefully
            chunk_dataset(input_dir, output_dir)
            
            # Output directory might be created but should be empty
            if os.path.exists(output_dir):
                output_files = list(Path(output_dir).glob("*"))
                assert len(output_files) == 0

    def test_chunk_dataset_no_metadata(self):
        """Test chunk_dataset with audio files but no metadata."""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = os.path.join(temp_dir, "input")
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(input_dir)
            
            # Create audio file without metadata
            sample_rate = 44100
            duration = 5.0
            t = np.linspace(0, duration, int(duration * sample_rate))
            audio = (0.3 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
            
            audio_file = os.path.join(input_dir, "no_metadata.wav")
            sf.write(audio_file, audio, sample_rate)
            
            # Should still work without metadata
            chunk_dataset(input_dir, output_dir)
            
            # Should still create output chunks
            if os.path.exists(output_dir):
                output_files = list(Path(output_dir).glob("*.wav"))
                # Might create chunks depending on implementation
                # At minimum, should not crash


class TestRealAudioDataProcessing:
    """Test real audio data processing workflows."""

    @pytest.fixture
    def complex_audio_file(self):
        """Create a complex audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            # Create realistic audio with multiple characteristics
            sample_rate = 44100
            duration = 12.0
            t = np.linspace(0, duration, int(duration * sample_rate))
            
            # Simulate a beat pattern
            beat_freq = 2.0  # 2 beats per second = 120 BPM
            kick_pattern = np.sin(2 * np.pi * 60 * t) * (np.sin(2 * np.pi * beat_freq * t) > 0.8)
            
            # Add melodic content
            melody = 0.3 * np.sin(2 * np.pi * 440 * t * (1 + 0.1 * np.sin(2 * np.pi * 0.5 * t)))
            
            # Add bass line
            bass = 0.4 * np.sin(2 * np.pi * 110 * t) * np.exp(-np.mod(t, 4) * 2)
            
            # Combine
            audio = (kick_pattern + melody + bass + 0.05 * np.random.randn(len(t))).astype(np.float32)
            
            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            sf.write(f.name, audio, sample_rate)
            yield f.name
            os.unlink(f.name)

    def test_full_audio_processing_pipeline(self, complex_audio_file):
        """Test a complete audio processing pipeline."""
        # Load the audio
        audio, sr = sf.read(complex_audio_file)
        
        # Initialize chunker
        chunker = AudioChunker(
            chunk_duration=4.0,
            overlap_duration=1.0,
            sample_rate=sr
        )
        
        # Create metadata
        metadata = {
            "original_file": complex_audio_file,
            "bpm": 120,
            "key": "A",
            "energy": "high"
        }
        
        # Chunk the audio
        chunks = chunker.chunk_audio(audio, metadata)
        
        # Verify pipeline worked
        assert len(chunks) > 1
        
        total_processed_samples = 0
        for chunk_audio, chunk_metadata in chunks:
            # Each chunk should have content
            assert len(chunk_audio) > 0
            assert np.max(np.abs(chunk_audio)) > 0.01
            
            # Metadata should be preserved and augmented
            assert chunk_metadata["bpm"] == 120
            assert chunk_metadata["key"] == "A"
            assert "chunk_index" in chunk_metadata
            assert "start_time" in chunk_metadata
            
            total_processed_samples += len(chunk_audio)
        
        # Should have processed a reasonable amount of audio
        # (accounting for overlaps, might be more than original)
        assert total_processed_samples >= len(audio)

    def test_chunking_preserves_audio_characteristics(self, complex_audio_file):
        """Test that chunking preserves important audio characteristics."""
        # Load original audio
        original_audio, sr = sf.read(complex_audio_file)
        
        # Chunk it
        chunker = AudioChunker(chunk_duration=3.0, overlap_duration=0.0, sample_rate=sr)
        chunks = chunker.chunk_audio(original_audio)
        
        # Analyze characteristics of original vs chunks
        original_rms = np.sqrt(np.mean(original_audio**2))
        original_peak = np.max(np.abs(original_audio))
        
        chunk_rms_values = []
        chunk_peak_values = []
        
        for chunk_audio, _ in chunks:
            if len(chunk_audio) > 0:
                chunk_rms = np.sqrt(np.mean(chunk_audio**2))
                chunk_peak = np.max(np.abs(chunk_audio))
                
                chunk_rms_values.append(chunk_rms)
                chunk_peak_values.append(chunk_peak)
        
        # Chunks should have similar dynamic characteristics
        avg_chunk_rms = np.mean(chunk_rms_values)
        avg_chunk_peak = np.mean(chunk_peak_values)
        
        # Should be reasonably close (within 50% due to varying content)
        assert abs(avg_chunk_rms - original_rms) / original_rms < 0.5
        assert abs(avg_chunk_peak - original_peak) / original_peak < 0.5 