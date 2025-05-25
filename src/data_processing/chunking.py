"""
Audio chunking utilities for creating training data from longer audio files.
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional
import soundfile as sf
import json
from pathlib import Path


class AudioChunker:
    """
    Split audio files into smaller chunks for training.
    """
    
    def __init__(self,
                 chunk_duration: float = 5.0,
                 overlap_duration: float = 0.5,
                 sample_rate: int = 44100):
        """
        Initialize the audio chunker.
        
        Args:
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            sample_rate: Audio sample rate
        """
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate
        
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.step_samples = self.chunk_samples - self.overlap_samples
    
    def chunk_audio(self, 
                   audio: np.ndarray,
                   metadata: Optional[Dict] = None) -> List[Tuple[np.ndarray, Dict]]:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio: Input audio array
            metadata: Optional metadata to propagate to chunks
            
        Returns:
            List of (chunk_audio, chunk_metadata) tuples
        """
        chunks = []
        
        # Calculate number of chunks
        if len(audio) <= self.chunk_samples:
            # Audio is shorter than chunk size
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "start_time": 0.0,
                "end_time": len(audio) / self.sample_rate,
                "chunk_index": 0,
                "total_chunks": 1
            })
            return [(audio, chunk_metadata)]
        
        num_chunks = (len(audio) - self.chunk_samples) // self.step_samples + 1
        
        for i in range(num_chunks):
            start_sample = i * self.step_samples
            end_sample = start_sample + self.chunk_samples
            
            # Extract chunk
            if end_sample <= len(audio):
                chunk = audio[start_sample:end_sample]
            else:
                # Pad the last chunk if necessary
                chunk = np.zeros(self.chunk_samples)
                remaining_samples = len(audio) - start_sample
                chunk[:remaining_samples] = audio[start_sample:]
            
            # Create chunk metadata
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                "start_time": start_sample / self.sample_rate,
                "end_time": end_sample / self.sample_rate,
                "chunk_index": i,
                "total_chunks": num_chunks,
                "original_duration": len(audio) / self.sample_rate
            })
            
            # Add beat/segment information if available
            if metadata and "beats" in metadata:
                chunk_metadata["beats"] = self._filter_beats_for_chunk(
                    metadata["beats"], 
                    start_sample / self.sample_rate,
                    end_sample / self.sample_rate
                )
            
            if metadata and "segments" in metadata:
                chunk_metadata["segments"] = self._filter_segments_for_chunk(
                    metadata["segments"],
                    start_sample / self.sample_rate,
                    end_sample / self.sample_rate
                )
            
            chunks.append((chunk, chunk_metadata))
        
        return chunks
    
    def chunk_stems(self,
                   stems: Dict[str, np.ndarray],
                   metadata: Optional[Dict] = None) -> List[Tuple[Dict[str, np.ndarray], Dict]]:
        """
        Chunk all stems consistently.
        
        Args:
            stems: Dictionary of stem_name -> audio_array
            metadata: Optional metadata
            
        Returns:
            List of (chunked_stems, chunk_metadata) tuples
        """
        # Find the minimum length across all stems
        min_length = min(len(audio) for audio in stems.values())
        
        # Trim all stems to the same length
        trimmed_stems = {}
        for stem_name, audio in stems.items():
            trimmed_stems[stem_name] = audio[:min_length]
        
        # Use one stem as reference for chunking
        reference_stem = next(iter(trimmed_stems.values()))
        reference_chunks = self.chunk_audio(reference_stem, metadata)
        
        # Create chunks for all stems
        chunked_stems_list = []
        
        for chunk_audio, chunk_metadata in reference_chunks:
            chunk_stems = {}
            start_sample = int(chunk_metadata["start_time"] * self.sample_rate)
            end_sample = int(chunk_metadata["end_time"] * self.sample_rate)
            
            for stem_name, stem_audio in trimmed_stems.items():
                if end_sample <= len(stem_audio):
                    chunk_stems[stem_name] = stem_audio[start_sample:end_sample]
                else:
                    # Pad if necessary
                    chunk = np.zeros(self.chunk_samples)
                    remaining_samples = len(stem_audio) - start_sample
                    if remaining_samples > 0:
                        chunk[:remaining_samples] = stem_audio[start_sample:]
                    chunk_stems[stem_name] = chunk
            
            chunked_stems_list.append((chunk_stems, chunk_metadata))
        
        return chunked_stems_list
    
    def _filter_beats_for_chunk(self, 
                               beats: List[float],
                               start_time: float,
                               end_time: float) -> List[float]:
        """Filter beats that fall within the chunk time range."""
        chunk_beats = []
        for beat_time in beats:
            if start_time <= beat_time < end_time:
                # Adjust beat time relative to chunk start
                chunk_beats.append(beat_time - start_time)
        return chunk_beats
    
    def _filter_segments_for_chunk(self,
                                  segments: List[Dict],
                                  start_time: float,
                                  end_time: float) -> List[Dict]:
        """Filter segments that overlap with the chunk time range."""
        chunk_segments = []
        
        for segment in segments:
            seg_start = segment.get("start", 0)
            seg_end = segment.get("end", float('inf'))
            
            # Check if segment overlaps with chunk
            if seg_start < end_time and seg_end > start_time:
                # Adjust segment times relative to chunk
                adjusted_segment = segment.copy()
                adjusted_segment["start"] = max(0, seg_start - start_time)
                adjusted_segment["end"] = min(end_time - start_time, seg_end - start_time)
                chunk_segments.append(adjusted_segment)
        
        return chunk_segments
    
    def save_chunks(self,
                   chunks: List[Tuple[np.ndarray, Dict]],
                   output_dir: str,
                   base_name: str) -> List[str]:
        """
        Save chunks to individual files.
        
        Args:
            chunks: List of (audio, metadata) tuples
            output_dir: Output directory
            base_name: Base name for chunk files
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        
        for i, (chunk_audio, chunk_metadata) in enumerate(chunks):
            # Save audio
            audio_filename = f"{base_name}_chunk_{i:03d}.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            sf.write(audio_path, chunk_audio, self.sample_rate)
            
            # Save metadata
            json_filename = f"{base_name}_chunk_{i:03d}.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, 'w') as f:
                json.dump(chunk_metadata, f, indent=2)
            
            saved_paths.append(audio_path)
        
        return saved_paths
    
    def save_stem_chunks(self,
                        chunked_stems: List[Tuple[Dict[str, np.ndarray], Dict]],
                        output_dir: str,
                        base_name: str) -> List[str]:
        """
        Save stem chunks to organized directories.
        
        Args:
            chunked_stems: List of (stems_dict, metadata) tuples
            output_dir: Output directory
            base_name: Base name for chunk files
            
        Returns:
            List of saved chunk directories
        """
        saved_dirs = []
        
        for i, (stems_dict, chunk_metadata) in enumerate(chunked_stems):
            chunk_dir = os.path.join(output_dir, f"{base_name}_chunk_{i:03d}")
            stems_dir = os.path.join(chunk_dir, "stems")
            os.makedirs(stems_dir, exist_ok=True)
            
            # Save individual stems
            for stem_name, stem_audio in stems_dict.items():
                stem_path = os.path.join(stems_dir, f"{stem_name}.wav")
                sf.write(stem_path, stem_audio, self.sample_rate)
            
            # Save metadata
            json_path = os.path.join(chunk_dir, f"{base_name}_chunk_{i:03d}.json")
            with open(json_path, 'w') as f:
                json.dump(chunk_metadata, f, indent=2)
            
            saved_dirs.append(chunk_dir)
        
        return saved_dirs


def chunk_dataset(input_dir: str,
                 output_dir: str,
                 chunk_duration: float = 5.0,
                 overlap_duration: float = 0.5) -> None:
    """
    Process an entire dataset directory by chunking all songs.
    
    Args:
        input_dir: Input directory with artist/song structure
        output_dir: Output directory for chunked data
        chunk_duration: Duration of each chunk
        overlap_duration: Overlap between chunks
    """
    chunker = AudioChunker(chunk_duration, overlap_duration)
    
    for artist_dir in os.listdir(input_dir):
        artist_path = os.path.join(input_dir, artist_dir)
        if not os.path.isdir(artist_path):
            continue
        
        print(f"Processing artist: {artist_dir}")
        
        for song_dir in os.listdir(artist_path):
            song_path = os.path.join(artist_path, song_dir)
            if not os.path.isdir(song_path):
                continue
            
            print(f"  Processing song: {song_dir}")
            
            # Find main audio file and metadata
            audio_file = None
            json_file = None
            stems_dir = None
            
            for file in os.listdir(song_path):
                if file.endswith('.wav') and not os.path.isdir(os.path.join(song_path, file)):
                    audio_file = os.path.join(song_path, file)
                elif file.endswith('.json'):
                    json_file = os.path.join(song_path, file)
                elif file == 'stems' and os.path.isdir(os.path.join(song_path, file)):
                    stems_dir = os.path.join(song_path, file)
            
            if not audio_file:
                print(f"    No audio file found, skipping")
                continue
            
            # Load metadata
            metadata = {}
            if json_file and os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
            
            # Load stems if available
            stems = {}
            if stems_dir and os.path.exists(stems_dir):
                for stem_file in os.listdir(stems_dir):
                    if stem_file.endswith('.wav'):
                        stem_name = os.path.splitext(stem_file)[0]
                        stem_path = os.path.join(stems_dir, stem_file)
                        stem_audio, sr = sf.read(stem_path)
                        if len(stem_audio.shape) > 1:
                            stem_audio = np.mean(stem_audio, axis=1)  # Convert to mono
                        stems[stem_name] = stem_audio
            
            # Create output directory
            output_artist_dir = os.path.join(output_dir, artist_dir)
            output_song_dir = os.path.join(output_artist_dir, song_dir)
            
            if stems:
                # Chunk stems
                chunked_stems = chunker.chunk_stems(stems, metadata)
                saved_dirs = chunker.save_stem_chunks(
                    chunked_stems, output_song_dir, song_dir
                )
                print(f"    Saved {len(saved_dirs)} stem chunks")
            else:
                # Chunk main audio only
                audio_data, sr = sf.read(audio_file)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                chunks = chunker.chunk_audio(audio_data, metadata)
                saved_paths = chunker.save_chunks(chunks, output_song_dir, song_dir)
                print(f"    Saved {len(saved_paths)} audio chunks")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk audio dataset for training")
    parser.add_argument("--input", type=str, required=True, help="Input dataset directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory for chunks")
    parser.add_argument("--duration", type=float, default=5.0, help="Chunk duration in seconds")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap duration in seconds")
    
    args = parser.parse_args()
    
    chunk_dataset(args.input, args.output, args.duration, args.overlap) 