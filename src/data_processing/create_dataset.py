"""
Dataset creation pipeline that processes raw audio files into the structured format
used by the Hip-Hop Producer model. This handles:
1. Beat/tempo analysis using all-in-one
2. Stem separation using Demucs
3. Metadata extraction and JSON creation
"""

import os
import sys
import subprocess
import json
import shutil
from typing import Dict, List, Optional, Tuple
import argparse
import librosa
import soundfile as sf
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.data_processing.stem_extraction import StemExtractor


class DatasetProcessor:
    """
    Process raw audio files into structured dataset format.
    """
    
    def __init__(self, 
                 output_dir: str = "data",
                 use_all_in_one: bool = True,
                 demucs_model: str = "htdemucs"):
        """
        Initialize the dataset processor.
        
        Args:
            output_dir: Base output directory for processed data
            use_all_in_one: Whether to use all-in-one for beat tracking
            demucs_model: Demucs model to use for stem separation
        """
        self.output_dir = output_dir
        self.use_all_in_one = use_all_in_one
        self.demucs_model = demucs_model
        
        # Initialize stem extractor
        self.stem_extractor = StemExtractor(model_name=demucs_model)
        
        print(f"Dataset processor initialized:")
        print(f"  Output directory: {output_dir}")
        print(f"  All-in-one: {use_all_in_one}")
        print(f"  Demucs model: {demucs_model}")
    
    def extract_metadata_with_all_in_one(self, audio_path: str) -> Optional[Dict]:
        """
        Extract beat tracking and tempo information using all-in-one.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with beats, downbeats, BPM, etc.
        """
        try:
            # Run all-in-one beat tracking
            cmd = ["python", "-m", "all_in_one", "--input", audio_path, "--output-dir", "temp_analysis"]
            
            print(f"Running all-in-one analysis on {audio_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
            
            if result.returncode != 0:
                print(f"All-in-one failed: {result.stderr}")
                return None
            
            # Find the output JSON file
            audio_name = Path(audio_path).stem
            json_files = list(Path("temp_analysis").glob(f"*{audio_name}*.json"))
            
            if not json_files:
                print(f"No JSON output found for {audio_path}")
                return None
            
            # Load the analysis results
            with open(json_files[0], 'r') as f:
                analysis_data = json.load(f)
            
            # Clean up temp files
            shutil.rmtree("temp_analysis", ignore_errors=True)
            
            return analysis_data
            
        except Exception as e:
            print(f"Error running all-in-one on {audio_path}: {e}")
            return None
    
    def extract_metadata_with_librosa(self, audio_path: str) -> Dict:
        """
        Extract basic metadata using librosa as fallback.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Basic metadata dictionary
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path)
            
            # Extract tempo and beats
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(beats, sr=sr)
            
            # Estimate downbeats (every 4th beat)
            downbeat_indices = range(0, len(beat_times), 4)
            downbeat_times = [beat_times[i] for i in downbeat_indices if i < len(beat_times)]
            
            # Create beat positions (1,2,3,4 pattern)
            beat_positions = [(i % 4) + 1 for i in range(len(beat_times))]
            
            # Create simple segments (for now, just verse/chorus alternating)
            duration = len(y) / sr
            segments = [
                {"start": 0.0, "end": duration/2, "label": "verse"},
                {"start": duration/2, "end": duration, "label": "chorus"}
            ]
            
            return {
                "path": audio_path,
                "bpm": float(tempo),
                "beats": beat_times.tolist(),
                "downbeats": downbeat_times,
                "beat_positions": beat_positions,
                "segments": segments
            }
            
        except Exception as e:
            print(f"Error extracting metadata with librosa: {e}")
            # Return minimal metadata
            return {
                "path": audio_path,
                "bpm": 120.0,  # Default BPM
                "beats": [],
                "downbeats": [],
                "beat_positions": [],
                "segments": [{"start": 0.0, "end": 30.0, "label": "unknown"}]
            }
    
    def process_single_file(self, 
                          audio_path: str,
                          artist_name: str,
                          output_subdir: Optional[str] = None) -> Optional[str]:
        """
        Process a single audio file into the structured format.
        
        Args:
            audio_path: Path to input audio file
            artist_name: Name of the artist
            output_subdir: Optional subdirectory name (uses filename if None)
            
        Returns:
            Path to created song directory or None if failed
        """
        try:
            # Determine output paths
            audio_filename = Path(audio_path).stem
            if output_subdir is None:
                output_subdir = audio_filename
            
            artist_dir = os.path.join(self.output_dir, artist_name)
            song_dir = os.path.join(artist_dir, output_subdir)
            stems_dir = os.path.join(song_dir, "stems")
            
            # Create directories
            os.makedirs(song_dir, exist_ok=True)
            os.makedirs(stems_dir, exist_ok=True)
            
            print(f"\nProcessing: {audio_path}")
            print(f"Output directory: {song_dir}")
            
            # 1. Copy/convert main audio file
            main_audio_path = os.path.join(song_dir, f"{audio_filename}.wav")
            if not os.path.exists(main_audio_path):
                if audio_path.endswith('.wav'):
                    shutil.copy2(audio_path, main_audio_path)
                else:
                    # Convert to WAV
                    audio_data, sr = librosa.load(audio_path, sr=None)
                    sf.write(main_audio_path, audio_data, sr)
                print(f"  Copied main audio: {main_audio_path}")
            
            # 2. Extract metadata
            if self.use_all_in_one:
                metadata = self.extract_metadata_with_all_in_one(main_audio_path)
                if metadata is None:
                    print("  All-in-one failed, falling back to librosa")
                    metadata = self.extract_metadata_with_librosa(main_audio_path)
            else:
                metadata = self.extract_metadata_with_librosa(main_audio_path)
            
            # Update path in metadata
            metadata["path"] = main_audio_path
            
            # 3. Extract stems using Demucs
            print("  Extracting stems with Demucs...")
            stems, _ = self.stem_extractor.extract_stems_from_file(
                main_audio_path, 
                output_dir=stems_dir
            )
            
            if stems:
                print(f"  Extracted {len(stems)} stems: {list(stems.keys())}")
                
                # Save individual stem files
                for stem_name, stem_audio in stems.items():
                    stem_path = os.path.join(stems_dir, f"{stem_name}.wav")
                    if not os.path.exists(stem_path):
                        # Convert from (channels, samples) to (samples, channels)
                        if len(stem_audio.shape) == 2:
                            stem_audio = stem_audio.T
                        sf.write(stem_path, stem_audio, self.stem_extractor.sample_rate)
            else:
                print("  Warning: No stems extracted")
            
            # 4. Save metadata JSON
            json_path = os.path.join(song_dir, f"{audio_filename}.json")
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"  Saved metadata: {json_path}")
            
            print(f"  ✓ Successfully processed {audio_filename}")
            return song_dir
            
        except Exception as e:
            print(f"  ✗ Error processing {audio_path}: {e}")
            return None
    
    def process_directory(self, 
                         input_dir: str,
                         artist_name: str) -> List[str]:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Directory containing audio files
            artist_name: Name of the artist
            
        Returns:
            List of successfully processed song directories
        """
        processed_dirs = []
        
        # Find all audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac'}
        audio_files = []
        
        for file in os.listdir(input_dir):
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(input_dir, file))
        
        print(f"Found {len(audio_files)} audio files in {input_dir}")
        
        # Process each file
        for audio_file in audio_files:
            result = self.process_single_file(audio_file, artist_name)
            if result:
                processed_dirs.append(result)
        
        print(f"\nCompleted processing {len(processed_dirs)}/{len(audio_files)} files")
        return processed_dirs


def scan_processed_directory(data_dir: str) -> Dict[str, List[str]]:
    """
    Scan the processed data directory and return organized structure.
    
    Args:
        data_dir: Base data directory
        
    Returns:
        Dictionary mapping artist -> list of song directories
    """
    structure = {}
    
    if not os.path.exists(data_dir):
        return structure
    
    for artist_dir in os.listdir(data_dir):
        artist_path = os.path.join(data_dir, artist_dir)
        if os.path.isdir(artist_path):
            songs = []
            for song_dir in os.listdir(artist_path):
                song_path = os.path.join(artist_path, song_dir)
                if os.path.isdir(song_path):
                    # Check if it has the expected structure
                    has_audio = any(f.endswith('.wav') for f in os.listdir(song_path))
                    has_json = any(f.endswith('.json') for f in os.listdir(song_path))
                    has_stems = os.path.exists(os.path.join(song_path, 'stems'))
                    
                    if has_audio and has_json and has_stems:
                        songs.append(song_path)
            
            if songs:
                structure[artist_dir] = songs
    
    return structure


def main():
    parser = argparse.ArgumentParser(description="Process audio files into structured dataset")
    parser.add_argument("--input", type=str, required=True, help="Input audio file or directory")
    parser.add_argument("--artist", type=str, required=True, help="Artist name")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--no-all-in-one", action="store_true", help="Skip all-in-one analysis")
    parser.add_argument("--demucs-model", type=str, default="htdemucs", help="Demucs model to use")
    parser.add_argument("--scan", action="store_true", help="Scan and report current dataset structure")
    
    args = parser.parse_args()
    
    if args.scan:
        # Scan existing dataset
        structure = scan_processed_directory(args.output)
        print(f"Dataset structure in {args.output}:")
        for artist, songs in structure.items():
            print(f"  {artist}: {len(songs)} songs")
        print(f"Total: {sum(len(songs) for songs in structure.values())} songs")
        return
    
    # Initialize processor
    processor = DatasetProcessor(
        output_dir=args.output,
        use_all_in_one=not args.no_all_in_one,
        demucs_model=args.demucs_model
    )
    
    # Process input
    if os.path.isfile(args.input):
        # Single file
        result = processor.process_single_file(args.input, args.artist)
        if result:
            print(f"Successfully processed file to: {result}")
        else:
            print("Processing failed")
    elif os.path.isdir(args.input):
        # Directory
        results = processor.process_directory(args.input, args.artist)
        print(f"Successfully processed {len(results)} files")
    else:
        print(f"Input path does not exist: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main() 