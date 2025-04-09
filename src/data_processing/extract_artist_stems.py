import os
import argparse
import glob
import subprocess
from tqdm import tqdm

def extract_artist_stems(input_dir, output_dir, model_name='htdemucs', limit=None):
    """
    Extract stems from artist audio files using demucs directly.
    """
    # Get all artist directories
    artist_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"Found {len(artist_dirs)} artist directories: {', '.join(artist_dirs)}")
    
    # Process each artist
    for artist in artist_dirs:
        artist_path = os.path.join(input_dir, artist)
        artist_output_path = os.path.join(output_dir, artist)
        
        # Create output directory
        os.makedirs(artist_output_path, exist_ok=True)
        
        # Get all audio files
        audio_files = glob.glob(os.path.join(artist_path, "*.wav"))
        
        if not audio_files:
            print(f"No audio files found for {artist}. Skipping.")
            continue
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            audio_files = audio_files[:limit]
            print(f"Processing {len(audio_files)} songs for {artist} (limited to {limit})")
        else:
            print(f"Processing {len(audio_files)} songs for {artist}")
        
        # Process each audio file
        for audio_file in tqdm(audio_files, desc=f"Extracting stems for {artist}"):
            # Get song name
            song_name = os.path.splitext(os.path.basename(audio_file))[0]
            
            # Create output directory
            song_output_dir = os.path.join(artist_output_path, song_name)
            os.makedirs(song_output_dir, exist_ok=True)
            
            # Check if stems already exist
            stem_dir = os.path.join(song_output_dir, model_name, song_name)
            if os.path.exists(stem_dir) and os.path.isdir(stem_dir):
                # Count existing stems
                stem_files = [f for f in os.listdir(stem_dir) if f.endswith('.wav')]
                if len(stem_files) >= 4:  # Most models provide 4 stems
                    print(f"Stems for {song_name} already exist. Skipping.")
                    continue
            
            # Get absolute path to audio file
            audio_file_abs = os.path.abspath(audio_file)
            
            # Check if the file exists
            if not os.path.isfile(audio_file_abs):
                print(f"Audio file not found: {audio_file_abs}")
                continue
            
            # Run demucs as a subprocess
            demucs_cmd = ["demucs", "-v", "-n", model_name, "-o", song_output_dir, audio_file_abs]
            print(f"Running command: {' '.join(demucs_cmd)}")
            
            try:
                subprocess.run(demucs_cmd, check=True)
                print(f"Successfully extracted stems for {song_name}")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting stems for {song_name}: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description="Extract stems from artist songs (simplified version)")
    parser.add_argument("--input_dir", default="data/artists", help="Directory containing artist folders")
    parser.add_argument("--output_dir", default="data/stems/artists", help="Directory to store extracted stems")
    parser.add_argument("--model", default="htdemucs", help="Demucs model to use")
    parser.add_argument("--artists", nargs="+", help="Specific artists to process (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of songs to process per artist")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # If specific artists are provided, process only those
    if args.artists:
        # Create a temporary input directory with just the requested artists
        temp_input_dir = {}
        for artist in args.artists:
            artist_path = os.path.join(args.input_dir, artist)
            if os.path.isdir(artist_path):
                temp_input_dir[artist] = artist_path
            else:
                print(f"Artist directory not found: {artist_path}")
                
        if not temp_input_dir:
            print("No valid artist directories found. Exiting.")
            return
            
        # Process each artist individually
        for artist, artist_path in temp_input_dir.items():
            # Create a temporary directory structure for extract_artist_stems
            single_artist_dir = os.path.dirname(artist_path)
            print(f"Processing artist: {artist} from {artist_path}")
            
            # Process directly without using the extract_artist_stems function
            artist_output_path = os.path.join(args.output_dir, artist)
            os.makedirs(artist_output_path, exist_ok=True)
            
            # Get all audio files
            audio_files = glob.glob(os.path.join(artist_path, "*.wav"))
            
            if not audio_files:
                print(f"No audio files found for {artist}. Skipping.")
                continue
            
            # Apply limit if specified
            if args.limit is not None and args.limit > 0:
                audio_files = audio_files[:args.limit]
                print(f"Processing {len(audio_files)} songs for {artist} (limited to {args.limit})")
            else:
                print(f"Processing {len(audio_files)} songs for {artist}")
            
            # Process each audio file
            for audio_file in tqdm(audio_files, desc=f"Extracting stems for {artist}"):
                # Get song name
                song_name = os.path.splitext(os.path.basename(audio_file))[0]
                
                # Create output directory
                song_output_dir = os.path.join(artist_output_path, song_name)
                os.makedirs(song_output_dir, exist_ok=True)
                
                # Check if stems already exist
                stem_dir = os.path.join(song_output_dir, args.model, song_name)
                if os.path.exists(stem_dir) and os.path.isdir(stem_dir):
                    # Count existing stems
                    stem_files = [f for f in os.listdir(stem_dir) if f.endswith('.wav')]
                    if len(stem_files) >= 4:  # Most models provide 4 stems
                        print(f"Stems for {song_name} already exist. Skipping.")
                        continue
                
                # Get absolute path to audio file
                audio_file_abs = os.path.abspath(audio_file)
                
                # Check if the file exists
                if not os.path.isfile(audio_file_abs):
                    print(f"Audio file not found: {audio_file_abs}")
                    continue
                
                # Run demucs as a subprocess
                demucs_cmd = ["demucs", "-v", "-n", args.model, "-o", song_output_dir, audio_file_abs]
                print(f"Running command: {' '.join(demucs_cmd)}")
                
                try:
                    subprocess.run(demucs_cmd, check=True)
                    print(f"Successfully extracted stems for {song_name}")
                except subprocess.CalledProcessError as e:
                    print(f"Error extracting stems for {song_name}: {e}")
                    continue
    else:
        # Process all artists
        extract_artist_stems(args.input_dir, args.output_dir, args.model, args.limit)
    
    print("Stem extraction complete!")

if __name__ == "__main__":
    main() 