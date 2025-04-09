import os
import argparse
import pandas as pd
import numpy as np
import glob
import subprocess
from tqdm import tqdm
import ast

def load_fma_metadata(metadata_dir):
    """
    Load FMA metadata and return a dataframe of tracks with genre information.
    """
    print("Loading tracks metadata...")
    tracks_file = os.path.join(metadata_dir, "tracks.csv")
    
    # Load multi-level column headers
    metadata = pd.read_csv(tracks_file, header=[0, 1], index_col=0, low_memory=False)
    
    # Print metadata structure
    print(f"Metadata columns shape: {metadata.columns.shape}")
    print(f"First level column headers: {metadata.columns.levels[0].tolist()}")
    print(f"Second level column headers: {metadata.columns.levels[1].tolist()}")
    
    # Check if genre information exists
    if 'track' in metadata.columns.levels[0] and 'genres' in metadata.columns.levels[1]:
        print("Found genres information in the metadata")
    else:
        print("Warning: Could not find genre information in expected location")
        
    # Load genres
    print("Loading genres metadata...")
    genres_file = os.path.join(metadata_dir, "genres.csv")
    genres = pd.read_csv(genres_file)
    
    # Map genre IDs to genres
    genre_map = dict(zip(genres['genre_id'], genres['title']))
    
    # Get hip-hop genre IDs
    hiphop_ids = genres[genres['title'].str.contains('Hip-Hop|Rap', case=False, regex=True)]['genre_id'].tolist()
    print(f"Found {len(hiphop_ids)} hip-hop related genre IDs: {hiphop_ids}")
    
    return metadata, hiphop_ids, genre_map

def filter_hiphop_tracks(metadata, hiphop_genre_ids):
    """
    Filter tracks for hip-hop genre.
    """
    print("Filtering tracks for hip-hop genre...")
    
    # Convert genre string to list for each track and check if any hip-hop genre is in the list
    def is_hiphop(genres_str):
        if pd.isna(genres_str):
            return False
        try:
            # Parse the string representation of list
            genres = ast.literal_eval(genres_str)
            # Check if any hip-hop genre ID is in the track's genres
            return any(genre_id in hiphop_genre_ids for genre_id in genres)
        except (ValueError, SyntaxError):
            return False
    
    # Filter tracks based on genre
    try:
        # Try to use multilevel index
        hiphop_tracks = metadata[metadata[('track', 'genres')].apply(is_hiphop)]
    except (KeyError, ValueError) as e:
        print(f"Error accessing multilevel index: {e}")
        # Try alternative column names
        for col in metadata.columns:
            if 'genre' in col[1].lower():
                print(f"Trying column: {col}")
                try:
                    hiphop_tracks = metadata[metadata[col].apply(is_hiphop)]
                    print(f"Found hip-hop tracks using column: {col}")
                    break
                except:
                    print(f"Failed with column: {col}")
        else:
            raise ValueError("Could not find a suitable genre column")
    
    print(f"Found {len(hiphop_tracks)} hip-hop tracks out of {len(metadata)} total tracks")
    
    # Return the index as a list (track IDs)
    return hiphop_tracks.index.tolist()

def find_track_paths(fma_dir, track_ids):
    """
    Find the file paths for the given track IDs in the FMA directory.
    """
    track_paths = {}
    
    print(f"Finding file paths for {len(track_ids)} tracks...")
    
    # For each track ID, find the corresponding file
    for track_id in track_ids:
        # FMA organizes files into subdirectories based on the first 3 digits of the track ID
        # Convert track ID to 6-digit string with leading zeros
        track_id_str = str(track_id).zfill(6)
        subdir = track_id_str[:3]
        
        # Construct the expected file path
        file_path = os.path.join(fma_dir, subdir, f"{track_id_str}.mp3")
        
        if os.path.exists(file_path):
            track_paths[track_id] = file_path
    
    print(f"Found {len(track_paths)} track files out of {len(track_ids)} track IDs")
    return track_paths

def extract_stems_from_tracks(track_paths, output_dir, model_name='htdemucs', limit=None):
    """
    Extract stems from the given tracks using demucs directly.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Limit the number of tracks if specified
    if limit:
        track_ids = list(track_paths.keys())[:limit]
    else:
        track_ids = list(track_paths.keys())
    
    print(f"Extracting stems from {len(track_ids)} hip-hop tracks...")
    
    # Process each track
    for track_id in tqdm(track_ids, desc="Processing tracks"):
        file_path = track_paths[track_id]
        
        # Create a directory structure based on track ID
        track_id_str = str(track_id).zfill(6)
        subdir = track_id_str[:3]
        track_output_dir = os.path.join(output_dir, subdir, track_id_str)
        
        # Check if stems already exist
        stem_dir = os.path.join(track_output_dir, model_name, track_id_str)
        if os.path.exists(stem_dir) and os.path.isdir(stem_dir):
            # Count existing stems
            stem_files = [f for f in os.listdir(stem_dir) if f.endswith('.wav')]
            if len(stem_files) >= 4:  # Most models provide 4 stems
                print(f"Stems for track {track_id} already exist. Skipping.")
                continue
        
        # Create output directory
        os.makedirs(track_output_dir, exist_ok=True)
        
        # Get absolute path to audio file
        audio_file_abs = os.path.abspath(file_path)
        
        # Check if the file exists
        if not os.path.isfile(audio_file_abs):
            print(f"Audio file not found: {audio_file_abs}")
            continue
        
        # Run demucs as a subprocess
        demucs_cmd = ["demucs", "-v", "-n", model_name, "-o", track_output_dir, audio_file_abs]
        print(f"Running command: {' '.join(demucs_cmd)}")
        
        try:
            subprocess.run(demucs_cmd, check=True)
            print(f"Successfully extracted stems for track {track_id}")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting stems for track {track_id}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Filter FMA dataset for hip-hop tracks and extract stems (simplified version)")
    parser.add_argument("--metadata_dir", default="data/fma_metadata", help="Directory containing FMA metadata")
    parser.add_argument("--fma_dir", default="data/fma_small", help="Directory containing FMA dataset")
    parser.add_argument("--output_dir", default="data/stems/fma_hiphop", help="Directory to store extracted stems")
    parser.add_argument("--model", default="htdemucs", help="Demucs model to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tracks to process")
    parser.add_argument("--save_metadata", action="store_true", help="Save filtered hip-hop metadata to CSV")
    
    args = parser.parse_args()
    
    # Load metadata
    metadata, hiphop_genre_ids, genre_map = load_fma_metadata(args.metadata_dir)
    
    # Filter hip-hop tracks
    hiphop_track_ids = filter_hiphop_tracks(metadata, hiphop_genre_ids)
    
    if args.save_metadata:
        # Save the filtered dataset
        output_csv = os.path.join(args.output_dir, "hiphop_tracks.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        
        # Get the hip-hop tracks from the metadata
        hiphop_metadata = metadata.loc[hiphop_track_ids]
        hiphop_metadata.to_csv(output_csv)
        print(f"Saved hip-hop tracks metadata to {output_csv}")
    
    # Find file paths for hip-hop tracks
    track_paths = find_track_paths(args.fma_dir, hiphop_track_ids)
    
    # Extract stems
    extract_stems_from_tracks(track_paths, args.output_dir, args.model, args.limit)
    
    print("Stem extraction complete!")

if __name__ == "__main__":
    main() 