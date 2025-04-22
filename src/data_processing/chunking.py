import os
import glob
import json
from tqdm import tqdm
import argparse
import numpy as np
import soundfile as sf
from pydub import AudioSegment

def get_song_files(artists_dir, stems_base_dir, specific_song=None):
    """Find all song files (audio WAV, JSON, and stem files)."""
    songs = []
    
    # Walk through artist directories
    for artist_path, _, files in os.walk(artists_dir):
        # Skip if it's not a direct artist directory
        if os.path.dirname(artist_path) != artists_dir:
            continue
        
        artist_name = os.path.basename(artist_path)
        
        # Look for song directories within artist directory
        for song_dir in os.listdir(artist_path):
            song_path = os.path.join(artist_path, song_dir)
            
            # Skip if not a directory or if we're looking for a specific song
            if not os.path.isdir(song_path) or (specific_song and song_dir != specific_song):
                continue
            
            # Look for audio WAV and JSON files
            wav_file = None
            json_file = None
            for file in os.listdir(song_path):
                file_path = os.path.join(song_path, file)
                if file.endswith(".wav") and not os.path.isdir(file_path):
                    wav_file = file_path
                elif file.endswith(".json"):
                    json_file = file_path
            
            # Skip if both WAV and JSON are not found
            if not wav_file or not json_file:
                print(f"Skipping {song_path}: WAV or JSON file not found")
                continue
            
            # Find stems folder
            stems_dir = os.path.join(song_path, "stems")
            stem_files = []
            if os.path.exists(stems_dir) and os.path.isdir(stems_dir):
                stem_files = [os.path.join(stems_dir, f) for f in os.listdir(stems_dir) 
                             if f.endswith(".wav") and not os.path.isdir(os.path.join(stems_dir, f))]
            
            # Add song info to list
            songs.append({
                "artist": artist_name,
                "song_name": song_dir,
                "wav_file": wav_file,
                "json_file": json_file,
                "stem_files": stem_files,
                "song_dir": song_path
            })
    
    return songs

def cut_song_into_segments(song_info, output_dir):
    """Cut a song and its stems into segments based on the JSON file."""
    # Make sure output directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        song_output_dir = os.path.join(output_dir, song_info["artist"], song_info["song_name"])
        os.makedirs(song_output_dir, exist_ok=True)
    else:
        song_output_dir = os.path.join(os.path.dirname(song_info["wav_file"]), "segments")
        os.makedirs(song_output_dir, exist_ok=True)
    
    # Load JSON data
    with open(song_info["json_file"], 'r') as f:
        json_data = json.load(f)
    
    # Get segments from JSON
    segments = json_data.get("segments", [])
    if not segments:
        print(f"No segments found in JSON for {song_info['song_name']}")
        return
    
    # Load the WAV file
    full_audio = AudioSegment.from_wav(song_info["wav_file"])
    
    # Cut the full audio into segments
    for i, segment in enumerate(segments):
        start_time = segment["start"] * 1000  # Convert to milliseconds
        end_time = segment["end"] * 1000
        label = segment["label"]
        
        # Extract the segment
        segment_audio = full_audio[start_time:end_time]
        
        # Save the segment
        segment_filename = f"{i:02d}_{label}_{start_time/1000:.2f}_{end_time/1000:.2f}.wav"
        segment_path = os.path.join(song_output_dir, segment_filename)
        segment_audio.export(segment_path, format="wav")
        
        print(f"Saved segment: {segment_path}")
    
    # Process each stem file similarly
    stems_output_dir = os.path.join(song_output_dir, "stems")
    os.makedirs(stems_output_dir, exist_ok=True)
    
    for stem_file in song_info["stem_files"]:
        stem_name = os.path.basename(stem_file).split('.')[0]  # Get stem name without extension
        stem_audio = AudioSegment.from_wav(stem_file)
        
        # Create directory for this stem
        stem_segments_dir = os.path.join(stems_output_dir, stem_name)
        os.makedirs(stem_segments_dir, exist_ok=True)
        
        # Cut the stem audio into segments
        for i, segment in enumerate(segments):
            start_time = segment["start"] * 1000  # Convert to milliseconds
            end_time = segment["end"] * 1000
            label = segment["label"]
            
            # Extract the segment
            segment_audio = stem_audio[start_time:end_time]
            
            # Save the segment
            segment_filename = f"{i:02d}_{label}_{start_time/1000:.2f}_{end_time/1000:.2f}.wav"
            segment_path = os.path.join(stem_segments_dir, segment_filename)
            segment_audio.export(segment_path, format="wav")
    
    return song_output_dir

def process_songs(artists_dir, output_dir=None, limit=None, specific_song=None):
    """Process all songs found in the artists directory."""
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(artists_dir), "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find stem files base location
    stems_base_dir = os.path.join(os.path.dirname(artists_dir), "stems") if os.path.dirname(artists_dir) else "stems"
    
    # Get all song files
    songs = get_song_files(artists_dir, stems_base_dir, specific_song)
    
    # Apply limit if specified
    if limit and limit > 0:
        songs = songs[:limit]
    
    print(f"Found {len(songs)} songs to process")
    
    for i, song in enumerate(songs):
        print(f"Processing song {i+1}/{len(songs)}: {song['artist']}/{song['song_name']}")
        print(f"  Audio: {song['wav_file']}")
        print(f"  JSON: {song['json_file']}")
        print(f"  Stems: {len(song['stem_files'])} stem files found")
        
        # Cut the song into segments
        segments_dir = cut_song_into_segments(song, output_dir)
        print(f"  Segments saved to: {segments_dir}")
        
    return songs

def main():
    parser = argparse.ArgumentParser(description="Process audio files with their JSON and stems")
    parser.add_argument("--artists_dir", default="data", help="Base directory containing artist folders")
    parser.add_argument("--output_dir", default="data/processed", help="Directory to save processing results")
    parser.add_argument("--artist", help="Process a specific artist (optional)")
    parser.add_argument("--song", help="Process a specific song from the artist (requires --artist)")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of files to process")
    args = parser.parse_args()

    artists_dir = args.artists_dir
    if args.artist:
        artists_dir = os.path.join(args.artists_dir, args.artist)

    process_songs(artists_dir, args.output_dir, args.limit, args.song)

if __name__ == "__main__":
    main()