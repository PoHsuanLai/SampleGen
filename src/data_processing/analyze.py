#!/usr/bin/env python3
"""
Script to process all audio files from the artists directory using allin1
and save the analysis results to the struct directory.
"""

import os
import sys
import glob
import time
import argparse
from tqdm import tqdm
import allin1

def find_audio_files(base_dir):
    """Find all .wav files recursively in the given directory."""
    audio_files = []
    for root, _, _ in os.walk(base_dir):
        wav_files = glob.glob(os.path.join(root, "*.wav"))
        audio_files.extend(wav_files)
    return audio_files

def process_artist_directory(artist_dir, output_dir, limit=None, overwrite=True):
    """Process all audio files in a specific artist directory."""
    audio_files = glob.glob(os.path.join(artist_dir, "*.wav"))
    
    if limit:
        audio_files = audio_files[:limit]
    
    print(f"Found {len(audio_files)} audio files in {artist_dir}")
    
    results = []
    for audio_file in tqdm(audio_files, desc=f"Processing {os.path.basename(artist_dir)}"):
        try:
            result = allin1.analyze(
                audio_file,
                out_dir=artist_dir,
                overwrite=overwrite,
            )
            results.append((audio_file, result))
            print(f"Results saved to {output_dir}")
        except Exception as e:
            print(f"Error processing {output_dir}: {e}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Process audio files with allin1")
    parser.add_argument("--artists_dir", default="data/artists", help="Base directory containing artist folders")
    parser.add_argument("--output_dir", default="data/stems/artists", help="Directory to save analysis results")
    parser.add_argument("--artist", help="Process a specific artist (optional)")
    parser.add_argument("--limit", type=int, help="Limit the number of songs to process per artist")
    parser.add_argument("--file", help="Process a single audio file")
    parser.add_argument("--overwrite", help="Overwrite existing analysis results")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.file:
        # Process a single file
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}")
            return
        print(f"Processing single file: {args.file}")
        result = allin1.analyze(args.file, out_dir=args.output_dir)
        print(f"Analysis complete. Result saved to {args.output_dir}")
        return
    
    if args.artist:
        # Process a specific artist
        artist_dir = os.path.join(args.artists_dir, args.artist)
        output_dir = os.path.join(args.output_dir, args.artist)
        if not os.path.exists(artist_dir):
            print(f"Artist directory not found: {artist_dir}")
            return
        
        results = process_artist_directory(artist_dir, output_dir, args.limit, args.overwrite)
        print(f"Processed {len(results)} files for {args.artist}")
    else:
        # Process all artists
        if not os.path.exists(args.artists_dir):
            print(f"Artists directory not found: {args.artists_dir}")
            return
        
        artist_dirs = [d for d in glob.glob(os.path.join(args.artists_dir, "*")) if os.path.isdir(d)]
        print(f"Found {len(artist_dirs)} artist directories")
        
        total_processed = 0
        for artist_dir in artist_dirs:
            artist_name = os.path.basename(artist_dir)
            output_dir = os.path.join(args.output_dir, artist_name)
            print(f"Processing artist: {artist_name}")
            results = process_artist_directory(artist_dir, output_dir, args.limit, args.overwrite)
            total_processed += len(results)
            print(f"Completed {artist_name}: {len(results)} files processed")
        
        print(f"All done! Processed {total_processed} files across {len(artist_dirs)} artists")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds") 