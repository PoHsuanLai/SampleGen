#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

def scan_processed_directory(processed_dir):
    """
    Scan a processed directory to find all audio segments and create a dataset index.
    
    Args:
        processed_dir: Path to the processed directory containing segmented audio files
        
    Returns:
        A dataset dictionary with metadata about all segments
    """
    # Initialize dataset structure
    dataset = {
        "segments": [],
        "stems": {},
        "artists": {},
        "song_sections": {}
    }
    
    # Find all WAV files in the processed directory
    segment_files = []
    for root, _, files in os.walk(processed_dir):
        for file in files:
            if file.endswith(".wav") and not file.startswith("._"):
                segment_files.append(os.path.join(root, file))
    
    # Process each segment file
    print(f"Found {len(segment_files)} segment files")
    for segment_file in tqdm(segment_files, desc="Processing segments"):
        # Parse file path to extract metadata
        rel_path = os.path.relpath(segment_file, processed_dir)
        path_parts = Path(rel_path).parts
        
        # Skip stem files for now (we'll process them differently)
        if "stems" in path_parts:
            stem_type = path_parts[-2]  # Get the stem type (vocals, drums, etc.)
            
            # Find the parent segment (non-stem)
            parent_segment_path = os.path.join(*path_parts[:-3])  # Remove "stems/stem_type/filename.wav"
            parent_segment_name = os.path.basename(segment_file).split('.')[0]
            
            # Add to stems dictionary
            if parent_segment_path not in dataset["stems"]:
                dataset["stems"][parent_segment_path] = {}
            
            dataset["stems"][parent_segment_path][stem_type] = {
                "path": rel_path,
                "segment_name": parent_segment_name
            }
            
            continue
        
        # Extract metadata from the filename
        # Format: 01_verse_1.20_10.50.wav
        filename = os.path.basename(segment_file)
        name_parts = os.path.splitext(filename)[0].split('_')
        
        if len(name_parts) >= 4:
            segment_idx = int(name_parts[0])
            section_type = name_parts[1]
            start_time = float(name_parts[2])
            end_time = float(name_parts[3])
            
            # Extract artist and song name from the path
            artist = path_parts[0] if len(path_parts) > 0 else "unknown"
            song_name = path_parts[1] if len(path_parts) > 1 else "unknown"
            
            # Create segment entry
            segment_entry = {
                "path": rel_path,
                "artist": artist,
                "song": song_name,
                "section": section_type,
                "index": segment_idx,
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time
            }
            
            # Add to main segments list
            dataset["segments"].append(segment_entry)
            
            # Update artists index
            if artist not in dataset["artists"]:
                dataset["artists"][artist] = {}
            if song_name not in dataset["artists"][artist]:
                dataset["artists"][artist][song_name] = []
            dataset["artists"][artist][song_name].append(segment_entry["path"])
            
            # Update song sections index
            if section_type not in dataset["song_sections"]:
                dataset["song_sections"][section_type] = []
            dataset["song_sections"][section_type].append(segment_entry["path"])
    
    # Add some summary statistics
    dataset["stats"] = {
        "total_segments": len(dataset["segments"]),
        "total_artists": len(dataset["artists"]),
        "total_stems": sum(len(stems) for stems in dataset["stems"].values()),
        "section_counts": {section: len(paths) for section, paths in dataset["song_sections"].items()}
    }
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Create a dataset index from processed audio segments")
    parser.add_argument("--processed_dir", default="data/processed", help="Directory containing processed segments")
    parser.add_argument("--output_file", default="data/dataset.json", help="Output JSON file for the dataset index")
    args = parser.parse_args()
    
    # Create the dataset index
    dataset = scan_processed_directory(args.processed_dir)
    
    # Save the dataset index as JSON
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Dataset index saved to {args.output_file}")
    print(f"Total segments: {dataset['stats']['total_segments']}")
    print(f"Total artists: {dataset['stats']['total_artists']}")
    print(f"Total stems: {dataset['stats']['total_stems']}")
    print("Section counts:")
    for section, count in dataset['stats']['section_counts'].items():
        print(f"  {section}: {count}")

if __name__ == "__main__":
    main() 