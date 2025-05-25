"""
YouTube downloader for hip-hop music data collection.
Downloads audio and extracts metadata for training data.
"""

import os
import sys
import subprocess
import json
from typing import Dict, List, Optional
import argparse


def download_youtube_audio(url: str, output_dir: str, artist_name: str = None) -> Optional[str]:
    """
    Download audio from YouTube URL using yt-dlp.
    
    Args:
        url: YouTube URL
        output_dir: Directory to save downloaded audio
        artist_name: Optional artist name for organization
        
    Returns:
        Path to downloaded file or None if failed
    """
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure output filename template
        if artist_name:
            output_template = os.path.join(output_dir, f"%(autonumber)03d - %(title)s.%(ext)s")
        else:
            output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
        
        # yt-dlp command
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",  # Best quality
            "--output", output_template,
            "--write-info-json",
            "--no-playlist",
            url
        ]
        
        print(f"Downloading audio from: {url}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Download successful!")
            # Find the downloaded file
            for file in os.listdir(output_dir):
                if file.endswith('.wav') and url.split('/')[-1] in file or 'youtube' in file.lower():
                    return os.path.join(output_dir, file)
            return None
        else:
            print(f"Download failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


def download_from_playlist(playlist_url: str, output_dir: str, artist_name: str = None, limit: int = None) -> List[str]:
    """
    Download multiple tracks from a YouTube playlist.
    
    Args:
        playlist_url: YouTube playlist URL
        output_dir: Directory to save downloaded audio
        artist_name: Artist name for organization
        limit: Maximum number of tracks to download
        
    Returns:
        List of downloaded file paths
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure output template
        if artist_name:
            output_template = os.path.join(output_dir, f"%(autonumber)03d - %(title)s.%(ext)s")
        else:
            output_template = os.path.join(output_dir, "%(playlist_index)03d - %(title)s.%(ext)s")
        
        # yt-dlp command for playlist
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav", 
            "--audio-quality", "0",
            "--output", output_template,
            "--write-info-json",
            "--yes-playlist",
            playlist_url
        ]
        
        if limit:
            cmd.extend(["--playlist-end", str(limit)])
        
        print(f"Downloading playlist: {playlist_url}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Playlist download successful!")
            # Find downloaded files
            downloaded_files = []
            for file in os.listdir(output_dir):
                if file.endswith('.wav'):
                    downloaded_files.append(os.path.join(output_dir, file))
            return downloaded_files
        else:
            print(f"Playlist download failed: {result.stderr}")
            return []
            
    except Exception as e:
        print(f"Error downloading playlist {playlist_url}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Download hip-hop music from YouTube for training data")
    parser.add_argument("--url", type=str, help="YouTube URL to download")
    parser.add_argument("--playlist", type=str, help="YouTube playlist URL")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    parser.add_argument("--artist", type=str, help="Artist name for organization")
    parser.add_argument("--limit", type=int, help="Limit number of downloads from playlist")
    
    args = parser.parse_args()
    
    if args.url:
        # Single video download
        if args.artist:
            artist_dir = os.path.join(args.output, args.artist)
        else:
            artist_dir = args.output
            
        downloaded_file = download_youtube_audio(args.url, artist_dir, args.artist)
        if downloaded_file:
            print(f"Downloaded: {downloaded_file}")
        else:
            print("Download failed")
            
    elif args.playlist:
        # Playlist download
        if args.artist:
            artist_dir = os.path.join(args.output, args.artist)
        else:
            artist_dir = args.output
            
        downloaded_files = download_from_playlist(args.playlist, artist_dir, args.artist, args.limit)
        print(f"Downloaded {len(downloaded_files)} files")
        
    else:
        print("Please provide either --url or --playlist")
        sys.exit(1)


if __name__ == "__main__":
    main() 