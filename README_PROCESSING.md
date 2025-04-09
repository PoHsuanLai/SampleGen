# Audio Processing with All-in-One

This README provides instructions for processing audio files using the All-in-One (allin1) library with the provided script.

## Prerequisites

1. Make sure you have the allin1 library installed and configured correctly.
2. The NATTEN library should be properly integrated with allin1 as per previous fixes.

## Usage

The `process_all_artists.py` script allows you to process audio files from the artists directory using allin1 and save the analysis results to a specified output directory.

### Basic Usage

```bash
# Process all artists with default settings
python process_all_artists.py

# Process a specific artist
python process_all_artists.py --artist kanye

# Process a single audio file
python process_all_artists.py --file data/artists/kanye/001\ -\ White\ Dress.wav

# Set a custom output directory
python process_all_artists.py --output_dir custom_output

# Limit the number of songs processed per artist
python process_all_artists.py --limit 5
```

### Command-line Arguments

- `--artists_dir`: Base directory containing artist folders (default: "data/artists")
- `--output_dir`: Directory to save analysis results (default: "struct")
- `--artist`: Process a specific artist (optional)
- `--limit`: Limit the number of songs to process per artist (optional)
- `--file`: Process a single audio file (optional)

## Output

The script will:
1. Process each audio file using allin1.analyze
2. Save the analysis results to the specified output directory
3. Print progress information and summary statistics
4. Handle errors gracefully, allowing processing to continue even if some files fail

## Examples

### Process a Single Artist (e.g., Kanye West)

```bash
python process_all_artists.py --artist kanye
```

### Process Just 3 Songs Per Artist

```bash
python process_all_artists.py --limit 3
```

### Process All Artists and Save to a Custom Directory

```bash
python process_all_artists.py --output_dir analysis_results
```

## Troubleshooting

If you encounter CUDA-related errors, you can disable CUDA by setting the environment variable:

```bash
CUDA_VISIBLE_DEVICES="" python process_all_artists.py
```

This will run the processing on CPU only, which may be slower but more reliable in some environments.

## Script Structure

- `find_audio_files()`: Recursively finds all .wav files in a directory
- `process_artist_directory()`: Processes all audio files in a specific artist directory
- `main()`: Handles command-line arguments and orchestrates the processing workflow 