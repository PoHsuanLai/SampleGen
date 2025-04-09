# Stem Extraction Data Processing Scripts

This directory contains scripts for processing music data and extracting stems using Demucs.

## Requirements

Make sure you have the required dependencies installed:

```bash
pip install tqdm pandas numpy soundfile demucs torch
```

## Scripts

### 1. Extract Stems from Artist Songs

This script processes complete songs from artists in the `data/artists/` directory and extracts stems using Demucs.

```bash
python src/data_processing/extract_artist_stems.py [OPTIONS]
```

#### Options:
- `--input_dir`: Directory containing artist folders (default: data/artists)
- `--output_dir`: Directory to store extracted stems (default: data/stems/artists)
- `--model`: Demucs model to use (default: htdemucs)
- `--transcribe`: Transcribe stems to MIDI (optional)
- `--artists`: Specific artists to process (optional, default: all)

#### Example:
```bash
# Process all artists
python src/data_processing/extract_artist_stems.py

# Process specific artists
python src/data_processing/extract_artist_stems.py --artists drake kendrick

# Use a different model and enable MIDI transcription
python src/data_processing/extract_artist_stems.py --model htdemucs_6s --transcribe
```

### 2. Extract Stems from FMA Hip-Hop Tracks

This script filters the FMA dataset for hip-hop tracks and extracts stems using Demucs.

```bash
python src/data_processing/extract_fma_hiphop_stems.py [OPTIONS]
```

#### Options:
- `--metadata_dir`: Directory containing FMA metadata (default: data/fma_metadata)
- `--fma_dir`: Directory containing FMA dataset (default: data/fma_small)
- `--output_dir`: Directory to store extracted stems (default: data/stems/fma_hiphop)
- `--model`: Demucs model to use (default: htdemucs)
- `--transcribe`: Transcribe stems to MIDI (optional)
- `--limit`: Limit number of tracks to process (optional)
- `--save_metadata`: Save filtered hip-hop metadata to CSV (optional)

#### Example:
```bash
# Process hip-hop tracks from the small FMA dataset
python src/data_processing/extract_fma_hiphop_stems.py

# Process hip-hop tracks from the medium FMA dataset
python src/data_processing/extract_fma_hiphop_stems.py --fma_dir data/fma_medium

# Process a limited number of tracks and save metadata
python src/data_processing/extract_fma_hiphop_stems.py --limit 100 --save_metadata
```

## Output Directory Structure

### Artist Stems

```
data/stems/artists/
├── drake/
│   ├── Headlines (Explicit)/
│   │   ├── htdemucs/
│   │   │   ├── Headlines (Explicit)/
│   │   │   │   ├── vocals.wav
│   │   │   │   ├── drums.wav
│   │   │   │   ├── bass.wav
│   │   │   │   └── other.wav
│   │   │   └── midi/ (if --transcribe is specified)
│   │   │       ├── vocals.mid
│   │   │       ├── drums.mid
│   │   │       └── ...
│   ├── HYFR/
│   │   └── ...
├── kendrick/
│   └── ...
└── ...
```

### FMA Hip-Hop Stems

```
data/stems/fma_hiphop/
├── hiphop_tracks.csv (if --save_metadata is specified)
├── 000/
│   ├── 000123/
│   │   ├── htdemucs/
│   │   │   ├── 000123/
│   │   │   │   ├── vocals.wav
│   │   │   │   ├── drums.wav
│   │   │   │   ├── bass.wav
│   │   │   │   └── other.wav
│   │   │   └── midi/ (if --transcribe is specified)
│   ├── 000456/
│   │   └── ...
├── 001/
│   └── ...
└── ...
```

## Available Demucs Models

- `htdemucs`: Hybrid Transformer Demucs (default)
- `htdemucs_ft`: Fine-tuned version
- `htdemucs_6s`: 6-source version (vocals, drums, bass, guitar, piano, other)
- `mdx`: Music Demixing model
- `mdx_extra`: Enhanced MDX
- `mdx_extra_q`: Quantized version 