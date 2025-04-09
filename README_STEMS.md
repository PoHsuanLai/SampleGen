# Stem Extraction Scripts

These scripts extract stems from audio files using the Demucs model. They work with both artist songs and the FMA dataset.

## Prerequisites

Make sure you have the required dependencies installed:
```bash
pip install pandas numpy tqdm demucs soundfile
```

## Scripts

### 1. Extract Stems from Artist Songs

```bash
python src/data_processing/extract_artist_stems_simple.py [OPTIONS]
```

#### Options:
- `--input_dir`: Directory containing artist folders (default: data/artists)
- `--output_dir`: Directory to store extracted stems (default: data/stems/artists)
- `--model`: Demucs model to use (default: htdemucs)
- `--artists`: Specific artists to process (optional, default: all)
- `--limit`: Limit number of songs to process per artist (optional)

#### Examples:
```bash
# Process all artists
python src/data_processing/extract_artist_stems_simple.py

# Process specific artists
python src/data_processing/extract_artist_stems_simple.py --artists drake kendrick

# Process a limited number of songs per artist
python src/data_processing/extract_artist_stems_simple.py --limit 5
```

### 2. Extract Stems from FMA Hip-Hop Tracks

```bash
python src/data_processing/extract_fma_hiphop_stems_simple.py [OPTIONS]
```

#### Options:
- `--metadata_dir`: Directory containing FMA metadata (default: data/fma_metadata)
- `--fma_dir`: Directory containing FMA dataset (default: data/fma_small)
- `--output_dir`: Directory to store extracted stems (default: data/stems/fma_hiphop)
- `--model`: Demucs model to use (default: htdemucs)
- `--limit`: Limit number of tracks to process (optional)
- `--save_metadata`: Save filtered hip-hop metadata to CSV (optional)

#### Examples:
```bash
# Process hip-hop tracks from the small FMA dataset
python src/data_processing/extract_fma_hiphop_stems_simple.py

# Process hip-hop tracks from the medium FMA dataset
python src/data_processing/extract_fma_hiphop_stems_simple.py --fma_dir data/fma_medium

# Process a limited number of tracks and save metadata
python src/data_processing/extract_fma_hiphop_stems_simple.py --limit 100 --save_metadata
```

## Output Structure

Each script produces stems in a structured directory format:

### Artist Stems
```
data/stems/artists/
├── drake/
│   ├── Headlines (Explicit)/
│   │   └── htdemucs/
│   │       └── Headlines (Explicit)/
│   │           ├── vocals.wav
│   │           ├── drums.wav
│   │           ├── bass.wav
│   │           └── other.wav
```

### FMA Hip-Hop Stems
```
data/stems/fma_hiphop/
├── 000/
│   ├── 000002/
│   │   └── htdemucs/
│   │       └── 000002/
│   │           ├── vocals.wav
│   │           ├── drums.wav
│   │           ├── bass.wav
│   │           └── other.wav
```

## Available Demucs Models

- `htdemucs`: Hybrid Transformer Demucs (default)
- `htdemucs_ft`: Fine-tuned version
- `htdemucs_6s`: 6-source version (vocals, drums, bass, guitar, piano, other)
- `mdx`: Music Demixing model
- `mdx_extra`: Enhanced MDX
- `mdx_extra_q`: Quantized version 