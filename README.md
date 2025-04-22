# SampleGen: AI-Powered Hip-Hop Beat Generation

SampleGen is an AI-powered beat generation system that combines sample-based production with machine learning to create hip-hop beats. It uses a pipeline of planning, generation, and mixing modules to produce high-quality music from text prompts and audio inputs.

## Features

- **Text-to-Beat Generation**: Create beats from natural language descriptions
- **Sample Analysis and Re-synthesis**: Extract stems from existing tracks and transform them
- **Multi-stage Pipeline**:
  - **Planner**: Understands music structure and creates beat blueprints
  - **Generator**: Creates individual musical components (drums, bass, others)
  - **Mixer**: Professionally mixes the components for high-quality output
- **Stem Separation**: Splits audio tracks into vocals, drums, bass, and other components
- **MIDI Transcription**: Can transcribe audio to MIDI for further editing

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/samplegen.git
   cd samplegen
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Download pre-trained models:
   ```bash
   python scripts/download_models.py
   ```

## Usage

### Command Line Interface

Generate a beat from an input audio file and text prompt:

```bash
python src/main.py --input samples/input.wav --output output/my_beat.wav --prompt "Boom bap style beat with heavy bass and jazzy piano samples"
```

### Options

- `--input`, `-i`: Path to the input audio file
- `--output`, `-o`: Path to save the output audio file (default: generated_song.wav)
- `--prompt`, `-p`: Natural language prompt describing the desired music
- `--duration`, `-d`: Maximum duration of the output in seconds (default: 30.0)
- `--stem-model`, `-s`: Stem separation model to use (default: spleeter:5stems)

## Project Structure

```
samplegen/
├── data/                   # Data storage directory
│   ├── stems/              # Extracted audio stems
│   └── fma_metadata/       # FMA dataset metadata
├── models/                 # Pre-trained model storage
├── src/                    # Source code
│   ├── data_processing/    # Data processing and stem extraction
│   ├── music/              # Music generation modules
│   │   ├── planner/        # Beat structure planning
│   │   ├── generator/      # Component generators
│   │   └── mixer/          # Audio mixing and mastering
│   ├── tuning/             # Fine-tuning scripts
│   ├── utils/              # Utility functions
│   ├── main.py             # CLI entry point
│   └── samplegen.py        # Main SampleGen class
├── samples/                # Example input audio samples
└── output/                 # Generated output directory
```

## Core Components

### 1. Stem Extraction

The system uses Meta's Demucs to separate audio tracks into stems (vocals, drums, bass, other). These stems can be used for analysis, conditioning, or directly in the output mix.

### 2. Music Planning

The planner uses audio and language understanding models to create a structured plan for the beat, determining sections, instruments, and style.

### 3. Music Generation

Multiple specialized generators create different components of the beat:
- **MelodyGenerator**: Creates melodic lines
- **BassGenerator**: Creates bass lines and 808s
- **DrumGenerator**: Creates drum patterns
- **HarmonyGenerator**: Creates chord progressions and harmonic elements

### 4. Audio Mixing

The MixingModule professionally mixes all generated components with frequency-specific processing, spatial effects, compression, and mastering.

## Examples

### Basic Beat Generation

```python
from samplegen import SampleGen

# Initialize SampleGen
samplegen = SampleGen()

# Generate a beat from an input file and prompt
summary = samplegen.process(
    input_audio_path="samples/loop.wav",
    user_prompt="Dark trap beat with heavy 808s and atmospheric pads",
    output_audio_path="output/dark_trap.wav",
    duration=30.0
)

print(summary)
```

### Custom Generation

```python
from samplegen import SampleGen

# Initialize with custom settings
samplegen = SampleGen(
    stem_model='htdemucs_6s',  # 6-stem separation model
    sample_rate=48000,         # Higher sample rate
    max_duration=60.0          # Longer output
)

# Generate a complete track from scratch
audio = samplegen.generate(
    prompt="Lo-fi hip-hop beat with jazzy piano, vinyl crackle, and laid-back drums"
)

# Save the output
import soundfile as sf
sf.write("output/lofi_beat.wav", audio.T, 48000)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Demucs](https://github.com/facebookresearch/demucs) for stem separation
- [MusicGen](https://github.com/facebookresearch/audiocraft) for music generation
- [FMA Dataset](https://github.com/mdeff/fma) for training data

# Audio Mixer Training Pipeline

This repository contains code for processing audio data and training a mixing model that learns to apply various mixing operations to improve audio quality.

## Overview

The mixer model is designed to take distorted audio as input and generate a plan of mixing operations to improve the audio quality. It uses a combination of audio and text encoders connected to a decoder language model to generate the appropriate mixing actions.

## Project Structure

- `src/data_processing/`: Contains scripts for audio processing and dataset creation
  - `chunking.py`: Splits audio files into segments based on JSON annotations
  - `audio_distorter.py`: Contains distortion methods for on-the-fly augmentation
  - `create_dataset.py`: Creates a dataset index of processed audio segments
- `src/music/mixer/`: Contains the mixer model implementation
  - `mixer.py`: The main mixer model that generates mixing actions
  - `mixing_tools.py`: Various audio mixing tools used by the mixer
- `src/tuning/`: Contains training and evaluation scripts
  - `tune_mixer.py`: Main script for training the mixer model

## Data Processing

The pipeline involves several steps:

1. **Chunking**: Split full songs into segments based on JSON annotations
2. **Dataset Creation**: Create an index of all segments for easier access
3. **Runtime Distortion**: During training, apply random distortions to original audio segments

## Training Pipeline

### Option 1: End-to-End Pipeline

Process data and train the model in a single command:

```bash
python -m src.tuning.tune_mixer \
  --config src/tuning/config/mixer_config.yaml \
  --data_dir data \
  --processed_dir data/processed \
  --dataset_file data/dataset.json
```

### Option 2: Step-by-Step

If you prefer to run the pipeline in steps:

1. Process the data:

```bash
python -m src.data_processing.chunking \
  --artists_dir data \
  --output_dir data/processed
```

2. Create the dataset index:

```bash
python -m src.data_processing.create_dataset \
  --processed_dir data/processed \
  --output_file data/dataset.json
```

3. Train the model:

```bash
python -m src.tuning.tune_mixer \
  --config src/tuning/config/mixer_config.yaml \
  --skip_processing
```

## Configuration Options

The mixer training can be configured through the `mixer_config.yaml` file:

- Data parameters (segment duration, sample rate)
- Model parameters (backbone models)
- Augmentation settings (types and probabilities of distortions)
- Training parameters (batch size, learning rate, etc.)

## Runtime Distortions

The audio distorter applies various transformations to create training examples:

- **Volume Changes**: Adjust audio volume up or down
- **Panning**: Create left/right imbalance
- **Filtering**: Add high/low frequency content that needs filtering
- **Compression**: Create dynamic range issues
- **EQ Adjustments**: Create frequency imbalances
- **Effects**: Training examples for delay and reverb

These distortions are applied on-the-fly during training, rather than pre-generating distorted files.

## Example Usage

```python
# Apply combined distortions to an audio file
from pydub import AudioSegment
from src.data_processing import AudioDistorter

# Load an audio file
audio = AudioSegment.from_file("path/to/audio.wav")

# Create distorter
distorter = AudioDistorter(audio)

# Apply random distortions
distorted_audio, actions = distorter.get_combined_distortions(num_distortions=2)

# Save the distorted audio
distorted_audio.export("distorted.wav", format="wav")

# The actions contain the mixing operations needed to fix the audio
print(actions)
``` 