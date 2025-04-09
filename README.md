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