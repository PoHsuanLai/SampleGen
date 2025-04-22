# Mixer Training Pipeline

This document describes how to train the mixer model which generates mixing instructions for audio stems based on input audio and text prompts.

## Overview

The mixer is a multimodal model that takes a combination of text prompts and audio waveforms as input, and generates a sequence of mixing tool instructions as output. It uses LoRA (Low-Rank Adaptation) to efficiently fine-tune the model.

The training process involves:
1. Preparing audio data by chunking songs into segments
2. Creating on-the-fly augmentations by distorting audio stems
3. Training the model to generate instructions that would "fix" these distortions
4. Saving LoRA weights that can be applied to the base model

## Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Required Python packages (can be installed via `pip`):
  - transformers
  - peft
  - librosa
  - pydub
  - numpy
  - tqdm
  - scipy

## Directory Structure

```
├── data/                      # Data directory
│   ├── raw/                   # Raw audio files
│   ├── processed/             # Processed audio segments
│   └── dataset.json           # Dataset index
├── models/                    # Trained models
│   └── finetuned/
│       └── mixer/             # Mixer model checkpoints
├── src/
│   ├── data_processing/       # Data processing utilities
│   ├── music/                 # Music generation models
│   │   └── mixer/             # Mixer model and tools
│   └── tuning/                # Training scripts
│       ├── config/            # Configuration files
│       │   └── mixer_config.yaml  # Mixer training configuration
│       ├── tune_mixer.py      # Mixer training script
│       └── tune_mixer_dataset.py  # Dataset class for mixer training
```

## Configuration

The mixer training is configured through `src/tuning/config/mixer_config.yaml`. The main configuration sections are:

- `default`: General training parameters like batch size, learning rate, etc.
- `data`: Data processing parameters
- `mixer`: Model parameters and LoRA configuration
- `generators`: Configuration for music generators (if used)
- `augmentation`: Audio augmentation parameters

## Training the Mixer

### 1. Prepare Your Data

Place your raw audio files in the `data/raw/` directory. The files should be organized by artist:

```
data/raw/
├── artist1/
│   ├── song1.wav
│   └── song2.wav
├── artist2/
│   └── song1.wav
...
```

### 2. Run the Training Pipeline

The main training script is `src/tuning/tune_mixer.py`. You can run it with various command-line arguments:

```bash
python -m src.tuning.tune_mixer \
  --config src/tuning/config/mixer_config.yaml \
  --data_dir data/raw \
  --processed_dir data/processed \
  --dataset_file data/dataset.json \
  --num_epochs 3
```

#### Command Line Arguments

- `--config`: Path to the configuration file (default: `src/tuning/config/mixer_config.yaml`)
- `--data_dir`: Directory containing raw audio data (default: `data`)
- `--processed_dir`: Directory to save processed segments (default: `data/processed`)
- `--dataset_file`: Path to save dataset index (default: `data/dataset.json`)
- `--artist`: Process a specific artist only (optional)
- `--limit`: Limit the number of songs to process (optional)
- `--num_epochs`: Override the number of epochs in the config (optional)

### 3. Testing the Training Pipeline

You can test the training pipeline with a small sample of data:

```bash
python -m src.tuning.test_mixer_training \
  --config src/tuning/config/mixer_config.yaml \
  --epochs 1
```

This will create sample audio files, process them, and run a quick training loop to verify everything works correctly.

## Model Architecture

The mixer model consists of:
1. A text encoder (BERT-based) for processing text prompts
2. An audio encoder (CLAP-based) for processing audio waveforms
3. A fusion layer to combine text and audio embeddings
4. A decoder (Mistral-based) that generates mixing instructions

LoRA is applied to the decoder part of the model for efficient fine-tuning.

## Output Format

The model outputs mixing instructions as a sequence of tool calls, each with parameters. For example:

```
change_volume(db=-3.0) apply_eq(low_gain=2.0, mid_gain=-1.0, high_gain=1.5) apply_reverb()
```

These instructions can be interpreted and executed by the `MixingTools` class.

## Troubleshooting

- **Out of memory errors**: Decrease batch size or use gradient accumulation
- **Slow training**: Enable mixed precision training with `fp16: true` in the config
- **Poor convergence**: Adjust learning rate or increase warmup steps
- **NaN losses**: Check for extreme audio values or gradient explosion

## Advanced Usage

You can customize the training process by modifying the configuration file:

- To change the strength of augmentations, adjust parameters in the `augmentation` section
- To focus on specific mixing tools, modify their probability values
- To use a different base model, update the `mistral_model` parameter in the `mixer` section 