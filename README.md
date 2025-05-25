# Unified Hip-Hop Producer AI

This document describes the new **Unified Architecture** that simplifies the hip-hop music production system by removing the complex mixer implementations and creating a single unified model.

## Architecture Overview

### New Unified Architecture

The unified system consists of:

1. **UnifiedProducerModel** - Single model that handles:
   - Planning which stems to generate
   - Requesting generators to create stems
   - Generating Faust DSP mixing scripts
   - Quality assessment

2. **UnifiedProducerDataset** - Simplified dataset for training:
   - Stem generation planning
   - Faust script generation
   - Quality assessment

3. **UnifiedProducerTrainer** - Streamlined training pipeline:
   - Planning loss (which stems to generate)
   - Quality assessment loss
   - Faust script generation loss

## Key Improvements

✅ **Simplified Architecture**: Single unified model instead of separate mixer + generators  
✅ **Direct Faust Generation**: Generates Faust DSP scripts directly  
✅ **Cleaner Training**: Focused training objectives without complex multimodal fusion  
✅ **Easier Maintenance**: Reduced complexity while maintaining functionality  
✅ **Better Separation**: Clear separation between planning, generation, and mixing  

## Usage

### Demo Mode

```bash
# Run with audio input and text prompt
uv run src/main_unified_producer.py --mode demo \
    --input your_song.wav \
    --prompt "Create a dark trap beat with heavy 808s" \
    --output result.wav
```

### Training Mode

```bash
# Train the unified model
uv run src/main_unified_producer.py --mode train \
    --data-dir your_data_directory \
    --epochs 5 \
    --checkpoint model_checkpoint
```

### Faust Script Training

```bash
# Train specifically for Faust script generation
uv run src/main_unified_producer.py --mode faust-train \
    --data-dir your_data_directory \
    --epochs 3
```

## Model Components

### ProducerModel

```python
from src.models.unified_producer import UnifiedProducerModel

model = UnifiedProducerModel(
    device='cuda',
    text_model_name='bert-base-uncased',
    decoder_model_name='mistralai/Mistral-7B-v0.1'
)

# Plan production
plan = model.plan_production(text_prompt="Create a trap beat")

# Generate stems
stems = model.generate_stems(plan, duration=5.0)

# Generate Faust script
faust_script = model.generate_faust_script(stems, text_prompt)
```

### Training Pipeline

```python
from src.training.unified_trainer import UnifiedProducerTrainer

trainer = UnifiedProducerTrainer(model, data_dir, device='cuda')

# Train for one epoch
results = trainer.train_epoch()
print(f"Planning loss: {results['planning_loss']}")
print(f"Quality loss: {results['quality_loss']}")
print(f"Faust loss: {results['faust_loss']}")
```

## Data Structure

The unified system expects data in the following structure:

```
data/
├── artist1/
│   ├── song1/
│   │   ├── song1.wav          # Main audio file
│   │   ├── song1.json         # Metadata (BPM, key, segments)
│   │   └── stems/             # Pre-extracted stems (optional)
│   │       ├── vocals.wav
│   │       ├── drums.wav
│   │       ├── bass.wav
│   │       └── other.wav
│   │
│   └── song2/
│       └── ...
└── artist2/
    └── ...
```

## Faust Script Generation

The unified model generates Faust DSP scripts for mixing:

```faust
import("stdfaust.lib");

bass_process = _ : fi.highpass(1, 40) : co.compressor_mono(4, -20, 0.003, 0.1);
drums_process = _ : fi.peak_eq(2, 100, 1) : co.compressor_mono(3, -15, 0.001, 0.05);
melody_process = _ : fi.peak_eq(1.5, 2000, 0.7) : re.mono_freeverb(0.3, 0.5, 0.5, 0.5);
harmony_process = _ : fi.peak_eq(1.5, 2000, 0.7) : re.mono_freeverb(0.3, 0.5, 0.5, 0.5);

process = _, _, _, _ : bass_process, drums_process, melody_process, harmony_process :> _;
```

## Training Objectives

The unified model is trained with three main objectives:

1. **Planning Loss**: Binary classification for which stems to generate
2. **Quality Loss**: Regression for audio quality assessment
3. **Faust Loss**: Token-level similarity for Faust script generation

## Comparison with Old Architecture

| Aspect | Old Mixer-Based | New Unified |
|--------|----------------|-------------|
| Models | Mixer + Generators | UnifiedProducer + Generators |
| Complexity | High (multimodal fusion) | Low (focused objectives) |
| Training | Complex mixer dataset | Simplified unified dataset |
| Output | Mixing instructions | Direct Faust scripts |
| Maintenance | Difficult | Easy |
| Extensibility | Limited | High |

## Dependencies

- PyTorch
- Transformers (BERT, Mistral)
- Demucs (stem separation)
- SoundFile
- NumPy

## Getting Started

1. Install dependencies:
```bash
pip install torch transformers demucs soundfile numpy
```

2. Run the demo:
```bash
uv run src/main_unified_producer.py
```

3. Train on your data:
```bash
uv run src/main_unified_producer.py --mode train --data-dir your_data
```

## Future Enhancements

- [ ] Advanced Faust script templates
- [ ] Real-time audio processing
- [ ] Web interface for easy interaction
- [ ] Integration with DAWs
- [ ] Advanced quality metrics
- [ ] Multi-genre support beyond hip-hop 