#!/usr/bin/env python3
"""
Test script for the mixer training pipeline.
This script verifies that the mixer training pipeline works as expected
by training on a small sample of data.
"""

import argparse
import logging
import os
from pathlib import Path
import torch

from .tune_mixer_sft import run_pipeline, train_mixer
from .config.config_utils import load_mixer_config

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_cuda_availability():
    """Verify CUDA availability and display device information."""
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        logger.info(f"Device capability: {torch.cuda.get_device_capability(0)}")
        logger.info(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA is not available. Training will be slow on CPU.")

def create_sample_data(data_dir, num_samples=5):
    """
    Create a small sample of data for testing the training pipeline.
    This is a placeholder - in a real setting, you'd have actual audio files to process.
    
    Args:
        data_dir: Directory to create the sample data in
        num_samples: Number of sample audio files to create
    """
    logger.info(f"Creating sample data in {data_dir}")
    
    # Create directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if we need to create sample data
    if len(list(Path(data_dir).rglob("*.wav"))) > 0:
        logger.info(f"Sample data already exists in {data_dir}")
        return
    
    try:
        # Import required libraries
        import numpy as np
        from scipy.io import wavfile
        from pydub import AudioSegment
        
        # Generate a few sample audio files
        for i in range(num_samples):
            # Generate a simple sine wave
            sample_rate = 44100
            duration = 5  # seconds
            frequencies = [440 * (i+1) for i in range(3)]  # Create harmonics
            
            t = np.linspace(0, duration, sample_rate * duration)
            signal = np.zeros_like(t)
            
            # Add multiple frequencies
            for freq in frequencies:
                signal += np.sin(2 * np.pi * freq * t) / len(frequencies)
            
            # Normalize
            signal = signal / np.max(np.abs(signal)) * 0.8
            
            # Convert to 16-bit PCM
            signal_16bit = (signal * 32767).astype(np.int16)
            
            # Save as WAV
            filename = os.path.join(data_dir, f"sample_{i+1}.wav")
            wavfile.write(filename, sample_rate, signal_16bit)
            
            logger.info(f"Created sample audio file: {filename}")
            
    except ImportError:
        logger.error("Could not import required libraries for sample data generation.")
        raise

def test_mixer_training(config_path, data_dir=None, processed_dir=None, dataset_file=None, num_epochs=1, max_train_samples=None):
    """
    Test the mixer training pipeline.
    
    Args:
        config_path: Path to the mixer configuration file
        data_dir: Directory containing raw audio data (will create sample data if not provided)
        processed_dir: Directory to save processed segments
        dataset_file: Path to save dataset index
        num_epochs: Number of epochs to train for
        max_train_samples: Maximum number of training samples to use
    """
    # Verify CUDA availability
    verify_cuda_availability()
    
    # Set default paths if not provided
    if not data_dir:
        data_dir = os.path.join("data", "test_samples")
        create_sample_data(data_dir)
    
    if not processed_dir:
        processed_dir = os.path.join("data", "test_processed")
    
    if not dataset_file:
        dataset_file = os.path.join("data", "test_dataset.json")
    
    # Load the config
    config = load_mixer_config(config_path)
    
    # Override the number of epochs if specified
    if num_epochs is not None:
        if "default" not in config:
            config["default"] = {}
        config["default"]["num_epochs"] = num_epochs
    
    # Override the max_train_samples if specified
    if max_train_samples is not None:
        if "data" not in config:
            config["data"] = {}
        config["data"]["max_train_samples"] = max_train_samples
    
    # Run the pipeline
    logger.info("Testing the mixer training pipeline")
    model = run_pipeline(
        config_path,
        data_dir,
        processed_dir,
        dataset_file,
        num_epochs=num_epochs
    )
    
    logger.info("Mixer training test completed successfully")
    return model

def main():
    parser = argparse.ArgumentParser(description='Test the mixer training pipeline')
    parser.add_argument('--config', type=str, default='src/tuning/config/mixer_config.yaml',
                        help='Path to the mixer configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing raw audio data (will create sample data if not provided)')
    parser.add_argument('--processed_dir', type=str, default=None,
                        help='Directory to save processed segments')
    parser.add_argument('--dataset_file', type=str, default=None,
                        help='Path to save dataset index')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs to train for')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Maximum number of training samples to use')
    args = parser.parse_args()
    
    # Run the test
    test_mixer_training(
        args.config,
        args.data_dir,
        args.processed_dir,
        args.dataset_file,
        args.epochs,
        args.max_train_samples
    )

if __name__ == "__main__":
    main() 