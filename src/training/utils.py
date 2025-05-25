"""
Utility functions for the Hip-Hop Producer training pipeline.
Extracted from the original mixer inference and training code.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import soundfile as sf
from pydub import AudioSegment
import json
import os
import tempfile
from datetime import datetime


def safe_normalize_audio(audio: np.ndarray, target_level: float = 0.95) -> np.ndarray:
    """
    Safely normalize audio to prevent clipping.
    
    Args:
        audio: Audio array to normalize
        target_level: Target peak level (0.0 to 1.0)
        
    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio
    
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio * (target_level / max_val)
    return audio


def load_audio_with_fallback(file_path: str, sample_rate: int = 44100) -> Tuple[np.ndarray, int]:
    """
    Load audio file with fallback error handling.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_data, actual_sample_rate)
    """
    try:
        # Try with soundfile first
        audio_data, sr = sf.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        return audio_data, sr
        
    except Exception as e:
        print(f"Warning: Could not load {file_path} with soundfile: {e}")
        
        try:
            # Fallback to pydub
            audio_segment = AudioSegment.from_file(file_path)
            audio_data = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            
            # Normalize to [-1, 1] range
            if audio_segment.sample_width == 2:  # 16-bit
                audio_data = audio_data / 32768.0
            elif audio_segment.sample_width == 3:  # 24-bit
                audio_data = audio_data / 8388608.0
            elif audio_segment.sample_width == 4:  # 32-bit
                audio_data = audio_data / 2147483648.0
            
            return audio_data, audio_segment.frame_rate
            
        except Exception as e2:
            print(f"Error: Could not load {file_path} with any method: {e2}")
            # Return silence as fallback
            return np.zeros(sample_rate), sample_rate


def prepare_audio_for_model(audio: np.ndarray, 
                          target_length: Optional[int] = None,
                          sample_rate: int = 44100) -> torch.Tensor:
    """
    Prepare audio for model input with consistent formatting.
    
    Args:
        audio: Input audio array
        target_length: Target length in samples (optional)
        sample_rate: Sample rate
        
    Returns:
        Preprocessed audio tensor
    """
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Handle target length
    if target_length is not None:
        if len(audio) > target_length:
            # Random crop
            start = np.random.randint(0, len(audio) - target_length)
            audio = audio[start:start + target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
    
    # Normalize
    audio = safe_normalize_audio(audio)
    
    # Convert to tensor
    return torch.from_numpy(audio)


def extract_segment_from_audio(audio: np.ndarray, 
                             segment_info: Dict[str, Any],
                             sample_rate: int = 44100) -> np.ndarray:
    """
    Extract a specific segment from audio based on metadata.
    
    Args:
        audio: Full audio array
        segment_info: Dictionary containing start/end times
        sample_rate: Audio sample rate
        
    Returns:
        Extracted audio segment
    """
    start_sec = segment_info.get("start", 0)
    end_sec = segment_info.get("end", len(audio) / sample_rate)
    
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)
    
    # Ensure valid bounds
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    if start_sample >= end_sample:
        # Return small silence if invalid segment
        return np.zeros(int(0.1 * sample_rate))
    
    return audio[start_sample:end_sample]


def validate_stems_dict(stems: Dict[str, np.ndarray], 
                       required_stems: List[str] = None) -> Dict[str, np.ndarray]:
    """
    Validate and clean stems dictionary.
    
    Args:
        stems: Dictionary of stem_name -> audio_array
        required_stems: List of required stem names
        
    Returns:
        Validated stems dictionary
    """
    if required_stems is None:
        required_stems = ["vocals", "drums", "bass", "other"]
    
    validated_stems = {}
    
    for stem_name in required_stems:
        if stem_name in stems and isinstance(stems[stem_name], np.ndarray):
            audio = stems[stem_name]
            if len(audio) > 0:
                validated_stems[stem_name] = safe_normalize_audio(audio)
            else:
                # Create silence if empty
                validated_stems[stem_name] = np.zeros(44100)  # 1 second of silence
        else:
            # Create silence if missing
            validated_stems[stem_name] = np.zeros(44100)
    
    return validated_stems


def save_inference_results(results: Dict[str, Any], 
                         output_dir: str,
                         prefix: str = "inference") -> Dict[str, str]:
    """
    Save inference results with proper organization.
    
    Args:
        results: Dictionary containing inference results
        output_dir: Directory to save results
        prefix: Prefix for output files
        
    Returns:
        Dictionary of saved file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    saved_paths = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save audio if present
    if "audio" in results:
        audio_path = os.path.join(output_dir, f"{prefix}_{timestamp}.wav")
        sf.write(audio_path, results["audio"], results.get("sample_rate", 44100))
        saved_paths["audio"] = audio_path
    
    # Save generated text/instructions
    if "instructions" in results:
        text_path = os.path.join(output_dir, f"{prefix}_{timestamp}_instructions.txt")
        with open(text_path, "w") as f:
            f.write(results["instructions"])
        saved_paths["instructions"] = text_path
    
    # Save metadata
    metadata = {
        "timestamp": timestamp,
        "parameters": results.get("parameters", {}),
        "prompt": results.get("prompt", ""),
        "model_info": results.get("model_info", {})
    }
    
    metadata_path = os.path.join(output_dir, f"{prefix}_{timestamp}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    saved_paths["metadata"] = metadata_path
    
    return saved_paths


def compute_audio_similarity(audio1: np.ndarray, audio2: np.ndarray) -> float:
    """
    Compute similarity between two audio arrays.
    
    Args:
        audio1: First audio array
        audio2: Second audio array
        
    Returns:
        Similarity score between 0 and 1
    """
    # Ensure same length
    min_len = min(len(audio1), len(audio2))
    if min_len == 0:
        return 0.0
    
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    # Compute normalized cross-correlation
    if np.std(audio1) == 0 or np.std(audio2) == 0:
        return 0.0
    
    correlation = np.corrcoef(audio1, audio2)[0, 1]
    
    # Handle NaN values
    if np.isnan(correlation):
        return 0.0
    
    return abs(correlation)


def create_training_batch(samples: List[Dict[str, Any]], 
                        device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Create a training batch from list of samples.
    
    Args:
        samples: List of sample dictionaries
        device: Target device for tensors
        
    Returns:
        Batched data dictionary
    """
    batch = {}
    
    # Handle stems
    if "stems" in samples[0]:
        stem_names = list(samples[0]["stems"].keys())
        batch["stems"] = {}
        
        for stem_name in stem_names:
            stem_batch = []
            for sample in samples:
                if stem_name in sample["stems"]:
                    stem_audio = sample["stems"][stem_name]
                    stem_tensor = prepare_audio_for_model(stem_audio)
                    stem_batch.append(stem_tensor)
                else:
                    # Pad with zeros if missing
                    stem_batch.append(torch.zeros(44100))
            
            batch["stems"][stem_name] = torch.stack(stem_batch).to(device)
    
    # Handle text prompts
    if "style_prompt" in samples[0]:
        batch["prompts"] = [sample["style_prompt"] for sample in samples]
    
    # Handle other fields
    for field in ["ground_truth_dsp", "segments", "song_info"]:
        if field in samples[0]:
            batch[field] = [sample[field] for sample in samples]
    
    return batch


def log_training_metrics(metrics: Dict[str, float], 
                        step: int,
                        writer=None) -> None:
    """
    Log training metrics to console and tensorboard.
    
    Args:
        metrics: Dictionary of metric_name -> value
        step: Training step number
        writer: Optional tensorboard writer
    """
    # Console logging
    metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    print(f"Step {step}: {metric_str}")
    
    # Tensorboard logging
    if writer is not None:
        for metric_name, value in metrics.items():
            writer.add_scalar(metric_name, value, step) 