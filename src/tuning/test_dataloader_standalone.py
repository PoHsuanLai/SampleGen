#!/usr/bin/env python3
# test_dataloader_standalone.py
"""
Standalone script to test the MixerDataset with stem-level distortions and song structure awareness.
This script loads songs, applies distortions to individual stems, processes JSON metadata,
and generates the appropriate text prompts and tool tokens for training.

Usage:
    python test_dataloader_standalone.py --data_dir /path/to/data --num_samples 10 --output_dir ./data_test_output
"""

import sys
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from pydub import AudioSegment
import random
import soundfile as sf
from tqdm import tqdm

# Add the parent directory to the path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the MixerDataset class
from src.tuning.mixer_dataset import MixerDataset

def plot_waveform(audio_array, sr, title, file_path):
    """Plot audio waveform and save to file"""
    plt.figure(figsize=(10, 3))
    plt.plot(np.linspace(0, len(audio_array) / sr, len(audio_array)), audio_array)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def plot_spectrogram(audio_array, sr, title, file_path):
    """Plot audio spectrogram and save to file"""
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_array)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def safe_normalize(audio_array):
    """Safely normalize audio array to avoid division by zero warnings"""
    max_val = np.max(np.abs(audio_array))
    if max_val > 1e-10:  # Only normalize if there's a significant value
        audio_array = audio_array / max_val * 0.9
    return audio_array

def save_audio(audio_array, sr, file_path):
    """Save audio array to file"""
    # Ensure audio is normalized safely
    audio_array = safe_normalize(audio_array)
    sf.write(file_path, audio_array, sr)

def test_dataloader(data_dir, num_samples=5, output_dir='./data_test_output'):
    """
    Test the MixerDataset with stem-level distortions and song structure awareness.
    
    Args:
        data_dir: Path to the data directory
        num_samples: Number of random samples to analyze
        output_dir: Directory to save output files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer (same one used for mixer training)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create a basic config for the dataset
    config = {
        "segment_duration": 4.0,
        "sample_rate": 48000,
        "num_distortions": 3,  # Increased from 2 to 3 for more aggressive distortions
        "max_segments_per_song": 3,
        "segment_selection_strategy": "sequential",
        "include_json": True
    }
    
    # Initialize the dataset
    print(f"Initializing dataset from {data_dir}")
    dataset = MixerDataset(
        processed_dir=data_dir,
        tokenizer=tokenizer,
        config=config,
        segment_duration=config["segment_duration"],
        max_seq_len=1024,  # Increased from 512 to 1024 to handle longer prompts
        sample_rate=config["sample_rate"],
        num_distortions=config["num_distortions"],
        train_mode=True,
        max_segments_per_song=config["max_segments_per_song"],
        segment_selection_strategy=config["segment_selection_strategy"],
        include_json=config["include_json"]
    )
    
    print(f"Dataset contains {len(dataset)} songs")
    
    # Select random indices if there are enough songs
    if len(dataset) > 0:
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    else:
        print("No songs found in the dataset.")
        return
    
    # Process selected samples
    for sample_idx, idx in enumerate(tqdm(indices, desc="Processing samples")):
        print(f"\nProcessing sample {sample_idx+1}/{len(indices)} (dataset index {idx})")
        
        # Get the sample
        try:
            sample = dataset[idx]
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
        
        # Create a directory for this sample
        sample_dir = os.path.join(output_dir, f"sample_{sample_idx+1}")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Extract basic info
        song_info = sample.get("song_info", {})
        segments = sample.get("segments", [])
        stem_types = sample.get("stem_types", [])
        
        # Write summary info to a text file
        info_file = os.path.join(sample_dir, "info.txt")
        with open(info_file, "w") as f:
            f.write(f"Source audio: {song_info.get('full_path', 'unknown')}\n")
            
            # Write segment info
            f.write("Segments:\n")
            for i, segment in enumerate(segments):
                segment_label = segment.get('segment_label', segment.get('label', 'unknown'))
                segment_start = segment.get('start', 0)
                segment_end = segment.get('end', 0)
                
                # Check if both start and end are 0, indicating metadata is probably in a different format
                if segment_start == 0 and segment_end == 0:
                    if 'segment_idx' in segment:
                        # This is likely a metadata entry rather than a true segment
                        segment_idx = segment.get('segment_idx', i)
                        if 'stem_name' in segment:
                            stem_name = segment.get('stem_name', 'unknown')
                            chunk_idx = segment.get('chunk_idx', 0)
                            f.write(f"{i+1}. {segment_label}: [Metadata] stem={stem_name}, segment_idx={segment_idx}, chunk={chunk_idx}\n")
                        else:
                            f.write(f"{i+1}. {segment_label}: [Metadata] segment_idx={segment_idx}\n")
                    else:
                        f.write(f"{i+1}. {segment_label}: [No timing information available]\n")
                else:
                    f.write(f"{i+1}. {segment_label}: {segment_start:.2f}s - {segment_end:.2f}s\n")
                
                # Add chunk information for each segment
                if "chunks" in segment:
                    f.write(f"   Chunks per stem:\n")
                    for stem_name, chunk_data in segment["chunks"].items():
                        if isinstance(chunk_data, dict):  # New format
                            chunk_count = chunk_data.get("count", 0)
                            loop_info = ""
                            if "loop_counts" in chunk_data:
                                loop_info = f" (loops: {chunk_data['loop_counts']})"
                            f.write(f"     {stem_name}: {chunk_count}{loop_info}\n")
                        else:  # Old format (for backward compatibility)
                            f.write(f"     {stem_name}: {chunk_data}\n")
            
            # Write token identifiers section
            f.write("\nToken Identifiers:\n")
            if "token_identifiers" in sample:
                token_ids = sample["token_identifiers"]
                if token_ids and isinstance(token_ids, list):
                    for stem_idx, stem_tokens in enumerate(token_ids):
                        if stem_idx < len(stem_types):
                            stem_name = stem_types[stem_idx]
                        else:
                            stem_name = f"stem_{stem_idx}"
                        f.write(f"  {stem_name}: {stem_tokens}\n")
                else:
                    f.write("  [No token identifiers found or invalid format]\n")
            else:
                f.write("  [No token identifiers section in data]\n")
            
            f.write("\nText prompt:\n")
            f.write(sample["text_prompt"])
            f.write("\n\n")
            
            f.write("Tool tokens:\n")
            # Format tool tokens for better readability
            if "tool_tokens" in sample and sample["tool_tokens"]:
                tool_tokens = sample["tool_tokens"]
                
                # Break down by stem if the tokens use the standard format with <stem> markers
                stem_sections = {}
                current_stem = "unknown"
                
                # Try to parse into stem sections
                for token in tool_tokens.split():
                    if token.startswith("<") and token.endswith(">") and not token.startswith("<loop:") and ":" not in token[1:-1]:
                        potential_stem = token[1:-1]  # Remove < and >
                        if potential_stem in stem_types:
                            current_stem = potential_stem
                            if current_stem not in stem_sections:
                                stem_sections[current_stem] = []
                        else:
                            # Add to current stem
                            if current_stem in stem_sections:
                                stem_sections[current_stem].append(token)
                    else:
                        # Add to current stem
                        if current_stem in stem_sections:
                            stem_sections[current_stem].append(token)
                
                # If we successfully parsed into stems, output by stem
                if stem_sections and all(stem in stem_sections for stem in stem_types):
                    for stem_name in stem_types:
                        if stem_name in stem_sections:
                            f.write(f"  {stem_name}: ")
                            # Add stem marker back and format tokens nicely
                            formatted_tokens = f"<{stem_name}> " + " ".join(stem_sections[stem_name])
                            # Add line breaks for readability
                            formatted_tokens = formatted_tokens.replace("> <", ">\n    <")
                            f.write(formatted_tokens + "\n\n")
                else:
                    # If parsing failed, just output the tokens with some formatting
                    formatted_tokens = tool_tokens.replace("> <", ">\n  <")
                    f.write(formatted_tokens)
            else:
                f.write("[No tool tokens found]\n")
            
            f.write("\n")
            
            f.write("Input IDs (text prompt):\n")
            f.write(str(sample["input_ids"]))
            f.write("\n\n")
            
            f.write("Tool Input IDs:\n")
            f.write(str(sample["tool_input_ids"]))
            f.write("\n\n")
            
            # Add encoded audio information
            if "encoded_audio" in sample:
                f.write("Encoded Audio Information:\n")
                for key, value in sample["encoded_audio"].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Write stem actions
            f.write("Stem Actions:\n")
            for stem_name, segment_actions in sample.get("stem_actions", {}).items():
                f.write(f"{stem_name.upper()} STEM:\n")
                for action_idx, actions in enumerate(segment_actions):
                    # Check if loop count is available in the tool tokens
                    loop_info = ""
                    if "tool_tokens" in sample:
                        tokens = sample["tool_tokens"].split()
                        for i, token in enumerate(tokens):
                            if token.startswith(f"stem({stem_name}") and i+1 < len(tokens) and tokens[i+1].startswith("loop_count("):
                                loop_count = tokens[i+1].replace("loop_count(", "").replace(")", "")
                                loop_info = f" (loops: {loop_count})"
                                break
                    
                    f.write(f"  Chunk {action_idx+1}{loop_info}:\n")
                    for action in actions:
                        try:
                            f.write(f"    - {action.get('tool', 'unknown')}: {json.dumps(action.get('params', {}))}\n")
                        except Exception as e:
                            f.write(f"    - Error parsing action: {str(e)}\n")
            
            f.write("\nAudio Shape Information:\n")
            try:
                f.write(f"Distorted Audio Shape: {sample.get('distorted_audio', torch.tensor([])).shape}\n")
                f.write(f"Original Audio Shape: {sample.get('original_audio', torch.tensor([])).shape}\n")
            except Exception as e:
                f.write(f"Error getting audio shape: {str(e)}\n")
        
        # Process each stem
        for stem_idx, stem_name in enumerate(stem_types):
            try:
                stem_dir = os.path.join(sample_dir, stem_name)
                os.makedirs(stem_dir, exist_ok=True)
                
                # Get the number of chunks
                num_chunks = 0
                if "distorted_audio" in sample and sample["distorted_audio"] is not None and stem_idx < sample["distorted_audio"].shape[0]:
                    num_chunks = sample["distorted_audio"].shape[1]
                
                # Process each chunk for this stem
                for chunk_idx in range(num_chunks):
                    # Generate chunk label
                    chunk_prefix = f"chunk_{chunk_idx+1}"
                    
                    try:
                        # Get audio tensors (with error handling)
                        distorted_audio = sample["distorted_audio"][stem_idx, chunk_idx].numpy()
                        original_audio = sample["original_audio"][stem_idx, chunk_idx].numpy()
                        
                        # Save waveforms
                        plot_waveform(
                            distorted_audio, 
                            config["sample_rate"],
                            f"{stem_name} - {chunk_prefix} - Distorted Waveform",
                            os.path.join(stem_dir, f"{chunk_prefix}_distorted_waveform.png")
                        )
                        
                        plot_waveform(
                            original_audio, 
                            config["sample_rate"],
                            f"{stem_name} - {chunk_prefix} - Original Waveform",
                            os.path.join(stem_dir, f"{chunk_prefix}_original_waveform.png")
                        )
                        
                        # Save spectrograms
                        plot_spectrogram(
                            distorted_audio, 
                            config["sample_rate"],
                            f"{stem_name} - {chunk_prefix} - Distorted Spectrogram",
                            os.path.join(stem_dir, f"{chunk_prefix}_distorted_spectrogram.png")
                        )
                        
                        plot_spectrogram(
                            original_audio, 
                            config["sample_rate"],
                            f"{stem_name} - {chunk_prefix} - Original Spectrogram",
                            os.path.join(stem_dir, f"{chunk_prefix}_original_spectrogram.png")
                        )
                        
                        # Save audio files
                        save_audio(
                            distorted_audio,
                            config["sample_rate"],
                            os.path.join(stem_dir, f"{chunk_prefix}_distorted.wav")
                        )
                        
                        save_audio(
                            original_audio,
                            config["sample_rate"],
                            os.path.join(stem_dir, f"{chunk_prefix}_original.wav")
                        )
                    except Exception as e:
                        error_message = f"Error processing {stem_name} chunk {chunk_idx+1}: {str(e)}"
                        print(error_message)
                        with open(os.path.join(stem_dir, f"{chunk_prefix}_error.txt"), "w") as f:
                            f.write(error_message)
            except Exception as e:
                print(f"Error processing stem {stem_name}: {str(e)}")

        # Create a combined view for each chunk (this is an approximation since chunks may belong to different segments)
        try:
            num_chunks = 0
            if "distorted_audio" in sample and sample["distorted_audio"] is not None and sample["distorted_audio"].shape[0] > 0:
                num_chunks = sample["distorted_audio"].shape[1]
            
            for chunk_idx in range(num_chunks):
                try:
                    chunk_dir = os.path.join(sample_dir, f"chunk_{chunk_idx+1}")
                    os.makedirs(chunk_dir, exist_ok=True)
                    
                    # Create a combined audio by adding all stems for this chunk
                    combined_distorted = None
                    combined_original = None
                    
                    for stem_idx, stem_name in enumerate(stem_types):
                        if (stem_idx < sample["distorted_audio"].shape[0] and 
                            chunk_idx < sample["distorted_audio"].shape[1]):
                            try:
                                distorted_stem = sample["distorted_audio"][stem_idx, chunk_idx].numpy()
                                original_stem = sample["original_audio"][stem_idx, chunk_idx].numpy()
                                
                                # Initialize combined arrays if not done yet
                                if combined_distorted is None:
                                    combined_distorted = np.zeros_like(distorted_stem)
                                    combined_original = np.zeros_like(original_stem)
                                    
                                # Simple mixing (add stems)
                                combined_distorted += distorted_stem
                                combined_original += original_stem
                            except Exception as e:
                                print(f"Error mixing {stem_name} for chunk {chunk_idx+1}: {str(e)}")
                    
                    # Skip if we couldn't create combined audio
                    if combined_distorted is None or combined_original is None:
                        print(f"Skipping combined view for chunk {chunk_idx+1}: No valid audio data")
                        continue
                    
                    # Normalize combined audio
                    max_distorted = np.max(np.abs(combined_distorted))
                    if max_distorted > 1e-10:
                        combined_distorted = combined_distorted / max_distorted * 0.9
                        
                    max_original = np.max(np.abs(combined_original))
                    if max_original > 1e-10:
                        combined_original = combined_original / max_original * 0.9
                    
                    # Save combined waveforms
                    plot_waveform(
                        combined_distorted, 
                        config["sample_rate"],
                        f"Combined - Chunk {chunk_idx+1} - Distorted Waveform",
                        os.path.join(chunk_dir, f"combined_distorted_waveform.png")
                    )
                    
                    plot_waveform(
                        combined_original, 
                        config["sample_rate"],
                        f"Combined - Chunk {chunk_idx+1} - Original Waveform",
                        os.path.join(chunk_dir, f"combined_original_waveform.png")
                    )
                    
                    # Save combined spectrograms
                    plot_spectrogram(
                        combined_distorted, 
                        config["sample_rate"],
                        f"Combined - Chunk {chunk_idx+1} - Distorted Spectrogram",
                        os.path.join(chunk_dir, f"combined_distorted_spectrogram.png")
                    )
                    
                    plot_spectrogram(
                        combined_original, 
                        config["sample_rate"],
                        f"Combined - Chunk {chunk_idx+1} - Original Spectrogram",
                        os.path.join(chunk_dir, f"combined_original_spectrogram.png")
                    )
                    
                    # Save combined audio files
                    save_audio(
                        combined_distorted, 
                        config["sample_rate"],
                        os.path.join(chunk_dir, f"combined_distorted.wav")
                    )
                    
                    save_audio(
                        combined_original, 
                        config["sample_rate"],
                        os.path.join(chunk_dir, f"combined_original.wav")
                    )
                except Exception as e:
                    print(f"Error processing combined view for chunk {chunk_idx+1}: {str(e)}")
        except Exception as e:
            print(f"Error creating combined views: {str(e)}")
    
    print(f"\nCompleted processing {len(indices)} samples. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MixerDataset with stem-level distortions")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to process")
    parser.add_argument("--output_dir", type=str, default="./data_test_output", help="Directory to save outputs")
    args = parser.parse_args()
    
    test_dataloader(args.data_dir, args.num_samples, args.output_dir) 