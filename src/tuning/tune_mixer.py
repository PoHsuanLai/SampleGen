# train_mixer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
from ..music.mixer.mixer import Mixer
from ..music.mixer.mixing_tools import MixingTools
from ..music.generator import OtherGenerator, BassGenerator, DrumGenerator
from .config.config_utils import (
    load_mixer_config, 
    get_mixer_config, 
    get_mixer_data_config, 
    get_mixer_augmentation_config,
    get_models_dir,
    get_mixer_stems
)
import glob
import os
import json
import random
import numpy as np
from pydub import AudioSegment
import tqdm
import logging
import librosa
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MixerDataset(Dataset):
    def __init__(self, 
                root_dir: str, 
                artists: List[str] = None,
                config: Dict[str, Any] = None,
                mistral_tokenizer=None):
        """
        Dataset for the mixer training.
        
        Args:
            root_dir: Root directory containing artist data
            artists: List of artists to include (None for all)
            config: Configuration dictionary
            mistral_tokenizer: Tokenizer for Mistral model
        """
        self.root_dir = root_dir
        
        # Load configuration
        if config is None:
            config = load_mixer_config()
        
        data_config = get_mixer_data_config(config)
        augmentation_config = get_mixer_augmentation_config(config)
        
        # Set data parameters from config
        self.segment_duration = data_config.get('segment_duration', 5.0)
        self.max_seq_len = data_config.get('max_seq_len', 64)
        self.sample_rate = data_config.get('sample_rate', 44100)
        self.stem_types = get_mixer_stems(config)
        
        # Set augmentation parameters
        self.augmentation_config = augmentation_config
        
        # Load the tokenizer if not provided
        if mistral_tokenizer is None:
            # Get the Mistral model path from config
            mixer_config = config.get('mixer', {})
            mistral_model_path = mixer_config.get('mistral_model', 'mistralai/Mistral-7B-v0.3')
            self.tokenizer = AutoTokenizer.from_pretrained(mistral_model_path)
            # Add special tokens for mixer operations from config
            special_tokens = config.get('mixer', {}).get('special_tokens', [])
            if special_tokens:
                self.tokenizer.add_special_tokens({
                    'additional_special_tokens': special_tokens
                })
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = mistral_tokenizer
            # Ensure pad token is set if using provided tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Find all artist directories if none specified
        if artists is None:
            artists = [d for d in os.listdir(os.path.join(root_dir, 'stems', 'artists')) 
                     if os.path.isdir(os.path.join(root_dir, 'stems', 'artists', d))]
        self.artists = artists
        
        # Load all track metadata and prepare examples
        self.examples = self._load_tracks()
        logger.info(f"Loaded {len(self.examples)} track segments for training")
    
    def _load_tracks(self) -> List[Dict]:
        """Load all track metadata and prepare dataset examples"""
        examples = []
        
        for artist in self.artists:
            artist_stems_dir = os.path.join(self.root_dir, 'stems', 'artists', artist)
            artist_metadata_dir = os.path.join(self.root_dir, 'artists', artist)
            
            # Get all tracks with stems available
            track_dirs = [d for d in os.listdir(artist_stems_dir) 
                        if os.path.isdir(os.path.join(artist_stems_dir, d))]
            

            pbar = tqdm.tqdm(track_dirs, desc="Loading tracks", total=len(track_dirs))
            for track_dir in pbar:
                # Find the corresponding metadata file
                track_name = track_dir.split('/')[-1]
                json_file = os.path.join(artist_metadata_dir, f"{track_name}.json")
                
                if not os.path.exists(json_file):
                    logger.warning(f"No metadata found for {track_name}, skipping")
                    continue
                
                # Get stem files
                stems_path = os.path.join(artist_stems_dir, track_dir, 'htdemucs', track_name)
                if not os.path.exists(stems_path):
                    logger.warning(f"No stems found for {track_name}, skipping")
                    continue
                
                # Load metadata
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                
                # Create examples from segments
                track_examples = self._create_segment_examples(stems_path, metadata)
                examples.extend(track_examples)
                pbar.update(1)
                
        return examples
    
    def _create_segment_examples(self, stems_path: str, metadata: Dict) -> List[Dict]:
        """
        Create training examples by segmenting stems according to song structure
        
        Args:
            stems_path: Path to the stem files
            metadata: Track metadata with beats and segments information
            
        Returns:
            List of examples for training
        """
        examples = []
        segments = metadata.get('segments', [])
        if not segments:
            logger.warning(f"No segments found for {stems_path}, skipping")
            return examples
            
        # Load stem files (excluding vocals)
        stems = {}
        for stem_type in self.stem_types:
            stem_file = os.path.join(stems_path, f"{stem_type}.wav")
            if os.path.exists(stem_file):
                stems[stem_type] = AudioSegment.from_file(stem_file)
        
        if not stems:
            logger.warning(f"No stems found at {stems_path}, skipping")
            return examples
        
        # Process each segment that's longer than our desired duration
        for segment in segments:
            start_sec = segment['start']
            end_sec = segment['end']
            duration = end_sec - start_sec
            
            # Skip if segment is too short
            if duration < self.segment_duration:
                continue
                
            # Divide segment into chunks of segment_duration
            for chunk_start in np.arange(start_sec, end_sec - self.segment_duration, self.segment_duration):
                chunk_end = chunk_start + self.segment_duration
                
                # Extract stems for this chunk
                chunk_stems = {}
                for stem_type, audio in stems.items():
                    # Convert seconds to milliseconds for pydub
                    start_ms = int(chunk_start * 1000)
                    end_ms = int(chunk_end * 1000)
                    chunk_stems[stem_type] = audio[start_ms:end_ms]
                
                # Create training example with random augmentations
                example = self._create_augmented_example(
                    chunk_stems, 
                    metadata,
                    segment['label'],
                    chunk_start,
                    chunk_end
                )
                examples.append(example)
        
        return examples
    
    def _create_augmented_example(self, 
                                 stems: Dict[str, AudioSegment],
                                 metadata: Dict,
                                 segment_type: str,
                                 start_time: float,
                                 end_time: float) -> Dict:
        """
        Apply random mixing tools to stems and record the operations
        
        Args:
            stems: Dictionary of stem audio segments
            metadata: Track metadata
            segment_type: Type of segment (verse, chorus, etc.)
            start_time: Start time of segment in seconds
            end_time: End time of segment in seconds
            
        Returns:
            Example with augmented audio and corresponding operations
        """
        augmentation_config = self.augmentation_config
        
        # Choose a random number of augmentations to apply from config
        min_ops = augmentation_config.get('min_operations', 1)
        max_ops = augmentation_config.get('max_operations', 4)
        num_augmentations = random.randint(min_ops, max_ops)
        
        # Choose which stems to modify
        available_stems = list(stems.keys())
        stems_to_modify = random.sample(available_stems, min(num_augmentations, len(available_stems)))
        
        # Initialize the mixer tools for each stem
        stem_tools = {stem: MixingTools(stems[stem]) for stem in stems}
        
        # Track all operations with their parameters
        operations = []
        
        # Get operations config
        operations_config = augmentation_config.get('operations', {})
        
        # Apply random operations to selected stems
        for stem in stems_to_modify:
            # Choose a random operation based on probabilities
            operation_choices = []
            operation_weights = []
            
            for op_name, op_config in operations_config.items():
                operation_choices.append(op_name)
                operation_weights.append(op_config.get('probability', 0.1))
                
            # Normalize weights if they don't sum to 1
            weight_sum = sum(operation_weights)
            if weight_sum != 1.0:
                operation_weights = [w / weight_sum for w in operation_weights]
                
            operation = random.choices(operation_choices, weights=operation_weights, k=1)[0]
            
            mixer_tool = stem_tools[stem]
            op_params = operations_config.get(operation, {}).get('params', {})
            
            # Apply the operation with parameters from config
            if operation == "loop":
                # Calculate beat times that fall within our segment
                beat_times = [b for b in metadata['beats'] if start_time <= b < end_time]
                if len(beat_times) >= 4:
                    # Choose a loop length from config or default
                    loop_lengths = op_params.get('loop_lengths', [4, 8])
                    loop_length = random.choice(loop_lengths)
                    
                    if len(beat_times) >= loop_length:
                        # Choose a random starting point
                        start_idx = random.randint(0, len(beat_times) - loop_length)
                        loop_start = beat_times[start_idx] - start_time  # Convert to segment time
                        loop_end = beat_times[start_idx + loop_length - 1] - start_time
                        
                        # Choose repeat count from config or default
                        counts = op_params.get('counts', [2, 3, 4])
                        count = random.choice(counts)
                        
                        # Convert to milliseconds for pydub
                        loop_start_ms = int(loop_start * 1000)
                        loop_end_ms = int(loop_end * 1000)
                        
                        mixer_tool.loop_section(loop_start_ms, loop_end_ms, count)
                        operations.append(f"loop(stem={stem}, start={loop_start:.1f}, end={loop_end:.1f}, count={count})")
            
            elif operation == "apply_eq":
                # Get gain range from config or default
                gain_range = op_params.get('gain_range', [-6, 6])
                low_gain = random.uniform(gain_range[0], gain_range[1])
                mid_gain = random.uniform(gain_range[0], gain_range[1])
                high_gain = random.uniform(gain_range[0], gain_range[1])
                
                mixer_tool.apply_eq(low_gain, mid_gain, high_gain)
                operations.append(f"apply_eq(stem={stem}, low={low_gain:.1f}, mid={mid_gain:.1f}, high={high_gain:.1f})")
            
            elif operation == "apply_reverb":
                mixer_tool.apply_reverb()
                operations.append(f"apply_reverb(stem={stem})")
            
            elif operation == "apply_delay":
                # Get delay parameters from config or default
                delay_ms_options = op_params.get('delay_ms', [100, 150, 200, 250, 300])
                delay_ms = random.choice(delay_ms_options)
                
                decay_range = op_params.get('decay_range', [0.3, 0.7])
                decay = random.uniform(decay_range[0], decay_range[1])
                
                mixer_tool.apply_delay(delay_ms, decay)
                operations.append(f"apply_delay(stem={stem}, delay_ms={delay_ms}, decay={decay:.1f})")
            
            elif operation == "change_volume":
                # Get volume range from config or default
                db_range = op_params.get('db_range', [-3, 3])
                db = random.uniform(db_range[0], db_range[1])
                
                mixer_tool.change_volume(db)
                operations.append(f"change_volume(stem={stem}, db={db:.1f})")
            
            elif operation == "pan":
                # Get pan range from config or default
                pan_range = op_params.get('pan_range', [-0.5, 0.5])
                pan_value = random.uniform(pan_range[0], pan_range[1])
                
                mixer_tool.pan(pan_value)
                operations.append(f"pan(stem={stem}, value={pan_value:.1f})")
            
            elif operation == "apply_low_pass":
                # Get cutoff options from config or default
                cutoffs = op_params.get('cutoffs', [500, 800, 1000, 1500, 2000])
                cutoff = random.choice(cutoffs)
                
                mixer_tool.apply_low_pass(cutoff)
                operations.append(f"apply_low_pass(stem={stem}, cutoff={cutoff})")
            
            elif operation == "apply_high_pass":
                # Get cutoff options from config or default
                cutoffs = op_params.get('cutoffs', [100, 150, 200, 300, 400])
                cutoff = random.choice(cutoffs)
                
                mixer_tool.apply_high_pass(cutoff)
                operations.append(f"apply_high_pass(stem={stem}, cutoff={cutoff})")
            
            elif operation == "apply_normalize":
                mixer_tool.apply_normalize()
                operations.append(f"apply_normalize(stem={stem})")
            
            elif operation == "apply_compression":
                # Get compression parameters from config or default
                threshold_range = op_params.get('threshold_range', [-30, -10])
                threshold = random.uniform(threshold_range[0], threshold_range[1])
                
                ratio_range = op_params.get('ratio_range', [2, 6])
                ratio = random.uniform(ratio_range[0], ratio_range[1])
                
                mixer_tool.apply_compression(threshold, ratio)
                operations.append(f"apply_compression(stem={stem}, threshold={threshold:.1f}, ratio={ratio:.1f})")
        
        # Export modified stems
        modified_stems = {stem: tool.export() for stem, tool in stem_tools.items()}
        
        # Mix all stems together
        mixed = sum(list(modified_stems.values())[1:], modified_stems[available_stems[0]])
        
        # Convert to numpy array for model input
        mixed_np = np.array(mixed.get_array_of_samples()).astype(np.float32) / 32767.0
        
        # Create text description based on segment type and track info
        text_description = f"Type: {segment_type}, BPM: {metadata.get('bpm', 120)}"
        
        # Join operations with semicolons to form the target sequence
        target_sequence = "; ".join(operations)
        
        return {
            "text": text_description,
            "audio": mixed_np,
            "target": target_sequence
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text_prompt = ex['text']
        audio_waveform = torch.tensor(ex['audio']).float()
        dsp_code = ex['target']  # string like: apply_eq(...)
        
        return {
            'text': text_prompt,
            'audio': audio_waveform,
            'target': dsp_code
        }

class MixerGeneratorTools:
    """Tools for the Mixer to generate new stem content"""
    
    def __init__(self, device=None, model_dir=None, config=None):
        """
        Initialize generator tools for the mixer
        
        Args:
            device: Device to run generators on (None for auto)
            model_dir: Directory for model storage
            config: Configuration dictionary
        """
        self.device = device
        self.model_dir = model_dir
        self.config = config or load_mixer_config()
        self.generators = {}
        
        # Get generator settings from config
        self.generator_config = self.config.get('generators', {})
        
        # Lazy load - will initialize when needed
        self._melody_generator = None
        self._bass_generator = None
        self._drums_generator = None
    
    def get_melody_generator(self):
        """Get or initialize melody generator"""
        if not self._melody_generator:
            model_name = self.generator_config.get('melody', {}).get('model_name', "facebook/musicgen-melody")
            self._melody_generator = OtherGenerator(model_name=model_name, device=self.device, model_dir=self.model_dir)
        return self._melody_generator
    
    def get_bass_generator(self):
        """Get or initialize bass generator"""
        if not self._bass_generator:
            model_name = self.generator_config.get('bass', {}).get('model_name', "facebook/musicgen-small")
            self._bass_generator = BassGenerator(model_name=model_name, device=self.device, model_dir=self.model_dir)
        return self._bass_generator
    
    def get_drums_generator(self):
        """Get or initialize drums generator"""
        if not self._drums_generator:
            model_name = self.generator_config.get('drums', {}).get('model_name', "facebook/musicgen-small")
            self._drums_generator = DrumGenerator(model_name=model_name, device=self.device, model_dir=self.model_dir)
        return self._drums_generator
    
    def generate_melody(self, prompt, duration=5.0, temperature=1.0):
        """Generate melody from prompt"""
        generator = self.get_melody_generator()
        return generator.generate(prompt=prompt, duration=duration, temperature=temperature)
    
    def generate_bass(self, prompt, duration=5.0, temperature=1.0):
        """Generate bass from prompt"""
        generator = self.get_bass_generator()
        return generator.generate(prompt=prompt, duration=duration, temperature=temperature)
    
    def generate_drums(self, prompt, duration=5.0, temperature=1.0):
        """Generate drums from prompt"""
        generator = self.get_drums_generator()
        return generator.generate(prompt=prompt, duration=duration, temperature=temperature)

def collate_fn(batch):
    """
    Custom collate function to handle variable length audio
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched data with padded audio
    """
    # Get max audio length
    max_audio_len = max(x['audio'].shape[0] for x in batch)
    
    # Prepare lists for batched data
    text_list = []
    audio_list = []
    target_list = []
    
    # Process each sample
    for sample in batch:
        text_list.append(sample['text'])
        
        # Pad audio to max length
        audio = sample['audio']
        audio_len = audio.shape[0]
        if audio_len < max_audio_len:
            padding = torch.zeros(max_audio_len - audio_len, dtype=audio.dtype)
            audio = torch.cat([audio, padding], dim=0)
        audio_list.append(audio)
        
        target_list.append(sample['target'])
    
    # Stack audio tensors
    audio_batch = torch.stack(audio_list, dim=0)
    
    return {
        'text': text_list,
        'audio': audio_batch,
        'target': target_list
    }

def train(model, 
         dataloader, 
         optimizer, 
         scheduler, 
         device, 
         epochs, 
         save_path,
         grad_accum_steps=4,  # Add gradient accumulation steps
         log_interval=10):
    """
    Train the mixer model
    
    Args:
        model: Mixer model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epochs: Number of epochs to train for
        save_path: Path to save model checkpoints
        grad_accum_steps: Number of gradient accumulation steps
        log_interval: How often to log training progress
    """
    model.text_encoder.train()
    model.fusion_proj.train()
    model.decoder.train()
    
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Use CrossEntropyLoss with ignore_index set to pad_token_id
    loss_fn = nn.CrossEntropyLoss(ignore_index=model.decoder_tokenizer.pad_token_id)

    # Track best validation loss
    best_loss = float('inf')
    
    # Create a wrapper for the forward method
    def forward_wrapper(model, text, audio, target):
        """
        Wrapper for the model's forward method to handle the inputs correctly.
        Returns logits and labels for computing loss.
        """
        # Base device for non-distributed components
        primary_device = audio.device
        vocab_size = model.decoder.config.vocab_size
        
        # Encode text
        text_inputs = model.text_tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(primary_device)
        text_emb = model.text_encoder(**text_inputs).last_hidden_state[:, 0]  # [batch, dim]

        # Encode audio
        audio_emb = model.get_audio_embedding(audio)
        if audio_emb.device != primary_device:
            audio_emb = audio_emb.to(primary_device)

        # Combine and project
        fused = torch.cat([text_emb, audio_emb], dim=-1)
        mistral_cond_emb = model.fusion_proj(fused).unsqueeze(1)  # [batch, 1, mistral_hidden_dim]

        # Tokenize targets
        target_inputs = model.decoder_tokenizer(target, return_tensors='pt', padding=True, truncation=True).to(primary_device)
        input_ids = target_inputs.input_ids
        attention_mask = target_inputs.attention_mask
        
        # Check if there are any token ids outside the vocab range
        if (input_ids >= vocab_size).any():
            # Clamp input_ids to be within vocabulary range
            input_ids = torch.clamp(input_ids, 0, vocab_size-1)
        
        # Forward pass through Mistral
        try:
            # For distributed models, device mapping is handled internally
            # Set up labels for causal language modeling
            labels = input_ids.clone()
            
            # Move tensors to model devices as needed (handled automatically)
            outputs = model.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                encoder_hidden_states=mistral_cond_emb,
                encoder_attention_mask=torch.ones(mistral_cond_emb.shape[:2], dtype=torch.bool).to(primary_device)
            )
            return outputs.logits, outputs.loss
        except RuntimeError as e:
            logger.error(f"Runtime error in forward pass: {str(e)}")
            raise e
    
    for epoch in range(epochs):
        total_loss = 0
        
        progress_bar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
        
        # Track steps for gradient accumulation
        optimizer.zero_grad()
        
        for step, batch in progress_bar:
            # Get batch data
            text = batch['text']
            audio = batch['audio'].to(device)
            target = batch['target']

            # Forward pass using our wrapper
            logits, loss = forward_wrapper(model, text, audio, target)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            # Only update weights after accumulating gradients
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update metrics (use the original non-scaled loss for display)
            batch_loss = loss.item() * grad_accum_steps
            total_loss += batch_loss
            
            # Update progress bar
            progress_bar.set_postfix(loss=batch_loss, avg_loss=total_loss/(step+1))
            
            # Log training progress
            if step % log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Step {step}/{len(dataloader)}, Loss: {batch_loss:.4f}")
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_path, f"mixer_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(save_path, "mixer_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, best_model_path)
            logger.info(f"New best model saved with loss: {avg_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train the mixer model")
    
    parser.add_argument("--data_root", type=str, default="data", 
                       help="Root directory containing artist data")
    parser.add_argument("--save_path", type=str, default=None,
                       help="Directory to save model checkpoints (default: from config)")
    parser.add_argument("--artists", nargs='+', default=None,
                       help="List of artists to include (default: all)")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to custom config file (default: mixer_config.yaml)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size for training (default: from config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (default: from config)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs to train for (default: from config)")
    parser.add_argument("--device", type=str, default='auto',
                       help="Device to train on (default: auto for distributed)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_mixer_config(args.config)
    default_config = config.get('default', {})
    
    # Get root directory for relative paths
    root_dir = os.getcwd()
    
    # Set up distributed training if using multiple GPUs
    if args.device == 'auto':
        device = 'cuda:0'  # Primary device - only used for non-model components
        logger.info(f"Using device: {device} with model distributed across available GPUs")
    else:
        device = args.device
        logger.info(f"Using single device: {device}")
    
    # Get model and data parameters from config
    mixer_config = get_mixer_config(config)
    data_config = get_mixer_data_config(config)
    
    # Get path for saving models (from args or config)
    if args.save_path is None:
        save_path = get_models_dir(root_dir, config)
    else:
        save_path = args.save_path
        if not os.path.isabs(save_path):
            save_path = os.path.join(root_dir, save_path)
    
    # Get training parameters (from args or config)
    batch_size = args.batch_size or default_config.get('batch_size', 4)
    learning_rate = args.lr or default_config.get('learning_rate', 5e-5)
    num_epochs = args.epochs or default_config.get('num_epochs', 5)
    grad_accum_steps = default_config.get('gradient_accumulation_steps', 4)
    
    # Initialize tokenizer with special tokens for mixer operations
    mistral_model = mixer_config.get('mistral_model', 'mistralai/Mistral-7B-v0.3')
    tokenizer = AutoTokenizer.from_pretrained(mistral_model)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    special_tokens = mixer_config.get('special_tokens', [])
    if special_tokens:
        tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
    
    # Initialize model
    logger.info("Initializing Mixer model...")
    text_model = mixer_config.get('text_model', 'bert-base-uncased')
    
    # Initialize model with device_map="auto" to distribute across GPUs
    model = Mixer(
        decoder_model_name=mistral_model,
        text_model_name=text_model
    )
    
    # The Mistral model is already distributed across GPUs
    # Move only the non-distributed components to the primary device
    model.text_encoder.to(device)
    model.audio_encoder.to(device)
    model.fusion_proj.to(device)
    model.decoder_tokenizer = tokenizer
    
    # Create dataset and dataloader
    logger.info("Creating dataset...")
    dataset = MixerDataset(
        root_dir=args.data_root,
        artists=args.artists,
        config=config,
        mistral_tokenizer=tokenizer
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Setup optimizer and scheduler
    weight_decay = default_config.get('weight_decay', 0.01)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    total_steps = len(dataloader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Train the model
    logger.info("Starting training...")
    logger.info(f"Training configuration: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    logger.info(f"Model will be saved to {save_path}")
    
    train(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=num_epochs,
        save_path=save_path,
        grad_accum_steps=grad_accum_steps
    )
    
    logger.info("Training complete!")
    
    # Save the final model
    final_model_path = os.path.join(save_path, "mixer_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    main()