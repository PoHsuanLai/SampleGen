# train_mixer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup, AutoTokenizer, get_constant_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
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
from .mixer_dataset import MixerDataset
import glob
import os
import json
import random
import numpy as np
from pydub import AudioSegment
import tqdm
from tqdm import tqdm
import logging
import librosa
import argparse
import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import types
import traceback
import gc  # For garbage collection

# Set TOKENIZERS_PARALLELISM environment variable to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import AudioDistorter from data_processing
from ..data_processing.audio_distorter import AudioDistorter
from ..data_processing.create_dataset import scan_processed_directory

# Set up logging
logging.basicConfig(level=logging.INFO, 
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MixerTrainer:
    def __init__(self, config, data_dir):
        """
        Initialize the MixerTrainer.
        
        Args:
            config: Configuration dictionary loaded from mixer_config.yaml
            data_dir: Directory containing raw audio data
        """
        self.config = config
        self.data_dir = data_dir
        
        # Get command line arguments
        parser = argparse.ArgumentParser(description='Process data and train a mixer model')
        args, _ = parser.parse_known_args()
        self.artist = args.artist if hasattr(args, 'artist') else None
        self.limit = args.limit if hasattr(args, 'limit') else None
        
        # Extract optional parameters if they are in the config
        default_config = config.get("default", {})
        self.num_epochs = args.num_epochs if (hasattr(args, 'num_epochs') and args.num_epochs is not None) else default_config.get("num_epochs")
        
        # Mixed precision training flag
        self.mixed_precision = default_config.get("mixed_precision", True)
        
        # Set up parameters from config
        self.set_parameters(self.config)
        
        # Filter data_dir to specific artist if provided
        if self.artist:
            self.data_dir = os.path.join(self.data_dir, self.artist)
            logger.info(f"Focusing on artist directory: {self.data_dir}")

    def set_parameters(self, config):
        # Get configuration sections
        self.mixer_config = get_mixer_config(config)
        self.data_config = get_mixer_data_config(config)
        self.augmentation_config = get_mixer_augmentation_config(config)
        self.default_config = config.get("default", {})
        self.models_dir = get_models_dir(config)
        
        # Extract training parameters
        if self.num_epochs is None:
            self.num_epochs = self.default_config.get("num_epochs", 3)
        self.batch_size = self.default_config.get("batch_size", 4)
        self.learning_rate = self.default_config.get("learning_rate", 5.0e-5)
        self.warmup_ratio = self.default_config.get("warmup_ratio", 0.10)
        self.weight_decay = self.default_config.get("weight_decay", 0.01)
        self.logging_steps = self.default_config.get("logging_steps", 100)
        self.save_strategy = self.default_config.get("save_strategy", "epoch")
        self.save_steps = self.default_config.get("save_steps", 1000)
        self.save_total_limit = self.default_config.get("save_total_limit", 2)
        self.eval_steps = self.default_config.get("eval_steps", 500)
        self.evaluation_strategy = self.default_config.get("evaluation_strategy", "steps")
        self.lr_patience = self.default_config.get("lr_patience", 2)
        self.lr_factor = self.default_config.get("lr_factor", 0.5)
        
        # Extract data parameters
        self.segment_duration = self.data_config.get("segment_duration", 4.0)
        self.max_seq_len = self.data_config.get("max_seq_len", 128)
        self.sample_rate = self.data_config.get("sample_rate", 48000)  # Use 48000 Hz to match CLAP model
        self.validation_split = self.data_config.get("validation_split", 0.05)
        self.max_train_samples = self.data_config.get("max_train_samples", 2000)
        
        # Extract model parameters
        self.mistral_model_name = self.mixer_config.get("mistral_model", "mistralai/Mistral-7B-v0.1")
        self.text_model_name = self.mixer_config.get("text_model", "bert-base-uncased")
        
        # Extract LoRA parameters
        self.lora_config = self.mixer_config.get("lora", {})
        self.lora_r = self.lora_config.get("r", 16)
        self.lora_alpha = self.lora_config.get("lora_alpha", 32)
        self.lora_dropout = self.lora_config.get("lora_dropout", 0.05)
        self.lora_bias = self.lora_config.get("bias", "none")
        self.lora_task_type = self.lora_config.get("task_type", "CAUSAL_LM")
        self.lora_target_modules = self.lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ])
        
        # Extract special tokens
        self.special_tokens = self.mixer_config.get("special_tokens", [])
        
        # Set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

    def create_dataset(self):
        """
        Create a dataset directly from the data directory using on-the-fly chunking.
        """
        logger.info("Setting up dataset creation...")

        # Initialize the tokenizer for text input
        logger.info(f"Initializing tokenizer from {self.text_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        
        # Add special tokens for mixer operations
        if self.special_tokens:
            logger.info(f"Adding {len(self.special_tokens)} special tokens to the tokenizer")
            special_tokens_dict = {"additional_special_tokens": self.special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # Create dataset using the modified MixerDataset with on-the-fly chunking
        logger.info(f"Creating MixerDataset from {self.data_dir}")
        dataset = MixerDataset(
            processed_dir=self.data_dir,  # Now processed_dir is used as the main data directory
            tokenizer=self.tokenizer,
            config=self.config,
            dataset_file=None,  # Don't use a dataset file for on-the-fly chunking
            segment_duration=self.segment_duration,
            max_seq_len=self.max_seq_len,
            sample_rate=self.sample_rate,
            num_distortions=self.augmentation_config.get("min_operations", 2),
            train_mode=True
        )
        
        # Split into train and validation sets
        dataset_size = len(dataset)
        val_size = max(1, int(dataset_size * self.validation_split))
        train_size = dataset_size - val_size
        
        # Apply max_train_samples limit if specified
        if self.max_train_samples and train_size > self.max_train_samples:
            train_size = self.max_train_samples
            # We need to make sure train_size + val_size equals the subset we're actually using
            dataset_size = train_size + val_size
            
            # Create a subset of the full dataset first
            indices = torch.randperm(len(dataset)).tolist()
            subset_indices = indices[:dataset_size]
            dataset = torch.utils.data.Subset(dataset, subset_indices)
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        logger.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
        
        return train_dataset, val_dataset

    def train_mixer(self):
        """
        Training logic for the mixer model using LoRA fine-tuning.
        """
        try:
            # Create model directories if they don't exist
            os.makedirs(self.models_dir, exist_ok=True)
            logger.info(f"Models will be saved to {self.models_dir}")
            
            # Get the train and validation datasets
            train_dataset, val_dataset = self.create_dataset()
            
            # Create data loaders with the custom collate function
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True,
                collate_fn=default_collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=False,
                collate_fn=default_collate_fn
            )
            
            # Initialize the mixer model
            logger.info(f"Initializing mixer model with {self.mistral_model_name} and {self.text_model_name}")
            model = Mixer(
                decoder_model_name=self.mistral_model_name,
                text_model_name=self.text_model_name
            )
            
            # Set up LoRA configuration
            logger.info("Setting up LoRA fine-tuning configuration")
            peft_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                task_type=self.lora_task_type,
                target_modules=self.lora_target_modules,
                modules_to_save=None,
                inference_mode=False,
            )
            
            # Apply LoRA to the decoder part of the model
            logger.info("Applying LoRA to the model")
            
            # Explicitly freeze the base model and enable gradient only for LoRA layers
            for param in model.decoder.parameters():
                param.requires_grad = False
            
            # Free up memory before loading LoRA
            gc.collect()
            torch.cuda.empty_cache()
            
            # Now apply LoRA 
            model.decoder = get_peft_model(model.decoder, peft_config)

            # Ensure all encoder components are in evaluation mode and decoder in training mode
            model.text_encoder.eval()
            model.audio_encoder.eval()
            model.clap_model.eval()
            
            # Ensure fusion projections and token embedder are trainable
            for param in model.fusion_proj.parameters():
                param.requires_grad = True
            for param in model.stem_token_embedder.parameters():
                param.requires_grad = True
            
            # Add debug information to check trainable parameters
            logger.info("Checking trainable parameters after applying LoRA")
            trainable_params = 0
            all_param = 0
            for name, param in model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    logger.info(f"Trainable parameter: {name}, shape: {param.shape}")
            
            logger.info(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )
            
            # Move model to device
            device = self.device
            model = model.to(device)
            
            # Set up optimizer and scheduler
            logger.info("Setting up optimizer and scheduler")
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], 
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Calculate total training steps
            total_steps = len(train_loader) * self.num_epochs
            warmup_steps = int(total_steps * self.warmup_ratio)
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
            # Training loop
            logger.info("Starting training...")
            best_val_loss = float('inf')
            patience_counter = 0
            progress_bar = tqdm(range(total_steps), desc="Training")
            
            # Set up scaler for mixed precision training
            scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
            
            # Training loop
            for epoch in range(self.num_epochs):
                model.train()
                train_loss = 0.0
                
                for batch_idx, batch in enumerate(train_loader):
                    try:
                        # Skip empty batches (can happen with our custom collate_fn)
                        if batch.get("distorted_audio") is None or len(batch["distorted_audio"]) == 0:
                            logger.warning(f"Skipping empty batch in epoch {epoch}, batch {batch_idx}")
                            continue
                        
                        # Move batch to device
                        distorted_audio = batch["distorted_audio"].to(device)
                        input_ids = batch["input_ids"].to(device) if "input_ids" in batch else None
                        attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
                        tool_input_ids = batch["tool_input_ids"].to(device) if "tool_input_ids" in batch else None
                        tool_attention_mask = batch["tool_attention_mask"].to(device) if "tool_attention_mask" in batch else None
                        token_identifiers = batch.get("token_identifiers", None)  # This stays as a list
                        
                        # Clear gradients
                        optimizer.zero_grad()
                        
                        # Forward pass with mixed precision if enabled
                        if self.mixed_precision:
                            with torch.cuda.amp.autocast():
                                outputs = model(
                                    text_prompt=input_ids,
                                    audio_waveform=distorted_audio,
                                    token_identifiers=token_identifiers,
                                    target_tool_tokens=None,
                                    attention_mask=attention_mask,
                                    tool_input_ids=tool_input_ids,
                                    tool_attention_mask=tool_attention_mask,
                                    labels=tool_input_ids  # Use the tool tokens as labels for the decoder
                                )
                                loss = outputs.loss
                        else:
                            outputs = model(
                                text_prompt=input_ids,
                                audio_waveform=distorted_audio,
                                token_identifiers=token_identifiers,
                                target_tool_tokens=None,
                                attention_mask=attention_mask,
                                tool_input_ids=tool_input_ids,
                                tool_attention_mask=tool_attention_mask,
                                labels=tool_input_ids  # Use the tool tokens as labels for the decoder
                            )
                            loss = outputs.loss
                        
                        # Backward pass with mixed precision if enabled
                        if self.mixed_precision:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                        
                        # Update learning rate
                        scheduler.step()
                        
                        # Update progress
                        train_loss += loss.item()
                        progress_bar.update(1)
                        progress_bar.set_postfix({
                            "epoch": epoch + 1,
                            "train_loss": train_loss / (batch_idx + 1),
                            "lr": scheduler.get_last_lr()[0]
                        })
                        
                        # Log training metrics every log_every steps
                        if (batch_idx + 1) % self.logging_steps == 0:
                            logger.info(
                                f"Epoch: {epoch + 1}/{self.num_epochs}, "
                                f"Batch: {batch_idx + 1}/{len(train_loader)}, "
                                f"Loss: {train_loss / (batch_idx + 1):.4f}, "
                                f"LR: {scheduler.get_last_lr()[0]:.6f}"
                            )
                            
                        # Periodically clean up memory
                        if batch_idx % 10 == 0:
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {str(e)}")
                        logger.error(traceback.format_exc())
                        continue
                
                # Validation
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        try:
                            # Skip empty batches
                            if batch.get("distorted_audio") is None or len(batch["distorted_audio"]) == 0:
                                logger.warning(f"Skipping empty validation batch in epoch {epoch}, batch {batch_idx}")
                                continue
                            
                            # Move batch to device
                            distorted_audio = batch["distorted_audio"].to(device)
                            input_ids = batch["input_ids"].to(device) if "input_ids" in batch else None
                            attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
                            tool_input_ids = batch["tool_input_ids"].to(device) if "tool_input_ids" in batch else None
                            tool_attention_mask = batch["tool_attention_mask"].to(device) if "tool_attention_mask" in batch else None
                            token_identifiers = batch.get("token_identifiers", None)  # This stays as a list
                            
                            # Forward pass with mixed precision if enabled
                            if self.mixed_precision:
                                with torch.cuda.amp.autocast():
                                    outputs = model(
                                        text_prompt=input_ids,
                                        audio_waveform=distorted_audio,
                                        token_identifiers=token_identifiers,
                                        target_tool_tokens=None,
                                        attention_mask=attention_mask,
                                        tool_input_ids=tool_input_ids,
                                        tool_attention_mask=tool_attention_mask,
                                        labels=tool_input_ids  # Use the tool tokens as labels for the decoder
                                    )
                                    loss = outputs.loss
                            else:
                                outputs = model(
                                    text_prompt=input_ids,
                                    audio_waveform=distorted_audio,
                                    token_identifiers=token_identifiers,
                                    target_tool_tokens=None,
                                    attention_mask=attention_mask,
                                    tool_input_ids=tool_input_ids,
                                    tool_attention_mask=tool_attention_mask,
                                    labels=tool_input_ids  # Use the tool tokens as labels for the decoder
                                )
                                loss = outputs.loss
                            
                            val_loss += loss.item()
                        except Exception as e:
                            logger.error(f"Error in validation batch {batch_idx}: {str(e)}")
                            logger.error(traceback.format_exc())
                            continue
                
                # Calculate average validation loss
                avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
                
                # Log validation metrics
                logger.info(
                    f"Epoch: {epoch + 1}/{self.num_epochs}, "
                    f"Train Loss: {train_loss / len(train_loader):.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}"
                )
                
                # Save model checkpoint if it's better than previous best
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # Save the model checkpoint
                    checkpoint_path = os.path.join(self.models_dir, f"checkpoint-epoch{epoch+1}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    
                    try:
                        # Save the decoder LoRA weights
                        model.decoder.save_pretrained(checkpoint_path)
                        
                        # Save the tokenizer
                        self.tokenizer.save_pretrained(checkpoint_path)
                        
                        # Save stem token embeddings
                        torch.save(model.stem_token_embedder.state_dict(), os.path.join(checkpoint_path, "stem_token_embedder.pt"))
                        torch.save(model.stem_token_dict, os.path.join(checkpoint_path, "stem_token_dict.pt"))
                        
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    except Exception as e:
                        logger.error(f"Error saving checkpoint: {str(e)}")
                        logger.error(traceback.format_exc())
                else:
                    patience_counter += 1
                    
                    # Check for early stopping
                    if patience_counter >= self.lr_patience:
                        # Reduce learning rate after patience is exhausted
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= self.lr_factor
                        logger.info(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
                        patience_counter = 0
                
                # Clean up memory after each epoch
                gc.collect()
                torch.cuda.empty_cache()
            
            # Save the final model
            final_model_path = os.path.join(self.models_dir, "final-model")
            os.makedirs(final_model_path, exist_ok=True)
            
            try:
                # Save the decoder LoRA weights
                model.decoder.save_pretrained(final_model_path)
                
                # Save the tokenizer
                self.tokenizer.save_pretrained(final_model_path)
                
                # Save stem token embeddings
                torch.save(model.stem_token_embedder.state_dict(), os.path.join(final_model_path, "stem_token_embedder.pt"))
                torch.save(model.stem_token_dict, os.path.join(final_model_path, "stem_token_dict.pt"))
                
                logger.info(f"Training complete. Final model saved to {final_model_path}")
            except Exception as e:
                logger.error(f"Error saving final model: {str(e)}")
                logger.error(traceback.format_exc())
            
            return model
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def evaluate_model(self, model, val_loader, device):
        """
        Evaluate the model on the validation set.
        
        Args:
            model: The mixer model to evaluate
            val_loader: DataLoader for the validation dataset
            device: Device to run evaluation on
        
        Returns:
            Average validation loss
        """
        # Set model to evaluation mode
        model.eval()
        
        # Make sure encoders are in eval mode
        model.text_encoder.eval()
        model.audio_encoder.eval()
        if hasattr(model, 'clap_model'):
            model.clap_model.eval()
        
        val_losses = []
        
        with torch.no_grad():  # Ensure no gradients are computed during evaluation
            for batch in tqdm(val_loader, desc="Evaluation"):
                # Skip empty batches
                if batch.get("distorted_audio") is None or len(batch["distorted_audio"]) == 0:
                    continue
                
                # Move inputs to device
                distorted_audio = batch["distorted_audio"].to(device)
                input_ids = batch["input_ids"].to(device) if "input_ids" in batch else None
                attention_mask = batch["attention_mask"].to(device) if "attention_mask" in batch else None
                tool_input_ids = batch["tool_input_ids"].to(device) if "tool_input_ids" in batch else None
                tool_attention_mask = batch["tool_attention_mask"].to(device) if "tool_attention_mask" in batch else None
                
                # Get token identifiers (these are already lists of strings)
                token_identifiers = batch.get("token_identifiers", None)
                
                # Check for NaN or Inf values and handle them
                if (torch.isnan(distorted_audio).any() or torch.isinf(distorted_audio).any() or
                    torch.isnan(input_ids).any() or torch.isinf(input_ids).any() or
                    (tool_input_ids is not None and (torch.isnan(tool_input_ids).any() or torch.isinf(tool_input_ids).any()))):
                    
                    # Replace NaN/Inf values with zeros
                    distorted_audio = torch.nan_to_num(distorted_audio)
                    input_ids = torch.nan_to_num(input_ids).long() if input_ids is not None else None
                    tool_input_ids = torch.nan_to_num(tool_input_ids).long() if tool_input_ids is not None else None
                
                # Forward pass
                outputs = model(
                    text_prompt=input_ids,
                    audio_waveform=distorted_audio,
                    token_identifiers=token_identifiers,
                    target_tool_tokens=None,
                    attention_mask=attention_mask,
                    tool_input_ids=tool_input_ids,
                    tool_attention_mask=tool_attention_mask,
                    labels=tool_input_ids  # Use the tool tokens as labels for causal LM
                )
                loss = outputs.loss
                val_losses.append(loss.item())
        
        # Calculate average validation loss
        if val_losses:
            avg_val_loss = sum(val_losses) / len(val_losses)
        else:
            # Return a high loss if no valid batches were processed
            avg_val_loss = 9999.0
            logger.warning("No valid losses were recorded during evaluation.")
        
        return avg_val_loss

def default_collate_fn(batch):
    """
    Custom collate function to handle batches with potentially problematic samples.
    
    Args:
        batch: List of samples from the dataset
    
    Returns:
        Collated batch with tensors and metadata
    """
    # Filter out None samples if present
    valid_samples = [sample for sample in batch if sample is not None]
    
    if not valid_samples:
        # Return empty default tensors if no valid samples
        return {
            "distorted_audio": torch.empty(0),
            "input_ids": torch.empty(0, dtype=torch.long),
            "attention_mask": torch.empty(0, dtype=torch.long),
            "tool_input_ids": torch.empty(0, dtype=torch.long),
            "tool_attention_mask": torch.empty(0, dtype=torch.long),
            "token_identifiers": []
        }

    # Create separate lists for each element
    distorted_audio = []
    input_ids = []
    attention_mask = []
    tool_input_ids = []
    tool_attention_mask = []
    token_identifiers = []

    for sample in valid_samples:
        try:
            distorted_audio.append(sample["distorted_audio"])
            input_ids.append(sample["input_ids"])
            attention_mask.append(sample["attention_mask"])
            tool_input_ids.append(sample["tool_input_ids"])
            tool_attention_mask.append(sample["tool_attention_mask"])
            token_identifiers.append(sample["token_identifiers"])
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Skipping problematic sample: {e}")
            continue

    # Stack tensors only if we have valid entries
    if distorted_audio:
        batch_dict = {
            "distorted_audio": torch.stack(distorted_audio),
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "tool_input_ids": torch.stack(tool_input_ids),
            "tool_attention_mask": torch.stack(tool_attention_mask),
            "token_identifiers": token_identifiers  # This remains a list of lists of strings
        }
        return batch_dict
    else:
        # Return empty default tensors if all samples were invalid
        return {
            "distorted_audio": torch.empty(0),
            "input_ids": torch.empty(0, dtype=torch.long),
            "attention_mask": torch.empty(0, dtype=torch.long),
            "tool_input_ids": torch.empty(0, dtype=torch.long),
            "tool_attention_mask": torch.empty(0, dtype=torch.long),
            "token_identifiers": []
        }

def main():
    parser = argparse.ArgumentParser(description='Process data and train a mixer model')
    parser.add_argument('--config', type=str, default='src/tuning/config/mixer_config.yaml', 
                        help='Path to the mixer configuration file')
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='Directory containing raw audio data')
    parser.add_argument('--artist', type=str, default=None,
                        help='Process a specific artist only (optional)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of songs to process (optional)')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override the number of epochs in the config (optional)')
    args = parser.parse_args()
    
    # Load the configuration
    config = load_mixer_config(args.config)
    
    # Create the trainer with all arguments
    trainer = MixerTrainer(config, args.data_dir)
    
    # Set command line arguments explicitly
    trainer.artist = args.artist
    trainer.limit = args.limit
    if args.num_epochs is not None:
        trainer.num_epochs = args.num_epochs
    
    # Train the mixer
    trainer.train_mixer()

if __name__ == "__main__":
    main()