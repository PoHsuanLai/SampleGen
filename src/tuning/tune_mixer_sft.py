# train_mixer.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
        parser.add_argument('--artist', type=str, help='Process a specific artist only (optional)')
        parser.add_argument('--limit', type=int, help='Limit the number of songs to process (optional)')
        parser.add_argument('--num_epochs', type=int, help='Override the number of epochs in the config (optional)')
        parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
        parser.add_argument('--gradient_accumulation_steps', type=int, help='Number of batches to accumulate gradients (default: 1)')
        parser.add_argument('--embedding_scale', type=float, help='Scaling factor for embeddings (default: 2.0)')
        parser.add_argument('--disable_mixed_precision_epochs', type=int, help='Number of initial epochs to disable mixed precision (default: 1)')
        parser.add_argument('--fp16', action='store_true', help='Use fp16 mixed precision training')
        args, _ = parser.parse_known_args()
        
        self.artist = args.artist if hasattr(args, 'artist') else None
        self.limit = args.limit if hasattr(args, 'limit') else None
        self.multi_gpu = args.multi_gpu if hasattr(args, 'multi_gpu') else False
        self.gradient_accumulation_steps = args.gradient_accumulation_steps if hasattr(args, 'gradient_accumulation_steps') and args.gradient_accumulation_steps is not None else 1
        
        # Set the embedding scale factor - default to a more conservative value
        self.embedding_scale = args.embedding_scale if hasattr(args, 'embedding_scale') and args.embedding_scale is not None else 2.0
        
        # Number of initial epochs to disable mixed precision
        self.disable_mixed_precision_epochs = args.disable_mixed_precision_epochs if hasattr(args, 'disable_mixed_precision_epochs') and args.disable_mixed_precision_epochs is not None else 1
        
        # Extract optional parameters if they are in the config
        default_config = config.get("default", {})
        self.num_epochs = args.num_epochs if (hasattr(args, 'num_epochs') and args.num_epochs is not None) else default_config.get("num_epochs")
        
        # Mixed precision training flag - use args or config, defaults to True
        self.mixed_precision = args.fp16 if hasattr(args, 'fp16') else default_config.get("mixed_precision", True)
        
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
        # Increase warmup ratio for better stability
        self.warmup_ratio = self.default_config.get("warmup_ratio", 0.15)
        self.weight_decay = self.default_config.get("weight_decay", 0.01)
        self.logging_steps = self.default_config.get("logging_steps", 100)
        self.save_strategy = self.default_config.get("save_strategy", "epoch")
        self.save_steps = self.default_config.get("save_steps", 1000)
        self.save_total_limit = self.default_config.get("save_total_limit", 2)
        self.eval_steps = self.default_config.get("eval_steps", 500)
        self.evaluation_strategy = self.default_config.get("evaluation_strategy", "steps")
        self.lr_patience = self.default_config.get("lr_patience", 2)
        self.lr_factor = self.default_config.get("lr_factor", 0.5)
        
        # Add gradient clipping parameters
        self.max_grad_norm = self.default_config.get("max_grad_norm", 1.0)
        self.max_grad_value = self.default_config.get("max_grad_value", 1.0)

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
        
        # Set devices for multi-GPU setup
        if self.multi_gpu and torch.cuda.device_count() > 1:
            logger.info(f"Using multi-GPU setup with {torch.cuda.device_count()} GPUs")
            self.encoder_device = torch.device("cuda:0")
            self.decoder_device = torch.device("cuda:1")
            self.main_device = self.encoder_device
        else:
            logger.info("Using single GPU setup")
            self.encoder_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.decoder_device = self.encoder_device
            self.main_device = self.encoder_device
            
        logger.info(f"Encoder device: {self.encoder_device}, Decoder device: {self.decoder_device}")

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
            
            # Initialize model
            logger.info("Creating mixer model...")
            logger.info(f"Using Mistral model: {self.mistral_model_name}")
            logger.info(f"Using text model: {self.text_model_name}")
            
            # Enable fp16 model creation to ensure all components are created in half precision
            # This helps avoid dtype mismatches later
            model_dtype = torch.float16 if self.mixed_precision else torch.float32
            logger.info(f"Creating model with dtype: {model_dtype}")
            
            # Create model with fp16 support
            model = Mixer(
                decoder_model_name=self.mistral_model_name,
                text_model_name=self.text_model_name,
                multi_gpu=self.multi_gpu
            )
            
            # Make sure model device attributes match trainer's configuration
            logger.info("Aligning model device configuration with trainer settings...")
            logger.info(f"Trainer devices: encoder={self.encoder_device}, decoder={self.decoder_device}")
            logger.info(f"Model devices: encoder={model.encoder_device}, decoder={model.decoder_device}")
            
            # Explicitly convert fusion projection to fp16 if using mixed precision
            if self.mixed_precision:
                logger.info("Converting fusion projection to fp16")
                model.fusion_proj = model.fusion_proj.half()
                model.stem_token_embedder = model.stem_token_embedder.half()
            
            # Configure LoRA
            logger.info("Configuring LoRA adapter...")
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                task_type=self.lora_task_type,
                # Add dtype parameter to ensure LoRA weights are in correct format
                modules_to_save=None,  # We're not saving non-LoRA modules
                init_lora_weights="gaussian"  # More stable initialization
            )
            
            # Apply LoRA to the model's decoder
            try:
                # First get the decoder from the model
                decoder = model.decoder
                
                # Create LoRA wrapper around the decoder
                logger.info("Applying LoRA to decoder...")
                model.decoder = get_peft_model(decoder, lora_config)
                logger.info("LoRA applied successfully")
                
                # Make sure LoRA adapter uses the same dtype as the base model
                if self.mixed_precision:
                    for name, module in model.decoder.named_modules():
                        if "lora_" in name:
                            module.to(dtype=torch.float16)
                            
            except Exception as e:
                logger.error(f"Error applying LoRA: {str(e)}")
                raise
            
            # Set up the optimizer (AdamW typically works well with LoRA)
            logger.info("Setting up optimizer...")
            optimizer = optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=self.learning_rate, 
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8  # Increased epsilon for numerical stability with fp16
            )
            
            # Calculate the number of training steps
            epochs = self.num_epochs
            num_update_steps_per_epoch = len(train_loader) // self.gradient_accumulation_steps
            max_train_steps = epochs * num_update_steps_per_epoch
            
            # Calculate warmup steps (as ratio of total steps)
            warmup_steps = int(max_train_steps * self.warmup_ratio)
            logger.info(f"Training for {max_train_steps} steps with {warmup_steps} warmup steps")
            
            # Set up learning rate scheduler
            logger.info("Setting up learning rate scheduler...")
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_train_steps
            )
            
            # Calculate the total number of steps for progress bar
            total_steps = max_train_steps
            
            # Training loop
            logger.info("Starting training...")
            best_val_loss = float('inf')
            patience_counter = 0
            progress_bar = tqdm(range(total_steps), desc="Training")
            
            # Set up scaler for mixed precision training with improved settings
            scaler = torch.cuda.amp.GradScaler(
                enabled=self.mixed_precision,  # Enable only if mixed precision is requested
                init_scale=2**10,  # Start with a smaller scale
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000)
            
            # Log the embedding scale being used
            logger.info(f"Using embedding scale factor: {self.embedding_scale}")
            logger.info(f"Mixed precision training enabled: {self.mixed_precision}")
            
            # Training loop
            for epoch in range(self.num_epochs):
                model.train()
                train_loss = 0.0
                accumulated_batches = 0
                
                # Determine if we should use mixed precision for this epoch
                use_mixed_precision = self.mixed_precision and epoch >= self.disable_mixed_precision_epochs
                if not use_mixed_precision:
                    logger.info(f"Epoch {epoch+1}: Running in full precision mode for stability")
                else:
                    logger.info(f"Epoch {epoch+1}: Running with mixed precision")
                
                for batch_idx, batch in enumerate(train_loader):
                    try:
                        # Skip empty batches (can happen with our custom collate_fn)
                        if batch.get("distorted_audio") is None or len(batch["distorted_audio"]) == 0:
                            logger.warning(f"Skipping empty batch in epoch {epoch}, batch {batch_idx}")
                            continue
                        
                        # Move batch to encoder device (data lives on GPU 0)
                        encoder_device = model.encoder_device
                        distorted_audio = batch["distorted_audio"].to(encoder_device)
                        input_ids = batch["input_ids"].to(encoder_device) if "input_ids" in batch else None
                        attention_mask = batch["attention_mask"].to(encoder_device) if "attention_mask" in batch else None
                        tool_input_ids = batch["tool_input_ids"].to(encoder_device) if "tool_input_ids" in batch else None
                        tool_attention_mask = batch["tool_attention_mask"].to(encoder_device) if "tool_attention_mask" in batch else None
                        token_identifiers = batch.get("token_identifiers", None)  # This stays as a list
                        
                        # If using mixed precision, convert input tensors to fp16
                        if use_mixed_precision:
                            if distorted_audio is not None:
                                distorted_audio = distorted_audio.half()
                        
                        # Don't clear gradients yet - will do after accumulation steps
                        
                        # Forward pass with mixed precision if enabled for this epoch
                        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                            outputs = model(
                                text_prompt=input_ids,
                                audio_waveform=distorted_audio,
                                token_identifiers=token_identifiers,
                                target_tool_tokens=None,
                                attention_mask=attention_mask,
                                tool_input_ids=tool_input_ids,
                                tool_attention_mask=tool_attention_mask,
                                labels=tool_input_ids,  # Use the tool tokens as labels for the decoder
                                embedding_scale=self.embedding_scale  # Use configured embedding scale
                            )
                            loss = outputs.loss
                            
                        # Normalize loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                            
                        # Move loss to encoder device for backward pass
                        if loss.device != encoder_device:
                            loss = loss.to(encoder_device)
                        
                        # Check for NaN loss and skip the batch if found
                        if torch.isnan(loss).any() or torch.isinf(loss).any():
                            logger.warning(f"NaN or Inf loss detected in batch {batch_idx}. Skipping backward pass.")
                            # Skip this batch but continue training
                            continue
                            
                        # Backward pass with mixed precision 
                        scaler.scale(loss).backward()
                        
                        # Apply gradient value clipping to catch extreme values before they cause NaNs
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_value_(model.parameters(), self.max_grad_value)
                        
                        # Check for NaN or Inf values in gradients
                        nan_detected = False
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                    nan_detected = True
                                    logger.warning(f"NaN/Inf gradient detected in {name}. Zeroing this gradient.")
                                    param.grad = torch.zeros_like(param.grad)
                        
                        # Update weights only after accumulating gradients
                        accumulated_batches += 1
                        if accumulated_batches % self.gradient_accumulation_steps == 0 or batch_idx == len(train_loader) - 1:
                            # Apply gradient norm clipping 
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.max_grad_norm)
                            
                            # Skip optimizer step if NaN gradients were detected and zeroed
                            if not nan_detected:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                logger.warning(f"NaN gradients detected in batch {batch_idx}. Skipping optimizer step.")
                            
                            optimizer.zero_grad()  # Clear gradients after optimization step
                            
                            # Update learning rate scheduler
                            scheduler.step()
                            
                            # Update progress bar
                            progress_bar.update(1)
                        
                        # Use loss.item() only if loss is finite
                        if not torch.isnan(loss).any() and not torch.isinf(loss).any():
                            # Multiply by gradient_accumulation_steps to get the actual loss value
                            train_loss += loss.item() * self.gradient_accumulation_steps
                        else:
                            logger.warning(f"NaN/Inf loss in batch {batch_idx}. Not including in average.")
                            
                        progress_bar.set_postfix({
                            "epoch": epoch + 1,
                            "train_loss": train_loss / max(1, (batch_idx + 1) / self.gradient_accumulation_steps),  # Normalize by number of updates
                            "lr": scheduler.get_last_lr()[0]
                        })
                        
                        # Log training metrics every log_every steps
                        if accumulated_batches % self.logging_steps == 0:
                            num_updates = accumulated_batches // self.gradient_accumulation_steps
                            logger.info(
                                f"Epoch: {epoch + 1}/{self.num_epochs}, "
                                f"Updates: {num_updates}/{num_update_steps_per_epoch}, "
                                f"Loss: {train_loss / max(1, num_updates):.4f}, "  # Normalize by number of updates
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
                            
                            # Move inputs to encoder device
                            encoder_device = model.encoder_device
                            distorted_audio = batch["distorted_audio"].to(encoder_device)
                            input_ids = batch["input_ids"].to(encoder_device) if "input_ids" in batch else None
                            attention_mask = batch["attention_mask"].to(encoder_device) if "attention_mask" in batch else None
                            tool_input_ids = batch["tool_input_ids"].to(encoder_device) if "tool_input_ids" in batch else None
                            tool_attention_mask = batch["tool_attention_mask"].to(encoder_device) if "tool_attention_mask" in batch else None
                            token_identifiers = batch.get("token_identifiers", None)  # This stays as a list
                            
                            # If using mixed precision, convert input tensors to fp16
                            if use_mixed_precision:
                                if distorted_audio is not None:
                                    distorted_audio = distorted_audio.half()
                                    logger.debug(f"Validation: Converted distorted_audio to dtype: {distorted_audio.dtype}")
                            
                            # Forward pass with the same precision as training
                            with torch.cuda.amp.autocast(enabled=use_mixed_precision):
                                outputs = model(
                                    text_prompt=input_ids,
                                    audio_waveform=distorted_audio,
                                    token_identifiers=token_identifiers,
                                    target_tool_tokens=None,
                                    attention_mask=attention_mask,
                                    tool_input_ids=tool_input_ids,
                                    tool_attention_mask=tool_attention_mask,
                                    labels=tool_input_ids,  # Use the tool tokens as labels for the decoder
                                    embedding_scale=self.embedding_scale  # Use configured embedding scale
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
                    f"Train Loss: {train_loss / max(1, num_update_steps_per_epoch):.4f}, "
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
                
                # Move inputs to encoder device
                encoder_device = model.encoder_device
                distorted_audio = batch["distorted_audio"].to(encoder_device)
                input_ids = batch["input_ids"].to(encoder_device) if "input_ids" in batch else None
                attention_mask = batch["attention_mask"].to(encoder_device) if "attention_mask" in batch else None
                tool_input_ids = batch["tool_input_ids"].to(encoder_device) if "tool_input_ids" in batch else None
                tool_attention_mask = batch["tool_attention_mask"].to(encoder_device) if "tool_attention_mask" in batch else None
                
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
                    labels=tool_input_ids,  # Use the tool tokens as labels for causal LM
                    embedding_scale=10.0  # Try a smaller scale for stability
                )
                loss = outputs.loss
                
                # Move loss to encoder device for accumulation
                if loss.device != encoder_device:
                    loss = loss.to(encoder_device)
                    
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

    # First, determine the max number of stems and chunks
    max_stems = 0
    max_chunks = 0
    for sample in valid_samples:
        try:
            if "distorted_audio" in sample and sample["distorted_audio"].size(0) > max_stems:
                max_stems = sample["distorted_audio"].size(0)
            if "distorted_audio" in sample and sample["distorted_audio"].size(1) > max_chunks:
                max_chunks = sample["distorted_audio"].size(1)
        except (KeyError, ValueError, TypeError, IndexError) as e:
            logger.warning(f"Error determining sample dimensions: {e}")
            continue

    # Now process each sample
    for sample in valid_samples:
        try:
            # Get the original shapes
            audio = sample["distorted_audio"]
            curr_stems = audio.size(0)
            curr_chunks = audio.size(1)
            audio_length = audio.size(2)
            
            # Pad stems if needed
            if curr_stems < max_stems:
                stem_padding = torch.zeros((max_stems - curr_stems, curr_chunks, audio_length), dtype=audio.dtype)
                audio = torch.cat([audio, stem_padding], dim=0)
            
            # Pad chunks if needed
            if curr_chunks < max_chunks:
                chunk_padding = torch.zeros((max_stems, max_chunks - curr_chunks, audio_length), dtype=audio.dtype)
                audio = torch.cat([audio, chunk_padding], dim=1)
            
            # Add the padded audio to the batch
            distorted_audio.append(audio)
            
            # Process token_identifiers to match padded audio
            sample_tokens = sample["token_identifiers"]
            padded_tokens = []
            
            for stem_idx, stem_tokens in enumerate(sample_tokens):
                # Pad token list with "[PAD]" to match max_chunks
                padded_stem_tokens = stem_tokens + ["[PAD]"] * (max_chunks - len(stem_tokens))
                padded_tokens.append(padded_stem_tokens)
            
            # Add additional padded stems if needed
            while len(padded_tokens) < max_stems:
                padded_tokens.append(["[PAD]"] * max_chunks)
            
            token_identifiers.append(padded_tokens)
            
            # Add other tensors
            input_ids.append(sample["input_ids"])
            attention_mask.append(sample["attention_mask"])
            tool_input_ids.append(sample["tool_input_ids"])
            tool_attention_mask.append(sample["tool_attention_mask"])
        except (KeyError, ValueError, TypeError, IndexError) as e:
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
    parser.add_argument('--multi_gpu', action='store_true',
                        help='Enable multi-GPU training using device_map (default: False)')
    parser.add_argument('--lora_r', type=int, default=None,
                        help='Override the LoRA rank parameter (default: config value)')
    parser.add_argument('--lora_alpha', type=int, default=None,
                        help='Override the LoRA alpha parameter (default: config value)')
    parser.add_argument('--lora_dropout', type=float, default=None,
                        help='Override the LoRA dropout parameter (default: config value)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Override the learning rate in the config (optional)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of update steps to accumulate before performing a backward/update pass (default: 1)')
    parser.add_argument('--embedding_scale', type=float, default=2.0,
                        help='Scaling factor for embeddings (default: 2.0)')
    parser.add_argument('--disable_mixed_precision_epochs', type=int, default=1,
                        help='Number of initial epochs to disable mixed precision (default: 1)')
                        
    args = parser.parse_args()
    
    # Load the configuration
    config = load_mixer_config(args.config)
    
    # Override LoRA parameters in config if provided
    if args.lora_r is not None or args.lora_alpha is not None or args.lora_dropout is not None:
        if "mixer" not in config:
            config["mixer"] = {}
        if "lora" not in config["mixer"]:
            config["mixer"]["lora"] = {}
            
        if args.lora_r is not None:
            config["mixer"]["lora"]["r"] = args.lora_r
            print(f"Overriding LoRA rank with {args.lora_r}")
            
        if args.lora_alpha is not None:
            config["mixer"]["lora"]["lora_alpha"] = args.lora_alpha
            print(f"Overriding LoRA alpha with {args.lora_alpha}")
            
        if args.lora_dropout is not None:
            config["mixer"]["lora"]["lora_dropout"] = args.lora_dropout
            print(f"Overriding LoRA dropout with {args.lora_dropout}")
    
    # Override learning rate if provided
    if args.learning_rate is not None:
        if "default" not in config:
            config["default"] = {}
        config["default"]["learning_rate"] = args.learning_rate
        print(f"Overriding learning rate with {args.learning_rate}")
        
    # Add gradient clipping parameters
    if "default" not in config:
        config["default"] = {}
    config["default"]["max_grad_norm"] = 1.0
    config["default"]["max_grad_value"] = 1.0
        
    # Enable mixed precision by default
    if "default" not in config:
        config["default"] = {}
    config["default"]["mixed_precision"] = True
    print("Mixed precision (FP16) training is enabled")
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        for i in range(torch.cuda.device_count()):
            free_memory, total_memory = torch.cuda.mem_get_info(i)
            free_gb = free_memory / (1024**3)
            total_gb = total_memory / (1024**3)
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {device_name}, {free_gb:.2f}GB free / {total_gb:.2f}GB total")
    
    # Create the trainer with all arguments
    trainer = MixerTrainer(config, args.data_dir)
    
    # Set command line arguments explicitly
    trainer.artist = args.artist
    trainer.limit = args.limit
    trainer.multi_gpu = args.multi_gpu
    trainer.gradient_accumulation_steps = args.gradient_accumulation_steps
    trainer.embedding_scale = args.embedding_scale
    trainer.disable_mixed_precision_epochs = args.disable_mixed_precision_epochs
    if args.num_epochs is not None:
        trainer.num_epochs = args.num_epochs
        
    print(f"Training with:")
    print(f"- Gradient accumulation steps: {trainer.gradient_accumulation_steps}")
    print(f"- Embedding scale: {trainer.embedding_scale}")
    print(f"- Mixed precision disabled for first {trainer.disable_mixed_precision_epochs} epochs")
    print(f"- Multi-GPU mode: {'Enabled' if trainer.multi_gpu else 'Disabled'}")
    
    # Train the mixer
    trainer.train_mixer()

if __name__ == "__main__":
    main()