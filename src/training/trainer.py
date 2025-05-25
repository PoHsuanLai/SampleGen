"""
Unified Trainer for the Unified Producer Model.
Handles training for stem generation planning and Faust script generation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from typing import Dict, Optional, List
import soundfile as sf
import numpy as np

# Import the unified model and dataset
from ..models.producer import UnifiedProducerModel
from .dataset import UnifiedProducerDataset, FaustScriptDataset


class UnifiedProducerTrainer:
    """
    Trainer for the Unified Producer Model.
    Handles training for:
    1. Planning which stems to generate
    2. Creating generation prompts
    3. Generating Faust mixing scripts
    4. Quality assessment
    """
    
    def __init__(self,
                 model: UnifiedProducerModel,
                 data_dir: str,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 batch_size: int = 1):
        """
        Initialize unified trainer.
        
        Args:
            model: UnifiedProducerModel to train
            data_dir: Directory containing training data
            device: Device to train on
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
        """
        self.model = model
        self.data_dir = data_dir
        self.device = device
        self.batch_size = batch_size
        
        # Setup optimizer for trainable parameters
        trainable_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                print(f"Trainable parameter: {name}, shape: {param.shape}")
        
        print(f"Found {len(trainable_params)} trainable parameters")
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        
        # Setup datasets
        self.train_dataset = UnifiedProducerDataset(
            data_dir=data_dir,
            use_distortion=True,
            faust_script_training=True
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_fn
        )
        
        # Loss functions
        self.planning_loss_fn = nn.BCEWithLogitsLoss()
        self.quality_loss_fn = nn.MSELoss()
        
    def _collate_fn(self, batch):
        """Custom collate function to handle variable-length data."""
        # Since we're using batch_size=1, just return the first item
        return batch[0]
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_planning_loss = 0.0
        total_quality_loss = 0.0
        total_faust_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Process batch
                losses = self._process_batch(batch)
                
                # Combine losses
                total_loss = (
                    losses['planning_loss'] + 
                    losses['quality_loss'] + 
                    losses.get('faust_loss', 0.0)
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Accumulate losses
                total_planning_loss += losses['planning_loss'].item()
                total_quality_loss += losses['quality_loss'].item()
                if 'faust_loss' in losses:
                    total_faust_loss += losses['faust_loss'].item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}: Planning Loss = {losses['planning_loss'].item():.4f}, "
                          f"Quality Loss = {losses['quality_loss'].item():.4f}")
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_planning_loss = total_planning_loss / max(num_batches, 1)
        avg_quality_loss = total_quality_loss / max(num_batches, 1)
        avg_faust_loss = total_faust_loss / max(num_batches, 1)
        
        return {
            'planning_loss': avg_planning_loss,
            'quality_loss': avg_quality_loss,
            'faust_loss': avg_faust_loss,
            'total_loss': avg_planning_loss + avg_quality_loss + avg_faust_loss
        }
    
    def _process_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Process a training batch."""
        stems = batch['stems']
        style_prompt = batch['style_prompt']
        planning_targets = batch['planning_targets']
        generation_prompts = batch['generation_prompts']
        faust_script = batch.get('faust_script')
        
        losses = {}
        
        # 1. Planning loss - train the model to predict which stems to generate
        with torch.no_grad():
            # Encode the style prompt
            inputs = self.model.text_tokenizer(
                style_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            text_embedding = self.model.text_encoder(**inputs).last_hidden_state.mean(dim=1)
        
        # Get planning predictions
        planning_logits = self.model.planning_head(text_embedding)
        planning_targets_tensor = planning_targets.to(self.device).unsqueeze(0)
        
        planning_loss = self.planning_loss_fn(planning_logits, planning_targets_tensor)
        losses['planning_loss'] = planning_loss
        
        # 2. Quality assessment loss
        # Create a simple mixed audio for quality assessment
        mixed_audio = self._create_simple_mix(stems)
        predicted_quality = self.model.assess_quality(mixed_audio, style_prompt)
        
        # Target quality based on whether distortions were applied
        if 'distorted_stems' in batch:
            target_quality = 0.6  # Lower quality for distorted audio
        else:
            target_quality = 0.8  # Higher quality for clean audio
        
        quality_loss = self.quality_loss_fn(
            torch.tensor(predicted_quality, device=self.device),
            torch.tensor(target_quality, device=self.device)
        )
        losses['quality_loss'] = quality_loss
        
        # 3. Faust script generation loss (if available)
        if faust_script and self.model.text_generator is not None:
            try:
                faust_loss = self._compute_faust_loss(stems, style_prompt, faust_script)
                losses['faust_loss'] = faust_loss
            except Exception as e:
                print(f"Faust loss computation failed: {e}")
        
        return losses
    
    def _create_simple_mix(self, stems: Dict[str, np.ndarray]) -> np.ndarray:
        """Create a simple mix of stems for quality assessment."""
        if not stems:
            return np.array([])
        
        # Find the maximum length
        max_length = max(len(stem) for stem in stems.values())
        
        # Mix stems with appropriate levels
        mixed = np.zeros(max_length)
        stem_weights = {'vocals': 0.7, 'drums': 0.9, 'bass': 0.8, 'other': 0.5}
        
        for stem_name, audio in stems.items():
            # Pad or truncate to match length
            if len(audio) < max_length:
                audio = np.pad(audio, (0, max_length - len(audio)))
            else:
                audio = audio[:max_length]
            
            # Apply weight
            weight = stem_weights.get(stem_name, 0.7)
            mixed += audio * weight
        
        # Normalize
        if np.max(np.abs(mixed)) > 0:
            mixed = mixed / np.max(np.abs(mixed)) * 0.8
        
        return mixed
    
    def _compute_faust_loss(self, stems: Dict[str, np.ndarray], style_prompt: str, target_script: str) -> torch.Tensor:
        """Compute loss for Faust script generation."""
        # Generate Faust script using the model
        generated_script = self.model.generate_faust_script(stems, style_prompt)
        
        # For now, use a simple string similarity loss
        # In practice, you might want to use more sophisticated metrics
        
        # Convert scripts to tokens for comparison
        target_tokens = self.model.generator_tokenizer(
            target_script,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        generated_tokens = self.model.generator_tokenizer(
            generated_script,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Simple token-level loss (this is a simplified approach)
        target_ids = target_tokens['input_ids'].to(self.device)
        generated_ids = generated_tokens['input_ids'].to(self.device)
        
        # Pad to same length
        max_len = max(target_ids.shape[1], generated_ids.shape[1])
        if target_ids.shape[1] < max_len:
            target_ids = torch.cat([
                target_ids,
                torch.zeros(target_ids.shape[0], max_len - target_ids.shape[1], dtype=torch.long, device=self.device)
            ], dim=1)
        if generated_ids.shape[1] < max_len:
            generated_ids = torch.cat([
                generated_ids,
                torch.zeros(generated_ids.shape[0], max_len - generated_ids.shape[1], dtype=torch.long, device=self.device)
            ], dim=1)
        
        # Compute similarity loss
        similarity = torch.sum(target_ids == generated_ids).float() / max_len
        faust_loss = 1.0 - similarity  # Loss is 1 - similarity
        
        return faust_loss
    
    def demo_full_pipeline(self, 
                          audio_file_path: str,
                          text_prompt: str,
                          output_path: str) -> None:
        """
        Demonstrate the full pipeline on a single file.
        
        Args:
            audio_file_path: Path to input audio file
            text_prompt: User's description
            output_path: Where to save the result
        """
        print(f"Processing: {audio_file_path}")
        print(f"Prompt: {text_prompt}")
        
        self.model.eval()
        
        with torch.no_grad():
            # Step 1: Plan production
            print("\n1. Planning production...")
            plan = self.model.plan_production(
                text_prompt=text_prompt,
                audio_file_path=audio_file_path
            )
            print(f"Plan: {plan}")
            
            # Step 2: Extract original stems
            print("\n2. Extracting original stems...")
            original_stems, _ = self.model.stem_extractor.extract_stems_from_file(audio_file_path)
            print(f"Extracted stems: {list(original_stems.keys())}")
            
            # Step 3: Generate new stems
            print("\n3. Generating new stems...")
            generated_stems = self.model.generate_stems(plan, duration=10.0)
            print(f"Generated stems: {list(generated_stems.keys())}")
            
            # Step 4: Generate Faust script
            print("\n4. Generating Faust mixing script...")
            all_stems = {**original_stems, **generated_stems}
            faust_script = self.model.generate_faust_script(all_stems, text_prompt)
            print(f"Generated Faust script:\n{faust_script}")
            
            # Step 5: Create initial mix
            print("\n5. Creating initial mix...")
            mixed_audio = self.model.create_mix(original_stems, generated_stems, text_prompt)
            
            # Step 6: Iterative refinement
            print("\n6. Applying iterative refinement...")
            final_mix, final_quality = self.model.iterative_refinement(
                original_stems, generated_stems, text_prompt,
                max_iterations=2, quality_threshold=0.7
            )
            
            print(f"Final quality score: {final_quality:.3f}")
            
            # Save result
            sf.write(output_path, final_mix, self.model.sample_rate)
            print(f"Saved result to: {output_path}")
            
            # Save Faust script
            faust_output_path = output_path.replace('.wav', '_faust_script.dsp')
            with open(faust_output_path, 'w') as f:
                f.write(faust_script)
            print(f"Saved Faust script to: {faust_output_path}")
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {path}")


class FaustScriptTrainer:
    """
    Specialized trainer for Faust script generation.
    """
    
    def __init__(self,
                 model: UnifiedProducerModel,
                 audio_files: List[str],
                 device: str = 'cuda',
                 learning_rate: float = 1e-4):
        """
        Initialize Faust script trainer.
        
        Args:
            model: UnifiedProducerModel to train
            audio_files: List of audio files for training
            device: Device to train on
            learning_rate: Learning rate
        """
        self.model = model
        self.device = device
        
        # Setup optimizer for text generator only
        trainable_params = [p for p in model.text_generator.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        
        # Setup dataset
        self.dataset = FaustScriptDataset(audio_files)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch on Faust script generation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.dataloader):
            try:
                prompt = batch['prompt'][0]  # Remove batch dimension
                target_script = batch['target_script'][0]
                
                # Generate script using model
                generated_script = self.model.generate_faust_script({}, prompt)
                
                # Compute loss (simplified token-level comparison)
                loss = self._compute_script_loss(generated_script, target_script)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.text_generator.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    print(f"Faust Batch {batch_idx}: Loss = {loss.item():.4f}")
                    
            except Exception as e:
                print(f"Error in Faust batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'faust_script_loss': avg_loss}
    
    def _compute_script_loss(self, generated_script: str, target_script: str) -> torch.Tensor:
        """Compute loss between generated and target Faust scripts."""
        # Tokenize both scripts
        target_tokens = self.model.generator_tokenizer(
            target_script,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        generated_tokens = self.model.generator_tokenizer(
            generated_script,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Move to device
        target_ids = target_tokens['input_ids'].to(self.device)
        generated_ids = generated_tokens['input_ids'].to(self.device)
        
        # Compute cross-entropy loss
        # This is a simplified approach - in practice you might want more sophisticated metrics
        min_len = min(target_ids.shape[1], generated_ids.shape[1])
        
        if min_len > 0:
            target_subset = target_ids[:, :min_len]
            generated_subset = generated_ids[:, :min_len]
            
            # Simple token matching loss
            matches = (target_subset == generated_subset).float()
            accuracy = matches.mean()
            loss = 1.0 - accuracy
        else:
            loss = torch.tensor(1.0, device=self.device)
        
        return loss 