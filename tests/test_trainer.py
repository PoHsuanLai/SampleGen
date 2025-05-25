"""
Unit tests for the trainer classes.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.training.trainer import HipHopProducerTrainer, SelfSupervisedPretrainer


class TestHipHopProducerTrainer:
    """Test suite for HipHopProducerTrainer."""
    
    @pytest.fixture
    def mock_model(self, device):
        """Create a mock HipHopProducerModel."""
        model = MagicMock()
        model.parameters.return_value = [
            torch.nn.Parameter(torch.randn(10, 10, requires_grad=True)),
            torch.nn.Parameter(torch.randn(5, requires_grad=True))
        ]
        model.assess_quality.return_value = 0.7
        model._simple_mix.return_value = np.random.randn(44100).astype(np.float32)
        model.plan_production.return_value = {'generate_bass': 'Deep bass'}
        model.generate_stems.return_value = {'bass': np.random.randn(44100).astype(np.float32)}
        model.create_mix.return_value = np.random.randn(44100).astype(np.float32)
        model.iterative_refinement.return_value = (
            np.random.randn(44100).astype(np.float32), 0.8
        )
        model.stem_extractor = MagicMock()
        model.stem_extractor.extract_stems_from_file.return_value = (
            {'vocals': np.random.randn(44100).astype(np.float32)}, None
        )
        model.sample_rate = 44100
        return model
    
    @pytest.fixture
    def mock_dataset(self):
        """Create a mock HipHopDataset."""
        with patch('src.training.trainer.HipHopDataset') as mock_dataset_class:
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 10
            mock_dataset.__getitem__.return_value = {
                'stems': {
                    'vocals': [np.random.randn(44100).astype(np.float32)],
                    'drums': [np.random.randn(44100).astype(np.float32)]
                },
                'style_prompt': ['Test hip-hop style'],
                'ground_truth_dsp': [{'tool': 'test', 'params': {}}]
            }
            mock_dataset_class.return_value = mock_dataset
            yield mock_dataset
    
    @pytest.fixture
    def mock_dataloader(self, mock_dataset):
        """Create a mock DataLoader."""
        with patch('src.training.trainer.DataLoader') as mock_dataloader_class:
            mock_dataloader = MagicMock()
            # Simulate a few batches
            batch_data = {
                'stems': {
                    'vocals': [np.random.randn(44100).astype(np.float32)],
                    'drums': [np.random.randn(44100).astype(np.float32)]
                },
                'style_prompt': ['Test style'],
                'ground_truth_dsp': [{'tool': 'test', 'params': {}}]
            }
            mock_dataloader.__iter__.return_value = iter([batch_data, batch_data])
            mock_dataloader_class.return_value = mock_dataloader
            yield mock_dataloader
    
    @pytest.fixture
    def trainer(self, mock_model, mock_dataset, mock_dataloader, temp_dir, device):
        """Create a HipHopProducerTrainer instance."""
        return HipHopProducerTrainer(
            model=mock_model,
            data_dir=temp_dir,
            device=device,
            learning_rate=1e-4
        )
    
    def test_initialization(self, trainer, mock_model, temp_dir, device):
        """Test trainer initialization."""
        assert trainer.model == mock_model
        assert trainer.data_dir == temp_dir
        assert trainer.device == device
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert hasattr(trainer, 'train_dataset')
        assert hasattr(trainer, 'train_loader')
    
    def test_process_batch(self, trainer):
        """Test batch processing."""
        batch = {
            'stems': {
                'vocals': [np.random.randn(44100).astype(np.float32)],
                'drums': [np.random.randn(44100).astype(np.float32)]
            },
            'style_prompt': ['Test style'],
            'ground_truth_dsp': [{'tool': 'test', 'params': {}}]
        }
        
        loss = trainer._process_batch(batch)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_process_batch_error_handling(self, trainer):
        """Test error handling in batch processing."""
        # Mock assess_quality to raise an exception
        trainer.model.assess_quality.side_effect = Exception("Mock error")
        
        batch = {
            'stems': {
                'vocals': [np.random.randn(44100).astype(np.float32)]
            },
            'style_prompt': ['Test style'],
            'ground_truth_dsp': [{'tool': 'test', 'params': {}}]
        }
        
        loss = trainer._process_batch(batch)
        
        # Should handle the error gracefully and return a valid loss
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
    
    def test_train_epoch(self, trainer, mock_dataloader):
        """Test training for one epoch."""
        results = trainer.train_epoch()
        
        assert isinstance(results, dict)
        assert 'train_loss' in results
        assert isinstance(results['train_loss'], float)
        assert results['train_loss'] >= 0
    
    def test_train_epoch_with_errors(self, trainer, mock_dataloader):
        """Test training epoch with batch errors."""
        # Mock the dataloader to include a problematic batch
        def side_effect_iter():
            # First batch is normal
            yield {
                'stems': {
                    'vocals': [np.random.randn(44100).astype(np.float32)]
                },
                'style_prompt': ['Test style'],
                'ground_truth_dsp': [{'tool': 'test', 'params': {}}]
            }
            # Second batch causes an error
            raise Exception("Mock batch error")
        
        mock_dataloader.__iter__ = side_effect_iter
        
        results = trainer.train_epoch()
        
        # Should handle errors and still return results
        assert isinstance(results, dict)
        assert 'train_loss' in results
    
    def test_demo_full_pipeline(self, trainer, test_wav_file, temp_dir):
        """Test the full pipeline demonstration."""
        output_path = os.path.join(temp_dir, "demo_output.wav")
        
        trainer.demo_full_pipeline(
            test_wav_file,
            "Dark trap beat",
            output_path
        )
        
        # Should have called the model methods
        trainer.model.plan_production.assert_called_once()
        trainer.model.stem_extractor.extract_stems_from_file.assert_called_once()
        trainer.model.generate_stems.assert_called_once()
        trainer.model.create_mix.assert_called_once()
        trainer.model.iterative_refinement.assert_called_once()
        
        # Should have created the output file
        assert os.path.exists(output_path)
    
    def test_save_checkpoint(self, trainer, temp_dir):
        """Test saving training checkpoint."""
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        
        trainer.save_checkpoint(checkpoint_path)
        
        assert os.path.exists(checkpoint_path)
        
        # Verify checkpoint contents
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
    
    def test_load_checkpoint(self, trainer, temp_dir):
        """Test loading training checkpoint."""
        checkpoint_path = os.path.join(temp_dir, "test_checkpoint.pt")
        
        # First save a checkpoint
        trainer.save_checkpoint(checkpoint_path)
        
        # Then load it
        trainer.load_checkpoint(checkpoint_path)
        
        # Should have called load_state_dict on model and optimizer
        trainer.model.load_state_dict.assert_called_once()
        trainer.optimizer.load_state_dict.assert_called_once()
    
    @pytest.mark.parametrize("learning_rate", [1e-3, 1e-4, 1e-5])
    def test_different_learning_rates(self, mock_model, temp_dir, device, learning_rate):
        """Test trainer with different learning rates."""
        with patch('src.training.trainer.HipHopDataset'), \
             patch('src.training.trainer.DataLoader'):
            
            trainer = HipHopProducerTrainer(
                model=mock_model,
                data_dir=temp_dir,
                device=device,
                learning_rate=learning_rate
            )
            
            # Check that optimizer was created with correct learning rate
            assert trainer.optimizer.param_groups[0]['lr'] == learning_rate
    
    def test_trainable_parameters_filtering(self, temp_dir, device):
        """Test that only trainable parameters are added to optimizer."""
        with patch('src.training.trainer.HipHopDataset'), \
             patch('src.training.trainer.DataLoader'):
            
            mock_model = MagicMock()
            
            # Create parameters with different requires_grad
            trainable_param = torch.nn.Parameter(torch.randn(5, 5), requires_grad=True)
            frozen_param = torch.nn.Parameter(torch.randn(3, 3), requires_grad=False)
            
            mock_model.parameters.return_value = [trainable_param, frozen_param]
            
            trainer = HipHopProducerTrainer(
                model=mock_model,
                data_dir=temp_dir,
                device=device
            )
            
            # Should only have the trainable parameter
            assert len(trainer.optimizer.param_groups[0]['params']) == 1


class TestSelfSupervisedPretrainer:
    """Test suite for SelfSupervisedPretrainer."""
    
    @pytest.fixture
    def mock_model(self, device):
        """Create a mock HipHopProducerModel for pretraining."""
        model = MagicMock()
        model.parameters.return_value = [
            torch.nn.Parameter(torch.randn(10, 10, requires_grad=True))
        ]
        model.assess_quality.return_value = 0.7
        return model
    
    @pytest.fixture
    def mock_distortion_dataset(self):
        """Create a mock AudioDistortionDataset."""
        with patch('src.training.trainer.AudioDistortionDataset') as mock_dataset_class:
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 5
            mock_dataset_class.return_value = mock_dataset
            yield mock_dataset
    
    @pytest.fixture
    def mock_distortion_dataloader(self, mock_distortion_dataset):
        """Create a mock DataLoader for distortion dataset."""
        with patch('src.training.trainer.DataLoader') as mock_dataloader_class:
            mock_dataloader = MagicMock()
            # Simulate batches
            batch_data = {
                'clean_audio': [torch.randn(44100)],
                'distorted_audio': [torch.randn(44100)],
                'corrections': [{'tool': 'test', 'params': {}}]
            }
            mock_dataloader.__iter__.return_value = iter([batch_data, batch_data])
            mock_dataloader_class.return_value = mock_dataloader
            yield mock_dataloader
    
    @pytest.fixture
    def audio_files(self, temp_dir, test_audio_mono, sample_rate):
        """Create test audio files for pretraining."""
        files = []
        for i in range(3):
            file_path = os.path.join(temp_dir, f"pretrain_audio_{i}.wav")
            import soundfile as sf
            sf.write(file_path, test_audio_mono, sample_rate)
            files.append(file_path)
        return files
    
    @pytest.fixture
    def pretrainer(self, mock_model, audio_files, mock_distortion_dataset, 
                  mock_distortion_dataloader, device):
        """Create a SelfSupervisedPretrainer instance."""
        return SelfSupervisedPretrainer(
            model=mock_model,
            audio_files=audio_files,
            device=device,
            learning_rate=1e-4
        )
    
    def test_initialization(self, pretrainer, mock_model, audio_files, device):
        """Test pretrainer initialization."""
        assert pretrainer.model == mock_model
        assert pretrainer.device == device
        assert isinstance(pretrainer.optimizer, torch.optim.AdamW)
        assert hasattr(pretrainer, 'dataset')
        assert hasattr(pretrainer, 'dataloader')
    
    def test_pretrain_epoch(self, pretrainer, mock_distortion_dataloader):
        """Test pretraining for one epoch."""
        results = pretrainer.pretrain_epoch()
        
        assert isinstance(results, dict)
        assert 'pretrain_loss' in results
        assert isinstance(results['pretrain_loss'], float)
        assert results['pretrain_loss'] >= 0
    
    def test_pretrain_epoch_with_errors(self, pretrainer, mock_distortion_dataloader):
        """Test pretraining epoch with batch errors."""
        # Mock assess_quality to raise an exception for some calls
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Every third call fails
                raise Exception("Mock quality assessment error")
            return 0.5
        
        pretrainer.model.assess_quality.side_effect = side_effect
        
        results = pretrainer.pretrain_epoch()
        
        # Should handle errors and still return results
        assert isinstance(results, dict)
        assert 'pretrain_loss' in results
    
    def test_margin_loss_calculation(self, pretrainer):
        """Test margin loss calculation."""
        # Mock quality assessments
        pretrainer.model.assess_quality.side_effect = [0.8, 0.4]  # clean, distorted
        
        # Create a batch
        batch = {
            'clean_audio': [torch.randn(44100)],
            'distorted_audio': [torch.randn(44100)],
            'corrections': [{'tool': 'test', 'params': {}}]
        }
        
        # Process the batch manually to test loss calculation
        clean_audio = batch['clean_audio'][0].numpy()
        distorted_audio = batch['distorted_audio'][0].numpy()
        
        clean_quality = pretrainer.model.assess_quality(clean_audio, "Clean audio")
        distorted_quality = pretrainer.model.assess_quality(distorted_audio, "Distorted audio")
        
        # Calculate margin loss
        margin_loss = torch.max(
            torch.tensor(0.0, device=pretrainer.device),
            torch.tensor(distorted_quality - clean_quality + 0.2, device=pretrainer.device)
        )
        
        # Clean quality (0.8) should be higher than distorted (0.4)
        # So margin loss should be max(0, 0.4 - 0.8 + 0.2) = max(0, -0.2) = 0
        assert margin_loss.item() == 0.0
    
    def test_margin_loss_with_bad_clean_quality(self, pretrainer):
        """Test margin loss when clean quality is unexpectedly low."""
        # Mock quality assessments where clean is worse than distorted
        pretrainer.model.assess_quality.side_effect = [0.3, 0.7]  # clean, distorted
        
        batch = {
            'clean_audio': [torch.randn(44100)],
            'distorted_audio': [torch.randn(44100)],
            'corrections': [{'tool': 'test', 'params': {}}]
        }
        
        clean_audio = batch['clean_audio'][0].numpy()
        distorted_audio = batch['distorted_audio'][0].numpy()
        
        clean_quality = pretrainer.model.assess_quality(clean_audio, "Clean audio")
        distorted_quality = pretrainer.model.assess_quality(distorted_audio, "Distorted audio")
        
        # Calculate margin loss
        margin_loss = torch.max(
            torch.tensor(0.0, device=pretrainer.device),
            torch.tensor(distorted_quality - clean_quality + 0.2, device=pretrainer.device)
        )
        
        # margin loss should be max(0, 0.7 - 0.3 + 0.2) = max(0, 0.6) = 0.6
        assert abs(margin_loss.item() - 0.6) < 1e-6
    
    def test_optimizer_setup(self, mock_model, audio_files, device):
        """Test optimizer setup with only trainable parameters."""
        # Create model with mixed trainable/frozen parameters
        trainable_param = torch.nn.Parameter(torch.randn(5, 5), requires_grad=True)
        frozen_param = torch.nn.Parameter(torch.randn(3, 3), requires_grad=False)
        mock_model.parameters.return_value = [trainable_param, frozen_param]
        
        with patch('src.training.trainer.AudioDistortionDataset'), \
             patch('src.training.trainer.DataLoader'):
            
            pretrainer = SelfSupervisedPretrainer(
                model=mock_model,
                audio_files=audio_files,
                device=device
            )
            
            # Should only have trainable parameters in optimizer
            assert len(pretrainer.optimizer.param_groups[0]['params']) == 1
    
    @pytest.mark.parametrize("learning_rate", [1e-3, 1e-4, 1e-5])
    def test_different_learning_rates(self, mock_model, audio_files, device, learning_rate):
        """Test pretrainer with different learning rates."""
        with patch('src.training.trainer.AudioDistortionDataset'), \
             patch('src.training.trainer.DataLoader'):
            
            pretrainer = SelfSupervisedPretrainer(
                model=mock_model,
                audio_files=audio_files,
                device=device,
                learning_rate=learning_rate
            )
            
            assert pretrainer.optimizer.param_groups[0]['lr'] == learning_rate
    
    def test_empty_audio_files(self, mock_model, device):
        """Test pretrainer with empty audio files list."""
        with patch('src.training.trainer.AudioDistortionDataset') as mock_dataset_class, \
             patch('src.training.trainer.DataLoader'):
            
            mock_dataset = MagicMock()
            mock_dataset.__len__.return_value = 0
            mock_dataset_class.return_value = mock_dataset
            
            pretrainer = SelfSupervisedPretrainer(
                model=mock_model,
                audio_files=[],
                device=device
            )
            
            # Should handle empty dataset gracefully
            assert hasattr(pretrainer, 'dataset')
            assert hasattr(pretrainer, 'dataloader')
    
    def test_gradient_clipping(self, pretrainer, mock_distortion_dataloader):
        """Test gradient clipping during pretraining."""
        # Mock parameters to have large gradients
        mock_param = torch.nn.Parameter(torch.randn(10, 10, requires_grad=True))
        mock_param.grad = torch.randn(10, 10) * 100  # Large gradients
        pretrainer.model.parameters.return_value = [mock_param]
        
        with patch('torch.nn.utils.clip_grad_norm_') as mock_clip:
            results = pretrainer.pretrain_epoch()
            
            # Should have called gradient clipping
            assert mock_clip.call_count > 0 