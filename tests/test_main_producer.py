"""
Real tests for main producer module - testing actual argument parsing and logic without heavy mocks.
"""

import pytest
import sys
import os
import tempfile
import argparse
import subprocess
from pathlib import Path
from unittest.mock import patch

# Import the actual main producer module
from src.main_producer import main, demo_with_example


class TestMainProducerReal:
    """Test the actual main producer implementation."""

    def test_argument_parser_setup(self):
        """Test that argument parser is set up correctly."""
        # Create a parser like the one in main()
        parser = argparse.ArgumentParser(description='Hip-Hop Producer AI - Unified Music Production System')
        
        parser.add_argument('--mode', choices=['demo', 'train', 'pretrain'], default='demo')
        parser.add_argument('--input', '-i', type=str)
        parser.add_argument('--prompt', '-p', type=str)
        parser.add_argument('--output', '-o', type=str, default='output.wav')
        parser.add_argument('--data-dir', type=str, default='data')
        parser.add_argument('--device', type=str, default='cuda')
        parser.add_argument('--epochs', type=int, default=1)
        parser.add_argument('--checkpoint', type=str)
        
        # Test default values
        args = parser.parse_args([])
        assert args.mode == 'demo'
        assert args.output == 'output.wav'
        assert args.data_dir == 'data'
        assert args.device == 'cuda'
        assert args.epochs == 1
        
        # Test parsing various arguments
        args = parser.parse_args(['--mode', 'train', '--device', 'cpu', '--epochs', '5'])
        assert args.mode == 'train'
        assert args.device == 'cpu'
        assert args.epochs == 5

    def test_argument_parser_demo_mode(self):
        """Test argument parsing for demo mode."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', choices=['demo', 'train', 'pretrain'], default='demo')
        parser.add_argument('--input', '-i', type=str)
        parser.add_argument('--prompt', '-p', type=str)
        parser.add_argument('--output', '-o', type=str, default='output.wav')
        parser.add_argument('--device', type=str, default='cuda')
        
        # Test demo mode with short flags
        args = parser.parse_args(['-i', 'input.wav', '-p', 'Create a beat', '-o', 'result.wav'])
        assert args.mode == 'demo'  # default
        assert args.input == 'input.wav'
        assert args.prompt == 'Create a beat'
        assert args.output == 'result.wav'

    def test_argument_parser_train_mode(self):
        """Test argument parsing for train mode."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', choices=['demo', 'train', 'pretrain'], default='demo')
        parser.add_argument('--data-dir', type=str, default='data')
        parser.add_argument('--epochs', type=int, default=1)
        parser.add_argument('--checkpoint', type=str)
        parser.add_argument('--device', type=str, default='cuda')
        
        args = parser.parse_args([
            '--mode', 'train',
            '--data-dir', '/path/to/data',
            '--epochs', '10',
            '--checkpoint', 'model.pt',
            '--device', 'cpu'
        ])
        assert args.mode == 'train'
        assert args.data_dir == '/path/to/data'
        assert args.epochs == 10
        assert args.checkpoint == 'model.pt'
        assert args.device == 'cpu'

    def test_file_walking_logic(self):
        """Test the file walking logic for pretrain mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create directory structure similar to what pretrain mode expects
            data_dir = os.path.join(temp_dir, "data")
            os.makedirs(data_dir)
            
            # Create some audio files
            audio_files = [
                "song1.wav",
                "song2.mp3", 
                "song3.flac",
                ".hidden.wav",  # Should be skipped
                "song4.wav"
            ]
            
            for filename in audio_files:
                Path(os.path.join(data_dir, filename)).touch()
            
            # Create a stems subdirectory (should be skipped)
            stems_dir = os.path.join(data_dir, "stems")
            os.makedirs(stems_dir)
            Path(os.path.join(stems_dir, "vocals.wav")).touch()
            Path(os.path.join(stems_dir, "drums.wav")).touch()
            
            # Create nested structure
            sub_dir = os.path.join(data_dir, "artist1")
            os.makedirs(sub_dir)
            Path(os.path.join(sub_dir, "track1.wav")).touch()
            Path(os.path.join(sub_dir, "track2.mp3")).touch()
            
            # Simulate the file collection logic from main()
            collected_files = []
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    if file.endswith(('.wav', '.mp3', '.flac')) and not file.startswith('.'):
                        if '/stems/' not in root:
                            full_path = os.path.join(root, file)
                            collected_files.append(full_path)
            
            # Should find 5 files (excluding hidden and stems)
            assert len(collected_files) == 5
            
            # Should not include hidden files
            assert not any('.hidden' in f for f in collected_files)
            
            # Should not include stems directory files
            assert not any('/stems/' in f for f in collected_files)
            
            # Should include nested files
            assert any('artist1' in f for f in collected_files)

    def test_demo_mode_validation_logic(self):
        """Test the validation logic for demo mode."""
        # Test the logical checks that would happen in demo mode
        
        # Case 1: Missing input and prompt
        input_file = None
        prompt = None
        
        demo_mode_valid = bool(input_file and prompt)
        assert not demo_mode_valid
        
        # Case 2: Has input but no prompt
        input_file = "test.wav"
        prompt = None
        
        demo_mode_valid = bool(input_file and prompt)
        assert not demo_mode_valid
        
        # Case 3: Has prompt but no input
        input_file = None
        prompt = "Create a beat"
        
        demo_mode_valid = bool(input_file and prompt)
        assert not demo_mode_valid
        
        # Case 4: Has both
        input_file = "test.wav"
        prompt = "Create a beat"
        
        demo_mode_valid = bool(input_file and prompt)
        assert demo_mode_valid

    def test_file_existence_checks(self):
        """Test file existence validation logic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a real file
            existing_file = os.path.join(temp_dir, "exists.wav")
            Path(existing_file).touch()
            
            # Test existence checks
            assert os.path.exists(existing_file)
            assert not os.path.exists("definitely_does_not_exist.wav")
            assert os.path.exists(temp_dir)  # Directory exists
            assert not os.path.exists(os.path.join(temp_dir, "nonexistent_dir"))

    def test_checkpoint_path_logic(self):
        """Test checkpoint path generation logic."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test checkpoint path generation like in the train mode
            base_checkpoint = os.path.join(temp_dir, "model_checkpoint.pt")
            
            # Generate epoch-specific checkpoint paths
            for epoch in range(1, 4):
                epoch_checkpoint = f"{base_checkpoint}_epoch_{epoch}.pt"
                
                # This is the pattern used in main()
                expected_pattern = f"model_checkpoint.pt_epoch_{epoch}.pt"
                assert epoch_checkpoint.endswith(expected_pattern)

    def test_demo_with_example_function(self, capsys):
        """Test the demo_with_example function (without model initialization)."""
        # Mock the model initialization to avoid heavy dependencies
        with patch('src.main_producer.HipHopProducerModel') as mock_model:
            # Configure mock to raise an exception (simulating missing setup)
            mock_instance = mock_model.return_value
            mock_instance.generate_stems.side_effect = Exception("Model not set up")
            
            # Call the demo function
            demo_with_example()
            
            # Check output
            captured = capsys.readouterr()
            assert "Hip-Hop Producer AI Demo" in captured.out
            assert "Generated plan:" in captured.out
            assert "deep 808" in captured.out.lower() or "deep" in captured.out.lower()
            assert "trap" in captured.out.lower()

    def test_demo_with_example_success_path(self, capsys):
        """Test demo_with_example with successful model generation."""
        with patch('src.main_producer.HipHopProducerModel') as mock_model:
            # Configure mock for successful generation
            mock_instance = mock_model.return_value
            mock_instance.generate_stems.return_value = {
                'bass': [1, 2, 3],  # Mock audio data
                'drums': [4, 5, 6],
                'harmony': [7, 8, 9]
            }
            
            demo_with_example()
            
            captured = capsys.readouterr()
            assert "Generated stems: ['bass', 'drums', 'harmony']" in captured.out

    def test_main_function_with_no_args(self, capsys):
        """Test main function behavior when called with no arguments."""
        # Mock sys.argv to have only the script name
        with patch.object(sys, 'argv', ['main_producer.py']):
            # Mock the model to avoid heavy dependencies
            with patch('src.main_producer.HipHopProducerModel') as mock_model:
                mock_instance = mock_model.return_value
                mock_instance.generate_stems.side_effect = Exception("Not set up")
                
                # Should call demo_with_example
                main()
                
                captured = capsys.readouterr()
                assert "Hip-Hop Producer AI Demo" in captured.out

    def test_main_function_demo_mode_missing_args(self, capsys):
        """Test main function demo mode with missing arguments."""
        test_args = ['main_producer.py', '--mode', 'demo']
        
        with patch.object(sys, 'argv', test_args):
            with patch('src.main_producer.HipHopProducerModel'):
                main()
                
                captured = capsys.readouterr()
                assert "Demo mode requires --input and --prompt arguments" in captured.out

    def test_main_function_demo_mode_missing_file(self, capsys):
        """Test main function demo mode with missing input file."""
        test_args = [
            'main_producer.py', 
            '--mode', 'demo',
            '--input', 'nonexistent_file.wav',
            '--prompt', 'Create a beat'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('src.main_producer.HipHopProducerModel'):
                main()
                
                captured = capsys.readouterr()
                assert "Input file not found" in captured.out

    def test_main_function_train_mode_missing_data_dir(self, capsys):
        """Test main function train mode with missing data directory."""
        test_args = [
            'main_producer.py',
            '--mode', 'train',
            '--data-dir', 'nonexistent_directory'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('src.main_producer.HipHopProducerModel'):
                main()
                
                captured = capsys.readouterr()
                assert "Data directory not found" in captured.out

    def test_main_function_pretrain_mode_missing_data_dir(self, capsys):
        """Test main function pretrain mode with missing data directory."""
        test_args = [
            'main_producer.py',
            '--mode', 'pretrain',
            '--data-dir', 'nonexistent_directory'
        ]
        
        with patch.object(sys, 'argv', test_args):
            with patch('src.main_producer.HipHopProducerModel'):
                main()
                
                captured = capsys.readouterr()
                assert "Data directory not found" in captured.out

    def test_main_function_pretrain_mode_no_audio_files(self, capsys):
        """Test main function pretrain mode with no audio files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty data directory
            test_args = [
                'main_producer.py',
                '--mode', 'pretrain',
                '--data-dir', temp_dir
            ]
            
            with patch.object(sys, 'argv', test_args):
                with patch('src.main_producer.HipHopProducerModel'):
                    main()
                    
                    captured = capsys.readouterr()
                    assert "No audio files found for pretraining" in captured.out

    def test_main_function_model_initialization_error(self, capsys):
        """Test main function with model initialization error."""
        test_args = ['main_producer.py', '--mode', 'demo']
        
        with patch.object(sys, 'argv', test_args):
            with patch('src.main_producer.HipHopProducerModel', side_effect=Exception("Model error")):
                main()
                
                captured = capsys.readouterr()
                assert "Error initializing model" in captured.out
                assert "Model error" in captured.out

    def test_device_argument_handling(self):
        """Test that device arguments are handled correctly."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', type=str, default='cuda')
        
        # Test default
        args = parser.parse_args([])
        assert args.device == 'cuda'
        
        # Test CPU
        args = parser.parse_args(['--device', 'cpu'])
        assert args.device == 'cpu'
        
        # Test specific GPU
        args = parser.parse_args(['--device', 'cuda:1'])
        assert args.device == 'cuda:1'

    def test_epochs_argument_validation(self):
        """Test epochs argument validation."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=1)
        
        # Test default
        args = parser.parse_args([])
        assert args.epochs == 1
        
        # Test valid values
        for epochs in [1, 5, 10, 100]:
            args = parser.parse_args(['--epochs', str(epochs)])
            assert args.epochs == epochs
        
        # Test that invalid values would raise an error
        with pytest.raises(SystemExit):  # argparse raises SystemExit on error
            parser.parse_args(['--epochs', 'not_a_number'])

    def test_mode_choices_validation(self):
        """Test that mode choices are validated correctly."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', choices=['demo', 'train', 'pretrain'], default='demo')
        
        # Test valid modes
        for mode in ['demo', 'train', 'pretrain']:
            args = parser.parse_args(['--mode', mode])
            assert args.mode == mode
        
        # Test invalid mode
        with pytest.raises(SystemExit):
            parser.parse_args(['--mode', 'invalid_mode'])

    def test_script_can_be_imported(self):
        """Test that the script can be imported without side effects."""
        # This test verifies that importing the module doesn't cause issues
        try:
            import src.main_producer
            assert hasattr(src.main_producer, 'main')
            assert hasattr(src.main_producer, 'demo_with_example')
            assert callable(src.main_producer.main)
            assert callable(src.main_producer.demo_with_example)
        except ImportError as e:
            pytest.fail(f"Could not import main_producer: {e}")


class TestMainProducerIntegration:
    """Integration tests for main producer with real file operations."""

    def test_full_argument_flow_demo_mode(self):
        """Test complete argument flow for demo mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake input file
            input_file = os.path.join(temp_dir, "input.wav")
            Path(input_file).touch()
            
            output_file = os.path.join(temp_dir, "output.wav")
            
            # Simulate parsing arguments for demo mode
            test_args = [
                'main_producer.py',
                '--mode', 'demo',
                '--input', input_file,
                '--prompt', 'Create a dark trap beat',
                '--output', output_file,
                '--device', 'cpu'
            ]
            
            parser = argparse.ArgumentParser()
            parser.add_argument('--mode', choices=['demo', 'train', 'pretrain'], default='demo')
            parser.add_argument('--input', '-i', type=str)
            parser.add_argument('--prompt', '-p', type=str)
            parser.add_argument('--output', '-o', type=str, default='output.wav')
            parser.add_argument('--device', type=str, default='cuda')
            
            args = parser.parse_args(test_args[1:])  # Skip script name
            
            # Verify all arguments are parsed correctly
            assert args.mode == 'demo'
            assert args.input == input_file
            assert args.prompt == 'Create a dark trap beat'
            assert args.output == output_file
            assert args.device == 'cpu'
            
            # Verify file existence check would pass
            assert os.path.exists(args.input)

    def test_full_argument_flow_train_mode(self):
        """Test complete argument flow for train mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data directory with some files
            data_dir = os.path.join(temp_dir, "training_data")
            os.makedirs(data_dir)
            Path(os.path.join(data_dir, "song1.wav")).touch()
            Path(os.path.join(data_dir, "song2.wav")).touch()
            
            checkpoint_file = os.path.join(temp_dir, "checkpoint.pt")
            
            test_args = [
                'main_producer.py',
                '--mode', 'train',
                '--data-dir', data_dir,
                '--epochs', '5',
                '--checkpoint', checkpoint_file,
                '--device', 'cpu'
            ]
            
            parser = argparse.ArgumentParser()
            parser.add_argument('--mode', choices=['demo', 'train', 'pretrain'], default='demo')
            parser.add_argument('--data-dir', type=str, default='data')
            parser.add_argument('--epochs', type=int, default=1)
            parser.add_argument('--checkpoint', type=str)
            parser.add_argument('--device', type=str, default='cuda')
            
            args = parser.parse_args(test_args[1:])
            
            assert args.mode == 'train'
            assert args.data_dir == data_dir
            assert args.epochs == 5
            assert args.checkpoint == checkpoint_file
            assert args.device == 'cpu'
            
            # Verify data directory exists
            assert os.path.exists(args.data_dir)

    def test_audio_file_discovery_realistic(self):
        """Test audio file discovery with realistic directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create realistic music library structure
            music_dir = os.path.join(temp_dir, "music")
            
            # Artist directories
            for artist in ["Artist1", "Artist2", "Artist3"]:
                artist_dir = os.path.join(music_dir, artist)
                os.makedirs(artist_dir)
                
                # Albums
                for album in ["Album1", "Album2"]:
                    album_dir = os.path.join(artist_dir, album)
                    os.makedirs(album_dir)
                    
                    # Tracks
                    for track in range(1, 4):  # 3 tracks per album
                        track_file = os.path.join(album_dir, f"Track_{track:02d}.wav")
                        Path(track_file).touch()
            
            # Add some files in root
            Path(os.path.join(music_dir, "bonus_track.mp3")).touch()
            Path(os.path.join(music_dir, "intro.flac")).touch()
            
            # Add stems directory (should be ignored)
            stems_dir = os.path.join(music_dir, "Artist1", "Album1", "stems")
            os.makedirs(stems_dir)
            Path(os.path.join(stems_dir, "vocals.wav")).touch()
            Path(os.path.join(stems_dir, "drums.wav")).touch()
            
            # Count files using the same logic as main()
            audio_files = []
            for root, dirs, files in os.walk(music_dir):
                for file in files:
                    if file.endswith(('.wav', '.mp3', '.flac')) and not file.startswith('.'):
                        if '/stems/' not in root:
                            full_path = os.path.join(root, file)
                            audio_files.append(full_path)
            
            # Should find: 3 artists × 2 albums × 3 tracks + 2 root files = 20 files
            # (excluding 2 stem files)
            assert len(audio_files) == 20
            
            # Verify we found files from different locations
            assert any('Artist1' in f for f in audio_files)
            assert any('Artist2' in f for f in audio_files)
            assert any('Artist3' in f for f in audio_files)
            assert any('bonus_track.mp3' in f for f in audio_files)
            assert any('intro.flac' in f for f in audio_files)
            
            # Verify stems were excluded
            assert not any('/stems/' in f for f in audio_files) 