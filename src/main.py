"""
Main entry point for the Unified Hip-Hop Producer AI system.
Uses the unified model that can both request generators and generate Faust scripts.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path if running as script
if __name__ == "__main__":
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    # Add the parent of src (the project root) to sys.path
    project_root = script_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# Try relative imports first (when running as module), fall back to absolute imports
try:
    from .models.producer import UnifiedProducerModel
    from .training.trainer import UnifiedProducerTrainer, FaustScriptTrainer
except ImportError:
    # Fallback for when running as script
    from src.models.producer import UnifiedProducerModel
    from src.training.trainer import UnifiedProducerTrainer, FaustScriptTrainer


def main():
    parser = argparse.ArgumentParser(description='Unified Hip-Hop Producer AI - Stem Generation and Faust Mixing')
    
    parser.add_argument('--mode', choices=['demo', 'train', 'faust-train'], default='demo',
                       help='Mode to run: demo, train, or faust-train')
    
    parser.add_argument('--input', '-i', type=str,
                       help='Path to input audio file (for demo mode)')
    
    parser.add_argument('--prompt', '-p', type=str,
                       help='Text prompt describing desired output')
    
    parser.add_argument('--output', '-o', type=str, default='output.wav',
                       help='Path to save generated audio')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing training data')
    
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of training epochs')
    
    parser.add_argument('--checkpoint', type=str,
                       help='Path to load/save model checkpoint')
    
    parser.add_argument('--text-model', type=str, default='bert-base-uncased',
                       help='Text encoder model name')
    
    parser.add_argument('--decoder-model', type=str, default='mistralai/Mistral-7B-v0.1',
                       help='Text generation model name')
    
    args = parser.parse_args()
    
    # Initialize the unified Hip-Hop Producer model
    print("Initializing Unified Hip-Hop Producer model...")
    try:
        model = UnifiedProducerModel(
            device=args.device,
            text_model_name=args.text_model,
            decoder_model_name=args.decoder_model
        )
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        print("This is expected if you don't have all the required models downloaded.")
        print("The model will download them automatically on first run.")
        return
    
    if args.mode == 'demo':
        if not args.input or not args.prompt:
            print("Demo mode requires --input and --prompt arguments")
            return
            
        if not os.path.exists(args.input):
            print(f"Input file not found: {args.input}")
            return
        
        # Initialize trainer for demo
        trainer = UnifiedProducerTrainer(model, args.data_dir, args.device)
        
        # Load checkpoint if provided
        if args.checkpoint and os.path.exists(args.checkpoint):
            trainer.load_checkpoint(args.checkpoint)
        
        # Run demo
        trainer.demo_full_pipeline(args.input, args.prompt, args.output)
    
    elif args.mode == 'train':
        if not os.path.exists(args.data_dir):
            print(f"Data directory not found: {args.data_dir}")
            return
            
        # Initialize trainer
        trainer = UnifiedProducerTrainer(model, args.data_dir, args.device)
        
        # Load checkpoint if provided
        if args.checkpoint and os.path.exists(args.checkpoint):
            trainer.load_checkpoint(args.checkpoint)
        
        # Training loop
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            results = trainer.train_epoch()
            print(f"Planning loss: {results['planning_loss']:.4f}")
            print(f"Quality loss: {results['quality_loss']:.4f}")
            print(f"Faust loss: {results['faust_loss']:.4f}")
            print(f"Total loss: {results['total_loss']:.4f}")
            
            # Save checkpoint
            if args.checkpoint:
                trainer.save_checkpoint(f"{args.checkpoint}_epoch_{epoch + 1}.pt")
    
    elif args.mode == 'faust-train':
        # Specialized Faust script training
        if not os.path.exists(args.data_dir):
            print(f"Data directory not found: {args.data_dir}")
            return
        
        # Collect audio files for Faust training
        audio_files = []
        for root, dirs, files in os.walk(args.data_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')) and not file.startswith('.'):
                    # Skip stem files, look for main audio files
                    if '/stems/' not in root:
                        full_path = os.path.join(root, file)
                        audio_files.append(full_path)
        
        if not audio_files:
            print("No audio files found for Faust training")
            return
        
        print(f"Found {len(audio_files)} files for Faust script training")
        
        # Initialize Faust trainer
        faust_trainer = FaustScriptTrainer(model, audio_files, args.device)
        
        # Training loop
        for epoch in range(args.epochs):
            print(f"\nFaust Training Epoch {epoch + 1}/{args.epochs}")
            results = faust_trainer.train_epoch()
            print(f"Faust script loss: {results['faust_script_loss']:.4f}")


def demo_with_example():
    """
    Demo function that shows how to use the unified system programmatically.
    """
    print("=== Unified Hip-Hop Producer AI Demo ===")
    
    # Initialize model
    model = UnifiedProducerModel(device='cuda')
    
    # Example usage without actual files (for demonstration)
    text_prompt = "Create a dark trap beat with heavy 808s and atmospheric pads"
    
    print(f"Prompt: {text_prompt}")
    
    # Simulate the planning phase
    plan = model.plan_production(text_prompt=text_prompt)
    print(f"Generated plan: {plan}")
    
    # Generate stems (this would work with actual model)
    try:
        generated_stems = model.generate_stems(plan, duration=5.0)
        print(f"Generated stems: {list(generated_stems.keys())}")
        
        # Generate Faust script
        if generated_stems:
            faust_script = model.generate_faust_script(generated_stems, text_prompt)
            print(f"Generated Faust script:\n{faust_script}")
        
    except Exception as e:
        print(f"Generation would work with proper setup: {e}")
    
    print("\nThis demo shows the unified architecture. To run with actual audio:")
    print("python src/main_unified_producer.py --mode demo --input your_song.wav --prompt 'your prompt'")
    print("\nTo train the model:")
    print("python src/main_unified_producer.py --mode train --data-dir your_data_dir --epochs 5")
    print("\nTo train Faust script generation:")
    print("python src/main_unified_producer.py --mode faust-train --data-dir your_data_dir --epochs 3")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments provided, run the demo
        demo_with_example()
        print("\n" + "="*50)
    else:
        main() 