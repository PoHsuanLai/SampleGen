from .prompts import get_hip_hop_prompts, get_custom_prompts
from .config.config_utils import load_config, get_generator_config, get_style_config, get_models_dir
from ..music.generator.models import OtherGenerator, BassGenerator, DrumGenerator
import os
import sys
import torch
import argparse
from datasets import load_dataset, Dataset
from transformers import (
    Trainer, 
    TrainingArguments,
    AutoProcessor
)
import numpy as np
from typing import Optional, Dict, Any, List

class Finetuner:
    def __init__(self, model_name: str, device: str = None, config_path: str = None, model_dir: str = None):
        self.model_name = model_name
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.model = None
        self.processor = None
        self.generator = None
        
        # Load configuration
        self.config = load_config(config_path)
        self.generator_config = get_generator_config(self.config, model_name)
        
        # Set models directory
        if model_dir is None:
            self.model_dir = get_models_dir(os.getcwd(), self.config)
        else:
            self.model_dir = model_dir
            
        # Ensure the models directory exists
        os.makedirs(self.model_dir, exist_ok=True)

    def load_model(self):
        # Get the model name from config if available
        config_model_name = self.generator_config.get('model_name', None)
        
        if self.model_name == "other":
            self.generator = OtherGenerator(
                model_name=config_model_name or "facebook/musicgen-other",
                device=self.device,
                model_dir=self.model_dir
            )
        elif self.model_name == "bass":
            self.generator = BassGenerator(
                model_name=config_model_name or "facebook/musicgen-small", 
                device=self.device,
                model_dir=self.model_dir
            )
        elif self.model_name == "drums":
            self.generator = DrumGenerator(
                model_name=config_model_name or "facebook/musicgen-small", 
                device=self.device,
                model_dir=self.model_dir
            )
        
        # Extract the actual model and processor from the generator wrapper
        self.processor = self.generator.processor
        self.model = self.generator.model

    def preprocess_dataset(self, dataset: Dataset, hip_hop_prompts: List[str]) -> Dataset:
        """
        Preprocess the dataset for finetuning on hip-hop beats
        
        Args:
            dataset: The input dataset containing audio samples
            hip_hop_prompts: List of hip-hop related text prompts to use
            
        Returns:
            Processed dataset ready for training
        """
        def preprocess_function(examples):
            # Get audio from dataset
            audio_arrays = examples["audio"] if "audio" in examples else examples["audio_array"]
            
            # Ensure we have raw audio arrays
            if isinstance(audio_arrays[0], dict) and "array" in audio_arrays[0]:
                audio_arrays = [audio["array"] for audio in audio_arrays]
            
            # Sample rate in the dataset
            sample_rate = examples.get("sampling_rate", 32000)
            
            # Assign hip-hop related prompts to each example
            import random
            text = [random.choice(hip_hop_prompts) for _ in range(len(audio_arrays))]
            
            # Process inputs using the model's processor
            inputs = self.processor(
                text=text,
                audio=audio_arrays,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            
            return inputs
        
        # Apply preprocessing to dataset
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return processed_dataset

    def finetune(self, dataset_path: str, output_path: str, 
                 num_epochs: int = None, 
                 batch_size: int = None,
                 learning_rate: float = None,
                 hip_hop_style: str = "modern",
                 custom_prompts_path: str = None):
        """
        Finetune a music generator model on hip-hop beats
        
        Args:
            dataset_path: Path to the dataset of audio samples
            output_path: Directory to save the finetuned model
            num_epochs: Number of training epochs (None to use config)
            batch_size: Batch size for training (None to use config)
            learning_rate: Learning rate for training (None to use config)
            hip_hop_style: Style of hip-hop to focus on (modern, classic, trap, etc.)
            custom_prompts_path: Path to custom prompts file (None to use default prompts)
        """
        if self.model is None:
            self.load_model()

        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Get style configuration
        style_config = get_style_config(self.config, hip_hop_style)
        
        # Get prompts based on configuration
        prompt_config = self.config.get('prompts', {})
        use_default_prompts = prompt_config.get('use_default_prompts', True)
        
        hip_hop_prompts = []
        
        # Load custom prompts if provided or specified in config
        config_prompt_file = prompt_config.get('custom_prompt_file')
        custom_file = custom_prompts_path or config_prompt_file
        
        if custom_file:
            custom_prompts = get_custom_prompts(custom_file, self.model_name)
            hip_hop_prompts.extend(custom_prompts)
            print(f"Loaded {len(custom_prompts)} custom prompts from {custom_file}")
        
        # Use default prompts if no custom prompts or if explicitly enabled
        if not hip_hop_prompts or use_default_prompts:
            default_prompts = get_hip_hop_prompts(self.model_name, hip_hop_style)
            hip_hop_prompts.extend(default_prompts)
            print(f"Using {len(default_prompts)} default prompts for {self.model_name} generator ({hip_hop_style} style)")
        
        print(f"Loading dataset from {dataset_path}...")
        # Load dataset
        dataset = load_dataset(dataset_path)
        train_dataset = dataset["train"] if "train" in dataset else dataset
        
        # Preprocess the dataset
        print("Preprocessing dataset...")
        processed_dataset = self.preprocess_dataset(train_dataset, hip_hop_prompts)
        
        # Get training parameters from config with fallbacks to provided values or defaults
        config_epochs = self.generator_config.get('num_epochs', 3)
        config_batch_size = self.generator_config.get('batch_size', 4)
        config_lr = self.generator_config.get('learning_rate', 5e-5)
        
        # Use provided values if available, otherwise use config values
        actual_epochs = num_epochs if num_epochs is not None else config_epochs
        actual_batch_size = batch_size if batch_size is not None else config_batch_size
        actual_lr = learning_rate if learning_rate is not None else config_lr
        
        # Get other training parameters from config
        weight_decay = self.generator_config.get('weight_decay', 0.01)
        gradient_accumulation = self.generator_config.get('gradient_accumulation_steps', 4)
        fp16 = self.generator_config.get('fp16', torch.cuda.is_available())
        save_strategy = self.generator_config.get('save_strategy', 'epoch')
        save_total_limit = self.generator_config.get('save_total_limit', 2)
        logging_steps = self.generator_config.get('logging_steps', 10)
        report_to = self.generator_config.get('report_to', ["tensorboard"])
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=actual_epochs,
            per_device_train_batch_size=actual_batch_size,
            learning_rate=actual_lr,
            weight_decay=weight_decay,
            logging_dir=os.path.join(output_path, "logs"),
            logging_steps=logging_steps,
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
            fp16=fp16,
            gradient_accumulation_steps=gradient_accumulation,
            report_to=report_to,
        )
        
        # Initialize the trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset,
        )
        
        # Start training
        print(f"Starting finetuning {self.model_name} generator for {hip_hop_style} hip-hop...")
        print(f"Training parameters: epochs={actual_epochs}, batch_size={actual_batch_size}, lr={actual_lr}")
        trainer.train()
        
        # Save the model
        model_save_path = os.path.join(output_path, f"finetuned_{self.model_name}")
        self.model.save_pretrained(model_save_path)
        self.processor.save_pretrained(model_save_path)
        
        print(f"Finetuning complete! Model saved to {model_save_path}")
        
        return model_save_path
    
    def evaluate_model(self, test_dataset_path: str, num_samples: int = 5) -> Dict[str, Any]:
        """
        Evaluate the finetuned model on a test dataset
        
        Args:
            test_dataset_path: Path to test dataset
            num_samples: Number of samples to generate for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        # Load test dataset
        test_dataset = load_dataset(test_dataset_path)
        test_dataset = test_dataset["test"] if "test" in test_dataset else test_dataset
        
        # Get sample prompts
        sample_prompts = get_hip_hop_prompts(self.model_name, "modern")[:num_samples]
        
        # Original generator for comparison
        original_generator = None
        if self.model_name == "other":
            original_generator = OtherGenerator(device=self.device, model_dir=self.model_dir)
        elif self.model_name == "bass":
            original_generator = BassGenerator(device=self.device, model_dir=self.model_dir)
        elif self.model_name == "drums":
            original_generator = DrumGenerator(device=self.device, model_dir=self.model_dir)
        
        # Create a wrapper for the finetuned model
        finetuned_generator = original_generator.__class__(device=self.device, model_dir=self.model_dir)
        finetuned_generator.model = self.model
        finetuned_generator.processor = self.processor
        
        # Generate samples with both models
        results = {
            "original_samples": [],
            "finetuned_samples": [],
            "prompts": sample_prompts
        }
        
        for prompt in sample_prompts:
            # Generate with original model
            original_audio = original_generator.generate(prompt=prompt, duration=5.0)
            
            # Generate with finetuned model
            finetuned_audio = finetuned_generator.generate(prompt=prompt, duration=5.0)
            
            results["original_samples"].append(original_audio)
            results["finetuned_samples"].append(finetuned_audio)
        
        return results


def parse_arguments():
    """Parse command line arguments for finetuning"""
    parser = argparse.ArgumentParser(description="Finetune music generators for hip-hop production")
    
    # Required arguments
    parser.add_argument("--model", type=str, required=True, choices=["melody", "harmony", "bass", "drums"],
                        help="Type of generator to finetune")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path to the dataset for finetuning")
    parser.add_argument("--output", type=str, required=True,
                        help="Directory to save the finetuned model")
    
    # Optional arguments
    parser.add_argument("--style", type=str, default="modern", 
                        choices=["modern", "classic", "trap", "lofi"],
                        help="Hip-hop style to focus training on")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for training (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate for training (overrides config)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom config file")
    parser.add_argument("--custom-prompts", type=str, default=None,
                        help="Path to custom prompts file")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model after training")
    parser.add_argument("--test-dataset", type=str, default=None,
                        help="Path to test dataset for evaluation (required if --evaluate is set)")
    
    return parser.parse_args()

def finetune_model(args):
    """Finetune a model with the specified parameters"""
    print(f"Starting finetuning process for {args.model} generator")
    print(f"Hip-hop style: {args.style}")
    
    # Load configuration to display info
    config = load_config(args.config)
    generator_config = get_generator_config(config, args.model)
    
    # Show configuration info
    if generator_config:
        model_name = generator_config.get('model_name', 'default model')
        description = generator_config.get('description', '')
        print(f"Using {model_name} - {description}")
    
    # Initialize finetuner
    finetuner = Finetuner(model_name=args.model, device=args.device, config_path=args.config)
    
    # Finetune model
    model_path = finetuner.finetune(
        dataset_path=args.dataset,
        output_path=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hip_hop_style=args.style,
        custom_prompts_path=args.custom_prompts
    )
    
    print(f"Finetuning complete! Model saved to: {model_path}")
    
    # Evaluate if requested
    if args.evaluate:
        if args.test_dataset is None:
            print("Error: --test-dataset must be provided when --evaluate is set")
            sys.exit(1)
            
        print(f"Evaluating finetuned {args.model} generator...")
        eval_results = finetuner.evaluate_model(args.test_dataset)
        
        # Save some sample outputs
        samples_dir = os.path.join(args.output, "evaluation_samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        for i, (orig, finetuned) in enumerate(zip(eval_results["original_samples"], 
                                                eval_results["finetuned_samples"])):
            prompt = eval_results["prompts"][i]
            
            # Save original and finetuned samples
            orig_path = os.path.join(samples_dir, f"original_sample_{i}.wav")
            finetuned_path = os.path.join(samples_dir, f"finetuned_sample_{i}.wav")
            
            finetuner.generator.save_audio(orig, orig_path)
            finetuner.generator.save_audio(finetuned, finetuned_path)
            
            # Save prompt text
            with open(os.path.join(samples_dir, f"prompt_{i}.txt"), "w") as f:
                f.write(prompt)
                
        print(f"Evaluation samples saved to {samples_dir}")

def main():
    """Main function for finetuning script"""
    args = parse_arguments()
    finetune_model(args)

if __name__ == "__main__":
    main() 