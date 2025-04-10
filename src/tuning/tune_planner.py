import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertModel
from open_clip import create_model_and_transforms
import torchaudio
import os
from config.config_utils import load_config
import sys
import argparse

class PlannerInputProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1792, 1024)

    def forward(self, x):
        return self.fc(x)

# --- Dataset ---
class BeatPlanningDataset(Dataset):
    def __init__(self, data, clap_model, bert_tokenizer, t5_decoder, device):
        self.data = data  # List of tuples: (text_prompt, audio_path, target_plan)
        self.clap_model = clap_model
        self.bert_tokenizer = bert_tokenizer
        self.t5_decoder = t5_decoder
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text_prompt, audio_path, target_plan = self.data[idx]

        tokens = self.bert_tokenizer(text_prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            text_embed = self.bert_model(**tokens).last_hidden_state.mean(dim=1)  # [1, 1024]

        waveform, sr = torchaudio.load(audio_path)
        if sr != 48000:
            waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
        waveform = waveform.to(self.device)
        with torch.no_grad():
            audio_embed = self.clap_model.encode_audio(waveform).mean(dim=1)  # [1, 768]

        combined = torch.cat([text_embed, audio_embed], dim=1).squeeze(0)  # [1792]

        target = self.t5_tokenizer(target_plan, return_tensors="pt", padding=True, truncation=True).input_ids.squeeze(0)

        return combined, target
    
class PlannerFinetuner:
    def __init__(self, model_name: str, device: str = None, config_path: str = None, model_dir: str = None):
        self.model_name = model_name
        self.device = device
        self.config = load_config(config_path)
        self.model_dir = model_dir
        self.decoder = None
        self.tokenizer = None
        self.projector = None
        self.dataset = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        self.decoder = T5ForConditionalGeneration.from_pretrained("t5-large").to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        self.projector = PlannerInputProjector().to(self.device)

    def load_dataset(self):
        self.dataset = BeatPlanningDataset(self.data, self.clap_model, self.bert_tokenizer, self.t5_decoder, self.device)

    # --- Training ---
    def train(self, config):
        dataloader = DataLoader(self.dataset, batch_size=config.get('batch_size', 2), shuffle=True)
        optimizer = optim.AdamW(list(self.projector.parameters()) + list(self.decoder.parameters()), lr=config.get('learning_rate', 2e-5))
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        self.decoder.train()
        self.projector.train()

        for epoch in range(config.get('num_epochs', 5)):
            total_loss = 0
            for combined, target in dataloader:
                combined = combined.to(self.device)
                target = target.to(self.device)

                projected = self.projector(combined)  # [B, 1024]
                input_ids = torch.full((projected.size(0), 1), self.tokenizer.pad_token_id, dtype=torch.long).to(self.device)

                outputs = self.decoder(input_ids=input_ids, encoder_outputs=(projected.unsqueeze(1),), labels=target)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{config.get('num_epochs', 5)}, Loss: {total_loss/len(dataloader):.4f}")

# Example dataset format
# dummy_data = [
#     ("upbeat jazzy beat", "path/to/audio1.wav", "start with swing drums, then add sax loop"),
#     ("dark trap", "path/to/audio2.wav", "use 808s, detuned melody, sparse hats")
# ]
# dataset = BeatPlanningDataset(dummy_data)
# train_loop(dataset)

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


def main():
    """Main function for finetuning script"""
    args = parse_arguments()
    config = load_config(args.config)

    planner_finetuner = PlannerFinetuner(args.model, args.device, args.config, args.model_dir)
    
    # Load models
    planner_finetuner.load_model()

    # Load dataset
    planner_finetuner.load_dataset()

    # Train
    planner_finetuner.train(config)

if __name__ == "__main__":
    main() 