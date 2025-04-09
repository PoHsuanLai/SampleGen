import torch
import torchaudio
from transformers import BertTokenizer, BertModel, T5Tokenizer, T5ForConditionalGeneration
from open_clip import create_model_and_transforms

# --- Load Models ---

# Audio: CLAP (HTSAT-large)
def load_clap_large():
    model, _, preprocess = create_model_and_transforms('HTSAT-large', pretrained='laion2b_sdat')
    model.eval()
    return model, preprocess

# Text: BERT-large
bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
bert_model = BertModel.from_pretrained("bert-large-uncased")
bert_model.eval()

# Planner: T5-large
t5_tokenizer = T5Tokenizer.from_pretrained("t5-large")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-large")
t5_model.eval()

# Projector: 1024 (BERT) + 768 (CLAP) -> 1024 (T5)
class PlannerInputProjector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1792, 1024)

    def forward(self, x):
        return self.fc(x)

projector = PlannerInputProjector()
projector.eval()

# --- Utility Functions ---

def encode_audio(audio_path, clap_model):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 48000:
        waveform = torchaudio.transforms.Resample(sr, 48000)(waveform)
    with torch.no_grad():
        audio_embed = clap_model.encode_audio(waveform).mean(dim=1)  # [1, 768]
    return audio_embed

def encode_text(prompt):
    tokens = bert_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = bert_model(**tokens)
    return output.last_hidden_state.mean(dim=1)  # [1, 1024]

def generate_beat_plan(audio_path, text_prompt, clap_model):
    audio_embed = encode_audio(audio_path, clap_model)       # [1, 768]
    text_embed = encode_text(text_prompt)                    # [1, 1024]
    combined = torch.cat([text_embed, audio_embed], dim=1)   # [1, 1792]
    projected = projector(combined)                          # [1, 1024]

    # Feed a text input (can be conditioned richer in the future)
    input_str = "audio_text_plan: " + text_prompt
    input_ids = t5_tokenizer(input_str, return_tensors="pt").input_ids

    with torch.no_grad():
        output_ids = t5_model.generate(input_ids=input_ids, max_length=128)
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
