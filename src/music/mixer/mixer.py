import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer, MistralForCausalLM

# Import and patch CLAP module
import laion_clap
import laion_clap.training.data as clap_data
from laion_clap import CLAP_Module

# Add numpy types to torch's safe globals
torch.serialization.add_safe_globals([np.ndarray, np.dtype, np.generic, 
                                     np.float32, np.float64, np.int64, 
                                     np.int32, np.uint8, np.bool_])

# Create patched versions of CLAP data conversion functions
original_float32_to_int16 = clap_data.float32_to_int16
original_int16_to_float32 = clap_data.int16_to_float32

def patched_float32_to_int16(x):
    if isinstance(x, torch.Tensor):
        return (x * 32767.).to(torch.int16)
    return original_float32_to_int16(x)

def patched_int16_to_float32(x):
    if isinstance(x, torch.Tensor):
        return (x.to(torch.float32) / 32767.)
    return original_int16_to_float32(x)

# Apply patches
clap_data.float32_to_int16 = patched_float32_to_int16
clap_data.int16_to_float32 = patched_int16_to_float32

class Mixer(nn.Module):
    def __init__(self, decoder_model_name='mistralai/Mistral-7B-v0.3', text_model_name='bert-base-uncased'):
        super().__init__()

        # Text Encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Audio Encoder (CLAP)
        self.audio_encoder = CLAP_Module(enable_fusion=False)
        
        # Load CLAP weights with patched torch.load
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
            
        try:
            # Apply patch
            torch.load = patched_torch_load
            self.audio_encoder.load_ckpt()  # loads pretrained CLAP weights
        finally:
            # Restore original function
            torch.load = original_torch_load

        # Mistral Decoder - distribute across available GPUs
        self.decoder = MistralForCausalLM.from_pretrained(
            decoder_model_name, 
            device_map="auto",  # Auto-distribute across available GPUs
            torch_dtype=torch.float16  # Use half precision to reduce memory usage
        )
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        
        # Ensure pad token is set
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        # Project combined embedding to Mistral hidden size
        text_dim = self.text_encoder.config.hidden_size
        audio_dim = 512  # CLAP audio embedding size
        mistral_hidden_dim = self.decoder.config.hidden_size

        self.fusion_proj = nn.Linear(text_dim + audio_dim, mistral_hidden_dim)
    
    def get_audio_embedding(self, audio_waveform):
        """Wrapper for CLAP audio embedding that handles tensor conversion issues"""
        # If input is a torch tensor, convert to numpy
        orig_device = None
        if isinstance(audio_waveform, torch.Tensor):
            orig_device = audio_waveform.device
            audio_waveform = audio_waveform.cpu().numpy()
        
        # Process with CLAP
        try:
            # Apply the patches again just to be sure
            clap_data.float32_to_int16 = patched_float32_to_int16
            clap_data.int16_to_float32 = patched_int16_to_float32
            
            # Get audio embedding using numpy arrays
            audio_emb = self.audio_encoder.get_audio_embedding_from_data(audio_waveform)
            
            # Convert back to tensor
            if orig_device is not None:
                audio_emb = torch.tensor(audio_emb, device=orig_device)
                
            return audio_emb.squeeze(1)  # [batch, 512]
        except Exception as e:
            print(f"Error in audio processing: {e}")
            # Fallback: return a zero embedding of the correct shape
            return torch.zeros((audio_waveform.shape[0], 512), device=orig_device)

    def forward(self, text_prompt, audio_waveform, target_tool_tokens=None):
        # Primary device for non-distributed components
        text_device = next(self.text_encoder.parameters()).device
        
        # Encode text
        text_inputs = self.text_tokenizer(text_prompt, return_tensors='pt', padding=True, truncation=True).to(text_device)
        text_emb = self.text_encoder(**text_inputs).last_hidden_state[:, 0]  # [batch, dim]

        # Encode audio using our patched wrapper
        audio_emb = self.get_audio_embedding(audio_waveform)
        if audio_emb.device != text_device:
            audio_emb = audio_emb.to(text_device)

        # Combine and project
        fused = torch.cat([text_emb, audio_emb], dim=-1)
        mistral_cond_emb = self.fusion_proj(fused).unsqueeze(1)  # [batch, 1, mistral_hidden_dim]

        # Prepare decoder input
        if target_tool_tokens is not None:
            target_inputs = self.decoder_tokenizer(target_tool_tokens, return_tensors='pt', padding=True, truncation=True).to(text_device)
            input_ids = target_inputs.input_ids
            attention_mask = target_inputs.attention_mask

            # The decoder model handles its own device placement with device_map="auto"
            outputs = self.decoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=mistral_cond_emb,
                encoder_attention_mask=torch.ones(mistral_cond_emb.shape[:2], dtype=torch.bool).to(text_device),
            )
            return outputs
        else:
            # For generation
            input_ids = self.decoder_tokenizer(self.decoder_tokenizer.bos_token, return_tensors='pt').input_ids.to(text_device)
            outputs = self.decoder.generate(
                input_ids=input_ids,
                max_length=64,
                encoder_hidden_states=mistral_cond_emb,
                encoder_attention_mask=torch.ones(mistral_cond_emb.shape[:2], dtype=torch.bool).to(text_device),
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
            )
            return self.decoder_tokenizer.decode(outputs[0], skip_special_tokens=True)
