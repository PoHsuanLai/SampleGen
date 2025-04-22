import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, ClapModel, ClapProcessor
from transformers import MistralForCausalLM, AutoTokenizer

class Mixer(nn.Module):
    def __init__(self, decoder_model_name='mistralai/Mistral-7B-v0.1', text_model_name='bert-base-uncased'):
        super().__init__()

        # Set the device for the whole model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Text Encoder
        self.text_encoder = AutoModel.from_pretrained(text_model_name).to(self.device)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Audio Encoder (Hugging Face's CLAP implementation - more stable)
        # Use the more stable music and speech specialized model
        self.clap_model_name = "laion/larger_clap_music_and_speech"
        self.clap_processor = ClapProcessor.from_pretrained(self.clap_model_name)
        self.clap_model = ClapModel.from_pretrained(self.clap_model_name).to(self.device)
        
        # We'll use the clap_model's audio encoder
        self.audio_encoder = self.clap_model.audio_model
        
        # Freeze the audio encoder parameters to prevent gradient issues
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        
        # Also freeze the text encoder for more stable training with LoRA
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Mistral Decoder - use a single GPU (cuda:0)
        self.decoder = MistralForCausalLM.from_pretrained(
            decoder_model_name, 
            device_map="cuda:0",  # Force to a single CUDA device
            torch_dtype=torch.float16  # Use half precision to reduce memory usage
        )
        
        # Make sure at least some parameters in the decoder require gradients
        # Important: The LoRA adapter will be applied to these parameters later
        for param in self.decoder.parameters():
            param.requires_grad = True
            
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        
        # Ensure pad token is set
        if self.decoder_tokenizer.pad_token is None:
            self.decoder_tokenizer.pad_token = self.decoder_tokenizer.eos_token

        # Project combined embedding to Mistral hidden size
        text_dim = self.text_encoder.config.hidden_size
        audio_dim = 512  # CLAP audio embedding size
        mistral_hidden_dim = self.decoder.config.hidden_size

        # Improved fusion architecture with residual connections and normalization
        self.fusion_proj = nn.Sequential(
            nn.Linear(text_dim + audio_dim, mistral_hidden_dim),
            nn.LayerNorm(mistral_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mistral_hidden_dim, mistral_hidden_dim),
            nn.LayerNorm(mistral_hidden_dim),
            nn.Dropout(0.1)
        ).to(self.device)
        
        # Make sure fusion projection requires gradients
        for param in self.fusion_proj.parameters():
            param.requires_grad = True
            
        # Apply Xavier initialization to fusion layers 
        self._init_fusion_weights()
        
        # Create a stem token embedder
        self.stem_token_embedder = nn.Embedding(100, audio_dim).to(self.device)  # Support up to 100 different tokens
        
        # Dictionary to map token names to indices
        self.stem_token_dict = {}
        self.next_token_id = 0
        
        # Initialize some common stem tokens
        self._initialize_stem_tokens()
    
    def _initialize_stem_tokens(self):
        """Initialize common stem position tokens"""
        common_stems = ["vocals", "bass", "drums", "other"]
        max_positions = 10  # Support up to 10 positions per stem
        
        for stem in common_stems:
            for pos in range(1, max_positions + 1):
                token = f"[{stem.upper()}_{pos}]"
                self.stem_token_dict[token] = self.next_token_id
                self.next_token_id += 1
                
        # Add PAD token
        self.stem_token_dict["[PAD]"] = self.next_token_id
        self.next_token_id += 1
    
    def _get_token_id(self, token):
        """Get the embedding ID for a token, creating a new entry if needed"""
        if token not in self.stem_token_dict:
            # If we're seeing a new token, add it to the dictionary
            self.stem_token_dict[token] = self.next_token_id
            self.next_token_id += 1
            
            # If we've exceeded the embedding size, resize the embedding layer
            if self.next_token_id >= self.stem_token_embedder.num_embeddings:
                old_embedder = self.stem_token_embedder
                self.stem_token_embedder = nn.Embedding(
                    self.next_token_id + 50,  # Add some extra space
                    old_embedder.embedding_dim
                ).to(self.device)
                # Copy the old weights
                with torch.no_grad():
                    self.stem_token_embedder.weight[:old_embedder.num_embeddings] = old_embedder.weight
        
        return self.stem_token_dict[token]
    
    def _init_fusion_weights(self):
        """Initialize fusion layer weights carefully for stable training"""
        with torch.no_grad():
            # Initialize linear layers with Xavier initialization
            for module in self.fusion_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
            # Initialize stem token embeddings
            nn.init.normal_(self.stem_token_embedder.weight, mean=0.0, std=0.02)
    
    def to(self, device):
        """Override to method to ensure all components go to the specified device."""
        # Update the device attribute
        self.device = device
        
        # Move base module
        super().to(device)
        
        # Move encoders explicitly
        self.text_encoder = self.text_encoder.to(device)
        self.clap_model = self.clap_model.to(device)
        self.audio_encoder = self.audio_encoder.to(device)
        
        # Note: Decoder is set to use cuda:0 specifically, so we don't move it
        
        # Move fusion projection and token embedder
        self.fusion_proj = self.fusion_proj.to(device)
        self.stem_token_embedder = self.stem_token_embedder.to(device)
        
        return self
    
    def get_embedding(self, text_prompt, audio_waveform, token_identifiers=None):
        """
        Use the CLAP model to extract audio embeddings with token identifiers
        
        Args:
            text_prompt: Text prompt tensor
            audio_waveform: Audio waveform tensor [batch, stems, chunks, samples]
            token_identifiers: Optional list of stem position tokens [stems, chunks]
        
        Returns:
            text_emb: Text embedding
            audio_emb: Audio embedding with token information incorporated
        """
        # Always use the defined device
        target_device = self.device
        
        # Handle numpy arrays
        with torch.no_grad():
            # Get text embedding
            text_emb = self.text_encoder(text_prompt)
            
            # Prepare to collect audio embeddings
            batch_size = audio_waveform.size(0)
            stems = audio_waveform.size(1)
            chunks = audio_waveform.size(2)
            
            # Reshape audio_waveform to process all chunks at once
            # From [batch, stems, chunks, samples] to [batch*stems*chunks, samples]
            audio_flat = audio_waveform.view(-1, audio_waveform.size(-1))
            
            # Create inputs with processor
            inputs = self.clap_processor(audios=audio_flat, return_tensors="pt", sampling_rate=48000)
            
            # Move inputs to target device
            inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Get audio features - shape [batch*stems*chunks, embed_dim]
            audio_features = self.clap_model.get_audio_features(**inputs)
            
            # Reshape back to [batch, stems, chunks, embed_dim]
            audio_features = audio_features.view(batch_size, stems, chunks, -1)
            
            # If token identifiers are provided, incorporate them
            if token_identifiers is not None:
                # Convert token strings to embedding IDs
                token_embeds = []
                
                for stem_idx in range(stems):
                    chunk_embeds = []
                    for chunk_idx in range(chunks):
                        # Get the token for this stem and chunk position
                        if stem_idx < len(token_identifiers) and chunk_idx < len(token_identifiers[stem_idx]):
                            token = token_identifiers[stem_idx][chunk_idx]
                        else:
                            # Default token if missing
                            token = f"[UNKNOWN]"
                        
                        # Get the token ID and embedding
                        token_id = self._get_token_id(token)
                        token_embed = self.stem_token_embedder(torch.tensor([token_id], device=target_device))
                        chunk_embeds.append(token_embed)
                    
                    # Stack chunk embeddings
                    if chunk_embeds:
                        stem_embed = torch.cat(chunk_embeds, dim=0)
                        token_embeds.append(stem_embed)
                
                # Stack stem embeddings and add batch dimension
                if token_embeds:
                    token_embeds = torch.stack(token_embeds, dim=0).unsqueeze(0)
                    
                    # Broadcast to match batch size if needed
                    if token_embeds.size(0) < batch_size:
                        token_embeds = token_embeds.expand(batch_size, -1, -1, -1)
                    
                    # Add token embeddings to audio features (residual connection)
                    audio_features = audio_features + token_embeds
            
            # Average across stems and chunks to get a single embedding per batch
            audio_features = audio_features.mean(dim=[1, 2])

        return text_emb, audio_features

    def forward(self, text_prompt=None, audio_waveform=None, token_identifiers=None, target_tool_tokens=None, 
              input_ids=None, attention_mask=None, tool_input_ids=None, tool_attention_mask=None, labels=None):
        """
        Forward pass for the Mixer model.
        
        Args:
            text_prompt: Raw text prompt (will be tokenized)
            audio_waveform: Audio waveform data
            token_identifiers: List of token identifiers for stems and positions
            target_tool_tokens: Raw target tokens (will be tokenized)
            input_ids: Pre-tokenized input IDs (if text_prompt is None)
            attention_mask: Pre-tokenized attention mask (if text_prompt is None)
            tool_input_ids: Pre-tokenized tool input IDs (if target_tool_tokens is None)
            tool_attention_mask: Pre-tokenized tool attention mask (if target_tool_tokens is None)
            labels: Optional direct labels for computing loss (will override internal label creation)
        """
        # Always use the defined device
        target_device = self.device

        # Encode audio using our stable wrapper
        if text_prompt is None:
            raise ValueError("text_prompt must be provided")
        
        if audio_waveform is None:
            raise ValueError("audio_waveform must be provided")
        
        # Ensure text_prompt is on the correct device
        if isinstance(text_prompt, torch.Tensor) and text_prompt.device != target_device:
            text_prompt = text_prompt.to(target_device)

        # Ensure audio_waveform is on the correct device
        if isinstance(audio_waveform, torch.Tensor) and audio_waveform.device != target_device:
            audio_waveform = audio_waveform.to(target_device)
            
        # Get audio embedding using the updated method that handles token identifiers
        with torch.no_grad():  # Use no_grad to prevent gradient propagation through the CLAP model
            text_emb, audio_emb = self.get_embedding(text_prompt, audio_waveform, token_identifiers)
        

        # Apply L2 normalization to both embeddings
        text_emb = nn.functional.normalize(text_emb, p=2, dim=-1)
        audio_emb = nn.functional.normalize(audio_emb, p=2, dim=-1)
        
        # Combine embeddings
        fused = torch.cat([text_emb, audio_emb], dim=-1)
        
        # Apply fusion projection with the improved architecture
        condition_emb = self.fusion_proj(fused)  # [batch, mistral_hidden_dim]
        
        # Additional normalization for stability
        condition_emb = torch.clamp(condition_emb, min=-1.0, max=1.0)
        
        # Prepare the inputs for the decoder
        # Use the features from the fusion module as the decoder input
        outputs = self.decoder(
            inputs_embeds=condition_emb.unsqueeze(1),  # Add sequence dimension
            attention_mask=torch.ones(condition_emb.size(0), 1, dtype=torch.long, device=target_device),
            labels=labels,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False  # Disable KV cache during training for gradient checkpointing
        )
        
        return outputs
