import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, ClapModel, ClapProcessor
from transformers import MistralForCausalLM, AutoTokenizer
import numpy as np

class Mixer(nn.Module):
    def __init__(self, decoder_model_name='mistralai/Mistral-7B-v0.1', text_model_name='bert-base-uncased', multi_gpu=False):
        super().__init__()

        # Set devices based on multi_gpu flag
        if multi_gpu and torch.cuda.device_count() > 1:
            print(f"Using multi-GPU setup with {torch.cuda.device_count()} GPUs")
            self.encoder_device = torch.device("cuda:0")
            self.decoder_device = torch.device("cuda:1")
        else:
            print("Using single GPU setup")
            self.encoder_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.decoder_device = self.encoder_device
            
        self.main_device = self.encoder_device  # For compatibility
        self.device = self.main_device  # For backward compatibility

        # Text Encoder on first GPU
        self.text_encoder = AutoModel.from_pretrained(text_model_name).to(self.encoder_device)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # Audio Encoder (Hugging Face's CLAP implementation - more stable) on first GPU
        # Use the more stable music and speech specialized model
        self.clap_model_name = "laion/larger_clap_music_and_speech"
        self.clap_processor = ClapProcessor.from_pretrained(self.clap_model_name)
        self.clap_model = ClapModel.from_pretrained(self.clap_model_name).to(self.encoder_device)
        
        # We'll use the clap_model's audio encoder
        self.audio_encoder = self.clap_model.audio_model
        
        # Freeze the audio encoder parameters to prevent gradient issues
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        
        # Also freeze the text encoder for more stable training with LoRA
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Mistral Decoder - distributed based on device setup
        print(f"Loading Mistral model on {self.decoder_device}...")
        
        # Set up device map according to multi-GPU configuration
        if self.decoder_device != self.encoder_device:
            # Multi-GPU: explicit placement on second GPU
            device_map = str(self.decoder_device)
        else:
            # Single-GPU: use 'auto' for automatic placement
            device_map = "auto"
            
        print(f"Using device_map: {device_map}")
            
        self.decoder = MistralForCausalLM.from_pretrained(
            decoder_model_name, 
            device_map=device_map,
            torch_dtype=torch.float16
        )
        
        # Store the model's dtype for consistent handling in forward pass
        self.model_dtype = next(self.decoder.parameters()).dtype
        print(f"Model using dtype: {self.model_dtype}")
        
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
        # Place fusion layers on same GPU as encoder for efficient transfer
        self.fusion_proj = nn.Sequential(
            nn.Linear(text_dim + audio_dim, mistral_hidden_dim),
            nn.LayerNorm(mistral_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mistral_hidden_dim, mistral_hidden_dim),
            nn.LayerNorm(mistral_hidden_dim),
            nn.Dropout(0.1)
        ).to(self.encoder_device)
        
        # Make sure fusion projection requires gradients
        for param in self.fusion_proj.parameters():
            param.requires_grad = True
            
        # Create a stem token embedder - MUST BE CREATED BEFORE _init_fusion_weights is called
        self.stem_token_embedder = nn.Embedding(100, audio_dim).to(self.encoder_device)  # Support up to 100 different tokens
        
        # Dictionary to map token names to indices
        self.stem_token_dict = {}
        self.next_token_id = 0
        
        # Initialize some common stem tokens
        self._initialize_stem_tokens()
            
        # Apply Xavier initialization to fusion layers 
        self._init_fusion_weights()
    
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
    
    def safe_normalize(self, x):
        """Safely normalize a tensor along the last dimension, handling zero vectors."""
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        if (norm == 0).any():
            print("Zero vector detected in embedding")
        return torch.where(norm > 0, x / norm, torch.zeros_like(x))
    
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
        # Always use encoder device for embedding extraction
        target_device = self.encoder_device
        
        # Handle numpy arrays
        with torch.no_grad():
            # Ensure text_prompt is on the encoder device
            if text_prompt.device != target_device:
                text_prompt = text_prompt.to(target_device)
                
            # Ensure audio_waveform is on the encoder device 
            if audio_waveform.device != target_device:
                audio_waveform = audio_waveform.to(target_device)
                
            # Get text embedding
            text_encoder_output = self.text_encoder(text_prompt)
            # Extract the tensor from the model output
            if hasattr(text_encoder_output, 'last_hidden_state'):
                # Get the [CLS] token embedding or average all token embeddings
                text_emb = text_encoder_output.last_hidden_state.mean(dim=1)  # Average pooling
            elif hasattr(text_encoder_output, 'pooler_output'):
                # Use the pooler output if available
                text_emb = text_encoder_output.pooler_output
            else:
                raise ValueError("Cannot extract embedding from text encoder output")
            
            # Debug text embedding
            print(f"DEBUG: Text embedding shape: {text_emb.shape}, device: {text_emb.device}")
            if torch.isnan(text_emb).any():
                print("WARNING: NaN values detected in text embedding")
            
            # Prepare to collect audio embeddings
            batch_size = audio_waveform.size(0)
            stems = audio_waveform.size(1)
            chunks = audio_waveform.size(2)
            
            # Debug audio shape
            print(f"DEBUG: Audio waveform shape: {audio_waveform.shape}, device: {audio_waveform.device}")
            
            # Create a mask to identify padded chunks (zeros)
            # Shape: [batch, stems, chunks]
            valid_mask = (audio_waveform.abs().sum(dim=-1) > 1e-5).float()
            
            # Reshape audio_waveform to process all chunks at once
            # From [batch, stems, chunks, samples] to [batch*stems*chunks, samples]
            audio_flat = audio_waveform.view(-1, audio_waveform.size(-1))
            
            # Debug flattened audio
            print(f"DEBUG: Flattened audio shape: {audio_flat.shape}")
            
            # Move audio_flat to CPU before processing with clap_processor
            audio_flat_cpu = audio_flat.cpu()
            
            # Process in smaller batches to avoid memory issues and use the full CLAP pipeline
            # instead of our custom implementation which had dimension issues
            safe_batch_size = 32  # Process in small batches
            total_samples = audio_flat_cpu.shape[0]
            all_audio_features = []
            
            for i in range(0, total_samples, safe_batch_size):
                end_idx = min(i + safe_batch_size, total_samples)
                batch_audio = audio_flat_cpu[i:end_idx]
                
                try:
                    # Use the processor's standard pipeline but with smaller batches
                    # This ensures all tensor dimensions are correct
                    batch_inputs = []
                    for audio in batch_audio:
                        # Convert to numpy array
                        audio_np = audio.numpy()
                        
                        # Check for NaN or Inf values in audio samples
                        if np.isnan(audio_np).any():
                            print(f"WARNING: NaN values in audio sample at batch {i}:{end_idx}")
                            # Replace NaN with zeros for stability
                            audio_np = np.nan_to_num(audio_np)
                        
                        # Use processor for one sample at a time (memory efficient)
                        inputs = self.clap_processor(
                            audios=audio_np,
                            return_tensors="pt",
                            sampling_rate=48000
                        )
                        
                        batch_inputs.append(inputs)
                    
                    # Process each sample through the CLAP model
                    batch_features = []
                    for inputs in batch_inputs:
                        # Move inputs to target device
                        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v 
                                for k, v in inputs.items()}
                        
                        # Get audio features
                        features = self.clap_model.get_audio_features(**inputs)
                        
                        # Check feature values
                        if torch.isnan(features).any():
                            print("WARNING: NaN values in CLAP features")
                            # Replace NaN with zeros
                            features = torch.nan_to_num(features)
                        
                        batch_features.append(features)
                    
                    # Stack the batch
                    if batch_features:
                        batch_features_tensor = torch.cat(batch_features, dim=0)
                        all_audio_features.append(batch_features_tensor)
                
                except Exception as e:
                    # If there's an error, print detailed information for debugging
                    print(f"Error processing batch {i}-{end_idx}: {str(e)}")
                    print(f"Batch audio shape: {batch_audio.shape}")
                    # Continue with the next batch instead of failing completely
                    continue
            
            # Make sure we got at least some embeddings
            if not all_audio_features:
                print("ERROR: Failed to extract any audio features!")
                # Return zero embeddings as fallback
                audio_features = torch.zeros(batch_size, 512, device=target_device)
                return text_emb, audio_features
            
            # Concatenate all batches
            if len(all_audio_features) > 1:
                audio_features = torch.cat(all_audio_features, dim=0)
            else:
                audio_features = all_audio_features[0]
            
            # Debug audio features
            print(f"DEBUG: Initial audio features shape: {audio_features.shape}")
            
            # Pad any missing embeddings if some batches failed (use zeros)
            if audio_features.shape[0] < total_samples:
                missing = total_samples - audio_features.shape[0]
                print(f"Warning: {missing} audio samples failed processing. Padding with zeros.")
                padding = torch.zeros(missing, audio_features.shape[1], device=audio_features.device)
                audio_features = torch.cat([audio_features, padding], dim=0)
            
            # Reshape back to [batch, stems, chunks, embed_dim]
            embed_dim = audio_features.size(-1)
            audio_features = audio_features.view(batch_size, stems, chunks, embed_dim)
            
            # Debug reshaped features
            print(f"DEBUG: Reshaped audio features: {audio_features.shape}")
            
            # If token identifiers are provided, incorporate them
            if token_identifiers is not None:
                # Convert token strings to embedding IDs
                token_embeds = []
                
                for batch_idx in range(batch_size):
                    batch_token_embeds = []
                    
                    # Process each stem in this batch item
                    for stem_idx in range(stems):
                        stem_embeds = []
                        
                        # Process each chunk in this stem
                        for chunk_idx in range(chunks):
                            # Get the token for this position
                            token = "[PAD]"  # Default
                            
                            # Check if we have a valid token identifier
                            if (batch_idx < len(token_identifiers) and 
                                token_identifiers[batch_idx] is not None and
                                stem_idx < len(token_identifiers[batch_idx]) and 
                                token_identifiers[batch_idx][stem_idx] is not None and
                                chunk_idx < len(token_identifiers[batch_idx][stem_idx]) and
                                token_identifiers[batch_idx][stem_idx][chunk_idx] is not None):
                                
                                token = token_identifiers[batch_idx][stem_idx][chunk_idx]
                            
                            # Get token ID and embedding
                            token_id = self._get_token_id(token)
                            token_embed = self.stem_token_embedder(torch.tensor([token_id], device=target_device))
                            stem_embeds.append(token_embed)
                        
                        # Stack chunk embeddings for this stem
                        if stem_embeds:
                            stem_embed = torch.cat(stem_embeds, dim=0)
                            batch_token_embeds.append(stem_embed)
                    
                    # Stack all stems for this batch item
                    if batch_token_embeds:
                        batch_embed = torch.stack(batch_token_embeds, dim=0)
                        token_embeds.append(batch_embed)
                
                # Stack across batch dimension
                if token_embeds:
                    token_embeds = torch.stack(token_embeds, dim=0)
                    
                    # Add token embeddings to audio features (weighted by valid mask)
                    # Apply the mask to avoid adding embeddings to padded regions
                    token_embeds = token_embeds * valid_mask.unsqueeze(-1)
                    audio_features = audio_features + token_embeds
            
            # Apply mask again to zero out padded regions
            audio_features = audio_features * valid_mask.unsqueeze(-1)
            
            # Average across valid stems and chunks to get a single embedding per batch
            # First compute the sum and the count of valid elements
            audio_features_sum = audio_features.sum(dim=[1, 2])
            valid_count = valid_mask.sum(dim=[1, 2]).clamp(min=1.0).unsqueeze(-1)
            
            # Compute the average (sum / count)
            audio_features = audio_features_sum / valid_count
            
            # Final check for NaN values
            if torch.isnan(audio_features).any():
                print("WARNING: NaN values in final audio features, replacing with zeros")
                audio_features = torch.nan_to_num(audio_features)
            
            print(f"DEBUG: Final audio features shape: {audio_features.shape}")
            
        return text_emb, audio_features
    
    # @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
    def forward(self, text_prompt=None, audio_waveform=None, token_identifiers=None, target_tool_tokens=None, 
              input_ids=None, attention_mask=None, tool_input_ids=None, tool_attention_mask=None, labels=None,
              embedding_scale=None):
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
            embedding_scale: Optional custom scaling factor for embeddings
        """
        # Move inputs to encoder device
        encoder_device = self.encoder_device
        decoder_device = self.decoder_device

        # Get model dtype from stored attribute to ensure consistent dtypes
        model_dtype = self.model_dtype
        print(f"DEBUG: Model dtype is {model_dtype}")

        # Log input shapes and types for debugging
        print(f"DEBUG INPUT - text_prompt shape: {text_prompt.shape if isinstance(text_prompt, torch.Tensor) else 'N/A'}")
        print(f"DEBUG INPUT - audio_waveform shape: {audio_waveform.shape}")
        if labels is not None:
            print(f"DEBUG INPUT - labels shape: {labels.shape}, dtype: {labels.dtype}")

        # Encode audio using our stable wrapper
        if text_prompt is None:
            raise ValueError("text_prompt must be provided")
        
        if audio_waveform is None:
            raise ValueError("audio_waveform must be provided")
        
        # Ensure text_prompt is on the correct device
        if isinstance(text_prompt, torch.Tensor) and text_prompt.device != encoder_device:
            text_prompt = text_prompt.to(encoder_device)

        # Ensure audio_waveform is on the correct device
        if isinstance(audio_waveform, torch.Tensor) and audio_waveform.device != encoder_device:
            audio_waveform = audio_waveform.to(encoder_device)
            
        # Get audio embedding using the updated method that handles token identifiers
        with torch.no_grad():  # Use no_grad to prevent gradient propagation through the CLAP model
            text_emb, audio_emb = self.get_embedding(text_prompt, audio_waveform, token_identifiers)

        # Log embedding statistics
        print(f"DEBUG EMBED - text_emb shape: {text_emb.shape}, min: {text_emb.min().item():.4f}, max: {text_emb.max().item():.4f}, mean: {text_emb.mean().item():.4f}")
        print(f"DEBUG EMBED - audio_emb shape: {audio_emb.shape}, min: {audio_emb.min().item():.4f}, max: {audio_emb.max().item():.4f}, mean: {audio_emb.mean().item():.4f}")

        # Apply normalization to both embeddings
        text_emb = self.safe_normalize(text_emb)
        audio_emb = self.safe_normalize(audio_emb)
        
        # Log normalized embedding statistics
        print(f"DEBUG NORM - text_emb min: {text_emb.min().item():.4f}, max: {text_emb.max().item():.4f}, mean: {text_emb.mean().item():.4f}")
        print(f"DEBUG NORM - audio_emb min: {audio_emb.min().item():.4f}, max: {audio_emb.max().item():.4f}, mean: {audio_emb.mean().item():.4f}")
        
        # Combine embeddings
        fused = torch.cat([text_emb, audio_emb], dim=-1)
        
        # Log fusion statistics
        print(f"DEBUG FUSED - shape: {fused.shape}, min: {fused.min().item():.4f}, max: {fused.max().item():.4f}, mean: {fused.mean().item():.4f}")
        
        # Apply fusion projection with the improved architecture
        condition_emb = self.fusion_proj(fused)  # [batch, mistral_hidden_dim]
        
        # Log condition embedding before scaling
        print(f"DEBUG COND - before scaling shape: {condition_emb.shape}, min: {condition_emb.min().item():.4f}, max: {condition_emb.max().item():.4f}, mean: {condition_emb.mean().item():.4f}")
        
        # FIXED: Scale embeddings properly for Mistral
        # Transformer models like Mistral expect embeddings scaled by sqrt(d_model)
        # Original clamp to [-1, 1] was too restrictive
        d_model = self.decoder.config.hidden_size
        
        # Use provided scale or default to sqrt(d_model)
        if embedding_scale is not None:
            scaling_factor = embedding_scale
            print(f"DEBUG SCALE - Using custom scaling factor: {scaling_factor}")
        else:
            scaling_factor = (d_model ** 0.5)  # square root of d_model (typically ~64 for 4096 dim)
            print(f"DEBUG SCALE - Using default scaling factor (sqrt(d_model)): {scaling_factor}")
        
        # Scale the condition embedding but keep it within reasonable bounds
        condition_emb = condition_emb * scaling_factor
        # Use a more generous clamping to allow for proper scaled values but prevent extreme outliers
        condition_emb = torch.clamp(condition_emb, min=-10.0, max=10.0)
        
        # Log condition embedding after scaling
        print(f"DEBUG COND - after scaling shape: {condition_emb.shape}, min: {condition_emb.min().item():.4f}, max: {condition_emb.max().item():.4f}, mean: {condition_emb.mean().item():.4f}")

        # We need to handle the mismatch between condition_emb (sequence length 1) 
        # and labels (tool_input_ids) which have longer sequence length
        if labels is not None:
            # Create causal inputs from condition_emb for decoder
            if labels.dim() > 1:
                # Get sequence length from labels
                seq_len = labels.size(1)
                batch_size = condition_emb.size(0)
                hidden_dim = condition_emb.size(-1)
                
                
                # First, get the embedding weight matrix from the decoder
                # This allows us to use the true token embeddings instead of zeros
                try:
                    word_embeddings = None
                    
                    # Try multiple paths to get embeddings
                    if hasattr(self.decoder, 'get_input_embeddings'):
                        print("DEBUG EMBED - Using get_input_embeddings")
                        word_embeddings = self.decoder.get_input_embeddings().weight
                    elif hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'get_input_embeddings'):
                        print("DEBUG EMBED - Using model.get_input_embeddings")
                        word_embeddings = self.decoder.model.get_input_embeddings().weight
                    # Special handling for PeftModel
                    elif 'PeftModel' in self.decoder.__class__.__name__:
                        print("DEBUG EMBED - Detected PeftModel, trying base_model path")
                        # PeftModel has a base_model attribute which contains the original model
                        if hasattr(self.decoder, 'base_model'):
                            if hasattr(self.decoder.base_model, 'get_input_embeddings'):
                                print("DEBUG EMBED - Using base_model.get_input_embeddings")
                                word_embeddings = self.decoder.base_model.get_input_embeddings().weight
                            elif hasattr(self.decoder.base_model, 'model') and hasattr(self.decoder.base_model.model, 'get_input_embeddings'):
                                print("DEBUG EMBED - Using base_model.model.get_input_embeddings")
                                word_embeddings = self.decoder.base_model.model.get_input_embeddings().weight
                            elif hasattr(self.decoder.base_model, 'model') and hasattr(self.decoder.base_model.model, 'embed_tokens'):
                                print("DEBUG EMBED - Using base_model.model.embed_tokens")
                                word_embeddings = self.decoder.base_model.model.embed_tokens.weight
                    # Fallback for Mistral model structure
                    elif hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'embed_tokens'):
                        print("DEBUG EMBED - Using model.embed_tokens")
                        word_embeddings = self.decoder.model.embed_tokens.weight
                    elif hasattr(self.decoder, 'model') and hasattr(self.decoder.model, 'model') and hasattr(self.decoder.model.model, 'embed_tokens'):
                        print("DEBUG EMBED - Using model.model.embed_tokens")
                        word_embeddings = self.decoder.model.model.embed_tokens.weight
                    
                    if word_embeddings is not None:
                        print(f"DEBUG EMBED - Found word embeddings of shape: {word_embeddings.shape}")
                        
                        # Create inputs_embeds using real token embeddings from labels
                        inputs_embeds = torch.zeros(
                            batch_size, seq_len, hidden_dim,
                            dtype=model_dtype,  # Use model dtype for consistency
                            device=encoder_device
                        )
                        
                        # Set the first position to our condition embedding
                        inputs_embeds[:, 0, :] = condition_emb.to(model_dtype)  # Convert to model dtype
                        
                        # For positions 1+, use actual token embeddings corresponding to the previous label
                        # This creates a proper autoregressive pattern where inputs at position t
                        # corresponds to the embedding of the target at position t-1
                        for t in range(1, seq_len):
                            # Get the previous token's embedding for each item in batch
                            for b in range(batch_size):
                                if t-1 < labels.size(1):  # Ensure we're within bounds
                                    token_idx = labels[b, t-1].item() 
                                    if token_idx >= 0 and token_idx < word_embeddings.size(0):
                                        # Get embedding for this token and convert to model dtype
                                        token_embedding = word_embeddings[token_idx].to(model_dtype)
                                        # Scale it properly
                                        token_embedding = token_embedding * scaling_factor / 10.0  # Scale a bit less than condition
                                        inputs_embeds[b, t, :] = token_embedding
                    else:
                        raise ValueError("Could not find word embeddings through any known path")
                        
                except Exception as e:
                    print(f"Warning: Could not use token embeddings: {str(e)}")
                    # Print detailed information about the decoder architecture for debugging
                    print(f"DEBUG ARCH - Decoder type: {type(self.decoder)}")
                    if hasattr(self.decoder, 'model'):
                        print(f"DEBUG ARCH - Decoder.model type: {type(self.decoder.model)}")
                    
                    # Fallback to original implementation
                    inputs_embeds = torch.zeros(
                        batch_size, seq_len, hidden_dim,
                        dtype=model_dtype,  # Use model dtype for consistency
                        device=encoder_device
                    )
                    inputs_embeds[:, 0, :] = condition_emb.to(model_dtype)  # Convert to model dtype
                    
                    # FIXED: Instead of zeros, add small random noise to other positions
                    # This prevents numerical instability while still allowing the model to focus on condition_emb
                    noise = torch.randn(batch_size, seq_len-1, hidden_dim, 
                                      dtype=model_dtype,  # Use model dtype for consistency
                                      device=encoder_device) * 0.01
                    inputs_embeds[:, 1:, :] = noise
                
                # Create attention mask that allows the model to see all tokens
                attention_mask = torch.ones(
                    batch_size, seq_len, 
                    dtype=torch.long, device=encoder_device
                )
            else:
                # If labels is just a 1D tensor, use simple setup
                inputs_embeds = condition_emb.to(model_dtype).unsqueeze(1)  # Convert to model dtype
                attention_mask = torch.ones(
                    condition_emb.size(0), 1, 
                    dtype=torch.long, device=encoder_device
                )
        else:
            # If no labels, just use the condition embedding
            inputs_embeds = condition_emb.to(model_dtype).unsqueeze(1)  # Convert to model dtype
            attention_mask = torch.ones(
                condition_emb.size(0), 1, 
                dtype=torch.long, device=encoder_device
            )
            
        # Add small epsilon to inputs_embeds to prevent exact zeros
        inputs_embeds = inputs_embeds + 1e-6 * torch.randn_like(inputs_embeds)
        
        # Check for NaN or Inf in inputs before passing to decoder
        if torch.isnan(inputs_embeds).any() or torch.isinf(inputs_embeds).any():
            print("WARNING: NaN or Inf detected in inputs_embeds, replacing with small random values")
            mask = torch.isnan(inputs_embeds) | torch.isinf(inputs_embeds)
            inputs_embeds = torch.where(mask, 
                                      torch.randn_like(inputs_embeds) * 0.01, 
                                      inputs_embeds)
        
        # Move inputs to decoder device before forwarding through the model
        # Make sure all inputs have the correct dtype
        inputs_embeds = inputs_embeds.to(decoder_device, dtype=model_dtype)
        attention_mask = attention_mask.to(decoder_device)

        if labels is not None:
            labels = labels.to(decoder_device)
        # Forward through the decoder
        try:
            # Compute loss scaling factor (for numerical stability)
            if labels is not None:
                # Count non-masked tokens in labels (-100 are masked)
                if labels.dim() > 1:
                    non_masked = (labels != -100).sum().item()
                    total = labels.numel()
                    print(f"DEBUG LOSS - Non-masked tokens: {non_masked}/{total} = {non_masked/total:.2%}")
                    
                    # Modified loss calculation to prevent NaN
                    # If only using the first token for loss (common issue), warn about it
                    if non_masked < 5 and total > 10:
                        print("WARNING: Very few non-masked tokens for loss calculation. This may cause unstable gradients.")
                    
                    # Check if labels contain out-of-vocabulary indices
                    vocab_size = self.decoder.config.vocab_size
                    if (labels >= 0).any() and (labels >= vocab_size).any():
                        out_of_vocab = ((labels >= 0) & (labels >= vocab_size)).sum().item()
                        print(f"WARNING: {out_of_vocab} label indices exceed vocabulary size ({vocab_size})!")
                        
                        # Fix labels to prevent out-of-bounds errors
                        labels = torch.where(
                            (labels >= 0) & (labels >= vocab_size),
                            torch.tensor(-100, device=labels.device, dtype=labels.dtype),
                            labels
                        )
                        print(f"DEBUG LOSS - Fixed out-of-vocab labels")
            
            # Try using a custom loss scale for stability
            custom_loss = False
            
            if custom_loss and labels is not None:
                # Use custom loss calculation (might be more stable)
                outputs = self.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False
                )
                
                # Get logits
                logits = outputs.logits
                
                # Compute loss manually with extra stability measures
                if labels.dim() > 1:
                    # Shift logits and labels for causal language modeling
                    shift_logits = logits[:, :-1, :]
                    shift_labels = labels[:, 1:]
                    
                    # Apply label smoothing (0.1) for stability
                    loss_fct = torch.nn.CrossEntropyLoss(
                        ignore_index=-100, 
                        reduction='mean',
                        label_smoothing=0.1
                    )
                    
                    # Flatten the tokens
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                    
                    # Add loss to outputs
                    outputs.loss = loss
                else:
                    # Handle 1D labels case
                    loss_fct = torch.nn.CrossEntropyLoss(
                        ignore_index=-100, 
                        reduction='mean',
                        label_smoothing=0.1
                    )
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    outputs.loss = loss
            else:
                # Use standard decoder with labels
                outputs = self.decoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False
                )
            
            # Check logits for abnormal values
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
                print(f"DEBUG LOGITS - Shape: {logits.shape}, min: {logits.min().item():.4f}, max: {logits.max().item():.4f}")
                print(f"DEBUG LOGITS - Mean: {logits.mean().item():.4f}, std: {logits.std().item():.4f}")
                
                if torch.isnan(logits).any():
                    print("WARNING: NaN values detected in logits")
                    # Try to repair NaN logits
                    nan_mask = torch.isnan(logits)
                    if nan_mask.any():
                        nan_count = nan_mask.sum().item()
                        print(f"WARNING: Replacing {nan_count} NaN logits with zeros")
                        logits = torch.where(nan_mask, torch.zeros_like(logits), logits)
                        outputs.logits = logits
                        
                if torch.isinf(logits).any():
                    print("WARNING: Inf values detected in logits")
                    # Try to repair Inf logits
                    inf_mask = torch.isinf(logits)
                    if inf_mask.any():
                        inf_count = inf_mask.sum().item()
                        print(f"WARNING: Replacing {inf_count} Inf logits with large finite values")
                        max_val = 1e4
                        logits = torch.where(
                            inf_mask, 
                            torch.sign(logits) * max_val, 
                            logits
                        )
                        outputs.logits = logits
            
            # Check hidden states for abnormal values
            # if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # # Print layer-by-layer NaN detection to find where the issue starts
                # print("DEBUG LAYERS - Checking NaN status in hidden states")
                # for i, hidden in enumerate(outputs.hidden_states):
                #     has_nan = torch.isnan(hidden).any().item()
                #     print(f"Layer {i} hidden state: {'HAS NaN' if has_nan else 'OK'} | Shape: {hidden.shape}")
                
                # # Check final layer's hidden states
                # final_hidden = outputs.hidden_states[-1]
                # print(f"DEBUG HIDDEN - Final layer shape: {final_hidden.shape}")
                # print(f"DEBUG HIDDEN - Final layer min: {final_hidden.min().item():.4f}, max: {final_hidden.max().item():.4f}")
                # print(f"DEBUG HIDDEN - Final layer mean: {final_hidden.mean().item():.4f}, std: {final_hidden.std().item():.4f}")
                
                # if torch.isnan(final_hidden).any():
                #     print("WARNING: NaN values detected in final hidden states")
                # if torch.isinf(final_hidden).any():
                #     print("WARNING: Inf values detected in final hidden states")
            
            # # Check loss
            # if hasattr(outputs, 'loss') and outputs.loss is not None:
            #     loss = outputs.loss
            #     print(f"DEBUG LOSS - Value: {loss.item():.4f}")
                
            #     if torch.isnan(loss).any():
            #         print("WARNING: NaN loss detected")
            #         # Try to diagnose by checking logits and labels
            #         if hasattr(outputs, 'logits'):
            #             # Check if there are extreme logit values that could cause overflow
            #             extreme_logits = (logits.abs() > 1000).sum().item()
            #             if extreme_logits > 0:
            #                 print(f"WARNING: Found {extreme_logits} extreme logit values (abs > 1000)")
            #     if torch.isinf(loss).any():
            #         print("WARNING: Inf loss detected")
                
        except Exception as e:
            print(f"Error in decoder forward pass: {e}")
            import traceback
            traceback.print_exc()
            # Return dummy outputs with NaN loss to signal the problem
            from transformers.modeling_outputs import CausalLMOutputWithPast
            outputs = CausalLMOutputWithPast(
                loss=torch.tensor(float('nan'), device=decoder_device),
                logits=torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1], 
                                 self.decoder.config.vocab_size, device=decoder_device),
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )
        
        # Store embeddings in outputs for debugging
        # Move them back to encoder device to avoid cross-device issues
        outputs.text_embedding = text_emb
        outputs.audio_embedding = audio_emb
        
        return outputs
