# hsa/wrappers/hsa_wrapper.py
import torch
import numpy as np
from typing import Dict, Optional, Any, List, Union

class HSAGPTModel(torch.nn.Module):
    """
    Custom GPT model with HSA attention.
    
    This class matches the architecture used in checkpoints.
    """
    
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        max_position_embeddings=1024,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        **kwargs
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        
        self.transformer = torch.nn.ModuleDict({
            # Embeddings
            "word_embeddings": torch.nn.Embedding(vocab_size, hidden_size),
            "position_embeddings": torch.nn.Embedding(max_position_embeddings, hidden_size),
            "LayerNorm": torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps),
            
            # Transformer layers
            "layers": torch.nn.ModuleList([self._create_layer() for _ in range(num_hidden_layers)])
        })
        
        # Register position IDs buffer
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).unsqueeze(0))
        
        # LM head
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Tie weights (if desired)
        self.lm_head.weight = self.transformer.word_embeddings.weight
        
        # Initialize weights
        self._init_weights()
    
    def _create_layer(self):
        """Create a single transformer layer."""
        return torch.nn.ModuleDict({
            # Self attention
            "attention": torch.nn.ModuleDict({
                "q_proj": torch.nn.Linear(self.hidden_size, self.hidden_size),
                "k_proj": torch.nn.Linear(self.hidden_size, self.hidden_size),
                "v_proj": torch.nn.Linear(self.hidden_size, self.hidden_size),
                "out_proj": torch.nn.Linear(self.hidden_size, self.hidden_size),
            }),
            
            # Layer normalization
            "layer_norm1": torch.nn.LayerNorm(self.hidden_size),
            "layer_norm2": torch.nn.LayerNorm(self.hidden_size),
            
            # Feed-forward network
            "ffn": torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.intermediate_size),
                torch.nn.GELU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.intermediate_size, self.hidden_size),
            )
        })
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs
    ):
        """Forward pass."""
        batch_size, seq_length = input_ids.size()
        
        # Get embeddings
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        word_embeds = self.transformer.word_embeddings(input_ids)
        position_embeds = self.transformer.position_embeddings(position_ids)
        
        # Add embeddings
        hidden_states = word_embeds + position_embeds
        hidden_states = self.transformer.LayerNorm(hidden_states)
        
        # Process through layers
        for i, layer in enumerate(self.transformer.layers):
            residual = hidden_states
            
            # Self attention
            hidden_states = layer.layer_norm1(hidden_states)
            
            # Apply attention (simplified for demonstration)
            q = layer.attention.q_proj(hidden_states)
            k = layer.attention.k_proj(hidden_states)
            v = layer.attention.v_proj(hidden_states)
            
            # Reshape for attention heads
            head_dim = self.hidden_size // self.num_attention_heads
            
            q = q.view(batch_size, seq_length, self.num_attention_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_length, self.num_attention_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_length, self.num_attention_heads, head_dim).transpose(1, 2)
            
            # Calculate attention scores (dot product)
            attention_scores = torch.matmul(q, k.transpose(-1, -2))
            attention_scores = attention_scores / (head_dim ** 0.5)
            
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
            
            # Apply softmax
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            
            # Apply attention weights
            context = torch.matmul(attention_probs, v)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
            
            # Output projection
            hidden_states = layer.attention.out_proj(context)
            
            # Add residual
            hidden_states = residual + hidden_states
            
            # Feed-forward network
            residual = hidden_states
            hidden_states = layer.layer_norm2(hidden_states)
            hidden_states = layer.ffn(hidden_states)
            hidden_states = residual + hidden_states
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        return {"logits": logits, "past_key_values": None}
    
    def generate(
        self,
        input_ids,
        max_new_tokens=20,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs
    ):
        """
        Generate text using the model.
        A simplified auto-regressive generation implementation.
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Start with input_ids as our sequence
        generated_ids = input_ids.clone()
        attention_mask = torch.ones_like(generated_ids)
        
        # Auto-regressive generation
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated_ids,
                    attention_mask=attention_mask
                )
            
            # Get next token logits
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p (nucleus) sampling
            if do_sample and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample from the distribution
            if do_sample:
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Check for EOS token
            if eos_token_id is not None and (next_token == eos_token_id).any():
                eos_indices = (next_token == eos_token_id).nonzero()
                if eos_indices.shape[0] > 0:
                    # Stop generation for sequences that hit EOS
                    for idx in eos_indices[:, 0]:
                        if idx < batch_size:
                            next_token[idx] = pad_token_id
            
            # Append new token to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            
            # Break if all sequences have hit EOS or max length
            if eos_token_id is not None and (generated_ids == eos_token_id).any(dim=1).all():
                break
        
        return generated_ids

def load_custom_model(
    checkpoint: Dict[str, Any], 
    config: Optional[Dict[str, Any]] = None,
    device: str = "cpu"
) -> HSAGPTModel:
    """
    Load a custom model and its state dict.
    
    Args:
        checkpoint: The loaded checkpoint
        config: Optional model configuration
        device: Device to load the model to
        
    Returns:
        The loaded model
    """
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Try to extract config from checkpoint
    if config is None and "config" in checkpoint:
        config = checkpoint["config"]
    
    # Create a default config if none provided
    if config is None:
        config = {
            "vocab_size": 50257,  # GPT-2 vocabulary size
            "hidden_size": 128,   # Reduced for TinyStories
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 512,
            "max_position_embeddings": 1024
        }
    
    # Check if we can infer hidden_size from state dict
    if "hidden_size" not in config:
        # Try to infer from weight shapes
        for key, tensor in state_dict.items():
            if "word_embeddings.weight" in key:
                config["hidden_size"] = tensor.shape[1]
                break
    
    # Create custom model
    model = HSAGPTModel(**config)
    
    # Load state dict
    try:
        model.load_state_dict(state_dict)
        print("Successfully loaded model using custom HSAGPTModel architecture")
        return model.to(device)
    except Exception as e:
        print(f"Error loading state dict into custom model: {e}")
        print("Trying to load with custom keys...")
        
        # Try to manually map keys if standard load fails
        new_state_dict = {}
        for k, v in state_dict.items():
            # Skip position_ids buffer to avoid errors
            if "position_ids" in k:
                continue
            new_state_dict[k] = v
        
        # Load with strict=False to ignore missing keys
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded with missing keys: {len(missing)} and unexpected keys: {len(unexpected)}")
        return model.to(device)

def apply_hsa_to_model(model: torch.nn.Module, hsa_config: Dict[str, Any]) -> torch.nn.Module:
    """
    Apply HSA attention to a model.
    
    Args:
        model: The model to modify
        hsa_config: HSA configuration
        
    Returns:
        Modified model with HSA attention
    """
    try:
        from hsa.core import create_hsa
        from hsa.model_integration import replace_attention_with_hsa
        
        # Create HSA
        hsa = create_hsa(hsa_config)
        
        # Apply HSA to model
        model = replace_attention_with_hsa(model, hsa_config)
        print("Applied HSA to model")
        
        return model
    except ImportError:
        print("HSA modules not available")
        return model
    except Exception as e:
        print(f"Error applying HSA to model: {e}")
        return model
