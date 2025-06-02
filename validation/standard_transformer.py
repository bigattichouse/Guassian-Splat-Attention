"""
Standard Transformer Implementation for Baseline Comparison

This module provides a clean implementation of standard transformer models
for comparison against SplatFlow and other attention mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class StandardTransformerGPT(nn.Module):
    """Standard transformer GPT-style model for comparison"""
    
    def __init__(self, vocab_size: int, model_dim: int = 256, 
                 num_layers: int = 4, num_heads: int = 4, max_seq_len: int = 512):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the transformer"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)
        
        x = self.embedding_dropout(token_emb + pos_emb)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Transformer layers
        x = self.transformer(x, mask=causal_mask)
        
        # Output projection
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        return logits


class StandardMultiHeadAttention(nn.Module):
    """Standard multi-head attention implementation for fine-grained comparison"""
    
    def __init__(self, model_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert model_dim % num_heads == 0
        
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard scaled dot-product attention"""
        batch_size, seq_len, model_dim = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if attention_mask is None:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        else:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(1), -1e9)
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        out = self.out_proj(out)
        
        return out


class StandardTransformerLayer(nn.Module):
    """Single transformer layer for custom implementations"""
    
    def __init__(self, model_dim: int, num_heads: int = 8, 
                 ff_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = model_dim * 4
        
        # Self-attention
        self.self_attn = StandardMultiHeadAttention(model_dim, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(model_dim)
        
        # Feed-forward
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout)
        )
        self.ff_norm = nn.LayerNorm(model_dim)
        
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through transformer layer"""
        
        # Self-attention with residual connection
        attn_out = self.self_attn(x, attention_mask)
        x = self.attn_norm(x + attn_out)
        
        # Feed-forward with residual connection
        ff_out = self.feed_forward(x)
        x = self.ff_norm(x + ff_out)
        
        return x


class CustomStandardTransformerGPT(nn.Module):
    """Custom standard transformer built from our own layers for detailed comparison"""
    
    def __init__(self, vocab_size: int, model_dim: int = 256, 
                 num_layers: int = 4, num_heads: int = 8, max_seq_len: int = 512):
        super().__init__()
        self.model_dim = model_dim
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            StandardTransformerLayer(model_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Output
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, vocab_size, bias=False)
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)
        
        x = self.embedding_dropout(token_emb + pos_emb)
        
        # Process through layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Output
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        return logits


def create_standard_model(model_type: str = "pytorch", **kwargs):
    """Factory function to create different standard transformer variants"""
    
    if model_type == "pytorch":
        return StandardTransformerGPT(**kwargs)
    elif model_type == "custom":
        return CustomStandardTransformerGPT(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'pytorch' or 'custom'.")


def test_standard_models():
    """Test the standard transformer implementations"""
    print("Testing Standard Transformer Implementations")
    print("=" * 45)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test parameters
    vocab_size = 1000
    model_dim = 128
    seq_len = 32
    batch_size = 4
    
    # Create models
    pytorch_model = create_standard_model("pytorch", 
                                         vocab_size=vocab_size, 
                                         model_dim=model_dim, 
                                         num_layers=3).to(device)
    
    custom_model = create_standard_model("custom", 
                                        vocab_size=vocab_size, 
                                        model_dim=model_dim, 
                                        num_layers=3).to(device)
    
    # Test data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Test both models
    models = {'PyTorch Standard': pytorch_model, 'Custom Standard': custom_model}
    
    for name, model in models.items():
        print(f"\nTesting {name}:")
        
        # Parameter count
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(input_ids)
            print(f"  Output shape: {output.shape}")
            print(f"  Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
        
        # Training step
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        logits = model(input_ids[:, :-1])
        targets = input_ids[:, 1:]
        loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"  Training loss: {loss.item():.4f}")
    
    print(f"\n✅ All standard models working correctly!")


if __name__ == "__main__":
    test_standard_models()
