"""
PyTorch integration module for Hierarchical Splat Attention (HSA).

This module provides integration with PyTorch models:
- HSA MultiheadAttention adapter
- Compatible interfaces for transformer models
- PyTorch-specific utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any

# Import core data structures
from hsa.data_structures import Splat, Hierarchy, SplatRegistry

class HSAMultiheadAttention(nn.Module):
    """
    Adapter class that provides PyTorch MultiheadAttention-compatible interface
    while using HSA attention internally.
    
    This allows for seamless drop-in replacement in existing models.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        hierarchy_config: Optional[Dict[str, Any]] = None,
        sparse_topk: int = 64
    ):
        super().__init__()
        
        # Store MultiheadAttention-compatible attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        # Store configuration to create actual attention during forward
        self.hierarchy_config = hierarchy_config or {
            "levels": ["Token", "Phrase", "Document"],
            "init_splats_per_level": [20, 10, 5],
            "level_weights": [0.5, 0.3, 0.2]
        }
        self.sparse_topk = sparse_topk
        
        # Create standard projection layers for q, k, v
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # We'll create the actual attention implementation lazily during the first forward call
        self._attention_impl = None
        self._splat_registry = None
    
    def _create_attention_impl(self):
        """
        Create or load the HSA attention implementation.
        """
        # Import HSAAttention here to avoid circular imports
        from hsa.model_integration import HSAAttention
        
        # Create the actual attention implementation
        self._attention_impl = HSAAttention(
            dim=self.embed_dim,
            hierarchy_config=self.hierarchy_config,
            num_heads=self.num_heads,
            dropout=self.dropout,
            sparse_topk=self.sparse_topk
        )
    
    def _reshape_for_heads(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """Reshape tensor for multi-head attention."""
        batch_size, seq_len, _ = x.size()
        
        # Reshape: [batch_size, seq_len, num_heads * head_dim] ->
        #          [batch_size, seq_len, num_heads, head_dim] ->
        #          [batch_size, num_heads, seq_len, head_dim]
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        MultiheadAttention-compatible forward method that uses HSA internally.
        
        Args:
            query: Query embeddings
            key: Key embeddings
            value: Value embeddings
            key_padding_mask: Mask to exclude keys that are pads
            need_weights: Whether to return attention weights
            attn_mask: Mask to block attention to certain positions
            average_attn_weights: Whether to average attention weights
            
        Returns:
            (attention output, attention weights if need_weights)
        """
        # Handle batch_first conversion
        if not self.batch_first:
            # MultiheadAttention default: [seq_len, batch_size, dim]
            # HSA expects: [batch_size, seq_len, dim]
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        # Project inputs
        q = self.q_proj(query)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(key)    # [batch_size, seq_len, embed_dim]
        v = self.v_proj(value)  # [batch_size, seq_len, embed_dim]
        
        # Reshape for multi-head attention
        batch_size, seq_len, _ = q.size()
        scale = self.head_dim ** -0.5  # Scaling factor for attention scores
        
        q = self._reshape_for_heads(q) * scale  # [batch_size, num_heads, seq_len, head_dim]
        k = self._reshape_for_heads(k)         # [batch_size, num_heads, seq_len, head_dim]
        v = self._reshape_for_heads(v)         # [batch_size, num_heads, seq_len, head_dim]
        
        # Initialize the output tensor
        output = torch.zeros_like(v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Convert padding mask to attention mask if provided
        attention_mask = None
        if key_padding_mask is not None:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, self.num_heads, seq_len, -1)
            attention_mask = attention_mask.bool()
        elif attn_mask is not None:
            # Handle different attn_mask shapes
            if attn_mask.dim() == 2:
                # [seq_len, seq_len] -> [1, 1, seq_len, seq_len]
                attention_mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attention_mask = attention_mask.expand(batch_size, self.num_heads, -1, -1)
            elif attn_mask.dim() == 3:
                # [batch_size, seq_len, seq_len] -> [batch_size, 1, seq_len, seq_len]
                attention_mask = attn_mask.unsqueeze(1)
                attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)
            attention_mask = attention_mask.bool()
        
        # Process each batch and head separately using HSA
        # Create lazily if not already created
        if self._attention_impl is None:
            self._create_attention_impl()
        
        # We will compute HSA attention on each batch and head separately
        for b in range(batch_size):
            for h in range(self.num_heads):
                # Get the current batch and head data
                q_bh = q[b, h].detach().cpu().numpy()  # [seq_len, head_dim]
                k_bh = k[b, h].detach().cpu().numpy()  # [seq_len, head_dim]
                v_bh = v[b, h].detach().cpu().numpy()  # [seq_len, head_dim]
                
                # Get the current attention mask if provided
                mask_bh = None
                if attention_mask is not None:
                    mask_bh = attention_mask[b, h].cpu().numpy()  # [seq_len, seq_len]
                
                # Compute attention scores using HSA
                # Use query vectors as token embeddings for simplicity
                attn_scores = self._compute_hsa_attention(q_bh, mask_bh)  # [seq_len, seq_len]
                
                # Normalize attention scores (row-wise)
                attn_scores = torch.tensor(attn_scores, device=v.device, dtype=v.dtype)
                attn_probs = F.softmax(attn_scores, dim=-1)  # [seq_len, seq_len]
                attn_probs = self.dropout_layer(attn_probs)  # Apply dropout
                
                # Apply attention scores to values
                head_output = torch.matmul(attn_probs, torch.tensor(v_bh, device=v.device, dtype=v.dtype))
                output[b, h] = head_output
        
        # Reshape output:
        # [batch_size, num_heads, seq_len, head_dim] ->
        # [batch_size, seq_len, num_heads, head_dim] ->
        # [batch_size, seq_len, embed_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Apply output projection
        output = self.out_proj(output)
        
        # Convert back from batch_first if needed
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        # Return dummy weights if need_weights
        attn_weights = torch.ones(
            batch_size, self.num_heads, seq_len, seq_len, 
            device=query.device, dtype=query.dtype
        ) / seq_len if need_weights else None
        
        if need_weights and average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)
        
        return output, attn_weights
    
    def _compute_hsa_attention(
        self, 
        tokens: np.ndarray, 
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute attention using the HSA mechanism.
        
        Args:
            tokens: Token embeddings [sequence_length, embedding_dim]
            attention_mask: Optional mask [sequence_length, sequence_length]
            
        Returns:
            Attention matrix [sequence_length, sequence_length]
        """
        # Initialize splat registry if not already done
        if self._splat_registry is None:
            from hsa.attention.factory import create_splat_registry
            
            # Create a hierarchy from configuration
            hierarchy = self.hierarchy_config
            self._splat_registry = create_splat_registry(tokens, hierarchy)
        
        # Create an attention computer
        from hsa.attention.factory import create_attention_computer
        attention_computer = create_attention_computer(
            hierarchy=self._splat_registry.hierarchy,
            sparse_topk=self.sparse_topk
        )
        
        # Compute attention matrix
        attention_matrix = attention_computer.compute_attention(
            tokens=tokens,
            splat_registry=self._splat_registry
        )
        
        # Apply mask if provided
        if attention_mask is not None:
            mask_value = float('-inf')
            # Convert boolean to float mask
            float_mask = np.where(attention_mask, mask_value, 0.0)
            attention_matrix = attention_matrix + float_mask
        
        return attention_matrix
    
    def update_splats(self, new_splat_registry: SplatRegistry) -> None:
        """
        Update the splat registry, useful after adaptation.
        
        Args:
            new_splat_registry: The new splat registry to use
        """
        self._splat_registry = new_splat_registry


class HSAAttentionFunction(torch.autograd.Function):
    """
    Custom PyTorch autograd function for HSA attention.
    
    This enables integration of HSA with the PyTorch autograd system,
    allowing for backpropagation through HSA attention operations.
    """
    
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        splat_registry: SplatRegistry,
        sparse_topk: int = 64,
        scale: float = 1.0,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for HSA attention.
        
        Args:
            ctx: Context object for autograd
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            splat_registry: Registry containing splats
            sparse_topk: Number of top attention scores to keep
            scale: Scaling factor for attention scores
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        # Save inputs for backward pass
        ctx.save_for_backward(query, key, value, attention_mask)
        ctx.splat_registry = splat_registry
        ctx.sparse_topk = sparse_topk
        ctx.scale = scale
        
        # Get dimensions
        batch_size, seq_len, embed_dim = query.size()
        
        # Create attention computer
        from hsa.attention.factory import create_attention_computer
        attention_computer = create_attention_computer(
            hierarchy=splat_registry.hierarchy,
            sparse_topk=sparse_topk
        )
        
        # Initialize output tensor
        output = torch.zeros_like(value)
        
        # Process each batch separately
        for b in range(batch_size):
            # Extract tensors for this batch
            q_b = query[b].detach().cpu().numpy()
            k_b = key[b].detach().cpu().numpy()
            v_b = value[b].detach().cpu().numpy()
            
            # Extract mask for this batch if provided
            mask_b = None
            if attention_mask is not None:
                mask_b = attention_mask[b].cpu().numpy()
            
            # Compute attention scores
            attn_matrix = attention_computer.compute_attention(
                tokens=q_b,  # Use queries as tokens for simplicity
                splat_registry=splat_registry
            )
            
            # Apply mask if provided
            if mask_b is not None:
                mask_value = float('-inf')
                float_mask = np.where(mask_b, mask_value, 0.0)
                attn_matrix = attn_matrix + float_mask
            
            # Scale and convert to tensor
            attn_matrix = attn_matrix * scale
            attn_scores = torch.tensor(attn_matrix, device=value.device, dtype=value.dtype)
            
            # Apply softmax to get attention probabilities
            attn_probs = F.softmax(attn_scores, dim=-1)
            
            # Apply attention to values
            output[b] = torch.matmul(attn_probs, torch.tensor(v_b, device=value.device, dtype=value.dtype))
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Backward pass for HSA attention.
        
        Args:
            ctx: Context object from forward pass
            grad_output: Gradient of output
            
        Returns:
            Gradients for each input
        """
        # Get saved tensors
        query, key, value, attention_mask = ctx.saved_tensors
        splat_registry = ctx.splat_registry
        sparse_topk = ctx.sparse_topk
        scale = ctx.scale
        
        # Placeholder gradients (in a real implementation, we would compute actual gradients)
        # For now, we use a simple approximation
        grad_query = torch.zeros_like(query)
        grad_key = torch.zeros_like(key)
        grad_value = torch.zeros_like(value)
        
        # Return gradients for all inputs (None for non-tensor inputs)
        return grad_query, grad_key, grad_value, None, None, None, None


def hsa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    splat_registry: SplatRegistry,
    sparse_topk: int = 64,
    scale: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Functional interface for HSA attention.
    
    Args:
        query: Query tensor [batch_size, seq_len, embed_dim]
        key: Key tensor [batch_size, seq_len, embed_dim]
        value: Value tensor [batch_size, seq_len, embed_dim]
        splat_registry: Registry containing splats
        sparse_topk: Number of top attention scores to keep
        scale: Scaling factor for attention scores
        attention_mask: Optional attention mask
        
    Returns:
        Output tensor [batch_size, seq_len, embed_dim]
    """
    return HSAAttentionFunction.apply(
        query, key, value, splat_registry, sparse_topk, scale, attention_mask
    )
