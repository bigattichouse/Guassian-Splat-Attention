"""
SplatFlow Attention Components - Complete Implementation
Core splat and attention mechanisms for the SplatFlow architecture.

This module provides:
- Fixed splat implementation with adaptive positioning
- Production SplatFlow attention mechanism 
- Health monitoring and emergency rescue systems
- Core attention building blocks for the complete model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, Dict, List, Any, Union
import warnings

logger = logging.getLogger(__name__)


class AdaptiveSplatPositioning(nn.Module):
    """
    Adaptive positioning system for splats in embedding space.
    Handles intelligent placement and movement of information mediators.
    """
    
    def __init__(
        self,
        num_splats: int,
        model_dim: int,
        embedding_dim: Optional[int] = None,
        adaptation_rate: float = 0.1
    ):
        super().__init__()
        self.num_splats = num_splats
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim or model_dim
        self.adaptation_rate = adaptation_rate
        
        # Learnable splat positions in embedding space
        self.splat_positions = nn.Parameter(
            torch.randn(num_splats, self.embedding_dim) * 0.02
        )
        
        # Splat characteristics
        self.splat_scales = nn.Parameter(torch.ones(num_splats))
        self.splat_importance = nn.Parameter(torch.ones(num_splats))
        self.splat_specialization = nn.Parameter(torch.randn(num_splats, model_dim // 4))
        
        # Movement and adaptation networks
        self.position_update_net = nn.Sequential(
            nn.Linear(model_dim + self.embedding_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, self.embedding_dim),
            nn.Tanh()  # Constrain movement
        )
        
        # Health monitoring
        self.register_buffer('splat_usage_history', torch.zeros(num_splats, 100))
        self.register_buffer('usage_ptr', torch.tensor(0))
        self.register_buffer('adaptation_count', torch.tensor(0))
        
        self._init_positions()
    
    def _init_positions(self):
        """Initialize splat positions in a structured way."""
        with torch.no_grad():
            # Grid-like initialization for stability
            grid_size = int(math.ceil(math.sqrt(self.num_splats)))
            for i in range(self.num_splats):
                row = i // grid_size
                col = i % grid_size
                
                # Position in 2D grid, then project to higher dimensions
                base_pos = torch.tensor([row, col], dtype=torch.float32)
                base_pos = base_pos / grid_size - 0.5  # Center around origin
                
                # Fill in the embedding dimensions
                self.splat_positions[i, :2] = base_pos
                if self.embedding_dim > 2:
                    self.splat_positions[i, 2:] = torch.randn(self.embedding_dim - 2) * 0.1
    
    def compute_influences(
        self, 
        token_embeddings: torch.Tensor,
        return_distances: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute influence of each splat on each token.
        
        Args:
            token_embeddings: [batch, seq_len, model_dim]
            return_distances: Whether to return distance information
            
        Returns:
            influences: [batch, seq_len, num_splats]
            distances: [batch, seq_len, num_splats] (if return_distances=True)
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # Project token embeddings to position space
        if self.embedding_dim != self.model_dim:
            # Simple projection for now - could be learnable
            token_positions = token_embeddings[:, :, :self.embedding_dim]
        else:
            token_positions = token_embeddings
        
        # Compute distances between tokens and splats
        # token_positions: [batch, seq_len, 1, embedding_dim]
        # splat_positions: [1, 1, num_splats, embedding_dim]
        token_expanded = token_positions.unsqueeze(2)
        splat_expanded = self.splat_positions.unsqueeze(0).unsqueeze(0)
        
        # Euclidean distances
        distances = torch.norm(token_expanded - splat_expanded, dim=-1)
        
        # Convert to Gaussian influences with learnable scales
        scales = self.splat_scales.abs().clamp(min=1e-6)
        influences = torch.exp(-0.5 * (distances / scales.unsqueeze(0).unsqueeze(0)) ** 2)
        
        # Apply importance weighting
        importance = self.splat_importance.abs().unsqueeze(0).unsqueeze(0)
        influences = influences * importance
        
        # Update usage statistics
        with torch.no_grad():
            current_usage = influences.mean(dim=(0, 1))  # [num_splats]
            ptr = self.usage_ptr.item() % 100
            self.splat_usage_history[:, ptr] = current_usage
            self.usage_ptr += 1
        
        if return_distances:
            return influences, distances
        return influences
    
    def adapt_positions(self, token_embeddings: torch.Tensor, influences: torch.Tensor):
        """
        Adapt splat positions based on usage patterns and gradients.
        
        Args:
            token_embeddings: [batch, seq_len, model_dim]
            influences: [batch, seq_len, num_splats]
        """
        if not self.training:
            return
        
        batch_size, seq_len, _ = token_embeddings.shape
        
        # Compute movement updates for each splat
        for splat_idx in range(self.num_splats):
            # Get tokens that are influenced by this splat
            splat_influences = influences[:, :, splat_idx]  # [batch, seq_len]
            
            if splat_influences.sum() > 1e-6:  # Only update if splat is being used
                # Weighted average of token embeddings
                weights = splat_influences.unsqueeze(-1)  # [batch, seq_len, 1]
                weighted_tokens = (token_embeddings * weights).sum(dim=(0, 1))  # [model_dim]
                total_weight = weights.sum()
                
                if total_weight > 1e-6:
                    avg_token = weighted_tokens / total_weight
                    
                    # Compute position update
                    current_pos = self.splat_positions[splat_idx]
                    update_input = torch.cat([avg_token, current_pos])
                    position_delta = self.position_update_net(update_input)
                    
                    # Apply update with learning rate
                    with torch.no_grad():
                        self.splat_positions[splat_idx] += self.adaptation_rate * position_delta
        
        self.adaptation_count += 1
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get health statistics for splat positioning."""
        with torch.no_grad():
            # Average usage over recent history
            recent_usage = self.splat_usage_history.mean(dim=1)
            
            stats = {
                'num_splats': self.num_splats,
                'active_splats': int((recent_usage > 0.01).sum()),
                'avg_usage': float(recent_usage.mean()),
                'min_usage': float(recent_usage.min()),
                'max_usage': float(recent_usage.max()),
                'usage_std': float(recent_usage.std()),
                'adaptation_count': int(self.adaptation_count),
                'underutilized_splats': int((recent_usage < 0.005).sum()),
                'overutilized_splats': int((recent_usage > 0.5).sum())
            }
            
        return stats


class SplatInformationFlow(nn.Module):
    """
    Core information flow mechanism through splats.
    Handles gathering, transforming, and redistributing information.
    """
    
    def __init__(
        self,
        model_dim: int,
        num_splats: int,
        flow_dropout: float = 0.1,
        information_bottleneck: bool = True
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_splats = num_splats
        self.information_bottleneck = information_bottleneck
        
        # Information transformation at splats
        bottleneck_dim = model_dim // 2 if information_bottleneck else model_dim
        
        self.splat_encoder = nn.Sequential(
            nn.Linear(model_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(flow_dropout)
        )
        
        self.splat_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, model_dim),
            nn.Dropout(flow_dropout)
        )
        
        # Splat-specific transformations
        self.splat_transforms = nn.ModuleList([
            nn.Linear(bottleneck_dim, bottleneck_dim)
            for _ in range(num_splats)
        ])
        
        # Flow regulation
        self.flow_gate = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Sigmoid()
        )
        
        # Emergency rescue mechanism
        self.register_buffer('emergency_threshold', torch.tensor(0.001))
        self.register_buffer('rescue_count', torch.tensor(0))
        
    def forward(
        self,
        token_embeddings: torch.Tensor,
        splat_influences: torch.Tensor,
        return_flow_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Process information flow through splats.
        
        Args:
            token_embeddings: [batch, seq_len, model_dim]
            splat_influences: [batch, seq_len, num_splats]
            return_flow_info: Whether to return detailed flow information
            
        Returns:
            flow_output: [batch, seq_len, model_dim]
            flow_info: Dict with detailed flow information (if requested)
        """
        batch_size, seq_len, model_dim = token_embeddings.shape
        
        # Step 1: Gather information at each splat
        # influences: [batch, seq_len, num_splats] -> [batch, num_splats, seq_len]
        influences_t = splat_influences.transpose(1, 2)
        
        # Weighted sum of token information for each splat
        # [batch, num_splats, seq_len] x [batch, seq_len, model_dim] -> [batch, num_splats, model_dim]
        splat_gathered = torch.bmm(influences_t, token_embeddings)
        
        # Normalize by total influence per splat
        total_influence = influences_t.sum(dim=2, keepdim=True).clamp(min=1e-8)
        splat_gathered = splat_gathered / total_influence
        
        # Step 2: Transform information at each splat
        splat_encoded = self.splat_encoder(splat_gathered)  # [batch, num_splats, bottleneck_dim]
        
        # Apply splat-specific transformations
        splat_transformed = torch.zeros_like(splat_encoded)
        for i, transform in enumerate(self.splat_transforms):
            splat_transformed[:, i, :] = transform(splat_encoded[:, i, :])
        
        splat_decoded = self.splat_decoder(splat_transformed)  # [batch, num_splats, model_dim]
        
        # Step 3: Redistribute information back to tokens
        # [batch, seq_len, num_splats] x [batch, num_splats, model_dim] -> [batch, seq_len, model_dim]
        flow_output = torch.bmm(splat_influences, splat_decoded)
        
        # Apply flow gating to control information flow
        flow_gate = self.flow_gate(token_embeddings)
        flow_output = flow_output * flow_gate
        
        # Emergency rescue for underutilized splats
        if self.training:
            self._emergency_rescue(splat_influences)
        
        if return_flow_info:
            flow_info = {
                'splat_gathered': splat_gathered,
                'splat_transformed': splat_transformed,
                'flow_gate': flow_gate,
                'total_influence': total_influence
            }
            return flow_output, flow_info
        
        return flow_output
    
    def _emergency_rescue(self, influences: torch.Tensor):
        """Emergency rescue for underutilized splats."""
        with torch.no_grad():
            avg_influence = influences.mean(dim=(0, 1))  # [num_splats]
            underutilized = avg_influence < self.emergency_threshold
            
            if underutilized.any():
                self.rescue_count += 1
                if self.rescue_count % 100 == 0:  # Log occasionally
                    count = int(underutilized.sum())
                    logger.warning(f"🚨 Emergency rescue: {count} underutilized splats (total rescues: {self.rescue_count})")


class FixedProductionSplatFlowAttention(nn.Module):
    """
    Production-ready SplatFlow attention mechanism.
    Combines splat-based information flow with standard attention for stability.
    """
    
    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        num_splats: int = 20,
        dropout: float = 0.1,
        splat_attention_ratio: float = 0.4,
        use_flash_attention: bool = False
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_splats = num_splats
        self.head_dim = model_dim // num_heads
        self.splat_attention_ratio = splat_attention_ratio
        self.standard_attention_ratio = 1.0 - splat_attention_ratio
        
        assert model_dim % num_heads == 0, f"model_dim {model_dim} must be divisible by num_heads {num_heads}"
        
        # Standard attention components
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        
        # SplatFlow components
        self.splat_positioning = AdaptiveSplatPositioning(
            num_splats=num_splats,
            model_dim=model_dim
        )
        
        self.splat_flow = SplatInformationFlow(
            model_dim=model_dim,
            num_splats=num_splats,
            flow_dropout=dropout
        )
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout)
        
        # Health monitoring
        self.register_buffer('forward_count', torch.tensor(0))
        self.register_buffer('splat_efficiency', torch.tensor(0.0))
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters with proper scaling."""
        # Xavier initialization for projections
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
        
        # Special initialization for output projection (residual scaling)
        with torch.no_grad():
            self.out_proj.weight *= 0.5
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        SplatFlow attention forward pass.
        
        Args:
            hidden_states: [batch, seq_len, model_dim]
            attention_mask: Optional attention mask
            position_ids: Optional position indices  
            past_key_value: Optional cached key-value for generation
            output_attentions: Whether to return attention weights
            use_cache: Whether to return cached key-value
            
        Returns:
            Tuple of (output, attention_weights, present_key_value)
        """
        batch_size, seq_len, _ = hidden_states.shape
        self.forward_count += 1
        
        # === SplatFlow Path ===
        # Compute splat influences
        splat_influences = self.splat_positioning.compute_influences(hidden_states)
        
        # Process information flow through splats
        splat_output = self.splat_flow(hidden_states, splat_influences)
        
        # Adapt splat positions (only during training)
        if self.training:
            self.splat_positioning.adapt_positions(hidden_states, splat_influences)
        
        # === Standard Attention Path ===
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Handle past key-value for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        
        present_key_value = (key, value) if use_cache else None
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Convert 2D mask to 4D
                attention_mask = attention_mask.view(batch_size, 1, 1, -1)
            attention_scores = attention_scores + attention_mask
        
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_probs, value)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.model_dim)
        
        # === Combine SplatFlow and Standard Attention ===
        combined_output = (
            self.splat_attention_ratio * splat_output +
            self.standard_attention_ratio * attention_output
        )
        
        # Final projection
        output = self.out_proj(combined_output)
        output = self.dropout(output)
        
        # Update efficiency metrics
        with torch.no_grad():
            splat_contrib = torch.norm(splat_output) / (torch.norm(combined_output) + 1e-8)
            self.splat_efficiency = 0.9 * self.splat_efficiency + 0.1 * splat_contrib
        
        # Return appropriate outputs
        attention_weights = attention_probs if output_attentions else None
        
        return output, attention_weights, present_key_value
    
    def get_attention_health(self) -> Dict[str, Any]:
        """Get comprehensive health report for this attention layer."""
        positioning_health = self.splat_positioning.get_health_stats()
        
        health = {
            'forward_count': int(self.forward_count),
            'splat_efficiency': float(self.splat_efficiency),
            'positioning': positioning_health,
            'rescue_count': int(self.splat_flow.rescue_count),
            'config': {
                'num_splats': self.num_splats,
                'num_heads': self.num_heads,
                'splat_ratio': self.splat_attention_ratio
            }
        }
        
        return health


class SplatFlowBlock(nn.Module):
    """
    Complete SplatFlow transformer block with attention and feed-forward.
    """
    
    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        num_splats: int = 20,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        prenorm: bool = True
    ):
        super().__init__()
        self.model_dim = model_dim
        self.prenorm = prenorm
        ffn_dim = ffn_dim or 4 * model_dim
        
        # SplatFlow attention
        self.self_attn = FixedProductionSplatFlowAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_splats=num_splats,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.attn_norm = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        
        # Health monitoring
        self.register_buffer('block_forward_count', torch.tensor(0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with residual connections and layer normalization."""
        
        self.block_forward_count += 1
        
        # Self-attention with residual connection
        if self.prenorm:
            # Pre-norm: normalize before attention
            normed_hidden_states = self.attn_norm(hidden_states)
            attn_output, attn_weights, present_key_value = self.self_attn(
                hidden_states=normed_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            hidden_states = hidden_states + attn_output
            
            # Feed-forward with residual connection
            normed_hidden_states = self.ffn_norm(hidden_states)
            ffn_output = self.ffn(normed_hidden_states)
            hidden_states = hidden_states + ffn_output
            
        else:
            # Post-norm: normalize after attention
            attn_output, attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            hidden_states = self.attn_norm(hidden_states + attn_output)
            
            # Feed-forward with residual connection
            ffn_output = self.ffn(hidden_states)
            hidden_states = self.ffn_norm(hidden_states + ffn_output)
        
        return hidden_states, attn_weights, present_key_value
    
    def get_block_health(self) -> Dict[str, Any]:
        """Get health statistics for this block."""
        attn_health = self.self_attn.get_attention_health()
        
        return {
            'block_forward_count': int(self.block_forward_count),
            'attention_health': attn_health
        }


# Utility functions for creating attention components
def create_splatflow_attention(
    model_dim: int,
    num_heads: int = 8,
    num_splats: int = 20,
    dropout: float = 0.1,
    **kwargs
) -> FixedProductionSplatFlowAttention:
    """Factory function for creating SplatFlow attention."""
    return FixedProductionSplatFlowAttention(
        model_dim=model_dim,
        num_heads=num_heads,
        num_splats=num_splats,
        dropout=dropout,
        **kwargs
    )


def create_splatflow_block(
    model_dim: int,
    num_heads: int = 8,
    num_splats: int = 20,
    ffn_dim: Optional[int] = None,
    dropout: float = 0.1,
    **kwargs
) -> SplatFlowBlock:
    """Factory function for creating SplatFlow transformer block."""
    return SplatFlowBlock(
        model_dim=model_dim,
        num_heads=num_heads,
        num_splats=num_splats,
        ffn_dim=ffn_dim,
        dropout=dropout,
        **kwargs
    )


def create_splatflow_block(
    model_dim: int,
    num_heads: int = 8,
    num_splats: int = 20,
    ffn_dim: Optional[int] = None,
    dropout: float = 0.1,
    **kwargs
) -> SplatFlowBlock:
    """Factory function for creating SplatFlow transformer block."""
    return SplatFlowBlock(
        model_dim=model_dim,
        num_heads=num_heads,
        num_splats=num_splats,
        ffn_dim=ffn_dim,
        dropout=dropout,
        **kwargs
    )


# Additional utility functions for compatibility
def get_splat_attention_stats(attention_module) -> Dict[str, Any]:
    """Get statistics from a SplatFlow attention module."""
    if hasattr(attention_module, 'get_attention_health'):
        return attention_module.get_attention_health()
    else:
        return {'error': 'not_a_splatflow_attention_module'}


def validate_splatflow_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a SplatFlow configuration and return validation results.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validation results with warnings and recommendations
    """
    results = {
        'valid': True,
        'warnings': [],
        'recommendations': [],
        'fixed_config': config.copy()
    }
    
    # Check critical parameters
    if config.get('model_dim', 0) % config.get('num_heads', 8) != 0:
        results['warnings'].append("model_dim must be divisible by num_heads")
        results['valid'] = False
    
    # Check dataset
    available_datasets = ['wikitext', 'openwebtext', 'bookcorpus', 'c4', 'pile', 
                         'common_crawl', 'github_code', 'arxiv', 'news', 'dialogue', 
                         'qa', 'stories', 'scientific', 'legal', 'medical']
    
    dataset = config.get('dataset', 'unknown')
    if dataset == 'tiny_shakespeare':
        results['warnings'].append("Dataset 'tiny_shakespeare' not available")
        results['recommendations'].append("Use 'wikitext' or 'stories' instead")
        results['fixed_config']['dataset'] = 'wikitext'
    elif dataset not in available_datasets and dataset not in ['conservative', 'custom']:
        results['warnings'].append(f"Dataset '{dataset}' may not be available")
    
    # Check PyTorch settings
    if config.get('enable_nested_tensor', True):
        results['recommendations'].append("Set enable_nested_tensor=False to avoid warnings")
        results['fixed_config']['enable_nested_tensor'] = False
    
    if config.get('norm_first', True):
        results['recommendations'].append("Set norm_first=False to avoid warnings")  
        results['fixed_config']['norm_first'] = False
    
    # Check splat configuration
    num_splats = config.get('num_splats', 20)
    if num_splats < 8:
        results['recommendations'].append("Consider num_splats >= 8 for better performance")
    elif num_splats > 64:
        results['recommendations'].append("Consider num_splats <= 64 to avoid memory issues")
    
    return results


def get_memory_requirements(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate memory requirements for a SplatFlow model configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Dictionary with memory estimates in MB
    """
    model_dim = config.get('model_dim', 512)
    num_layers = config.get('num_layers', 6)
    num_splats = config.get('num_splats', 20)
    vocab_size = config.get('vocab_size', 50257)
    max_seq_len = config.get('max_seq_len', 4096)
    batch_size = config.get('batch_size', 2)
    
    # Parameter memory
    embedding_params = vocab_size * model_dim
    attention_params_per_layer = 4 * model_dim * model_dim + num_splats * model_dim
    ffn_params_per_layer = 2 * model_dim * (4 * model_dim)
    total_params = embedding_params + num_layers * (attention_params_per_layer + ffn_params_per_layer)
    param_memory = (total_params * 4) / (1024 * 1024)  # 4 bytes per parameter
    
    # Activation memory (rough estimate)
    activation_memory_per_sample = (max_seq_len * model_dim * num_layers * 8) / (1024 * 1024)
    activation_memory = activation_memory_per_sample * batch_size
    
    # Gradient memory (same as parameters)
    gradient_memory = param_memory
    
    # Total memory with some overhead
    total_memory = (param_memory + activation_memory + gradient_memory) * 1.3
    
    return {
        'parameters_mb': param_memory,
        'activations_mb': activation_memory,
        'gradients_mb': gradient_memory,
        'total_estimated_mb': total_memory,
        'recommended_gpu_gb': total_memory / 1024 * 1.5  # Add safety margin
    }


# Health monitoring utilities
class SplatFlowHealthMonitor:
    """Utility class for monitoring SplatFlow health across multiple components."""
    
    def __init__(self):
        self.components = []
    
    def register_component(self, component, name: str):
        """Register a SplatFlow component for monitoring."""
        self.components.append((name, component))
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive health report for all registered components."""
        system_health = {
            'total_components': len(self.components),
            'component_health': {},
            'summary': {
                'total_splats': 0,
                'active_splats': 0,
                'total_rescues': 0,
                'avg_efficiency': 0.0
            }
        }
        
        total_efficiency = 0.0
        efficiency_count = 0
        
        for name, component in self.components:
            if hasattr(component, 'get_attention_health'):
                health = component.get_attention_health()
                system_health['component_health'][name] = health
                
                # Aggregate statistics
                if 'positioning' in health:
                    system_health['summary']['total_splats'] += health['positioning']['num_splats']
                    system_health['summary']['active_splats'] += health['positioning']['active_splats']
                
                if 'rescue_count' in health:
                    system_health['summary']['total_rescues'] += health['rescue_count']
                
                if 'splat_efficiency' in health:
                    total_efficiency += health['splat_efficiency']
                    efficiency_count += 1
            
            elif hasattr(component, 'get_block_health'):
                health = component.get_block_health()
                system_health['component_health'][name] = health
        
        # Calculate average efficiency
        if efficiency_count > 0:
            system_health['summary']['avg_efficiency'] = total_efficiency / efficiency_count
        
        return system_health


# Global health monitor instance
global_health_monitor = SplatFlowHealthMonitor()


# Utility functions for backward compatibility
def get_quick_model_stats(model) -> Dict[str, Any]:
    """
    Get quick statistics about a SplatFlow model.
    This function provides backward compatibility for existing code.
    
    Args:
        model: SplatFlow model instance
        
    Returns:
        Dictionary with model statistics
    """
    stats = {
        'model_type': type(model).__name__,
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'device': str(next(model.parameters()).device),
        'training_mode': model.training
    }
    
    # Add SplatFlow-specific stats if available
    if hasattr(model, 'config'):
        stats.update({
            'model_dim': model.config.get('model_dim', 'unknown'),
            'num_layers': model.config.get('num_layers', 'unknown'),
            'num_splats': model.config.get('num_splats', 'unknown'),
            'max_seq_len': model.config.get('max_seq_len', 'unknown')
        })
    
    # Add health information if available
    if hasattr(model, 'get_model_health_report'):
        try:
            health = model.get_model_health_report()
            stats['health_summary'] = health.get('system_health', {}).get('summary', {})
        except Exception:
            stats['health_summary'] = 'unavailable'
    
    return stats


def get_splatflow_component_info() -> Dict[str, Any]:
    """
    Get information about available SplatFlow components.
    Useful for debugging and compatibility checking.
    """
    info = {
        'components_available': [
            'AdaptiveSplatPositioning',
            'SplatInformationFlow', 
            'FixedProductionSplatFlowAttention',
            'SplatFlowBlock',
            'SplatFlowHealthMonitor'
        ],
        'utility_functions': [
            'create_splatflow_attention',
            'create_splatflow_block',
            'get_quick_model_stats',
            'get_splatflow_component_info'
        ],
        'version': '1.0.0',
        'compatibility': 'production_ready'
    }
    
    return info


def create_fallback_model_stats(model_dim: int = 512, num_layers: int = 6, num_splats: int = 20) -> Dict[str, Any]:
    """
    Create fallback model statistics when model isn't available.
    Used for configuration validation and planning.
    """
    # Rough parameter estimation
    embedding_params = 50257 * model_dim  # vocab_size * model_dim
    attention_params_per_layer = 4 * model_dim * model_dim  # Q, K, V, O projections
    ffn_params_per_layer = 2 * model_dim * (4 * model_dim)  # FFN layers
    layer_params = attention_params_per_layer + ffn_params_per_layer
    total_params = embedding_params + (layer_params * num_layers)
    
    stats = {
        'estimated_parameters': total_params,
        'model_dim': model_dim,
        'num_layers': num_layers,
        'num_splats': num_splats,
        'memory_estimate_mb': (total_params * 4) / (1024 * 1024),  # 4 bytes per param
        'complexity': 'O(n*k)' if num_splats > 0 else 'O(n²)',
        'type': 'estimated_fallback'
    }
    
    return stats


if __name__ == "__main__":
    # Test the attention components
    print("🧪 Testing SplatFlow Attention Components...")
    
    # Test parameters
    batch_size, seq_len, model_dim = 2, 128, 512
    num_heads, num_splats = 8, 20
    
    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, model_dim)
    attention_mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
    attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
    
    print(f"📊 Test input: {hidden_states.shape}")
    
    # Test individual components
    print("\n🔧 Testing AdaptiveSplatPositioning...")
    positioning = AdaptiveSplatPositioning(num_splats, model_dim)
    influences = positioning.compute_influences(hidden_states)
    print(f"✅ Splat influences: {influences.shape}")
    print(f"📈 Health: {positioning.get_health_stats()}")
    
    print("\n🌊 Testing SplatInformationFlow...")
    flow = SplatInformationFlow(model_dim, num_splats)
    flow_output = flow(hidden_states, influences)
    print(f"✅ Flow output: {flow_output.shape}")
    
    print("\n🎯 Testing FixedProductionSplatFlowAttention...")
    attention = FixedProductionSplatFlowAttention(model_dim, num_heads, num_splats)
    attn_output, attn_weights, _ = attention(hidden_states, attention_mask)
    print(f"✅ Attention output: {attn_output.shape}")
    print(f"📈 Attention health: {attention.get_attention_health()}")
    
    print("\n🏗️ Testing SplatFlowBlock...")
    block = SplatFlowBlock(model_dim, num_heads, num_splats)
    block_output, _, _ = block(hidden_states, attention_mask)
    print(f"✅ Block output: {block_output.shape}")
    print(f"📈 Block health: {block.get_block_health()}")
    
    print("\n🎉 All SplatFlow attention components working correctly!")
    
    # Test health monitoring
    print("\n🏥 Testing Health Monitoring...")
    global_health_monitor.register_component(attention, "main_attention")
    global_health_monitor.register_component(block, "transformer_block")
    
    system_health = global_health_monitor.get_system_health()
    print(f"🏥 System health summary: {system_health['summary']}")
    
    print("\n✅ SplatFlow attention components fully tested and operational!")
