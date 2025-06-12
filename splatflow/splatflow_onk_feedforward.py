"""
SplatFlow O(n*k) Feed-Forward Components - Phase 2 Enhanced
FIXED: Defensive parameter handling to prevent dropout conflicts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class SplatAwareFeedForward(nn.Module):
    """
    O(n*k) splat-aware feed-forward network that processes tokens based on splat groupings.
    Achieves computational efficiency through intelligent grouping and selective processing.
    """
    
    def __init__(self, model_dim: int, num_splats: int, ff_dim: Optional[int] = None,
                 layer_idx: int = 0, dropout: float = 0.1, **kwargs):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_splats = num_splats
        self.ff_dim = ff_dim or model_dim * 4
        self.layer_idx = layer_idx
        self.dropout = dropout
        
        # Remove any conflicting parameters from kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['dropout', 'model_dim', 'num_splats', 'ff_dim', 'layer_idx']}
        
        # Core feed-forward networks per splat group
        self.splat_networks = nn.ModuleList()
        for i in range(num_splats):
            network = nn.Sequential(
                nn.Linear(model_dim, self.ff_dim // num_splats),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim // num_splats, model_dim)
            )
            self.splat_networks.append(network)
        
        # Global aggregation network
        self.global_aggregator = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim)
        )
        
        # Processing statistics
        self.processing_stats = {
            'total_calls': 0,
            'splat_utilization': defaultdict(int),
            'efficiency_scores': []
        }
        
        logger.info(f"🚀 SplatAware FF initialized: {model_dim}d -> {self.ff_dim}d with {num_splats} splats")
    
    def forward(self, x: torch.Tensor, 
                splat_groups: Optional[Dict] = None,
                attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with splat-aware processing
        
        Args:
            x: Input tensor [batch_size, seq_len, model_dim]
            splat_groups: Grouping information from attention
            attention_weights: Attention weights for efficiency calculation
        """
        
        self.processing_stats['total_calls'] += 1
        
        try:
            if splat_groups and len(splat_groups) > 0:
                return self._splat_aware_processing(x, splat_groups, attention_weights)
            else:
                return self._fallback_processing(x)
                
        except Exception as e:
            logger.warning(f"SplatAware FF processing failed: {e}")
            return self._fallback_processing(x)
    
    def _splat_aware_processing(self, x: torch.Tensor, splat_groups: Dict,
                               attention_weights: Optional[torch.Tensor]) -> torch.Tensor:
        """Process tokens with splat grouping awareness"""
        
        batch_size, seq_len, model_dim = x.shape
        output = torch.zeros_like(x)
        
        # Process each splat group
        for splat_idx, token_indices in splat_groups.items():
            if isinstance(token_indices, torch.Tensor) and len(token_indices) > 0:
                # Extract tokens for this splat
                splat_tokens = x[:, token_indices, :]
                
                # Process with corresponding splat network
                network_idx = min(splat_idx, len(self.splat_networks) - 1)
                processed_tokens = self.splat_networks[network_idx](splat_tokens)
                
                # Place back in output
                output[:, token_indices, :] = processed_tokens
                
                # Update statistics
                self.processing_stats['splat_utilization'][splat_idx] += len(token_indices)
        
        # Global aggregation
        output = self.global_aggregator(output)
        
        return output
    
    def _fallback_processing(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback to standard processing when splat info unavailable"""
        
        # Use first splat network as fallback
        processed = self.splat_networks[0](x)
        return self.global_aggregator(processed)
    
    def get_processing_statistics(self) -> Dict:
        """Get detailed processing statistics"""
        
        total_utilization = sum(self.processing_stats['splat_utilization'].values())
        
        return {
            'type': 'SplatAware FeedForward',
            'model_dim': self.model_dim,
            'ff_dim': self.ff_dim,
            'num_splats': self.num_splats,
            'total_calls': self.processing_stats['total_calls'],
            'splat_utilization': dict(self.processing_stats['splat_utilization']),
            'avg_utilization_per_call': total_utilization / max(self.processing_stats['total_calls'], 1),
            'optimization_type': 'O(n*k) Splat-Aware'
        }


class SplatSpecializedFFBank(nn.Module):
    """
    Bank of specialized feed-forward networks, each optimized for different content patterns.
    Each specialization handles different content patterns optimally.
    """
    
    def __init__(self, model_dim: int, num_specializations: int = 4, 
                 ff_dim: Optional[int] = None, dropout: float = 0.1, **kwargs):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_specializations = num_specializations
        self.ff_dim = ff_dim or model_dim * 4
        self.dropout = dropout
        
        # Remove any conflicting parameters from kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['dropout', 'model_dim', 'num_specializations', 'ff_dim']}
        
        # Define specialization types
        self.specialization_types = [
            "syntax",      # Grammar, punctuation, structure
            "semantics",   # Meaning, concepts, entities
            "long_range",  # Dependencies across long distances
            "general"      # Catch-all for unspecialized content
        ][:num_specializations]
        
        # Create specialized networks
        self.specialized_networks = nn.ModuleDict()
        for spec_type in self.specialization_types:
            self.specialized_networks[spec_type] = self._create_specialized_network(spec_type)
        
        # Specialization classifier
        self.specialization_classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, num_specializations),
            nn.Softmax(dim=-1)
        )
        
        # Adaptive mixing weights
        self.mixing_weights = nn.Parameter(torch.ones(num_specializations) / num_specializations)
        
        # Statistics tracking
        self.specialization_usage = defaultdict(int)
        self.specialization_performance = defaultdict(list)
        
        logger.info(f"🏦 SplatSpecialized FF Bank initialized: {num_specializations} specializations")
    
    def _create_specialized_network(self, spec_type: str) -> nn.Module:
        """Create a specialized network based on content type"""
        
        if spec_type == "syntax":
            # Syntax processing: smaller, faster networks for structural patterns
            return nn.Sequential(
                nn.Linear(self.model_dim, self.ff_dim // 2),
                nn.ReLU(),  # ReLU for syntax (sharper decisions)
                nn.Dropout(self.dropout),
                nn.Linear(self.ff_dim // 2, self.model_dim)
            )
        
        elif spec_type == "semantics":
            # Semantic processing: larger networks for complex meaning
            return nn.Sequential(
                nn.Linear(self.model_dim, self.ff_dim),
                nn.GELU(),  # GELU for smoother semantic processing
                nn.Dropout(self.dropout),
                nn.Linear(self.ff_dim, self.model_dim)
            )
        
        elif spec_type == "long_range":
            # Long-range dependencies: specialized architecture
            return nn.Sequential(
                nn.Linear(self.model_dim, self.ff_dim // 4),
                nn.GELU(),
                nn.Linear(self.ff_dim // 4, self.ff_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.ff_dim, self.model_dim)
            )
        
        else:  # general
            # Standard processing for unspecialized content
            return nn.Sequential(
                nn.Linear(self.model_dim, self.ff_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.ff_dim, self.model_dim)
            )
    
    def forward(self, x: torch.Tensor, 
                splat_groups: Optional[Dict] = None,
                attention_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with specialization selection"""
        
        try:
            # Classify specialization needs
            specialization_scores = self.specialization_classifier(x)
            
            # Process through each specialization
            specialized_outputs = []
            for i, spec_type in enumerate(self.specialization_types):
                network_output = self.specialized_networks[spec_type](x)
                specialized_outputs.append(network_output)
                
                # Update usage statistics
                usage_weight = specialization_scores[:, :, i].mean().item()
                self.specialization_usage[spec_type] += usage_weight
            
            # Weighted combination of outputs
            specialized_outputs = torch.stack(specialized_outputs, dim=-1)  # [B, S, D, num_spec]
            specialization_scores = specialization_scores.unsqueeze(-2)  # [B, S, 1, num_spec]
            
            # Weighted sum
            output = (specialized_outputs * specialization_scores).sum(dim=-1)
            
            return output
            
        except Exception as e:
            logger.warning(f"Specialized FF processing failed: {e}")
            # Fallback to general network
            return self.specialized_networks['general'](x)
    
    def get_specialization_statistics(self) -> Dict:
        """Get specialization usage statistics"""
        
        total_usage = sum(self.specialization_usage.values())
        
        return {
            'type': 'SplatSpecialized FF Bank',
            'num_specializations': self.num_specializations,
            'specialization_types': self.specialization_types,
            'usage_distribution': {
                spec: usage / max(total_usage, 1) 
                for spec, usage in self.specialization_usage.items()
            },
            'most_used_specialization': max(self.specialization_usage.items(), key=lambda x: x[1])[0] if self.specialization_usage else 'none',
            'optimization_type': 'Specialized FF Bank'
        }


class HierarchicalFFProcessor(nn.Module):
    """
    Hierarchical feed-forward processor that adapts computational complexity 
    based on token importance and attention patterns.
    """
    
    def __init__(self, model_dim: int, max_ff_dim: int = None, 
                 num_levels: int = 3, dropout: float = 0.1, **kwargs):
        super().__init__()
        
        self.model_dim = model_dim
        self.max_ff_dim = max_ff_dim or model_dim * 4
        self.num_levels = num_levels
        self.dropout = dropout
        
        # Remove any conflicting parameters from kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['dropout', 'model_dim', 'max_ff_dim', 'num_levels']}
        
        # Create hierarchical processing levels
        self.processing_levels = nn.ModuleList()
        for level in range(num_levels):
            # Increasing complexity for higher importance levels
            ff_dim = self.max_ff_dim // (2 ** (num_levels - level - 1))
            ff_dim = max(ff_dim, model_dim)  # Minimum size
            
            level_processor = nn.Sequential(
                nn.Linear(model_dim, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, model_dim)
            )
            self.processing_levels.append(level_processor)
        
        # Importance classifier
        self.importance_classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, num_levels),
            nn.Softmax(dim=-1)
        )
        
        # Adaptive thresholds
        self.importance_thresholds = nn.Parameter(
            torch.linspace(0.2, 0.8, num_levels)
        )
        
        logger.info(f"🏗️ Hierarchical FF Processor initialized: {num_levels} levels")
    
    def forward(self, token_embeddings: torch.Tensor,
                attention_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process tokens through hierarchical levels based on importance
        
        Args:
            token_embeddings: [batch_size, seq_len, model_dim]
            attention_scores: [batch_size, seq_len] importance scores
        """
        
        try:
            # Determine processing levels
            if attention_scores is not None:
                processing_levels = self._assign_processing_levels(attention_scores)
            else:
                # Use importance classifier
                importance_scores = self.importance_classifier(token_embeddings)
                processing_levels = torch.argmax(importance_scores, dim=-1)
            
            # Process tokens by level
            output = torch.zeros_like(token_embeddings)
            
            for level in range(self.num_levels):
                level_mask = (processing_levels == level)
                if level_mask.any():
                    level_tokens = token_embeddings[level_mask]
                    processed_tokens = self.processing_levels[level](level_tokens)
                    output[level_mask] = processed_tokens
            
            return output
            
        except Exception as e:
            logger.warning(f"Hierarchical FF processing failed: {e}")
            # Fallback to highest level processing
            return self.processing_levels[-1](token_embeddings)
    
    def _assign_processing_levels(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """Assign processing levels based on attention scores"""
        
        batch_size, seq_len = attention_scores.shape
        levels = torch.zeros_like(attention_scores, dtype=torch.long)
        
        # Create cumulative thresholds
        thresholds = self.importance_thresholds.detach()
        cumulative_thresholds = torch.cumsum(thresholds, dim=0)
        
        for level in range(self.num_levels):
            if level == 0:
                mask = attention_scores <= cumulative_thresholds[level]
            else:
                mask = (attention_scores > cumulative_thresholds[level-1]) & \
                       (attention_scores <= cumulative_thresholds[level])
            
            levels[mask] = level
        
        return levels
    
    def get_hierarchy_statistics(self) -> Dict:
        """Get hierarchical processing statistics"""
        
        try:
            ff_dims = []
            for level_processor in self.processing_levels:
                # Get FF dimension from first linear layer
                first_linear = level_processor[0]
                ff_dims.append(first_linear.out_features)
            
            return {
                'num_levels': self.num_levels,
                'ff_dimensions_by_level': ff_dims,
                'importance_thresholds': self.importance_thresholds.detach().cpu().tolist(),
                'max_ff_dim': self.max_ff_dim,
                'complexity_ratio': max(ff_dims) / min(ff_dims) if ff_dims else 1.0,
                'optimization_type': 'Hierarchical FF Processor'
            }
        except Exception as e:
            logger.warning(f"Failed to get hierarchy statistics: {e}")
            return {
                'error': str(e),
                'optimization_type': 'Hierarchical FF Processor'
            }


# FIXED: Utility functions for integration with proper parameter handling

def create_onk_feedforward_layer(model_dim: int, num_splats: int, 
                                ff_type: str = "splat_aware",
                                layer_idx: int = 0, 
                                dropout: float = 0.1,
                                **kwargs) -> nn.Module:
    """
    FIXED: Factory function to create O(n*k) feed-forward layers with defensive parameter handling
    
    Args:
        model_dim: Model dimension
        num_splats: Number of splats for splat-aware processing
        ff_type: Type of FF layer ('splat_aware', 'specialized', 'hierarchical')
        layer_idx: Layer index for initialization
        dropout: Dropout rate (explicit parameter takes precedence)
        **kwargs: Additional arguments (conflicting parameters removed)
    
    Returns:
        Configured feed-forward layer
    """
    
    # DEFENSIVE FIX: Remove any duplicate parameters from kwargs to prevent conflicts
    conflicting_params = ['dropout', 'model_dim', 'num_splats', 'layer_idx']
    clean_kwargs = {k: v for k, v in kwargs.items() if k not in conflicting_params}
    
    # Log if we found conflicting parameters
    for param in conflicting_params:
        if param in kwargs:
            logger.warning(f"Removed conflicting parameter '{param}' from kwargs - using explicit value")
    
    try:
        if ff_type == "splat_aware":
            return SplatAwareFeedForward(
                model_dim=model_dim, 
                num_splats=num_splats, 
                layer_idx=layer_idx, 
                dropout=dropout,
                **clean_kwargs
            )
        
        elif ff_type == "specialized":
            return SplatSpecializedFFBank(
                model_dim=model_dim, 
                dropout=dropout,
                **clean_kwargs
            )
        
        elif ff_type == "hierarchical":
            return HierarchicalFFProcessor(
                model_dim=model_dim, 
                dropout=dropout,
                **clean_kwargs
            )
        
        else:
            logger.warning(f"Unknown FF type '{ff_type}', defaulting to splat_aware")
            return SplatAwareFeedForward(
                model_dim=model_dim, 
                num_splats=num_splats, 
                layer_idx=layer_idx, 
                dropout=dropout,
                **clean_kwargs
            )
            
    except Exception as e:
        logger.error(f"Failed to create {ff_type} feedforward layer: {e}")
        # Emergency fallback to simple feedforward
        return nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim)
        )


def get_onk_feedforward_statistics(ff_layer: nn.Module) -> Dict:
    """Get statistics from any O(n*k) feed-forward layer"""
    
    try:
        if hasattr(ff_layer, 'get_processing_statistics'):
            return ff_layer.get_processing_statistics()
        elif hasattr(ff_layer, 'get_specialization_statistics'):
            return ff_layer.get_specialization_statistics()
        elif hasattr(ff_layer, 'get_hierarchy_statistics'):
            return ff_layer.get_hierarchy_statistics()
        else:
            return {
                'type': type(ff_layer).__name__,
                'parameters': sum(p.numel() for p in ff_layer.parameters()),
                'optimization_type': 'Unknown O(n*k) FF'
            }
    except Exception as e:
        logger.warning(f"Failed to get FF statistics: {e}")
        return {
            'error': str(e),
            'type': type(ff_layer).__name__,
            'optimization_type': 'O(n*k) FF (Error)'
        }


def validate_feedforward_config(config: Dict) -> Dict:
    """Validate and clean feedforward configuration to prevent parameter conflicts"""
    
    cleaned_config = config.copy()
    
    # Remove problematic parameters that should be passed explicitly
    problematic_params = ['dropout', 'model_dim', 'num_splats', 'layer_idx']
    
    if 'feedforward_kwargs' in cleaned_config:
        feedforward_kwargs = cleaned_config['feedforward_kwargs']
        
        for param in problematic_params:
            if param in feedforward_kwargs:
                removed_value = feedforward_kwargs.pop(param)
                logger.warning(f"Removed '{param}={removed_value}' from feedforward_kwargs to prevent conflicts")
        
        cleaned_config['feedforward_kwargs'] = feedforward_kwargs
    
    return cleaned_config
