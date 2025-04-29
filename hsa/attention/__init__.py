"""
Attention submodule for Hierarchical Splat Attention (HSA).

This module provides implementations of hierarchical attention mechanisms.
"""

# Import from renamed modules to match file names
from .base import AttentionComputer, mahalanobis_batch, gauss_kernel_batch, find_relevant_tokens
from .implementations import DenseAttentionComputer, SparseAttentionComputer, SpatialAttentionComputer
from .metrics import SplatAttentionMetrics
from .factory import create_attention_computer
from .pytorch import HSAMultiheadAttention

__all__ = [
    'AttentionComputer',
    'DenseAttentionComputer',
    'SparseAttentionComputer',
    'SpatialAttentionComputer',
    'SplatAttentionMetrics',
    'create_attention_computer',
    'HSAMultiheadAttention',
    'mahalanobis_batch',
    'gauss_kernel_batch',
    'find_relevant_tokens'
]
