"""
Core computation methods for sparse attention in Hierarchical Splat Attention (HSA).

This module provides the implementation details for computing attention in different
scenarios, including dense computation for small inputs and sparse computation for
larger inputs.
"""

from typing import Dict, List, Optional, Tuple, Set, Any, Callable
import numpy as np
import scipy.sparse as sp
import logging

from .splat import Splat
from .registry import SplatRegistry
from .attention_interface import AttentionConfig
from .attention_cache import AttentionCache
from .sparse_attention_utils import (
    compute_token_relevance, create_sparse_matrix_from_values,
    find_relevant_splats_for_token, create_causal_mask
)

# Configure logging
logger = logging.getLogger(__name__)


def compute_dense_attention_map(
    tokens: np.ndarray,
    splat: Splat,
    max_tokens: Optional[int] = None,
    cache: Optional[AttentionCache] = None
) -> np.ndarray:
    """Compute attention map using dense computation.
    
    Args:
        tokens: Token embeddings of shape [seq_len, embedding_dim]
        splat: Splat to compute attention for
        max_tokens: Maximum number of tokens to compute
        cache: Optional cache for attention computations
        
    Returns:
        Attention contribution matrix
    """
    seq_len = tokens.shape[0]
    
    # Limit computation to max_tokens if specified
    if max_tokens is not None and seq_len > max_tokens:
        # Sample tokens uniformly
        indices = np.linspace(0, seq_len - 1, max_tokens, dtype=int)
        sampled_tokens = tokens[indices]
        
        # Use vectorized batch computation
        if hasattr(splat, 'compute_attention_batch'):
            sampled_attention = splat.compute_attention_batch(sampled_tokens)
        else:
            # Fall back to pairwise computation
            sampled_attention = np.zeros((max_tokens, max_tokens))
            for i in range(max_tokens):
                for j in range(max_tokens):
                    sampled_attention[i, j] = splat.compute_attention(
                        sampled_tokens[i], sampled_tokens[j]
                    )
        
        # Expand back to full size using nearest neighbor
        attention = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            i_sampled = np.argmin(np.abs(i - indices))
            for j in range(seq_len):
                j_sampled = np.argmin(np.abs(j - indices))
                attention[i, j] = sampled_attention[i_sampled, j_sampled]
        
        return attention
    
    # Use vectorized computation if available
    if hasattr(splat, 'compute_attention_batch'):
        return splat.compute_attention_batch(tokens)
    
    # Fall back to original pairwise computation
    attention = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            attention[i, j] = splat.compute_attention(tokens[i], tokens[j])
    
    return attention


def compute_optimized_attention_map(
    tokens: np.ndarray,
    splat: Splat,
    relevant_indices: Optional[np.ndarray] = None,
    relevant_tokens: Optional[np.ndarray] = None,
    sparsity_threshold: float = 0.01
) -> np.ndarray:
    """Compute attention map using optimized sparse methods.
    
    Args:
        tokens: Token embeddings of shape [seq_len, embedding_dim]
        splat: Splat to compute attention for
        relevant_indices: Optional indices of relevant tokens
        relevant_tokens: Optional embeddings of relevant tokens
        sparsity_threshold: Threshold below which values are considered zero
        
    Returns:
        Attention contribution matrix
    """
    seq_len = tokens.shape[0]
    
    # If relevant indices are provided, compute sparse attention
    if relevant_indices is not None and relevant_tokens is not None:
        # Create sparse matrix
        data = []
        row_ind = []
        col_ind = []
        
        # Compute attentions between relevant tokens
        for i, idx_i in enumerate(relevant_indices):
            for j, idx_j in enumerate(relevant_indices):
                # Compute attention
                att_value = splat.compute_attention(
                    relevant_tokens[i], relevant_tokens[j]
                )
                
                # Only store non-zero values
                if att_value >= sparsity_threshold:
                    data.append(att_value)
                    row_ind.append(idx_i)
                    col_ind.append(idx_j)
        
        # Create sparse matrix and convert to dense
        return create_sparse_matrix_from_values(
            data, row_ind, col_ind, (seq_len, seq_len)
        )
    
    # Otherwise, compute using token relevance approximation
    
    # Compute token relevance to this splat
    token_relevance = compute_token_relevance(tokens, splat)
    
    # Compute attention matrix using outer product approximation
    # This is a mathematical simplification of the full computation
    # that works well in practice
    attention = np.outer(token_relevance, token_relevance)
    
    # Scale by amplitude (if not already included in token_relevance)
    if not hasattr(splat, 'amplitude') or splat.amplitude != 1.0:
        attention *= splat.amplitude
    
    return attention


def compute_dense_attention(
    tokens: np.ndarray,
    splat_registry: SplatRegistry,
    config: AttentionConfig,
    splat_attention_fn: Callable
) -> np.ndarray:
    """Compute attention using dense methods.
    
    Args:
        tokens: Token embeddings of shape [seq_len, embedding_dim]
        splat_registry: Registry containing splats to use for attention
        config: Attention configuration
        splat_attention_fn: Function to compute attention for a single splat
        
    Returns:
        Attention matrix of shape [seq_len, seq_len]
    """
    seq_len = tokens.shape[0]
    
    # Initialize attention matrix
    attention_matrix = np.zeros((seq_len, seq_len))
    
    # Get hierarchy and level weights
    hierarchy = splat_registry.hierarchy
    level_weights = {}
    
    if config.level_weights:
        # Use custom weights if provided
        level_weights = config.level_weights
    else:
        # Use hierarchy defaults
        for level in hierarchy.levels:
            level_weights[level] = hierarchy.get_level_weight(level)
    
    # Compute attention per level
    for level in hierarchy.levels:
        level_splats = list(splat_registry.get_splats_at_level(level))
        
        # Skip empty levels
        if not level_splats:
            continue
        
        # Compute level attention efficiently
        level_attention = compute_level_attention_dense(
            tokens, level_splats, splat_attention_fn
        )
        
        # Normalize level attention if requested
        if config.normalize_levels:
            max_val = np.max(level_attention)
            if max_val > 0:
                level_attention = level_attention / max_val
        
        # Apply causal mask if requested
        if config.causal:
            causal_mask = create_causal_mask(seq_len)
            level_attention *= causal_mask
        
        # Apply level weight
        weighted_level_attention = level_attention * level_weights[level]
        
        # Add to total attention
        attention_matrix += weighted_level_attention
    
    # Apply top-k sparsity if configured
    if config.topk is not None:
        attention_matrix = apply_topk_sparsity(attention_matrix, config.topk)
    elif config.threshold is not None:
        # Apply threshold sparsity
        attention_matrix = np.where(
            attention_matrix >= config.threshold, 
            attention_matrix, 
            0.0
        )
    
    # Normalize rows efficiently
    if config.normalize_rows:
        row_sums = np.sum(attention_matrix, axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        attention_matrix = attention_matrix / row_sums
    
    return attention_matrix


def compute_level_attention_dense(
    tokens: np.ndarray, 
    level_splats: List[Splat],
    splat_attention_fn: Callable
) -> np.ndarray:
    """Compute attention for all splats at a specific level using dense methods.
    
    Args:
        tokens: Token embeddings of shape [seq_len, embedding_dim]
        level_splats: List of splats at this level
        splat_attention_fn: Function to compute attention for a single splat
        
    Returns:
        Level attention matrix of shape [seq_len, seq_len]
    """
    seq_len = tokens.shape[0]
    level_attention = np.zeros((seq_len, seq_len))
    
    # Shared cache for this computation
    shared_cache = {}
    
    # Process splats in chunks to balance memory usage and performance
    splat_chunks = [level_splats[i:i+10] for i in range(0, len(level_splats), 10)]
    
    for splat_chunk in splat_chunks:
        # Compute attention for this chunk of splats
        for splat in splat_chunk:
            # Get from cache if available
            if splat.id in shared_cache:
                splat_attention = shared_cache[splat.id]
            else:
                # Compute attention
                splat_attention = splat_attention_fn(tokens, splat)
                # Store in cache
                shared_cache[splat.id] = splat_attention
            
            # Add to level attention
            level_attention += splat_attention
    
    return level_attention


def compute_level_attention_sparse(
    tokens: np.ndarray,
    splat_registry: SplatRegistry,
    level: str,
    spatial_indexes: Optional[Dict[str, Any]] = None,
    level_attention_cache: Optional[Dict[str, np.ndarray]] = None,
    sparsity_threshold: float = 0.01,
    splat_attention_fn: Optional[Callable] = None
) -> np.ndarray:
    """Compute attention for all splats at a specific level using sparse methods.
    
    Args:
        tokens: Token embeddings of shape [seq_len, embedding_dim]
        splat_registry: Registry containing splats
        level: Hierarchical level to compute attention for
        spatial_indexes: Optional dictionary of spatial indexes by level
        level_attention_cache: Optional cache for level attention matrices
        sparsity_threshold: Threshold below which values are considered zero
        splat_attention_fn: Function to compute attention for a single splat
        
    Returns:
        Level attention matrix of shape [seq_len, seq_len]
    """
    seq_len = tokens.shape[0]
    
    # Try to get from cache
    if level_attention_cache is not None and level in level_attention_cache:
        return level_attention_cache[level]
    
    # Get splats at this level
    level_splats = list(splat_registry.get_splats_at_level(level))
    
    # Skip empty levels
    if not level_splats:
        return np.zeros((seq_len, seq_len))
    
    if spatial_indexes is not None and level in spatial_indexes:
        # Use spatial index to efficiently find relevant splats for each token
        spatial_index = spatial_indexes[level]
        
        # Initialize sparse matrix arrays
        data = []
        row_ind = []
        col_ind = []
        
        # For each token, find nearby splats that could contribute to attention
        for i in range(seq_len):
            token_i = tokens[i]
            
            # Find splats near this token
            nearby_splats = find_relevant_splats_for_token(
                token_i, spatial_index, sparsity_threshold, max_splats=5
            )
            
            for j in range(seq_len):
                token_j = tokens[j]
                
                # Compute attention through nearby splats
                token_pair_attention = 0.0
                
                for splat, relevance in nearby_splats:
                    att = splat.compute_attention(token_i, token_j)
                    token_pair_attention += att
                
                # Add non-zero attention to sparse matrix
                if token_pair_attention >= sparsity_threshold:
                    data.append(token_pair_attention)
                    row_ind.append(i)
                    col_ind.append(j)
        
        # Create sparse matrix and convert to dense
        level_attention = create_sparse_matrix_from_values(
            data, row_ind, col_ind, (seq_len, seq_len)
        )
    else:
        # Use dense computation as a fallback
        level_attention = compute_level_attention_dense(
            tokens, level_splats, splat_attention_fn
        )
    
    # Store in cache if available
    if level_attention_cache is not None:
        level_attention_cache[level] = level_attention
    
    return level_attention


def apply_topk_sparsity(
    attention_matrix: np.ndarray,
    k: int
) -> np.ndarray:
    """Apply top-k sparsity to attention matrix.
    
    Args:
        attention_matrix: Dense attention matrix of shape [seq_len, seq_len]
        k: Number of attention values to keep per row
        
    Returns:
        Sparse attention matrix of shape [seq_len, seq_len]
    """
    if k >= attention_matrix.shape[1]:
        return attention_matrix.copy()
    
    # Create output matrix
    result = np.zeros_like(attention_matrix)
    
    # Process each row
    for i in range(attention_matrix.shape[0]):
        # Get indices of top-k values in this row
        row = attention_matrix[i]
        
        if k == 1:
            # Fast path for k=1
            top_idx = np.argmax(row)
            result[i, top_idx] = row[top_idx]
        else:
            # General case
            top_k_indices = np.argpartition(row, -k)[-k:]
            result[i, top_k_indices] = row[top_k_indices]
    
    return result


def get_level_weight(
    level: str,
    registry: SplatRegistry,
    level_weights: Optional[Dict[str, float]] = None
) -> float:
    """Get weight for a hierarchical level.
    
    Args:
        level: Level name
        registry: Registry containing splats
        level_weights: Optional custom level weights
        
    Returns:
        Weight for the given level
    """
    if level_weights is not None and level in level_weights:
        return level_weights[level]
    
    # Use default from hierarchy
    return registry.hierarchy.get_level_weight(level)


def optimize_computation_method(
    tokens: np.ndarray,
    splat: Splat
) -> str:
    """Determine the most efficient computation method for attention.
    
    Args:
        tokens: Token embeddings
        splat: Splat to compute attention for
        
    Returns:
        String indicating computation method ('dense', 'sparse', or 'hybrid')
    """
    seq_len = tokens.shape[0]
    dim = tokens.shape[1]
    
    # For very small sequences, always use dense
    if seq_len <= 256:
        return 'dense'
    
    # For medium sequences, check token relevance
    if seq_len <= 1024:
        # Compute a sample of token relevances
        sample_indices = np.random.choice(seq_len, min(100, seq_len), replace=False)
        sample_tokens = tokens[sample_indices]
        
        # Calculate relevance for sample
        relevances = np.array([
            compute_token_relevance(sample_tokens, splat)
        ])
        
        # Estimate number of relevant tokens
        relevance_ratio = np.mean(relevances > 0.01)
        
        if relevance_ratio < 0.2:
            return 'sparse'
        else:
            return 'hybrid'
    
    # For very large sequences, prefer sparse or hybrid
    relevance_ratio = 0.1  # Assume low relevance ratio for large sequences
    
    if relevance_ratio < 0.1:
        return 'sparse'
    else:
        return 'hybrid'

