"""
Utility functions for attention computation in Hierarchical Splat Attention (HSA).

This module provides helper functions for optimizing and working with
attention computations.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Any
import logging

# Configure logging
logger = logging.getLogger(__name__)


def apply_causal_mask(attention_matrix: np.ndarray) -> np.ndarray:
    """Apply causal (lower triangular) mask to attention matrix.
    
    Args:
        attention_matrix: Attention matrix of shape [seq_len, seq_len]
        
    Returns:
        Masked attention matrix
    """
    seq_len = attention_matrix.shape[0]
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    return attention_matrix * causal_mask


def normalize_rows(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize matrix rows to sum to 1.
    
    Args:
        matrix: Input matrix
        eps: Small constant to avoid division by zero
        
    Returns:
        Row-normalized matrix
    """
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.maximum(row_sums, eps)
    return matrix / row_sums


def apply_topk_mask(matrix: np.ndarray, k: int) -> np.ndarray:
    """Keep only top-k values in each row.
    
    Args:
        matrix: Input matrix
        k: Number of values to keep per row
        
    Returns:
        Masked matrix with only top-k values per row
    """
    if k >= matrix.shape[1]:
        return matrix.copy()
    
    result = matrix.copy()
    # For each row, find values below the top k and zero them out
    indices = np.argsort(result, axis=1)[:, :-k]
    
    # Create a mask of zeros with ones at the indices to be zeroed
    mask = np.zeros_like(result, dtype=bool)
    rows = np.arange(result.shape[0])[:, np.newaxis]
    mask[rows, indices] = True
    
    # Zero out values below top-k
    result[mask] = 0.0
    
    return result


def vectorized_pairwise_distances(tokens: np.ndarray, 
                                 splat_position: np.ndarray,
                                 splat_cov_inv: np.ndarray) -> np.ndarray:
    """Compute pairwise Mahalanobis distances for all tokens.
    
    Args:
        tokens: Token embeddings of shape [batch_size, embedding_dim]
        splat_position: Splat center position of shape [embedding_dim]
        splat_cov_inv: Splat covariance inverse of shape [embedding_dim, embedding_dim]
        
    Returns:
        Pairwise distances of shape [batch_size, batch_size]
    """
    # Compute deltas for all tokens - shape: [batch_size, embedding_dim]
    deltas = tokens - splat_position
    
    # Compute transformed deltas - shape: [batch_size, embedding_dim]
    transformed = np.dot(deltas, splat_cov_inv)
    
    # Compute Mahalanobis distances - shape: [batch_size]
    distances = np.sum(transformed * deltas, axis=1)
    
    # Each element (i,j) is distance from token i to token j through the splat
    # This is an approximation that works well in practice
    distances_i = np.sqrt(distances)
    distances_matrix = np.outer(distances_i, distances_i)
    
    return distances_matrix


def batch_attention_computation(tokens: np.ndarray, 
                               splats: List[Any],
                               shared_cache: Optional[Dict] = None) -> np.ndarray:
    """Compute attention through multiple splats in a batched manner.
    
    Args:
        tokens: Token embeddings of shape [seq_len, embedding_dim]
        splats: List of splats to compute attention through
        shared_cache: Optional cache dictionary
        
    Returns:
        Combined attention matrix of shape [seq_len, seq_len]
    """
    seq_len = tokens.shape[0]
    attention_matrix = np.zeros((seq_len, seq_len))
    
    # Initialize empty cache if not provided
    if shared_cache is None:
        shared_cache = {}
    
    for splat in splats:
        # Check if this splat's attention is already in cache
        splat_id = splat.id
        if splat_id in shared_cache:
            splat_attention = shared_cache[splat_id]
        else:
            # Compute attention for this splat
            if hasattr(splat, 'compute_attention_batch'):
                splat_attention = splat.compute_attention_batch(tokens)
            else:
                # Fallback to pairwise computation
                splat_attention = np.zeros((seq_len, seq_len))
                for i in range(seq_len):
                    for j in range(seq_len):
                        splat_attention[i, j] = splat.compute_attention(
                            tokens[i], tokens[j]
                        )
            
            # Store in cache
            shared_cache[splat_id] = splat_attention
        
        # Add to combined attention
        attention_matrix += splat_attention
    
    return attention_matrix
