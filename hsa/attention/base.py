"""
Base module for Hierarchical Splat Attention (HSA).

This module defines the core interfaces and base classes for HSA attention computation:
- Abstract base class for attention computation
- Common utility functions for attention calculations
- Interface definitions for various attention implementations
"""

import numpy as np
from typing import List, Set, Dict, Tuple, Optional, Union, Any
import math
from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix, lil_matrix
import time

# Import core data structures
from hsa.data_structures import Splat, Hierarchy, SplatRegistry

class AttentionComputer(ABC):
    """
    Abstract base class for computing attention matrices in Hierarchical Splat Attention.
    
    This class defines the interface for attention computation with various optimizations.
    Concrete implementations can use different strategies for efficiency.
    """
    
    def __init__(self, hierarchy: Hierarchy, sparse_topk: int = 64, max_splat_radius: float = 3.0):
        """
        Initialize the attention computer.
        
        Args:
            hierarchy: Hierarchy configuration for the attention mechanism
            sparse_topk: Number of top attention scores to keep per token (sparsity)
            max_splat_radius: Maximum radius of influence for splats in std deviations
        """
        self.hierarchy = hierarchy
        self.sparse_topk = sparse_topk
        self.max_splat_radius = max_splat_radius
        self.max_computation_time = 60  # Maximum seconds for attention computation
    
    @abstractmethod
    def compute_attention(
        self, 
        tokens: np.ndarray, 
        splat_registry: SplatRegistry
    ) -> np.ndarray:
        """
        Compute the attention matrix using hierarchical splat attention.
        
        Args:
            tokens: Token embeddings of shape [sequence_length, embedding_dim]
            splat_registry: Registry containing all splats
            
        Returns:
            Attention matrix of shape [sequence_length, sequence_length]
        """
        pass
    
    def compute_splat_attention_map(
        self, 
        tokens: np.ndarray, 
        splat: Splat,
        max_tokens: int = 200
    ) -> np.ndarray:
        """
        Compute the attention map for a single splat across all token pairs.
        
        This is useful for visualization and debugging.
        
        Args:
            tokens: Token embeddings of shape [sequence_length, embedding_dim]
            splat: The splat to compute attention for
            max_tokens: Maximum number of tokens to use (for efficiency)
            
        Returns:
            Attention matrix from this splat of shape [sequence_length, sequence_length]
        """
        sequence_length = tokens.shape[0]
        
        # For optimization, limit the token count
        if sequence_length > max_tokens:
            # Subsample the tokens
            indices = np.linspace(0, sequence_length-1, max_tokens, dtype=int)
            tokens = tokens[indices]
            sequence_length = max_tokens
        
        # Extract splat parameters
        pos = splat.position
        cov_inv = splat.covariance_inverse
        amp = splat.amplitude
        
        # Find tokens within splat's influence using vectorized operations
        diffs = tokens - pos
        distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
        relevant_indices = np.where(distances < self.max_splat_radius)[0]
        
        # Initialize attention map
        attention_map = np.zeros((sequence_length, sequence_length))
        
        # Compute attention only for relevant token pairs
        for i_idx in relevant_indices:
            for j_idx in relevant_indices:
                # Compute distance between tokens relative to this splat
                diff = (tokens[i_idx] - tokens[j_idx]) - pos
                dist = np.sqrt(diff @ cov_inv @ diff)
                
                # Compute attention score
                attention_map[i_idx, j_idx] = amp * np.exp(-dist**2)
        
        return attention_map
    
    def _apply_topk_sparsity(self, attention_matrix: np.ndarray) -> np.ndarray:
        """
        Apply top-k sparsification to a dense attention matrix.
        
        Args:
            attention_matrix: Raw attention matrix
            
        Returns:
            Sparsified attention matrix
        """
        sequence_length = attention_matrix.shape[0]
        sparse_matrix = np.zeros_like(attention_matrix)
        
        for i in range(sequence_length):
            # Get indices of top-k elements in this row
            row = attention_matrix[i, :]
            
            # Efficiently find top-k indices
            if sequence_length <= self.sparse_topk:
                # Keep all if sequence is shorter than topk
                topk_indices = np.arange(sequence_length)
            else:
                # Find top-k indices
                topk_indices = np.argpartition(row, -self.sparse_topk)[-self.sparse_topk:]
            
            # Keep only the top-k values
            sparse_matrix[i, topk_indices] = attention_matrix[i, topk_indices]
        
        return sparse_matrix
    
    def _apply_topk_sparsity_sparse(self, attention_matrix: csr_matrix) -> csr_matrix:
        """
        Apply top-k sparsification to a sparse attention matrix.
        
        Args:
            attention_matrix: Raw sparse attention matrix
            
        Returns:
            Sparsified attention matrix
        """
        # Convert to LIL format for row-wise operations
        lil_matrix_data = attention_matrix.tolil()
        sequence_length = lil_matrix_data.shape[0]
        
        # For each row, keep only top-k elements
        for i in range(sequence_length):
            row_data = lil_matrix_data.data[i]
            row_indices = lil_matrix_data.rows[i]
            
            # Skip if row has fewer than k elements
            if len(row_data) <= self.sparse_topk:
                continue
            
            # Find the top-k values and their indices
            paired = [(val, idx) for val, idx in zip(row_data, row_indices)]
            paired.sort(reverse=True)
            
            # Keep only the top-k values
            top_k_pairs = paired[:self.sparse_topk]
            
            # Update the row
            lil_matrix_data.data[i] = [val for val, _ in top_k_pairs]
            lil_matrix_data.rows[i] = [idx for _, idx in top_k_pairs]
        
        # Convert back to CSR format
        return lil_matrix_data.tocsr()


# Common utility functions for attention

def mahalanobis_batch(
    points: np.ndarray, 
    center: np.ndarray, 
    cov_inv: np.ndarray
) -> np.ndarray:
    """
    Compute Mahalanobis distances for a batch of points.
    
    This is an optimized implementation for computing many distances at once.
    
    Args:
        points: Points of shape [batch_size, dim]
        center: Center point of shape [dim]
        cov_inv: Inverse covariance matrix of shape [dim, dim]
        
    Returns:
        Distances of shape [batch_size]
    """
    # Compute differences for all points at once
    diffs = points - center  # [batch_size, dim]
    
    # Compute squared Mahalanobis distances efficiently
    # This is equivalent to np.sqrt(np.sum(diffs @ cov_inv * diffs, axis=1))
    # but more efficient with einsum
    return np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))


def gauss_kernel_batch(
    distances: np.ndarray, 
    amplitude: float = 1.0
) -> np.ndarray:
    """
    Apply Gaussian kernel to a batch of distances.
    
    Args:
        distances: Distances of shape [batch_size]
        amplitude: Amplitude scaling factor
        
    Returns:
        Kernel values of shape [batch_size]
    """
    return amplitude * np.exp(-distances**2)


def find_relevant_tokens(
    tokens: np.ndarray, 
    splat: Splat, 
    max_radius: float = 3.0
) -> np.ndarray:
    """
    Find tokens within a splat's influence radius.
    
    Args:
        tokens: Token embeddings of shape [sequence_length, embedding_dim]
        splat: The splat to check against
        max_radius: Maximum radius of influence in std deviations
        
    Returns:
        Indices of relevant tokens
    """
    # Extract splat parameters
    pos = splat.position
    cov_inv = splat.covariance_inverse
    
    # Compute Mahalanobis distances
    distances = mahalanobis_batch(tokens, pos, cov_inv)
    
    # Find tokens within radius
    return np.where(distances < max_radius)[0]
