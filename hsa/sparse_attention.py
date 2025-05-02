"""
Sparse implementation of attention computation for Hierarchical Splat Attention (HSA).

This module provides the main SparseAttentionComputer class that implements
the AttentionComputer interface with sparse optimizations.
"""

from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import logging

from .splat import Splat
from .registry import SplatRegistry
from .attention_interface import AttentionComputer, AttentionConfig, AttentionResult
from .attention_cache import AttentionCache, cache_attention
from .sparse_attention_utils import (
    compute_token_relevance, update_spatial_indexes, 
    initialize_spatial_indexes, apply_sparse_topk
)
from .sparse_attention_computation import (
    compute_dense_attention_map, compute_optimized_attention_map,
    compute_dense_attention, compute_level_attention_dense,
    compute_level_attention_sparse
)

# Configure logging
logger = logging.getLogger(__name__)


class SparseAttentionComputer(AttentionComputer):
    """
    Memory-efficient sparse implementation of the HSA attention mechanism.
    
    This implementation uses sparse matrices and spatial indexing to efficiently
    compute attention for long sequences, reducing both memory usage and
    computation time.
    """
    
    def __init__(
        self, 
        config: Optional[AttentionConfig] = None, 
        cache_size: int = 1000,
        sparsity_threshold: float = 0.01,
        use_spatial_index: bool = True
    ):
        """Initialize a sparse attention computer.
        
        Args:
            config: Configuration for attention computation
            cache_size: Maximum number of entries to cache
            sparsity_threshold: Threshold below which attention values are considered zero
            use_spatial_index: Whether to use spatial indexing for optimization
        """
        self.config = config or AttentionConfig()
        self.cache = AttentionCache(max_size=cache_size)
        self.sparsity_threshold = sparsity_threshold
        self.use_spatial_index = use_spatial_index
        
        # Spatial indexes for each level
        self.spatial_indexes = {}
        
        # Store computed attention matrices per level for reuse
        self.level_attention_cache = {}
    
    @cache_attention()
    def compute_splat_attention_map(
        self,
        tokens: np.ndarray,
        splat: Splat,
        max_tokens: Optional[int] = None
    ) -> np.ndarray:
        """Compute the attention contribution map for a single splat.
        
        Uses sparse computation when possible.
        
        Args:
            tokens: Token embeddings of shape [seq_len, embedding_dim]
            splat: Splat to compute attention for
            max_tokens: Maximum number of tokens to compute (for efficiency)
            
        Returns:
            Attention contribution matrix of shape [seq_len, seq_len]
        """
        seq_len = tokens.shape[0]
        
        # For small sequences, use the dense computation
        if seq_len <= 256:
            # Compute using dense method
            attention = compute_dense_attention_map(
                tokens, splat, max_tokens, self.cache
            )
            
            # Sparsify the result
            attention[attention < self.sparsity_threshold] = 0.0
            
            return attention
        
        # For larger sequences, use sparse computation
        
        # Compute token relevance to this splat
        token_relevance = compute_token_relevance(tokens, splat)
        
        # Filter tokens that are relevant to this splat
        relevant_indices = np.where(token_relevance > self.sparsity_threshold)[0]
        
        # If no tokens are relevant, return zero matrix
        if len(relevant_indices) == 0:
            return np.zeros((seq_len, seq_len))
        
        # If few tokens are relevant, use sparse matrix directly
        if len(relevant_indices) <= 0.2 * seq_len:
            # Create sparse matrix computation based on relevant tokens only
            relevant_tokens = tokens[relevant_indices]
            attention = compute_optimized_attention_map(
                tokens, splat, relevant_indices, relevant_tokens, self.sparsity_threshold
            )
            return attention
        
        # If many tokens are relevant, compute full attention but use optimizations
        attention = compute_optimized_attention_map(
            tokens, splat, None, None, self.sparsity_threshold
        )
        
        # Sparsify the result
        attention[attention < self.sparsity_threshold] = 0.0
        
        return attention
    
    def compute_attention(
        self, 
        tokens: np.ndarray, 
        splat_registry: SplatRegistry
    ) -> np.ndarray:
        """Compute the attention matrix for a sequence of tokens.
        
        Uses sparse computation for efficiency.
        
        Args:
            tokens: Token embeddings of shape [seq_len, embedding_dim]
            splat_registry: Registry containing splats to use for attention
            
        Returns:
            Attention matrix of shape [seq_len, seq_len]
        """
        seq_len = tokens.shape[0]
        
        # For small sequences, use dense computation
        if seq_len <= 256:
            return compute_dense_attention(
                tokens, splat_registry, self.config, self.compute_splat_attention_map
            )
        
        # For larger sequences, use sparse computation
        
        # Initialize attention matrix
        attention_matrix = np.zeros((seq_len, seq_len))
        
        # Get hierarchy and level weights
        hierarchy = splat_registry.hierarchy
        level_weights = {}
        
        if self.config.level_weights:
            # Use custom weights if provided
            level_weights = self.config.level_weights
        else:
            # Use hierarchy defaults
            for level in hierarchy.levels:
                level_weights[level] = hierarchy.get_level_weight(level)
        
        # Initialize or update spatial indexes if using them
        if self.use_spatial_index:
            if not self.spatial_indexes:
                self.spatial_indexes = initialize_spatial_indexes(splat_registry)
            else:
                update_spatial_indexes(splat_registry, self.spatial_indexes)
        
        # Compute attention per level
        for level in hierarchy.levels:
            # Compute level attention efficiently
            level_attention = compute_level_attention_sparse(
                tokens, 
                splat_registry, 
                level, 
                self.spatial_indexes if self.use_spatial_index else None,
                self.level_attention_cache,
                self.sparsity_threshold,
                self.compute_splat_attention_map
            )
            
            # Normalize level attention if requested
            if self.config.normalize_levels:
                max_val = np.max(level_attention)
                if max_val > 0:
                    level_attention = level_attention / max_val
            
            # Apply causal mask if requested
            if self.config.causal:
                causal_mask = np.tril(np.ones_like(level_attention))
                level_attention *= causal_mask
            
            # Apply level weight
            weighted_level_attention = level_attention * level_weights[level]
            
            # Add to total attention
            attention_matrix += weighted_level_attention
        
        # Apply top-k sparsity if configured
        if self.config.topk is not None or self.config.threshold is not None:
            attention_matrix = self.apply_topk_sparsity(
                attention_matrix, 
                k=self.config.topk, 
                threshold=self.config.threshold
            )
        
        # Normalize rows efficiently
        if self.config.normalize_rows:
            row_sums = np.sum(attention_matrix, axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            attention_matrix = attention_matrix / row_sums
        
        return attention_matrix
    
    def compute_attention_with_details(
        self,
        tokens: np.ndarray,
        splat_registry: SplatRegistry
    ) -> AttentionResult:
        """Compute attention with detailed contributions.
        
        Args:
            tokens: Token embeddings of shape [seq_len, embedding_dim]
            splat_registry: Registry containing splats to use for attention
            
        Returns:
            AttentionResult with full attention matrix and contribution details
        """
        seq_len = tokens.shape[0]
        
        # Compute overall attention matrix
        attention_matrix = self.compute_attention(tokens, splat_registry)
        
        # Initialize contribution tracking
        level_contributions = {}
        splat_contributions = {}
        active_splats = []
        
        # Get hierarchy and level weights
        hierarchy = splat_registry.hierarchy
        level_weights = {}
        
        if self.config.level_weights:
            # Use custom weights if provided
            level_weights = self.config.level_weights
        else:
            # Use hierarchy defaults
            for level in hierarchy.levels:
                level_weights[level] = hierarchy.get_level_weight(level)
        
        # Compute contributions per level
        for level in hierarchy.levels:
            level_splats = list(splat_registry.get_splats_at_level(level))
            
            # Skip empty levels
            if not level_splats:
                continue
            
            # Initialize level attention
            level_attention = np.zeros((seq_len, seq_len))
            
            # Compute attention through each splat
            for splat in level_splats:
                splat_attention = self.compute_splat_attention_map(tokens, splat)
                
                # Store splat contribution
                splat_contributions[splat.id] = splat_attention
                
                # Add to level attention
                level_attention += splat_attention
                
                # Check if splat is active
                if np.max(splat_attention) > 0.01:  # Activation threshold
                    active_splats.append(splat)
            
            # Normalize level attention if requested
            if self.config.normalize_levels and np.max(level_attention) > 0:
                level_attention = level_attention / np.max(level_attention)
            
            # Apply causal mask if requested
            if self.config.causal:
                causal_mask = np.tril(np.ones_like(level_attention))
                level_attention *= causal_mask
            
            # Store level contribution
            level_contributions[level] = level_attention
        
        # Create result object
        return AttentionResult(
            attention_matrix=attention_matrix,
            level_contributions=level_contributions,
            splat_contributions=splat_contributions,
            active_splats=active_splats
        )
    
    def apply_topk_sparsity(
        self,
        attention_matrix: np.ndarray,
        k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Apply top-k sparsity to attention matrix.
        
        Either k or threshold must be specified.
        
        Args:
            attention_matrix: Dense attention matrix of shape [seq_len, seq_len]
            k: Number of attention values to keep per row (if None, use threshold)
            threshold: Minimum attention value to keep (if k is None)
            
        Returns:
            Sparse attention matrix of shape [seq_len, seq_len]
        """
        if k is None and threshold is None:
            # If neither is specified, use a default threshold
            threshold = self.config.default_threshold or self.sparsity_threshold
        
        if k is not None:
            # Apply top-k sparsity
            return apply_sparse_topk(attention_matrix, k)
        else:
            # Apply threshold sparsity
            return np.where(attention_matrix >= threshold, attention_matrix, 0.0)
