"""
Optimized dense implementation of attention computation for Hierarchical Splat Attention (HSA).

This module provides an efficient implementation of the HSA attention mechanism
using various optimization techniques.
"""

from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import logging

from .splat import Splat
from .registry import SplatRegistry
from .hierarchy import Hierarchy
from .attention_interface import AttentionComputer, AttentionConfig, AttentionResult
from .attention_cache import AttentionCache, cache_attention
from .attention_utils import (
    apply_causal_mask, normalize_rows, apply_topk_mask, 
    vectorized_pairwise_distances, batch_attention_computation
)

# Configure logging
logger = logging.getLogger(__name__)


class DenseAttentionComputer(AttentionComputer):
    """
    Optimized dense implementation of the HSA attention mechanism.
    
    This provides an efficient implementation that uses vectorization,
    caching, and other optimizations to improve performance.
    """
    
    def __init__(self, config: Optional[AttentionConfig] = None, cache_size: int = 1000):
        """Initialize an optimized dense attention computer.
        
        Args:
            config: Configuration for attention computation
            cache_size: Maximum number of entries to cache
        """
        self.config = config or AttentionConfig()
        self.cache = AttentionCache(max_size=cache_size)
        
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
        
        Args:
            tokens: Token embeddings of shape [seq_len, embedding_dim]
            splat: Splat to compute attention for
            max_tokens: Maximum number of tokens to compute (for efficiency)
            
        Returns:
            Attention contribution matrix of shape [seq_len, seq_len]
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
    
    def compute_attention(
        self, 
        tokens: np.ndarray, 
        splat_registry: SplatRegistry
    ) -> np.ndarray:
        """Compute the attention matrix for a sequence of tokens.
        
        Args:
            tokens: Token embeddings of shape [seq_len, embedding_dim]
            splat_registry: Registry containing splats to use for attention
            
        Returns:
            Attention matrix of shape [seq_len, seq_len]
        """
        seq_len = tokens.shape[0]
        
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
        
        # Pre-compute causal mask if needed
        causal_mask = None
        if self.config.causal:
            causal_mask = np.tril(np.ones((seq_len, seq_len)))
        
        # Clear level attention cache
        self.level_attention_cache = {}
        
        # Compute attention per level
        for level in hierarchy.levels:
            level_splats = list(splat_registry.get_splats_at_level(level))
            
            # Skip empty levels
            if not level_splats:
                continue
            
            # Compute level attention efficiently
            level_attention = self._compute_level_attention(tokens, level_splats)
            
            # Store in cache for reuse
            self.level_attention_cache[level] = level_attention
            
            # Normalize level attention if requested
            if self.config.normalize_levels:
                max_val = np.max(level_attention)
                if max_val > 0:
                    level_attention = level_attention / max_val
            
            # Apply causal mask if requested
            if self.config.causal and causal_mask is not None:
                level_attention = level_attention * causal_mask
            
            # Apply level weight
            level_attention *= level_weights[level]
            
            # Add to total attention
            attention_matrix += level_attention
        
        # Apply top-k sparsity if configured
        if self.config.topk is not None or self.config.threshold is not None:
            attention_matrix = self.apply_topk_sparsity(
                attention_matrix, 
                k=self.config.topk, 
                threshold=self.config.threshold
            )
        
        # Normalize rows efficiently
        if self.config.normalize_rows:
            attention_matrix = normalize_rows(attention_matrix)
        
        return attention_matrix
    
    def _compute_level_attention(
        self, 
        tokens: np.ndarray, 
        level_splats: List[Splat]
    ) -> np.ndarray:
        """Compute attention for all splats at a specific level.
        
        Args:
            tokens: Token embeddings of shape [seq_len, embedding_dim]
            level_splats: List of splats at this level
            
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
                splat_attention = self.compute_splat_attention_map(tokens, splat)
                level_attention += splat_attention
                
                # Store in shared cache
                shared_cache[splat.id] = splat_attention
        
        return level_attention
    
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
            threshold = self.config.default_threshold
        
        result = attention_matrix.copy()
        
        if k is not None:
            # Vectorized top-k computation
            result = apply_topk_mask(result, k)
        else:
            # Simple threshold sparsity
            result = np.where(result >= threshold, result, 0.0)
        
        return result
    
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
        # First compute the full attention matrix
        attention_matrix = self.compute_attention(tokens, splat_registry)
        
        # Prepare contribution tracking
        level_contributions = {}
        splat_contributions = {}
        active_splats = []
        
        # Process levels - can reuse cached level attention
        for level in splat_registry.hierarchy.levels:
            # Skip empty levels
            if not splat_registry.get_splats_at_level(level):
                continue
            
            # Get level attention from cache if available
            if level in self.level_attention_cache:
                level_attention = self.level_attention_cache[level]
            else:
                # Compute if not in cache
                level_splats = list(splat_registry.get_splats_at_level(level))
                level_attention = self._compute_level_attention(tokens, level_splats)
            
            # Store level contribution
            level_contributions[level] = level_attention
        
        # Process individual splats - we need this for splat contributions
        activation_threshold = 0.01  # Threshold for considering a splat active
        
        for splat in splat_registry.get_all_splats():
            # Compute contribution for this splat
            splat_attention = self.compute_splat_attention_map(tokens, splat)
            splat_contributions[splat.id] = splat_attention
            
            # Check if splat is active
            max_contribution = np.max(splat_attention)
            if max_contribution > activation_threshold:
                active_splats.append(splat)
        
        # Create and return result object
        return AttentionResult(
            attention_matrix=attention_matrix,
            level_contributions=level_contributions,
            splat_contributions=splat_contributions,
            active_splats=active_splats
        )
