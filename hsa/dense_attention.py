"""
Dense implementation of attention computation for Hierarchical Splat Attention (HSA).

This module provides a complete but potentially inefficient implementation
of the HSA attention mechanism.
"""

from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from .splat import Splat
from .registry import SplatRegistry
from .hierarchy import Hierarchy
from .attention_interface import AttentionComputer, AttentionConfig, AttentionResult


class DenseAttentionComputer(AttentionComputer):
    """
    Dense implementation of the HSA attention mechanism.
    
    This provides a complete but potentially inefficient implementation that
    computes all pairwise token-token attention values through all splats.
    """
    
    def __init__(self, config: Optional[AttentionConfig] = None):
        """Initialize a dense attention computer.
        
        Args:
            config: Configuration for attention computation
        """
        self.config = config or AttentionConfig()
    
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
        
        # Compute attention per level
        level_contributions = {}
        
        for level in hierarchy.levels:
            level_attention = np.zeros((seq_len, seq_len))
            level_splats = splat_registry.get_splats_at_level(level)
            
            # Skip empty levels
            if not level_splats:
                continue
            
            # Compute attention through each splat at this level
            for splat in level_splats:
                splat_attention = self.compute_splat_attention_map(tokens, splat)
                level_attention += splat_attention
            
            # Normalize level attention if requested
            if self.config.normalize_levels and np.max(level_attention) > 0:
                level_attention = level_attention / np.max(level_attention)
            
            # Apply causal mask if requested
            if self.config.causal:
                # Create causal mask (lower triangular)
                causal_mask = np.tril(np.ones_like(level_attention))
                # Apply mask - set upper triangle to zero
                level_attention = level_attention * causal_mask
            
            # Apply level weight
            weighted_level_attention = level_attention * level_weights[level]
            
            # Add to total attention
            attention_matrix += weighted_level_attention
            
            # Store level contribution
            level_contributions[level] = level_attention
        
        # Apply top-k sparsity if configured
        if self.config.topk is not None or self.config.threshold is not None:
            attention_matrix = self.apply_topk_sparsity(
                attention_matrix, 
                k=self.config.topk, 
                threshold=self.config.threshold
            )
        
        # Normalize rows if requested
        if self.config.normalize_rows:
            row_sums = np.sum(attention_matrix, axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            attention_matrix = attention_matrix / row_sums
        
        return attention_matrix
    
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
            
            # Compute attention for sampled tokens
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
        
        # Compute full attention matrix
        attention = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                attention[i, j] = splat.compute_attention(tokens[i], tokens[j])
        
        return attention
    
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
        seq_len = attention_matrix.shape[0]
        
        if k is not None:
            # Apply top-k sparsity
            for i in range(seq_len):
                # Get indices of top-k values in this row
                if k < seq_len:
                    # Find values below the top k and zero them out
                    row = result[i]
                    threshold_value = np.sort(row)[-k]
                    result[i] = np.where(row >= threshold_value, row, 0.0)
        else:
            # Apply threshold sparsity
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
        
        # Prepare contribution tracking
        level_contributions = {}
        splat_contributions = {}
        active_splats = []
        
        # Compute attention per level
        for level in hierarchy.levels:
            level_attention = np.zeros((seq_len, seq_len))
            level_splats = splat_registry.get_splats_at_level(level)
            
            # Skip empty levels
            if not level_splats:
                continue
            
            # Compute attention through each splat at this level
            for splat in level_splats:
                splat_attention = self.compute_splat_attention_map(tokens, splat)
                
                # Store splat contribution
                splat_contributions[splat.id] = splat_attention
                
                # Check if splat is active
                if np.max(splat_attention) > 0.01:  # Activation threshold
                    active_splats.append(splat)
                
                level_attention += splat_attention
            
            # Normalize level attention if requested
            if self.config.normalize_levels and np.max(level_attention) > 0:
                level_attention = level_attention / np.max(level_attention)
            
            # Apply causal mask if requested
            if self.config.causal:
                # Create causal mask (lower triangular)
                causal_mask = np.tril(np.ones_like(level_attention))
                # Apply mask - set upper triangle to zero
                level_attention = level_attention * causal_mask
            
            # Apply level weight
            weighted_level_attention = level_attention * level_weights[level]
            
            # Add to total attention
            attention_matrix += weighted_level_attention
            
            # Store level contribution
            level_contributions[level] = level_attention
        
        # Apply top-k sparsity if configured
        if self.config.topk is not None or self.config.threshold is not None:
            attention_matrix = self.apply_topk_sparsity(
                attention_matrix, 
                k=self.config.topk, 
                threshold=self.config.threshold
            )
        
        # Normalize rows if requested
        if self.config.normalize_rows:
            row_sums = np.sum(attention_matrix, axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.where(row_sums > 0, row_sums, 1.0)
            attention_matrix = attention_matrix / row_sums
        
        # Create and return result object
        return AttentionResult(
            attention_matrix=attention_matrix,
            level_contributions=level_contributions,
            splat_contributions=splat_contributions,
            active_splats=active_splats
        )
