"""
Sparse implementation of attention computation for Hierarchical Splat Attention (HSA).

This module provides a memory-efficient sparse implementation of the HSA
attention mechanism, which is especially useful for long sequences.
"""

from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import scipy.sparse as sp
import logging

from .splat import Splat
from .registry import SplatRegistry
from .attention_interface import AttentionComputer, AttentionConfig, AttentionResult
from .attention_cache import AttentionCache, cache_attention
from .attention_utils import (
    apply_causal_mask, normalize_rows, apply_topk_mask, 
    vectorized_pairwise_distances, batch_attention_computation
)
from .spatial_index import SpatialIndexFactory

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
            attention = self._compute_dense_attention_map(tokens, splat, max_tokens)
            
            # Sparsify the result
            attention[attention < self.sparsity_threshold] = 0.0
            
            return attention
        
        # For larger sequences, use sparse computation
        
        # Compute token relevance to this splat
        token_relevance = self._compute_token_relevance(tokens, splat)
        
        # Filter tokens that are relevant to this splat
        relevant_indices = np.where(token_relevance > self.sparsity_threshold)[0]
        
        # If no tokens are relevant, return zero matrix
        if len(relevant_indices) == 0:
            return np.zeros((seq_len, seq_len))
        
        # If few tokens are relevant, use sparse matrix directly
        if len(relevant_indices) <= 0.2 * seq_len:
            # Create sparse matrix
            data = []
            row_ind = []
            col_ind = []
            
            # Compute attentions between relevant tokens
            relevant_tokens = tokens[relevant_indices]
            
            for i, idx_i in enumerate(relevant_indices):
                for j, idx_j in enumerate(relevant_indices):
                    # Compute attention
                    att_value = splat.compute_attention(
                        relevant_tokens[i], relevant_tokens[j]
                    )
                    
                    # Only store non-zero values
                    if att_value >= self.sparsity_threshold:
                        data.append(att_value)
                        row_ind.append(idx_i)
                        col_ind.append(idx_j)
            
            # Create sparse matrix
            sparse_att = sp.csr_matrix(
                (data, (row_ind, col_ind)), 
                shape=(seq_len, seq_len)
            )
            
            # Convert to dense for now (in production, you'd keep it sparse)
            return sparse_att.toarray()
        
        # If many tokens are relevant, compute full attention but use optimizations
        attention = self._compute_optimized_attention_map(tokens, splat)
        
        # Sparsify the result
        attention[attention < self.sparsity_threshold] = 0.0
        
        return attention
    
    def _compute_dense_attention_map(
        self,
        tokens: np.ndarray,
        splat: Splat,
        max_tokens: Optional[int] = None
    ) -> np.ndarray:
        """Compute attention map using dense computation.
        
        Args:
            tokens: Token embeddings of shape [seq_len, embedding_dim]
            splat: Splat to compute attention for
            max_tokens: Maximum number of tokens to compute
            
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
    
    def _compute_token_relevance(
        self,
        tokens: np.ndarray,
        splat: Splat
    ) -> np.ndarray:
        """Compute relevance of each token to a splat.
        
        Args:
            tokens: Token embeddings of shape [seq_len, embedding_dim]
            splat: Splat to compute relevance for
            
        Returns:
            Relevance scores for each token
        """
        # Compute Mahalanobis distances from tokens to splat center
        deltas = tokens - splat.position
        
        if hasattr(splat, 'covariance_inverse') and splat.covariance_inverse is not None:
            try:
                # Transform deltas
                transformed = np.dot(deltas, splat.covariance_inverse)
                
                # Compute Mahalanobis distances
                distances = np.sum(transformed * deltas, axis=1)
                
                # Convert to relevance scores
                relevance = np.exp(-0.5 * distances)
                
                # Apply amplitude
                relevance = splat.amplitude * relevance
            except:
                # Fall back to Euclidean distance
                distances = np.linalg.norm(deltas, axis=1)
                relevance = np.exp(-0.5 * distances ** 2)
        else:
            # Fall back to Euclidean distance
            distances = np.linalg.norm(deltas, axis=1)
            relevance = np.exp(-0.5 * distances ** 2)
        
        return relevance
    
    def _compute_optimized_attention_map(
        self,
        tokens: np.ndarray,
        splat: Splat
    ) -> np.ndarray:
        """Compute attention map using optimized methods.
        
        Args:
            tokens: Token embeddings of shape [seq_len, embedding_dim]
            splat: Splat to compute attention for
            
        Returns:
            Attention contribution matrix
        """
        seq_len = tokens.shape[0]
        
        # Compute token relevance to this splat
        token_relevance = self._compute_token_relevance(tokens, splat)
        
        # Compute attention matrix using outer product approximation
        # This is a mathematical simplification of the full computation
        # that works well in practice
        attention = np.outer(token_relevance, token_relevance)
        
        # Apply any necessary corrections
        # This would be more sophisticated in a production implementation
        
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
            return self._compute_dense_attention(tokens, splat_registry)
        
        # For larger sequences, use sparse computation
        
        # Initialize sparse attention matrix
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
            self._update_spatial_indexes(splat_registry)
        
        # Compute attention per level
        for level in hierarchy.levels:
            # Compute level attention efficiently
            level_attention = self._compute_level_attention_sparse(
                tokens, splat_registry, level
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
    
    def _compute_dense_attention(
        self,
        tokens: np.ndarray,
        splat_registry: SplatRegistry
    ) -> np.ndarray:
        """Compute attention using dense methods.
        
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
        for level in hierarchy.levels:
            level_splats = list(splat_registry.get_splats_at_level(level))
            
            # Skip empty levels
            if not level_splats:
                continue
            
            # Compute level attention efficiently
            level_attention = self._compute_level_attention_dense(tokens, level_splats)
            
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
    
    def _compute_level_attention_dense(
        self, 
        tokens: np.ndarray, 
        level_splats: List[Splat]
    ) -> np.ndarray:
        """Compute attention for all splats at a specific level using dense methods.
        
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
    
    def _compute_level_attention_sparse(
        self,
        tokens: np.ndarray,
        splat_registry: SplatRegistry,
        level: str
    ) -> np.ndarray:
        """Compute attention for all splats at a specific level using sparse methods.
        
        Args:
            tokens: Token embeddings of shape [seq_len, embedding_dim]
            splat_registry: Registry containing splats
            level: Hierarchical level to compute attention for
            
        Returns:
            Level attention matrix of shape [seq_len, seq_len]
        """
        seq_len = tokens.shape[0]
        
        # Try to get from cache
        if level in self.level_attention_cache:
            return self.level_attention_cache[level]
        
        # Get splats at this level
        level_splats = list(splat_registry.get_splats_at_level(level))
        
        # Skip empty levels
        if not level_splats:
            return np.zeros((seq_len, seq_len))
        
        if self.use_spatial_index and level in self.spatial_indexes:
            # Use spatial index to efficiently find relevant splats for each token
            spatial_index = self.spatial_indexes[level]
            
            # Initialize sparse matrix arrays
            data = []
            row_ind = []
            col_ind = []
            
            # For each token, find nearby splats that could contribute to attention
            for i in range(seq_len):
                token_i = tokens[i]
                
                # Find splats near this token
                nearby_splats = spatial_index.find_nearest(token_i, k=5)
                
                for j in range(seq_len):
                    token_j = tokens[j]
                    
                    # Compute attention through nearby splats
                    token_pair_attention = 0.0
                    
                    for splat, _ in nearby_splats:
                        att = splat.compute_attention(token_i, token_j)
                        token_pair_attention += att
                    
                    # Add non-zero attention to sparse matrix
                    if token_pair_attention >= self.sparsity_threshold:
                        data.append(token_pair_attention)
                        row_ind.append(i)
                        col_ind.append(j)
            
            # Create sparse matrix
            sparse_att = sp.csr_matrix(
                (data, (row_ind, col_ind)), 
                shape=(seq_len, seq_len)
            )
            
            # Convert to dense for now (in production, keep sparse longer)
            level_attention = sparse_att.toarray()
        else:
            # Use dense computation as a fallback
            level_attention = self._compute_level_attention_dense(tokens, level_splats)
        
        # Store in cache
        self.level_attention_cache[level] = level_attention
        
        return level_attention
    
    def _update_spatial_indexes(self, splat_registry: SplatRegistry) -> None:
        """Update spatial indexes for efficient splat queries.
        
        Args:
            splat_registry: Registry containing splats
        """
        if not self.use_spatial_index:
            return
        
        # Create or update indexes for each level
        for level in splat_registry.hierarchy.levels:
            # Get splats at this level
            level_splats = list(splat_registry.get_splats_at_level(level))
            
            # Skip empty levels
            if not level_splats:
                continue
            
            # Create or replace index
            dim = splat_registry.embedding_dim
            self.spatial_indexes[level] = SpatialIndexFactory.create_index(
                dim=dim,
                splats=level_splats,
                index_type="auto"  # Let factory choose optimal type
            )
    
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
        
        result = attention_matrix.copy()
        
        if k is not None:
            # Top-k sparsity (keep only k largest values per row)
            result = apply_topk_mask(result, k)
        else:
            # Threshold sparsity (remove values below threshold)
            result = np.where(result >= threshold, result, 0.0)
        
        return result
