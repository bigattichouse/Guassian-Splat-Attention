"""
Attention interface definitions for Hierarchical Splat Attention (HSA).

This module defines the interfaces for attention computation in HSA.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from .splat import Splat
from .registry import SplatRegistry


class AttentionComputer(ABC):
    """Abstract interface for attention computation mechanisms in HSA."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass


class AttentionConfig:
    """Configuration for attention computation."""
    
    def __init__(
        self,
        level_weights: Optional[Dict[str, float]] = None,
        topk: Optional[int] = None,
        threshold: Optional[float] = None,
        normalize_levels: bool = True,
        normalize_rows: bool = True,
        use_sparse: bool = False,
        causal: bool = False
    ):
        """Initialize attention configuration.
        
        Args:
            level_weights: Weights for each hierarchy level (overrides hierarchy defaults)
            topk: Number of attention values to keep per row (for sparsity)
            threshold: Minimum attention value to keep (for sparsity)
            normalize_levels: Whether to normalize attention across hierarchy levels
            normalize_rows: Whether to normalize attention rows to sum to 1
            use_sparse: Whether to use sparse computation
            causal: Whether to enforce causal attention (lower triangular)
        """
        self.level_weights = level_weights
        self.topk = topk
        self.threshold = threshold
        self.normalize_levels = normalize_levels
        self.normalize_rows = normalize_rows
        self.use_sparse = use_sparse
        self.causal = causal


class AttentionResult:
    """Result of attention computation."""
    
    def __init__(
        self,
        attention_matrix: np.ndarray,
        level_contributions: Optional[Dict[str, np.ndarray]] = None,
        splat_contributions: Optional[Dict[str, np.ndarray]] = None,
        active_splats: Optional[List[Splat]] = None
    ):
        """Initialize attention result.
        
        Args:
            attention_matrix: Final attention matrix of shape [seq_len, seq_len]
            level_contributions: Attention contribution from each level
            splat_contributions: Attention contribution from each splat
            active_splats: List of splats that contributed to attention
        """
        self.attention_matrix = attention_matrix
        self.level_contributions = level_contributions or {}
        self.splat_contributions = splat_contributions or {}
        self.active_splats = active_splats or []
    
    def get_attention_for_token(self, token_idx: int) -> np.ndarray:
        """Get attention vector for a specific token.
        
        Args:
            token_idx: Index of the token
            
        Returns:
            Attention vector of shape [seq_len]
        """
        return self.attention_matrix[token_idx]
    
    def get_level_contribution(self, level: str) -> Optional[np.ndarray]:
        """Get attention contribution from a specific level.
        
        Args:
            level: Hierarchy level name
            
        Returns:
            Attention matrix for the level or None if not available
        """
        return self.level_contributions.get(level)
    
    def get_splat_contribution(self, splat_id: str) -> Optional[np.ndarray]:
        """Get attention contribution from a specific splat.
        
        Args:
            splat_id: ID of the splat
            
        Returns:
            Attention matrix for the splat or None if not available
        """
        return self.splat_contributions.get(splat_id)
    
    def get_active_splats_for_token(self, token_idx: int, threshold: float = 0.01) -> List[Splat]:
        """Get list of splats that actively contribute to a token's attention.
        
        Args:
            token_idx: Index of the token
            threshold: Minimum contribution threshold
            
        Returns:
            List of active splats
        """
        if not self.splat_contributions or not self.active_splats:
            return []
        
        active = []
        for splat in self.active_splats:
            contribution = self.splat_contributions.get(splat.id)
            if contribution is not None:
                # Check if this splat contributes meaningfully to this token's attention
                if np.max(contribution[token_idx]) > threshold:
                    active.append(splat)
        
        return active
