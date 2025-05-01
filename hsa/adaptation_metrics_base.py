"""
Base adaptation metrics interfaces for Hierarchical Splat Attention (HSA).

This module defines the base interfaces for computing metrics used in
adaptation decisions for HSA.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from .splat import Splat
from .registry import SplatRegistry
from .adaptation_types import AdaptationMetrics


class AdaptationMetricsComputer(ABC):
    """Abstract interface for computing adaptation metrics."""
    
    @abstractmethod
    def compute_metrics(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> AdaptationMetrics:
        """Compute adaptation metrics for a splat.
        
        Args:
            splat: Splat to compute metrics for
            registry: Registry containing all splats
            tokens: Optional token embeddings for context-aware metrics
            
        Returns:
            AdaptationMetrics with computed values
        """
        pass
    
    @abstractmethod
    def compute_splat_activation(
        self,
        splat: Splat,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        """Compute activation metric for a splat.
        
        Args:
            splat: Splat to compute activation for
            tokens: Optional token embeddings
            
        Returns:
            Activation value between 0 and 1
        """
        pass
    
    @abstractmethod
    def compute_activation_trend(
        self,
        splat: Splat
    ) -> float:
        """Compute activation trend over time.
        
        Args:
            splat: Splat to compute trend for
            
        Returns:
            Trend value (positive for increasing, negative for decreasing)
        """
        pass
    
    @abstractmethod
    def compute_splat_variance(
        self,
        splat: Splat,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        """Compute internal variance of a splat.
        
        Args:
            splat: Splat to compute variance for
            tokens: Optional token embeddings
            
        Returns:
            Variance value between 0 and 1
        """
        pass
    
    @abstractmethod
    def compute_similarity(
        self,
        splat_a: Splat,
        splat_b: Splat
    ) -> float:
        """Compute similarity between two splats.
        
        Args:
            splat_a: First splat
            splat_b: Second splat
            
        Returns:
            Similarity value between 0 and 1
        """
        pass
    
    @abstractmethod
    def compute_coverage_uniformity(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        """Compute how uniformly a splat covers its region.
        
        Args:
            splat: Splat to compute coverage for
            registry: Registry containing all splats
            tokens: Optional token embeddings
            
        Returns:
            Uniformity value between 0 and 1
        """
        pass
    
    @abstractmethod
    def compute_information_contribution(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        """Compute information-theoretic contribution of a splat.
        
        Args:
            splat: Splat to compute contribution for
            registry: Registry containing all splats
            tokens: Optional token embeddings
            
        Returns:
            Contribution value (higher means more important)
        """
        pass


class SplatCandidateEvaluator(ABC):
    """Abstract interface for evaluating splat candidates during adaptation."""
    
    @abstractmethod
    def evaluate_mitosis_candidates(
        self,
        original_splat: Splat,
        candidates: List[Tuple[Splat, Splat]],
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> Tuple[Splat, Splat]:
        """Evaluate candidate splat pairs for mitosis operation.
        
        Args:
            original_splat: Original splat being split
            candidates: List of (splat1, splat2) tuples to evaluate
            registry: Registry containing all splats
            tokens: Optional token embeddings
            
        Returns:
            Best (splat1, splat2) pair from candidates
        """
        pass
    
    @abstractmethod
    def evaluate_merge_candidates(
        self,
        splat_a: Splat,
        splat_b: Splat,
        merge_candidates: List[Splat],
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> Splat:
        """Evaluate candidate splats for merge operation.
        
        Args:
            splat_a: First splat being merged
            splat_b: Second splat being merged
            merge_candidates: List of potential merged splats
            registry: Registry containing all splats
            tokens: Optional token embeddings
            
        Returns:
            Best merged splat from candidates
        """
        pass
    
    @abstractmethod
    def evaluate_birth_candidates(
        self,
        candidates: List[Splat],
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> Splat:
        """Evaluate candidate splats for birth operation.
        
        Args:
            candidates: List of potential new splats
            registry: Registry containing all splats
            tokens: Optional token embeddings
            
        Returns:
            Best new splat from candidates
        """
        pass
    
    @abstractmethod
    def evaluate_adjust_candidates(
        self,
        original_splat: Splat,
        candidates: List[Splat],
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> Splat:
        """Evaluate candidate parameter adjustments.
        
        Args:
            original_splat: Original splat being adjusted
            candidates: List of potential adjusted splats
            registry: Registry containing all splats
            tokens: Optional token embeddings
            
        Returns:
            Best adjusted splat from candidates
        """
        pass


class AdaptationMetricsAggregator:
    """Aggregates metrics across multiple splats for system-level analysis."""
    
    def __init__(self, metrics_computer: AdaptationMetricsComputer):
        """Initialize metrics aggregator.
        
        Args:
            metrics_computer: Metrics computer to use for individual splats
        """
        self.metrics_computer = metrics_computer
    
    def compute_all_metrics(
        self,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> Dict[str, AdaptationMetrics]:
        """Compute metrics for all splats in registry.
        
        Args:
            registry: Registry containing all splats
            tokens: Optional token embeddings
            
        Returns:
            Dictionary mapping splat IDs to metrics
        """
        metrics = {}
        for splat in registry.get_all_splats():
            metrics[splat.id] = self.metrics_computer.compute_metrics(
                splat, registry, tokens
            )
        return metrics
    
    def compute_level_metrics(
        self,
        level: str,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> Dict[str, AdaptationMetrics]:
        """Compute metrics for all splats at a specific level.
        
        Args:
            level: Hierarchical level name
            registry: Registry containing all splats
            tokens: Optional token embeddings
            
        Returns:
            Dictionary mapping splat IDs to metrics
        """
        metrics = {}
        for splat in registry.get_splats_at_level(level):
            metrics[splat.id] = self.metrics_computer.compute_metrics(
                splat, registry, tokens
            )
        return metrics
    
    def compute_similarity_matrix(
        self,
        splats: List[Splat]
    ) -> np.ndarray:
        """Compute pairwise similarity matrix between splats.
        
        Args:
            splats: List of splats to compare
            
        Returns:
            Similarity matrix of shape [len(splats), len(splats)]
        """
        n = len(splats)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            # Diagonal is always 1 (self-similarity)
            similarity_matrix[i, i] = 1.0
            
            # Compute upper triangle
            for j in range(i + 1, n):
                sim = self.metrics_computer.compute_similarity(splats[i], splats[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim  # Matrix is symmetric
        
        return similarity_matrix
    
    def find_similar_splats(
        self,
        registry: SplatRegistry,
        threshold: float = 0.9,
        same_level_only: bool = True
    ) -> List[Tuple[Splat, Splat, float]]:
        """Find pairs of similar splats in the registry.
        
        Args:
            registry: Registry containing all splats
            threshold: Similarity threshold
            same_level_only: Whether to only compare splats at the same level
            
        Returns:
            List of (splat_a, splat_b, similarity) tuples
        """
        similar_pairs = []
        
        if same_level_only:
            # Process each level separately
            for level in registry.hierarchy.levels:
                splats = list(registry.get_splats_at_level(level))
                similarity_matrix = self.compute_similarity_matrix(splats)
                
                # Find similar pairs (upper triangle only to avoid duplicates)
                for i in range(len(splats)):
                    for j in range(i + 1, len(splats)):
                        sim = similarity_matrix[i, j]
                        if sim >= threshold:
                            similar_pairs.append((splats[i], splats[j], sim))
        else:
            # Compare all splats regardless of level
            splats = registry.get_all_splats()
            similarity_matrix = self.compute_similarity_matrix(splats)
            
            # Find similar pairs (upper triangle only to avoid duplicates)
            for i in range(len(splats)):
                for j in range(i + 1, len(splats)):
                    sim = similarity_matrix[i, j]
                    if sim >= threshold:
                        similar_pairs.append((splats[i], splats[j], sim))
        
        # Sort by similarity (highest first)
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        
        return similar_pairs
    
    def find_splats_for_death(
        self,
        registry: SplatRegistry,
        activation_threshold: float = 0.01,
        min_lifetime: int = 10
    ) -> List[Tuple[Splat, float]]:
        """Find splats that are candidates for removal.
        
        Args:
            registry: Registry containing all splats
            activation_threshold: Maximum activation for death candidates
            min_lifetime: Minimum lifetime before considering for death
            
        Returns:
            List of (splat, activation) tuples sorted by activation (lowest first)
        """
        candidates = []
        
        for splat in registry.get_all_splats():
            # Skip splats that haven't lived long enough
            if splat.lifetime < min_lifetime:
                continue
            
            activation = self.metrics_computer.compute_splat_activation(splat)
            if activation <= activation_threshold:
                candidates.append((splat, activation))
        
        # Sort by activation (lowest first)
        candidates.sort(key=lambda x: x[1])
        
        return candidates
    
    def find_splats_for_mitosis(
        self,
        registry: SplatRegistry,
        activation_threshold: float = 0.8,
        variance_threshold: float = 0.5,
        min_lifetime: int = 10,
        tokens: Optional[np.ndarray] = None
    ) -> List[Tuple[Splat, float, float]]:
        """Find splats that are candidates for splitting.
        
        Args:
            registry: Registry containing all splats
            activation_threshold: Minimum activation for mitosis candidates
            variance_threshold: Minimum variance for mitosis candidates
            min_lifetime: Minimum lifetime before considering for mitosis
            tokens: Optional token embeddings
            
        Returns:
            List of (splat, activation, variance) tuples
        """
        candidates = []
        
        for splat in registry.get_all_splats():
            # Skip splats that haven't lived long enough
            if splat.lifetime < min_lifetime:
                continue
            
            activation = self.metrics_computer.compute_splat_activation(splat)
            variance = self.metrics_computer.compute_splat_variance(splat, tokens)
            
            if activation >= activation_threshold or variance >= variance_threshold:
                candidates.append((splat, activation, variance))
        
        # Sort by combined score (activation + variance)
        candidates.sort(key=lambda x: x[1] + x[2], reverse=True)
        
        return candidates
