"""
Implementation of information-theoretic candidate evaluation for HSA adaptation.

This module implements the InfoTheoreticCandidateEvaluator class, which evaluates
adaptation candidates using information-theoretic metrics.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import logging

from .splat import Splat
from .registry import SplatRegistry
from .adaptation_metrics_base import SplatCandidateEvaluator
from .metrics_computation import InfoTheoreticMetricsComputer

# Configure logging
logger = logging.getLogger(__name__)


class InfoTheoreticCandidateEvaluator(SplatCandidateEvaluator):
    """
    Evaluates splat candidates using information-theoretic principles.
    
    This class provides methods to select the best candidates for various
    adaptation operations based on information theory.
    """
    
    def __init__(
        self,
        metrics_computer: Optional[InfoTheoreticMetricsComputer] = None,
        coverage_weight: float = 0.4,
        uniformity_weight: float = 0.3,
        info_gain_weight: float = 0.3
    ):
        """Initialize candidate evaluator.
        
        Args:
            metrics_computer: Metrics computer to use
            coverage_weight: Weight for coverage score
            uniformity_weight: Weight for uniformity score
            info_gain_weight: Weight for information gain score
        """
        self.metrics_computer = metrics_computer or InfoTheoreticMetricsComputer()
        self.coverage_weight = coverage_weight
        self.uniformity_weight = uniformity_weight
        self.info_gain_weight = info_gain_weight
    
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
        if not candidates:
            raise ValueError("No candidates provided")
        
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0]
        
        # Compute scores for each candidate
        scores = []
        
        for splat1, splat2 in candidates:
            # Compute metrics for both splats
            metrics1 = self.metrics_computer.compute_metrics(splat1, registry, tokens)
            metrics2 = self.metrics_computer.compute_metrics(splat2, registry, tokens)
            
            # Coverage score: how well do they cover the original splat's region
            coverage_score = self._compute_coverage_score(
                original_splat, splat1, splat2, tokens
            )
            
            # Uniformity score: how uniform is each splat's coverage
            uniformity_score = (
                metrics1.coverage_uniformity + metrics2.coverage_uniformity
            ) / 2
            
            # Information gain score: how much info do they provide
            info_gain_score = (
                metrics1.information_contribution + metrics2.information_contribution
            ) / 2
            
            # Compute overall score
            overall_score = (
                self.coverage_weight * coverage_score +
                self.uniformity_weight * uniformity_score +
                self.info_gain_weight * info_gain_score
            )
            
            scores.append(overall_score)
        
        # Find best candidate
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
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
        if not merge_candidates:
            raise ValueError("No candidates provided")
        
        # If only one candidate, return it
        if len(merge_candidates) == 1:
            return merge_candidates[0]
        
        # Compute scores for each candidate
        scores = []
        
        for candidate in merge_candidates:
            # Compute metrics for the candidate
            metrics = self.metrics_computer.compute_metrics(candidate, registry, tokens)
            
            # Coverage score: how well does it cover both original splats
            coverage_score = self._compute_merge_coverage_score(
                splat_a, splat_b, candidate, tokens
            )
            
            # Uniformity score: how uniform is its coverage
            uniformity_score = metrics.coverage_uniformity
            
            # Information gain score
            info_gain_score = metrics.information_contribution
            
            # Compute overall score
            overall_score = (
                self.coverage_weight * coverage_score +
                self.uniformity_weight * uniformity_score +
                self.info_gain_weight * info_gain_score
            )
            
            scores.append(overall_score)
        
        # Find best candidate
        best_idx = np.argmax(scores)
        return merge_candidates[best_idx]
    
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
        if not candidates:
            raise ValueError("No candidates provided")
        
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0]
        
        # Compute scores for each candidate
        scores = []
        
        for candidate in candidates:
            # Compute metrics for the candidate
            metrics = self.metrics_computer.compute_metrics(candidate, registry, tokens)
            
            # For birth, we're primarily interested in information gain
            info_gain_score = metrics.information_contribution
            
            # We also want good uniformity
            uniformity_score = metrics.coverage_uniformity
            
            # For coverage, we want to cover regions not already covered
            # This is implicitly part of information_contribution
            
            # Compute overall score
            overall_score = (
                0.2 * uniformity_score +
                0.8 * info_gain_score
            )
            
            scores.append(overall_score)
        
        # Find best candidate
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
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
        if not candidates:
            raise ValueError("No candidates provided")
        
        # If only one candidate, return it
        if len(candidates) == 1:
            return candidates[0]
        
        # Compute scores for each candidate
        scores = []
        
        for candidate in candidates:
            # Compute metrics for the candidate
            metrics = self.metrics_computer.compute_metrics(candidate, registry, tokens)
            
            # For adjustment, we want to improve information contribution
            info_gain_score = metrics.information_contribution
            
            # We also want to maintain good coverage
            coverage_score = self._compute_adjust_coverage_score(
                original_splat, candidate, tokens
            )
            
            # And good uniformity
            uniformity_score = metrics.coverage_uniformity
            
            # Compute overall score
            overall_score = (
                self.coverage_weight * coverage_score +
                self.uniformity_weight * uniformity_score +
                self.info_gain_weight * info_gain_score
            )
            
            scores.append(overall_score)
        
        # Find best candidate
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def _compute_coverage_score(
        self,
        original_splat: Splat,
        splat1: Splat,
        splat2: Splat,
        tokens: Optional[np.ndarray]
    ) -> float:
        """Compute how well two splats cover the original splat's region.
        
        Args:
            original_splat: Original splat
            splat1: First candidate splat
            splat2: Second candidate splat
            tokens: Optional token embeddings
            
        Returns:
            Coverage score between 0 and 1
        """
        if tokens is None or tokens.shape[0] == 0:
            # Without tokens, use a geometric heuristic
            # Measure overlap of covariance ellipsoids
            
            # Simple approximation: use distance between centers
            dist1 = np.linalg.norm(splat1.position - original_splat.position)
            dist2 = np.linalg.norm(splat2.position - original_splat.position)
            
            # Normalize by original splat's "radius"
            radius = np.sqrt(np.trace(original_splat.covariance) / original_splat.dim)
            norm_dist1 = dist1 / max(radius, 1e-6)
            norm_dist2 = dist2 / max(radius, 1e-6)
            
            # Compute coverage scores (closer is better)
            coverage1 = np.exp(-norm_dist1)
            coverage2 = np.exp(-norm_dist2)
            
            # Average coverage
            return (coverage1 + coverage2) / 2
        
        # With tokens, compute actual coverage
        # Get token weights for the original splat
        original_weights = self._compute_token_weights(original_splat, tokens)
        
        # Get token weights for each candidate
        weights1 = self._compute_token_weights(splat1, tokens)
        weights2 = self._compute_token_weights(splat2, tokens)
        
        # Combined coverage
        combined_weights = np.maximum(weights1, weights2)
        
        # Compute coverage ratio
        total_original = np.sum(original_weights)
        covered = np.sum(np.minimum(original_weights, combined_weights))
        
        if total_original > 0:
            coverage_ratio = covered / total_original
        else:
            coverage_ratio = 0.0
        
        return coverage_ratio
