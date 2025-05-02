"""
Implementation of information-theoretic metrics computation for HSA adaptation.

This module implements the InfoTheoreticMetricsComputer class, which computes
adaptation metrics based on information theory principles.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import logging
import math

from .splat import Splat
from .registry import SplatRegistry
from .adaptation_types import AdaptationMetrics
from .adaptation_metrics_base import AdaptationMetricsComputer

# Configure logging
logger = logging.getLogger(__name__)


class InfoTheoreticMetricsComputer(AdaptationMetricsComputer):
    """
    Computes adaptation metrics based on information theory principles.
    
    This class provides a rigorous framework for measuring the effectiveness of
    splats and guiding adaptation decisions using information-theoretic concepts.
    """
    
    def __init__(
        self, 
        entropy_smoothing: float = 0.001,
        activation_scale: float = 1.0,
        trend_window: int = 5,
        dynamic_thresholds: bool = True
    ):
        """Initialize metrics computer.
        
        Args:
            entropy_smoothing: Smoothing factor for entropy calculations
            activation_scale: Scaling factor for activation values
            trend_window: Window size for trend calculations
            dynamic_thresholds: Whether to use dynamic thresholds
        """
        self.entropy_smoothing = entropy_smoothing
        self.activation_scale = activation_scale
        self.trend_window = trend_window
        self.dynamic_thresholds = dynamic_thresholds
        
        # Track statistics for dynamic thresholds
        self.activation_stats = {
            "mean": 0.0,
            "std": 0.0,
            "count": 0
        }
        self.variance_stats = {
            "mean": 0.0,
            "std": 0.0,
            "count": 0
        }
    
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
        # Compute individual metrics
        activation_mean = self.compute_splat_activation(splat, tokens)
        activation_trend = self.compute_activation_trend(splat)
        variance = self.compute_splat_variance(splat, tokens)
        
        # Compute coverage uniformity
        coverage_uniformity = self.compute_coverage_uniformity(splat, registry, tokens)
        
        # Compute information contribution
        info_contribution = self.compute_information_contribution(splat, registry, tokens)
        
        # Compute similarity to other splats
        similarity_dict = {}
        
        # Get splats at the same level
        same_level_splats = list(registry.get_splats_at_level(splat.level))
        
        for other_splat in same_level_splats:
            if other_splat.id != splat.id:
                similarity = self.compute_similarity(splat, other_splat)
                similarity_dict[other_splat.id] = similarity
        
        # Update statistics for dynamic thresholds
        if self.dynamic_thresholds:
            self._update_stats(activation_mean, variance)
        
        # Create and return metrics
        return AdaptationMetrics(
            activation_mean=activation_mean,
            activation_trend=activation_trend,
            information_contribution=info_contribution,
            coverage_uniformity=coverage_uniformity,
            variance=variance,
            similarity_to_others=similarity_dict
        )
    
    def compute_splat_activation(
        self,
        splat: Splat,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        """Compute activation metric for a splat.
        
        If tokens are provided, computes context-specific activation.
        Otherwise, uses historical activation values.
        
        Args:
            splat: Splat to compute activation for
            tokens: Optional token embeddings
            
        Returns:
            Activation value between 0 and 1
        """
        if tokens is not None and tokens.shape[0] > 0:
            # Compute activation with respect to provided tokens
            total_activation = 0.0
            count = 0
            
            # Sample tokens if there are too many
            max_tokens = 100
            if tokens.shape[0] > max_tokens:
                indices = np.linspace(0, tokens.shape[0] - 1, max_tokens, dtype=int)
                sample_tokens = tokens[indices]
            else:
                sample_tokens = tokens
            
            # Compute average activation
            for i in range(sample_tokens.shape[0]):
                for j in range(sample_tokens.shape[0]):
                    token_i = sample_tokens[i]
                    token_j = sample_tokens[j]
                    
                    activation = splat.compute_attention(token_i, token_j)
                    total_activation += activation
                    count += 1
            
            if count > 0:
                avg_activation = total_activation / count
            else:
                avg_activation = 0.0
            
            # Blend with historical activation
            historical_activation = splat.get_average_activation()
            blended_activation = 0.7 * avg_activation + 0.3 * historical_activation
            
            return blended_activation * self.activation_scale
        else:
            # Use historical activation
            return splat.get_average_activation() * self.activation_scale
    
    def compute_activation_trend(self, splat: Splat) -> float:
        """Compute activation trend over time.
        
        Args:
            splat: Splat to compute trend for
            
        Returns:
            Trend value (positive for increasing, negative for decreasing)
        """
        # Get activation history
        history = splat.activation_history.get_values()
        
        # If not enough history, return neutral value
        if len(history) < 2:
            return 0.0
        
        # Use a window to focus on recent trend
        window = min(self.trend_window, len(history))
        recent = history[-window:]
        
        # Simple linear regression
        x = np.arange(window)
        y = np.array(recent)
        
        # Compute trend (slope)
        if np.std(x) == 0:
            return 0.0
            
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sum((x - mean_x) ** 2)
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        
        # Normalize to [-1, 1]
        normalized_slope = np.tanh(slope * 10)
        
        return normalized_slope
    
    def compute_splat_variance(
        self,
        splat: Splat,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        """Compute internal variance of a splat.
        
        This measures how diverse the token embeddings covered by this splat are.
        
        Args:
            splat: Splat to compute variance for
            tokens: Optional token embeddings
            
        Returns:
            Variance value between 0 and 1
        """
        # If no tokens provided, use covariance trace as a measure
        if tokens is None or tokens.shape[0] == 0:
            if hasattr(splat, 'covariance') and splat.covariance is not None:
                # Use normalized trace of covariance matrix
                trace = np.trace(splat.covariance)
                normalized_trace = min(1.0, trace / splat.dim)
                return normalized_trace
            else:
                return 0.5  # Default value
        
        # With tokens, compute actual variance of tokens covered by this splat
        token_weights = self._compute_token_weights(splat, tokens)
        
        # If no tokens have significant weight, return default
        if np.sum(token_weights) < 1e-6:
            return 0.5
        
        # Normalize weights
        weights = token_weights / np.sum(token_weights)
        
        # Compute weighted mean
        weighted_mean = np.sum(tokens * weights[:, np.newaxis], axis=0)
        
        # Compute weighted variance
        variance = 0.0
        for i in range(tokens.shape[0]):
            if weights[i] > 0:
                delta = tokens[i] - weighted_mean
                variance += weights[i] * np.sum(delta ** 2)
        
        # Normalize variance
        normalized_variance = min(1.0, variance / splat.dim)
        
        return normalized_variance
    
    def compute_similarity(self, splat_a: Splat, splat_b: Splat) -> float:
        """Compute similarity between two splats.
        
        Args:
            splat_a: First splat
            splat_b: Second splat
            
        Returns:
            Similarity value between 0 and 1
        """
        # Ensure dimensions match
        if splat_a.dim != splat_b.dim:
            return 0.0
        
        try:
            # Position similarity (based on Mahalanobis distance)
            delta = splat_a.position - splat_b.position
            
            # Use average of both covariance matrices
            avg_cov_inv = 0.5 * (splat_a.covariance_inverse + splat_b.covariance_inverse)
            mahalanobis = delta @ avg_cov_inv @ delta
            
            # Convert to similarity
            pos_similarity = np.exp(-0.5 * mahalanobis)
            
            # Covariance similarity using Frobenius norm
            cov_diff = splat_a.covariance - splat_b.covariance
            cov_norm = np.linalg.norm(cov_diff, 'fro')
            max_norm = np.sqrt(np.linalg.norm(splat_a.covariance, 'fro') * 
                              np.linalg.norm(splat_b.covariance, 'fro'))
            
            # Convert to similarity
            cov_similarity = np.exp(-cov_norm / max(1e-6, max_norm))
            
            # Amplitude similarity
            amp_diff = abs(splat_a.amplitude - splat_b.amplitude)
            amp_similarity = np.exp(-amp_diff)
            
            # Combine similarities
            similarity = 0.6 * pos_similarity + 0.3 * cov_similarity + 0.1 * amp_similarity
            
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            
            # Fall back to position-based similarity
            dist = np.linalg.norm(splat_a.position - splat_b.position)
            return float(np.exp(-dist))
    
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
        if tokens is None or tokens.shape[0] == 0:
            # Without tokens, use a heuristic based on covariance
            try:
                # Compute eigenvalues of covariance
                eigenvalues = np.linalg.eigvalsh(splat.covariance)
                
                # Compute ratio of smallest to largest eigenvalue
                # A more uniform coverage would have more similar eigenvalues
                min_eig = max(1e-6, eigenvalues[0])
                max_eig = max(1e-6, eigenvalues[-1])
                
                ratio = min_eig / max_eig
                
                # Convert to uniformity
                return float(ratio)
                
            except Exception as e:
                logger.warning(f"Error computing coverage uniformity: {e}")
                return 0.5  # Default value
        
        # With tokens, compute entropy of token weights
        token_weights = self._compute_token_weights(splat, tokens)
        
        # If no tokens have significant weight, return default
        if np.sum(token_weights) < 1e-6:
            return 0.5
        
        # Normalize weights
        weights = token_weights / np.sum(token_weights)
        
        # Compute entropy
        entropy = 0.0
        for w in weights:
            if w > 0:
                entropy -= w * math.log(w + self.entropy_smoothing)
        
        # Normalize to [0, 1]
        max_entropy = math.log(len(weights))
        if max_entropy > 0:
            normalized_entropy = entropy / max_entropy
        else:
            normalized_entropy = 1.0
        
        return float(normalized_entropy)
    
    def compute_information_contribution(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        """Compute information-theoretic contribution of a splat.
        
        This measures how much unique information this splat provides
        compared to other splats in the registry.
        
        Args:
            splat: Splat to compute contribution for
            registry: Registry containing all splats
            tokens: Optional token embeddings
            
        Returns:
            Contribution value (higher means more important)
        """
        if tokens is None or tokens.shape[0] == 0:
            # Without tokens, use a heuristic based on activation and uniqueness
            activation = splat.get_average_activation()
            
            # Compute average similarity to other splats at same level
            avg_similarity = 0.0
            count = 0
            
            same_level_splats = list(registry.get_splats_at_level(splat.level))
            for other_splat in same_level_splats:
                if other_splat.id != splat.id:
                    similarity = self.compute_similarity(splat, other_splat)
                    avg_similarity += similarity
                    count += 1
            
            if count > 0:
                avg_similarity /= count
            else:
                avg_similarity = 0.0
            
            # Compute uniqueness (inverse of similarity)
            uniqueness = 1.0 - avg_similarity
            
            # Contribution = activation * uniqueness
            contribution = activation * uniqueness
            
            return float(contribution)
        
        # With tokens, compute information gain
        return self._compute_token_information_gain(splat, registry, tokens)
    
    def _compute_token_weights(
        self,
        splat: Splat,
        tokens: np.ndarray
    ) -> np.ndarray:
        """Compute weight of each token for a splat.
        
        Args:
            splat: Splat to compute weights for
            tokens: Token embeddings
            
        Returns:
            Array of token weights
        """
        # Compute Mahalanobis distances
        deltas = tokens - splat.position
        
        if hasattr(splat, 'covariance_inverse') and splat.covariance_inverse is not None:
            try:
                # Transform deltas
                transformed = np.dot(deltas, splat.covariance_inverse)
                
                # Compute Mahalanobis distances
                distances = np.sum(transformed * deltas, axis=1)
                
                # Convert to weights
                weights = np.exp(-0.5 * distances)
                
                return weights
            except:
                # Fall back to Euclidean distance
                pass
        
        # Fall back to Euclidean distance
        distances = np.linalg.norm(deltas, axis=1)
        weights = np.exp(-0.5 * distances ** 2)
        
        return weights
    
    def _compute_token_information_gain(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: np.ndarray
    ) -> float:
        """Compute information gain for tokens covered by a splat.
        
        Args:
            splat: Splat to compute gain for
            registry: Registry containing all splats
            tokens: Token embeddings
            
        Returns:
            Information gain value
        """
        # Compute token weights for this splat
        token_weights = self._compute_token_weights(splat, tokens)
        
        # Compute token weights for all other splats
        other_weights = np.zeros_like(token_weights)
        
        # Get all splats at same level
        same_level_splats = list(registry.get_splats_at_level(splat.level))
        
        for other_splat in same_level_splats:
            if other_splat.id != splat.id:
                weights = self._compute_token_weights(other_splat, tokens)
                other_weights = np.maximum(other_weights, weights)
        
        # Compute difference in coverage
        unique_coverage = np.maximum(token_weights - other_weights, 0.0)
        
        # Information gain is the sum of unique coverage
        gain = np.sum(unique_coverage)
        
        # Normalize
        total_weight = np.sum(token_weights)
        if total_weight > 0:
            normalized_gain = gain / total_weight
        else:
            normalized_gain = 0.0
        
        return float(normalized_gain)
    
    def _update_stats(self, activation: float, variance: float) -> None:
        """Update statistics for dynamic thresholds.
        
        Args:
            activation: Current activation value
            variance: Current variance value
        """
        # Update activation stats with Welford's online algorithm
        count = self.activation_stats["count"] + 1
        delta = activation - self.activation_stats["mean"]
        self.activation_stats["mean"] += delta / count
        delta2 = activation - self.activation_stats["mean"]
        self.activation_stats["std"] += delta * delta2
        self.activation_stats["count"] = count
        
        # Update variance stats
        count = self.variance_stats["count"] + 1
        delta = variance - self.variance_stats["mean"]
        self.variance_stats["mean"] += delta / count
        delta2 = variance - self.variance_stats["mean"]
        self.variance_stats["std"] += delta * delta2
        self.variance_stats["count"] = count
        
        # Compute standard deviations if we have enough samples
        if self.activation_stats["count"] > 1:
            self.activation_stats["std"] = np.sqrt(
                self.activation_stats["std"] / (self.activation_stats["count"] - 1)
            )
        
        if self.variance_stats["count"] > 1:
            self.variance_stats["std"] = np.sqrt(
                self.variance_stats["std"] / (self.variance_stats["count"] - 1)
            )
