"""
Attention metrics module for Hierarchical Splat Attention (HSA).

This module provides metrics and monitoring for HSA:
- Splat activation calculation
- Error contribution measurement
- Attention quality analysis
- Statistics for adaptation decisions
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union, Any
import math

# Import core data structures
from hsa.data_structures import Splat, Hierarchy, SplatRegistry

class SplatAttentionMetrics:
    """
    Computes and tracks metrics related to splat attention.
    
    This class is useful for monitoring splat effectiveness and
    informing adaptation decisions.
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.splat_activations = {}  # Maps splat ID to activation level
        self.splat_error_contributions = {}  # Maps splat ID to error contribution
        self.recent_values = {}  # Maps metric name to recent values (for statistics)
        self.attention_statistics = {}  # Maps statistic name to value
    
    def compute_splat_activation(
        self, 
        splat: Splat, 
        tokens: np.ndarray, 
        attention_matrix: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute the activation level of a splat.
        
        This measures how much the splat contributes to the overall attention pattern.
        
        Args:
            splat: The splat to evaluate
            tokens: Token embeddings
            attention_matrix: The full attention matrix (optional)
            
        Returns:
            Activation level (higher means more active)
        """
        sequence_length = tokens.shape[0]
        
        # Extract splat parameters for efficiency
        pos = splat.position
        cov_inv = splat.covariance_inverse
        amp = splat.amplitude
        
        # First, find tokens within splat's influence
        diffs = tokens - pos
        distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
        relevant_indices = np.where(distances < 3.0)[0]  # 3 std deviations
        
        # If no relevant tokens, activation is zero
        if len(relevant_indices) == 0:
            self.splat_activations[splat.id] = 0.0
            return 0.0
        
        # Compute activation based on relevant tokens only
        total_attention = 0.0
        count = 0
        
        # Limit number of tokens for efficiency
        max_tokens = min(50, len(relevant_indices))
        if len(relevant_indices) > max_tokens:
            # Take a sample of relevant tokens
            relevant_indices = np.random.choice(relevant_indices, size=max_tokens, replace=False)
        
        for i in relevant_indices:
            for j in relevant_indices:
                # Compute attention score
                diff = (tokens[i] - tokens[j]) - pos
                dist = np.sqrt(diff @ cov_inv @ diff)
                score = amp * np.exp(-dist**2)
                
                total_attention += score
                count += 1
        
        # Average attention
        activation = total_attention / max(1, count)
        
        # Store the result
        self.splat_activations[splat.id] = activation
        
        # Update recent values for computing statistics
        if "activation" not in self.recent_values:
            self.recent_values["activation"] = []
        self.recent_values["activation"].append(activation)
        # Keep only recent values (last 100)
        if len(self.recent_values["activation"]) > 100:
            self.recent_values["activation"] = self.recent_values["activation"][-100:]
        
        return activation
    
    def compute_splat_error_contribution(
        self,
        splat: Splat,
        tokens: np.ndarray,
        target_attention: np.ndarray,
        current_attention: np.ndarray
    ) -> float:
        """
        Compute how much a splat contributes to attention error.
        
        This helps identify splats that need adaptation.
        
        Args:
            splat: The splat to evaluate
            tokens: Token embeddings
            target_attention: The target/ideal attention matrix
            current_attention: The current computed attention matrix
            
        Returns:
            Error contribution (higher means more error)
        """
        sequence_length = tokens.shape[0]
        
        # Extract splat parameters for efficiency
        pos = splat.position
        cov_inv = splat.covariance_inverse
        amp = splat.amplitude
        
        # Find tokens within splat's influence (vectorized)
        diffs = tokens - pos
        distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
        relevant_indices = np.where(distances < 3.0)[0]  # 3 std deviations
        
        # If no relevant tokens, error contribution is zero
        if len(relevant_indices) == 0:
            self.splat_error_contributions[splat.id] = 0.0
            return 0.0
        
        # Limit number of tokens for efficiency
        max_tokens = min(50, len(relevant_indices))
        if len(relevant_indices) > max_tokens:
            # Take a sample of relevant tokens
            relevant_indices = np.random.choice(relevant_indices, size=max_tokens, replace=False)
        
        # Compute the attention map for just this splat (sparse)
        splat_attention = np.zeros((len(relevant_indices), len(relevant_indices)))
        for i_idx, i in enumerate(relevant_indices):
            for j_idx, j in enumerate(relevant_indices):
                diff = (tokens[i] - tokens[j]) - pos
                dist = np.sqrt(diff @ cov_inv @ diff)
                splat_attention[i_idx, j_idx] = amp * np.exp(-dist**2)
 
         # Add safety checks before error calculation
        if np.isnan(target_attention).any() or np.isnan(current_attention).any():
            # Replace NaN with zeros
            target_attention = np.nan_to_num(target_attention, nan=0.0)
            current_attention = np.nan_to_num(current_attention, nan=0.0)
        
        # Calculate attention error for the relevant tokens
        error = np.zeros_like(splat_attention)
        for i_idx, i in enumerate(relevant_indices):
            for j_idx, j in enumerate(relevant_indices):
                error[i_idx, j_idx] = target_attention[i, j] - current_attention[i, j]
                # Check for invalid results and fix them
                if np.isnan(error[i_idx, j_idx]) or np.isinf(error[i_idx, j_idx]):
                    error[i_idx, j_idx] = 0.0
                        
        # Compute how much this splat's attention aligns with the error
        error_contribution = 0.0
        for i_idx in range(len(relevant_indices)):
            for j_idx in range(len(relevant_indices)):
                error_contribution += splat_attention[i_idx, j_idx] * error[i_idx, j_idx]
        
        # Take absolute value
        error_contribution = abs(error_contribution)
        
        # Normalize by number of relevant token pairs
        error_contribution /= max(1, len(relevant_indices)**2)
        
        # Store the result
        self.splat_error_contributions[splat.id] = error_contribution
        
        # Update recent values for computing statistics
        if "error_contribution" not in self.recent_values:
            self.recent_values["error_contribution"] = []
        self.recent_values["error_contribution"].append(error_contribution)
        # Keep only recent values (last 100)
        if len(self.recent_values["error_contribution"]) > 100:
            self.recent_values["error_contribution"] = self.recent_values["error_contribution"][-100:]
        
        return error_contribution
    
    def compute_attention_sparsity(self, attention_matrix: np.ndarray) -> float:
        """
        Compute the sparsity (percentage of zeros) of an attention matrix.
        
        Args:
            attention_matrix: Attention matrix
            
        Returns:
            Sparsity percentage (0.0 to 1.0)
        """
        # Count zeros
        zeros = np.count_nonzero(attention_matrix == 0)
        
        # Calculate percentage
        sparsity = zeros / attention_matrix.size
        
        # Store statistic
        self.attention_statistics["sparsity"] = sparsity
        
        return sparsity
    
    def compute_attention_entropy(self, attention_matrix: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Compute the entropy of the attention distribution.
        
        Higher entropy means more uniform (less focused) attention.
        
        Args:
            attention_matrix: Attention matrix
            epsilon: Small value to avoid log(0)
            
        Returns:
            Attention entropy
        """
        # Normalize rows to create probability distributions
        row_sums = attention_matrix.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, epsilon)  # Avoid division by zero
        normalized = attention_matrix / row_sums
        
        # Compute entropy for each row
        row_entropies = []
        for row in normalized:
            # Only consider non-zero elements
            mask = row > epsilon
            p = row[mask]
            
            # Entropy formula: -sum(p * log(p))
            entropy = -np.sum(p * np.log2(p + epsilon))
            row_entropies.append(entropy)
        
        # Average entropy across all rows
        mean_entropy = np.mean(row_entropies)
        
        # Store statistic
        self.attention_statistics["entropy"] = mean_entropy
        
        return mean_entropy
    
    def compute_level_contributions(
        self,
        tokens: np.ndarray,
        splat_registry: SplatRegistry,
        attention_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute the contribution of each hierarchical level to the attention.
        
        Args:
            tokens: Token embeddings
            splat_registry: Registry containing all splats
            attention_matrix: The full attention matrix
            
        Returns:
            Dictionary mapping level names to contribution percentages
        """
        from .attention_implementations import SparseAttentionComputer
        
        # Create temporary attention computer
        computer = SparseAttentionComputer(splat_registry.hierarchy)
        
        # Compute per-level contributions
        level_contributions = {}
        total_attention = np.sum(attention_matrix)
        
        if total_attention <= 0:
            # Default to hierarchy weights if total attention is zero
            for level in splat_registry.hierarchy.levels:
                level_contributions[level] = splat_registry.hierarchy.get_level_weight(level)
            return level_contributions
        
        for level in splat_registry.hierarchy.levels:
            # Create temporary registry with only this level's splats
            level_registry = SplatRegistry(splat_registry.hierarchy)
            for splat in splat_registry.get_splats_at_level(level):
                level_registry.register(splat)
            
            # Skip if no splats at this level
            if len(level_registry.splats) == 0:
                level_contributions[level] = 0.0
                continue
            
            # Compute attention for just this level
            level_attention = computer.compute_attention(tokens, level_registry)
            
            # Calculate relative contribution
            level_sum = np.sum(level_attention)
            level_contributions[level] = level_sum / total_attention if total_attention > 0 else 0.0
        
        # Store statistics
        self.attention_statistics["level_contributions"] = level_contributions
        
        return level_contributions
    
    def compute_attention_quality_metrics(
        self,
        attention_matrix: np.ndarray,
        target_attention: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute metrics for attention quality.
        
        Args:
            attention_matrix: Computed attention matrix
            target_attention: Optional target/ideal attention matrix
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Compute sparsity
        metrics["sparsity"] = self.compute_attention_sparsity(attention_matrix)
        
        # Compute entropy
        metrics["entropy"] = self.compute_attention_entropy(attention_matrix)
        
        # Compute error if target available
        if target_attention is not None:
            # Mean squared error
            mse = np.mean((attention_matrix - target_attention) ** 2)
            metrics["mse"] = mse
            
            # Frobenius norm of error
            frob_norm = np.linalg.norm(attention_matrix - target_attention, 'fro')
            metrics["frob_error"] = frob_norm
        
        # Store statistics
        for key, value in metrics.items():
            self.attention_statistics[key] = value
        
        return metrics
    
    def get_splat_metrics(self, splat_id: str) -> Dict[str, float]:
        """
        Get all metrics for a specific splat.
        
        Args:
            splat_id: ID of the splat
            
        Returns:
            Dictionary of metrics for the splat
        """
        return {
            "activation": self.splat_activations.get(splat_id, 0.0),
            "error_contribution": self.splat_error_contributions.get(splat_id, 0.0)
        }
    
    def get_top_active_splats(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top-n most active splats.
        
        Args:
            n: Number of splats to return
            
        Returns:
            List of (splat_id, activation) tuples
        """
        return sorted(
            self.splat_activations.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
    
    def get_top_error_contributors(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top-n splats contributing most to error.
        
        Args:
            n: Number of splats to return
            
        Returns:
            List of (splat_id, error_contribution) tuples
        """
        return sorted(
            self.splat_error_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
    
    def compute_statistics(self) -> Dict[str, float]:
        """
        Compute statistical information about metrics.
        
        These statistics are useful for thresholding in adaptation decisions.
        
        Returns:
            Dictionary of statistics
        """
        statistics = {}
        
        # Process each type of metric
        for metric_name, values in self.recent_values.items():
            if not values:
                continue
                
            # Basic statistics
            statistics[f"{metric_name}_mean"] = np.mean(values)
            statistics[f"{metric_name}_median"] = np.median(values)
            statistics[f"{metric_name}_std"] = np.std(values)
            
            # Percentiles
            statistics[f"{metric_name}_p10"] = np.percentile(values, 10)
            statistics[f"{metric_name}_p25"] = np.percentile(values, 25)
            statistics[f"{metric_name}_p75"] = np.percentile(values, 75)
            statistics[f"{metric_name}_p90"] = np.percentile(values, 90)
        
        # Add to attention statistics
        self.attention_statistics.update(statistics)
        
        return statistics
    
    def get_adaptive_threshold(
        self, 
        metric_name: str,
        percentile: float = 90.0,
        default_value: float = 0.1
    ) -> float:
        """
        Compute an adaptive threshold based on recent metrics.
        
        This is useful for adapting to different data patterns dynamically.
        
        Args:
            metric_name: Name of the metric
            percentile: Percentile to use (e.g., 90.0 for 90th percentile)
            default_value: Default value if not enough data
            
        Returns:
            Adaptive threshold value
        """
        values = self.recent_values.get(metric_name, [])
        
        if len(values) < 10:
            return default_value
            
        return float(np.percentile(values, percentile))
    
    def reset(self) -> None:
        """Reset all stored metrics."""
        self.splat_activations.clear()
        self.splat_error_contributions.clear()
        # Keep recent values for continuity in statistics
        # but clear attention statistics
        self.attention_statistics.clear()


def compute_pairwise_attention_correlation(
    attention_matrices: List[np.ndarray]
) -> np.ndarray:
    """
    Compute correlation matrix between multiple attention matrices.
    
    This is useful for comparing attention patterns from different models.
    
    Args:
        attention_matrices: List of attention matrices
        
    Returns:
        Correlation matrix of shape [num_matrices, num_matrices]
    """
    num_matrices = len(attention_matrices)
    
    # Flatten matrices for correlation computation
    flattened = [matrix.flatten() for matrix in attention_matrices]
    
    # Compute correlation matrix
    corr_matrix = np.zeros((num_matrices, num_matrices))
    
    for i in range(num_matrices):
        for j in range(num_matrices):
            corr = np.corrcoef(flattened[i], flattened[j])[0, 1]
            corr_matrix[i, j] = corr
    
    return corr_matrix


def compute_attention_similarity(
    matrix1: np.ndarray, 
    matrix2: np.ndarray,
    method: str = 'cosine'
) -> float:
    """
    Compute similarity between two attention matrices.
    
    Args:
        matrix1: First attention matrix
        matrix2: Second attention matrix
        method: Similarity method ('cosine', 'frobenius', 'correlation')
        
    Returns:
        Similarity score (higher means more similar)
    """
    # Ensure same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError(f"Matrices must have same shape: {matrix1.shape} vs {matrix2.shape}")
    
    # Flatten matrices
    flat1 = matrix1.flatten()
    flat2 = matrix2.flatten()
    
    if method == 'cosine':
        # Cosine similarity
        dot = np.dot(flat1, flat2)
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        similarity = dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        
    elif method == 'frobenius':
        # Similarity based on Frobenius norm of difference
        diff_norm = np.linalg.norm(matrix1 - matrix2, 'fro')
        max_norm = max(np.linalg.norm(matrix1, 'fro'), np.linalg.norm(matrix2, 'fro'))
        similarity = 1.0 - (diff_norm / max_norm) if max_norm > 0 else 1.0
        
    elif method == 'correlation':
        # Pearson correlation
        correlation = np.corrcoef(flat1, flat2)[0, 1]
        similarity = max(0.0, correlation)  # Ensure non-negative
        
    else:
        raise ValueError(f"Unknown similarity method: {method}")
    
    return similarity


def analyze_attention_patterns(
    attention_matrix: np.ndarray,
    sequence_length: int
) -> Dict[str, float]:
    """
    Analyze attention patterns for common structures.
    
    Identifies patterns like diagonal attention, local window attention,
    and global token attention.
    
    Args:
        attention_matrix: Attention matrix
        sequence_length: Length of the sequence
        
    Returns:
        Dictionary of pattern strengths
    """
    patterns = {}
    
    # Diagonal pattern (self-attention and nearby tokens)
    diagonal_mask = np.zeros_like(attention_matrix)
    width = max(1, int(0.1 * sequence_length))  # 10% diagonal width
    for i in range(sequence_length):
        for j in range(max(0, i-width), min(sequence_length, i+width+1)):
            diagonal_mask[i, j] = 1.0
    
    diagonal_attention = np.sum(attention_matrix * diagonal_mask) / np.sum(attention_matrix)
    patterns["diagonal"] = float(diagonal_attention)
    
    # Local window pattern (block diagonal)
    window_size = max(1, int(0.2 * sequence_length))  # 20% window size
    window_mask = np.zeros_like(attention_matrix)
    for i in range(sequence_length):
        window_start = max(0, i - window_size // 2)
        window_end = min(sequence_length, i + window_size // 2 + 1)
        window_mask[i, window_start:window_end] = 1.0
    
    local_window_attention = np.sum(attention_matrix * window_mask) / np.sum(attention_matrix)
    patterns["local_window"] = float(local_window_attention)
    
    # Global token pattern (rows or columns with high attention)
    row_sums = np.sum(attention_matrix, axis=1)
    col_sums = np.sum(attention_matrix, axis=0)
    
    # Find global tokens (tokens that attend to many others or are attended by many)
    row_threshold = np.percentile(row_sums, 90)  # Top 10% rows
    col_threshold = np.percentile(col_sums, 90)  # Top 10% columns
    
    global_rows = np.where(row_sums > row_threshold)[0]
    global_cols = np.where(col_sums > col_threshold)[0]
    
    # Calculate global token attention
    global_mask = np.zeros_like(attention_matrix)
    for i in global_rows:
        global_mask[i, :] = 1.0
    for j in global_cols:
        global_mask[:, j] = 1.0
    
    global_attention = np.sum(attention_matrix * global_mask) / np.sum(attention_matrix)
    patterns["global_tokens"] = float(global_attention)
    
    return patterns
