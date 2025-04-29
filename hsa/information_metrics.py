"""
Information Metrics module for Hierarchical Splat Attention (HSA).

This module provides information-theoretic metrics to enhance adaptation decisions:
- Attention entropy computation
- Information contribution quantification
- Mutual information analysis
- Information gradient calculation

These metrics help make more intelligent decisions about when to split, merge, or remove splats,
ensuring that the attention mechanism preserves important patterns while maintaining efficiency.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core data structures
from hsa.data_structures import Splat, SplatRegistry
from hsa.attention import create_attention_computer

def compute_attention_entropy(
    attention_matrix: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute the entropy of an attention distribution.
    
    Higher entropy indicates more diffuse attention patterns, while
    lower entropy indicates more focused attention.
    
    Args:
        attention_matrix: Attention matrix [sequence_length, sequence_length]
        epsilon: Small value to avoid log(0)
        
    Returns:
        Entropy value (higher means more diffuse attention)
    """
    # Normalize attention matrix to represent a probability distribution
    row_sums = attention_matrix.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, epsilon)  # Avoid division by zero
    norm_attn = attention_matrix / row_sums
    
    # Compute entropy for each row (token)
    entropies = []
    for row in norm_attn:
        # Only consider non-zero elements to avoid log(0)
        mask = row > epsilon
        p = row[mask]
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -np.sum(p * np.log2(p))
        entropies.append(entropy)
    
    # Return average entropy across all tokens
    return np.mean(entropies)

def compute_kl_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute the KL divergence between two attention matrices.
    
    KL(P||Q) measures how much information is lost when using Q to approximate P.
    
    Args:
        p: First attention matrix (ground truth) [sequence_length, sequence_length]
        q: Second attention matrix (approximation) [sequence_length, sequence_length]
        epsilon: Small value to avoid log(0) and division by zero
        
    Returns:
        KL divergence value (higher means more information loss)
    """
    # Normalize matrices to represent probability distributions
    p_sum = p.sum(axis=1, keepdims=True)
    q_sum = q.sum(axis=1, keepdims=True)
    
    p_sum = np.maximum(p_sum, epsilon)  # Avoid division by zero
    q_sum = np.maximum(q_sum, epsilon)  # Avoid division by zero
    
    p_norm = p / p_sum
    q_norm = q / q_sum
    
    # Compute KL divergence row by row
    kl_divs = []
    for p_row, q_row in zip(p_norm, q_norm):
        # Avoid log(0) and division by zero
        mask = (p_row > epsilon) & (q_row > epsilon)
        p_valid = p_row[mask]
        q_valid = q_row[mask]
        
        # Calculate KL divergence: sum(p * log(p/q))
        kl_div = np.sum(p_valid * np.log2(p_valid / q_valid))
        kl_divs.append(kl_div)
    
    # Return average KL divergence across all tokens
    return np.mean(kl_divs)

def compute_js_divergence(
    p: np.ndarray,
    q: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute the Jensen-Shannon divergence between two attention matrices.
    
    JS divergence is a symmetrized and smoothed version of KL divergence.
    
    Args:
        p: First attention matrix [sequence_length, sequence_length]
        q: Second attention matrix [sequence_length, sequence_length]
        epsilon: Small value to avoid numerical issues
        
    Returns:
        JS divergence value (higher means more dissimilar distributions)
    """
    # Create mixture distribution
    m = 0.5 * (p + q)
    
    # Compute KL divergences to the mixture
    kl_p_m = compute_kl_divergence(p, m, epsilon)
    kl_q_m = compute_kl_divergence(q, m, epsilon)
    
    # JS divergence is the average of the two KL divergences
    return 0.5 * (kl_p_m + kl_q_m)

def compute_mutual_information(
    joint_distribution: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute the mutual information of a joint distribution.
    
    Mutual information measures how much knowing one variable
    reduces uncertainty about the other.
    
    Args:
        joint_distribution: Joint probability distribution (e.g., attention matrix)
        epsilon: Small value to avoid numerical issues
        
    Returns:
        Mutual information value
    """
    # Ensure the input is a proper probability distribution
    joint = joint_distribution / (np.sum(joint_distribution) + epsilon)
    
    # Compute marginal distributions
    p_x = np.sum(joint, axis=1)
    p_y = np.sum(joint, axis=0)
    
    # Compute mutual information: sum(p(x,y) * log(p(x,y) / (p(x) * p(y))))
    mi = 0.0
    
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > epsilon and p_x[i] > epsilon and p_y[j] > epsilon:
                mi += joint[i, j] * np.log2(joint[i, j] / (p_x[i] * p_y[j]))
    
    return mi

def compute_splat_information_contribution(
    splat: Splat,
    tokens: np.ndarray,
    attention_matrix: np.ndarray,
    splat_registry: SplatRegistry,
    sparse_topk: int = 64
) -> float:
    """
    Compute how much unique information a splat contributes to the overall attention.
    
    This is measured by calculating the KL divergence between the full attention matrix
    and the attention matrix computed without this splat.
    
    Args:
        splat: The splat to evaluate
        tokens: Token embeddings [sequence_length, embedding_dim]
        attention_matrix: Current full attention matrix
        splat_registry: Registry containing all splats
        sparse_topk: Number of top attention scores to keep per token
        
    Returns:
        Information contribution score (higher means more important)
    """
    try:
        # Create a temporary registry without this splat
        temp_registry = SplatRegistry(splat_registry.hierarchy)
        for s in splat_registry.splats.values():
            if s.id != splat.id:
                temp_registry.register(s)
        
        # Create temporary attention computer
        temp_computer = create_attention_computer(
            hierarchy=splat_registry.hierarchy,
            sparse_topk=sparse_topk,
            efficient=True
        )
        
        # Compute attention without this splat
        attention_without_splat = temp_computer.compute_attention(tokens, temp_registry)
        
        # Compute KL divergence between full attention and attention without this splat
        # This measures how much information is lost when removing this splat
        kl_div = compute_kl_divergence(attention_matrix, attention_without_splat)
        
        return kl_div
    
    except Exception as e:
        logger.error(f"Error computing information contribution: {e}")
        return 0.0  # Default to zero on error

def compute_splat_attention_entropy(
    splat: Splat,
    tokens: np.ndarray
) -> float:
    """
    Compute the entropy of a splat's attention distribution.
    
    This measures how focused or diffuse a splat's attention pattern is.
    
    Args:
        splat: The splat to evaluate
        tokens: Token embeddings [sequence_length, embedding_dim]
        
    Returns:
        Entropy value (higher means more diffuse attention)
    """
    try:
        sequence_length = tokens.shape[0]
        
        # Compute the attention map for just this splat
        splat_attention = np.zeros((sequence_length, sequence_length))
        for i in range(sequence_length):
            for j in range(sequence_length):
                splat_attention[i, j] = splat.compute_attention(tokens[i], tokens[j])
        
        # Compute the entropy of this attention distribution
        return compute_attention_entropy(splat_attention)
    
    except Exception as e:
        logger.error(f"Error computing splat attention entropy: {e}")
        return 0.0  # Default to zero on error

def compute_information_gradient(
    splat: Splat,
    tokens: np.ndarray,
    attention_matrix: np.ndarray,
    epsilon: float = 1e-6
) -> np.ndarray:
    """
    Compute the gradient of information contribution with respect to splat position.
    
    This can be used to guide the direction of mitosis (splitting) to maximize
    information gain.
    
    Args:
        splat: The splat to evaluate
        tokens: Token embeddings [sequence_length, embedding_dim]
        attention_matrix: Current full attention matrix
        epsilon: Small perturbation amount for finite difference approximation
        
    Returns:
        Gradient vector with the same dimensions as splat position
    """
    try:
        dim = splat.position.shape[0]
        gradient = np.zeros(dim)
        
        # Store original position
        original_position = splat.position.copy()
        
        # Compute base information contribution
        base_attention = compute_splat_attention_map(splat, tokens)
        base_entropy = compute_attention_entropy(base_attention)
        
        # Compute gradient using finite differences
        for i in range(dim):
            # Create perturbation vector
            perturbation = np.zeros(dim)
            perturbation[i] = epsilon
            
            # Perturb position in positive direction
            splat.position = original_position + perturbation
            pos_attention = compute_splat_attention_map(splat, tokens)
            pos_entropy = compute_attention_entropy(pos_attention)
            
            # Perturb position in negative direction
            splat.position = original_position - perturbation
            neg_attention = compute_splat_attention_map(splat, tokens)
            neg_entropy = compute_attention_entropy(neg_attention)
            
            # Compute symmetric finite difference
            gradient[i] = (pos_entropy - neg_entropy) / (2 * epsilon)
            
            # Restore original position
            splat.position = original_position.copy()
        
        # Normalize gradient
        norm = np.linalg.norm(gradient)
        if norm > 1e-10:
            gradient = gradient / norm
        
        return gradient
    
    except Exception as e:
        logger.error(f"Error computing information gradient: {e}")
        # Return a random gradient as fallback
        dim = splat.position.shape[0]
        random_grad = np.random.randn(dim)
        return random_grad / (np.linalg.norm(random_grad) + 1e-10)

def compute_splat_attention_map(
    splat: Splat,
    tokens: np.ndarray
) -> np.ndarray:
    """
    Compute the attention map for a single splat across all token pairs.
    
    Args:
        splat: The splat to evaluate
        tokens: Token embeddings [sequence_length, embedding_dim]
        
    Returns:
        Attention matrix from this splat [sequence_length, sequence_length]
    """
    sequence_length = tokens.shape[0]
    attention_map = np.zeros((sequence_length, sequence_length))
    
    for i in range(sequence_length):
        for j in range(sequence_length):
            attention_map[i, j] = splat.compute_attention(tokens[i], tokens[j])
    
    return attention_map

def compute_level_information_flow(
    splat_registry: SplatRegistry,
    tokens: np.ndarray,
    from_level: str,
    to_level: str
) -> float:
    """
    Compute the information flow between hierarchical levels.
    
    This measures how much information from one level influences another level,
    which is useful for analyzing hierarchical relationships.
    
    Args:
        splat_registry: Registry containing all splats
        tokens: Token embeddings [sequence_length, embedding_dim]
        from_level: Source level name
        to_level: Target level name
        
    Returns:
        Information flow score (higher means stronger influence)
    """
    try:
        # Create attention computer
        attention_computer = create_attention_computer(
            hierarchy=splat_registry.hierarchy,
            efficient=True
        )
        
        # Create registries with only the specified levels
        from_registry = SplatRegistry(splat_registry.hierarchy)
        for splat in splat_registry.get_splats_at_level(from_level):
            from_registry.register(splat)
        
        to_registry = SplatRegistry(splat_registry.hierarchy)
        for splat in splat_registry.get_splats_at_level(to_level):
            to_registry.register(splat)
        
        # Compute attention for both levels
        from_attention = attention_computer.compute_attention(tokens, from_registry)
        to_attention = attention_computer.compute_attention(tokens, to_registry)
        
        # Compute mutual information between the attention patterns
        # This measures how much knowing one level's attention helps predict the other
        joint = np.stack((from_attention.flatten(), to_attention.flatten())).T
        joint_norm = joint / (np.sum(joint) + 1e-10)
        
        # Compute mutual information
        mi = compute_mutual_information(joint_norm)
        
        return mi
    
    except Exception as e:
        logger.error(f"Error computing level information flow: {e}")
        return 0.0  # Default to zero on error

class InformationMetricsTracker:
    """
    Tracks information-theoretic metrics for splats.
    
    This class maintains records of information contribution, entropy,
    and other information-theoretic metrics for all splats, which can
    be used to make more informed adaptation decisions.
    """
    
    def __init__(self):
        """Initialize the information metrics tracker."""
        self.splat_info_contributions = {}  # Maps splat ID to information contribution
        self.splat_entropies = {}  # Maps splat ID to attention entropy
        self.level_info_flows = {}  # Maps level pairs to information flow
        
    def compute_all_metrics(
        self,
        splat_registry: SplatRegistry,
        tokens: np.ndarray,
        attention_matrix: np.ndarray
    ) -> None:
        """
        Compute all information metrics for all splats.
        
        Args:
            splat_registry: Registry containing all splats
            tokens: Token embeddings
            attention_matrix: Current full attention matrix
        """
        # Compute information contribution for each splat
        for splat_id, splat in splat_registry.splats.items():
            try:
                # Compute information contribution
                info_contribution = compute_splat_information_contribution(
                    splat, tokens, attention_matrix, splat_registry
                )
                self.splat_info_contributions[splat_id] = info_contribution
                
                # Compute attention entropy
                entropy = compute_splat_attention_entropy(splat, tokens)
                self.splat_entropies[splat_id] = entropy
                
            except Exception as e:
                logger.error(f"Error computing metrics for splat {splat_id}: {e}")
        
        # Compute level information flows
        hierarchy_levels = splat_registry.hierarchy.levels
        for i, from_level in enumerate(hierarchy_levels):
            for j, to_level in enumerate(hierarchy_levels):
                if i != j:
                    flow_key = f"{from_level}_to_{to_level}"
                    flow = compute_level_information_flow(
                        splat_registry, tokens, from_level, to_level
                    )
                    self.level_info_flows[flow_key] = flow
    
    def get_splat_metrics(self, splat_id: str) -> Dict[str, float]:
        """
        Get all information metrics for a specific splat.
        
        Args:
            splat_id: ID of the splat
            
        Returns:
            Dictionary of metrics for the splat
        """
        return {
            "info_contribution": self.splat_info_contributions.get(splat_id, 0.0),
            "entropy": self.splat_entropies.get(splat_id, 0.0)
        }
    
    def get_top_info_contributors(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top-n splats with highest information contribution.
        
        Args:
            n: Number of splats to return
            
        Returns:
            List of (splat_id, info_contribution) tuples
        """
        return sorted(
            self.splat_info_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
    
    def get_top_entropy_splats(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top-n splats with highest attention entropy.
        
        Args:
            n: Number of splats to return
            
        Returns:
            List of (splat_id, entropy) tuples
        """
        return sorted(
            self.splat_entropies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
    
    def get_strongest_level_flows(self, n: int = 3) -> List[Tuple[str, float]]:
        """
        Get the strongest level-to-level information flows.
        
        Args:
            n: Number of flows to return
            
        Returns:
            List of (level_pair, flow_strength) tuples
        """
        return sorted(
            self.level_info_flows.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
    
    def reset(self) -> None:
        """Reset all stored metrics."""
        self.splat_info_contributions.clear()
        self.splat_entropies.clear()
        self.level_info_flows.clear()
