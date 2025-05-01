"""
Merge operation implementation for Hierarchical Splat Attention (HSA).

This module provides functionality for merging similar splats in the HSA
structure to reduce redundancy and maintain a clean, efficient representation.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import uuid
import logging

from .splat import Splat
from .registry import SplatRegistry

# Configure logging
logger = logging.getLogger(__name__)


def generate_merge_candidates(
    splat_a: Splat,
    splat_b: Splat,
    num_candidates: int = 3
) -> List[Splat]:
    """Generate candidate splats for a merge operation.
    
    Args:
        splat_a: First splat to merge
        splat_b: Second splat to merge
        num_candidates: Number of candidates to generate
        
    Returns:
        List of candidate merged splats
    """
    # Validate input
    if splat_a.dim != splat_b.dim:
        logger.error(f"Cannot merge splats with different dimensions: {splat_a.dim} vs {splat_b.dim}")
        return []
    
    dim = splat_a.dim
    candidates = []
    
    # Simple merge: weighted average of parameters
    for i in range(num_candidates):
        # Vary the interpolation factor for different candidates
        t = (i / (num_candidates - 1)) if num_candidates > 1 else 0.5
        
        # Interpolate position
        position = (1 - t) * splat_a.position + t * splat_b.position
        
        # Interpolate covariance
        # This simple approach may not preserve positive-definiteness
        # In practice, use more sophisticated methods like geodesic interpolation
        covariance = (1 - t) * splat_a.covariance + t * splat_b.covariance
        
        # Ensure covariance is positive definite
        min_eigenvalue = np.min(np.linalg.eigvalsh(covariance))
        if min_eigenvalue <= 0:
            # Add to diagonal to ensure positive definiteness
            covariance += np.eye(dim) * (1e-5 - min_eigenvalue)
        
        # Interpolate amplitude
        amplitude = (1 - t) * splat_a.amplitude + t * splat_b.amplitude
        
        # Choose level (keep the higher level if different)
        if splat_a.level != splat_b.level:
            # Get level indices
            try:
                level_a_idx = get_level_index(splat_a.level)
                level_b_idx = get_level_index(splat_b.level)
                
                # Higher level has larger index in hierarchy
                level = splat_a.level if level_a_idx > level_b_idx else splat_b.level
            except:
                # Fallback to first splat's level
                level = splat_a.level
        else:
            level = splat_a.level
        
        # Choose parent (keep the parent of the higher level splat or None if same level)
        if splat_a.level != splat_b.level:
            parent = splat_a.parent if level == splat_a.level else splat_b.parent
        else:
            # Same level, use parent of first splat if they differ
            parent = splat_a.parent
        
        # Create candidate splat
        splat = Splat(
            dim=dim,
            position=position,
            covariance=covariance,
            amplitude=amplitude,
            level=level,
            parent=parent,
            id=f"merge_candidate_{uuid.uuid4()}"
        )
        
        candidates.append(splat)
    
    return candidates


def get_level_index(level: str) -> int:
    """Helper function to get hierarchy index for a level.
    
    This is a simplification - in production, get this from the hierarchy.
    
    Args:
        level: Level name
        
    Returns:
        Level index
    """
    # Common level names and their relative order
    level_order = {
        "token": 0,
        "subword": 0,
        "word": 1,
        "phrase": 2,
        "sentence": 3,
        "paragraph": 4,
        "document": 5
    }
    
    return level_order.get(level, 0)


def perform_merge(
    registry: SplatRegistry,
    splat_a_id: str,
    splat_b_id: str
) -> Optional[Splat]:
    """Merge two splats and update the registry.
    
    Args:
        registry: SplatRegistry to update
        splat_a_id: ID of first splat to merge
        splat_b_id: ID of second splat to merge
        
    Returns:
        The merged splat if successful, None if failed
    """
    try:
        # Get the splats
        splat_a = registry.get_splat(splat_a_id)
        splat_b = registry.get_splat(splat_b_id)
        
        # Generate merge candidates
        candidates = generate_merge_candidates(splat_a, splat_b)
        
        if not candidates:
            logger.warning("No merge candidates generated")
            return None
        
        # Select best candidate (just use the first one in this simple implementation)
        merged_splat = candidates[0]
        
        # Remove original splats
        registry.unregister(splat_a)
        registry.unregister(splat_b)
        
        # Register merged splat
        registry.register(merged_splat)
        
        # Transfer children from both parents to merged splat
        all_children = list(splat_a.children.union(splat_b.children))
        
        for child in all_children:
            # Skip if child is already assigned to merged splat
            if child.parent == merged_splat:
                continue
                
            # Update parent reference
            if child.parent:
                try:
                    child.parent.children.remove(child)
                except KeyError:
                    pass
                    
            child.parent = merged_splat
            merged_splat.children.add(child)
        
        return merged_splat
        
    except Exception as e:
        logger.error(f"Error during merge operation: {e}")
        return None


def find_merge_candidates(
    registry: SplatRegistry,
    similarity_threshold: float = 0.8,
    max_candidates: int = 5,
    same_level_only: bool = True
) -> List[Tuple[Splat, Splat, float]]:
    """Find pairs of splats that are candidates for merging.
    
    Args:
        registry: SplatRegistry to analyze
        similarity_threshold: Minimum similarity for merge candidates
        max_candidates: Maximum number of candidate pairs to return
        same_level_only: Whether to only consider splats at the same level
        
    Returns:
        List of (splat_a, splat_b, similarity) tuples
    """
    candidates = []
    
    # Get all splats
    all_splats = registry.get_all_splats()
    
    # Insufficient splats for merging
    if len(all_splats) < 2:
        return candidates
    
    # Compare all pairs of splats
    for i in range(len(all_splats)):
        splat_a = all_splats[i]
        
        for j in range(i + 1, len(all_splats)):
            splat_b = all_splats[j]
            
            # Skip if not at same level (if required)
            if same_level_only and splat_a.level != splat_b.level:
                continue
            
            # Calculate similarity
            similarity = calculate_similarity(splat_a, splat_b)
            
            # Add to candidates if above threshold
            if similarity >= similarity_threshold:
                candidates.append((splat_a, splat_b, similarity))
    
    # Sort by similarity (highest first)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Limit to max candidates
    return candidates[:max_candidates]


def calculate_similarity(splat_a: Splat, splat_b: Splat) -> float:
    """Calculate similarity between two splats.
    
    Args:
        splat_a: First splat
        splat_b: Second splat
        
    Returns:
        Similarity value between 0 and 1
    """
    # Validate that splats have the same dimension
    if splat_a.dim != splat_b.dim:
        return 0.0
    
    try:
        # Calculate position similarity (distance-based)
        pos_distance = np.linalg.norm(splat_a.position - splat_b.position)
        pos_similarity = np.exp(-pos_distance)
        
        # Calculate covariance similarity
        # Use Frobenius norm of difference
        cov_distance = np.linalg.norm(splat_a.covariance - splat_b.covariance, 'fro')
        cov_similarity = np.exp(-0.1 * cov_distance)
        
        # Calculate amplitude similarity
        amp_distance = abs(splat_a.amplitude - splat_b.amplitude)
        amp_similarity = np.exp(-amp_distance)
        
        # Level similarity (1.0 if same, 0.5 if different)
        level_similarity = 1.0 if splat_a.level == splat_b.level else 0.5
        
        # Combine similarities with weights
        weights = [0.5, 0.3, 0.1, 0.1]  # Position, covariance, amplitude, level
        similarities = [pos_similarity, cov_similarity, amp_similarity, level_similarity]
        
        combined_similarity = sum(w * s for w, s in zip(weights, similarities))
        
        return combined_similarity
        
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0


def calculate_merge_suitability(
    registry: SplatRegistry,
    level: Optional[str] = None
) -> float:
    """Calculate suitability for merge operations at a specific level.
    
    Higher values indicate more redundant splats that could benefit from merging.
    
    Args:
        registry: SplatRegistry to analyze
        level: Hierarchical level to analyze (if None, analyzes all levels)
        
    Returns:
        Merge suitability score (0-1 scale)
    """
    # Get splats to analyze
    if level is not None:
        splats = list(registry.get_splats_at_level(level))
    else:
        splats = registry.get_all_splats()
    
    # Not enough splats for meaningful analysis
    if len(splats) < 2:
        return 0.0
    
    try:
        # Calculate pairwise similarities
        similarities = []
        
        for i in range(len(splats)):
            for j in range(i + 1, len(splats)):
                similarity = calculate_similarity(splats[i], splats[j])
                similarities.append(similarity)
        
        # Calculate statistics
        if not similarities:
            return 0.0
            
        mean_similarity = sum(similarities) / len(similarities)
        
        # Find clusters of similar splats
        num_high_similarity = sum(1 for s in similarities if s > 0.8)
        high_similarity_ratio = num_high_similarity / len(similarities)
        
        # Combine metrics
        suitability = 0.5 * mean_similarity + 0.5 * high_similarity_ratio
        
        return suitability
        
    except Exception as e:
        logger.error(f"Error calculating merge suitability: {e}")
        return 0.0


def get_best_merge_strategy(
    registry: SplatRegistry
) -> Dict[str, float]:
    """Determine the best merge strategy based on level suitability.
    
    Args:
        registry: SplatRegistry to analyze
        
    Returns:
        Dictionary mapping level names to merge priority scores (0-1)
    """
    strategies = {}
    
    # Analyze each level
    for level in registry.hierarchy.levels:
        # Skip levels with too few splats
        if registry.count_splats(level) < 3:
            strategies[level] = 0.0
            continue
            
        # Calculate merge suitability
        suitability = calculate_merge_suitability(registry, level)
        strategies[level] = suitability
    
    return strategies
