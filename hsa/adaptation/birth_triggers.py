"""
Birth triggers module for Hierarchical Splat Attention (HSA).

This module implements the detection logic for when new splats should be created:
- Functions to identify regions needing new splats
- Logic for determining when a level needs more splats
- Utilities for finding optimal positions for new splats
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core data structures
from hsa.data_structures import Splat, SplatRegistry

def identify_empty_regions(
    tokens: np.ndarray,
    splat_registry: SplatRegistry,
    min_distance_threshold: float = 1.8,  # Reduced from 2.0
    max_regions: int = 3,
    min_tokens_per_region: int = 3  # New: minimum tokens to consider a region
) -> List[np.ndarray]:
    """
    Identify regions in the embedding space that are far from existing splats.
    
    Args:
        tokens: Token embeddings [sequence_length, embedding_dim]
        splat_registry: Registry containing all splats
        min_distance_threshold: Minimum distance to consider a region empty
        max_regions: Maximum number of empty regions to identify
        min_tokens_per_region: Minimum tokens needed to form a valid region
        
    Returns:
        List of positions for potential new splats
    """
    try:
        # Subsample tokens for efficiency if there are too many
        if tokens.shape[0] > 200:
            subsample_size = min(200, tokens.shape[0])
            indices = np.random.choice(tokens.shape[0], size=subsample_size, replace=False)
            sampled_tokens = tokens[indices]
        else:
            sampled_tokens = tokens
        
        # Get all splat positions
        all_splats = list(splat_registry.splats.values())
        if not all_splats:
            # If no splats, suggest token centers directly
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=max_regions, random_state=42)
            kmeans.fit(sampled_tokens)
            return [center for center in kmeans.cluster_centers_]
            
        # Calculate minimum distance from each token to any splat
        min_distances = np.ones(len(sampled_tokens)) * float('inf')
        
        for splat in all_splats:
            # Extract splat parameters
            pos = splat.position
            cov_inv = splat.covariance_inverse
            
            # Calculate distances from tokens to this splat
            diffs = sampled_tokens - pos
            try:
                distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
            except:
                # Fallback to Euclidean distance if Mahalanobis fails
                distances = np.sqrt(np.sum(diffs**2, axis=1))
            
            # Update minimum distances
            min_distances = np.minimum(min_distances, distances)
        
        # Find tokens that are far from any existing splat
        # This is more aggressive with the lowered threshold
        far_indices = np.where(min_distances > min_distance_threshold)[0]
        
        if len(far_indices) < min_tokens_per_region:  # Not enough far tokens
            # Fall back to the tokens furthest from any splat
            if len(sampled_tokens) > 0:
                # Take the most distant tokens as fallback
                sorted_indices = np.argsort(-min_distances)  # Descending order
                top_distant = sorted_indices[:max(min_tokens_per_region, 5)]
                
                # Create a simple mean position from the most distant tokens
                mean_position = np.mean(sampled_tokens[top_distant], axis=0)
                return [mean_position]
            return []
        
        far_tokens = sampled_tokens[far_indices]
        
        # Cluster the far tokens to identify distinct empty regions
        from sklearn.cluster import KMeans
        n_clusters = min(max_regions, len(far_indices) // min_tokens_per_region + 1)
        if n_clusters < 1:
            n_clusters = 1
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3)
        kmeans.fit(far_tokens)
        
        # Return cluster centers as potential birth locations
        return [center for center in kmeans.cluster_centers_]
        
    except Exception as e:
        logger.error(f"Error in identify_empty_regions: {e}")
        # Even on exception, try to return something reasonable
        if 'sampled_tokens' in locals() and len(sampled_tokens) > 0:
            # Return the mean of all tokens as a fallback
            return [np.mean(sampled_tokens, axis=0)]
        return []
        far_tokens = sampled_tokens[far_indices]
        
        # Cluster the far tokens to identify distinct empty regions
        from sklearn.cluster import KMeans
        n_clusters = min(max_regions, len(far_indices) // 5)  # Ensure enough tokens per cluster
        if n_clusters < 1:
            n_clusters = 1
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
        kmeans.fit(far_tokens)
        
        # Return cluster centers as potential birth locations
        return [center for center in kmeans.cluster_centers_]
        
    except Exception as e:
        logger.error(f"Error in identify_empty_regions: {e}")
        return []

def should_perform_birth(
    tokens: np.ndarray,
    splat_registry: SplatRegistry,
    level: str,
    min_tokens_per_birth: int = 5,  # Reduced from 10
    min_distance_threshold: float = 1.8,  # Reduced from 2.0 to detect more empty regions
    birth_level_threshold: float = 0.7  # Add parameter: Birth when below 70% of initial
) -> bool:
    """
    Determine if new splats should be created at a given level based on token coverage.
    
    Args:
        tokens: Token embeddings [sequence_length, embedding_dim]
        splat_registry: Registry containing all splats
        level: The hierarchy level to check
        min_tokens_per_birth: Minimum number of tokens needed to justify a birth
        min_distance_threshold: Minimum distance to consider a token uncovered
        birth_level_threshold: Ratio of current to initial splats below which birth is triggered
        
    Returns:
        True if birth should occur, False otherwise
    """
    # Get all splats at this level
    level_splats = list(splat_registry.get_splats_at_level(level))
    
    # If no splats at this level, definitely need birth
    if not level_splats:
        return True
    
    # Check splat density
    init_count = splat_registry.hierarchy.get_init_splats_count(level)
    current_count = len(level_splats)
    
    # If fewer than threshold of the initial count, consider birth
    # This is more aggressive than the previous 0.5 (50%) threshold
    if current_count < init_count * birth_level_threshold:
        # Find empty regions
        empty_regions = identify_empty_regions(
            tokens,
            splat_registry,
            min_distance_threshold=min_distance_threshold,
            max_regions=2  # Check up to 2 regions
        )
        
        # If found empty regions, suggest birth
        return len(empty_regions) > 0
    
    # Additional check: look for significant uncovered regions anyway
    covered_tokens = np.zeros(len(tokens), dtype=bool)
    
    # Check coverage for each splat
    for splat in level_splats:
        # Extract splat parameters
        pos = splat.position
        cov_inv = splat.covariance_inverse
        
        # Calculate distance for all tokens
        diffs = tokens - pos
        try:
            distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
        except:
            # Fallback to Euclidean distance if Mahalanobis fails
            distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Mark tokens as covered if within 3 standard deviations
        close_tokens = distances < 3.0
        covered_tokens = covered_tokens | close_tokens
    
    # Calculate percentage of uncovered tokens
    uncovered_ratio = 1.0 - np.mean(covered_tokens)
    
    # If more than 30% of tokens are uncovered, consider birth
    if uncovered_ratio > 0.3:
        # Verify with empty regions check
        empty_regions = identify_empty_regions(
            tokens,
            splat_registry,
            min_distance_threshold=min_distance_threshold,
            max_regions=1  # Just need to find one
        )
        return len(empty_regions) > 0
    
    return False
def find_optimal_birth_positions(
    tokens: np.ndarray,
    splat_registry: SplatRegistry,
    level: str,
    min_distance_threshold: float = 2.0,
    desired_count: Optional[int] = None
) -> List[np.ndarray]:
    """
    Find optimal positions for new splats.
    
    Args:
        tokens: Token embeddings [sequence_length, embedding_dim]
        splat_registry: Registry containing all splats
        level: The hierarchy level to check
        min_distance_threshold: Minimum distance to consider a token uncovered
        desired_count: Optional desired number of positions to find
        
    Returns:
        List of positions for potential new splats
    """
    # If desired_count not specified, use a default based on level
    if desired_count is None:
        init_count = splat_registry.hierarchy.get_init_splats_count(level)
        current_count = len(splat_registry.get_splats_at_level(level))
        desired_count = max(1, init_count - current_count)
    
    # Find empty regions
    max_regions = min(desired_count, 5)  # Limit to 5 regions for efficiency
    return identify_empty_regions(
        tokens,
        splat_registry,
        min_distance_threshold=min_distance_threshold,
        max_regions=max_regions
    )

def analyze_token_coverage(
    tokens: np.ndarray,
    splat_registry: SplatRegistry
) -> Dict[str, float]:
    """
    Analyze token coverage across hierarchy levels.
    
    Args:
        tokens: Token embeddings [sequence_length, embedding_dim]
        splat_registry: Registry containing all splats
        
    Returns:
        Dictionary mapping level names to coverage percentages
    """
    coverage = {}
    
    for level in splat_registry.hierarchy.levels:
        # Get splats at this level
        level_splats = list(splat_registry.get_splats_at_level(level))
        
        if not level_splats:
            coverage[level] = 0.0
            continue
        
        # Track covered tokens
        covered_tokens = np.zeros(len(tokens), dtype=bool)
        
        # Check coverage for each splat
        for splat in level_splats:
            pos = splat.position
            cov_inv = splat.covariance_inverse
            
            # Calculate distance for all tokens
            diffs = tokens - pos
            distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))
            
            # Mark tokens covered by this splat
            close_tokens = distances < 3.0  # Using 3.0 standard deviations
            covered_tokens = covered_tokens | close_tokens
        
        # Calculate coverage percentage
        coverage[level] = np.mean(covered_tokens)
    
    return coverage
