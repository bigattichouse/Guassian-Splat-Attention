# spatial_index_utils.py
"""
Utility functions for spatial indexing in Hierarchical Splat Attention (HSA).

This module provides helper functions for managing and optimizing spatial
indexes used in HSA.
"""

from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np

from .splat import Splat
from .registry import SplatRegistry


def compute_bounding_box(splats: List[Splat]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the bounding box containing all splats.
    
    Args:
        splats: List of splats
        
    Returns:
        Tuple of (min_coords, max_coords)
    """
    if not splats:
        return None, None
        
    # Extract positions
    positions = np.array([splat.position for splat in splats])
    
    # Compute min and max along each dimension
    min_coords = np.min(positions, axis=0)
    max_coords = np.max(positions, axis=0)
    
    return min_coords, max_coords


def compute_centroid(splats: List[Splat]) -> np.ndarray:
    """Compute the centroid of a set of splats.
    
    Args:
        splats: List of splats
        
    Returns:
        Centroid position
    """
    if not splats:
        return None
        
    # Extract positions
    positions = np.array([splat.position for splat in splats])
    
    # Compute mean position
    centroid = np.mean(positions, axis=0)
    
    return centroid


def compute_distance_matrix(splats: List[Splat]) -> np.ndarray:
    """Compute pairwise distances between splats.
    
    Args:
        splats: List of splats
        
    Returns:
        Distance matrix of shape [n_splats, n_splats]
    """
    if not splats:
        return np.array([])
        
    n_splats = len(splats)
    distances = np.zeros((n_splats, n_splats))
    
    # Compute all pairwise distances
    for i in range(n_splats):
        for j in range(i+1, n_splats):
            dist = np.linalg.norm(splats[i].position - splats[j].position)
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances


def find_closest_pairs(splats: List[Splat], k: int = 5) -> List[Tuple[int, int, float]]:
    """Find the k closest pairs of splats.
    
    Args:
        splats: List of splats
        k: Number of pairs to find
        
    Returns:
        List of (index_i, index_j, distance) tuples
    """
    if len(splats) < 2:
        return []
        
    # Compute distance matrix
    distances = compute_distance_matrix(splats)
    
    # Create list of all pairs
    pairs = []
    for i in range(len(splats)):
        for j in range(i+1, len(splats)):
            pairs.append((i, j, distances[i, j]))
    
    # Sort by distance
    pairs.sort(key=lambda x: x[2])
    
    # Return top k
    return pairs[:k]


def find_farthest_pairs(splats: List[Splat], k: int = 5) -> List[Tuple[int, int, float]]:
    """Find the k farthest pairs of splats.
    
    Args:
        splats: List of splats
        k: Number of pairs to find
        
    Returns:
        List of (index_i, index_j, distance) tuples
    """
    if len(splats) < 2:
        return []
        
    # Compute distance matrix
    distances = compute_distance_matrix(splats)
    
    # Create list of all pairs
    pairs = []
    for i in range(len(splats)):
        for j in range(i+1, len(splats)):
            pairs.append((i, j, distances[i, j]))
    
    # Sort by distance (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Return top k
    return pairs[:k]
