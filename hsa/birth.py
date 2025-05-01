"""
Birth operation implementation for Hierarchical Splat Attention (HSA).

This module provides functionality for creating new splats in the HSA structure,
particularly in regions with insufficient coverage.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import uuid
import logging

from .splat import Splat
from .registry import SplatRegistry

# Configure logging
logger = logging.getLogger(__name__)


def generate_birth_candidates(
    registry: SplatRegistry,
    level: str,
    position: Optional[np.ndarray] = None,
    tokens: Optional[np.ndarray] = None,
    num_candidates: int = 5
) -> List[Splat]:
    """Generate candidate splats for birth operation.
    
    Args:
        registry: SplatRegistry for context
        level: Hierarchical level for the new splat
        position: Optional target position (if None, inferred from tokens or random)
        tokens: Optional token embeddings to influence positioning
        num_candidates: Number of candidates to generate
        
    Returns:
        List of candidate splats
    """
    # Validate level
    if not registry.hierarchy.is_valid_level(level):
        logger.warning(f"Invalid level '{level}' for birth operation. Using lowest level.")
        level = registry.hierarchy.levels[0]
    
    dim = registry.embedding_dim
    candidates = []
    
    # Determine parent splat (if applicable)
    parent = None
    parent_level = registry.hierarchy.get_parent_level(level)
    if parent_level:
        parent_splats = list(registry.get_splats_at_level(parent_level))
        if parent_splats:
            # Select parent based on position or randomly
            if position is not None and len(parent_splats) > 1:
                # Find closest parent
                parent = min(
                    parent_splats,
                    key=lambda s: np.linalg.norm(s.position - position)
                )
            else:
                # Select random parent
                parent = parent_splats[np.random.randint(len(parent_splats))]
    
    # Generate positions
    positions = []
    
    if position is not None:
        # Use provided position as base
        base_position = position
    elif tokens is not None and tokens.shape[0] > 0:
        # Use token distribution to inform position
        # Simple approach: use mean of token embeddings
        base_position = np.mean(tokens, axis=0)
    else:
        # Generate random position
        base_position = np.random.normal(0, 1.0, dim)
    
    # Generate variations around the base position
    positions.append(base_position)  # Include base position
    
    for _ in range(num_candidates - 1):
        # Add random noise to base position
        noise = np.random.normal(0, 0.2, dim)
        positions.append(base_position + noise)
    
    # Generate candidates with varying parameters
    for i, pos in enumerate(positions):
        # Create covariance matrix (scaled identity for simplicity)
        # In production, adapt covariance based on surrounding tokens/splats
        scale = 1.0
        if parent is not None:
            # Make child splats smaller than parent
            scale = 0.5
        
        covariance = np.eye(dim) * scale
        
        # Create candidate splat
        splat = Splat(
            dim=dim,
            position=pos,
            covariance=covariance,
            amplitude=1.0,
            level=level,
            parent=parent,
            id=f"birth_candidate_{uuid.uuid4()}"
        )
        
        candidates.append(splat)
    
    return candidates


def perform_birth(
    registry: SplatRegistry,
    level: str,
    position: Optional[np.ndarray] = None,
    tokens: Optional[np.ndarray] = None,
    covariance: Optional[np.ndarray] = None,
    amplitude: float = 1.0,
    parent_id: Optional[str] = None
) -> Optional[Splat]:
    """Create and register a new splat.
    
    Args:
        registry: SplatRegistry to update
        level: Hierarchical level for the new splat
        position: Position in embedding space (if None, inferred from tokens or random)
        tokens: Optional token embeddings to influence positioning
        covariance: Covariance matrix (if None, uses default)
        amplitude: Attention strength factor
        parent_id: Optional ID of parent splat
        
    Returns:
        The new splat if successful, None if failed
    """
    try:
        dim = registry.embedding_dim
        
        # Determine parent splat if parent_id is provided
        parent = None
        if parent_id is not None:
            try:
                parent = registry.get_splat(parent_id)
            except ValueError:
                logger.warning(f"Parent splat {parent_id} not found")
        
        # Determine position
        if position is None:
            if tokens is not None and tokens.shape[0] > 0:
                # Use token distribution to inform position
                position = np.mean(tokens, axis=0)
            else:
                # Generate random position
                position = np.random.normal(0, 1.0, dim)
        
        # Determine covariance
        if covariance is None:
            # Create default covariance (identity matrix)
            covariance = np.eye(dim)
            
            # Scale based on level
            level_idx = registry.hierarchy.get_level_index(level)
            scale = 1.0 + level_idx * 0.5  # Scale factors: 1.0, 1.5, 2.0, 2.5...
            covariance *= scale
        
        # Create new splat
        new_splat = Splat(
            dim=dim,
            position=position,
            covariance=covariance,
            amplitude=amplitude,
            level=level,
            parent=parent
        )
        
        # Register new splat
        registry.register(new_splat)
        
        return new_splat
        
    except Exception as e:
        logger.error(f"Error during birth operation: {e}")
        return None


def identify_empty_regions(
    registry: SplatRegistry,
    tokens: Optional[np.ndarray] = None,
    coverage_threshold: float = 0.1,
    max_regions: int = 3
) -> List[np.ndarray]:
    """Identify regions in embedding space with insufficient splat coverage.
    
    Args:
        registry: SplatRegistry to analyze
        tokens: Optional token embeddings to find gaps in coverage
        coverage_threshold: Threshold below which a region is considered empty
        max_regions: Maximum number of regions to identify
        
    Returns:
        List of positions (centers of empty regions)
    """
    empty_regions = []
    dim = registry.embedding_dim
    
    if tokens is None or tokens.shape[0] == 0:
        # Without tokens, just generate some random positions
        # This is a placeholder - in production, use more sophisticated methods
        for _ in range(max_regions):
            position = np.random.normal(0, 1.0, dim)
            empty_regions.append(position)
            
        return empty_regions
    
    # With tokens, identify regions with low coverage
    # This is a simplified implementation
    
    # Get all splats
    all_splats = registry.get_all_splats()
    
    # If no splats, any region is empty
    if not all_splats:
        # Just use some token positions
        indices = np.random.choice(tokens.shape[0], min(max_regions, tokens.shape[0]), replace=False)
        for idx in indices:
            empty_regions.append(tokens[idx])
        return empty_regions
    
    # Compute coverage for each token
    coverages = []
    for token in tokens:
        # Compute maximum activation from any splat for this token
        max_coverage = 0.0
        for splat in all_splats:
            # Compute Mahalanobis distance to splat center
            delta = token - splat.position
            if hasattr(splat, 'covariance_inverse') and splat.covariance_inverse is not None:
                try:
                    mahalanobis = delta @ splat.covariance_inverse @ delta
                    coverage = np.exp(-0.5 * mahalanobis)  # Gaussian kernel
                    max_coverage = max(max_coverage, coverage)
                except Exception:
                    # Fallback to Euclidean distance
                    distance = np.linalg.norm(delta)
                    coverage = np.exp(-0.5 * distance**2)
                    max_coverage = max(max_coverage, coverage)
            else:
                # Fallback to Euclidean distance
                distance = np.linalg.norm(delta)
                coverage = np.exp(-0.5 * distance**2)
                max_coverage = max(max_coverage, coverage)
        
        coverages.append(max_coverage)
    
    # Convert to numpy array for easier processing
    coverages = np.array(coverages)
    
    # Find tokens with coverage below threshold
    low_coverage_indices = np.where(coverages < coverage_threshold)[0]
    
    # If no low coverage regions found, return random positions
    if len(low_coverage_indices) == 0:
        return empty_regions  # Already initialized as empty list
    
    # Select tokens with lowest coverage
    num_regions = min(max_regions, len(low_coverage_indices))
    selected_indices = low_coverage_indices[np.argsort(coverages[low_coverage_indices])[:num_regions]]
    
    # Use token positions as centers of empty regions
    for idx in selected_indices:
        empty_regions.append(tokens[idx])
    
    return empty_regions


def create_initial_splats(
    registry: SplatRegistry, 
    tokens: Optional[np.ndarray] = None
) -> int:
    """Create initial splats based on token distribution.
    
    Args:
        registry: SplatRegistry to populate
        tokens: Optional token embeddings to inform initialization
        
    Returns:
        Number of splats created
    """
    # Get initialization parameters from hierarchy
    hierarchy = registry.hierarchy
    levels = hierarchy.levels
    init_counts = hierarchy.init_splats_per_level
    
    total_created = 0
    dim = registry.embedding_dim
    
    # First, create splats at the highest level
    highest_level = levels[-1]
    highest_count = init_counts[-1]
    
    # Determine initial positions
    positions = []
    
    if tokens is not None and tokens.shape[0] > 0:
        # Use token-based initialization
        if tokens.shape[0] > highest_count:
            # Sample tokens
            indices = np.random.choice(tokens.shape[0], highest_count, replace=False)
            positions = [tokens[idx] for idx in indices]
        else:
            # Use all tokens and add random positions if needed
            positions = [tokens[idx] for idx in range(tokens.shape[0])]
            
            while len(positions) < highest_count:
                # Add random positions
                positions.append(np.random.normal(0, 1.0, dim))
    else:
        # Random initialization
        for _ in range(highest_count):
            positions.append(np.random.normal(0, 1.0, dim))
    
    # Create splats at highest level
    highest_splats = []
    highest_scale = 2.0  # Larger covariance at higher levels
    
    for pos in positions:
        covariance = np.eye(dim) * highest_scale
        
        splat = Splat(
            dim=dim,
            position=pos,
            covariance=covariance,
            amplitude=1.0,
            level=highest_level
        )
        
        registry.register(splat)
        highest_splats.append(splat)
        total_created += 1
    
    # Now create splats at lower levels with parent-child relationships
    for level_idx in range(len(levels) - 2, -1, -1):  # Iterate from second-highest to lowest
        level = levels[level_idx]
        count = init_counts[level_idx]
        parent_level = levels[level_idx + 1]
        parent_splats = list(registry.get_splats_at_level(parent_level))
        
        # Calculate splats per parent (approximately)
        splats_per_parent = max(1, count // len(parent_splats))
        
        # Scale for covariance - lower levels have smaller covariance
        scale = 1.0 / (level_idx + 1)
        
        # Create child splats for each parent
        splats_created = 0
        
        for parent in parent_splats:
            # Create specified number of children for this parent
            for _ in range(splats_per_parent):
                if splats_created >= count:
                    break
                
                # Generate position near parent
                offset = np.random.normal(0, 0.3, dim)
                position = parent.position + offset
                
                # Create covariance (smaller than parent)
                covariance = np.eye(dim) * scale
                
                # Create splat
                splat = Splat(
                    dim=dim,
                    position=position,
                    covariance=covariance,
                    amplitude=1.0,
                    level=level,
                    parent=parent
                )
                
                registry.register(splat)
                splats_created += 1
                total_created += 1
            
            if splats_created >= count:
                break
    
    return total_created
