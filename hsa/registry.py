"""
SplatRegistry class implementation for Hierarchical Splat Attention (HSA).

The SplatRegistry manages all splats in the system and provides functionality
to access and organize them by hierarchical level.
"""

from typing import Dict, Set, List, Optional, Tuple, Any, Union
import numpy as np
from collections import defaultdict

from .splat import Splat
from .hierarchy import Hierarchy


class SplatRegistry:
    """
    The SplatRegistry manages all splats in the system and provides functionality
    to access and organize them by hierarchical level.
    """
    
    def __init__(self, hierarchy: Hierarchy, embedding_dim: int):
        """Initialize a new SplatRegistry.
        
        Args:
            hierarchy: Hierarchy structure for organizing splats
            embedding_dim: Dimensionality of the embedding space
        """
        self.hierarchy = hierarchy
        self.embedding_dim = embedding_dim
        
        # Initialize data structures
        self.splats: Dict[str, Splat] = {}  # All splats by ID
        self.splats_by_level: Dict[str, Set[Splat]] = {
            level: set() for level in hierarchy.levels
        }
    
    def register(self, splat: Splat) -> None:
        """Register a splat in the registry.
        
        Args:
            splat: Splat to register
            
        Raises:
            ValueError: If splat with same ID already exists or level is invalid
        """
        # Validate splat
        if splat.id in self.splats:
            raise ValueError(f"Splat with ID {splat.id} already exists in registry")
        
        if not self.hierarchy.is_valid_level(splat.level):
            raise ValueError(f"Level '{splat.level}' is not valid in current hierarchy")
        
        if splat.dim != self.embedding_dim:
            raise ValueError(
                f"Splat dimension ({splat.dim}) does not match registry " +
                f"embedding dimension ({self.embedding_dim})"
            )
        
        # Add splat to registry
        self.splats[splat.id] = splat
        self.splats_by_level[splat.level].add(splat)
    
    def unregister(self, splat: Union[Splat, str]) -> None:
        """Unregister a splat from the registry.
        
        Args:
            splat: Splat object or splat ID to unregister
            
        Raises:
            ValueError: If splat is not found in registry
        """
        # Get splat ID
        splat_id = splat if isinstance(splat, str) else splat.id
        
        if splat_id not in self.splats:
            raise ValueError(f"Splat with ID {splat_id} not found in registry")
        
        splat_obj = self.splats[splat_id]
        
        # Remove parent-child relationships
        if splat_obj.parent is not None:
            splat_obj.parent.children.remove(splat_obj)
        
        # Move children to parent if any
        if splat_obj.parent is not None and splat_obj.children:
            for child in list(splat_obj.children):
                child.parent = splat_obj.parent
                splat_obj.parent.children.add(child)
                splat_obj.children.remove(child)
        
        # Remove splat from registry
        self.splats_by_level[splat_obj.level].remove(splat_obj)
        del self.splats[splat_id]
    
    def get_splat(self, splat_id: str) -> Splat:
        """Get a splat by ID.
        
        Args:
            splat_id: ID of the splat to retrieve
            
        Returns:
            Splat object
            
        Raises:
            ValueError: If splat is not found
        """
        if splat_id not in self.splats:
            raise ValueError(f"Splat with ID {splat_id} not found in registry")
        
        return self.splats[splat_id]
    
    def get_splats_at_level(self, level: str) -> Set[Splat]:
        """Get all splats at a specific hierarchical level.
        
        Args:
            level: Hierarchical level name
            
        Returns:
            Set of splats at the specified level
            
        Raises:
            ValueError: If level is not valid
        """
        if not self.hierarchy.is_valid_level(level):
            raise ValueError(f"Level '{level}' is not valid in current hierarchy")
        
        return self.splats_by_level[level].copy()
    
    def get_all_splats(self) -> List[Splat]:
        """Get all splats in the registry.
        
        Returns:
            List of all splats
        """
        return list(self.splats.values())
    
    def count_splats(self, level: Optional[str] = None) -> int:
        """Count the number of splats, optionally at a specific level.
        
        Args:
            level: Hierarchical level to count (if None, counts all splats)
            
        Returns:
            Number of splats
            
        Raises:
            ValueError: If level is not valid
        """
        if level is None:
            return len(self.splats)
        
        if not self.hierarchy.is_valid_level(level):
            raise ValueError(f"Level '{level}' is not valid in current hierarchy")
        
        return len(self.splats_by_level[level])
    
    def replace_splat(self, old_splat: Splat, new_splats: List[Splat]) -> None:
        """Replace a splat with one or more new splats.
        
        This is used during adaptation operations like mitosis (splitting).
        
        Args:
            old_splat: Splat to replace
            new_splats: List of new splats to add
            
        Raises:
            ValueError: If old splat is not in registry or new splats have invalid level
        """
        if old_splat.id not in self.splats:
            raise ValueError(f"Splat with ID {old_splat.id} not found in registry")
        
        # Check if we're just removing the splat without replacements
        if not new_splats:
            self.unregister(old_splat)
            return
        
        # Validate all new splats before making changes
        for splat in new_splats:
            if not self.hierarchy.is_valid_level(splat.level):
                raise ValueError(f"Level '{splat.level}' is not valid in current hierarchy")
            
            if splat.dim != self.embedding_dim:
                raise ValueError(
                    f"Splat dimension ({splat.dim}) does not match registry " +
                    f"embedding dimension ({self.embedding_dim})"
                )
        
        # Establish parent-child relationships
        for splat in new_splats:
            if old_splat.parent is not None:
                splat.parent = old_splat.parent
                old_splat.parent.children.add(splat)
        
        # Transfer children to the first new splat if there are any
        if old_splat.children:
            primary_splat = new_splats[0]
            for child in list(old_splat.children):
                child.parent = primary_splat
                primary_splat.children.add(child)
        
        # Remove old splat first
        self.unregister(old_splat)
        
        # Add new splats
        for splat in new_splats:
            self.register(splat)
    
    def initialize_splats(self, tokens: Optional[np.ndarray] = None) -> None:
        """Initialize splats according to hierarchy configuration.
        
        Can use token embeddings if provided to distribute splats in the embedding space.
        
        Args:
            tokens: Optional token embeddings to use for initialization
        """
        # Clear existing splats
        self.splats.clear()
        for level_set in self.splats_by_level.values():
            level_set.clear()
        
        for level_idx, level_name in enumerate(self.hierarchy.levels):
            num_splats = self.hierarchy.init_splats_per_level[level_idx]
            
            # Get parent level splats if any
            parent_level = self.hierarchy.get_parent_level(level_name)
            parent_splats = []
            if parent_level:
                parent_splats = list(self.splats_by_level[parent_level])
            
            for i in range(num_splats):
                # Create initial position - either random or based on tokens
                if tokens is not None and tokens.shape[0] > 0:
                    # Initialize from tokens - this is a simple strategy, can be improved
                    idx = i % tokens.shape[0]
                    position = tokens[idx].copy()
                    
                    # Add some noise for diversity
                    position += np.random.normal(0, 0.1, self.embedding_dim)
                else:
                    # Random initialization
                    position = np.random.normal(0, 1.0, self.embedding_dim)
                
                # Create initial covariance - identity matrix scaled by level
                # Higher levels have larger covariance (broader attention)
                scale = 1.0 + level_idx * 0.5  # Scale factors: 1.0, 1.5, 2.0, 2.5...
                covariance = np.eye(self.embedding_dim) * scale
                
                # Assign parent if available
                parent = None
                if parent_splats:
                    parent = parent_splats[i % len(parent_splats)]
                
                # Create and register splat
                splat = Splat(
                    dim=self.embedding_dim,
                    position=position,
                    covariance=covariance,
                    amplitude=1.0,
                    level=level_name,
                    parent=parent
                )
                
                self.register(splat)
    
    def get_active_splats(self, threshold: float = 0.01) -> List[Splat]:
        """Get splats with average activation above threshold.
        
        Args:
            threshold: Activation threshold
            
        Returns:
            List of active splats
        """
        return [
            splat for splat in self.splats.values()
            if splat.get_average_activation() > threshold
        ]
    
    def change_splat_level(self, splat: Splat, new_level: str) -> None:
        """Change the hierarchical level of a splat.
        
        Args:
            splat: Splat to modify
            new_level: New hierarchical level
            
        Raises:
            ValueError: If splat is not in registry or new level is invalid
        """
        if splat.id not in self.splats:
            raise ValueError(f"Splat with ID {splat.id} not found in registry")
        
        if not self.hierarchy.is_valid_level(new_level):
            raise ValueError(f"Level '{new_level}' is not valid in current hierarchy")
        
        # Remove from old level
        self.splats_by_level[splat.level].remove(splat)
        
        # Update level
        old_level = splat.level
        splat.level = new_level
        
        # Add to new level
        self.splats_by_level[new_level].add(splat)
        
        # Update parent-child relationships if needed
        if splat.parent is not None and not self.hierarchy.is_valid_level(splat.parent.level):
            splat.parent.children.remove(splat)
            splat.parent = None
    
    def __repr__(self) -> str:
        """String representation of the registry.
        
        Returns:
            String representation
        """
        level_counts = {
            level: len(splats) 
            for level, splats in self.splats_by_level.items()
        }
        return (
            f"SplatRegistry(total_splats={len(self.splats)}, " +
            f"embedding_dim={self.embedding_dim}, " +
            f"level_counts={level_counts})"
        )
