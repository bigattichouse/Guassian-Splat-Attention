"""
SplatRegistry class implementation for Hierarchical Splat Attention (HSA).

The SplatRegistry manages all splats in the system and provides functionality
to access and organize them by hierarchical level.
"""

from typing import Dict, Set, List, Optional, Tuple, Any, Union, Iterator
import numpy as np
from collections import defaultdict
import logging

from .splat import Splat
from .hierarchy import Hierarchy

# Configure logging
logger = logging.getLogger(__name__)


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
            
        Raises:
            ValueError: If embedding_dim is not positive
        """
        # Validate embedding dimension
        if embedding_dim <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {embedding_dim}")
            
        self.hierarchy = hierarchy
        self.embedding_dim = embedding_dim
        
        # Initialize data structures
        self.splats: Dict[str, Splat] = {}  # All splats by ID
        self.splats_by_level: Dict[str, Set[Splat]] = {
            level: set() for level in hierarchy.levels
        }
        
        # Stats for monitoring
        self.registered_count = 0
        self.unregistered_count = 0
        self.recovery_count = 0
    
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
        self.registered_count += 1
    
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
        
        # Store parent reference before modifying relationships
        parent = splat_obj.parent
        
        # First handle all children
        if splat_obj.children:
            children = list(splat_obj.children)  # Make a copy to avoid modification during iteration
            
            for child in children:
                # Update child's parent reference
                if parent is not None:
                    # Move child to grandparent
                    child.parent = parent
                    parent.children.add(child)
                else:
                    # If no grandparent, child becomes orphaned
                    child.parent = None
                    
                # Remove child from this splat's children
                splat_obj.children.remove(child)
        
        # Now handle parent relationship
        if parent is not None:
            try:
                parent.children.remove(splat_obj)
            except KeyError:
                logger.warning(f"Splat {splat_id} not found in parent's children set")
                self.recovery_count += 1
        
        # Remove splat from registry
        try:
            self.splats_by_level[splat_obj.level].remove(splat_obj)
        except KeyError:
            logger.warning(f"Splat {splat_id} not found in level {splat_obj.level}")
            self.recovery_count += 1
            
        del self.splats[splat_id]
        self.unregistered_count += 1
        
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
    
    def safe_get_splat(self, splat_id: str) -> Optional[Splat]:
        """Safely get a splat by ID without raising an exception.
        
        Args:
            splat_id: ID of the splat to retrieve
            
        Returns:
            Splat object or None if not found
        """
        return self.splats.get(splat_id)
    
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
    
    def iterate_splats(self, level: Optional[str] = None) -> Iterator[Splat]:
        """Iterate over splats, optionally filtered by level.
        
        Args:
            level: Hierarchical level to filter by (if None, iterates all splats)
            
        Yields:
            Splats in the registry
            
        Raises:
            ValueError: If level is not valid
        """
        if level is None:
            yield from self.splats.values()
        else:
            if not self.hierarchy.is_valid_level(level):
                raise ValueError(f"Level '{level}' is not valid in current hierarchy")
            
            yield from self.splats_by_level[level]
    
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
        
        # Store old splat's parent and children for reference
        old_parent = old_splat.parent
        old_children = set(old_splat.children)
        
        # Remove old splat first
        self.unregister(old_splat)
        
        # Establish parent-child relationships for new splats
        for splat in new_splats:
            if old_parent is not None:
                splat.parent = old_parent
                old_parent.children.add(splat)
        
        # Transfer children to the first new splat if there are any and new_splats is not empty
        if old_children and new_splats:
            primary_splat = new_splats[0]
            for child in old_children:
                # Important: Remove child from its current parent's children set
                # After unregister, the children would have been moved to old_parent
                if child.parent is not None:
                    try:
                        child.parent.children.remove(child)
                    except KeyError:
                        # Child might not be in parent's children set
                        logger.warning(f"Child {child.id} not found in parent's children set")
                
                # Update child's parent
                child.parent = primary_splat
                # Add child to new parent's children set
                primary_splat.children.add(child)
        
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
        
        # Reset counters
        self.registered_count = 0
        self.unregistered_count = 0
        self.recovery_count = 0
        
        # Check token dimensions if provided
        if tokens is not None and tokens.shape[1] != self.embedding_dim:
            logger.warning(
                f"Token embedding dimension ({tokens.shape[1]}) does not match " +
                f"registry embedding dimension ({self.embedding_dim}). Using random initialization."
            )
            tokens = None
        
        # Create all splats first without parents
        splats_by_level = {}
        for level_idx, level_name in enumerate(self.hierarchy.levels):
            num_splats = self.hierarchy.init_splats_per_level[level_idx]
            splats_by_level[level_name] = []
            
            for i in range(num_splats):
                try:
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
                    
                    # Create splat without parent for now
                    splat = Splat(
                        dim=self.embedding_dim,
                        position=position,
                        covariance=covariance,
                        amplitude=1.0,
                        level=level_name
                    )
                    
                    splats_by_level[level_name].append(splat)
                    
                except Exception as e:
                    logger.error(f"Failed to create splat: {e}")
                    self.recovery_count += 1
                    continue
        
        # Now establish parent-child relationships
        for level_idx, level_name in enumerate(self.hierarchy.levels):
            # Skip the highest level - no parent needed
            if level_idx == len(self.hierarchy.levels) - 1:
                continue
                
            parent_level = self.hierarchy.get_parent_level(level_name)
            if parent_level and parent_level in splats_by_level:
                parent_splats = splats_by_level[parent_level]
                if parent_splats:  # Only if we have parents available
                    for i, splat in enumerate(splats_by_level[level_name]):
                        parent = parent_splats[i % len(parent_splats)]
                        splat.parent = parent
                        parent.children.add(splat)
        
        # Finally register all splats
        for level_name in self.hierarchy.levels:
            for splat in splats_by_level[level_name]:
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
        
        # If already at the requested level, do nothing
        if splat.level == new_level:
            return
        
        # Handle inconsistency: splat might be in a different level set than its level attribute suggests
        try:
            # Remove from old level
            self.splats_by_level[splat.level].remove(splat)
        except KeyError:
            # Try to find it in another level set
            found = False
            for level, splats in self.splats_by_level.items():
                if splat in splats:
                    self.splats_by_level[level].remove(splat)
                    found = True
                    logger.warning(
                        f"Splat {splat.id} found in level {level} but has level attribute {splat.level}"
                    )
                    self.recovery_count += 1
                    break
            
            if not found:
                logger.warning(f"Splat {splat.id} not found in any level set")
                self.recovery_count += 1
        
        # Update level
        old_level = splat.level
        splat.level = new_level
        
        # Add to new level
        self.splats_by_level[new_level].add(splat)
        
        # Handle parent-child relationships if needed
        if splat.parent is not None:
            parent_level_idx = self.hierarchy.get_level_index(splat.parent.level)
            new_level_idx = self.hierarchy.get_level_index(new_level)
            
            # If parent level is not above new level, remove parent relationship
            if parent_level_idx <= new_level_idx:
                logger.info(
                    f"Removing parent relationship for splat {splat.id} as its new level " +
                    f"{new_level} is not below its parent's level {splat.parent.level}"
                )
                splat.parent.children.remove(splat)
                splat.parent = None
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the registry data structures.
        
        Returns:
            True if registry is consistent, False otherwise
        """
        # Check that all splats in the registry are in their respective level sets
        for splat_id, splat in self.splats.items():
            if splat not in self.splats_by_level[splat.level]:
                logger.warning(
                    f"Splat {splat_id} not found in its level set {splat.level}"
                )
                return False
        
        # Check that all splats in level sets are in the registry
        for level, splats in self.splats_by_level.items():
            for splat in splats:
                if splat.id not in self.splats:
                    logger.warning(
                        f"Splat {splat.id} found in level set {level} but not in registry"
                    )
                    return False
                
                # Check level attribute consistency
                if splat.level != level:
                    logger.warning(
                        f"Splat {splat.id} in level set {level} has level attribute {splat.level}"
                    )
                    return False
        
        # Check parent-child relationship consistency
        for splat in self.splats.values():
            # Check that children reference this splat as parent
            for child in splat.children:
                if child.parent != splat:
                    logger.warning(
                        f"Child {child.id} of splat {splat.id} has different parent {child.parent.id if child.parent else 'None'}"
                    )
                    return False
            
            # Check that parent includes this splat in its children
            if splat.parent is not None:
                if splat.parent.id not in self.splats:
                    logger.warning(
                        f"Parent {splat.parent.id} of splat {splat.id} not found in registry"
                    )
                    return False
                
                if splat not in splat.parent.children:
                    logger.warning(
                        f"Splat {splat.id} not found in its parent's children set"
                    )
                    return False
        
        return True
    
    def repair_integrity(self) -> int:
        """Repair inconsistencies in the registry data structures.
        
        Returns:
            Number of repairs made
        """
        repairs = 0
        
        # Ensure all levels defined in hierarchy exist in splats_by_level
        for level in self.hierarchy.levels:
            if level not in self.splats_by_level:
                self.splats_by_level[level] = set()
                logger.info(f"Created missing level set for {level}")
                repairs += 1
        
        # Repair splats missing from their level sets
        for splat_id, splat in self.splats.items():
            if splat.level not in self.splats_by_level:
                # Create the missing level set
                self.splats_by_level[splat.level] = set()
                logger.info(f"Created missing level set for {splat.level}")
                repairs += 1
                    
            if splat not in self.splats_by_level[splat.level]:
                self.splats_by_level[splat.level].add(splat)
                logger.info(f"Added splat {splat_id} to its level set {splat.level}")
                repairs += 1
        
        # Repair splats in wrong level sets
        for level, splats in list(self.splats_by_level.items()):
            for splat in list(splats):
                if splat.id not in self.splats:
                    # Splat in level set but not in registry - remove it
                    splats.remove(splat)
                    logger.info(
                        f"Removed splat {splat.id} from level set {level} (not in registry)"
                    )
                    repairs += 1
                elif splat.level != level:
                    # Splat in wrong level set - move it to correct one
                    splats.remove(splat)
                    self.splats_by_level[splat.level].add(splat)
                    logger.info(
                        f"Moved splat {splat.id} from level set {level} to {splat.level}"
                    )
                    repairs += 1
        
        # Repair parent-child relationships
        for splat in list(self.splats.values()):
            # Repair invalid parent references
            if splat.parent is not None and splat.parent.id not in self.splats:
                logger.info(
                    f"Removing invalid parent reference for splat {splat.id}"
                )
                splat.parent = None
                repairs += 1
            
            # Fix orphaned children
            orphaned = self.find_orphaned_children()
            for orphan in orphaned:
                if orphan.parent is None:
                    logger.info(f"Found orphaned child {orphan.id} - removing parent reference")
                    orphan.parent = None
                    repairs += 1
            
            # Repair missing child references in parent
            if splat.parent is not None and splat not in splat.parent.children:
                splat.parent.children.add(splat)
                logger.info(
                    f"Added splat {splat.id} to its parent's children set"
                )
                repairs += 1
            
            # Repair invalid child references
            for child in list(splat.children):
                if not hasattr(child, 'id') or not isinstance(child, Splat):
                    # Handle non-Splat objects in children set
                    splat.children.remove(child)
                    logger.info(
                        f"Removed non-Splat object from children of splat {splat.id}"
                    )
                    repairs += 1
                    continue
                    
                if child.id not in self.splats:
                    splat.children.remove(child)
                    logger.info(
                        f"Removed invalid child reference {child.id} from splat {splat.id}"
                    )
                    repairs += 1
                elif child.parent != splat:
                    # This is critical: Fix the inconsistency where a child is in parent's
                    # children set but doesn't reference this parent
                    child.parent = splat
                    logger.info(
                        f"Fixed parent reference for child {child.id} to splat {splat.id}"
                    )
                    repairs += 1
        
        return repairs
        
    def find_orphaned_children(self) -> List[Splat]:
        """Find splats that don't have parents but should according to hierarchy.
        
        Returns:
            List of orphaned splats
        """
        orphans = []
        
        # Check all levels except the highest level
        highest_level_idx = len(self.hierarchy.levels) - 1
        
        for splat in self.get_all_splats():
            level_idx = self.hierarchy.get_level_index(splat.level)
            
            # Skip the highest level - no parent expected
            if level_idx == highest_level_idx:
                continue
            
            # For all other levels, splats should have parents
            if splat.parent is None:
                orphans.append(splat)
            elif splat.parent.id not in self.splats:
                # Also check if the parent exists in the registry
                # This is an orphan with a reference to a non-existent parent
                orphans.append(splat)
            elif splat not in splat.parent.children:
                # Also check if the parent includes this splat in its children set
                # This is an orphan whose parent doesn't recognize it
                orphans.append(splat)
                    
        return orphans
        
    def find_empty_levels(self) -> List[str]:
        """Find hierarchical levels with no splats.
        
        Returns:
            List of empty level names
        """
        return [
            level for level, splats in self.splats_by_level.items()
            if not splats
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the registry state.
        
        Returns:
            Dictionary with summary information
        """
        level_counts = {
            level: len(splats) 
            for level, splats in self.splats_by_level.items()
        }
        
        empty_levels = self.find_empty_levels()
        orphaned_children = len(self.find_orphaned_children())
        is_consistent = self.verify_integrity()
        
        return {
            "total_splats": len(self.splats),
            "levels": level_counts,
            "empty_levels": empty_levels,
            "orphaned_children": orphaned_children,
            "registered_count": self.registered_count,
            "unregistered_count": self.unregistered_count,
            "recovery_count": self.recovery_count,
            "is_consistent": is_consistent
        }
    
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
