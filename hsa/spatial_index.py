"""
Tree-based spatial indexing for Hierarchical Splat Attention (HSA).

This module provides the primary tree-based spatial indexing capabilities to optimize 
attention computation by efficiently finding relevant splats in embedding space.
"""

from typing import List, Dict, Set, Tuple, Optional, Any, Union
import numpy as np
import logging
from queue import PriorityQueue
import heapq

from .splat import Splat
from .spatial_index_node import _Node

# Configure logging
logger = logging.getLogger(__name__)


class SpatialIndex:
    """
    Tree-based spatial index for efficient queries in embedding space.
    
    This class enables efficient queries such as finding the nearest splats to a point,
    range queries, and other spatial operations to optimize attention computation.
    """
    
    def __init__(self, dim: int, max_leaf_size: int = 10, max_depth: int = 10):
        """Initialize a spatial index.
        
        Args:
            dim: Dimensionality of the embedding space
            max_leaf_size: Maximum number of items in a leaf node
            max_depth: Maximum depth of the tree
        """
        self.dim = dim
        self.max_leaf_size = max_leaf_size
        self.max_depth = max_depth
        
        # Root node of the tree
        self.root = None
        
        # Map from splat ID to node containing the splat
        self.splat_to_node = {}
        
        # Statistics
        self.num_nodes = 0
        self.num_splats = 0
        self.max_current_depth = 0
        self.rebuild_count = 0
    
    def insert(self, splat: Splat) -> None:
        """Insert a splat into the index.
        
        Args:
            splat: Splat to insert
            
        Raises:
            ValueError: If splat dimension doesn't match index dimension
        """
        if splat.dim != self.dim:
            raise ValueError(
                f"Splat dimension ({splat.dim}) does not match index dimension ({self.dim})"
            )
        
        # If root doesn't exist, create it
        if self.root is None:
            self.root = _Node(
                dim=self.dim,
                depth=0,
                max_leaf_size=self.max_leaf_size,
                max_depth=self.max_depth
            )
        
        # Insert the splat
        node = self.root.insert(splat)
        
        # Update mapping
        self.splat_to_node[splat.id] = node
        
        # Update statistics
        self.num_splats += 1
        self.num_nodes = self._count_nodes(self.root)
        self.max_current_depth = self._max_depth(self.root)
        
        # Check if we need to rebuild
        if self.max_current_depth > self.max_depth:
            self._rebuild()
    
    def remove(self, splat_id: str) -> bool:
        """Remove a splat from the index.
        
        Args:
            splat_id: ID of the splat to remove
            
        Returns:
            True if splat was removed, False if not found
        """
        if splat_id not in self.splat_to_node:
            return False
        
        # Get node containing the splat
        node = self.splat_to_node[splat_id]
        
        # Remove splat from node
        if node.remove(splat_id):
            # Remove from mapping
            del self.splat_to_node[splat_id]
            
            # Update statistics
            self.num_splats -= 1
            self.num_nodes = self._count_nodes(self.root)
            
            # Check if we need to reorganize
            if self.num_splats > 0 and len(self.root.get_all_splats()) < self.max_leaf_size:
                # Tree is very sparse, rebuild
                self._rebuild()
                
            return True
        
        return False
    
    def update(self, splat: Splat) -> None:
        """Update a splat's position in the index.
        
        Args:
            splat: Splat with updated position
            
        Raises:
            ValueError: If splat is not in the index
        """
        if splat.id not in self.splat_to_node:
            raise ValueError(f"Splat {splat.id} not found in index")
        
        # Remove and reinsert
        self.remove(splat.id)
        self.insert(splat)
    
    def find_nearest(self, position: np.ndarray, k: int = 1) -> List[Tuple[Splat, float]]:
        """Find the k nearest splats to a position.
        
        Args:
            position: Query position in embedding space
            k: Number of nearest neighbors to find
            
        Returns:
            List of (splat, distance) tuples sorted by distance
            
        Raises:
            ValueError: If position dimension doesn't match index dimension
        """
        # Validate position dimension first, before any early returns
        if len(position) != self.dim:
            raise ValueError(
                f"Position dimension {len(position)} does not match index dimension {self.dim}"
            )
            
        if self.root is None or self.num_splats == 0:
            return []
        
        # Limit k to number of splats
        k = min(k, self.num_splats)
        
        # Get all splats and calculate exact distances
        all_splats = self.get_all_splats()
        distances = []
        
        for splat in all_splats:
            # Calculate Euclidean distance
            dist = np.linalg.norm(splat.position - position)
            distances.append((splat, dist))
        
        # Sort by distance (nearest first)
        # Use a stable sort for consistent results when distances are equal
        distances.sort(key=lambda x: (x[1], x[0].id))
        
        # Return top k
        return distances[:k]
    
    def find_nearest_efficient(self, position: np.ndarray, k: int = 1) -> List[Tuple[Splat, float]]:
        """Find the k nearest splats to a position using efficient tree traversal.
        
        This method is optimized for larger datasets where the tree structure
        provides significant acceleration.
        
        Args:
            position: Query position in embedding space
            k: Number of nearest neighbors to find
            
        Returns:
            List of (splat, distance) tuples sorted by distance
            
        Raises:
            ValueError: If position dimension doesn't match index dimension
        """
        # Validate position dimension first, before any early returns
        if len(position) != self.dim:
            raise ValueError(
                f"Position dimension {len(position)} does not match index dimension {self.dim}"
            )
            
        if self.root is None or self.num_splats == 0:
            return []
        
        # Limit k to number of splats
        k = min(k, self.num_splats)
        
        # Use priority queue for nearest neighbors
        # Use negative distance so smallest distance has highest priority
        nearest = PriorityQueue()
        
        # Search the tree
        self.root.find_nearest(position, k, nearest)
        
        # Extract results
        results = []
        while not nearest.empty():
            neg_dist, splat_id, splat = nearest.get()
            results.append((splat, -neg_dist))  # Convert back to positive distance
        
        # Return in order of increasing distance (nearest first)
        results.reverse()
        
        return results
    
    def range_query(self, center: np.ndarray, radius: float) -> List[Splat]:
        """Find all splats within a given radius of a center point.
        
        Args:
            center: Center position in embedding space
            radius: Search radius
            
        Returns:
            List of splats within the radius
            
        Raises:
            ValueError: If center dimension doesn't match index dimension
        """
        # Validate position dimension first, before any early returns
        if len(center) != self.dim:
            raise ValueError(
                f"Center dimension {len(center)} does not match index dimension {self.dim}"
            )
            
        if self.root is None or self.num_splats == 0:
            return []
        
        # Search the tree
        results = []
        self.root.range_query(center, radius, results)
        
        return results
    
    def find_by_level(self, level: str, position: np.ndarray, k: int = 5) -> List[Tuple[Splat, float]]:
        """Find the k nearest splats at a specific level.
        
        Args:
            level: Hierarchical level to filter by
            position: Query position in embedding space
            k: Number of nearest neighbors to find
            
        Returns:
            List of (splat, distance) tuples sorted by distance
            
        Raises:
            ValueError: If position dimension doesn't match index dimension
        """
        # Validate position dimension first
        if len(position) != self.dim:
            raise ValueError(
                f"Position dimension {len(position)} does not match index dimension {self.dim}"
            )
            
        if self.root is None or self.num_splats == 0:
            return []
        
        # Find all splats at this level
        splats_at_level = []
        self.root.filter_by_level(level, splats_at_level)
        
        if not splats_at_level:
            return []
        
        # Sort by distance to position
        distances = []
        for splat in splats_at_level:
            dist = np.linalg.norm(splat.position - position)
            distances.append((splat, dist))
        
        # Sort by distance (nearest first), then by splat ID for determinism
        distances.sort(key=lambda x: (x[1], x[0].id))
        
        # Limit to k
        return distances[:k]
    
    def get_all_splats(self) -> List[Splat]:
        """Get all splats in the index.
        
        Returns:
            List of all splats
        """
        if self.root is None:
            return []
        
        # Use a set to avoid duplicates
        splat_set = set()
        splats = self.root.get_all_splats()
        for splat in splats:
            splat_set.add(splat)
        
        return list(splat_set)
    
    def _rebuild(self) -> None:
        """Rebuild the tree to balance it."""
        # Get all splats
        all_splats = self.get_all_splats()
        
        # Clear the tree
        self.root = None
        self.splat_to_node = {}
        self.num_nodes = 0
        self.max_current_depth = 0
        
        # Save splat count before rebuilding
        temp_num_splats = self.num_splats
        
        # Reset rebuild count before incrementing
        self.rebuild_count = 0
        
        # Increment rebuild count
        self.rebuild_count += 1
        
        # Reset splat count to avoid double counting
        self.num_splats = 0
        
        # Create a new root
        self.root = _Node(
            dim=self.dim,
            depth=0,
            max_leaf_size=self.max_leaf_size,
            max_depth=self.max_depth
        )
        
        # Insert all splats
        for splat in all_splats:
            node = self.root.insert(splat)
            self.splat_to_node[splat.id] = node
        
        # Update statistics
        self.num_splats = len(all_splats)
        self.num_nodes = self._count_nodes(self.root)
        self.max_current_depth = self._max_depth(self.root)
    
    def _count_nodes(self, node) -> int:
        """Recursively count the number of nodes in the tree.
        
        Args:
            node: Node to start counting from
            
        Returns:
            Number of nodes
        """
        if node is None:
            return 0
            
        count = 1  # Count this node
        
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                count += self._count_nodes(child)
        
        return count
    
    def _max_depth(self, node) -> int:
        """Find the maximum depth of the tree.
        
        Args:
            node: Node to start from
            
        Returns:
            Maximum depth
        """
        if node is None:
            return 0
            
        if hasattr(node, 'children') and node.children:
            return 1 + max(self._max_depth(child) for child in node.children)
        
        return 1  # Leaf node
    
    def __repr__(self) -> str:
        """String representation of the index.
        
        Returns:
            String representation
        """
        return (
            f"SpatialIndex(dim={self.dim}, splats={self.num_splats}, " +
            f"nodes={self.num_nodes}, depth={self.max_current_depth})"
        )


# Add forward reference to SpatialIndexFactory to avoid circular imports
# This is a function that will be lazily imported when needed
def get_spatial_index_factory():
    """Get the SpatialIndexFactory class lazily to avoid circular imports."""
    from .spatial_index_factory import SpatialIndexFactory
    return SpatialIndexFactory

# Create a class-like object that forwards calls to the real factory
class SpatialIndexFactory:
    """Forward reference to the real SpatialIndexFactory.
    
    This class forwards all calls to the real SpatialIndexFactory in spatial_index_factory.py
    to avoid circular imports.
    """
    
    @staticmethod
    def create_index(*args, **kwargs):
        """Forward to the real create_index method."""
        factory = get_spatial_index_factory()
        return factory.create_index(*args, **kwargs)
    
    @staticmethod
    def analyze_data_distribution(*args, **kwargs):
        """Forward to the real analyze_data_distribution method."""
        factory = get_spatial_index_factory()
        return factory.analyze_data_distribution(*args, **kwargs)
    
    @staticmethod
    def optimize_grid_parameters(*args, **kwargs):
        """Forward to the real optimize_grid_parameters method."""
        factory = get_spatial_index_factory()
        return factory.optimize_grid_parameters(*args, **kwargs)
    
    @staticmethod
    def optimize_tree_parameters(*args, **kwargs):
        """Forward to the real optimize_tree_parameters method."""
        factory = get_spatial_index_factory()
        return factory.optimize_tree_parameters(*args, **kwargs)
    
    @staticmethod
    def create_optimized_index(*args, **kwargs):
        """Forward to the real create_optimized_index method."""
        factory = get_spatial_index_factory()
        return factory.create_optimized_index(*args, **kwargs)
    
    @staticmethod
    def benchmark_index(*args, **kwargs):
        """Forward to the real benchmark_index method."""
        factory = get_spatial_index_factory()
        return factory.benchmark_index(*args, **kwargs)
