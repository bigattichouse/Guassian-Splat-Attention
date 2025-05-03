"""
Base spatial indexing implementation for Hierarchical Splat Attention (HSA).

This module provides the primary tree-based spatial indexing capabilities to optimize 
attention computation by efficiently finding relevant splats in embedding space.
"""

from typing import List, Dict, Set, Tuple, Optional, Any, Union
import numpy as np
import logging
from queue import PriorityQueue

from .splat import Splat
from .registry import SplatRegistry

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
        """
        if self.root is None or self.num_splats == 0:
            return []
        
        # Validate position dimension
        if position.shape != (self.dim,):
            raise ValueError(
                f"Position shape {position.shape} does not match index dimension {(self.dim,)}"
            )
        
        # Limit k to number of splats
        k = min(k, self.num_splats)
        
        # Use priority queue for nearest neighbors
        nearest = PriorityQueue()
        
        # Search the tree
        self.root.find_nearest(position, k, nearest)
        
        # Extract results
        results = []
        while not nearest.empty():
            dist, splat = nearest.get()
            results.append((splat, dist))
        
        # Sort by distance (nearest first)
        results.reverse()
        
        return results
    
    def range_query(self, center: np.ndarray, radius: float) -> List[Splat]:
        """Find all splats within a given radius of a center point.
        
        Args:
            center: Center position in embedding space
            radius: Search radius
            
        Returns:
            List of splats within the radius
        """
        if self.root is None or self.num_splats == 0:
            return []
        
        # Validate position dimension
        if center.shape != (self.dim,):
            raise ValueError(
                f"Position shape {center.shape} does not match index dimension {(self.dim,)}"
            )
        
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
        """
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
        
        # Sort by distance (nearest first)
        distances.sort(key=lambda x: x[1])
        
        # Limit to k
        return distances[:k]
    
    def get_all_splats(self) -> List[Splat]:
        """Get all splats in the index.
        
        Returns:
            List of all splats
        """
        if self.root is None:
            return []
        
        return self.root.get_all_splats()
    
    def _rebuild(self) -> None:
        """Rebuild the tree to balance it."""
        # Get all splats
        all_splats = self.get_all_splats()
        
        # Clear the tree
        self.root = None
        self.splat_to_node = {}
        self.num_nodes = 0
        self.num_splats = 0
        self.max_current_depth = 0
        
        # Reinsert all splats
        for splat in all_splats:
            self.insert(splat)
        
        self.rebuild_count += 1
    
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


class _Node:
    """Internal node for the spatial index tree."""
    
    def __init__(self, dim: int, depth: int, max_leaf_size: int, max_depth: int):
        """Initialize a node.
        
        Args:
            dim: Dimensionality of the embedding space
            depth: Depth of this node in the tree
            max_leaf_size: Maximum number of items in a leaf node
            max_depth: Maximum depth of the tree
        """
        self.dim = dim
        self.depth = depth
        self.max_leaf_size = max_leaf_size
        self.max_depth = max_depth
        
        # For leaf nodes
        self.splats = []
        
        # For internal nodes
        self.split_axis = None
        self.split_value = None
        self.children = None  # [left, right] when split
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node.
        
        Returns:
            True if leaf node, False if internal node
        """
        return self.children is None
    
    def insert(self, splat: Splat) -> '_Node':
        """Insert a splat into this node or its descendants.
        
        Args:
            splat: Splat to insert
            
        Returns:
            Node containing the inserted splat
        """
        if self.is_leaf():
            # Add to this node
            self.splats.append(splat)
            
            # Check if we need to split
            if len(self.splats) > self.max_leaf_size and self.depth < self.max_depth:
                self._split()
                
                # After splitting, insert again to get the right node
                return self.insert(splat)
            
            return self
        else:
            # Internal node, pass to appropriate child
            if splat.position[self.split_axis] <= self.split_value:
                return self.children[0].insert(splat)
            else:
                return self.children[1].insert(splat)
    
    def remove(self, splat_id: str) -> bool:
        """Remove a splat from this node or its descendants.
        
        Args:
            splat_id: ID of the splat to remove
            
        Returns:
            True if splat was removed, False if not found
        """
        if self.is_leaf():
            # Check if splat is in this node
            for i, splat in enumerate(self.splats):
                if splat.id == splat_id:
                    # Remove the splat
                    self.splats.pop(i)
                    return True
            
            return False
        else:
            # Try both children
            for child in self.children:
                if child.remove(splat_id):
                    return True
            
            return False
    
    def find_nearest(self, position: np.ndarray, k: int, nearest: PriorityQueue) -> None:
        """Find the k nearest splats to a position.
        
        Args:
            position: Query position in embedding space
            k: Number of nearest neighbors to find
            nearest: Priority queue to store results
        """
        if self.is_leaf():
            # Check all splats in this leaf
            for splat in self.splats:
                # Calculate distance
                dist = np.linalg.norm(splat.position - position)
                
                # Add to queue
                if nearest.qsize() < k:
                    nearest.put((dist, splat))
                else:
                    # Replace furthest if this is closer
                    furthest_dist, _ = nearest.queue[0]  # Peek at worst item
                    if dist < furthest_dist:
                        nearest.get()  # Remove worst
                        nearest.put((dist, splat))
        else:
            # Determine which child to visit first
            if position[self.split_axis] <= self.split_value:
                first, second = self.children
            else:
                second, first = self.children
            
            # Visit first child
            first.find_nearest(position, k, nearest)
            
            # Check if we need to visit the second child
            if nearest.qsize() < k:
                # Queue not full, definitely visit second child
                second.find_nearest(position, k, nearest)
            else:
                # Check if second child could have closer points
                furthest_dist, _ = nearest.queue[0]  # Peek at worst item
                
                # Distance to the splitting plane
                plane_dist = abs(position[self.split_axis] - self.split_value)
                
                if plane_dist < furthest_dist:
                    # Second child could have closer points
                    second.find_nearest(position, k, nearest)
    
    def range_query(self, center: np.ndarray, radius: float, results: List[Splat]) -> None:
        """Find all splats within a given radius of a center point.
        
        Args:
            center: Center position in embedding space
            radius: Search radius
            results: List to store results
        """
        if self.is_leaf():
            # Check all splats in this leaf
            for splat in self.splats:
                # Calculate distance
                dist = np.linalg.norm(splat.position - center)
                
                # Add if within radius
                if dist <= radius:
                    results.append(splat)
        else:
            # Distance to the splitting plane
            plane_dist = abs(center[self.split_axis] - self.split_value)
            
            # Check if the sphere crosses the splitting plane
            if plane_dist <= radius:
                # Crosses the plane, check both children
                self.children[0].range_query(center, radius, results)
                self.children[1].range_query(center, radius, results)
            else:
                # Only check the side containing the center
                if center[self.split_axis] <= self.split_value:
                    self.children[0].range_query(center, radius, results)
                else:
                    self.children[1].range_query(center, radius, results)
    
    def filter_by_level(self, level: str, results: List[Splat]) -> None:
        """Find all splats at a specific level.
        
        Args:
            level: Hierarchical level to filter by
            results: List to store results
        """
        if self.is_leaf():
            # Add splats that match the level
            for splat in self.splats:
                if splat.level == level:
                    results.append(splat)
        else:
            # Check both children
            for child in self.children:
                child.filter_by_level(level, results)
    
    def get_all_splats(self) -> List[Splat]:
        """Get all splats in this node and its descendants.
        
        Returns:
            List of all splats
        """
        if self.is_leaf():
            return self.splats.copy()
        else:
            splats = []
            for child in self.children:
                splats.extend(child.get_all_splats())
            return splats
    
    def _split(self) -> None:
        """Split this node into two children."""
        # Determine split axis (choose the axis with highest variance)
        positions = np.array([splat.position for splat in self.splats])
        variances = np.var(positions, axis=0)
        self.split_axis = np.argmax(variances)
        
        # Determine split value (median of positions along the axis)
        values = positions[:, self.split_axis]
        self.split_value = np.median(values)
        
        # Create children
        self.children = [
            _Node(self.dim, self.depth + 1, self.max_leaf_size, self.max_depth),
            _Node(self.dim, self.depth + 1, self.max_leaf_size, self.max_depth)
        ]
        
        # Distribute splats to children
        for splat in self.splats:
            if splat.position[self.split_axis] <= self.split_value:
                self.children[0].splats.append(splat)
            else:
                self.children[1].splats.append(splat)
        
        # Clear splats from this node
        self.splats = []


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
