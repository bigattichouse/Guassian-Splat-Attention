"""
Node implementation for spatial indexing in Hierarchical Splat Attention (HSA).

This module provides the internal node class used by the spatial index to
organize splats in a tree structure for efficient spatial queries.
"""

from typing import List, Optional, Tuple, Dict, Set
import numpy as np
import heapq
from queue import PriorityQueue

from .splat import Splat


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
                
                # After splitting, find the right node for this splat
                if self.is_leaf():
                    return self
                elif splat.position[self.split_axis] <= self.split_value:
                    return self.children[0]
                else:
                    return self.children[1]
            
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
        """Find k nearest splats to position.
        
        Args:
            position: Query position
            k: Number of nearest neighbors to find
            nearest: Priority queue to store results (negated distance, id, splat)
        """
        if self.is_leaf():
            # For leaf nodes, compute distance to all splats
            for splat in self.splats:
                # Calculate Euclidean distance
                dist = np.linalg.norm(splat.position - position)
                
                # Add to priority queue (negative distance for max-heap behavior)
                item = (-dist, splat.id, splat)
                
                if nearest.qsize() < k:
                    nearest.put(item)
                else:
                    # Compare with the largest distance currently in the queue
                    # If this one is smaller, remove the largest and add this one
                    if -dist > nearest.queue[0][0]:
                        nearest.get()  # Remove the largest distance item
                        nearest.put(item)
        else:
            # For internal nodes, determine which child to search first
            first_child_idx = 0 if position[self.split_axis] <= self.split_value else 1
            second_child_idx = 1 - first_child_idx
            
            # Search the most promising child first
            self.children[first_child_idx].find_nearest(position, k, nearest)
            
            # Check if we need to search the other child
            # Only search if:
            # 1. We haven't found k points yet, or
            # 2. The distance to the splitting plane is less than the largest distance in our queue
            if nearest.qsize() < k:
                # Not enough points found yet
                self.children[second_child_idx].find_nearest(position, k, nearest)
            else:
                # Calculate distance to splitting plane
                dist_to_plane = abs(position[self.split_axis] - self.split_value)
                
                # Get the largest distance in our queue (negated)
                farthest_dist = -nearest.queue[0][0]
                
                # If the plane is closer than our current kth nearest neighbor,
                # we need to check the other side
                if dist_to_plane < farthest_dist:
                    self.children[second_child_idx].find_nearest(position, k, nearest)

    def range_query(self, center: np.ndarray, radius: float, results: List[Splat]) -> None:
        """Find all splats within radius of center.
        
        Args:
            center: Center position
            radius: Search radius
            results: List to store results
        """
        if self.is_leaf():
            # For leaf nodes, check all splats
            for splat in self.splats:
                dist = np.linalg.norm(splat.position - center)
                if dist <= radius:
                    results.append(splat)
        else:
            # For internal nodes, check both children if they could contain points within radius
            dist_to_plane = abs(center[self.split_axis] - self.split_value)
            
            # Always search the side containing the center
            if center[self.split_axis] <= self.split_value:
                # Center is in left subtree
                self.children[0].range_query(center, radius, results)
                
                # Check if we need to search the right subtree
                if dist_to_plane <= radius:
                    self.children[1].range_query(center, radius, results)
            else:
                # Center is in right subtree
                self.children[1].range_query(center, radius, results)
                
                # Check if we need to search the left subtree
                if dist_to_plane <= radius:
                    self.children[0].range_query(center, radius, results)        
                    
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
        """Get all splats in this node and its descendants."""
        if self.is_leaf():
            return self.splats.copy()  # Make sure to return a copy
        else:
            splats = []
            for child in self.children:
                if child is not None:  # Check if child exists
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
        left_splats = []
        right_splats = []
        
        for splat in self.splats:
            if splat.position[self.split_axis] <= self.split_value:
                left_splats.append(splat)
            else:
                right_splats.append(splat)
        
        # Set children's splats directly to avoid recursive insertion
        self.children[0].splats = left_splats
        self.children[1].splats = right_splats
        
        # Clear splats from this node
        self.splats = []
