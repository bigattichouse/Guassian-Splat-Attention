"""
Grid-based spatial indexing for Hierarchical Splat Attention (HSA).

This module provides a grid-based spatial indexing implementation which
divides the embedding space into cells for efficient spatial queries,
particularly in low-dimensional spaces.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import logging
from collections import defaultdict

from .splat import Splat

# Configure logging
logger = logging.getLogger(__name__)


class GridSpatialIndex:
    """
    Grid-based spatial index for efficient queries in embedding space.
    
    This class divides the embedding space into a grid of cells for faster
    spatial queries, particularly useful for low-dimensional spaces.
    """
    
    def __init__(self, dim: int, cell_size: float = 1.0, min_coord: float = -10.0, max_coord: float = 10.0):
        """Initialize a grid-based spatial index.
        
        Args:
            dim: Dimensionality of the embedding space
            cell_size: Size of each grid cell
            min_coord: Minimum coordinate value for the grid
            max_coord: Maximum coordinate value for the grid
        """
        self.dim = dim
        self.cell_size = cell_size
        self.min_coord = min_coord
        self.max_coord = max_coord
        
        # Calculate grid dimensions
        self.grid_size = int((max_coord - min_coord) / cell_size) + 1
        
        # Initialize empty grid
        self.grid = defaultdict(list)
        
        # Map from splat ID to grid cell
        self.splat_to_cell = {}
        
        # Statistics
        self.num_splats = 0
        self.num_cells = 0
    
    def _get_cell_index(self, position: np.ndarray) -> Tuple:
        """Get the grid cell index for a position.
        
        Args:
            position: Position in embedding space
            
        Returns:
            Tuple of cell indices
        """
        # Clip to grid bounds
        clipped = np.clip(position, self.min_coord, self.max_coord)
        
        # Convert to cell indices
        indices = tuple(int((coord - self.min_coord) / self.cell_size) for coord in clipped)
        
        return indices
    
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
        
        # Get cell index
        cell = self._get_cell_index(splat.position)
        
        # Add to cell
        self.grid[cell].append(splat)
        
        # Update mapping
        self.splat_to_cell[splat.id] = cell
        
        # Update statistics
        self.num_splats += 1
        self.num_cells = len(self.grid)
    
    def remove(self, splat_id: str) -> bool:
        """Remove a splat from the index.
        
        Args:
            splat_id: ID of the splat to remove
            
        Returns:
            True if splat was removed, False if not found
        """
        if splat_id not in self.splat_to_cell:
            return False
        
        # Get cell containing the splat
        cell = self.splat_to_cell[splat_id]
        
        # Find and remove the splat
        for i, splat in enumerate(self.grid[cell]):
            if splat.id == splat_id:
                self.grid[cell].pop(i)
                
                # Remove from mapping
                del self.splat_to_cell[splat_id]
                
                # Update statistics
                self.num_splats -= 1
                
                # Clean up empty cells
                if not self.grid[cell]:
                    del self.grid[cell]
                    self.num_cells = len(self.grid)
                
                return True
        
        return False
    
    def update(self, splat: Splat) -> None:
        """Update a splat's position in the index.
        
        Args:
            splat: Splat with updated position
            
        Raises:
            ValueError: If splat is not in the index
        """
        if splat.id not in self.splat_to_cell:
            raise ValueError(f"Splat {splat.id} not found in index")
        
        # Get old cell
        old_cell = self.splat_to_cell[splat.id]
        
        # Get new cell
        new_cell = self._get_cell_index(splat.position)
        
        # If cell has changed, update
        if old_cell != new_cell:
            # Remove from old cell
            for i, old_splat in enumerate(self.grid[old_cell]):
                if old_splat.id == splat.id:
                    self.grid[old_cell].pop(i)
                    break
            
            # Clean up empty cells
            if not self.grid[old_cell]:
                del self.grid[old_cell]
                
            # Add to new cell
            self.grid[new_cell].append(splat)
            
            # Update mapping
            self.splat_to_cell[splat.id] = new_cell
            
            # Update statistics
            self.num_cells = len(self.grid)
        else:
            # Just update in place
            for i, old_splat in enumerate(self.grid[old_cell]):
                if old_splat.id == splat.id:
                    self.grid[old_cell][i] = splat
                    break
    
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
            
        if self.num_splats == 0:
            return []
        
        # Limit k to number of splats
        k = min(k, self.num_splats)
        
        # Get cell index
        cell = self._get_cell_index(position)
        
        # Start with current cell
        candidates = list(self.grid.get(cell, []))
        
        # If we need more, explore neighboring cells
        if len(candidates) < k:
            # Generate neighbor cells in increasing distance order
            max_radius = max(
                int(np.ceil((self.max_coord - self.min_coord) / self.cell_size)),
                1
            )
            
            for radius in range(1, max_radius):
                # Generate neighbor cells at this radius
                neighbors = self._get_neighbor_cells(cell, radius)
                
                # Add splats from these cells
                for neighbor in neighbors:
                    candidates.extend(self.grid.get(neighbor, []))
                
                # If we have enough candidates, stop
                if len(candidates) >= k:
                    break
        
        # Calculate distances for all candidates
        distances = []
        for splat in candidates:
            dist = np.linalg.norm(splat.position - position)
            distances.append((splat, dist))
        
        # Sort by distance (nearest first), then by ID for determinism
        distances.sort(key=lambda x: (x[1], x[0].id))
        
        # Limit to k
        return distances[:k]

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
            
        if self.num_splats == 0:
            return []
        
        # Calculate cells that could contain points within radius
        cell_radius = int(np.ceil(radius / self.cell_size))
        center_cell = self._get_cell_index(center)
        
        # Get all cells within the radius
        cells = [center_cell]
        for r in range(1, cell_radius + 1):
            cells.extend(self._get_neighbor_cells(center_cell, r))
        
        # Collect all splats from these cells
        candidates = []
        for cell in cells:
            candidates.extend(self.grid.get(cell, []))
        
        # Filter by actual distance
        results = []
        for splat in candidates:
            dist = np.linalg.norm(splat.position - center)
            if dist <= radius:
                results.append(splat)
        
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
            
        if self.num_splats == 0:
            return []
        
        # Get all splats of this level
        level_splats = []
        for splats in self.grid.values():
            for splat in splats:
                if splat.level == level:
                    level_splats.append(splat)
        
        if not level_splats:
            return []
        
        # Calculate distances
        distances = []
        for splat in level_splats:
            dist = np.linalg.norm(splat.position - position)
            distances.append((splat, dist))
        
        # Sort by distance (nearest first), then by ID for determinism
        distances.sort(key=lambda x: (x[1], x[0].id))
        
        # Limit to k
        return distances[:k]
    
    def get_all_splats(self) -> List[Splat]:
        """Get all splats in the index.
        
        Returns:
            List of all splats
        """
        all_splats = []
        for splats in self.grid.values():
            all_splats.extend(splats)
        return all_splats
    
    def _get_neighbor_cells(self, cell: tuple, radius: int) -> list:
        """Get neighbor cells at a specific radius.
        
        Args:
            cell: Center cell indices
            radius: Distance in cell units
            
        Returns:
            List of neighbor cell indices
        """
        if radius == 0:
            return [cell]
            
        neighbors = []
        
        # This is a simple implementation for 2D and 3D cases
        # For higher dimensions, it's more complex
        if self.dim == 1:
            neighbors = [
                (cell[0] - radius,),
                (cell[0] + radius,)
            ]
        elif self.dim == 2:
            # Add cells at exactly radius distance
            for i in range(-radius, radius + 1):
                j_pos = radius - abs(i)
                j_neg = -j_pos
                
                if j_pos == 0:  # Only one cell at the extremes
                    neighbors.append((cell[0] + i, cell[1]))
                else:  # Two cells (positive and negative j)
                    neighbors.append((cell[0] + i, cell[1] + j_pos))
                    neighbors.append((cell[0] + i, cell[1] + j_neg))
        elif self.dim == 3:
            # This is an approximation - it doesn't give exact sphere shells
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    k_pos = radius - abs(i) - abs(j)
                    
                    if k_pos >= 0:  # Within radius
                        neighbors.append((cell[0] + i, cell[1] + j, cell[2] + k_pos))
                        if k_pos > 0:  # Skip duplicating the equator
                            neighbors.append((cell[0] + i, cell[1] + j, cell[2] - k_pos))
        else:
            # For higher dimensions, just use a simpler approximation
            # Generate neighbors along each axis
            for d in range(self.dim):
                for direction in [-1, 1]:
                    neighbor = list(cell)
                    neighbor[d] += direction * radius
                    neighbors.append(tuple(neighbor))
        
        return neighbors
