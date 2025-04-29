"""
Core data structures for Hierarchical Splat Attention (HSA).

This module implements the fundamental data structures required for HSA:
- Splat: The core element that represents attention zones in the embedding space
- Hierarchy: Manages the hierarchical levels and their configurations
- Utility functions for vector/matrix operations

This is the foundation module for the HSA system with no external dependencies.
"""

import math
import numpy as np
from typing import List, Set, Optional, Dict, Union, Tuple, Any

class Splat:
    """
    The core element in HSA that represents an attention zone in embedding space.
    
    A splat has a position in the embedding space, a covariance matrix that defines
    its shape, an amplitude that controls its strength, and a hierarchical level.
    It can also have parent-child relationships with splats at adjacent levels.
    """
    
    def __init__(
        self,
        position: np.ndarray,
        covariance: np.ndarray,
        amplitude: float,
        level: str,
        splat_id: Optional[str] = None
    ):
        """
        Initialize a new Splat.
        
        Args:
            position: Center position vector in embedding space
            covariance: Covariance matrix defining the shape/extent of the splat
            amplitude: Scalar value controlling the strength of the splat
            level: Hierarchical level name this splat belongs to
            splat_id: Optional unique identifier for the splat
        """
        self.position = position
        self.covariance = covariance
        self.amplitude = amplitude
        self.level = level
        self.id = splat_id or self._generate_id()
        
        # Relationship attributes
        self.parent: Optional[Splat] = None
        self.children: Set[Splat] = set()
        
        # Cached values for efficiency
        self._covariance_inverse = None
        self._normalization_factor = None
        
    def _generate_id(self) -> str:
        """Generate a unique identifier for this splat."""
        import uuid
        return f"splat_{uuid.uuid4().hex[:8]}"
    
    @property
    def covariance_inverse(self) -> np.ndarray:
        """
        Get the inverse of the covariance matrix, computing it if not already cached.
        
        Returns:
            The inverse covariance matrix
        """
        if self._covariance_inverse is None:
            self._covariance_inverse = np.linalg.inv(self.covariance)
        return self._covariance_inverse
    
    @property
    def normalization_factor(self) -> float:
        """
        Calculate the normalization factor for Mahalanobis distance.
        
        Returns:
            The normalization factor for the Gaussian distribution
        """
        if self._normalization_factor is None:
            dim = self.position.shape[0]
            det = np.linalg.det(self.covariance)
            self._normalization_factor = 1.0 / (math.sqrt((2 * math.pi) ** dim * det))
        return self._normalization_factor
    
    def compute_distance(self, token_i: np.ndarray, token_j: np.ndarray) -> float:
        """
        Compute Mahalanobis distance between tokens relative to this splat.
        
        Args:
            token_i: First token embedding
            token_j: Second token embedding
            
        Returns:
            Mahalanobis distance value
        """
        # Calculate the token difference relative to the splat position
        diff = (token_i - token_j) - self.position
        
        # Compute Mahalanobis distance: sqrt((diff)^T * cov^-1 * diff)
        distance = math.sqrt(diff @ self.covariance_inverse @ diff)
        return distance
    
    def compute_attention(self, token_i: np.ndarray, token_j: np.ndarray) -> float:
        """
        Compute attention score between tokens based on this splat.
        
        Args:
            token_i: First token embedding
            token_j: Second token embedding
            
        Returns:
            Attention score value
        """
        distance = self.compute_distance(token_i, token_j)
        return self.amplitude * math.exp(-distance**2)
    
    def clone(self) -> 'Splat':
        """
        Create a copy of this splat with the same parameters.
        
        Returns:
            A new Splat object with copied parameters
        """
        new_splat = Splat(
            position=self.position.copy(),
            covariance=self.covariance.copy(),
            amplitude=self.amplitude,
            level=self.level
        )
        return new_splat
    
    def add_child(self, child: 'Splat') -> None:
        """
        Add a child splat to this splat.
        
        Args:
            child: The splat to add as a child
            
        Raises:
            ValueError: If the child is not at the appropriate level
        """
        self.children.add(child)
        child.parent = self
    
    def remove_child(self, child: 'Splat') -> None:
        """
        Remove a child splat from this splat.
        
        Args:
            child: The splat to remove from children
        """
        if child in self.children:
            self.children.remove(child)
            child.parent = None
    
    def __repr__(self) -> str:
        """
        String representation of the Splat.
        
        Returns:
            String representation
        """
        return f"Splat(id={self.id}, level={self.level}, position={self.position.shape}, amplitude={self.amplitude:.3f})"

class Hierarchy:
    """
    Manages the hierarchical structure of splats across different levels.
    
    The hierarchy defines the levels, their relative weights, and initialization
    parameters for splats at each level.
    """
    
    def __init__(
        self, 
        levels: List[str],
        init_splats_per_level: List[int],
        level_weights: List[float]
    ):
        """
        Initialize a new Hierarchy.
        
        Args:
            levels: List of level names from finest to coarsest
            init_splats_per_level: Number of splats to initialize at each level
            level_weights: Weight for attention contribution at each level
            
        Raises:
            ValueError: If the lengths of the provided lists don't match
        """
        if not (len(levels) == len(init_splats_per_level) == len(level_weights)):
            raise ValueError("All hierarchy parameters must have the same length")
        
        self.levels = levels
        self.init_splats_per_level = init_splats_per_level
        self.level_weights = level_weights
        
        # Normalize weights if they don't sum to 1
        total_weight = sum(level_weights)
        if abs(total_weight - 1.0) > 1e-6:
            self.level_weights = [w / total_weight for w in level_weights]
    
    def get_level_index(self, level_name: str) -> int:
        """
        Get the index of a level by name.
        
        Args:
            level_name: Name of the level
            
        Returns:
            Integer index of the level
            
        Raises:
            ValueError: If the level name is not found
        """
        try:
            return self.levels.index(level_name)
        except ValueError:
            raise ValueError(f"Level '{level_name}' not found in hierarchy")
    
    def get_level_weight(self, level_name: str) -> float:
        """
        Get the weight for a specific level.
        
        Args:
            level_name: Name of the level
            
        Returns:
            Weight value for the level
        """
        idx = self.get_level_index(level_name)
        return self.level_weights[idx]
    
    def get_init_splats_count(self, level_name: str) -> int:
        """
        Get the initialization count for splats at a specific level.
        
        Args:
            level_name: Name of the level
            
        Returns:
            Number of splats to initialize at this level
        """
        idx = self.get_level_index(level_name)
        return self.init_splats_per_level[idx]
    
    def get_parent_level(self, level_name: str) -> Optional[str]:
        """
        Get the name of the parent level for a given level.
        
        Args:
            level_name: Name of the current level
            
        Returns:
            Name of the parent level or None if this is the top level
        """
        idx = self.get_level_index(level_name)
        if idx < len(self.levels) - 1:
            return self.levels[idx + 1]
        return None
    
    def get_child_level(self, level_name: str) -> Optional[str]:
        """
        Get the name of the child level for a given level.
        
        Args:
            level_name: Name of the current level
            
        Returns:
            Name of the child level or None if this is the bottom level
        """
        idx = self.get_level_index(level_name)
        if idx > 0:
            return self.levels[idx - 1]
        return None
    
    def __repr__(self) -> str:
        """
        String representation of the Hierarchy.
        
        Returns:
            String representation
        """
        return f"Hierarchy(levels={self.levels}, weights={self.level_weights})"

# Utility functions for vector/matrix operations

def ensure_positive_definite(matrix: np.ndarray, min_eigenval: float = 1e-6) -> np.ndarray:
    """
    Ensure a matrix is positive definite by adjusting its eigenvalues.
    
    Args:
        matrix: Input matrix to check and adjust
        min_eigenval: Minimum eigenvalue to enforce
        
    Returns:
        Adjusted positive definite matrix
    """
    # Make sure matrix is symmetric
    matrix = (matrix + matrix.T) / 2
    
    # Compute eigendecomposition
    eigenvals, eigenvecs = np.linalg.eigh(matrix)
    
    # Adjust eigenvalues to ensure positive definiteness
    eigenvals = np.maximum(eigenvals, min_eigenval)
    
    # Reconstruct the matrix
    adjusted_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    return adjusted_matrix

def sample_covariance_matrix(dim: int, scale: float = 1.0) -> np.ndarray:
    """
    Generate a random positive definite covariance matrix.
    
    Args:
        dim: Dimensionality of the matrix
        scale: Scale factor for the matrix values
        
    Returns:
        Random positive definite covariance matrix
    """
    # Create a random matrix
    random_matrix = np.random.randn(dim, dim) * scale
    
    # Make it positive definite
    covariance = random_matrix @ random_matrix.T
    
    # Ensure positive definiteness and return
    return ensure_positive_definite(covariance)

def mahalanobis_distance(x: np.ndarray, mean: np.ndarray, cov_inv: np.ndarray) -> float:
    """
    Compute the Mahalanobis distance between a vector and a distribution.
    
    Args:
        x: Input vector
        mean: Mean vector of the distribution
        cov_inv: Inverse covariance matrix of the distribution
        
    Returns:
        Mahalanobis distance
    """
    diff = x - mean
    return math.sqrt(diff @ cov_inv @ diff)

class SplatRegistry:
    """
    Registry to manage and track all splats in the system.
    
    This class provides functionality to register, retrieve, and organize splats
    across hierarchical levels.
    """
    
    def __init__(self, hierarchy: Hierarchy):
        """
        Initialize a new SplatRegistry.
        
        Args:
            hierarchy: The hierarchy configuration for the splats
        """
        self.hierarchy = hierarchy
        self.splats: Dict[str, Splat] = {}  # All splats by ID
        self.splats_by_level: Dict[str, Set[Splat]] = {level: set() for level in hierarchy.levels}
    
    def register(self, splat: Splat) -> None:
        """
        Register a splat in the registry.
        
        Args:
            splat: The splat to register
            
        Raises:
            ValueError: If a splat with the same ID already exists
        """
        if splat.id in self.splats:
            raise ValueError(f"Splat with ID {splat.id} already exists in the registry")
        
        self.splats[splat.id] = splat
        self.splats_by_level[splat.level].add(splat)
    
    def unregister(self, splat: Union[Splat, str]) -> None:
        """
        Remove a splat from the registry.
        
        Args:
            splat: The splat or splat ID to unregister
            
        Raises:
            KeyError: If the splat is not found
        """
        splat_id = splat.id if isinstance(splat, Splat) else splat
        
        if splat_id not in self.splats:
            raise KeyError(f"Splat with ID {splat_id} not found in registry")
        
        splat_obj = self.splats[splat_id]
        self.splats_by_level[splat_obj.level].remove(splat_obj)
        del self.splats[splat_id]
        
        # Handle parent-child relationships
        if splat_obj.parent:
            splat_obj.parent.remove_child(splat_obj)
        
        # Remove all child references
        for child in list(splat_obj.children):
            splat_obj.remove_child(child)
    
    def get_splat(self, splat_id: str) -> Splat:
        """
        Get a splat by its ID.
        
        Args:
            splat_id: The ID of the splat to retrieve
            
        Returns:
            The requested splat
            
        Raises:
            KeyError: If the splat is not found
        """
        if splat_id not in self.splats:
            raise KeyError(f"Splat with ID {splat_id} not found in registry")
        return self.splats[splat_id]
    
    def get_splats_at_level(self, level: str) -> Set[Splat]:
        """
        Get all splats at a specific level.
        
        Args:
            level: The hierarchy level name
            
        Returns:
            Set of splats at the specified level
            
        Raises:
            KeyError: If the level is not valid
        """
        if level not in self.splats_by_level:
            raise KeyError(f"Level {level} not found in hierarchy")
        return self.splats_by_level[level]
    
    def replace_splat(self, old_splat: Splat, new_splats: List[Splat]) -> None:
        """
        Replace a splat with one or more new splats (for adaptation mechanisms).
        
        Args:
            old_splat: The splat to replace
            new_splats: List of new splats to add
            
        Raises:
            KeyError: If the old splat is not found
        """
        # Remember parent reference before unregistering
        parent = old_splat.parent
        
        # Unregister the old splat
        self.unregister(old_splat)
        
        # Register all new splats
        for new_splat in new_splats:
            self.register(new_splat)
            
            # Maintain the parent relationship if it existed
            if parent:
                parent.add_child(new_splat)
    
    def __len__(self) -> int:
        """
        Get the total number of splats in the registry.
        
        Returns:
            Count of all registered splats
        """
        return len(self.splats)
