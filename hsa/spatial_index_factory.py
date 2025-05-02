"""
Spatial index factory and analysis tools for Hierarchical Splat Attention (HSA).

This module provides factory methods for creating the appropriate spatial index
based on data characteristics and analytical tools for optimizing index performance.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import logging

from .splat import Splat
from .registry import SplatRegistry
from .spatial_index import SpatialIndex
from .grid_spatial_index import GridSpatialIndex

# Configure logging
logger = logging.getLogger(__name__)


class SpatialIndexFactory:
    """Factory for creating the appropriate spatial index based on data characteristics."""
    
    @staticmethod
    def create_index(
        dim: int,
        splats: Optional[List[Splat]] = None,
        index_type: str = "auto",
        **kwargs
    ) -> Union[SpatialIndex, GridSpatialIndex]:
        """Create a spatial index of the appropriate type.
        
        Args:
            dim: Dimensionality of the embedding space
            splats: Optional list of splats to initialize the index with
            index_type: Type of index to create ("tree", "grid", or "auto")
            **kwargs: Additional arguments for the specific index type
            
        Returns:
            Appropriate spatial index instance
            
        Raises:
            ValueError: If index_type is invalid
        """
        if index_type == "auto":
            # Choose based on dimensionality
            if dim <= 3:
                # Low dimensions: grid is usually more efficient
                index_type = "grid"
            else:
                # Higher dimensions: tree is usually better
                index_type = "tree"
        
        # Create the appropriate index
        if index_type == "tree":
            index = SpatialIndex(
                dim=dim,
                max_leaf_size=kwargs.get("max_leaf_size", 10),
                max_depth=kwargs.get("max_depth", 10)
            )
        elif index_type == "grid":
            index = GridSpatialIndex(
                dim=dim,
                cell_size=kwargs.get("cell_size", 1.0),
                min_coord=kwargs.get("min_coord", -10.0),
                max_coord=kwargs.get("max_coord", 10.0)
            )
        else:
            raise ValueError(f"Invalid index type: {index_type}")
        
        # Add splats if provided
        if splats:
            for splat in splats:
                index.insert(splat)
        
        return index
    
    @staticmethod
    def analyze_data_distribution(splats: List[Splat]) -> Dict[str, Any]:
        """Analyze the distribution of splats to recommend an index type.
        
        Args:
            splats: List of splats to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not splats:
            return {
                "recommended_index": "tree",  # Default for empty data
                "reason": "No data to analyze"
            }
        
        # Get dimensionality
        dim = splats[0].dim
        
        # Extract positions
        positions = np.array([splat.position for splat in splats])
        
        # Calculate basic statistics
        mean_pos = np.mean(positions, axis=0)
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        std_pos = np.std(positions, axis=0)
        
        # Calculate spatial density
        volume = np.prod(max_pos - min_pos)
        if volume > 0:
            density = len(splats) / volume
        else:
            density = float('inf')
        
        # Determine if distribution is uniform or clustered
        try:
            from scipy.stats import entropy
            
            # Simple histogram-based approach
            num_bins = min(10, len(splats) // 5) if len(splats) > 5 else 2
            histograms = []
            for d in range(dim):
                hist, _ = np.histogram(positions[:, d], bins=num_bins, density=True)
                histograms.append(hist)
            
            # Calculate entropy along each dimension
            entropies = [entropy(hist) for hist in histograms]
            mean_entropy = np.mean(entropies)
            
            # Higher entropy = more uniform distribution
            uniform_threshold = 0.7 * np.log(num_bins)  # 70% of max entropy
            is_uniform = mean_entropy >= uniform_threshold
        except ImportError:
            # Fallback if scipy is not available
            logger.warning("scipy not available, using simplified uniformity analysis")
            
            # Simple variance-based approach
            variances = np.var(positions, axis=0)
            std_ratio = np.max(variances) / (np.min(variances) + 1e-10)
            is_uniform = std_ratio < 3.0  # Arbitrary threshold
            mean_entropy = 0.0
        
        # Make recommendation
        if dim <= 3 and is_uniform:
            recommended_index = "grid"
            reason = "Low dimensionality with uniform distribution"
        elif dim <= 3 and density > 0.1:
            recommended_index = "grid"
            reason = "Low dimensionality with high density"
        else:
            recommended_index = "tree"
            reason = f"{'High' if dim > 3 else 'Low'} dimensionality with {'clustered' if not is_uniform else 'uniform'} distribution"
        
        return {
            "recommended_index": recommended_index,
            "reason": reason,
            "dimensionality": dim,
            "num_splats": len(splats),
            "mean_position": mean_pos.tolist(),
            "min_position": min_pos.tolist(),
            "max_position": max_pos.tolist(),
            "std_position": std_pos.tolist(),
            "density": density,
            "uniformity": "uniform" if is_uniform else "clustered",
            "entropy": mean_entropy
        }
    
    @staticmethod
    def optimize_grid_parameters(splats: List[Splat]) -> Dict[str, Any]:
        """Optimize parameters for a grid-based spatial index.
        
        Args:
            splats: List of splats to analyze
            
        Returns:
            Dictionary with optimized parameters
        """
        if not splats:
            return {
                "cell_size": 1.0,
                "min_coord": -10.0,
                "max_coord": 10.0
            }
        
        # Extract positions
        positions = np.array([splat.position for splat in splats])
        
        # Calculate bounds
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        
        # Add margin to bounds
        margin = 0.1 * (max_pos - min_pos)
        min_coord = min_pos - margin
        max_coord = max_pos + margin
        
        # Choose cell size based on data distribution
        # Aim for ~5-10 splats per cell on average
        target_cells = max(1, len(splats) // 7)  # 7 is a good balance
        dim = splats[0].dim
        cells_per_dim = max(1, int(target_cells ** (1/dim)))
        
        # Calculate cell size
        cell_size = np.max((max_coord - min_coord) / cells_per_dim)
        
        return {
            "cell_size": float(cell_size),
            "min_coord": float(np.min(min_coord)),
            "max_coord": float(np.max(max_coord))
        }
    
    @staticmethod
    def optimize_tree_parameters(splats: List[Splat]) -> Dict[str, Any]:
        """Optimize parameters for a tree-based spatial index.
        
        Args:
            splats: List of splats to analyze
            
        Returns:
            Dictionary with optimized parameters
        """
        if not splats:
            return {
                "max_leaf_size": 10,
                "max_depth": 10
            }
        
        # Calculate appropriate leaf size
        # For smaller datasets, larger leaves are better
        # For larger datasets, smaller leaves improve query time
        if len(splats) < 100:
            max_leaf_size = 20  # Larger leaves for small datasets
        elif len(splats) < 1000:
            max_leaf_size = 10  # Medium leaves for medium datasets
        else:
            max_leaf_size = 5   # Small leaves for large datasets
        
        # Calculate appropriate max depth
        # This should be logarithmic in the number of splats
        dim = splats[0].dim
        base = 2.0  # Binary tree
        max_depth = max(5, int(np.log(len(splats)) / np.log(base)) + 3)
        
        # Adjust for dimensionality
        # Higher dimensions need deeper trees
        max_depth = max_depth + max(0, dim - 3)
        
        return {
            "max_leaf_size": max_leaf_size,
            "max_depth": max_depth
        }
    
    @staticmethod
    def create_optimized_index(
        splats: List[Splat],
        index_type: str = "auto"
    ) -> Union[SpatialIndex, GridSpatialIndex]:
        """Create an index with optimized parameters for a dataset.
        
        Args:
            splats: List of splats to analyze and index
            index_type: Type of index to create ("tree", "grid", or "auto")
            
        Returns:
            Optimized spatial index instance
        """
        if not splats:
            # Default parameters for empty dataset
            return SpatialIndexFactory.create_index(
                dim=2,  # Default dimension
                index_type=index_type
            )
        
        # Get dimensionality
        dim = splats[0].dim
        
        # Choose index type if auto
        if index_type == "auto":
            # Analyze distribution to determine best type
            analysis = SpatialIndexFactory.analyze_data_distribution(splats)
            index_type = analysis["recommended_index"]
        
        # Get optimized parameters
        if index_type == "grid":
            params = SpatialIndexFactory.optimize_grid_parameters(splats)
        else:  # tree
            params = SpatialIndexFactory.optimize_tree_parameters(splats)
        
        # Create index with optimized parameters
        return SpatialIndexFactory.create_index(
            dim=dim,
            splats=splats,
            index_type=index_type,
            **params
        )
    
    @staticmethod
    def benchmark_index(
        index: Union[SpatialIndex, GridSpatialIndex],
        test_positions: np.ndarray,
        query_types: List[str] = ["nearest", "range"]
    ) -> Dict[str, Any]:
        """Benchmark the performance of a spatial index.
        
        Args:
            index: Spatial index to benchmark
            test_positions: Positions to use for test queries
            query_types: Types of queries to benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        results = {
            "index_type": "tree" if isinstance(index, SpatialIndex) else "grid",
            "dim": index.dim,
            "num_splats": getattr(index, "num_splats", 0),
            "num_test_positions": len(test_positions),
            "query_times": {}
        }
        
        # Benchmark nearest neighbor queries
        if "nearest" in query_types:
            k_values = [1, 5, 10]
            nearest_times = {}
            
            for k in k_values:
                start_time = time.time()
                
                for position in test_positions:
                    index.find_nearest(position, k)
                
                total_time = time.time() - start_time
                avg_time = total_time / len(test_positions)
                nearest_times[f"k={k}"] = avg_time
            
            results["query_times"]["nearest"] = nearest_times
        
        # Benchmark range queries
        if "range" in query_types:
            radius_values = [0.1, 1.0, 5.0]
            range_times = {}
            
            for radius in radius_values:
                start_time = time.time()
                
                for position in test_positions:
                    index.range_query(position, radius)
                
                total_time = time.time() - start_time
                avg_time = total_time / len(test_positions)
                range_times[f"radius={radius}"] = avg_time
            
            results["query_times"]["range"] = range_times
        
        return results
