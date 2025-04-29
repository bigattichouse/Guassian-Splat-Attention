"""
Factory module for Hierarchical Splat Attention (HSA).

This module provides factory functions and utilities for creating:
- Attention computers with different implementations
- Splat registries with various configurations
- Hierarchies with appropriate level structures
- Preset configurations for common use cases
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any

# Import core data structures
from hsa.data_structures import Splat, Hierarchy, SplatRegistry

# Import attention implementations
from hsa.attention.base import AttentionComputer
from hsa.attention.implementations import (
    DenseAttentionComputer, 
    SparseAttentionComputer, 
    SpatialAttentionComputer
)

# Import initialization utilities
from hsa.initialization import initialize_splats

def create_attention_computer(
    hierarchy: Hierarchy, 
    sparse_topk: int = 64,
    efficient: bool = True,
    use_spatial: bool = False,
    max_splat_radius: float = 3.0,
    device: str = "cpu"
) -> AttentionComputer:
    """
    Create an appropriate attention computer based on configuration.
    
    Args:
        hierarchy: Hierarchy configuration for attention
        sparse_topk: Number of top-k connections to keep per token
        efficient: Whether to use optimized implementations
        use_spatial: Whether to use the spatial indexing implementation
        max_splat_radius: Maximum radius of influence for splats
        device: Compute device ("cpu" or "cuda")
        
    Returns:
        Attention computer instance appropriate for the configuration
    """
    # Import torch only if needed to check CUDA availability
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                device = "cpu"
        except ImportError:
            print("PyTorch not available, using CPU")
            device = "cpu"
    
    # Create the appropriate implementation based on arguments
    if not efficient:
        # Basic implementation for debugging or validation
        return DenseAttentionComputer(hierarchy, sparse_topk, max_splat_radius)
    
    elif use_spatial:
        # Spatial indexing for very large sequences
        spatial_computer = SpatialAttentionComputer(hierarchy, sparse_topk, max_splat_radius)
        return spatial_computer
    
    else:
        # Default to sparse attention for most cases
        sparse_computer = SparseAttentionComputer(hierarchy, sparse_topk, max_splat_radius)
        return sparse_computer

def create_splat_registry(
    tokens: np.ndarray,
    hierarchy_config: Dict[str, Any],
    n_neighbors: int = 10,
    affinity: str = 'nearest_neighbors',
    random_seed: Optional[int] = None
) -> SplatRegistry:
    """
    Create a splat registry with initialized splats.
    
    Args:
        tokens: Token embeddings for initialization
        hierarchy_config: Configuration for hierarchy (levels, counts, weights)
        n_neighbors: Number of neighbors for clustering
        affinity: Clustering affinity method
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Initialized splat registry
    """
    # Create hierarchy object
    hierarchy = create_hierarchy(hierarchy_config)
    
    # Initialize splats and registry
    registry = initialize_splats(
        tokens=tokens,
        hierarchy_config={
            "levels": hierarchy.levels,
            "init_splats_per_level": hierarchy.init_splats_per_level,
            "level_weights": hierarchy.level_weights
        },
        n_neighbors=n_neighbors,
        affinity=affinity,
        random_seed=random_seed
    )
    
    return registry

def create_hierarchy(config: Dict[str, Any]) -> Hierarchy:
    """
    Create a hierarchy from configuration.
    
    Args:
        config: Configuration dictionary with hierarchy parameters
        
    Returns:
        Hierarchy object
    """
    # Extract required parameters
    levels = config.get("levels", ["Token", "Phrase", "Section", "Document"])
    
    # Get init splats per level, with defaults based on levels if not provided
    init_splats_per_level = config.get("init_splats_per_level")
    if init_splats_per_level is None:
        # Default declining by factor of 2
        init_splats_per_level = [100 // (2**i) for i in range(len(levels))]
    
    # Get level weights, with defaults if not provided
    level_weights = config.get("level_weights")
    if level_weights is None:
        # Default to linear decline
        total = len(levels) * (len(levels) + 1) // 2  # Sum of 1..n
        level_weights = [(len(levels) - i) / total for i in range(len(levels))]
    
    # Create and return hierarchy
    return Hierarchy(
        levels=levels,
        init_splats_per_level=init_splats_per_level,
        level_weights=level_weights
    )

def create_preset_config(
    preset_name: str,
    embedding_dim: Optional[int] = None,
    sequence_length: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a preset configuration for a specific use case.
    
    Args:
        preset_name: Name of the preset configuration
        embedding_dim: Optional embedding dimension override
        sequence_length: Optional sequence length for optimization
        
    Returns:
        Configuration dictionary
    """
    # Standard text configuration (default)
    if preset_name == "standard_text":
        return {
            "levels": ["Token", "Phrase", "Section", "Document"],
            "init_splats_per_level": [100, 50, 20, 5],
            "level_weights": [0.4, 0.3, 0.2, 0.1],
            "sparse_topk": 64,
            "use_spatial": False
        }
    
    # Long context configuration
    elif preset_name == "long_context":
        # Scale splat counts based on sequence length if provided
        if sequence_length is not None:
            scale_factor = max(1.0, min(4.0, sequence_length / 1000))
            splat_counts = [
                int(200 * scale_factor),
                int(100 * scale_factor),
                int(50 * scale_factor),
                int(20 * scale_factor)
            ]
        else:
            splat_counts = [200, 100, 50, 20]
            
        return {
            "levels": ["Token", "Phrase", "Section", "Document", "Global"],
            "init_splats_per_level": splat_counts,
            "level_weights": [0.3, 0.25, 0.2, 0.15, 0.1],
            "sparse_topk": 128,
            "use_spatial": True
        }
    
    # Interpretable configuration (fewer splats, clearer hierarchy)
    elif preset_name == "interpretable":
        return {
            "levels": ["Token", "Phrase", "Section"],
            "init_splats_per_level": [50, 20, 5],
            "level_weights": [0.5, 0.3, 0.2],
            "sparse_topk": 32,
            "use_spatial": False
        }
    
    # Memory-efficient configuration for limited hardware
    elif preset_name == "memory_efficient":
        return {
            "levels": ["Token", "Document"],
            "init_splats_per_level": [50, 10],
            "level_weights": [0.7, 0.3],
            "sparse_topk": 32,
            "use_spatial": False
        }
    
    # High-speed configuration for real-time applications
    elif preset_name == "high_speed":
        return {
            "levels": ["Token", "Document"],
            "init_splats_per_level": [30, 5],
            "level_weights": [0.7, 0.3],
            "sparse_topk": 16,
            "use_spatial": False
        }
    
    # Domain-specific configurations
    elif preset_name == "code_understanding":
        return {
            "levels": ["Token", "Expression", "Block", "Function", "Module"],
            "init_splats_per_level": [100, 50, 20, 10, 5],
            "level_weights": [0.3, 0.25, 0.2, 0.15, 0.1],
            "sparse_topk": 64,
            "use_spatial": False
        }
    
    elif preset_name == "document_qa":
        return {
            "levels": ["Token", "Sentence", "Paragraph", "Section", "Document"],
            "init_splats_per_level": [100, 50, 20, 10, 5],
            "level_weights": [0.3, 0.25, 0.2, 0.15, 0.1],
            "sparse_topk": 64,
            "use_spatial": True
        }
    
    # Unknown preset
    else:
        raise ValueError(f"Unknown preset configuration: {preset_name}")

def estimate_optimal_params(
    embedding_dim: int,
    sequence_length: int,
    available_memory_mb: Optional[int] = None,
    target_complexity: str = "medium"
) -> Dict[str, Any]:
    """
    Estimate optimal HSA parameters based on input dimensions and resources.
    
    Args:
        embedding_dim: Embedding dimension
        sequence_length: Maximum sequence length
        available_memory_mb: Available memory in MB (optional)
        target_complexity: Complexity target ("low", "medium", "high")
        
    Returns:
        Configuration dictionary with estimated parameters
    """
    # Scale factors based on complexity target
    complexity_factors = {
        "low": 0.5,
        "medium": 1.0,
        "high": 2.0
    }
    factor = complexity_factors.get(target_complexity, 1.0)
    
    # Estimate optimal sparse_topk based on sequence length
    if sequence_length < 256:
        sparse_topk = 32
    elif sequence_length < 1024:
        sparse_topk = int(64 * factor)
    elif sequence_length < 4096:
        sparse_topk = int(128 * factor)
    else:
        sparse_topk = int(256 * factor)
    
    # Estimate splat counts based on embedding dimension and sequence length
    # Higher dimensionality requires more splats to cover the space
    base_splats = int(20 * (embedding_dim / 128)**0.5 * factor)
    
    # Adjust for sequence length (longer sequences need more splats)
    length_factor = (sequence_length / 512)**0.3
    
    # Create level structure based on sequence length
    if sequence_length < 512:
        levels = ["Token", "Phrase", "Document"]
        ratios = [1.0, 0.5, 0.2]
    elif sequence_length < 2048:
        levels = ["Token", "Phrase", "Section", "Document"]
        ratios = [1.0, 0.5, 0.2, 0.1]
    else:
        levels = ["Token", "Phrase", "Section", "Document", "Global"]
        ratios = [1.0, 0.5, 0.2, 0.1, 0.05]
    
    # Calculate splat counts for each level
    splat_counts = [int(base_splats * ratio * length_factor) for ratio in ratios]
    
    # Ensure minimum counts
    splat_counts = [max(5, count) for count in splat_counts]
    
    # Calculate level weights (more weight to lower levels for most cases)
    total = sum(1 / (i + 1) for i in range(len(levels)))
    level_weights = [1 / (i + 1) / total for i in range(len(levels))]
    
    # Determine whether to use spatial indexing based on sequence length
    use_spatial = sequence_length > 1024
    
    # Create configuration
    config = {
        "levels": levels,
        "init_splats_per_level": splat_counts,
        "level_weights": level_weights,
        "sparse_topk": sparse_topk,
        "use_spatial": use_spatial
    }
    
    # If available memory is provided, adjust for memory constraints
    if available_memory_mb is not None:
        mem_per_token = embedding_dim * 4 / 1024 / 1024  # 4 bytes per float
        tokens_memory = sequence_length * mem_per_token
        
        # If memory is constrained, make adjustments
        if tokens_memory > available_memory_mb * 0.3:  # Using 30% as threshold
            # Reduce splat counts
            scale = min(1.0, (available_memory_mb * 0.3) / tokens_memory)
            config["init_splats_per_level"] = [max(5, int(count * scale)) for count in splat_counts]
            
            # Reduce top-k for more sparsity
            config["sparse_topk"] = max(16, int(sparse_topk * scale))
            
            # More aggressive cleanup of temporaries
            config["aggressive_memory_cleanup"] = True
    
    return config

def adjust_config_for_task(
    base_config: Dict[str, Any],
    task_name: str
) -> Dict[str, Any]:
    """
    Adjust configuration parameters for a specific task.
    
    Args:
        base_config: Base configuration dictionary
        task_name: Name of the task for specialization
        
    Returns:
        Adjusted configuration dictionary
    """
    # Make a copy to avoid modifying original
    config = base_config.copy()
    
    if task_name == "translation":
        # Translation needs good cross-lingual attention
        config["level_weights"] = [0.5, 0.3, 0.2]  # More weight to token level
        config["sparse_topk"] = min(int(config["sparse_topk"] * 1.5), 256)  # More connections
    
    elif task_name == "summarization":
        # Summarization needs good document-level attention
        if "Document" in config["levels"]:
            doc_idx = config["levels"].index("Document")
            # Increase document-level weight
            config["level_weights"][doc_idx] = config["level_weights"][doc_idx] * 1.5
            # Normalize weights to sum to 1
            total = sum(config["level_weights"])
            config["level_weights"] = [w / total for w in config["level_weights"]]
    
    elif task_name == "code_completion":
        # Code completion needs special structure
        config["levels"] = ["Token", "Expression", "Block", "Function"]
        # Re-adjust splat counts and weights for code structure
        config["init_splats_per_level"] = [100, 50, 20, 10]
        config["level_weights"] = [0.4, 0.3, 0.2, 0.1]
    
    elif task_name == "question_answering":
        # QA needs good cross-attention between query and context
        config["sparse_topk"] = min(int(config["sparse_topk"] * 1.3), 256)
        # Add global level if not present
        if "Global" not in config["levels"]:
            config["levels"].append("Global")
            config["init_splats_per_level"].append(5)
            # Re-normalize weights
            old_weight_sum = sum(config["level_weights"])
            global_weight = 0.1
            config["level_weights"] = [(1 - global_weight) * w / old_weight_sum for w in config["level_weights"]]
            config["level_weights"].append(global_weight)
    
    return config

def create_hierarchy_for_data(
    tokens: np.ndarray,
    num_levels: int = 4,
    clustering_method: str = "spectral",
    random_seed: Optional[int] = None
) -> Hierarchy:
    """
    Create a data-driven hierarchy based on token embeddings.
    
    Args:
        tokens: Token embeddings
        num_levels: Number of hierarchical levels
        clustering_method: Method for clustering analysis
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Hierarchy object optimized for the data
    """
    # Import clustering utilities
    from sklearn.cluster import SpectralClustering
    
    # Determine hierarchy levels
    levels = ["Token"]  # Always start with token level
    
    # Add intermediate levels based on num_levels
    if num_levels == 2:
        levels.append("Document")
    elif num_levels == 3:
        levels.extend(["Phrase", "Document"])
    elif num_levels == 4:
        levels.extend(["Phrase", "Section", "Document"])
    elif num_levels >= 5:
        levels.extend(["Phrase", "Section", "Document", "Global"])
        # Add more levels if needed
        for i in range(5, num_levels):
            levels.append(f"Level_{i}")
    
    # Sample tokens for clustering analysis
    max_samples = min(1000, tokens.shape[0])
    if tokens.shape[0] > max_samples:
        indices = np.random.choice(tokens.shape[0], size=max_samples, replace=False)
        samples = tokens[indices]
    else:
        samples = tokens
    
    # Analyze data to determine splat counts
    # Use clustering to find natural groupings
    splat_counts = []
    
    # Sample different clustering levels
    for i, level in enumerate(levels):
        # Skip token level (no clustering needed)
        if i == 0:
            # Token level gets most splats
            base_count = min(100, tokens.shape[0] // 10)
            splat_counts.append(base_count)
            continue
        
        # For other levels, determine by clustering
        if clustering_method == "spectral" and tokens.shape[1] < 50:
            try:
                # Create spectral clustering with auto-tuning of clusters
                n_clusters_range = range(3, min(20, samples.shape[0] // 5))
                best_score = -1
                best_n_clusters = 5  # Default
                
                for n_clusters in n_clusters_range:
                    clustering = SpectralClustering(
                        n_clusters=n_clusters,
                        random_state=random_seed,
                        n_init=2  # Reduced for speed
                    )
                    labels = clustering.fit_predict(samples)
                    
                    # Score based on silhouette or simple variance ratio
                    from sklearn.metrics import silhouette_score
                    try:
                        score = silhouette_score(samples, labels)
                        if score > best_score:
                            best_score = score
                            best_n_clusters = n_clusters
                    except:
                        # Fallback if silhouette fails
                        continue
                
                # Adjust based on level depth
                level_factor = 1.0 / (i + 1)
                level_count = int(best_n_clusters * level_factor * 2)
                splat_counts.append(max(5, level_count))
                
            except Exception as e:
                # Fallback if clustering fails
                print(f"Clustering failed: {e}")
                base_count = splat_counts[0] if splat_counts else 100
                splat_counts.append(max(5, base_count // (2**i)))
        else:
            # Simple heuristic approach
            base_count = splat_counts[0] if splat_counts else 100
            splat_counts.append(max(5, base_count // (2**i)))
    
    # Calculate level weights (more weight to lower levels for most cases)
    total = sum(1 / (i + 1) for i in range(len(levels)))
    level_weights = [1 / (i + 1) / total for i in range(len(levels))]
    
    # Create and return hierarchy
    return Hierarchy(
        levels=levels,
        init_splats_per_level=splat_counts,
        level_weights=level_weights
    )
