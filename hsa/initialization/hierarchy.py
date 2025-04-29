"""
Hierarchy management module for Hierarchical Splat Attention (HSA) initialization.

This module handles the establishment and maintenance of hierarchical relationships
between splats at different levels, as well as adaptive hierarchy creation.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _establish_parent_child_relationships(
    registry: Any,  # SplatRegistry
    child_level: str, 
    parent_level: str
) -> None:
    """
    Establish parent-child relationships between splats at adjacent levels.
    
    Args:
        registry: The splat registry
        child_level: The name of the child level
        parent_level: The name of the parent level
    """
    child_splats = registry.get_splats_at_level(child_level)
    parent_splats = registry.get_splats_at_level(parent_level)
    
    # Convert to lists for indexing
    parent_list = list(parent_splats)
    
    # If no parents, can't establish relationships
    if not parent_list:
        return
    
    # For each child, find the closest parent
    for child in child_splats:
        min_distance = float('inf')
        closest_parent = None
        
        for parent in parent_list:
            # Compute Mahalanobis distance using the parent's covariance
            # This gives more weight to the parent's natural shape
            try:
                diff = child.position - parent.position
                distance = np.sqrt(diff @ parent.covariance_inverse @ diff)
            except:
                # Fallback to Euclidean distance if Mahalanobis fails
                distance = np.linalg.norm(child.position - parent.position)
            
            if distance < min_distance:
                min_distance = distance
                closest_parent = parent
        
        # Establish relationship
        if closest_parent:
            closest_parent.add_child(child)

def create_adaptive_hierarchy(
    tokens: np.ndarray,
    base_level_count: int = 3,
    min_splats_per_level: int = 16,
    max_splats_per_level: int = 256
) -> Dict[str, Any]:
    """
    Create an adaptive hierarchy configuration based on token distribution.
    
    Args:
        tokens: Token embeddings
        base_level_count: Base number of hierarchical levels
        min_splats_per_level: Minimum splats per level
        max_splats_per_level: Maximum splats per level
        
    Returns:
        Hierarchy configuration dictionary
    """
    logger.info(f"Creating adaptive hierarchy from {tokens.shape[0]} tokens")
    
    # Default hierarchy configuration
    hierarchy_config = {
        "levels": ["Token", "Phrase", "Document"],
        "init_splats_per_level": [64, 32, 16],
        "level_weights": [0.5, 0.3, 0.2]
    }
    
    try:
        # Calculate complexity of token space
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Import analysis function without circular dependency
        from .analysis import analyze_embedding_space
        
        # Analyze embedding space to estimate complexity
        analysis = analyze_embedding_space(tokens)
        
        if analysis["clusters"] is not None:
            # Use cluster information to guide hierarchy
            n_clusters = analysis["clusters"]["n_clusters"]
            
            # Adjust number of levels based on complexity
            level_count = base_level_count
            if n_clusters > 8:
                level_count += 1  # Add an extra level for complex spaces
            
            # Create level names
            if level_count == 2:
                levels = ["Token", "Document"]
            elif level_count == 3:
                levels = ["Token", "Phrase", "Document"]
            elif level_count == 4:
                levels = ["Token", "Phrase", "Section", "Document"]
            else:
                levels = ["Token", "Phrase", "Section", "Document", "Global"]
            
            # Calculate splats per level based on token count and clusters
            token_count = tokens.shape[0]
            base_splat_count = min(max(n_clusters * 2, min_splats_per_level), 
                                  max(token_count // 10, min_splats_per_level))
            base_splat_count = min(base_splat_count, max_splats_per_level)
            
            # Calculate decreasing counts for higher levels
            init_splats_per_level = []
            for i in range(level_count):
                level_splats = max(min_splats_per_level, int(base_splat_count / (2**i)))
                init_splats_per_level.append(level_splats)
            
            # Calculate level weights (more weight to lower levels)
            level_weights = []
            total_weight = sum(1 / (i + 1) for i in range(level_count))
            for i in range(level_count):
                weight = (1 / (i + 1)) / total_weight
                level_weights.append(weight)
            
            # Update the hierarchy configuration
            hierarchy_config["levels"] = levels
            hierarchy_config["init_splats_per_level"] = init_splats_per_level
            hierarchy_config["level_weights"] = level_weights
        
        logger.info(f"Created adaptive hierarchy: {hierarchy_config}")
    
    except Exception as e:
        logger.warning(f"Failed to create adaptive hierarchy: {e}")
        logger.info("Using default hierarchy configuration")
    
    return hierarchy_config

def adapt_hierarchy_for_sequence_length(
    sequence_length: int,
    base_hierarchy: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Adapt hierarchy configuration based on sequence length.
    
    Args:
        sequence_length: Length of the sequence
        base_hierarchy: Optional base hierarchy to adapt
        
    Returns:
        Adapted hierarchy configuration
    """
    # Start with default hierarchy if none provided
    if base_hierarchy is None:
        base_hierarchy = {
            "levels": ["Token", "Phrase", "Document"],
            "init_splats_per_level": [64, 32, 16],
            "level_weights": [0.5, 0.3, 0.2]
        }
    
    # Create a copy to modify
    adapted_hierarchy = base_hierarchy.copy()
    adapted_hierarchy["init_splats_per_level"] = base_hierarchy["init_splats_per_level"].copy()
    adapted_hierarchy["level_weights"] = base_hierarchy["level_weights"].copy()
    
    # For very short sequences, reduce splat counts
    if sequence_length < 64:
        # Scale down proportionally
        scale_factor = max(0.25, sequence_length / 64)
        adapted_hierarchy["init_splats_per_level"] = [
            max(4, int(count * scale_factor))
            for count in adapted_hierarchy["init_splats_per_level"]
        ]
    
    # For long sequences, increase splat counts and add levels if needed
    elif sequence_length > 1024:
        # Scale up proportionally, but sub-linearly
        scale_factor = 1.0 + 0.3 * np.log2(sequence_length / 1024)
        
        adapted_hierarchy["init_splats_per_level"] = [
            min(512, int(count * scale_factor))
            for count in adapted_hierarchy["init_splats_per_level"]
        ]
        
        # For very long sequences, consider adding an extra level
        if sequence_length > 4096 and len(adapted_hierarchy["levels"]) < 5:
            if "Global" not in adapted_hierarchy["levels"]:
                adapted_hierarchy["levels"].append("Global")
                adapted_hierarchy["init_splats_per_level"].append(8)
                
                # Recalculate weights (more weight to lower levels)
                level_count = len(adapted_hierarchy["levels"])
                total_weight = sum(1 / (i + 1) for i in range(level_count))
                adapted_hierarchy["level_weights"] = [
                    (1 / (i + 1)) / total_weight for i in range(level_count)
                ]
    
    logger.info(f"Adapted hierarchy for sequence length {sequence_length}: {adapted_hierarchy}")
    return adapted_hierarchy

def optimize_hierarchy_for_task(
    task_type: str,
    base_hierarchy: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Optimize hierarchy configuration for specific tasks.
    
    Args:
        task_type: Type of task ('translation', 'summarization', etc.)
        base_hierarchy: Optional base hierarchy to optimize
        
    Returns:
        Optimized hierarchy configuration
    """
    # Start with default hierarchy if none provided
    if base_hierarchy is None:
        base_hierarchy = {
            "levels": ["Token", "Phrase", "Document"],
            "init_splats_per_level": [64, 32, 16],
            "level_weights": [0.5, 0.3, 0.2]
        }
    
    # Create a copy to modify
    optimized_hierarchy = base_hierarchy.copy()
    optimized_hierarchy["level_weights"] = base_hierarchy["level_weights"].copy()
    
    # Optimize based on task type
    if task_type == "translation":
        # For translation, emphasize token and phrase levels
        if "Token" in optimized_hierarchy["levels"] and "Phrase" in optimized_hierarchy["levels"]:
            token_idx = optimized_hierarchy["levels"].index("Token")
            phrase_idx = optimized_hierarchy["levels"].index("Phrase")
            
            # Increase weight for token and phrase levels
            weights = optimized_hierarchy["level_weights"]
            # Apply adjustment but maintain weight sum of 1.0
            total_adjustment = 0.1
            weights[token_idx] += total_adjustment * 0.6
            weights[phrase_idx] += total_adjustment * 0.4
            
            # Reduce other weights proportionally
            other_indices = [i for i in range(len(weights)) if i != token_idx and i != phrase_idx]
            for idx in other_indices:
                weights[idx] *= (1.0 - total_adjustment)
            
            # Normalize weights
            total = sum(weights)
            optimized_hierarchy["level_weights"] = [w / total for w in weights]
    
    elif task_type == "summarization":
        # For summarization, emphasize document and section levels
        higher_level_boost = False
        for level in ["Document", "Section"]:
            if level in optimized_hierarchy["levels"]:
                idx = optimized_hierarchy["levels"].index(level)
                # Increase weight for document/section level
                weights = optimized_hierarchy["level_weights"]
                weights[idx] *= 1.3
                higher_level_boost = True
        
        # Normalize weights if changes were made
        if higher_level_boost:
            weights = optimized_hierarchy["level_weights"]
            total = sum(weights)
            optimized_hierarchy["level_weights"] = [w / total for w in weights]
    
    elif task_type == "code_completion":
        # For code completion, use code-specific levels if they don't exist
        code_levels = ["Token", "Expression", "Block", "Function"]
        if set(optimized_hierarchy["levels"]) != set(code_levels):
            # Replace with code-specific hierarchy
            optimized_hierarchy["levels"] = code_levels
            
            # Adjust splat counts if needed
            if len(optimized_hierarchy["init_splats_per_level"]) != len(code_levels):
                # Start with default counts and scale appropriately
                base_count = 64
                optimized_hierarchy["init_splats_per_level"] = [
                    int(base_count / (2**i)) for i in range(len(code_levels))
                ]
            
            # Recalculate weights with emphasis on token and expression levels
            total_weight = sum(1 / (i + 1) for i in range(len(code_levels)))
            base_weights = [(1 / (i + 1)) / total_weight for i in range(len(code_levels))]
            
            # Boost token and expression weights
            boost_factor = 1.2
            base_weights[0] *= boost_factor  # Token
            base_weights[1] *= boost_factor  # Expression
            
            # Normalize
            total = sum(base_weights)
            optimized_hierarchy["level_weights"] = [w / total for w in base_weights]
    
    logger.info(f"Optimized hierarchy for task {task_type}: {optimized_hierarchy}")
    return optimized_hierarchy

def rebuild_hierarchy_relationships(
    registry: Any,  # SplatRegistry
    force_rebuild: bool = False
) -> None:
    """
    Rebuild all parent-child relationships in a splat registry.
    
    Useful after major changes to splat positions or structure.
    
    Args:
        registry: The splat registry
        force_rebuild: Whether to force full rebuild even if relationships exist
    """
    # Check if rebuild is needed
    if not force_rebuild:
        # Count existing relationships
        child_count = 0
        for level in registry.hierarchy.levels[:-1]:  # Skip top level
            for splat in registry.get_splats_at_level(level):
                if splat.parent is not None:
                    child_count += 1
        
        # If we have at least some relationships, no need to rebuild
        if child_count > 0:
            logger.info(f"Found {child_count} existing relationships, skipping rebuild")
            return
    
    # Clear all existing relationships
    for splat in registry.splats.values():
        # Clear children
        splat.children.clear()
        # Clear parent
        splat.parent = None
    
    # Rebuild level by level
    for level_idx, level in enumerate(registry.hierarchy.levels[:-1]):  # Skip top level
        parent_level = registry.hierarchy.levels[level_idx + 1]
        
        logger.info(f"Rebuilding relationships between {level} and {parent_level}")
        
        # Establish relationships
        _establish_parent_child_relationships(registry, level, parent_level)
    
    # Count relationships after rebuild
    child_count = 0
    for level in registry.hierarchy.levels[:-1]:  # Skip top level
        for splat in registry.get_splats_at_level(level):
            if splat.parent is not None:
                child_count += 1
    
    logger.info(f"Rebuilt hierarchy relationships: {child_count} child-parent links")
