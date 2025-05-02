"""
Demonstration script for using the Hierarchical Splat Attention (HSA) visualization tools.

This script shows how to visualize splats, attention patterns, and adaptation operations
using the SplatVisualizer class.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Set
import os
import logging

# Import HSA components
from hsa.splat import Splat
from hsa.hierarchy import Hierarchy
from hsa.registry import SplatRegistry
from hsa.dense_attention import DenseAttentionComputer
from hsa.attention_interface import AttentionConfig
from hsa.adaptation_types import AdaptationType, AdaptationResult
from hsa.adaptation_metrics_base import AdaptationMetricsComputer
from hsa.adaptation_controller import AdaptationController

# Import visualization
from hsa.splat_visualization import SplatVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_demo_registry(dim: int = 2, n_tokens: int = 50) -> Tuple[SplatRegistry, np.ndarray]:
    """Create a demo registry and token embeddings for visualization.
    
    Args:
        dim: Dimensionality of the embedding space
        n_tokens: Number of token embeddings to generate
        
    Returns:
        Tuple of (registry, tokens)
    """
    # Create hierarchy with 3 levels
    hierarchy = Hierarchy(
        levels=["token", "phrase", "document"],
        init_splats_per_level=[10, 5, 2],
        level_weights=[0.5, 0.3, 0.2]
    )
    
    # Create registry
    registry = SplatRegistry(hierarchy=hierarchy, embedding_dim=dim)
    
    # Generate random token embeddings
    tokens = np.random.normal(0, 1.0, (n_tokens, dim))
    
    # Initialize splats based on token distribution
    registry.initialize_splats(tokens)
    
    return registry, tokens


def create_demo_adaptation_results(registry: SplatRegistry) -> List[AdaptationResult]:
    """Create demo adaptation results for visualization.
    
    Args:
        registry: SplatRegistry to create results for
        
    Returns:
        List of AdaptationResult objects
    """
    # Get all splats
    all_splats = registry.get_all_splats()
    
    if len(all_splats) < 5:
        logger.warning("Not enough splats for demo adaptation results")
        return []
    
    # Create demo results
    results = []
    
    # Mitosis example
    mitosis_splat = all_splats[0]
    child1_id = f"child1_of_{mitosis_splat.id}"
    child2_id = f"child2_of_{mitosis_splat.id}"
    
    mitosis_result = AdaptationResult(
        success=True,
        adaptation_type=AdaptationType.MITOSIS,
        original_splat_id=mitosis_splat.id,
        new_splat_ids=[child1_id, child2_id],
        removed_splat_ids=[mitosis_splat.id],
        metrics_before=None,
        metrics_after=None,
        message=f"Split splat {mitosis_splat.id} into {child1_id} and {child2_id}"
    )
    
    results.append(mitosis_result)
    
    # Birth example
    birth_id = "new_splat_id"
    
    birth_result = AdaptationResult(
        success=True,
        adaptation_type=AdaptationType.BIRTH,
        original_splat_id="",
        new_splat_ids=[birth_id],
        removed_splat_ids=[],
        metrics_before=None,
        metrics_after=None,
        message=f"Created new splat {birth_id}"
    )
    
    results.append(birth_result)
    
    # Death example
    death_splat = all_splats[1]
    
    death_result = AdaptationResult(
        success=True,
        adaptation_type=AdaptationType.DEATH,
        original_splat_id=death_splat.id,
        new_splat_ids=[],
        removed_splat_ids=[death_splat.id],
        metrics_before=None,
        metrics_after=None,
        message=f"Removed splat {death_splat.id}"
    )
    
    results.append(death_result)
    
    # Merge example
    merge_splat1 = all_splats[2]
    merge_splat2 = all_splats[3]
    merged_id = f"merged_{merge_splat1.id}_{merge_splat2.id}"
    
    merge_result = AdaptationResult(
        success=True,
        adaptation_type=AdaptationType.MERGE,
        original_splat_id=merge_splat1.id,
        new_splat_ids=[merged_id],
        removed_splat_ids=[merge_splat1.id, merge_splat2.id],
        metrics_before=None,
        metrics_after=None,
        message=f"Merged splats {merge_splat1.id} and {merge_splat2.id} into {merged_id}"
    )
    
    results.append(merge_result)
    
    # Adjust example
    adjust_splat = all_splats[4]
    
    adjust_result = AdaptationResult(
        success=True,
        adaptation_type=AdaptationType.ADJUST,
        original_splat_id=adjust_splat.id,
        new_splat_ids=[],
        removed_splat_ids=[],
        metrics_before=None,
        metrics_after=None,
        message=f"Adjusted parameters for splat {adjust_splat.id}"
    )
    
    results.append(adjust_result)
    
    return results


def create_demo_attention_matrix(n_tokens: int) -> np.ndarray:
    """Create a demo attention matrix for visualization.
    
    Args:
        n_tokens: Number of tokens
        
    Returns:
        Attention matrix of shape [n_tokens, n_tokens]
    """
    # Initialize with small values
    attention = np.random.uniform(0, 0.1, (n_tokens, n_tokens))
    
    # Add some structure (e.g., locality patterns)
    for i in range(n_tokens):
        # Self-attention
        attention[i, i] = 0.8
        
        # Attention to nearby tokens
        for j in range(max(0, i-5), min(n_tokens, i+6)):
            attention[i, j] = max(attention[i, j], 0.5 * np.exp(-0.1 * abs(i - j)))
        
        # Some random strong connections
        random_targets = np.random.choice(n_tokens, 3, replace=False)
        for j in random_targets:
            attention[i, j] = max(attention[i, j], 0.7)
    
    # Normalize rows
    row_sums = attention.sum(axis=1, keepdims=True)
    attention = attention / row_sums
    
    return attention


def run_visualizations(save_dir: Optional[str] = None):
    """Run various visualizations and optionally save them.
    
    Args:
        save_dir: Directory to save visualizations to (if None, just displays)
    """
    # Create directory if needed
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create demo registry and tokens
    dim = 2  # Use 2D for easy visualization
    n_tokens = 50
    
    logger.info("Creating demo registry and tokens...")
    registry, tokens = create_demo_registry(dim, n_tokens)
    
    # Create demo attention matrix
    logger.info("Creating demo attention matrix...")
    attention_matrix = create_demo_attention_matrix(n_tokens)
    
    # Create demo adaptation results
    logger.info("Creating demo adaptation results...")
    adaptation_results = create_demo_adaptation_results(registry)
    
    # Create visualizer
    logger.info("Creating visualizer...")
    visualizer = SplatVisualizer(figsize=(12, 10))
    
    # Run visualizations
    logger.info("Running visualizations...")
    
    # 1. Basic registry visualization
    logger.info("1. Basic registry visualization")
    fig = visualizer.visualize_registry(
        registry=registry,
        tokens=tokens,
        title="Hierarchical Splat Attention - Registry Visualization",
        save_path=os.path.join(save_dir, "registry.png") if save_dir else None
    )
    plt.show()
    
    # 2. Highlight specific splats
    logger.info("2. Highlight specific splats")
    # Get a few random splats to highlight
    all_splats = registry.get_all_splats()
    highlight_splats = set(np.random.choice([s.id for s in all_splats], 
                                            min(3, len(all_splats)), 
                                            replace=False))
    
    fig = visualizer.visualize_registry(
        registry=registry,
        tokens=tokens,
        highlight_splats=highlight_splats,
        title="Highlighting Specific Splats",
        save_path=os.path.join(save_dir, "highlight.png") if save_dir else None
    )
    plt.show()
    
    # 3. Visualize hierarchy levels
    logger.info("3. Visualize hierarchy levels")
    fig = visualizer.visualize_hierarchy_levels(
        registry=registry,
        tokens=tokens,
        title="Hierarchical Levels of Splat Attention",
        save_path=os.path.join(save_dir, "hierarchy_levels.png") if save_dir else None
    )
    plt.show()
    
    # 4. Visualize attention flow
    logger.info("4. Visualize attention flow")
    fig = visualizer.visualize_attention_flow(
        registry=registry,
        attention_matrix=attention_matrix,
        tokens=tokens,
        token_indices=list(range(10)),  # Use first 10 tokens
        title="Attention Flow through Splats",
        save_path=os.path.join(save_dir, "attention_flow.png") if save_dir else None
    )
    plt.show()
    
    # 5. Visualize adaptation operations
    if adaptation_results:
        logger.info("5. Visualize adaptation operations")
        for i, result in enumerate(adaptation_results):
            fig = visualizer.visualize_adaptation_operation(
                result=result,
                registry=registry,
                tokens=tokens,
                save_path=os.path.join(save_dir, f"adaptation_{i}.png") if save_dir else None
            )
            plt.show()
    
    # 6. Visualize adaptation history
    if adaptation_results:
        logger.info("6. Visualize adaptation history")
        fig = visualizer.visualize_adaptation_history(
            history=adaptation_results,
            registry=registry,
            tokens=tokens,
            save_path=os.path.join(save_dir, "adaptation_history.png") if save_dir else None
        )
        plt.show()
    
    # 7. Visualize splat attention map
    logger.info("7. Visualize splat attention map")
    if all_splats:
        # Get a random splat
        splat = np.random.choice(all_splats)
        
        # Create a simple attention map for this splat
        splat_attention = np.zeros((n_tokens, n_tokens))
        for i in range(n_tokens):
            for j in range(n_tokens):
                # Simple Gaussian attention based on distance from splat position
                dist_i = np.sum((tokens[i] - splat.position) ** 2)
                dist_j = np.sum((tokens[j] - splat.position) ** 2)
                splat_attention[i, j] = np.exp(-0.5 * (dist_i + dist_j))
        
        fig = visualizer.visualize_splat_attention_map(
            splat=splat,
            tokens=tokens,
            attention_map=splat_attention,
            token_indices=list(range(10)),  # Use first 10 tokens
            title=f"Attention Map for Splat {splat.id}",
            save_path=os.path.join(save_dir, "splat_attention_map.png") if save_dir else None
        )
        plt.show()
    
    logger.info("All visualizations completed.")


if __name__ == "__main__":
    # Run visualizations and save to "visualizations" directory
    run_visualizations(save_dir="visualizations")
