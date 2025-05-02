"""
Example of visualizing real adaptation operations with HSA.

This script demonstrates the visualization of real adaptation operations
by running actual mitosis, merge, birth, and death operations on splats.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from typing import List, Dict, Optional, Tuple, Set, Any

# Import HSA components
from hsa.splat import Splat
from hsa.hierarchy import Hierarchy
from hsa.registry import SplatRegistry
from hsa.dense_attention import DenseAttentionComputer
from hsa.attention_interface import AttentionConfig, AttentionResult
from hsa.adaptation_types import AdaptationType, AdaptationResult

# Import adaptation operations
from hsa.mitosis import perform_mitosis, identify_mitosis_candidates
from hsa.merge import perform_merge, find_merge_candidates
from hsa.birth import perform_birth, identify_empty_regions
from hsa.death import perform_death, identify_death_candidates

# Import visualization
from hsa.splat_visualization import SplatVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_tokens_and_registry(dim: int = 2, n_tokens: int = 100) -> Tuple[SplatRegistry, np.ndarray]:
    """Create tokens and initialize registry.
    
    Args:
        dim: Dimensionality of the embedding space
        n_tokens: Number of tokens
        
    Returns:
        Tuple of (registry, tokens)
    """
    # Create structured token distribution for better visualization
    tokens = np.zeros((n_tokens, dim))
    
    if dim == 2:
        # Create 2D token distribution with clusters
        n_clusters = 4
        points_per_cluster = n_tokens // n_clusters
        remaining = n_tokens - (points_per_cluster * n_clusters)
        
        # Cluster centers
        centers = [
            np.array([1.0, 1.0]),    # Top right
            np.array([-1.0, 1.0]),   # Top left
            np.array([1.0, -1.0]),   # Bottom right
            np.array([-1.0, -1.0])   # Bottom left
        ]
        
        # Generate tokens in clusters
        idx = 0
        for i in range(n_clusters):
            for j in range(points_per_cluster):
                # Add noise to cluster center
                noise = np.random.normal(0, 0.3, dim)
                tokens[idx] = centers[i] + noise
                idx += 1
        
        # Add remaining tokens randomly
        for i in range(remaining):
            tokens[idx] = np.random.normal(0, 1.0, dim)
            idx += 1
            
    else:
        # For other dimensions, just use random distribution
        tokens = np.random.normal(0, 1.0, (n_tokens, dim))
    
    # Create hierarchy
    hierarchy = Hierarchy(
        levels=["token", "phrase", "document"],
        init_splats_per_level=[15, 8, 3],
        level_weights=[0.5, 0.3, 0.2]
    )
    
    # Create registry
    registry = SplatRegistry(hierarchy=hierarchy, embedding_dim=dim)
    
    # Initialize splats based on token distribution
    registry.initialize_splats(tokens)
    
    return registry, tokens


def compute_attention(registry: SplatRegistry, tokens: np.ndarray) -> AttentionResult:
    """Compute attention matrix and details.
    
    Args:
        registry: SplatRegistry to use
        tokens: Token embeddings
        
    Returns:
        AttentionResult with matrix and contribution details
    """
    # Create attention computer
    attention_computer = DenseAttentionComputer(
        config=AttentionConfig(
            normalize_levels=True,
            normalize_rows=True,
            causal=False
        )
    )
    
    # Compute attention with details
    return attention_computer.compute_attention_with_details(
        tokens=tokens,
        splat_registry=registry
    )


def run_adaptation_cycle(
    registry: SplatRegistry,
    tokens: np.ndarray,
    attention_result: AttentionResult
) -> List[AdaptationResult]:
    """Run a complete adaptation cycle with all operations.
    
    Args:
        registry: SplatRegistry to adapt
        tokens: Token embeddings
        attention_result: Result of attention computation
        
    Returns:
        List of adaptation results
    """
    results = []
    
    # 1. Identify mitosis candidates
    logger.info("Identifying mitosis candidates...")
    mitosis_candidates = identify_mitosis_candidates(registry)
    
    if mitosis_candidates:
        # Take top candidate
        splat, activation, variance = mitosis_candidates[0]
        logger.info(f"Performing mitosis on splat {splat.id} (activation={activation:.3f}, variance={variance:.3f})")
        
        # Perform mitosis
        new_pair = perform_mitosis(registry, splat.id)
        
        if new_pair:
            results.append(AdaptationResult(
                success=True,
                adaptation_type=AdaptationType.MITOSIS,
                original_splat_id=splat.id,
                new_splat_ids=[new_pair[0].id, new_pair[1].id],
                removed_splat_ids=[splat.id],
                metrics_before=None,
                metrics_after=None,
                message=f"Split splat {splat.id} into {new_pair[0].id} and {new_pair[1].id}"
            ))
    
    # 2. Identify merge candidates
    logger.info("Identifying merge candidates...")
    merge_candidates = find_merge_candidates(registry)
    
    if merge_candidates:
        # Take top candidate
        splat_a, splat_b, similarity = merge_candidates[0]
        logger.info(f"Performing merge on splats {splat_a.id} and {splat_b.id} (similarity={similarity:.3f})")
        
        # Perform merge
        merged_splat = perform_merge(registry, splat_a.id, splat_b.id)
        
        if merged_splat:
            results.append(AdaptationResult(
                success=True,
                adaptation_type=AdaptationType.MERGE,
                original_splat_id=splat_a.id,
                new_splat_ids=[merged_splat.id],
                removed_splat_ids=[splat_a.id, splat_b.id],
                metrics_before=None,
                metrics_after=None,
                message=f"Merged splats {splat_a.id} and {splat_b.id} into {merged_splat.id}"
            ))
    
    # 3. Identify birth opportunities
    logger.info("Identifying empty regions for birth...")
    empty_regions = identify_empty_regions(registry, tokens)
    
    if empty_regions:
        # Take first region
        position = empty_regions[0]
        level = registry.hierarchy.levels[0]  # Use lowest level
        
        logger.info(f"Performing birth at position {position} for level {level}")
        
        # Perform birth
        new_splat = perform_birth(registry, level, position, tokens)
        
        if new_splat:
            results.append(AdaptationResult(
                success=True,
                adaptation_type=AdaptationType.BIRTH,
                original_splat_id="",
                new_splat_ids=[new_splat.id],
                removed_splat_ids=[],
                metrics_before=None,
                metrics_after=None,
                message=f"Created new splat {new_splat.id} at level {level}"
            ))
    
    # 4. Identify death candidates
    logger.info("Identifying death candidates...")
    death_candidates = identify_death_candidates(registry)
    
    if death_candidates:
        # Take top candidate
        splat, activation = death_candidates[0]
        logger.info(f"Performing death on splat {splat.id} (activation={activation:.3f})")
        
        # Perform death
        success = perform_death(registry, splat.id)
        
        if success:
            results.append(AdaptationResult(
                success=True,
                adaptation_type=AdaptationType.DEATH,
                original_splat_id=splat.id,
                new_splat_ids=[],
                removed_splat_ids=[splat.id],
                metrics_before=None,
                metrics_after=None,
                message=f"Removed splat {splat.id}"
            ))
    
    return results


def run_visualization_example(save_dir: Optional[str] = None):
    """Run a complete visualization example with real adaptation operations.
    
    Args:
        save_dir: Directory to save visualizations (if None, just display)
    """
    # Create directory if needed
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create tokens and registry
    dim = 2  # Use 2D for visualization
    n_tokens = 100
    
    logger.info("Creating tokens and registry...")
    registry, tokens = create_tokens_and_registry(dim, n_tokens)
    
    # Create visualizer
    visualizer = SplatVisualizer(figsize=(12, 10))
    
    # Initial visualization
    logger.info("Initial visualization...")
    fig = visualizer.visualize_registry(
        registry=registry,
        tokens=tokens,
        title="Initial State - Before Adaptation",
        save_path=os.path.join(save_dir, "initial.png") if save_dir else None
    )
    plt.show()
    
    # Compute attention
    logger.info("Computing attention...")
    attention_result = compute_attention(registry, tokens)
    
    # Run adaptation cycle
    logger.info("Running adaptation cycle...")
    adaptation_results = run_adaptation_cycle(registry, tokens, attention_result)
    
    # Visualize after adaptation
    logger.info("Visualizing after adaptation...")
    if adaptation_results:
        # Visualize each adaptation operation
        for i, result in enumerate(adaptation_results):
            logger.info(f"Visualizing {result.adaptation_type.name} operation...")
            fig = visualizer.visualize_adaptation_operation(
                result=result,
                registry=registry,
                tokens=tokens,
                save_path=os.path.join(save_dir, f"adaptation_{i}.png") if save_dir else None
            )
            plt.show()
        
        # Visualize hierarchy levels
        logger.info("Visualizing hierarchy levels...")
        fig = visualizer.visualize_hierarchy_levels(
            registry=registry,
            tokens=tokens,
            title="Hierarchy Levels After Adaptation",
            save_path=os.path.join(save_dir, "hierarchy_levels.png") if save_dir else None
        )
        plt.show()
        
        # Visualize attention flow
        logger.info("Visualizing attention flow...")
        # Recompute attention after adaptation
        attention_result = compute_attention(registry, tokens)
        
        fig = visualizer.visualize_attention_flow(
            registry=registry,
            attention_matrix=attention_result.attention_matrix,
            tokens=tokens,
            token_indices=list(range(10)),  # Use first 10 tokens
            title="Attention Flow After Adaptation",
            save_path=os.path.join(save_dir, "attention_flow.png") if save_dir else None
        )
        plt.show()
        
        # Visualize splat attention map
        if registry.get_all_splats():
            logger.info("Visualizing splat attention map...")
            # Get a random splat
            splat = np.random.choice(registry.get_all_splats())
            
            # Get its attention map
            splat_id = splat.id
            if splat_id in attention_result.splat_contributions:
                splat_attention = attention_result.splat_contributions[splat_id]
                
                fig = visualizer.visualize_splat_attention_map(
                    splat=splat,
                    tokens=tokens,
                    attention_map=splat_attention,
                    token_indices=list(range(10)),  # Use first 10 tokens
                    title=f"Attention Map for Splat {splat_id}",
                    save_path=os.path.join(save_dir, "splat_attention_map.png") if save_dir else None
                )
                plt.show()
    else:
        logger.info("No adaptation operations were performed.")
    
    logger.info("Example completed.")


if __name__ == "__main__":
    # Run example and save to "adaptation_examples" directory
    run_visualization_example(save_dir="adaptation_examples")
