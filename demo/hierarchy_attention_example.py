"""
Example of visualizing attention patterns across hierarchy levels in HSA.

This script demonstrates how to visualize attention patterns at different 
hierarchical levels, showing how attention flows through the hierarchy.
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

# Import visualization
from hsa.splat_visualization import SplatVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_structured_registry(dim: int = 2) -> Tuple[SplatRegistry, np.ndarray]:
    """Create a structured registry with clear hierarchical relationships.
    
    Args:
        dim: Dimensionality of the embedding space
        
    Returns:
        Tuple of (registry, tokens)
    """
    # Create hierarchy with 3 levels
    hierarchy = Hierarchy(
        levels=["token", "phrase", "document"],
        init_splats_per_level=[12, 4, 1],
        level_weights=[0.5, 0.3, 0.2]
    )
    
    # Create registry
    registry = SplatRegistry(hierarchy=hierarchy, embedding_dim=dim)
    
    # Create tokens distributed in 4 clusters (represent 4 phrases)
    n_tokens = 120
    n_clusters = 4
    tokens_per_cluster = n_tokens // n_clusters
    
    tokens = []
    
    if dim == 2:
        # 2D centers for the 4 clusters
        centers = [
            np.array([2.0, 2.0]),    # Top right
            np.array([-2.0, 2.0]),   # Top left
            np.array([2.0, -2.0]),   # Bottom right
            np.array([-2.0, -2.0])   # Bottom left
        ]
        
        # Create tokens for each cluster
        for i in range(n_clusters):
            for j in range(tokens_per_cluster):
                # Add random noise
                noise = np.random.normal(0, 0.5, dim)
                token = centers[i] + noise
                tokens.append(token)
    else:
        # For other dimensions, create structured tokens in a different way
        # For example, use PCA components as directions
        for i in range(n_clusters):
            direction = np.random.normal(0, 1.0, dim)
            direction = direction / np.linalg.norm(direction)
            
            for j in range(tokens_per_cluster):
                # Create token along direction with noise
                distance = 2.0 + np.random.normal(0, 0.2)
                token = direction * distance + np.random.normal(0, 0.5, dim)
                tokens.append(token)
    
    # Convert to numpy array
    tokens = np.array(tokens)
    
    # Manually create structured splats instead of using initialize_splats
    # This gives us more control over the hierarchical relationships
    
    # Document-level splat (covers everything)
    document_splat = Splat(
        dim=dim,
        position=np.zeros(dim),
        covariance=np.eye(dim) * 6.0,  # Large covariance to cover all tokens
        amplitude=1.0,
        level="document",
        id="document_splat"
    )
    registry.register(document_splat)
    
    # Phrase-level splats (one per cluster)
    phrase_splats = []
    for i in range(n_clusters):
        if dim == 2:
            position = centers[i]
        else:
            # Use cluster center
            cluster_start = i * tokens_per_cluster
            cluster_end = (i + 1) * tokens_per_cluster
            position = np.mean(tokens[cluster_start:cluster_end], axis=0)
        
        phrase_splat = Splat(
            dim=dim,
            position=position,
            covariance=np.eye(dim) * 1.5,  # Medium covariance
            amplitude=1.0,
            level="phrase",
            parent=document_splat,
            id=f"phrase_splat_{i}"
        )
        registry.register(phrase_splat)
        phrase_splats.append(phrase_splat)
    
    # Token-level splats (3 per phrase)
    token_splats = []
    for i, phrase_splat in enumerate(phrase_splats):
        # Sub-divide phrase into 3 token regions
        cluster_start = i * tokens_per_cluster
        tokens_per_token_splat = tokens_per_cluster // 3
        
        for j in range(3):
            # Get tokens for this sub-region
            sub_start = cluster_start + j * tokens_per_token_splat
            sub_end = sub_start + tokens_per_token_splat
            sub_tokens = tokens[sub_start:sub_end]
            
            # Use mean position
            position = np.mean(sub_tokens, axis=0)
            
            token_splat = Splat(
                dim=dim,
                position=position,
                covariance=np.eye(dim) * 0.5,  # Small covariance
                amplitude=1.0,
                level="token",
                parent=phrase_splat,
                id=f"token_splat_{i}_{j}"
            )
            registry.register(token_splat)
            token_splats.append(token_splat)
    
    return registry, tokens


def compute_attention_by_level(
    registry: SplatRegistry, 
    tokens: np.ndarray
) -> Dict[str, np.ndarray]:
    """Compute attention matrices separately for each hierarchy level.
    
    Args:
        registry: SplatRegistry to use
        tokens: Token embeddings
        
    Returns:
        Dictionary mapping level names to attention matrices
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
    attention_result = attention_computer.compute_attention_with_details(
        tokens=tokens,
        splat_registry=registry
    )
    
    # Extract level contributions
    return attention_result.level_contributions


def visualize_attention_hierarchy(
    registry: SplatRegistry,
    tokens: np.ndarray,
    attention_matrices: Dict[str, np.ndarray],
    save_dir: Optional[str] = None
):
    """Visualize attention patterns across hierarchy levels.
    
    Args:
        registry: SplatRegistry to visualize
        tokens: Token embeddings
        attention_matrices: Dictionary mapping level names to attention matrices
        save_dir: Directory to save visualizations (if None, just display)
    """
    # Create directory if needed
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create visualizer
    visualizer = SplatVisualizer(figsize=(15, 10))
    
    # 1. Basic registry visualization
    logger.info("1. Registry visualization...")
    fig = visualizer.visualize_registry(
        registry=registry,
        tokens=tokens,
        title="Hierarchical Splat Attention - Registry Overview",
        save_path=os.path.join(save_dir, "registry.png") if save_dir else None
    )
    plt.show()
    
    # 2. Hierarchy levels visualization
    logger.info("2. Hierarchy levels visualization...")
    fig = visualizer.visualize_hierarchy_levels(
        registry=registry,
        tokens=tokens,
        title="Hierarchical Levels of Splat Attention",
        save_path=os.path.join(save_dir, "hierarchy_levels.png") if save_dir else None
    )
    plt.show()
    
    # 3. Visualize attention matrices for each level
    logger.info("3. Visualizing attention matrices by level...")
    
    # Create figure with subplots
    hierarchy = registry.hierarchy
    levels = hierarchy.levels
    n_levels = len(levels)
    
    if n_levels > 0:
        # Setup figure
        fig, axes = plt.subplots(1, n_levels, figsize=(5 * n_levels, 5))
        if n_levels == 1:
            axes = [axes]  # Make it iterable
        
        # Plot attention matrix for each level
        for i, level in enumerate(levels):
            ax = axes[i]
            
            if level in attention_matrices:
                # Get attention matrix for this level
                attention = attention_matrices[level]
                
                # Plot as heatmap
                im = ax.imshow(attention, cmap='viridis', interpolation='nearest')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Attention Value')
                
                # Set title and labels
                ax.set_title(f"Level: {level}")
                ax.set_xlabel('Target Token')
                if i == 0:
                    ax.set_ylabel('Source Token')
                
                # For large matrices, don't show ticks
                if attention.shape[0] > 20:
                    ax.set_xticks([])
                    ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, f"No attention data for level: {level}", 
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Level: {level}")
        
        # Set overall title
        fig.suptitle("Attention Matrices by Hierarchy Level", fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=0.85)
        
        # Save if requested
        if save_dir:
            plt.savefig(os.path.join(save_dir, "attention_matrices.png"), dpi=150, bbox_inches='tight')
        
        plt.show()
    
    # 4. Visualize attention flow for each level
    logger.info("4. Visualizing attention flow by level...")
    
    # Calculate a representative subset of tokens for visualization
    subset_size = min(10, tokens.shape[0])
    subset_indices = list(range(0, tokens.shape[0], tokens.shape[0] // subset_size))[:subset_size]
    
    # Plot attention flow for each level
    for level in levels:
        if level in attention_matrices:
            logger.info(f"Visualizing attention flow for level: {level}")
            
            # Get attention matrix for this level
            attention = attention_matrices[level]
            
            # Get splats at this level
            level_splats = list(registry.get_splats_at_level(level))
            highlight_splats = set([splat.id for splat in level_splats[:3]])  # Highlight first 3
            
            # Visualize attention flow
            fig = visualizer.visualize_attention_flow(
                registry=registry,
                attention_matrix=attention,
                tokens=tokens,
                token_indices=subset_indices,
                title=f"Attention Flow - Level: {level}",
                save_path=os.path.join(save_dir, f"attention_flow_{level}.png") if save_dir else None
            )
            plt.show()
            
            # Visualize registry with highlighted splats at this level
            fig = visualizer.visualize_registry(
                registry=registry,
                tokens=tokens,
                highlight_splats=highlight_splats,
                title=f"Highlighted Splats - Level: {level}",
                save_path=os.path.join(save_dir, f"highlighted_{level}.png") if save_dir else None
            )
            plt.show()


def run_example(save_dir: Optional[str] = None):
    """Run the complete example.
    
    Args:
        save_dir: Directory to save visualizations (if None, just display)
    """
    # Create structured registry
    logger.info("Creating structured registry...")
    registry, tokens = create_structured_registry(dim=2)
    
    # Compute attention matrices by level
    logger.info("Computing attention by level...")
    attention_matrices = compute_attention_by_level(registry, tokens)
    
    # Visualize
    logger.info("Visualizing attention hierarchy...")
    visualize_attention_hierarchy(registry, tokens, attention_matrices, save_dir)
    
    logger.info("Example completed.")


if __name__ == "__main__":
    # Run example and save to "hierarchy_examples" directory
    run_example(save_dir="hierarchy_examples")
