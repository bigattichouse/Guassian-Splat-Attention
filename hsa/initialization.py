"""
Initialization module for Hierarchical Splat Attention (HSA).

This module provides interfaces for creating and initializing splats across hierarchical levels.
It serves as the main entry point for the initialization functionality, delegating to specialized
submodules for specific tasks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add submodule directory to path if it doesn't exist
# This is to ensure we can import from submodules even during development
current_dir = os.path.dirname(os.path.abspath(__file__))
initialization_dir = os.path.join(current_dir, 'initialization')
if os.path.exists(initialization_dir) and initialization_dir not in sys.path:
    sys.path.append(initialization_dir)

# Import core data structures
from hsa.data_structures import Splat, Hierarchy, SplatRegistry, ensure_positive_definite, sample_covariance_matrix

# Import from submodules
try:
    # Try importing from submodules first
    from hsa.initialization.core import HSAInitializer, initialize_splats as core_initialize_splats
    from hsa.initialization.core import reinitialize_splat as core_reinitialize_splat
    from hsa.initialization.tokenizer import initialize_from_tokenizer as tokenizer_initialize
    from hsa.initialization.tokenizer import initialize_from_chat_tokens as chat_tokens_initialize
    from hsa.initialization.analysis import analyze_embedding_space as analyze_space
    from hsa.initialization.hierarchy import create_adaptive_hierarchy as create_hierarchy
    
    logger.info("Successfully imported from submodules")
    USING_SUBMODULES = True

except ImportError as e:
    # If submodules are not available, fallback to this file's implementations
    logger.warning(f"Could not import from submodules, falling back to direct imports: {e}")
    USING_SUBMODULES = False

# Main entry point functions

def initialize_splats(
    tokens: np.ndarray,
    hierarchy_config: Dict[str, Any],
    tokenizer: Optional[Any] = None,
    n_neighbors: int = 10,
    affinity: str = 'nearest_neighbors',
    initialization_method: str = 'spectral',
    random_seed: Optional[int] = None
) -> SplatRegistry:
    """
    Main entry point function to initialize splats for HSA.
    
    Args:
        tokens: Token embeddings [sequence_length, embedding_dim]
        hierarchy_config: Configuration dictionary with hierarchy parameters
        tokenizer: Optional tokenizer for token-aware initialization
        n_neighbors: Number of neighbors for nearest neighbors affinity
        affinity: Affinity method for spectral clustering
        initialization_method: Method to use for initialization
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Registry containing all initialized splats
    """
    if USING_SUBMODULES:
        return core_initialize_splats(
            tokens=tokens,
            hierarchy_config=hierarchy_config,
            tokenizer=tokenizer,
            n_neighbors=n_neighbors,
            affinity=affinity,
            initialization_method=initialization_method,
            random_seed=random_seed
        )
    else:
        # Create a hierarchy object
        hierarchy = Hierarchy(
            levels=hierarchy_config['levels'],
            init_splats_per_level=hierarchy_config['init_splats_per_level'],
            level_weights=hierarchy_config['level_weights']
        )
        
        # Create initializer
        initializer = HSAInitializer(
            hierarchy=hierarchy,
            n_neighbors=n_neighbors,
            affinity=affinity,
            initialization_method=initialization_method,
            random_seed=random_seed
        )
        
        # Initialize splats
        registry = initializer.initialize_splats(tokens, tokenizer=tokenizer)
        
        return registry

def reinitialize_splat(
    splat: Splat, 
    data_points: np.ndarray
) -> Splat:
    """
    Convenience function to reinitialize a single splat.
    
    Args:
        splat: The splat to reinitialize
        data_points: Data points to use for reinitialization
        
    Returns:
        The reinitialized splat
    """
    if USING_SUBMODULES:
        return core_reinitialize_splat(splat, data_points)
    else:
        # Create temporary initializer
        initializer = HSAInitializer(
            hierarchy=Hierarchy(
                levels=[splat.level],
                init_splats_per_level=[1],
                level_weights=[1.0]
            )
        )
        
        # Reinitialize splat
        return initializer.reinitialize_splat(splat, data_points)

def initialize_from_tokenizer(
    tokenizer: Any,
    embedding_matrix: Optional[np.ndarray] = None,
    hierarchy_config: Optional[Dict[str, Any]] = None,
    sample_sentences: Optional[List[str]] = None,
    random_seed: Optional[int] = None
) -> SplatRegistry:
    """
    Initialize splats directly from a tokenizer, using its embedding matrix
    or by generating sample embeddings from text.
    
    Args:
        tokenizer: The tokenizer to use
        embedding_matrix: Optional embedding matrix from the model
        hierarchy_config: Optional hierarchy configuration
        sample_sentences: Optional sample sentences to generate token embeddings
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Registry containing all initialized splats
    """
    if USING_SUBMODULES:
        return tokenizer_initialize(
            tokenizer=tokenizer,
            embedding_matrix=embedding_matrix,
            hierarchy_config=hierarchy_config,
            sample_sentences=sample_sentences,
            random_seed=random_seed
        )
    else:
        # This would typically delegate to the tokenizer module implementation
        # For simplicity, we'll fall back to a basic initialization
        logger.warning("Tokenizer-aware initialization requires submodules - using basic initialization")
        
        # Create pseudo-embeddings
        embedding_dim = 64
        if hierarchy_config and 'embedding_dim' in hierarchy_config:
            embedding_dim = hierarchy_config['embedding_dim']
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create random embeddings
        token_embeddings = np.random.randn(100, embedding_dim)
        
        # Use standard initialization
        return initialize_splats(
            tokens=token_embeddings,
            hierarchy_config=hierarchy_config or {
                "levels": ["Token", "Phrase", "Document"],
                "init_splats_per_level": [64, 32, 16],
                "level_weights": [0.5, 0.3, 0.2]
            },
            random_seed=random_seed
        )

def initialize_from_chat_tokens(
    tokens: List[Any],
    embedding_fn: Optional[Callable] = None,
    hierarchy_config: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Any] = None,
    random_seed: Optional[int] = None
) -> SplatRegistry:
    """
    Initialize splats from tokens in a chat context.
    
    Args:
        tokens: The tokens from the chat
        embedding_fn: Optional function to convert tokens to embeddings
        hierarchy_config: Optional hierarchy configuration
        tokenizer: Optional tokenizer for additional context
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Registry containing all initialized splats
    """
    if USING_SUBMODULES:
        return chat_tokens_initialize(
            tokens=tokens,
            embedding_fn=embedding_fn,
            hierarchy_config=hierarchy_config,
            tokenizer=tokenizer,
            random_seed=random_seed
        )
    else:
        # This would typically delegate to the tokenizer module implementation
        # For simplicity, we'll fall back to a basic initialization
        logger.warning("Chat token initialization requires submodules - using basic initialization")
        
        # Create pseudo-embeddings
        embedding_dim = 64
        if hierarchy_config and 'embedding_dim' in hierarchy_config:
            embedding_dim = hierarchy_config['embedding_dim']
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create random embeddings based on token count
        token_embeddings = np.random.randn(len(tokens), embedding_dim)
        
        # Use standard initialization
        return initialize_splats(
            tokens=token_embeddings,
            hierarchy_config=hierarchy_config or {
                "levels": ["Token", "Phrase", "Document"],
                "init_splats_per_level": [64, 32, 16],
                "level_weights": [0.5, 0.3, 0.2]
            },
            random_seed=random_seed
        )

def analyze_embedding_space(
    tokens: np.ndarray,
    method: str = 'pca',
    n_components: int = 2
) -> Dict[str, Any]:
    """
    Analyze the embedding space to guide splat placement.
    
    Args:
        tokens: Token embeddings
        method: Dimensionality reduction method ('pca' or 'tsne')
        n_components: Number of components for the projection
        
    Returns:
        Dictionary with analysis results
    """
    if USING_SUBMODULES:
        return analyze_space(
            tokens=tokens,
            method=method,
            n_components=n_components
        )
    else:
        # This would typically delegate to the analysis module
        # For simplicity, we'll just return a basic analysis
        logger.warning("Embedding space analysis requires submodules - using basic analysis")
        
        return {
            "method": method,
            "projections": None,
            "clusters": None,
            "density": None
        }

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
    if USING_SUBMODULES:
        return create_hierarchy(
            tokens=tokens,
            base_level_count=base_level_count,
            min_splats_per_level=min_splats_per_level,
            max_splats_per_level=max_splats_per_level
        )
    else:
        # This would typically delegate to the hierarchy module
        # For simplicity, return a basic hierarchy
        logger.warning("Adaptive hierarchy creation requires submodules - using default hierarchy")
        
        return {
            "levels": ["Token", "Phrase", "Document"],
            "init_splats_per_level": [64, 32, 16],
            "level_weights": [0.5, 0.3, 0.2]
        }

# Expose these as public API
__all__ = [
    'HSAInitializer',
    'initialize_splats',
    'reinitialize_splat',
    'initialize_from_tokenizer',
    'initialize_from_chat_tokens',
    'analyze_embedding_space',
    'create_adaptive_hierarchy'
]
