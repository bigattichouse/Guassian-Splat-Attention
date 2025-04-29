"""
Tokenizer-aware initialization module for Hierarchical Splat Attention (HSA).

This module provides specialized initialization methods that leverage tokenizer
information to place splats in the embedding space more intelligently.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import clustering methods
from .clustering import _initialize_kmeans_centers

def _analyze_token_distribution(tokenizer: Any) -> Dict[str, Any]:
    """
    Analyze token distribution from a tokenizer to inform initialization.
    
    Args:
        tokenizer: The tokenizer object
        
    Returns:
        Dictionary of distribution statistics
    """
    # Initialize results dictionary
    distribution = {
        "frequent_tokens": [],
        "special_tokens": [],
        "vocab_size": 0,
        "token_frequency": {}
    }
    
    try:
        # Get vocabulary size
        if hasattr(tokenizer, 'vocab_size'):
            distribution["vocab_size"] = tokenizer.vocab_size
        elif hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            distribution["vocab_size"] = len(vocab)
        
        # Identify special tokens
        special_tokens = []
        if hasattr(tokenizer, 'all_special_tokens'):
            special_tokens = tokenizer.all_special_tokens
        elif hasattr(tokenizer, 'special_tokens_map'):
            special_tokens = list(tokenizer.special_tokens_map.values())
        
        distribution["special_tokens"] = special_tokens
        
        # For some tokenizers, we can access frequency information
        if hasattr(tokenizer, 'get_vocab'):
            vocab = tokenizer.get_vocab()
            
            # Use simple heuristic: lower token IDs are often more frequent
            sorted_tokens = sorted([(i, t) for t, i in vocab.items()])
            
            # Take first 100 tokens as potentially frequent
            frequent_tokens = [t for _, t in sorted_tokens[:100]]
            distribution["frequent_tokens"] = frequent_tokens
        
        logger.info(f"Tokenizer analysis: vocab size={distribution['vocab_size']}, special tokens={len(distribution['special_tokens'])}")
        
    except Exception as e:
        logger.warning(f"Error analyzing token distribution: {e}")
    
    return distribution

def _initialize_token_aware_centers(
    data_points: np.ndarray, 
    n_clusters: int,
    level: str,
    tokenizer: Any,
    token_distribution: Dict[str, Any],
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Initialize splat centers with awareness of token distribution.
    
    This method uses information from the tokenizer to place splats
    with consideration of token frequency, importance, and patterns.
    
    Args:
        data_points: Sampled data points (token embeddings)
        n_clusters: Number of clusters (splats) to create
        level: The hierarchy level
        tokenizer: The tokenizer object
        token_distribution: Token distribution information
        random_seed: Optional random seed for reproducibility
        
    Returns:
        Centers for splats [n_clusters, embedding_dim]
    """
    # Default values in case we can't extract level index
    level_idx = 0
    if level in ["Token", "Phrase", "Section", "Document", "Global"]:
        level_idx = ["Token", "Phrase", "Section", "Document", "Global"].index(level)
    
    # For token level, we want to ensure coverage of special/frequent tokens
    if level_idx == 0 and len(token_distribution.get("frequent_tokens", [])) > 0:
        # Try to identify embeddings for frequent tokens
        special_embeddings = []
        try:
            # This depends on having the tokenizer and model architecture working together
            # so we'll try a few approaches
            if hasattr(tokenizer, 'convert_tokens_to_ids'):
                # Get token IDs for special tokens
                special_token_ids = [
                    tokenizer.convert_tokens_to_ids(token) 
                    for token in token_distribution["special_tokens"]
                ]
                
                # Look for these embeddings in our data
                # This is heuristic and might not match exactly
                if len(special_token_ids) > 0 and len(data_points) > 20:
                    # Use a subset of points that might correspond to special tokens
                    subset = data_points[:20]  # Just use the first few as a heuristic
                    special_embeddings = subset
            
            logger.info(f"Found {len(special_embeddings)} potential special token embeddings")
            
            if len(special_embeddings) > 0:
                # Dedicate some clusters to special tokens
                special_count = min(len(special_embeddings), n_clusters // 4)
                if special_count > 0:
                    # Use K-means for the rest of the clusters
                    remaining_clusters = n_clusters - special_count
                    if remaining_clusters > 0:
                        kmeans = _initialize_kmeans_centers(
                            data_points, 
                            remaining_clusters,
                            random_seed
                        )
                        
                        # Combine special embeddings with K-means centers
                        return np.vstack([
                            special_embeddings[:special_count],
                            kmeans
                        ])
        except Exception as e:
            logger.warning(f"Token-aware initialization error: {e}. Falling back to K-means.")
    
    # If we reach here, use K-means as fallback
    return _initialize_kmeans_centers(data_points, n_clusters, random_seed)

def initialize_from_tokenizer(
    tokenizer: Any,
    embedding_matrix: Optional[np.ndarray] = None,
    hierarchy_config: Optional[Dict[str, Any]] = None,
    sample_sentences: Optional[List[str]] = None,
    random_seed: Optional[int] = None
) -> Any:  # Returns SplatRegistry
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
    logger.info("Initializing splats from tokenizer")
    
    # Use default hierarchy if none provided
    if hierarchy_config is None:
        hierarchy_config = {
            "levels": ["Token", "Phrase", "Document"],
            "init_splats_per_level": [64, 32, 16],
            "level_weights": [0.5, 0.3, 0.2]
        }
    
    token_embeddings = None
    
    # Try to get embeddings from the provided matrix
    if embedding_matrix is not None:
        logger.info(f"Using provided embedding matrix with shape {embedding_matrix.shape}")
        token_embeddings = embedding_matrix
    
    # If no embeddings yet, try to generate from sample sentences
    if token_embeddings is None and sample_sentences:
        logger.info("Generating embeddings from sample sentences")
        try:
            # Generate tokens from sample sentences
            all_tokens = []
            for sentence in sample_sentences:
                if hasattr(tokenizer, 'encode'):
                    tokens = tokenizer.encode(sentence, add_special_tokens=True)
                    all_tokens.extend(tokens)
                elif hasattr(tokenizer, 'tokenize'):
                    tokens = tokenizer.tokenize(sentence)
                    # Try to convert to IDs if possible
                    if hasattr(tokenizer, 'convert_tokens_to_ids'):
                        token_ids = tokenizer.convert_tokens_to_ids(tokens)
                        all_tokens.extend(token_ids)
                    else:
                        # Just use tokens directly
                        all_tokens.extend(tokens)
            
            # Convert token IDs to embeddings - this requires model integration
            # which we don't have direct access to, so we'll create
            # pseudo-embeddings by mapping tokens to a space
            logger.info(f"Generated {len(all_tokens)} tokens from sample sentences")
            
            # Create pseudo-embeddings based on token IDs
            # This is a placeholder until real embeddings are available
            vocab_size = getattr(tokenizer, 'vocab_size', 30000)
            embedding_dim = hierarchy_config.get('embedding_dim', 64)
            
            # Initialize a deterministic random embedding based on token IDs
            if random_seed is not None:
                np.random.seed(random_seed)
            
            # Create a pseudo-embedding matrix
            pseudo_embeddings = np.random.randn(vocab_size, embedding_dim)
            
            # Map tokens to embeddings
            token_embeddings = np.array([
                pseudo_embeddings[token_id % vocab_size] for token_id in all_tokens
            ])
            
            logger.info(f"Created pseudo-embeddings with shape {token_embeddings.shape}")
        except Exception as e:
            logger.warning(f"Error generating embeddings from sample sentences: {e}")
    
    # If still no embeddings, create random ones
    if token_embeddings is None:
        logger.warning("No embeddings available, creating random embeddings")
        embedding_dim = hierarchy_config.get('embedding_dim', 64)
        vocab_size = getattr(tokenizer, 'vocab_size', 30000)
        
        # Create random embeddings
        if random_seed is not None:
            np.random.seed(random_seed)
        
        token_embeddings = np.random.randn(100, embedding_dim)
    
    # Import here to avoid circular dependencies
    from ..initialization import initialize_splats
    
    # Initialize splats using the embeddings and tokenizer
    return initialize_splats(
        tokens=token_embeddings,
        hierarchy_config=hierarchy_config,
        tokenizer=tokenizer,
        initialization_method='token_aware',
        random_seed=random_seed
    )

def initialize_from_chat_tokens(
    tokens: List[Any],
    embedding_fn: Optional[Callable] = None,
    hierarchy_config: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Any] = None,
    random_seed: Optional[int] = None
) -> Any:  # Returns SplatRegistry
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
    logger.info(f"Initializing splats from {len(tokens)} chat tokens")
    
    # Use default hierarchy if none provided
    if hierarchy_config is None:
        hierarchy_config = {
            "levels": ["Token", "Phrase", "Document"],
            "init_splats_per_level": [64, 32, 16],
            "level_weights": [0.5, 0.3, 0.2]
        }
    
    # Try to convert tokens to embeddings
    token_embeddings = None
    
    if embedding_fn is not None:
        try:
            # Convert tokens to embeddings using the provided function
            token_embeddings = embedding_fn(tokens)
            logger.info(f"Generated embeddings with shape {token_embeddings.shape}")
        except Exception as e:
            logger.warning(f"Error converting tokens to embeddings: {e}")
    
    # If no embeddings yet, create pseudo-embeddings
    if token_embeddings is None:
        logger.warning("No embedding function available, creating pseudo-embeddings")
        embedding_dim = hierarchy_config.get('embedding_dim', 64)
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create pseudo-embeddings based on token values
        try:
            # Try to convert tokens to numeric values
            token_values = []
            for token in tokens:
                if isinstance(token, (int, float)):
                    token_values.append(token)
                elif hasattr(token, 'value'):
                    token_values.append(token.value)
                elif isinstance(token, str) and token.isdigit():
                    token_values.append(int(token))
                else:
                    # Use hash value
                    token_values.append(hash(str(token)) % 10000)
            
            # Create deterministic pseudo-embeddings based on token values
            pseudo_embeddings = np.zeros((len(token_values), embedding_dim))
            for i, value in enumerate(token_values):
                # Use the value to seed a deterministic embedding
                np.random.seed(value + (random_seed or 0))
                pseudo_embeddings[i] = np.random.randn(embedding_dim)
            
            token_embeddings = pseudo_embeddings
            logger.info(f"Created pseudo-embeddings with shape {token_embeddings.shape}")
        except Exception as e:
            logger.warning(f"Error creating pseudo-embeddings: {e}")
            # Fallback to random embeddings
            token_embeddings = np.random.randn(len(tokens), embedding_dim)
    
    # Import here to avoid circular dependencies
    from ..initialization import initialize_splats
    
    # Initialize splats using the embeddings
    return initialize_splats(
        tokens=token_embeddings,
        hierarchy_config=hierarchy_config,
        tokenizer=tokenizer,
        initialization_method='token_aware' if tokenizer else 'kmeans',
        random_seed=random_seed
    )
