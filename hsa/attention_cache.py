"""
Attention caching system for Hierarchical Splat Attention (HSA).

This module provides caching capabilities to improve performance by avoiding
redundant attention computations.
"""

import functools
from typing import Dict, Tuple, Optional, Any
import numpy as np
import time


class AttentionCache:
    """Cache system for attention computations to avoid redundant calculations."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        """Initialize attention cache.
        
        Args:
            max_size: Maximum number of entries to store
            ttl: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, splat_id: str, tokens_hash: str) -> str:
        """Generate a cache key.
        
        Args:
            splat_id: ID of the splat
            tokens_hash: Hash of token embeddings
            
        Returns:
            Cache key string
        """
        return f"{splat_id}:{tokens_hash}"
    
    def _hash_tokens(self, tokens: np.ndarray) -> str:
        """Generate a hash for token embeddings.
        
        Args:
            tokens: Token embeddings
            
        Returns:
            Hash string
        """
        # Simple hash based on shape and first/last values
        # For production, consider using a more robust hash function
        shape_str = f"shape={tokens.shape}"
        sum_str = f"sum={np.sum(tokens):.6f}"
        first_str = f"first={tokens[0, 0]:.6f}" if tokens.size > 0 else ""
        last_str = f"last={tokens[-1, -1]:.6f}" if tokens.size > 0 else ""
        
        return f"{shape_str};{sum_str};{first_str};{last_str}"
    
    def get(self, splat_id: str, tokens: np.ndarray) -> Optional[np.ndarray]:
        """Get cached attention matrix if available.
        
        Args:
            splat_id: ID of the splat
            tokens: Token embeddings
            
        Returns:
            Cached attention matrix or None if not found/expired
        """
        tokens_hash = self._hash_tokens(tokens)
        key = self._generate_key(splat_id, tokens_hash)
        
        if key in self.cache:
            attention_matrix, timestamp = self.cache[key]
            
            # Check if entry is still valid
            if time.time() - timestamp <= self.ttl:
                self.hits += 1
                return attention_matrix
            
            # Entry expired, remove it
            del self.cache[key]
        
        self.misses += 1
        return None
    
    def put(self, splat_id: str, tokens: np.ndarray, attention_matrix: np.ndarray) -> None:
        """Store attention matrix in cache.
        
        Args:
            splat_id: ID of the splat
            tokens: Token embeddings
            attention_matrix: Attention matrix to cache
        """
        tokens_hash = self._hash_tokens(tokens)
        key = self._generate_key(splat_id, tokens_hash)
        
        # Add to cache with current timestamp
        self.cache[key] = (attention_matrix, time.time())
        
        # Remove oldest entries if cache exceeds max size
        if len(self.cache) > self.max_size:
            # Sort by timestamp (oldest first)
            sorted_keys = sorted(self.cache.keys(), 
                                key=lambda k: self.cache[k][1])
            
            # Remove oldest entries
            for k in sorted_keys[:len(self.cache) - self.max_size]:
                del self.cache[k]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl": self.ttl
        }


# Function decorator for caching splat attention
def cache_attention(cache: Optional[AttentionCache] = None):
    """Decorator for caching attention computation results.
    
    Args:
        cache: AttentionCache instance (if None, creates a new one)
        
    Returns:
        Decorated function
    """
    # Create default cache if not provided
    if cache is None:
        cache = AttentionCache()
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, tokens: np.ndarray, splat: Any, *args, **kwargs):
            # Try to get from cache
            cached_result = cache.get(splat.id, tokens)
            if cached_result is not None:
                return cached_result
            
            # Compute if not in cache
            result = func(self, tokens, splat, *args, **kwargs)
            
            # Store in cache
            cache.put(splat.id, tokens, result)
            
            return result
        return wrapper
    return decorator
