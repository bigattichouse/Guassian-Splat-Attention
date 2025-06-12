"""
SplatFlow Hierarchical Cache Module
Multi-level caching system for O(n*k) optimization with intelligent cache management.
Provides sequence/chunk/local level caching for trajectories, attention patterns, and splat arrangements.
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import pickle
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque, OrderedDict
from dataclasses import dataclass
from enum import Enum
import json
import threading
from functools import lru_cache

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Enumeration of cache levels for hierarchical caching"""
    LOCAL = "local"           # Individual token neighborhoods  
    CHUNK = "chunk"          # Small sequence segments (64-256 tokens)
    SEQUENCE = "sequence"    # Full sequences (up to model max_len)
    GLOBAL = "global"        # Cross-sequence patterns
    

@dataclass
class CacheEntry:
    """Unified cache entry structure for all cache types"""
    key: str
    data: Any
    timestamp: float
    access_count: int
    cache_level: CacheLevel
    content_type: Optional[str] = None
    sequence_length: Optional[int] = None
    confidence_score: float = 1.0
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.timestamp = time.time()
    
    def age(self) -> float:
        """Get age in seconds"""
        return time.time() - self.timestamp
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'key': self.key,
            'timestamp': self.timestamp,
            'access_count': self.access_count,
            'cache_level': self.cache_level.value,
            'content_type': self.content_type,
            'sequence_length': self.sequence_length,
            'confidence_score': self.confidence_score,
            'metadata': self.metadata
        }


class IntelligentCacheKey:
    """Intelligent cache key generator with content-aware hashing"""
    
    @staticmethod
    def generate_trajectory_key(embeddings: torch.Tensor, 
                              layer_idx: int, 
                              cache_level: CacheLevel,
                              content_type: str = "general") -> str:
        """Generate cache key for trajectory data"""
        try:
            # Create content fingerprint
            if embeddings.numel() > 0:
                fingerprint = IntelligentCacheKey._create_tensor_fingerprint(embeddings)
            else:
                fingerprint = "empty"
            
            # Combine components
            key_components = [
                f"traj",
                f"l{layer_idx}",
                f"{cache_level.value}",
                f"{content_type}",
                f"{embeddings.shape}",
                fingerprint
            ]
            
            return "_".join(key_components)
        except Exception as e:
            logger.warning(f"Failed to generate trajectory key: {e}")
            return f"traj_fallback_{int(time.time())}"
    
    @staticmethod
    def generate_attention_key(attention_weights: torch.Tensor,
                             num_splats: int,
                             content_type: str = "general") -> str:
        """Generate cache key for attention patterns"""
        try:
            fingerprint = IntelligentCacheKey._create_tensor_fingerprint(attention_weights)
            
            key_components = [
                f"attn",
                f"s{num_splats}",
                f"{content_type}",
                f"{attention_weights.shape}",
                fingerprint
            ]
            
            return "_".join(key_components)
        except Exception as e:
            logger.warning(f"Failed to generate attention key: {e}")
            return f"attn_fallback_{int(time.time())}"
    
    @staticmethod
    def generate_constellation_key(splat_positions: List[torch.Tensor],
                                 model_dim: int,
                                 sequence_length: int,
                                 content_type: str = "general") -> str:
        """Generate cache key for splat constellations"""
        try:
            # Create fingerprint from splat positions
            position_data = torch.cat(splat_positions) if splat_positions else torch.tensor([])
            fingerprint = IntelligentCacheKey._create_tensor_fingerprint(position_data)
            
            key_components = [
                f"const",
                f"d{model_dim}",
                f"len{sequence_length}",
                f"n{len(splat_positions)}",
                f"{content_type}",
                fingerprint
            ]
            
            return "_".join(key_components)
        except Exception as e:
            logger.warning(f"Failed to generate constellation key: {e}")
            return f"const_fallback_{int(time.time())}"
    
    @staticmethod
    def generate_embedding_stats_key(embeddings: torch.Tensor,
                                   stat_type: str = "general") -> str:
        """Generate cache key for embedding statistics"""
        try:
            fingerprint = IntelligentCacheKey._create_tensor_fingerprint(embeddings)
            
            key_components = [
                f"stats",
                f"{stat_type}",
                f"{embeddings.shape}",
                fingerprint
            ]
            
            return "_".join(key_components)
        except Exception as e:
            logger.warning(f"Failed to generate embedding stats key: {e}")
            return f"stats_fallback_{int(time.time())}"
    
    @staticmethod
    def _create_tensor_fingerprint(tensor: torch.Tensor, max_elements: int = 1000) -> str:
        """Create a compact fingerprint of tensor content"""
        try:
            if tensor.numel() == 0:
                return "empty"
            
            # Sample tensor for large tensors
            if tensor.numel() > max_elements:
                flat = tensor.flatten()
                indices = torch.linspace(0, len(flat) - 1, max_elements, dtype=torch.long)
                sampled = flat[indices]
            else:
                sampled = tensor.flatten()
            
            # Create statistical fingerprint
            stats = {
                'mean': float(sampled.mean()),
                'std': float(sampled.std()),
                'min': float(sampled.min()),
                'max': float(sampled.max())
            }
            
            # Hash the statistics
            stats_str = json.dumps(stats, sort_keys=True)
            hash_obj = hashlib.md5(stats_str.encode())
            return hash_obj.hexdigest()[:12]  # First 12 characters
            
        except Exception as e:
            logger.warning(f"Failed to create tensor fingerprint: {e}")
            return f"fp_error_{int(time.time())}"


class HierarchicalTrajectoryCache(nn.Module):
    """
    Multi-level trajectory caching system with intelligent eviction and content-aware storage.
    Caches trajectories at local, chunk, sequence, and global levels.
    """
    
    def __init__(self, model_dim: int, cache_levels: List[str] = ["local", "chunk", "sequence"],
                 cache_sizes: Optional[Dict[str, int]] = None, 
                 max_age_seconds: float = 300.0):
        super().__init__()
        
        self.model_dim = model_dim
        self.cache_levels = [CacheLevel(level) for level in cache_levels]
        self.max_age_seconds = max_age_seconds
        
        # Default cache sizes per level
        default_sizes = {
            "local": 5000,     # Many small local patterns
            "chunk": 2000,     # Medium number of chunk patterns  
            "sequence": 500,   # Fewer full sequence patterns
            "global": 100      # Very few global patterns
        }
        self.cache_sizes = cache_sizes or default_sizes
        
        # Cache storage by level
        self.caches = {}
        for level in self.cache_levels:
            self.caches[level] = OrderedDict()
        
        # Cache statistics
        self.stats = {
            'hits': defaultdict(int),
            'misses': defaultdict(int),
            'evictions': defaultdict(int),
            'total_entries': defaultdict(int)
        }
        
        # Content type classification
        self.content_classifier = self._create_content_classifier()
        
        # Thread safety
        self._cache_lock = threading.RLock()
        
        logger.info(f"🗄️ Hierarchical trajectory cache initialized")
        logger.info(f"   Levels: {[level.value for level in self.cache_levels]}")
        logger.info(f"   Sizes: {self.cache_sizes}")
    
    def _create_content_classifier(self) -> nn.Module:
        """Create simple content type classifier"""
        return nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.GELU(),
            nn.Linear(self.model_dim // 2, 4),  # general, code, dialogue, narrative
            nn.Softmax(dim=-1)
        )
    
    def store_trajectory(self, embeddings: torch.Tensor, trajectory: torch.Tensor,
                        layer_idx: int, cache_level: CacheLevel = CacheLevel.LOCAL,
                        content_type: Optional[str] = None) -> bool:
        """Store trajectory in appropriate cache level"""
        
        with self._cache_lock:
            try:
                # Classify content type if not provided
                if content_type is None:
                    content_type = self._classify_content(embeddings)
                
                # Generate cache key
                cache_key = IntelligentCacheKey.generate_trajectory_key(
                    embeddings, layer_idx, cache_level, content_type
                )
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    data=trajectory.clone().detach(),
                    timestamp=time.time(),
                    access_count=1,
                    cache_level=cache_level,
                    content_type=content_type,
                    sequence_length=embeddings.size(1) if embeddings.dim() > 1 else None,
                    confidence_score=self._calculate_confidence(embeddings, trajectory),
                    metadata={
                        'layer_idx': layer_idx,
                        'shape': tuple(trajectory.shape),
                        'device': str(trajectory.device)
                    }
                )
                
                # Store in cache
                cache = self.caches[cache_level]
                cache[cache_key] = entry
                
                # Maintain cache size limits
                self._evict_if_needed(cache_level)
                
                self.stats['total_entries'][cache_level] += 1
                
                return True
                
            except Exception as e:
                logger.warning(f"Failed to store trajectory in cache: {e}")
                return False
    
    def retrieve_trajectory(self, embeddings: torch.Tensor, layer_idx: int,
                          cache_level: CacheLevel = CacheLevel.LOCAL,
                          content_type: Optional[str] = None,
                          similarity_threshold: float = 0.95) -> Optional[torch.Tensor]:
        """Retrieve cached trajectory if available"""
        
        with self._cache_lock:
            try:
                # Classify content type if not provided
                if content_type is None:
                    content_type = self._classify_content(embeddings)
                
                # Try exact match first
                cache_key = IntelligentCacheKey.generate_trajectory_key(
                    embeddings, layer_idx, cache_level, content_type
                )
                
                cache = self.caches[cache_level]
                
                if cache_key in cache:
                    entry = cache[cache_key]
                    
                    # Check if entry is still valid
                    if entry.age() < self.max_age_seconds:
                        entry.update_access()
                        # Move to end (LRU)
                        cache.move_to_end(cache_key)
                        
                        self.stats['hits'][cache_level] += 1
                        return entry.data.clone()
                    else:
                        # Remove expired entry
                        del cache[cache_key]
                
                # Try similarity search if exact match fails
                similar_entry = self._find_similar_trajectory(
                    embeddings, layer_idx, cache_level, content_type, similarity_threshold
                )
                
                if similar_entry is not None:
                    similar_entry.update_access()
                    self.stats['hits'][cache_level] += 1
                    return similar_entry.data.clone()
                
                self.stats['misses'][cache_level] += 1
                return None
                
            except Exception as e:
                logger.warning(f"Failed to retrieve trajectory from cache: {e}")
                self.stats['misses'][cache_level] += 1
                return None
    
    def _find_similar_trajectory(self, embeddings: torch.Tensor, layer_idx: int,
                               cache_level: CacheLevel, content_type: str,
                               threshold: float) -> Optional[CacheEntry]:
        """Find similar cached trajectory using content similarity"""
        
        try:
            cache = self.caches[cache_level]
            best_entry = None
            best_similarity = 0.0
            
            # Create fingerprint for comparison
            target_fingerprint = IntelligentCacheKey._create_tensor_fingerprint(embeddings)
            
            for entry in cache.values():
                # Filter by content type and layer
                if (entry.content_type == content_type and 
                    entry.metadata.get('layer_idx') == layer_idx):
                    
                    # Simple similarity based on fingerprint matching and shape
                    entry_key_parts = entry.key.split('_')
                    if len(entry_key_parts) >= 6:
                        entry_fingerprint = entry_key_parts[-1]
                        
                        # Check shape compatibility
                        if entry.data.shape == embeddings.shape:
                            # Simple fingerprint similarity (could be enhanced)
                            if entry_fingerprint == target_fingerprint:
                                similarity = 1.0
                            else:
                                similarity = 0.8  # Partial match
                            
                            if similarity > best_similarity and similarity >= threshold:
                                best_similarity = similarity
                                best_entry = entry
            
            return best_entry
            
        except Exception as e:
            logger.warning(f"Failed to find similar trajectory: {e}")
            return None
    
    def _classify_content(self, embeddings: torch.Tensor) -> str:
        """Classify content type of embeddings"""
        
        try:
            with torch.no_grad():
                if embeddings.numel() == 0:
                    return "general"
                
                # Use a representative sample
                if embeddings.dim() == 3:  # [batch, seq, dim]
                    sample = embeddings[0, :min(32, embeddings.size(1)), :].mean(dim=0)
                elif embeddings.dim() == 2:  # [seq, dim]
                    sample = embeddings[:min(32, embeddings.size(0)), :].mean(dim=0)
                else:
                    sample = embeddings
                
                # Classify using simple heuristics + learned classifier
                probs = self.content_classifier(sample.unsqueeze(0))
                class_idx = torch.argmax(probs, dim=-1).item()
                
                content_types = ["general", "code", "dialogue", "narrative"]
                return content_types[class_idx] if class_idx < len(content_types) else "general"
                
        except Exception as e:
            logger.warning(f"Content classification failed: {e}")
            return "general"
    
    def _calculate_confidence(self, embeddings: torch.Tensor, trajectory: torch.Tensor) -> float:
        """Calculate confidence score for cached trajectory"""
        
        try:
            # Simple confidence based on trajectory magnitude and consistency
            if trajectory.numel() == 0:
                return 0.0
            
            magnitude = torch.norm(trajectory).item()
            variance = torch.var(trajectory).item()
            
            # Higher magnitude and lower variance = higher confidence
            confidence = min(1.0, magnitude / (1.0 + variance))
            return max(0.0, confidence)
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _evict_if_needed(self, cache_level: CacheLevel):
        """Evict entries if cache size limit exceeded"""
        
        try:
            cache = self.caches[cache_level]
            max_size = self.cache_sizes.get(cache_level.value, 1000)
            
            while len(cache) > max_size:
                # Evict least recently used entry
                oldest_key, oldest_entry = cache.popitem(last=False)
                self.stats['evictions'][cache_level] += 1
                
                logger.debug(f"Evicted {cache_level.value} cache entry: {oldest_key}")
                
        except Exception as e:
            logger.warning(f"Cache eviction failed for {cache_level.value}: {e}")
    
    def clear_cache(self, cache_level: Optional[CacheLevel] = None):
        """Clear cache(s)"""
        
        with self._cache_lock:
            if cache_level is not None:
                self.caches[cache_level].clear()
                logger.info(f"Cleared {cache_level.value} cache")
            else:
                for level in self.cache_levels:
                    self.caches[level].clear()
                logger.info("Cleared all trajectory caches")
    
    def get_cache_statistics(self) -> Dict:
        """Get comprehensive cache statistics"""
        
        with self._cache_lock:
            try:
                stats = {}
                
                for level in self.cache_levels:
                    cache = self.caches[level]
                    hits = self.stats['hits'][level]
                    misses = self.stats['misses'][level]
                    total_requests = hits + misses
                    
                    stats[level.value] = {
                        'size': len(cache),
                        'max_size': self.cache_sizes.get(level.value, 0),
                        'hits': hits,
                        'misses': misses,
                        'hit_rate': hits / max(total_requests, 1),
                        'evictions': self.stats['evictions'][level],
                        'total_entries_created': self.stats['total_entries'][level]
                    }
                
                # Overall statistics
                total_hits = sum(self.stats['hits'].values())
                total_misses = sum(self.stats['misses'].values())
                total_requests = total_hits + total_misses
                
                stats['overall'] = {
                    'total_requests': total_requests,
                    'total_hits': total_hits,
                    'total_misses': total_misses,
                    'overall_hit_rate': total_hits / max(total_requests, 1),
                    'cache_type': 'HierarchicalTrajectoryCache'
                }
                
                return stats
                
            except Exception as e:
                logger.warning(f"Failed to get cache statistics: {e}")
                return {'error': str(e)}


class AttentionPatternCache(nn.Module):
    """
    Cache for common attention patterns organized by content type.
    Stores and retrieves attention weight patterns to avoid recomputation.
    """
    
    def __init__(self, cache_size: int = 2000, similarity_threshold: float = 0.9,
                 max_age_seconds: float = 600.0):
        super().__init__()
        
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        self.max_age_seconds = max_age_seconds
        
        # Pattern storage organized by content type
        self.pattern_cache = defaultdict(OrderedDict)
        
        # Pattern analysis
        self.pattern_analyzer = self._create_pattern_analyzer()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'patterns_stored': 0,
            'evictions': 0
        }
        
        # Thread safety
        self._cache_lock = threading.RLock()
        
        logger.info(f"🎯 Attention pattern cache initialized (size: {cache_size})")
    
    def _create_pattern_analyzer(self) -> nn.Module:
        """Create pattern analyzer for attention weights"""
        return nn.Sequential(
            nn.AdaptiveAvgPool1d(64),  # Normalize to fixed size
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 16)  # Pattern embedding
        )
    
    def store_pattern(self, attention_weights: torch.Tensor, num_splats: int,
                     content_type: str = "general", 
                     metadata: Optional[Dict] = None) -> bool:
        """Store attention pattern in cache"""
        
        with self._cache_lock:
            try:
                # Generate cache key
                cache_key = IntelligentCacheKey.generate_attention_key(
                    attention_weights, num_splats, content_type
                )
                
                # Analyze pattern
                pattern_embedding = self._analyze_pattern(attention_weights)
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    data={
                        'attention_weights': attention_weights.clone().detach(),
                        'pattern_embedding': pattern_embedding.detach(),
                        'num_splats': num_splats
                    },
                    timestamp=time.time(),
                    access_count=1,
                    cache_level=CacheLevel.SEQUENCE,
                    content_type=content_type,
                    sequence_length=attention_weights.size(1) if attention_weights.dim() > 1 else None,
                    metadata=metadata or {}
                )
                
                # Store in content-specific cache
                cache = self.pattern_cache[content_type]
                cache[cache_key] = entry
                
                # Maintain cache size
                self._evict_patterns_if_needed(content_type)
                
                self.stats['patterns_stored'] += 1
                
                return True
                
            except Exception as e:
                logger.warning(f"Failed to store attention pattern: {e}")
                return False
    
    def retrieve_pattern(self, query_weights: torch.Tensor, num_splats: int,
                        content_type: str = "general") -> Optional[torch.Tensor]:
        """Retrieve similar attention pattern from cache"""
        
        with self._cache_lock:
            try:
                # Try exact match first
                cache_key = IntelligentCacheKey.generate_attention_key(
                    query_weights, num_splats, content_type
                )
                
                cache = self.pattern_cache[content_type]
                
                if cache_key in cache:
                    entry = cache[cache_key]
                    if entry.age() < self.max_age_seconds:
                        entry.update_access()
                        cache.move_to_end(cache_key)
                        
                        self.stats['hits'] += 1
                        return entry.data['attention_weights'].clone()
                    else:
                        del cache[cache_key]
                
                # Try similarity search
                similar_pattern = self._find_similar_pattern(
                    query_weights, num_splats, content_type
                )
                
                if similar_pattern is not None:
                    self.stats['hits'] += 1
                    return similar_pattern
                
                self.stats['misses'] += 1
                return None
                
            except Exception as e:
                logger.warning(f"Failed to retrieve attention pattern: {e}")
                self.stats['misses'] += 1
                return None
    
    def _analyze_pattern(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Analyze attention pattern to create embedding"""
        
        try:
            with torch.no_grad():
                if attention_weights.dim() == 3:  # [batch, seq, splats]
                    # Average across batch
                    pattern = attention_weights.mean(dim=0)  # [seq, splats]
                else:
                    pattern = attention_weights
                
                if pattern.dim() == 2:
                    # Average across splats
                    pattern = pattern.mean(dim=-1)  # [seq]
                
                # Normalize pattern length and analyze
                if len(pattern) > 0:
                    pattern = pattern.unsqueeze(0)  # Add batch dim
                    embedding = self.pattern_analyzer(pattern.unsqueeze(1))  # Add channel dim
                    return embedding.squeeze()
                else:
                    return torch.zeros(16)  # Default embedding size
                    
        except Exception as e:
            logger.warning(f"Pattern analysis failed: {e}")
            return torch.zeros(16)
    
    def _find_similar_pattern(self, query_weights: torch.Tensor, num_splats: int,
                            content_type: str) -> Optional[torch.Tensor]:
        """Find similar pattern using embedding similarity"""
        
        try:
            query_embedding = self._analyze_pattern(query_weights)
            cache = self.pattern_cache[content_type]
            
            best_similarity = 0.0
            best_pattern = None
            
            for entry in cache.values():
                # Check compatibility
                if entry.data['num_splats'] == num_splats:
                    stored_embedding = entry.data['pattern_embedding']
                    
                    # Compute cosine similarity
                    similarity = F.cosine_similarity(
                        query_embedding.unsqueeze(0),
                        stored_embedding.unsqueeze(0),
                        dim=1
                    ).item()
                    
                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_pattern = entry.data['attention_weights']
                        entry.update_access()
            
            return best_pattern.clone() if best_pattern is not None else None
            
        except Exception as e:
            logger.warning(f"Failed to find similar pattern: {e}")
            return None
    
    def _evict_patterns_if_needed(self, content_type: str):
        """Evict old patterns if cache is full"""
        
        try:
            cache = self.pattern_cache[content_type]
            
            while len(cache) > self.cache_size // 4:  # Divide cache among content types
                oldest_key, _ = cache.popitem(last=False)
                self.stats['evictions'] += 1
                
        except Exception as e:
            logger.warning(f"Pattern eviction failed: {e}")
    
    def get_pattern_statistics(self) -> Dict:
        """Get attention pattern cache statistics"""
        
        with self._cache_lock:
            try:
                total_patterns = sum(len(cache) for cache in self.pattern_cache.values())
                total_requests = self.stats['hits'] + self.stats['misses']
                
                content_type_stats = {}
                for content_type, cache in self.pattern_cache.items():
                    content_type_stats[content_type] = {
                        'patterns': len(cache),
                        'avg_age': np.mean([entry.age() for entry in cache.values()]) if cache else 0
                    }
                
                return {
                    'total_patterns': total_patterns,
                    'patterns_by_content_type': content_type_stats,
                    'hits': self.stats['hits'],
                    'misses': self.stats['misses'],
                    'hit_rate': self.stats['hits'] / max(total_requests, 1),
                    'patterns_stored': self.stats['patterns_stored'],
                    'evictions': self.stats['evictions'],
                    'cache_type': 'AttentionPatternCache'
                }
                
            except Exception as e:
                logger.warning(f"Failed to get pattern statistics: {e}")
                return {'error': str(e)}


class SplatConstellationCache(nn.Module):
    """
    Cache for optimal splat arrangements (constellations) based on content type and sequence length.
    Stores proven splat configurations for fast initialization and adaptation.
    """
    
    def __init__(self, cache_size: int = 500, model_dim: int = 512):
        super().__init__()
        
        self.cache_size = cache_size
        self.model_dim = model_dim
        
        # Constellation storage
        self.constellation_cache = OrderedDict()
        
        # Quality metrics for stored constellations
        self.quality_analyzer = self._create_quality_analyzer()
        
        # Statistics
        self.stats = {
            'constellations_stored': 0,
            'constellations_retrieved': 0,
            'quality_improvements': 0,
            'evictions': 0
        }
        
        # Thread safety
        self._cache_lock = threading.RLock()
        
        logger.info(f"⭐ Splat constellation cache initialized (size: {cache_size})")
    
    def _create_quality_analyzer(self) -> nn.Module:
        """Create constellation quality analyzer"""
        return nn.Sequential(
            nn.Linear(self.model_dim * 2, self.model_dim),  # position + stats
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.GELU(),
            nn.Linear(self.model_dim // 2, 1),  # Quality score
            nn.Sigmoid()
        )
    
    def store_constellation(self, splat_positions: List[torch.Tensor],
                          performance_metrics: Dict,
                          content_type: str = "general",
                          sequence_length: int = 1024,
                          layer_idx: int = 0) -> bool:
        """Store successful splat constellation"""
        
        with self._cache_lock:
            try:
                # Generate cache key
                cache_key = IntelligentCacheKey.generate_constellation_key(
                    splat_positions, self.model_dim, sequence_length, content_type
                )
                
                # Analyze constellation quality
                quality_score = self._analyze_constellation_quality(
                    splat_positions, performance_metrics
                )
                
                # Create constellation data
                constellation_data = {
                    'positions': [pos.clone().detach() for pos in splat_positions],
                    'performance_metrics': performance_metrics.copy(),
                    'quality_score': quality_score,
                    'num_splats': len(splat_positions),
                    'layer_idx': layer_idx
                }
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    data=constellation_data,
                    timestamp=time.time(),
                    access_count=1,
                    cache_level=CacheLevel.SEQUENCE,
                    content_type=content_type,
                    sequence_length=sequence_length,
                    confidence_score=quality_score,
                    metadata={
                        'num_splats': len(splat_positions),
                        'layer_idx': layer_idx,
                        'performance_metrics': performance_metrics
                    }
                )
                
                # Store constellation
                self.constellation_cache[cache_key] = entry
                
                # Maintain cache size
                self._evict_constellations_if_needed()
                
                self.stats['constellations_stored'] += 1
                
                logger.debug(f"Stored constellation: {content_type}, quality={quality_score:.3f}")
                
                return True
                
            except Exception as e:
                logger.warning(f"Failed to store constellation: {e}")
                return False
    
    def retrieve_constellation(self, content_type: str = "general",
                             sequence_length: int = 1024,
                             num_splats: int = 20,
                             layer_idx: int = 0) -> Optional[Dict]:
        """Retrieve best matching constellation"""
        
        with self._cache_lock:
            try:
                best_constellation = None
                best_score = 0.0
                
                for entry in self.constellation_cache.values():
                    # Filter by compatibility
                    if (entry.content_type == content_type and
                        entry.metadata['layer_idx'] == layer_idx and
                        entry.metadata['num_splats'] == num_splats):
                        
                        # Calculate compatibility score
                        seq_len_ratio = min(sequence_length, entry.sequence_length) / max(sequence_length, entry.sequence_length)
                        compatibility_score = entry.confidence_score * seq_len_ratio
                        
                        if compatibility_score > best_score:
                            best_score = compatibility_score
                            best_constellation = entry
                
                if best_constellation is not None:
                    best_constellation.update_access()
                    # Move to end (LRU)
                    self.constellation_cache.move_to_end(best_constellation.key)
                    
                    self.stats['constellations_retrieved'] += 1
                    
                    return {
                        'positions': [pos.clone() for pos in best_constellation.data['positions']],
                        'quality_score': best_constellation.data['quality_score'],
                        'performance_metrics': best_constellation.data['performance_metrics'],
                        'adaptation_needed': sequence_length != best_constellation.sequence_length
                    }
                
                return None
                
            except Exception as e:
                logger.warning(f"Failed to retrieve constellation: {e}")
                return None
    
    def _analyze_constellation_quality(self, splat_positions: List[torch.Tensor],
                                     performance_metrics: Dict) -> float:
        """Analyze quality of splat constellation"""
        
        try:
            if not splat_positions:
                return 0.0
            
            # Calculate spatial distribution metrics
            positions = torch.stack(splat_positions)  # [num_splats, model_dim]
            
            # Coverage quality (how well distributed)
            pairwise_distances = torch.cdist(positions, positions)
            min_distances = pairwise_distances[pairwise_distances > 0].min()
            avg_distance = pairwise_distances[pairwise_distances > 0].mean()
            
            coverage_score = min(1.0, min_distances / (avg_distance + 1e-8))
            
            # Performance quality from metrics
            health_ratio = performance_metrics.get('health_ratio', 0.0)
            usefulness = performance_metrics.get('avg_usefulness', 0.0)
            trajectory_influence = performance_metrics.get('avg_trajectory_influence', 0.0)
            
            performance_score = (health_ratio + usefulness / 5.0 + trajectory_influence * 1e6) / 3.0
            performance_score = min(1.0, max(0.0, performance_score))
            
            # Combined quality score
            quality_score = 0.4 * coverage_score + 0.6 * performance_score
            
            return float(quality_score)
            
        except Exception as e:
            logger.warning(f"Failed to analyze constellation quality: {e}")
            return 0.5
    
    def _evict_constellations_if_needed(self):
        """Evict low-quality constellations if cache is full"""
        
        try:
            while len(self.constellation_cache) > self.cache_size:
                # Find lowest quality constellation
                worst_key = None
                worst_score = float('inf')
                
                for key, entry in self.constellation_cache.items():
                    score = entry.confidence_score * (1.0 + entry.access_count / 10.0)
                    if score < worst_score:
                        worst_score = score
                        worst_key = key
                
                if worst_key:
                    del self.constellation_cache[worst_key]
                    self.stats['evictions'] += 1
                
        except Exception as e:
            logger.warning(f"Constellation eviction failed: {e}")
    
    def get_constellation_statistics(self) -> Dict:
        """Get constellation cache statistics"""
        
        with self._cache_lock:
            try:
                if self.constellation_cache:
                    quality_scores = [entry.confidence_score for entry in self.constellation_cache.values()]
                    avg_quality = np.mean(quality_scores)
                    content_types = defaultdict(int)
                    
                    for entry in self.constellation_cache.values():
                        content_types[entry.content_type] += 1
                else:
                    avg_quality = 0.0
                    content_types = {}
                
                return {
                    'total_constellations': len(self.constellation_cache),
                    'avg_quality_score': avg_quality,
                    'constellations_by_content_type': dict(content_types),
                    'constellations_stored': self.stats['constellations_stored'],
                    'constellations_retrieved': self.stats['constellations_retrieved'],
                    'quality_improvements': self.stats['quality_improvements'],
                    'evictions': self.stats['evictions'],
                    'cache_type': 'SplatConstellationCache'
                }
                
            except Exception as e:
                logger.warning(f"Failed to get constellation statistics: {e}")
                return {'error': str(e)}


class EmbeddingStatsCache(nn.Module):
    """
    Cache for embedding statistics to speed up splat positioning and trajectory computation.
    Stores statistical properties of token embeddings by content type.
    """
    
    def __init__(self, cache_size: int = 1000, model_dim: int = 512):
        super().__init__()
        
        self.cache_size = cache_size
        self.model_dim = model_dim
        
        # Statistics cache
        self.stats_cache = OrderedDict()
        
        # Statistics aggregator
        self.stats_aggregator = self._create_stats_aggregator()
        
        # Statistics
        self.cache_stats = {
            'stats_stored': 0,
            'stats_retrieved': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread safety
        self._cache_lock = threading.RLock()
        
        logger.info(f"📊 Embedding stats cache initialized (size: {cache_size})")
    
    def _create_stats_aggregator(self) -> nn.Module:
        """Create statistics aggregator network"""
        return nn.Sequential(
            nn.Linear(8, 16),  # Input: [mean, std, min, max, median, q25, q75, range]
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 4)  # Compact stats representation
        )
    
    def store_embedding_stats(self, embeddings: torch.Tensor,
                            content_type: str = "general",
                            context_info: Optional[Dict] = None) -> bool:
        """Store embedding statistics"""
        
        with self._cache_lock:
            try:
                # Compute comprehensive statistics
                stats = self._compute_embedding_statistics(embeddings)
                
                # Generate cache key
                cache_key = IntelligentCacheKey.generate_embedding_stats_key(
                    embeddings, content_type
                )
                
                # Create compact representation
                compact_stats = self._create_compact_representation(stats)
                
                # Create cache entry
                entry = CacheEntry(
                    key=cache_key,
                    data={
                        'statistics': stats,
                        'compact_representation': compact_stats,
                        'embedding_shape': tuple(embeddings.shape),
                        'context_info': context_info or {}
                    },
                    timestamp=time.time(),
                    access_count=1,
                    cache_level=CacheLevel.SEQUENCE,
                    content_type=content_type,
                    sequence_length=embeddings.size(1) if embeddings.dim() > 1 else None,
                    confidence_score=self._calculate_stats_confidence(embeddings, stats)
                )
                
                # Store statistics
                self.stats_cache[cache_key] = entry
                
                # Maintain cache size
                self._evict_stats_if_needed()
                
                self.cache_stats['stats_stored'] += 1
                
                return True
                
            except Exception as e:
                logger.warning(f"Failed to store embedding stats: {e}")
                return False
    
    def retrieve_embedding_stats(self, embeddings: torch.Tensor,
                                content_type: str = "general") -> Optional[Dict]:
        """Retrieve cached embedding statistics"""
        
        with self._cache_lock:
            try:
                # Try exact match
                cache_key = IntelligentCacheKey.generate_embedding_stats_key(
                    embeddings, content_type
                )
                
                if cache_key in self.stats_cache:
                    entry = self.stats_cache[cache_key]
                    entry.update_access()
                    self.stats_cache.move_to_end(cache_key)
                    
                    self.cache_stats['cache_hits'] += 1
                    self.cache_stats['stats_retrieved'] += 1
                    
                    return entry.data['statistics'].copy()
                
                # Try to find compatible stats
                compatible_stats = self._find_compatible_stats(embeddings, content_type)
                
                if compatible_stats is not None:
                    self.cache_stats['cache_hits'] += 1
                    self.cache_stats['stats_retrieved'] += 1
                    return compatible_stats
                
                self.cache_stats['cache_misses'] += 1
                return None
                
            except Exception as e:
                logger.warning(f"Failed to retrieve embedding stats: {e}")
                self.cache_stats['cache_misses'] += 1
                return None
    
    def _compute_embedding_statistics(self, embeddings: torch.Tensor) -> Dict:
        """Compute comprehensive embedding statistics"""
        
        try:
            with torch.no_grad():
                flat_embeddings = embeddings.flatten()
                
                if flat_embeddings.numel() == 0:
                    return self._get_empty_stats()
                
                stats = {
                    'mean': float(flat_embeddings.mean()),
                    'std': float(flat_embeddings.std()),
                    'min': float(flat_embeddings.min()),
                    'max': float(flat_embeddings.max()),
                    'median': float(flat_embeddings.median()),
                    'q25': float(flat_embeddings.quantile(0.25)),
                    'q75': float(flat_embeddings.quantile(0.75)),
                    'range': float(flat_embeddings.max() - flat_embeddings.min()),
                    'norm': float(torch.norm(embeddings)),
                    'sparsity': float((flat_embeddings.abs() < 1e-6).float().mean())
                }
                
                # Per-dimension statistics
                if embeddings.dim() >= 2:
                    dim_means = embeddings.mean(dim=tuple(range(embeddings.dim() - 1)))
                    stats['dim_mean_std'] = float(dim_means.std())
                    stats['dim_mean_range'] = float(dim_means.max() - dim_means.min())
                
                return stats
                
        except Exception as e:
            logger.warning(f"Failed to compute embedding statistics: {e}")
            return self._get_empty_stats()
    
    def _get_empty_stats(self) -> Dict:
        """Get default statistics for empty embeddings"""
        return {
            'mean': 0.0, 'std': 1.0, 'min': 0.0, 'max': 0.0,
            'median': 0.0, 'q25': 0.0, 'q75': 0.0, 'range': 0.0,
            'norm': 0.0, 'sparsity': 0.0, 'dim_mean_std': 0.0, 'dim_mean_range': 0.0
        }
    
    def _create_compact_representation(self, stats: Dict) -> torch.Tensor:
        """Create compact neural representation of statistics"""
        
        try:
            # Extract key statistics for neural processing
            key_stats = torch.tensor([
                stats['mean'], stats['std'], stats['min'], stats['max'],
                stats['median'], stats['q25'], stats['q75'], stats['range']
            ], dtype=torch.float32)
            
            # Create compact representation
            with torch.no_grad():
                compact = self.stats_aggregator(key_stats)
            
            return compact
            
        except Exception as e:
            logger.warning(f"Failed to create compact representation: {e}")
            return torch.zeros(4)
    
    def _calculate_stats_confidence(self, embeddings: torch.Tensor, stats: Dict) -> float:
        """Calculate confidence in statistics quality"""
        
        try:
            # Higher confidence for larger, more varied embeddings
            size_factor = min(1.0, embeddings.numel() / 10000)  # Normalize by typical size
            variation_factor = min(1.0, stats['std'] / max(abs(stats['mean']), 1e-6))
            
            confidence = 0.7 * size_factor + 0.3 * variation_factor
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Failed to calculate stats confidence: {e}")
            return 0.5
    
    def _find_compatible_stats(self, embeddings: torch.Tensor, content_type: str) -> Optional[Dict]:
        """Find compatible cached statistics"""
        
        try:
            target_shape = embeddings.shape
            
            for entry in self.stats_cache.values():
                if (entry.content_type == content_type and
                    entry.data['embedding_shape'][-1] == target_shape[-1]):  # Same model_dim
                    
                    # Check shape compatibility
                    cached_shape = entry.data['embedding_shape']
                    if len(cached_shape) == len(target_shape):
                        entry.update_access()
                        return entry.data['statistics'].copy()
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to find compatible stats: {e}")
            return None
    
    def _evict_stats_if_needed(self):
        """Evict old statistics if cache is full"""
        
        try:
            while len(self.stats_cache) > self.cache_size:
                # Evict least recently used
                oldest_key, _ = self.stats_cache.popitem(last=False)
                
        except Exception as e:
            logger.warning(f"Stats eviction failed: {e}")
    
    def get_embedding_cache_statistics(self) -> Dict:
        """Get embedding statistics cache metrics"""
        
        with self._cache_lock:
            try:
                total_requests = self.cache_stats['cache_hits'] + self.cache_stats['cache_misses']
                
                content_distribution = defaultdict(int)
                for entry in self.stats_cache.values():
                    content_distribution[entry.content_type] += 1
                
                return {
                    'total_cached_stats': len(self.stats_cache),
                    'stats_by_content_type': dict(content_distribution),
                    'cache_hits': self.cache_stats['cache_hits'],
                    'cache_misses': self.cache_stats['cache_misses'],
                    'hit_rate': self.cache_stats['cache_hits'] / max(total_requests, 1),
                    'stats_stored': self.cache_stats['stats_stored'],
                    'stats_retrieved': self.cache_stats['stats_retrieved'],
                    'cache_type': 'EmbeddingStatsCache'
                }
                
            except Exception as e:
                logger.warning(f"Failed to get embedding cache statistics: {e}")
                return {'error': str(e)}


# Unified cache manager

class HierarchicalCacheManager(nn.Module):
    """
    Unified manager for all hierarchical caches with coordinated policies.
    Provides single interface for all caching operations.
    """
    
    def __init__(self, model_dim: int, cache_config: Optional[Dict] = None):
        super().__init__()
        
        self.model_dim = model_dim
        self.cache_config = cache_config or {}
        
        # Initialize individual caches
        self.trajectory_cache = HierarchicalTrajectoryCache(
            model_dim=model_dim,
            cache_levels=self.cache_config.get('trajectory_cache_levels', ['local', 'chunk', 'sequence']),
            cache_sizes=self.cache_config.get('trajectory_cache_sizes'),
            max_age_seconds=self.cache_config.get('trajectory_max_age', 300.0)
        )
        
        self.attention_cache = AttentionPatternCache(
            cache_size=self.cache_config.get('attention_cache_size', 2000),
            similarity_threshold=self.cache_config.get('attention_similarity_threshold', 0.9),
            max_age_seconds=self.cache_config.get('attention_max_age', 600.0)
        )
        
        self.constellation_cache = SplatConstellationCache(
            cache_size=self.cache_config.get('constellation_cache_size', 500),
            model_dim=model_dim
        )
        
        self.embedding_cache = EmbeddingStatsCache(
            cache_size=self.cache_config.get('embedding_cache_size', 1000),
            model_dim=model_dim
        )
        
        # Global statistics
        self.global_stats = {
            'total_cache_operations': 0,
            'cache_manager_hits': 0,
            'cache_manager_misses': 0
        }
        
        logger.info(f"🗄️ Hierarchical cache manager initialized")
        logger.info(f"   Trajectory cache levels: {self.trajectory_cache.cache_levels}")
        logger.info(f"   Total cache components: 4 (trajectory, attention, constellation, embedding)")
    
    def get_comprehensive_statistics(self) -> Dict:
        """Get statistics from all caches"""
        
        try:
            stats = {
                'cache_manager': {
                    'total_operations': self.global_stats['total_cache_operations'],
                    'manager_hits': self.global_stats['cache_manager_hits'],
                    'manager_misses': self.global_stats['cache_manager_misses'],
                    'manager_hit_rate': self.global_stats['cache_manager_hits'] / max(
                        self.global_stats['cache_manager_hits'] + self.global_stats['cache_manager_misses'], 1
                    )
                },
                'trajectory_cache': self.trajectory_cache.get_cache_statistics(),
                'attention_cache': self.attention_cache.get_pattern_statistics(),
                'constellation_cache': self.constellation_cache.get_constellation_statistics(),
                'embedding_cache': self.embedding_cache.get_embedding_cache_statistics()
            }
            
            # Calculate overall cache efficiency
            all_hit_rates = []
            for cache_name, cache_stats in stats.items():
                if cache_name != 'cache_manager' and isinstance(cache_stats, dict):
                    if 'hit_rate' in cache_stats:
                        all_hit_rates.append(cache_stats['hit_rate'])
                    elif 'overall' in cache_stats and 'overall_hit_rate' in cache_stats['overall']:
                        all_hit_rates.append(cache_stats['overall']['overall_hit_rate'])
            
            if all_hit_rates:
                stats['overall_efficiency'] = {
                    'avg_hit_rate': np.mean(all_hit_rates),
                    'min_hit_rate': np.min(all_hit_rates),
                    'max_hit_rate': np.max(all_hit_rates),
                    'cache_balance': np.std(all_hit_rates)  # Lower is better
                }
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get comprehensive cache statistics: {e}")
            return {'error': str(e)}
    
    def clear_all_caches(self):
        """Clear all caches"""
        
        try:
            self.trajectory_cache.clear_cache()
            self.attention_cache.pattern_cache.clear()
            self.constellation_cache.constellation_cache.clear()
            self.embedding_cache.stats_cache.clear()
            
            logger.info("🗑️ All caches cleared")
            
        except Exception as e:
            logger.warning(f"Failed to clear all caches: {e}")
    
    def optimize_cache_sizes(self, usage_stats: Dict):
        """Dynamically optimize cache sizes based on usage patterns"""
        
        try:
            # This could implement intelligent cache size adjustment
            # based on hit rates and usage patterns
            logger.info("🔧 Cache size optimization not yet implemented")
            
        except Exception as e:
            logger.warning(f"Cache optimization failed: {e}")


# Factory function for easy integration

def create_hierarchical_cache_system(model_dim: int, cache_config: Optional[Dict] = None) -> HierarchicalCacheManager:
    """
    Factory function to create complete hierarchical cache system
    
    Args:
        model_dim: Model dimension
        cache_config: Cache configuration dictionary
        
    Returns:
        Configured hierarchical cache manager
    """
    
    try:
        cache_manager = HierarchicalCacheManager(model_dim, cache_config)
        logger.info(f"🚀 Created hierarchical cache system for model_dim={model_dim}")
        return cache_manager
        
    except Exception as e:
        logger.error(f"Failed to create hierarchical cache system: {e}")
        raise
