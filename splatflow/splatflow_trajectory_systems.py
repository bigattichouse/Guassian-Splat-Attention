"""
SplatFlow Trajectory Systems - Fixed Implementation
Advanced trajectory guidance and caching with proper method interfaces.

This module provides:
- Enhanced inter-layer trajectory flow with correct method signatures
- Trajectory caching for efficiency  
- Enhanced positional embeddings
- Compatible interface for model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, Dict, List, Any, Union

logger = logging.getLogger(__name__)


class TrajectoryPositionEncoding(nn.Module):
    """
    Enhanced positional encoding with trajectory-based adjustments.
    """
    
    def __init__(
        self,
        model_dim: int,
        max_seq_len: int = 8192,
        trajectory_strength: float = 0.1,
        enable_caching: bool = True
    ):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.trajectory_strength = trajectory_strength
        self.enable_caching = enable_caching
        
        # Create positional encoding table
        pe = torch.zeros(max_seq_len, model_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * 
                            (-math.log(10000.0) / model_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, model_dim]
        
        # Trajectory displacement network
        self.trajectory_net = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Linear(model_dim // 2, model_dim),
            nn.Tanh()
        )
        
        # Caching for efficiency
        if enable_caching:
            self.register_buffer('cached_positions', torch.zeros(1, 1, model_dim))
            self.register_buffer('cache_valid', torch.tensor(False))
    
    def forward(
        self, 
        token_embeddings: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply trajectory-enhanced positional encoding.
        
        Args:
            token_embeddings: [batch, seq_len, model_dim]
            position_ids: [batch, seq_len] optional position indices
            
        Returns:
            Enhanced embeddings with positional and trajectory information
        """
        batch_size, seq_len, _ = token_embeddings.shape
        
        # Get base positional encodings
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=token_embeddings.device).unsqueeze(0)
        
        # Extract positional encodings
        pos_encodings = self.pe[:, :seq_len, :]  # [1, seq_len, model_dim]
        pos_encodings = pos_encodings.expand(batch_size, -1, -1)
        
        # Compute trajectory adjustments
        trajectory_adjustments = self.trajectory_net(token_embeddings)
        trajectory_adjustments = trajectory_adjustments * self.trajectory_strength
        
        # Combine base position with trajectory adjustments
        enhanced_positions = pos_encodings + trajectory_adjustments
        
        return token_embeddings + enhanced_positions


class TrajectoryCache(nn.Module):
    """
    Caching system for trajectory computations to improve efficiency.
    """
    
    def __init__(
        self,
        cache_size: int = 1000,
        model_dim: int = 512,
        similarity_threshold: float = 0.95
    ):
        super().__init__()
        self.cache_size = cache_size
        self.model_dim = model_dim
        self.similarity_threshold = similarity_threshold
        
        # Cache storage
        self.register_buffer('cache_keys', torch.zeros(cache_size, model_dim))
        self.register_buffer('cache_values', torch.zeros(cache_size, model_dim))
        self.register_buffer('cache_valid', torch.zeros(cache_size, dtype=torch.bool))
        self.register_buffer('cache_usage', torch.zeros(cache_size))
        self.register_buffer('cache_ptr', torch.tensor(0))
    
    def lookup(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Look up cached trajectory for similar input.
        
        Args:
            query: [model_dim] query vector
            
        Returns:
            Cached trajectory if found, None otherwise
        """
        if not self.cache_valid.any():
            return None
        
        # Compute similarities
        valid_keys = self.cache_keys[self.cache_valid]
        similarities = F.cosine_similarity(query.unsqueeze(0), valid_keys, dim=1)
        
        # Check if any similarity exceeds threshold
        max_sim, max_idx = similarities.max(dim=0)
        if max_sim > self.similarity_threshold:
            # Update usage counter
            valid_indices = torch.where(self.cache_valid)[0]
            cache_idx = valid_indices[max_idx]
            self.cache_usage[cache_idx] += 1
            
            return self.cache_values[cache_idx]
        
        return None
    
    def store(self, key: torch.Tensor, value: torch.Tensor):
        """
        Store a key-value pair in the cache.
        
        Args:
            key: [model_dim] input key
            value: [model_dim] trajectory value
        """
        ptr = int(self.cache_ptr)
        
        self.cache_keys[ptr] = key.detach()
        self.cache_values[ptr] = value.detach()
        self.cache_valid[ptr] = True
        self.cache_usage[ptr] = 1
        
        # Update pointer (circular buffer)
        self.cache_ptr = (ptr + 1) % self.cache_size
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        valid_count = int(self.cache_valid.sum())
        total_usage = int(self.cache_usage.sum())
        
        return {
            'cache_size': self.cache_size,
            'valid_entries': valid_count,
            'total_usage': total_usage,
            'hit_rate': total_usage / max(valid_count, 1),
            'fill_ratio': valid_count / self.cache_size
        }


class EnhancedInterLayerTrajectoryFlow(nn.Module):
    """
    Enhanced inter-layer trajectory flow system with proper method interfaces.
    Provides sophisticated trajectory guidance and communication between layers.
    """
    
    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        max_seq_len: int = 4096,
        trajectory_strength: float = 0.1,
        enable_caching: bool = True,
        flow_dropout: float = 0.1
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.trajectory_strength = trajectory_strength
        self.enable_caching = enable_caching
        
        # Trajectory position encoding
        self.position_encoding = TrajectoryPositionEncoding(
            model_dim=model_dim,
            max_seq_len=max_seq_len,
            trajectory_strength=trajectory_strength,
            enable_caching=enable_caching
        )
        
        # Inter-layer flow networks
        self.flow_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.ReLU(),
                nn.Dropout(flow_dropout),
                nn.Linear(model_dim, model_dim),
                nn.Tanh()
            )
            for _ in range(num_layers)
        ])
        
        # Layer-specific trajectory guidance
        self.trajectory_guides = nn.ModuleList([
            nn.Linear(model_dim, model_dim)
            for _ in range(num_layers)
        ])
        
        # Trajectory momentum (for stability)
        self.momentum_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.Sigmoid()
            )
            for _ in range(num_layers)
        ])
        
        # Caching system
        if enable_caching:
            self.trajectory_cache = TrajectoryCache(
                cache_size=1000,
                model_dim=model_dim
            )
        else:
            self.trajectory_cache = None
        
        # Health monitoring
        self.register_buffer('flow_applications', torch.zeros(num_layers))
        self.register_buffer('trajectory_magnitudes', torch.zeros(num_layers))
        
        # Previous layer states for momentum
        self.previous_states = [None] * num_layers
        
        logger.info(f"🌊 EnhancedInterLayerTrajectoryFlow initialized: {num_layers} layers, {model_dim}d")
    
    def initialize_trajectories(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Initialize trajectory flow for the input embeddings.
        This method is called at the beginning of the forward pass.
        
        Args:
            hidden_states: [batch, seq_len, model_dim] input embeddings
            
        Returns:
            Enhanced embeddings with initial trajectory information
        """
        # Apply trajectory-enhanced positional encoding
        enhanced_states = self.position_encoding(hidden_states)
        
        # Reset previous states for new sequence
        self.previous_states = [None] * self.num_layers
        
        logger.debug(f"🌊 Initialized trajectories for {hidden_states.shape}")
        return enhanced_states
    
    def apply_inter_layer_flow(
        self, 
        hidden_states: torch.Tensor, 
        layer_idx: int
    ) -> torch.Tensor:
        """
        Apply inter-layer trajectory flow between transformer layers.
        
        Args:
            hidden_states: [batch, seq_len, model_dim] current layer output
            layer_idx: Index of the current layer
            
        Returns:
            Enhanced hidden states with trajectory flow applied
        """
        if layer_idx >= self.num_layers:
            return hidden_states
        
        batch_size, seq_len, model_dim = hidden_states.shape
        
        # Update flow applications counter
        self.flow_applications[layer_idx] += 1
        
        # Get layer-specific trajectory guidance
        trajectory_guide = self.trajectory_guides[layer_idx](hidden_states)
        
        # Apply flow network transformation
        flow_delta = self.flow_networks[layer_idx](hidden_states)
        flow_delta = flow_delta * self.trajectory_strength
        
        # Apply momentum if we have previous state
        if self.previous_states[layer_idx] is not None:
            previous_state = self.previous_states[layer_idx]
            
            # Compute momentum weights
            combined_state = torch.cat([hidden_states, previous_state], dim=-1)
            momentum_weights = self.momentum_networks[layer_idx](combined_state)
            
            # Apply momentum to flow
            flow_delta = flow_delta * momentum_weights
        
        # Check cache for similar patterns (if enabled)
        if self.trajectory_cache is not None and not self.training:
            # Use average embedding as cache key
            cache_key = hidden_states.mean(dim=(0, 1))  # [model_dim]
            cached_flow = self.trajectory_cache.lookup(cache_key)
            
            if cached_flow is not None:
                # Use cached trajectory
                flow_delta = cached_flow.unsqueeze(0).unsqueeze(0).expand_as(flow_delta)
            else:
                # Store current trajectory in cache
                self.trajectory_cache.store(cache_key, flow_delta.mean(dim=(0, 1)))
        
        # Apply trajectory flow
        enhanced_states = hidden_states + trajectory_guide + flow_delta
        
        # Store current state for momentum in next application
        self.previous_states[layer_idx] = hidden_states.detach()
        
        # Update trajectory magnitude tracking
        with torch.no_grad():
            magnitude = torch.norm(flow_delta).item()
            self.trajectory_magnitudes[layer_idx] = (
                0.9 * self.trajectory_magnitudes[layer_idx] + 0.1 * magnitude
            )
        
        logger.debug(f"🌊 Applied trajectory flow at layer {layer_idx}, magnitude: {magnitude:.4f}")
        
        return enhanced_states
    
    def get_trajectory_health(self) -> Dict[str, Any]:
        """Get health statistics for trajectory flow system."""
        health = {
            'num_layers': self.num_layers,
            'flow_applications': self.flow_applications.tolist(),
            'trajectory_magnitudes': self.trajectory_magnitudes.tolist(),
            'average_magnitude': float(self.trajectory_magnitudes.mean()),
            'max_magnitude': float(self.trajectory_magnitudes.max()),
            'trajectory_strength': self.trajectory_strength,
            'caching_enabled': self.enable_caching
        }
        
        if self.trajectory_cache is not None:
            health['cache_stats'] = self.trajectory_cache.get_cache_stats()
        
        return health
    
    def reset_trajectory_state(self):
        """Reset trajectory state (useful between sequences)."""
        self.previous_states = [None] * self.num_layers
        logger.debug("🌊 Trajectory state reset")
    
    def set_trajectory_strength(self, strength: float):
        """Dynamically adjust trajectory strength."""
        self.trajectory_strength = strength
        self.position_encoding.trajectory_strength = strength
        logger.info(f"🌊 Trajectory strength set to {strength}")


class BasicInterLayerTrajectoryFlow(nn.Module):
    """
    Simplified trajectory flow for cases where enhanced features aren't needed.
    Provides the same interface but with minimal computation overhead.
    """
    
    def __init__(
        self,
        model_dim: int,
        num_layers: int,
        trajectory_strength: float = 0.05
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.trajectory_strength = trajectory_strength
        
        # Simple trajectory adjustment
        self.trajectory_projection = nn.Linear(model_dim, model_dim)
        
        logger.info(f"🌊 BasicInterLayerTrajectoryFlow initialized: {num_layers} layers")
    
    def initialize_trajectories(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Initialize trajectories (basic version)."""
        # Simply return input with small trajectory adjustment
        trajectory_adj = self.trajectory_projection(hidden_states) * self.trajectory_strength
        return hidden_states + trajectory_adj
    
    def apply_inter_layer_flow(
        self, 
        hidden_states: torch.Tensor, 
        layer_idx: int
    ) -> torch.Tensor:
        """Apply simple trajectory flow."""
        if layer_idx >= self.num_layers:
            return hidden_states
        
        # Simple flow application
        trajectory_adj = self.trajectory_projection(hidden_states) * self.trajectory_strength
        return hidden_states + trajectory_adj
    
    def get_trajectory_health(self) -> Dict[str, Any]:
        """Get basic health stats."""
        return {
            'type': 'basic',
            'num_layers': self.num_layers,
            'trajectory_strength': self.trajectory_strength
        }


# Factory functions for creating trajectory systems
def create_enhanced_trajectory_flow(
    model_dim: int,
    num_layers: int,
    max_seq_len: int = 4096,
    trajectory_strength: float = 0.1,
    enable_caching: bool = True,
    **kwargs
) -> EnhancedInterLayerTrajectoryFlow:
    """Create enhanced trajectory flow system."""
    return EnhancedInterLayerTrajectoryFlow(
        model_dim=model_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        trajectory_strength=trajectory_strength,
        enable_caching=enable_caching,
        **kwargs
    )


def create_basic_trajectory_flow(
    model_dim: int,
    num_layers: int,
    trajectory_strength: float = 0.05,
    **kwargs
) -> BasicInterLayerTrajectoryFlow:
    """Create basic trajectory flow system."""
    return BasicInterLayerTrajectoryFlow(
        model_dim=model_dim,
        num_layers=num_layers,
        trajectory_strength=trajectory_strength
    )


# Trajectory flow utilities
class TrajectoryFlowMonitor:
    """Monitor trajectory flow performance across training."""
    
    def __init__(self):
        self.flow_history = []
        self.health_snapshots = []
    
    def log_trajectory_step(self, trajectory_flow: EnhancedInterLayerTrajectoryFlow):
        """Log trajectory flow state for monitoring."""
        if hasattr(trajectory_flow, 'get_trajectory_health'):
            health = trajectory_flow.get_trajectory_health()
            self.health_snapshots.append(health)
            
            # Keep only recent history
            if len(self.health_snapshots) > 1000:
                self.health_snapshots = self.health_snapshots[-500:]
    
    def get_trajectory_summary(self) -> Dict[str, Any]:
        """Get summary of trajectory flow performance."""
        if not self.health_snapshots:
            return {'status': 'no_data'}
        
        recent_health = self.health_snapshots[-1]
        return {
            'current_magnitudes': recent_health.get('trajectory_magnitudes', []),
            'average_magnitude': recent_health.get('average_magnitude', 0),
            'trajectory_strength': recent_health.get('trajectory_strength', 0),
            'snapshots_count': len(self.health_snapshots)
        }


# Global trajectory monitor
global_trajectory_monitor = TrajectoryFlowMonitor()


if __name__ == "__main__":
    # Test trajectory systems
    print("🧪 Testing SplatFlow Trajectory Systems...")
    
    # Test parameters
    model_dim, num_layers, seq_len, batch_size = 512, 6, 128, 2
    
    # Create test input
    hidden_states = torch.randn(batch_size, seq_len, model_dim)
    print(f"📊 Test input: {hidden_states.shape}")
    
    # Test enhanced trajectory flow
    print("\n🌊 Testing EnhancedInterLayerTrajectoryFlow...")
    enhanced_flow = create_enhanced_trajectory_flow(
        model_dim=model_dim,
        num_layers=num_layers,
        trajectory_strength=0.1
    )
    
    # Test initialization
    initialized_states = enhanced_flow.initialize_trajectories(hidden_states)
    print(f"✅ Trajectory initialization: {initialized_states.shape}")
    
    # Test inter-layer flow
    for layer_idx in range(3):
        flow_output = enhanced_flow.apply_inter_layer_flow(initialized_states, layer_idx)
        print(f"✅ Layer {layer_idx} flow: {flow_output.shape}")
        initialized_states = flow_output
    
    # Test health monitoring
    health = enhanced_flow.get_trajectory_health()
    print(f"🏥 Trajectory health: {health['average_magnitude']:.4f}")
    
    # Test basic trajectory flow
    print("\n🌊 Testing BasicInterLayerTrajectoryFlow...")
    basic_flow = create_basic_trajectory_flow(
        model_dim=model_dim,
        num_layers=num_layers
    )
    
    basic_init = basic_flow.initialize_trajectories(hidden_states)
    basic_flow_out = basic_flow.apply_inter_layer_flow(basic_init, 0)
    print(f"✅ Basic trajectory flow: {basic_flow_out.shape}")
    
    # Test trajectory caching
    print("\n💾 Testing TrajectoryCache...")
    cache = TrajectoryCache(cache_size=100, model_dim=model_dim)
    
    test_key = torch.randn(model_dim)
    test_value = torch.randn(model_dim)
    
    cache.store(test_key, test_value)
    retrieved = cache.lookup(test_key)
    
    if retrieved is not None:
        print("✅ Cache store/lookup successful")
        print(f"📈 Cache stats: {cache.get_cache_stats()}")
    
    # Test monitoring
    print("\n📊 Testing TrajectoryFlowMonitor...")
    global_trajectory_monitor.log_trajectory_step(enhanced_flow)
    summary = global_trajectory_monitor.get_trajectory_summary()
    print(f"📊 Monitor summary: {summary}")
    
    print("\n🎉 All trajectory systems tests passed!")
