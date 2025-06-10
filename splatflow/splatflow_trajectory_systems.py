"""
SplatFlow Trajectory Systems Module
Advanced trajectory guidance, caching, and positional embedding systems for SplatFlow.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

def safe_tensor_to_scalar(tensor: torch.Tensor, default: float = 0.0) -> float:
    """Safely convert tensor to scalar with proper error handling"""
    try:
        if tensor.numel() == 1:
            return tensor.item()
        elif tensor.numel() > 1:
            return tensor.mean().item()
        else:
            return default
    except Exception:
        return default


class TrajectoryGuidanceSystem(nn.Module):
    """Advanced trajectory guidance system for goal-directed trajectory steering"""
    
    def __init__(self, model_dim: int, num_layers: int, max_seq_len: int = 2048):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Learnable trajectory guidance networks
        self.guidance_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim * 2, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
                nn.Tanh()
            ) for _ in range(num_layers)
        ])
        
        # Context-aware trajectory targets
        self.target_generator = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Linear(model_dim * 2, model_dim),
            nn.Dropout(0.1)
        )
        
        # Task-specific trajectory modulation
        self.task_modulators = nn.ParameterList([
            nn.Parameter(torch.randn(model_dim) * 0.1)
            for _ in range(num_layers)
        ])
        
        # Guidance strength controllers
        self.guidance_strengths = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5 + i * 0.2))
            for i in range(num_layers)
        ])
        
        logger.info(f"🎯 Trajectory Guidance System initialized for {num_layers} layers")
    
    def compute_contextual_targets(self, embeddings: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Compute context-aware trajectory targets"""
        try:
            batch_size, seq_len, dim = embeddings.shape
            
            if seq_len == 0 or batch_size == 0:
                return torch.zeros_like(embeddings)
            
            context = embeddings.mean(dim=1)
            targets = self.target_generator(context)
            
            if layer_idx < len(self.task_modulators):
                task_mod = self.task_modulators[layer_idx]
                targets = targets + task_mod.unsqueeze(0)
            
            targets = targets.unsqueeze(1).expand(-1, seq_len, -1)
            
            return targets
        except Exception as e:
            logger.warning(f"Failed to compute contextual targets: {e}")
            return torch.zeros_like(embeddings)
    
    def compute_guided_trajectories(self, embeddings: torch.Tensor, 
                                  base_trajectories: torch.Tensor,
                                  layer_idx: int) -> torch.Tensor:
        """Compute trajectories with guidance toward contextual targets"""
        
        try:
            targets = self.compute_contextual_targets(embeddings, layer_idx)
            
            combined_input = torch.cat([embeddings, targets], dim=-1)
            
            if layer_idx < len(self.guidance_networks):
                guidance_vectors = self.guidance_networks[layer_idx](combined_input)
            else:
                guidance_vectors = torch.zeros_like(embeddings)
            
            if layer_idx < len(self.guidance_strengths):
                guidance_strength = torch.sigmoid(self.guidance_strengths[layer_idx])
            else:
                guidance_strength = torch.tensor(0.5)
            
            guided_trajectories = (
                (1 - guidance_strength) * base_trajectories + 
                guidance_strength * guidance_vectors
            )
            
            return guided_trajectories
        except Exception as e:
            logger.warning(f"Failed to compute guided trajectories for layer {layer_idx}: {e}")
            return base_trajectories
    
    def get_guidance_statistics(self) -> Dict:
        """Get guidance system statistics with safe calculations"""
        try:
            strengths = []
            for s in self.guidance_strengths:
                try:
                    strength_val = torch.sigmoid(s).item()
                    strengths.append(strength_val)
                except Exception:
                    strengths.append(0.5)  # Default value
            
            if len(strengths) > 0:
                avg_strength = sum(strengths) / len(strengths)
                max_strength = max(strengths)
                active_layers = sum(1 for s in strengths if s > 0.1)
            else:
                avg_strength = 0.0
                max_strength = 0.0
                active_layers = 0
            
            return {
                'guidance_strengths_by_layer': strengths,
                'avg_guidance_strength': avg_strength,
                'max_guidance_strength': max_strength,
                'guidance_active_layers': active_layers
            }
        except Exception as e:
            logger.warning(f"Failed to get guidance statistics: {e}")
            return {
                'guidance_strengths_by_layer': [],
                'avg_guidance_strength': 0.0,
                'max_guidance_strength': 0.0,
                'guidance_active_layers': 0
            }


class TrajectoryCache(nn.Module):
    """Efficient trajectory caching system for storing and reusing computed trajectories"""
    
    def __init__(self, model_dim: int, cache_size: int = 25, similarity_threshold: float = 0.98):
        super().__init__()
        self.model_dim = model_dim
        self.cache_size = cache_size
        self.similarity_threshold = similarity_threshold
        
        self.trajectory_cache = {}
        self.cache_keys = []
        self.cache_usage_count = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Learned cache key generator
        self.key_generator = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 64),
            nn.Tanh()
        )
        
        logger.info(f"🗄️ Trajectory Cache initialized (size: {cache_size})")
    
    def generate_cache_key(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Generate cache key for given embeddings"""
        try:
            # Take mean across batch and sequence to get single representative vector
            if embeddings.dim() == 3:  # (batch, seq, dim)
                if embeddings.size(0) > 0 and embeddings.size(1) > 0:
                    sequence_repr = embeddings.mean(dim=(0, 1))  # (dim,)
                else:
                    sequence_repr = torch.zeros(embeddings.size(-1), device=embeddings.device)
            else:
                sequence_repr = embeddings.mean(dim=0)  # Handle other cases
            
            cache_key = self.key_generator(sequence_repr.unsqueeze(0))  # (1, key_dim)
            return cache_key.squeeze(0)  # (key_dim,) - single vector, not batched
        except Exception as e:
            logger.warning(f"Failed to generate cache key: {e}")
            return torch.zeros(64, device=embeddings.device)
    
    def find_similar_cached_trajectory(self, cache_key: torch.Tensor) -> Optional[torch.Tensor]:
        """Find similar cached trajectory if it exists"""
        if len(self.cache_keys) == 0:
            return None
        
        try:
            cache_key_tensor = torch.stack(self.cache_keys, dim=0)
            
            # Ensure cache_key is 1D
            if cache_key.dim() > 1:
                cache_key = cache_key.squeeze()
            
            similarities = F.cosine_similarity(
                cache_key.unsqueeze(0), 
                cache_key_tensor, 
                dim=1
            )
            
            max_similarity, best_idx = torch.max(similarities, dim=0)
            
            # Use safe conversion here
            max_sim_val = safe_tensor_to_scalar(max_similarity)
            if max_sim_val > self.similarity_threshold:
                best_key = self.cache_keys[best_idx.item()]
                key_str = str(best_key.detach().cpu().numpy().tobytes())
                
                if key_str in self.trajectory_cache:
                    self.cache_hits += 1
                    self.cache_usage_count[key_str] += 1
                    return self.trajectory_cache[key_str].clone()
            
            self.cache_misses += 1
            return None
            
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            self.cache_misses += 1
            return None
    
    def store_trajectory(self, cache_key: torch.Tensor, trajectory: torch.Tensor):
        """Store trajectory in cache"""
        try:
            if cache_key.dim() > 1:
                cache_key = cache_key.squeeze()
            
            key_str = str(cache_key.detach().cpu().numpy().tobytes())
            
            self.trajectory_cache[key_str] = trajectory.clone().detach()
            self.cache_keys.append(cache_key.clone().detach())
            
            if len(self.cache_keys) > self.cache_size:
                self._evict_least_used()
                
        except Exception as e:
            logger.warning(f"Failed to store trajectory in cache: {e}")
    
    def _evict_least_used(self):
        """Evict least recently used cache entries"""
        try:
            if not self.cache_usage_count:
                if len(self.cache_keys) > 0:
                    oldest_key = self.cache_keys[0]
                    key_str = str(oldest_key.detach().cpu().numpy().tobytes())
                else:
                    return
            else:
                least_used_key = min(self.cache_usage_count.keys(), 
                                    key=self.cache_usage_count.get)
                key_str = least_used_key
                
                # Find corresponding tensor key
                oldest_key = None
                for i, key_tensor in enumerate(self.cache_keys):
                    if str(key_tensor.detach().cpu().numpy().tobytes()) == key_str:
                        oldest_key = self.cache_keys[i]
                        break
                
                if oldest_key is None:
                    return
            
            if key_str in self.trajectory_cache:
                del self.trajectory_cache[key_str]
            if key_str in self.cache_usage_count:
                del self.cache_usage_count[key_str]
            
            self.cache_keys = [k for k in self.cache_keys 
                              if str(k.detach().cpu().numpy().tobytes()) != key_str]
        except Exception as e:
            logger.warning(f"Cache eviction failed: {e}")
    
    def get_cache_statistics(self) -> Dict:
        """Get cache performance statistics with safe division"""
        try:
            total_requests = self.cache_hits + self.cache_misses
            if total_requests > 0:
                hit_rate = self.cache_hits / total_requests
            else:
                hit_rate = 0.0
            
            return {
                'cache_size': len(self.cache_keys),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
        except Exception as e:
            logger.warning(f"Failed to get cache statistics: {e}")
            return {
                'cache_size': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'hit_rate': 0.0,
                'total_requests': 0
            }


class TrajectoryAwarePositionalEmbedding(nn.Module):
    """Enhanced positional embedding integrating trajectory information"""
    
    def __init__(self, model_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.trajectory_position_proj = nn.Linear(model_dim * 2, model_dim)
        
        self.trajectory_directions = nn.Parameter(
            torch.randn(max_seq_len, model_dim // 4) * 0.1
        )
        
        self.position_scales = nn.Parameter(
            torch.ones(max_seq_len) * 0.5
        )
        
        self.register_buffer('trajectory_frequencies', 
                           self._create_trajectory_frequencies())
        
        logger.info(f"📍 Trajectory-Aware Positional Embedding initialized")
    
    def _create_trajectory_frequencies(self) -> torch.Tensor:
        """Create sinusoidal frequencies for trajectory encoding"""
        try:
            position = torch.arange(self.max_seq_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, self.model_dim // 4, 2).float() *
                               -(math.log(10000.0) / max(self.model_dim // 4, 1)))
            
            freqs = torch.zeros(self.max_seq_len, self.model_dim // 4)
            freqs[:, 0::2] = torch.sin(position * div_term)
            freqs[:, 1::2] = torch.cos(position * div_term)
            
            return freqs
        except Exception as e:
            logger.warning(f"Failed to create trajectory frequencies: {e}")
            return torch.zeros(self.max_seq_len, self.model_dim // 4)
    
    def forward(self, input_embeddings: torch.Tensor, 
                trajectories: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute trajectory-aware positional embeddings"""
        try:
            batch_size, seq_len, model_dim = input_embeddings.shape
            device = input_embeddings.device
            
            # Clamp sequence length to avoid index errors
            seq_len = min(seq_len, self.max_seq_len)
            
            positions = torch.arange(seq_len, device=device)
            pos_embeddings = self.position_embedding(positions)
            
            if trajectories is not None and trajectories.size(1) >= seq_len:
                # Enhanced trajectory-aware positioning
                traj_dirs = self.trajectory_directions[:seq_len]
                traj_dir_expanded = traj_dirs.unsqueeze(0).expand(batch_size, -1, -1)
                
                traj_freqs = self.trajectory_frequencies[:seq_len]
                traj_freq_expanded = traj_freqs.unsqueeze(0).expand(batch_size, -1, -1)
                
                pos_scales = torch.sigmoid(self.position_scales[:seq_len])
                scaled_trajectories = trajectories[:, :seq_len, :] * pos_scales.unsqueeze(0).unsqueeze(-1)
                
                # Ensure trajectory component has correct dimensions
                traj_component_dim = min(model_dim//2, scaled_trajectories.size(-1))
                
                trajectory_component = torch.cat([
                    traj_dir_expanded,
                    traj_freq_expanded,
                    scaled_trajectories[:, :, :traj_component_dim]
                ], dim=-1)
                
                # Ensure trajectory_component has the right size for projection
                expected_size = model_dim * 2
                if trajectory_component.size(-1) != expected_size:
                    # Pad or truncate to expected size
                    if trajectory_component.size(-1) < expected_size:
                        padding = torch.zeros(batch_size, seq_len, expected_size - trajectory_component.size(-1), device=device)
                        trajectory_component = torch.cat([trajectory_component, padding], dim=-1)
                    else:
                        trajectory_component = trajectory_component[:, :, :expected_size]
                
                trajectory_pos = self.trajectory_position_proj(
                    torch.cat([input_embeddings[:, :seq_len, :], trajectory_component], dim=-1)
                )
                
                enhanced_pos = pos_embeddings.unsqueeze(0) + 0.3 * trajectory_pos
            else:
                enhanced_pos = pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            
            return enhanced_pos
        except Exception as e:
            logger.warning(f"Trajectory-aware positional embedding failed: {e}")
            # Fallback to basic positional embedding
            batch_size, seq_len, model_dim = input_embeddings.shape
            device = input_embeddings.device
            seq_len = min(seq_len, self.max_seq_len)
            positions = torch.arange(seq_len, device=device)
            pos_embeddings = self.position_embedding(positions)
            return pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)


class EnhancedInterLayerTrajectoryFlow(nn.Module):
    """Enhanced trajectory flow system with guidance, caching, and positional integration"""
    
    def __init__(self, num_layers: int, model_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        
        # Core trajectory flow
        self.trajectory_bridges = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
                nn.Dropout(0.1)
            ) for _ in range(1, num_layers)
        ])
        
        # Trajectory connections
        self.trajectory_strengths = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5 + i * 0.3))
            for i in range(1, num_layers)
        ])
        
        # Enhanced components
        self.guidance_system = TrajectoryGuidanceSystem(model_dim, num_layers, max_seq_len)
        self.trajectory_cache = TrajectoryCache(model_dim)
        self.enhanced_positional = TrajectoryAwarePositionalEmbedding(model_dim, max_seq_len)
        
        # Statistics
        self.layer_trajectories = {}
        self.flow_statistics = {}
        
        logger.info(f"🌟 Enhanced InterLayer trajectory flow initialized")
        logger.info(f"   ✅ Trajectory guidance system")
        logger.info(f"   ✅ Trajectory caching (size: {self.trajectory_cache.cache_size})")
        logger.info(f"   ✅ Enhanced positional embedding")
    
    def compute_enhanced_trajectory_flow(self, layer_idx: int, 
                                       embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute enhanced trajectory flow with all advanced features"""
        try:
            batch_size, seq_len, dim = embeddings.shape
            device = embeddings.device
            
            if batch_size == 0 or seq_len == 0:
                zero_trajectory = torch.zeros_like(embeddings)
                zero_position = torch.zeros_like(embeddings)
                return zero_trajectory, zero_position
            
            # Try cache first
            cache_key = self.trajectory_cache.generate_cache_key(embeddings)
            cached_trajectory = self.trajectory_cache.find_similar_cached_trajectory(cache_key)
            
            if cached_trajectory is not None and cached_trajectory.shape == embeddings.shape:
                base_trajectories = cached_trajectory
            else:
                base_trajectories = self._compute_base_trajectories(layer_idx, embeddings)
                self.trajectory_cache.store_trajectory(cache_key, base_trajectories)
            
            # Apply trajectory guidance
            guided_trajectories = self.guidance_system.compute_guided_trajectories(
                embeddings, base_trajectories, layer_idx
            )
            
            # Compute enhanced positional embeddings
            enhanced_positions = self.enhanced_positional(embeddings, guided_trajectories)
            
            # Store for analysis
            self.layer_trajectories[layer_idx] = guided_trajectories.detach().clone()
            flow_magnitude = safe_tensor_to_scalar(torch.norm(guided_trajectories, dim=-1).mean())
            self.flow_statistics[layer_idx] = flow_magnitude
            
            return guided_trajectories, enhanced_positions
        except Exception as e:
            logger.warning(f"Enhanced trajectory flow computation failed for layer {layer_idx}: {e}")
            # Return safe fallback
            zero_trajectory = torch.zeros_like(embeddings)
            zero_position = torch.zeros_like(embeddings)
            return zero_trajectory, zero_position
    
    def _compute_base_trajectories(self, layer_idx: int, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute base trajectory vectors with enhanced sensitivity"""
        try:
            batch_size, seq_len, dim = embeddings.shape
            device = embeddings.device
            
            if seq_len < 2:
                return torch.zeros_like(embeddings)
            
            trajectories = torch.zeros_like(embeddings)
            
            for pos in range(1, seq_len):
                try:
                    window_start = max(0, pos - 6)
                    
                    if window_start < pos:
                        window_embeddings = embeddings[:, window_start:pos, :]
                        next_embeddings = embeddings[:, window_start+1:pos+1, :]
                        
                        traj_vectors = next_embeddings - window_embeddings
                        traj_magnitudes = torch.norm(traj_vectors, dim=-1, keepdim=True)
                        
                        valid_mask = traj_magnitudes.squeeze(-1) > 1e-8
                        
                        if valid_mask.any():
                            normalized_trajs = torch.zeros_like(traj_vectors)
                            valid_mags = traj_magnitudes[valid_mask]
                            normalized_trajs[valid_mask] = traj_vectors[valid_mask] / (valid_mags + 1e-10)
                            
                            window_size = pos - window_start
                            weights = torch.exp(torch.linspace(-1, 0, window_size, device=device))
                            weights = weights.unsqueeze(0).unsqueeze(-1)
                            
                            depth_scale = 0.2 * (1 + layer_idx * 1.2)
                            
                            weighted_traj = (normalized_trajs * weights).sum(dim=1)
                            weight_sum = weights.sum(dim=1)
                            
                            # Safe division
                            final_traj = weighted_traj / (weight_sum + 1e-8)
                            trajectories[:, pos, :] = final_traj * depth_scale
                except Exception as e:
                    logger.warning(f"Failed to compute trajectory for position {pos}: {e}")
                    continue
            
            return trajectories
        except Exception as e:
            logger.warning(f"Base trajectory computation failed: {e}")
            return torch.zeros_like(embeddings)
    
    def apply_skip_connections(self, layer_trajectories: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply skip connections from Layer 0 to upper layers"""
        try:
            if len(layer_trajectories) == 0:
                return layer_trajectories
            
            enhanced_trajectories = [layer_trajectories[0]]
            base_trajectory = layer_trajectories[0]
            
            for i, (bridge, strength) in enumerate(zip(self.trajectory_bridges, self.trajectory_strengths)):
                layer_idx = i + 1
                
                try:
                    if layer_idx < len(layer_trajectories):
                        original_traj = layer_trajectories[layer_idx]
                        skip_traj = bridge(base_trajectory)
                        
                        gate_strength = torch.sigmoid(strength)
                        combined_traj = (1 - gate_strength) * original_traj + gate_strength * skip_traj
                        
                        enhanced_trajectories.append(combined_traj)
                    else:
                        skip_traj = bridge(base_trajectory)
                        enhanced_trajectories.append(skip_traj)
                except Exception as e:
                    logger.warning(f"Skip connection failed for layer {layer_idx}: {e}")
                    # Use original trajectory if available, otherwise zero
                    if layer_idx < len(layer_trajectories):
                        enhanced_trajectories.append(layer_trajectories[layer_idx])
                    else:
                        enhanced_trajectories.append(torch.zeros_like(base_trajectory))
            
            return enhanced_trajectories
        except Exception as e:
            logger.warning(f"Skip connections failed: {e}")
            return layer_trajectories
    
    def get_comprehensive_statistics(self) -> Dict:
        """Get comprehensive statistics including new features with safe calculations"""
        try:
            # Base statistics with safe calculations
            if len(self.flow_statistics) > 0:
                flow_values = list(self.flow_statistics.values())
                total_layers_with_flow = len([m for m in flow_values if m > 0.001])
                max_flow_magnitude = max(flow_values)
                avg_flow_magnitude = sum(flow_values) / len(flow_values)
            else:
                total_layers_with_flow = 0
                max_flow_magnitude = 0.0
                avg_flow_magnitude = 0.0
            
            base_stats = {
                'layer_flow_magnitudes': dict(self.flow_statistics),
                'total_layers_with_flow': total_layers_with_flow,
                'max_flow_magnitude': max_flow_magnitude,
                'avg_flow_magnitude': avg_flow_magnitude
            }
            
            # Get guidance and cache statistics safely
            try:
                guidance_stats = self.guidance_system.get_guidance_statistics()
            except Exception as e:
                logger.warning(f"Failed to get guidance stats: {e}")
                guidance_stats = {}
            
            try:
                cache_stats = self.trajectory_cache.get_cache_statistics()
            except Exception as e:
                logger.warning(f"Failed to get cache stats: {e}")
                cache_stats = {}
            
            return {
                **base_stats,
                'guidance': guidance_stats,
                'cache': cache_stats
            }
        except Exception as e:
            logger.warning(f"Failed to get comprehensive statistics: {e}")
            return {
                'layer_flow_magnitudes': {},
                'total_layers_with_flow': 0,
                'max_flow_magnitude': 0.0,
                'avg_flow_magnitude': 0.0,
                'guidance': {},
                'cache': {}
            }
