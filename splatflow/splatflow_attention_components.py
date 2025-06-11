"""
SplatFlow Attention Components Module - COMPLETE O(n*k) Optimization
Production-ready O(n*k) implementation with smart sampling throughout.
Maintains all functionality while ensuring linear scaling with sequence length.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
import logging
from typing import Tuple, Optional, Dict, List, Any

from .splatflow_core_systems import DeviceManager, safe_tensor_to_scalar

# Import O(n*k) birth system
try:
    from .splatflow_adaptive_birth import (
        AdaptiveSplatBirthManager,
        OnkCoverageAnalyzer,
        OnkTrajectoryAnalyzer,
        OnkSplatRepositioner
    )
    ONK_BIRTH_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("🚀 O(n*k) birth system loaded successfully")
except ImportError:
    ONK_BIRTH_AVAILABLE = False
    AdaptiveSplatBirthManager = None
    logger = logging.getLogger(__name__)
    logger.warning("⚠️  O(n*k) birth system not available - using basic mode")

# Enhanced coverage algorithms (optional)
try:
    from .enhanced_coverage_algorithms import (
        IntelligentSplatPositioner, 
        CoverageAwareSplatAdapter
    )
    ENHANCED_COVERAGE_AVAILABLE = True
except ImportError:
    ENHANCED_COVERAGE_AVAILABLE = False
    IntelligentSplatPositioner = None
    CoverageAwareSplatAdapter = None


class OnkSamplingStrategy:
    """Centralized O(n*k) sampling strategy with adaptive sizing"""
    
    @staticmethod
    def get_sample_size(seq_len: int, base_ratio: float = 0.2, 
                       min_size: int = 16, max_size: int = 128) -> int:
        """Get optimal sample size for O(n*k) operations"""
        if seq_len <= 64:
            return seq_len  # Use full sequence for short inputs
        
        target_size = max(min_size, int(seq_len * base_ratio))
        return min(target_size, max_size)
    
    @staticmethod
    def smart_sample_indices(seq_len: int, sample_size: int, 
                           strategy: str = "random", device: torch.device = None) -> torch.Tensor:
        """Generate smart sample indices using various strategies"""
        if sample_size >= seq_len:
            return torch.arange(seq_len, device=device)
        
        if strategy == "random":
            return torch.randperm(seq_len, device=device)[:sample_size]
        elif strategy == "strategic":
            # Mix of random and evenly spaced
            half_size = sample_size // 2
            random_indices = torch.randperm(seq_len, device=device)[:half_size]
            spaced_indices = torch.linspace(0, seq_len-1, sample_size - half_size, device=device).long()
            return torch.cat([random_indices, spaced_indices]).unique()
        elif strategy == "evenly_spaced":
            return torch.linspace(0, seq_len-1, sample_size, device=device).long()
        else:
            return torch.randperm(seq_len, device=device)[:sample_size]


class OnkOptimizedTrajectoryFlowSplat:
    """COMPLETE O(n*k) optimized splat with smart sampling throughout"""
    
    def __init__(self, position: torch.Tensor, scale: float, amplitude: float, 
                 splat_id: int, device: torch.device, layer_idx: int = 0):
        self.device = device
        self.id = splat_id
        self.layer_idx = layer_idx
        
        # Core parameters on correct device
        self.position = position.clone().detach().to(device).requires_grad_(True)
        self.log_scale = torch.tensor(math.log(scale), device=device, requires_grad=True)
        self.amplitude = torch.tensor(amplitude, device=device, requires_grad=True)
        
        self.velocity = torch.zeros_like(self.position, device=device)
        self.trajectory_momentum = 0.9
        
        # Enhanced learning rates
        base_lr = 0.12  # Slightly increased for better adaptation
        self.trajectory_learning_rate = base_lr * (1.0 + layer_idx * 0.8)
        
        self.age = 0
        self.usefulness = 2.0 + layer_idx * 0.5
        self.activation_history = []
        self.trajectory_influence_history = []
        
        self.splat_connections = {}
        self.flow_magnitude = 0.0
        
        # O(n*k) optimized embedding statistics
        self.embedding_stats = {
            'mean_magnitude': 2.0,
            'std_magnitude': 1.0,
            'median_inter_token_distance': 4.0,
            'sample_count': 0,
            'last_update_age': 0
        }
        
        # Coverage tracking
        self.coverage_contribution = 0.0
        self.last_coverage_update = 0
        
        # O(n*k) adaptive radius with birth requests
        self.max_healthy_radius = 10.0
        self.expansion_warning_threshold = 8.0
        self.birth_request_callback = None
        self.last_radius_expansion_attempt = 0
        self.birth_reason = None
        
        # O(n*k) optimization parameters
        self.radius_sample_ratio = 0.25  # 25% sample for radius computation
        self.stats_sample_ratio = 0.15   # 15% sample for statistics updates
        self.min_radius_samples = 16
        self.min_stats_samples = 12
        
        # Performance tracking
        self.sample_efficiency_history = []
        self.computation_savings = 0
    
    def update_embedding_statistics_onk(self, sample_embeddings: torch.Tensor):
        """COMPLETE O(n*k) embedding statistics update with efficient sampling"""
        with torch.no_grad():
            try:
                sample_embeddings = sample_embeddings.to(self.device)
                flat_embeddings = sample_embeddings.reshape(-1, sample_embeddings.shape[-1])
                
                if len(flat_embeddings) == 0:
                    return
                
                # O(n*k) OPTIMIZATION: Smart sampling for statistics
                stats_sample_size = OnkSamplingStrategy.get_sample_size(
                    len(flat_embeddings), 
                    self.stats_sample_ratio,
                    self.min_stats_samples,
                    48  # Max for statistics
                )
                
                if stats_sample_size < len(flat_embeddings):
                    sample_indices = OnkSamplingStrategy.smart_sample_indices(
                        len(flat_embeddings), stats_sample_size, "strategic", self.device
                    )
                    flat_embeddings = flat_embeddings[sample_indices]
                    self.computation_savings += len(sample_embeddings.reshape(-1, sample_embeddings.shape[-1])) - stats_sample_size
                
                # Update magnitude statistics with robust computation
                magnitudes = torch.norm(flat_embeddings, dim=-1)
                
                if len(magnitudes) > 0:
                    mean_mag = magnitudes.mean().item()
                    std_mag = magnitudes.std().item() if len(magnitudes) > 1 else 1.0
                    
                    # Exponential moving average for stability
                    alpha = 0.1  # Learning rate for statistics
                    self.embedding_stats['mean_magnitude'] = (
                        alpha * max(0.5, min(mean_mag, 15.0)) + 
                        (1 - alpha) * self.embedding_stats['mean_magnitude']
                    )
                    self.embedding_stats['std_magnitude'] = (
                        alpha * max(0.5, min(std_mag, 8.0)) + 
                        (1 - alpha) * self.embedding_stats['std_magnitude']
                    )
                
                # O(n*k) inter-token distance statistics with very small sample
                if len(flat_embeddings) > 1:
                    distance_sample_size = min(12, len(flat_embeddings))  # Very small for distance computation
                    if distance_sample_size < len(flat_embeddings):
                        distance_indices = torch.randperm(len(flat_embeddings), device=self.device)[:distance_sample_size]
                        distance_tokens = flat_embeddings[distance_indices]
                    else:
                        distance_tokens = flat_embeddings
                    
                    if len(distance_tokens) > 1:
                        # Compute only a subset of pairwise distances
                        max_pairs = min(20, len(distance_tokens) * (len(distance_tokens) - 1) // 2)
                        if max_pairs < len(distance_tokens) * (len(distance_tokens) - 1) // 2:
                            # Sample pairs instead of computing all
                            n_tokens = len(distance_tokens)
                            i_indices = torch.randint(0, n_tokens, (max_pairs,), device=self.device)
                            j_indices = torch.randint(0, n_tokens, (max_pairs,), device=self.device)
                            # Ensure i != j
                            mask = i_indices != j_indices
                            i_indices = i_indices[mask]
                            j_indices = j_indices[mask]
                            
                            if len(i_indices) > 0:
                                distances = torch.norm(distance_tokens[i_indices] - distance_tokens[j_indices], dim=-1)
                            else:
                                distances = torch.tensor([], device=self.device)
                        else:
                            distances = torch.cdist(distance_tokens, distance_tokens)
                            distances = distances[distances > 1e-6]
                        
                        if len(distances) > 0:
                            median_dist = torch.median(distances).item()
                            # Exponential moving average
                            self.embedding_stats['median_inter_token_distance'] = (
                                alpha * max(1.0, min(median_dist, 12.0)) + 
                                (1 - alpha) * self.embedding_stats['median_inter_token_distance']
                            )
                
                self.embedding_stats['sample_count'] += 1
                self.embedding_stats['last_update_age'] = self.age
                
            except Exception as e:
                logger.warning(f"Failed to update O(n*k) embedding statistics for splat {self.id}: {e}")
    
    def compute_adaptive_influence_radius_onk(self, token_embeddings: torch.Tensor) -> float:
        """COMPLETE O(n*k) radius computation with sophisticated sampling"""
        with torch.no_grad():
            try:
                token_embeddings = token_embeddings.to(self.device)
                
                base_radius = self.embedding_stats['median_inter_token_distance'] * 2.2  # Slightly increased
                flat_tokens = token_embeddings.reshape(-1, token_embeddings.shape[-1])
                
                if len(flat_tokens) <= 5:
                    return max(2.0, min(base_radius, self.max_healthy_radius))
                
                # O(n*k) OPTIMIZATION: Adaptive sampling for radius computation
                radius_sample_size = OnkSamplingStrategy.get_sample_size(
                    len(flat_tokens), 
                    self.radius_sample_ratio,
                    self.min_radius_samples,
                    80  # Max for radius computation
                )
                
                if radius_sample_size < len(flat_tokens):
                    # Use strategic sampling for better coverage
                    sample_indices = OnkSamplingStrategy.smart_sample_indices(
                        len(flat_tokens), radius_sample_size, "strategic", self.device
                    )
                    sampled_tokens = flat_tokens[sample_indices]
                    
                    # Track sampling efficiency
                    efficiency = radius_sample_size / len(flat_tokens)
                    self.sample_efficiency_history.append(efficiency)
                    if len(self.sample_efficiency_history) > 50:
                        self.sample_efficiency_history.pop(0)
                else:
                    sampled_tokens = flat_tokens
                
                # Calculate distance from splat to sampled tokens
                splat_expanded = self.position.unsqueeze(0)
                distances_to_splat = torch.norm(sampled_tokens - splat_expanded, dim=-1)
                
                if len(distances_to_splat) > 0:
                    radius_candidates = []
                    
                    # Enhanced percentile computation with multiple percentiles
                    try:
                        if len(distances_to_splat) >= 8:
                            sorted_distances, _ = torch.sort(distances_to_splat)
                            # Try multiple percentiles for robustness
                            percentile_15 = sorted_distances[max(0, len(sorted_distances) * 15 // 100)].item()
                            percentile_25 = sorted_distances[max(0, len(sorted_distances) * 25 // 100)].item()
                            radius_candidates.extend([percentile_15, percentile_25])
                    except Exception:
                        pass
                    
                    # Add adaptive candidates based on embedding statistics
                    radius_candidates.extend([
                        self.embedding_stats['median_inter_token_distance'] * 1.5,
                        self.embedding_stats['median_inter_token_distance'] * 2.0,
                        base_radius,
                        3.5  # Slightly increased minimum
                    ])
                    
                    # Choose radius with intelligent birth requests
                    for radius in sorted(radius_candidates):
                        if radius > self.max_healthy_radius:
                            self._request_onk_birth_for_distant_tokens(
                                sampled_tokens, flat_tokens, radius, radius_sample_size
                            )
                            return min(self.max_healthy_radius, base_radius)
                        
                        if radius > 0:
                            tokens_in_range = (distances_to_splat < radius).sum().item()
                            if tokens_in_range > 0:
                                if radius > self.expansion_warning_threshold:
                                    self._consider_onk_birth_for_expansion(
                                        sampled_tokens, flat_tokens, radius, radius_sample_size
                                    )
                                
                                return max(1.0, min(radius, self.max_healthy_radius))
                
                return max(2.5, min(base_radius, self.max_healthy_radius))  # Slightly increased fallback
                
            except Exception as e:
                logger.warning(f"Failed to compute O(n*k) adaptive radius for splat {self.id}: {e}")
                return min(3.5, self.max_healthy_radius)
    
    def _request_onk_birth_for_distant_tokens(self, sampled_tokens: torch.Tensor, 
                                            all_tokens: torch.Tensor, required_radius: float,
                                            sample_size: int):
        """COMPLETE O(n*k) birth request with accurate cluster size estimation"""
        try:
            if self.birth_request_callback is None:
                return
            
            sampled_tokens = sampled_tokens.to(self.device)
            
            # Find distant tokens in sample
            splat_expanded = self.position.unsqueeze(0)
            distances_to_splat = torch.norm(sampled_tokens - splat_expanded, dim=-1)
            
            distant_mask = distances_to_splat > self.max_healthy_radius
            if distant_mask.any():
                distant_tokens = sampled_tokens[distant_mask]
                
                if len(distant_tokens) > 0:
                    centroid = distant_tokens.mean(dim=0).to(self.device)
                    
                    # Improved cluster size estimation
                    sample_ratio = sample_size / max(len(all_tokens), 1)
                    raw_estimate = len(distant_tokens) / max(sample_ratio, 0.01)
                    
                    # Apply confidence scaling based on sample size
                    confidence = min(1.0, sample_size / 64)  # Full confidence at 64+ samples
                    estimated_cluster_size = int(raw_estimate * confidence + len(distant_tokens) * (1 - confidence))
                    
                    urgency = (len(distant_tokens) / len(sampled_tokens)) * (required_radius / self.max_healthy_radius)
                    urgency = min(1.0, urgency)  # Cap urgency
                    
                    self.birth_request_callback(
                        position=centroid,
                        reason="onk_radius_limit_reached",
                        urgency=urgency,
                        parent_splat_id=self.id,
                        token_cluster_size=estimated_cluster_size
                    )
                    
                    self.last_radius_expansion_attempt = self.age
                    
                    # Even more reduced debug spam
                    if self.id == 0 and random.random() < 0.002:  # Only 0.2% of the time
                        print(f"    🚀 O(n*k) Splat {self.id}: Birth requested for ~{estimated_cluster_size} distant tokens "
                              f"(confidence={confidence:.2f})")
                              
        except Exception as e:
            logger.warning(f"Failed to request O(n*k) birth for splat {self.id}: {e}")
    
    def _consider_onk_birth_for_expansion(self, sampled_tokens: torch.Tensor, 
                                        all_tokens: torch.Tensor, proposed_radius: float,
                                        sample_size: int):
        """COMPLETE O(n*k) expansion consideration with improved heuristics"""
        try:
            if (self.birth_request_callback is None or 
                self.age - self.last_radius_expansion_attempt < 8):  # Slightly reduced cooldown
                return
            
            sampled_tokens = sampled_tokens.to(self.device)
            
            current_scale = torch.exp(self.log_scale).item() if hasattr(self, 'log_scale') else 3.0
            if proposed_radius > current_scale * 1.4:  # Slightly reduced threshold
                
                splat_expanded = self.position.unsqueeze(0)
                distances_to_splat = torch.norm(sampled_tokens - splat_expanded, dim=-1)
                
                moderate_mask = ((distances_to_splat > current_scale * 1.1) & 
                               (distances_to_splat < self.max_healthy_radius))
                
                if moderate_mask.sum().item() >= 2:  # Reduced threshold
                    moderate_tokens = sampled_tokens[moderate_mask]
                    centroid = moderate_tokens.mean(dim=0).to(self.device)
                    
                    # Improved cluster size estimation
                    sample_ratio = sample_size / max(len(all_tokens), 1)
                    confidence = min(1.0, sample_size / 48)
                    raw_estimate = len(moderate_tokens) / max(sample_ratio, 0.01)
                    estimated_cluster_size = int(raw_estimate * confidence + len(moderate_tokens) * (1 - confidence))
                    
                    urgency = 0.35  # Slightly increased urgency for moderate gaps
                    
                    self.birth_request_callback(
                        position=centroid,
                        reason="onk_expansion_prevention",
                        urgency=urgency,
                        parent_splat_id=self.id,
                        token_cluster_size=estimated_cluster_size
                    )
                    
                    self.last_radius_expansion_attempt = self.age
                    
        except Exception as e:
            logger.warning(f"Failed to consider O(n*k) birth for splat {self.id}: {e}")
    
    def update_with_onk_trajectory_flow(self, layer_trajectory: torch.Tensor, 
                                      token_embeddings: torch.Tensor, 
                                      splat_network: Optional[Dict] = None,
                                      epoch: int = 0):
        """COMPLETE O(n*k) trajectory flow update with all optimizations"""
        self.age += 1
        
        try:
            layer_trajectory = layer_trajectory.to(self.device)
            token_embeddings = token_embeddings.to(self.device)
            
            # Update embedding statistics periodically with O(n*k) sampling
            if self.age % 40 == 0:  # Slightly more frequent updates
                self.update_embedding_statistics_onk(token_embeddings)
            
            # Compute trajectory influence with O(n*k) methods
            trajectory_influence = self.compute_onk_trajectory_influence(
                layer_trajectory, token_embeddings, epoch
            )
            
            influence_magnitude = safe_tensor_to_scalar(torch.norm(trajectory_influence))
            self.trajectory_influence_history.append(influence_magnitude)
            if len(self.trajectory_influence_history) > 100:
                self.trajectory_influence_history.pop(0)
            
            # Apply inter-splat flow if available
            if splat_network:
                try:
                    inter_splat_flow = self.compute_inter_splat_flow_onk(splat_network)
                    trajectory_influence = trajectory_influence + 0.35 * inter_splat_flow  # Slightly reduced
                except Exception as e:
                    logger.warning(f"Inter-splat flow computation failed for splat {self.id}: {e}")
            
            # Enhanced adaptive learning rate with epoch awareness
            adaptive_lr = self.trajectory_learning_rate
            if self.layer_idx > 0:
                layer_boost = 1.0 + self.layer_idx * 1.3  # Slightly reduced
                adaptive_lr *= layer_boost
            
            # Progressive learning rate schedule
            if epoch < 8:
                adaptive_lr *= 1.4  # Reduced early boost
            elif epoch < 20:
                adaptive_lr *= 1.1  # Modest mid-training boost
            
            self.velocity = (self.trajectory_momentum * self.velocity + 
                            adaptive_lr * trajectory_influence).to(self.device)
            
            # Improved movement bounds with adaptive scaling
            base_max_vel = self.embedding_stats['std_magnitude'] * 0.75
            max_vel = max(0.4, min(base_max_vel, 2.5))  # Slightly more conservative
            self.velocity = torch.clamp(self.velocity, -max_vel, max_vel)
            
            # Update position with improved bounds
            with torch.no_grad():
                new_position = self.position + self.velocity
                bound_scale = (self.embedding_stats['mean_magnitude'] + 
                             2.5 * self.embedding_stats['std_magnitude'])  # Slightly reduced
                bounds = max(6.0, min(bound_scale, 18.0))  # Tighter bounds
                self.position.data = torch.clamp(new_position, -bounds, bounds)
            
            # Enhanced progressive usefulness criteria with improved thresholds
            if len(self.trajectory_influence_history) > 0:
                recent_influence = np.mean(self.trajectory_influence_history[-15:])  # Slightly shorter window
            else:
                recent_influence = 0.0
            
            # More nuanced progressive thresholds
            if epoch < 4:
                baseline_influence = 5e-8  # Very lenient early on
            elif epoch < 12:
                baseline_influence = 5e-7  # Moderate
            elif epoch < 25:
                baseline_influence = 2e-6  # Normal
            else:
                baseline_influence = 5e-6  # Mature threshold
            
            usefulness_delta = 0.08 * (recent_influence - baseline_influence)  # Slightly reduced
            self.usefulness = np.clip(self.usefulness + usefulness_delta, 0.05, 4.5)
            self.flow_magnitude = influence_magnitude
            
        except Exception as e:
            logger.warning(f"Failed to update splat {self.id} with O(n*k) trajectory flow: {e}")
    
    def compute_onk_trajectory_influence(self, layer_trajectory: torch.Tensor, 
                                       token_embeddings: torch.Tensor,
                                       epoch: int = 0) -> torch.Tensor:
        """COMPLETE O(n*k) trajectory influence computation with all optimizations"""
        try:
            batch_size, seq_len, dim = token_embeddings.shape
            
            if seq_len == 0 or batch_size == 0:
                return torch.zeros_like(self.position, device=self.device)
            
            layer_trajectory = layer_trajectory.to(self.device)
            token_embeddings = token_embeddings.to(self.device)
            
            # Use O(n*k) adaptive influence radius
            influence_radius = self.compute_adaptive_influence_radius_onk(token_embeddings)
            
            splat_expanded = self.position.unsqueeze(0).unsqueeze(0)
            distances = torch.norm(token_embeddings - splat_expanded, dim=-1)
            
            # Extremely reduced debug spam with better insights
            if (distances.numel() > 0 and self.id == 0 and 
                random.random() < 0.0005 and epoch % 5 == 0):  # Only every 5th epoch, 0.05% chance
                min_dist = torch.min(distances).item()
                mean_dist = torch.mean(distances).item()
                efficiency = np.mean(self.sample_efficiency_history) if self.sample_efficiency_history else 1.0
                print(f"    🚀 O(n*k) Splat {self.id}: min_dist={min_dist:.3f}, mean_dist={mean_dist:.3f}, "
                      f"radius={influence_radius:.3f}, efficiency={efficiency:.2f}")
            
            # Progressive radius expansion with improved logic
            influence_mask = distances < influence_radius
            tokens_in_influence = influence_mask.sum().item()
            
            if tokens_in_influence == 0:
                # More intelligent expansion with better factors
                expansion_factors = [1.3, 1.8, 2.5, 3.5]  # More gradual expansion
                for expansion_factor in expansion_factors:
                    expanded_radius = influence_radius * expansion_factor
                    influence_mask = distances < expanded_radius
                    if influence_mask.any():
                        influence_radius = expanded_radius
                        tokens_in_influence = influence_mask.sum().item()
                        if (self.id == 0 and random.random() < 0.002 and 
                            epoch % 10 == 0):  # Very rare logging
                            print(f"    🚀 O(n*k) Splat {self.id}: expanded radius to {expanded_radius:.3f}, "
                                  f"found {tokens_in_influence} tokens")
                        break
                else:
                    # O(n*k) emergency repositioning with improved sampling
                    if epoch < 4 and distances.numel() > 0:  # Only very early epochs
                        flat_distances = distances.reshape(-1)
                        flat_tokens = token_embeddings.reshape(-1, dim)
                        
                        # Smarter emergency repositioning with sampling
                        if len(flat_distances) > 64:
                            # Sample for emergency repositioning
                            emergency_sample_size = min(64, len(flat_distances))
                            sample_indices = torch.randperm(len(flat_distances), device=self.device)[:emergency_sample_size]
                            sampled_distances = flat_distances[sample_indices]
                            closest_idx = torch.argmin(sampled_distances)
                            global_closest_idx = sample_indices[closest_idx]
                        else:
                            global_closest_idx = torch.argmin(flat_distances)
                        
                        closest_token = flat_tokens[global_closest_idx]
                        
                        with torch.no_grad():
                            direction = closest_token - self.position
                            # More conservative emergency movement
                            self.position.data += 0.25 * direction
                        
                        if (self.id == 0 and random.random() < 0.005 and 
                            epoch % 20 == 0):  # Very rare emergency logging
                            print(f"    🚨 O(n*k) EMERGENCY: Moved splat {self.id} toward sampled closest token")
                    
                    return torch.zeros_like(self.position, device=self.device)
            
            # Enhanced influence computation with better weighting
            proximity_weights = torch.exp(-distances / (influence_radius * 0.35))  # Slightly tighter
            proximity_weights = proximity_weights * influence_mask.float()
            
            # More sophisticated trajectory magnitude weighting
            traj_magnitudes = torch.norm(layer_trajectory, dim=-1)
            magnitude_weights = torch.sigmoid(traj_magnitudes * 0.9)  # Slightly more sensitive
            
            # Combined weighting with normalization
            total_weights = proximity_weights * magnitude_weights
            total_weight_sum = safe_tensor_to_scalar(total_weights.sum())
            
            if (self.id == 0 and random.random() < 0.0002 and 
                epoch % 15 == 0):  # Extremely rare detailed logging
                print(f"    🚀 O(n*k) Splat {self.id}: tokens_in_range={tokens_in_influence}, "
                      f"total_weight={total_weight_sum:.6f}, radius={influence_radius:.3f}")
            
            if total_weight_sum < 1e-12:
                return torch.zeros_like(self.position, device=self.device)
            
            # Compute weighted trajectory influence with improved normalization
            weighted_trajectories = layer_trajectory * total_weights.unsqueeze(-1)
            influence_vector = weighted_trajectories.sum(dim=(0, 1)) / max(total_weight_sum, 1e-12)
            
            # Enhanced layer-specific boost with epoch awareness
            layer_boost = 1.0 + self.layer_idx * 1.1  # Slightly reduced
            if epoch < 10:
                layer_boost *= 1.1  # Small boost for early training
            
            influence_vector = influence_vector * layer_boost
            
            return influence_vector.to(self.device)
            
        except Exception as e:
            logger.warning(f"Failed to compute O(n*k) trajectory influence for splat {self.id}: {e}")
            return torch.zeros_like(self.position, device=self.device)
    
    def compute_inter_splat_flow_onk(self, splat_network: Dict) -> torch.Tensor:
        """Enhanced inter-splat flow with O(n*k) optimizations"""
        inter_flow = torch.zeros_like(self.position, device=self.device)
        
        try:
            # Limit inter-splat computations for very large networks
            max_interactions = min(32, len(splat_network))
            if len(splat_network) > max_interactions:
                # Sample subset of splats for interaction
                splat_ids = list(splat_network.keys())
                selected_ids = random.sample([sid for sid in splat_ids if sid != self.id], 
                                           max_interactions - 1)
                selected_splats = [splat_network[sid] for sid in selected_ids]
            else:
                selected_splats = [s for s in splat_network.values() if s.id != self.id]
            
            for other_splat in selected_splats:
                try:
                    other_position = other_splat.position.to(self.device)
                    direction = other_position - self.position
                    distance = torch.norm(direction)
                    
                    if distance > 1e-6:
                        normalized_direction = direction / max(distance, 1e-6)
                        optimal_distance = 2.2 + self.layer_idx * 0.4  # Slightly adjusted
                        
                        if distance > optimal_distance:
                            flow_strength = 0.18 * (distance - optimal_distance)  # Slightly reduced
                            inter_flow += flow_strength * normalized_direction
                        else:
                            flow_strength = 0.25 * (optimal_distance - distance)  # Slightly reduced
                            inter_flow -= flow_strength * normalized_direction
                except Exception as e:
                    logger.warning(f"Inter-splat flow failed between {self.id} and {other_splat.id}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Inter-splat flow computation failed for splat {self.id}: {e}")
        
        return inter_flow.to(self.device)
    
    def is_healthy(self, epoch: int = 0) -> bool:
        """Enhanced health check with improved progressive criteria"""
        try:
            if len(self.trajectory_influence_history) > 0:
                recent_influence = np.mean(self.trajectory_influence_history[-12:])  # Slightly shorter window
            else:
                recent_influence = 0.0
            
            # More sophisticated progressive thresholds
            if epoch < 4:
                influence_threshold = 3e-8  # Very lenient for early training
                usefulness_threshold = 0.005
            elif epoch < 10:
                influence_threshold = 3e-7  # Moderate early phase
                usefulness_threshold = 0.03
            elif epoch < 20:
                influence_threshold = 1e-6  # Normal training
                usefulness_threshold = 0.08
            else:
                influence_threshold = 3e-6  # Mature training
                usefulness_threshold = 0.12
            
            is_influence_healthy = recent_influence > influence_threshold
            is_usefulness_healthy = self.usefulness > usefulness_threshold
            
            # Additional health factors
            is_position_stable = torch.norm(self.velocity).item() < 5.0  # Prevent runaway splats
            
            return is_influence_healthy and is_usefulness_healthy and is_position_stable
        except Exception as e:
            logger.warning(f"Health check failed for splat {self.id}: {e}")
            return True  # Default to healthy to avoid premature removal
    
    def get_production_stats(self, epoch: int = 0) -> Dict:
        """COMPLETE production-level statistics with O(n*k) metrics"""
        try:
            if len(self.trajectory_influence_history) > 0:
                recent_influence = np.mean(self.trajectory_influence_history[-20:])
                avg_influence = np.mean(self.trajectory_influence_history)
                influence_trend = (self.trajectory_influence_history[-5:] if len(self.trajectory_influence_history) >= 5 
                                 else self.trajectory_influence_history)
                trend_slope = np.polyfit(range(len(influence_trend)), influence_trend, 1)[0] if len(influence_trend) > 1 else 0.0
            else:
                recent_influence = 0.0
                avg_influence = 0.0
                trend_slope = 0.0
            
            avg_sample_efficiency = (np.mean(self.sample_efficiency_history) 
                                   if self.sample_efficiency_history else 1.0)
            
            return {
                'layer_idx': self.layer_idx,
                'age': self.age,
                'usefulness': self.usefulness,
                'recent_trajectory_influence': recent_influence,
                'avg_trajectory_influence': avg_influence,
                'influence_trend_slope': trend_slope,
                'flow_magnitude': self.flow_magnitude,
                'velocity_magnitude': torch.norm(self.velocity).item(),
                'position_magnitude': torch.norm(self.position).item(),
                'trajectory_learning_rate': self.trajectory_learning_rate,
                'is_healthy': self.is_healthy(epoch),
                'embedding_stats': self.embedding_stats,
                'coverage_contribution': self.coverage_contribution,
                'birth_reason': self.birth_reason,
                'max_radius': self.max_healthy_radius,
                'optimization': 'O(n*k) algorithms',
                'sample_efficiency': avg_sample_efficiency,
                'computation_savings': self.computation_savings,
                'sample_efficiency_history_length': len(self.sample_efficiency_history)
            }
        except Exception as e:
            logger.warning(f"Failed to get stats for splat {self.id}: {e}")
            return {
                'layer_idx': self.layer_idx,
                'age': self.age,
                'usefulness': 1.0,
                'recent_trajectory_influence': 0.0,
                'avg_trajectory_influence': 0.0,
                'influence_trend_slope': 0.0,
                'flow_magnitude': 0.0,
                'velocity_magnitude': 0.0,
                'position_magnitude': 0.0,
                'trajectory_learning_rate': self.trajectory_learning_rate,
                'is_healthy': True,
                'embedding_stats': self.embedding_stats,
                'coverage_contribution': 0.0,
                'birth_reason': self.birth_reason,
                'max_radius': self.max_healthy_radius,
                'optimization': 'O(n*k) algorithms',
                'sample_efficiency': 1.0,
                'computation_savings': 0,
                'sample_efficiency_history_length': 0
            }


class OnkOptimizedSplatFlowAttention(nn.Module):
    """COMPLETE O(n*k) optimized SplatFlow attention with all production features"""
    
    def __init__(self, model_dim: int, num_splats: int = 20, max_splats: int = 64,
                 dropout: float = 0.1, layer_idx: int = 0):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_splats = num_splats
        self.max_splats = max_splats
        self.layer_idx = layer_idx
        self.dropout = dropout
        
        self.trajectory_computer = None
        
        self.splats = []
        self.adaptation_enabled = True
        self.adaptation_frequency = 1
        self.forward_count = 0
        
        self.min_splats = max(4, num_splats // 4)
        self.recovery_enabled = True
        self.last_recovery_epoch = 0
        
        # Neural network components
        self.token_value_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.output_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Enhanced trajectory strength with layer-aware initialization
        initial_strength = 0.55 + layer_idx * 0.25  # Slightly reduced
        self.trajectory_strength = nn.Parameter(torch.tensor(initial_strength))
        
        # O(n*k) birth management system
        if ONK_BIRTH_AVAILABLE:
            try:
                self.birth_manager = AdaptiveSplatBirthManager(
                    max_splats=max_splats,
                    max_radius=10.0,
                    max_births_per_epoch=2,
                    birth_cooldown=2  # Slightly reduced cooldown
                )
                logger.info(f"🚀 O(n*k) birth system enabled for layer {layer_idx}")
            except Exception as e:
                logger.warning(f"Failed to initialize O(n*k) birth manager: {e}")
                self.birth_manager = None
        else:
            self.birth_manager = None
            logger.warning(f"O(n*k) birth system not available for layer {layer_idx}")
        
        # Enhanced coverage components (optional)
        if ENHANCED_COVERAGE_AVAILABLE:
            try:
                self.coverage_positioner = IntelligentSplatPositioner(
                    model_dim=model_dim,
                    num_splats=num_splats,
                    layer_idx=layer_idx
                )
                self.coverage_adapter = CoverageAwareSplatAdapter(self.coverage_positioner)
                logger.info(f"🎯 Enhanced coverage optimization enabled for layer {layer_idx}")
            except Exception as e:
                logger.warning(f"Failed to initialize coverage components: {e}")
                self.coverage_positioner = None
                self.coverage_adapter = None
        else:
            self.coverage_positioner = None
            self.coverage_adapter = None
        
        # Performance tracking
        self.attention_computation_times = []
        self.total_computations = 0
        self.total_savings = 0
        
        # Initialize splats and weights
        self._initialize_onk_splats()
        self._init_weights()
        
        logger.info(f"🚀 COMPLETE O(n*k) SplatFlow attention initialized for layer {layer_idx}")
    
    def _initialize_onk_splats(self):
        """Initialize splats with O(n*k) optimizations"""
        device = DeviceManager.get_primary_device()
        self.splats = []
        
        for i in range(self.num_splats):
            # Improved initialization with better distribution
            position = torch.randn(self.model_dim, device=device) * 0.4  # Slightly reduced variance
            scale = 2.2 + torch.rand(1).item() * 0.8  # Slightly adjusted range
            amplitude = 1.1 + torch.rand(1).item() * 0.4  # Slightly adjusted range
            
            splat = OnkOptimizedTrajectoryFlowSplat(position, scale, amplitude, i, device, self.layer_idx)
            
            # Connect O(n*k) birth request callback
            if self.birth_manager:
                try:
                    splat.birth_request_callback = self.birth_manager.request_splat_birth
                except Exception as e:
                    logger.warning(f"Failed to set birth callback for splat {i}: {e}")
            
            self.splats.append(splat)
        
        logger.info(f"🚀 Initialized {len(self.splats)} COMPLETE O(n*k) optimized splats for layer {self.layer_idx}")
    
    def fix_splat_positioning_based_on_embeddings(self, sample_embeddings: torch.Tensor):
        """COMPLETE O(n*k) splat positioning with all optimizations"""
        with torch.no_grad():
            try:
                device = self.splats[0].device if self.splats else DeviceManager.get_primary_device()
                sample_embeddings = sample_embeddings.to(device)
                
                print(f"🚀 COMPLETE O(n*k): Positioning splats for layer {self.layer_idx}...")
                
                # Use enhanced coverage positioning if available
                if self.coverage_positioner:
                    try:
                        intelligent_positions = self.coverage_positioner.generate_intelligent_positions(
                            sample_embeddings, num_splats=len(self.splats)
                        )
                        
                        for i, splat in enumerate(self.splats):
                            if i < len(intelligent_positions):
                                splat.position.data = intelligent_positions[i].to(device)
                                print(f"   ✅ Splat {i}: intelligently positioned at magnitude {torch.norm(splat.position).item():.3f}")
                        
                        logger.info(f"✅ COMPLETE O(n*k) ENHANCED: Layer {self.layer_idx} splats positioned with intelligent clustering")
                        return
                        
                    except Exception as e:
                        logger.warning(f"Enhanced positioning failed: {e}, falling back to O(n*k) method")
                
                # COMPLETE O(n*k) fallback positioning
                self._complete_onk_fallback_positioning(sample_embeddings, device)
                
            except Exception as e:
                logger.error(f"Failed to fix COMPLETE O(n*k) splat positioning for layer {self.layer_idx}: {e}")
    
    def _complete_onk_fallback_positioning(self, sample_embeddings: torch.Tensor, device: torch.device):
        """COMPLETE O(n*k) fallback positioning with all optimizations"""
        flat_embeddings = sample_embeddings.reshape(-1, self.model_dim)
        
        if len(flat_embeddings) == 0:
            logger.warning(f"Empty embeddings for layer {self.layer_idx}, skipping positioning")
            return
        
        # COMPLETE O(n*k) OPTIMIZATION: Advanced sampling for positioning
        positioning_sample_size = OnkSamplingStrategy.get_sample_size(
            len(flat_embeddings), 
            0.3,  # Higher ratio for positioning
            32,   # Higher minimum for better coverage
            160   # Higher maximum for positioning
        )
        
        if positioning_sample_size < len(flat_embeddings):
            sample_indices = OnkSamplingStrategy.smart_sample_indices(
                len(flat_embeddings), positioning_sample_size, "strategic", device
            )
            sampled_embeddings = flat_embeddings[sample_indices]
            self.total_savings += len(flat_embeddings) - positioning_sample_size
        else:
            sampled_embeddings = flat_embeddings
        
        # Enhanced statistics calculation
        mean_pos = sampled_embeddings.mean(dim=0)
        std_pos = sampled_embeddings.std(dim=0)
        std_pos = torch.clamp(std_pos, min=0.15, max=5.0)  # Better bounds
        
        print(f"🚀 COMPLETE O(n*k) Layer {self.layer_idx} embedding analysis:")
        print(f"   Sample token count: {len(sampled_embeddings)} / {len(flat_embeddings)}")
        print(f"   Sampling efficiency: {len(sampled_embeddings) / len(flat_embeddings):.2f}")
        print(f"   Mean magnitude: {torch.norm(mean_pos).item():.3f}")
        print(f"   Std magnitude: {torch.norm(std_pos).item():.3f}")
        
        # Update embedding statistics for all splats
        for splat in self.splats:
            splat.update_embedding_statistics_onk(sample_embeddings.unsqueeze(0))
        
        # COMPLETE O(n*k) positioning strategy with improved distribution
        for i, splat in enumerate(self.splats):
            try:
                if i < len(sampled_embeddings):
                    # Use sampled token as base with improved distribution
                    base_pos = sampled_embeddings[i % len(sampled_embeddings)]
                else:
                    # Sample from improved distribution
                    base_pos = mean_pos + torch.randn_like(mean_pos, device=device) * std_pos * 0.7
                
                # Enhanced perturbation for better coverage
                perturbation_strength = torch.norm(std_pos) * 0.25  # Slightly reduced
                perturbation = torch.randn_like(base_pos, device=device) * perturbation_strength
                new_position = base_pos + perturbation
                
                # Apply intelligent bounds
                position_bounds = torch.norm(mean_pos) + 4 * torch.norm(std_pos)
                position_bounds = max(8.0, min(position_bounds, 20.0))
                new_position = torch.clamp(new_position, -position_bounds, position_bounds)
                
                splat.position.data = new_position.to(device)
                splat.usefulness = 2.0 + random.uniform(-0.2, 0.2)  # Add small variation
                splat.velocity.zero_()
                splat.trajectory_influence_history.clear()
                
                print(f"   ✅ Splat {i}: COMPLETE O(n*k) positioned at magnitude {torch.norm(splat.position).item():.3f}")
            except Exception as e:
                logger.warning(f"Failed to reposition splat {i}: {e}")
    
    def progressive_splat_repositioning(self, layer_embeddings: torch.Tensor, epoch: int):
        """COMPLETE O(n*k) progressive splat repositioning"""
        if epoch % 3 == 0 and epoch > 0:
            with torch.no_grad():
                try:
                    # Use COMPLETE O(n*k) repositioning
                    if (self.birth_manager and 
                        hasattr(self.birth_manager, 'apply_progressive_repositioning_onk')):
                        
                        repositioned = self.birth_manager.apply_progressive_repositioning_onk(
                            self.splats, layer_embeddings
                        )
                        
                        if repositioned > 0:
                            print(f"    🚀 Layer {self.layer_idx}: COMPLETE O(n*k) repositioned {repositioned} splats (epoch {epoch})")
                    else:
                        # Fallback O(n*k) repositioning
                        repositioned = self._fallback_onk_repositioning(layer_embeddings)
                        if repositioned > 0:
                            print(f"    🚀 Layer {self.layer_idx}: Fallback O(n*k) repositioned {repositioned} splats (epoch {epoch})")
                        
                except Exception as e:
                    logger.warning(f"COMPLETE O(n*k) progressive repositioning failed for layer {self.layer_idx}: {e}")
    
    def _fallback_onk_repositioning(self, layer_embeddings: torch.Tensor) -> int:
        """Fallback O(n*k) repositioning when birth manager unavailable"""
        repositioned_count = 0
        
        try:
            device = layer_embeddings.device
            batch_size, seq_len, embed_dim = layer_embeddings.shape
            
            if seq_len == 0 or not self.splats:
                return 0
            
            # O(n*k) sampling for repositioning
            sample_size = OnkSamplingStrategy.get_sample_size(seq_len, 0.25, 24, 80)
            sample_indices = OnkSamplingStrategy.smart_sample_indices(
                seq_len, sample_size, "strategic", device
            )
            sampled_tokens = layer_embeddings[0][sample_indices]  # First batch
            
            for splat in self.splats:
                try:
                    # Find closest sampled tokens to this splat
                    distances = torch.norm(sampled_tokens - splat.position.unsqueeze(0), dim=-1)
                    
                    # Get top-k closest tokens for better positioning
                    k_closest = min(6, len(distances))  # Slightly reduced
                    _, closest_indices = torch.topk(distances, k_closest, largest=False)
                    closest_tokens = sampled_tokens[closest_indices]
                    
                    # Move toward centroid of closest tokens
                    target_position = closest_tokens.mean(dim=0)
                    direction = target_position - splat.position
                    
                    # Adaptive movement strength
                    move_strength = 0.04  # More conservative
                    
                    with torch.no_grad():
                        splat.position.data += move_strength * direction
                        repositioned_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to reposition splat {splat.id}: {e}")
                    continue
            
            return repositioned_count
            
        except Exception as e:
            logger.warning(f"Fallback repositioning failed: {e}")
            return 0
    
    def emergency_splat_rescue(self, layer_embeddings: torch.Tensor, epoch: int):
        """COMPLETE O(n*k) emergency system with all optimizations"""
        try:
            healthy_count = sum(1 for splat in self.splats if splat.is_healthy(epoch))
            
            # More conservative emergency trigger
            if len(self.splats) > 0 and healthy_count < len(self.splats) * 0.12:  # 12% threshold
                print(f"    🚨 COMPLETE O(n*k) EMERGENCY Layer {self.layer_idx}: Only {healthy_count}/{len(self.splats)} healthy splats")
                
                with torch.no_grad():
                    # Use COMPLETE O(n*k) emergency repositioning
                    if (self.birth_manager and 
                        hasattr(self.birth_manager, 'apply_emergency_repositioning_onk')):
                        
                        rescued = self.birth_manager.apply_emergency_repositioning_onk(
                            self.splats, layer_embeddings
                        )
                        
                        if rescued > 0:
                            print(f"      🚀 COMPLETE O(n*k) rescued {rescued} splats strategically")
                    else:
                        # Fallback emergency rescue
                        rescued = self._fallback_emergency_rescue(layer_embeddings)
                        if rescued > 0:
                            print(f"      🚀 Fallback emergency rescued {rescued} splats")
                        
        except Exception as e:
            logger.warning(f"COMPLETE O(n*k) emergency rescue failed for layer {self.layer_idx}: {e}")
    
    def _fallback_emergency_rescue(self, layer_embeddings: torch.Tensor) -> int:
        """Fallback emergency rescue with O(n*k) optimization"""
        rescued_count = 0
        
        try:
            device = layer_embeddings.device
            batch_size, seq_len, embed_dim = layer_embeddings.shape
            
            if seq_len == 0:
                return 0
            
            # O(n*k) emergency sampling
            emergency_sample_size = OnkSamplingStrategy.get_sample_size(seq_len, 0.2, 20, 64)
            sample_indices = OnkSamplingStrategy.smart_sample_indices(
                seq_len, emergency_sample_size, "strategic", device
            )
            sampled_tokens = layer_embeddings[0][sample_indices]
            
            # Only rescue truly problematic splats
            for splat in self.splats:
                try:
                    if hasattr(splat, 'usefulness') and splat.usefulness < 0.08:  # Slightly increased threshold
                        # Find closest token in sample
                        distances = torch.norm(sampled_tokens - splat.position.unsqueeze(0), dim=-1)
                        closest_idx = torch.argmin(distances)
                        closest_token = sampled_tokens[closest_idx]
                        
                        # Move splat toward closest token (conservative movement)
                        direction = closest_token - splat.position
                        move_strength = 0.15  # More conservative
                        
                        with torch.no_grad():
                            splat.position.data += move_strength * direction
                            
                            # Apply improved bounds
                            bounds = 16.0  # Slightly reduced
                            splat.position.data = torch.clamp(splat.position.data, -bounds, bounds)
                            
                            # Reset splat parameters
                            splat.usefulness = 1.2  # Modest reset
                            splat.velocity.zero_()
                        
                        rescued_count += 1
                        
                except Exception as e:
                    logger.warning(f"Emergency repositioning failed for splat {splat.id}: {e}")
                    continue
            
            return rescued_count
            
        except Exception as e:
            logger.warning(f"Fallback emergency rescue failed: {e}")
            return 0
    
    def _init_weights(self):
        """Initialize weights with production-level scaling"""
        std = 0.018 / math.sqrt(self.layer_idx + 1)  # Slightly reduced
        nn.init.normal_(self.token_value_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=std)
    
    def compute_production_attention_matrix(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """COMPLETE production attention matrix computation with optimizations"""
        device = DeviceManager.get_primary_device()
        batch_size, seq_len, model_dim = token_embeddings.shape
        
        token_embeddings = token_embeddings.to(device)
        
        if not self.splats:
            logger.warning(f"No splats available in layer {self.layer_idx}, using uniform attention")
            return torch.ones(batch_size, seq_len, 1, device=device) / 1.0
        
        try:
            # Track computation time
            import time
            start_time = time.time()
            
            centers = []
            scales = []
            amplitudes = []
            
            for splat in self.splats:
                try:
                    centers.append(splat.position.detach().to(device))
                    scales.append(torch.exp(splat.log_scale).detach().clamp(min=0.08, max=15.0).to(device))  # Slightly adjusted
                    amplitudes.append(splat.amplitude.detach().clamp(min=0.08, max=6.0).to(device))  # Slightly adjusted
                except Exception as e:
                    logger.warning(f"Failed to get parameters for splat {splat.id}: {e}")
                    continue
            
            if len(centers) == 0:
                return torch.ones(batch_size, seq_len, 1, device=device) / 1.0
            
            centers = torch.stack(centers, dim=0).to(device)
            scales = torch.stack(scales, dim=0).to(device)
            amplitudes = torch.stack(amplitudes, dim=0).to(device)
            
            # Optimized attention computation
            tokens_expanded = token_embeddings.unsqueeze(2)  # [batch, seq, 1, dim]
            centers_expanded = centers.unsqueeze(0).unsqueeze(0)  # [1, 1, splats, dim]
            
            # Efficient distance computation
            diff = tokens_expanded - centers_expanded
            distances_sq = torch.sum(diff ** 2, dim=-1)  # [batch, seq, splats]
            
            # Stable scale computation
            scales_sq = scales ** 2
            scales_sq = torch.clamp(scales_sq, min=1e-6, max=225.0)  # Slightly adjusted bounds
            normalized_distances = distances_sq / scales_sq.unsqueeze(0).unsqueeze(0)
            normalized_distances = torch.clamp(normalized_distances, max=45.0)  # Slightly reduced for stability
            
            # Gaussian weights with improved stability
            gaussian_weights = torch.exp(-0.5 * normalized_distances)
            attention_weights = gaussian_weights * amplitudes.unsqueeze(0).unsqueeze(0)
            
            # Improved normalization
            attention_sums = attention_weights.sum(dim=-1, keepdim=True)
            attention_sums = torch.clamp(attention_sums, min=1e-10)  # Slightly less aggressive
            attention_weights = attention_weights / attention_sums
            
            # Track performance
            computation_time = time.time() - start_time
            self.attention_computation_times.append(computation_time)
            if len(self.attention_computation_times) > 100:
                self.attention_computation_times.pop(0)
            
            self.total_computations += 1
            
            return attention_weights
            
        except Exception as e:
            logger.error(f"Production attention computation failed for layer {self.layer_idx}: {e}")
            fallback_attention = torch.ones(batch_size, seq_len, max(1, len(self.splats)), device=device)
            fallback_attention = fallback_attention / max(1, len(self.splats))
            return fallback_attention
    
    def forward(self, token_embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                epoch: int = 0) -> torch.Tensor:
        """COMPLETE O(n*k) production-level forward pass with all optimizations"""
        self.forward_count += 1
        device = DeviceManager.get_primary_device()
        batch_size, seq_len, model_dim = token_embeddings.shape
        
        token_embeddings = token_embeddings.to(device)
        
        try:
            # Enhanced trajectory computation
            if self.trajectory_computer is not None:
                trajectories, _ = self.trajectory_computer.compute_enhanced_trajectory_flow(
                    self.layer_idx, token_embeddings
                )
                trajectories = trajectories.to(device)
                
                traj_magnitude = torch.norm(trajectories).item()
                if traj_magnitude < 0.0008:  # Slightly reduced threshold
                    trajectories = trajectories + torch.randn_like(trajectories, device=device) * 0.04  # Slightly reduced noise
            else:
                trajectories = torch.randn_like(token_embeddings, device=device) * 0.04
                trajectories = trajectories.to(device)
                
        except Exception as e:
            logger.error(f"Trajectory computation failed for layer {self.layer_idx}: {e}")
            trajectories = torch.randn_like(token_embeddings, device=device) * 0.04
            trajectories = trajectories.to(device)
        
        # Compute attention weights with optimization tracking
        attention_weights = self.compute_production_attention_matrix(token_embeddings)
        
        if attention_weights.size(-1) == 0:
            logger.warning(f"No active splats in layer {self.layer_idx}")
            return token_embeddings
        
        try:
            # Core attention computation
            token_values = self.token_value_proj(token_embeddings)
            
            # Optimized Einstein summation
            splat_representations = torch.einsum('bsn,bsd->bnd', attention_weights, token_values)
            token_outputs = torch.einsum('bsn,bnd->bsd', attention_weights, splat_representations)
            
            # Apply dropout and output projection
            token_outputs = self.dropout_layer(token_outputs)
            output = self.output_proj(token_outputs)
            
            # Apply attention mask if provided
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                output = output * attention_mask.unsqueeze(-1)
            
            # COMPLETE O(n*k) adaptation during training
            if (self.training and self.adaptation_enabled and 
                self.forward_count % self.adaptation_frequency == 0):
                with torch.no_grad():
                    self.adapt_splats_complete_onk(token_embeddings, trajectories, attention_weights, epoch)
            
            return output
            
        except Exception as e:
            logger.error(f"Forward pass failed for layer {self.layer_idx}: {e}")
            return token_embeddings
    
    def adapt_splats_complete_onk(self, token_embeddings: torch.Tensor, 
                                trajectories: torch.Tensor, 
                                attention_weights: torch.Tensor,
                                epoch: int = 0):
        """COMPLETE O(n*k) production-level splat adaptation with all features"""
        if not self.adaptation_enabled or not self.splats:
            return
        
        device = DeviceManager.get_primary_device()
        
        try:
            # Ensure all tensors are on correct device
            token_embeddings = token_embeddings.to(device)
            trajectories = trajectories.to(device)
            attention_weights = attention_weights.to(device)
            
            # Compute splat activations
            if attention_weights.size(-1) > 0:
                splat_activations = attention_weights.mean(dim=(0, 1))
            else:
                splat_activations = torch.zeros(len(self.splats), device=device)
                
            trajectory_strength_value = torch.sigmoid(self.trajectory_strength) * 4.5  # Slightly reduced
            
            splat_network = {splat.id: splat for splat in self.splats}
            
            # COMPLETE O(n*k) birth processing with all optimizations
            if self.birth_manager:
                try:
                    new_splat_params = self.birth_manager.process_birth_requests(
                        current_splats=self.splats,
                        token_embeddings=token_embeddings,
                        trajectories=trajectories,
                        epoch=epoch
                    )
                    
                    # Create actual splats from parameters with enhanced initialization
                    for splat_params in new_splat_params:
                        try:
                            new_splat = OnkOptimizedTrajectoryFlowSplat(
                                position=splat_params['position'],
                                scale=splat_params['scale'],
                                amplitude=splat_params['amplitude'],
                                splat_id=splat_params['splat_id'],
                                device=splat_params['device'],
                                layer_idx=self.layer_idx
                            )
                            
                            # Enhanced initialization for new splats
                            new_splat.age = 0
                            new_splat.usefulness = 2.2  # Slightly higher initial usefulness
                            new_splat.velocity = torch.zeros_like(splat_params['position'])
                            new_splat.birth_reason = splat_params.get('birth_reason', 'onk_optimization')
                            
                            # Set up birth request callback
                            if self.birth_manager:
                                new_splat.birth_request_callback = self.birth_manager.request_splat_birth
                            
                            self.splats.append(new_splat)
                            splat_network[new_splat.id] = new_splat
                            
                        except Exception as e:
                            logger.warning(f"Failed to create splat from COMPLETE O(n*k) parameters: {e}")
                    
                    if new_splat_params:
                        # Enhanced logging for birth events
                        birth_count = getattr(self.birth_manager, 'total_births', 0)
                        if birth_count % 3 == 1:  # Every 3rd birth
                            logger.info(f"🚀 Layer {self.layer_idx}: COMPLETE O(n*k) birth added {len(new_splat_params)} splats "
                                      f"(total: {len(self.splats)}, births: {birth_count})")
                except Exception as e:
                    logger.warning(f"COMPLETE O(n*k) birth processing failed: {e}")
            
            # Update all splats with COMPLETE O(n*k) optimization
            successful_updates = 0
            for i, splat in enumerate(self.splats):
                if i >= len(splat_activations):
                    continue
                
                try:
                    activation = splat_activations[i].item() if i < len(splat_activations) else 0.0
                    
                    splat.update_with_onk_trajectory_flow(
                        trajectories,
                        token_embeddings,
                        splat_network,
                        epoch
                    )
                    successful_updates += 1
                except Exception as e:
                    logger.warning(f"Failed to update splat {i} in layer {self.layer_idx}: {e}")
            
            # Track adaptation success rate
            if len(self.splats) > 0:
                success_rate = successful_updates / len(self.splats)
                if success_rate < 0.8:  # 80% success threshold
                    logger.warning(f"Layer {self.layer_idx}: Low adaptation success rate: {success_rate:.2f}")
            
        except Exception as e:
            logger.error(f"COMPLETE O(n*k) production adaptation failed for layer {self.layer_idx}: {e}")
    
    def get_production_stats(self, epoch: int = 0) -> Dict:
        """COMPLETE production-level statistics with all O(n*k) metrics"""
        try:
            if not self.splats:
                return {
                    'layer_idx': self.layer_idx,
                    'num_splats': 0,
                    'healthy_splats': 0,
                    'avg_usefulness': 0.0,
                    'avg_trajectory_influence': 0.0,
                    'trajectory_strength': 0.0,
                    'coverage_efficiency': 0.0,
                    'birth_stats': {},
                    'optimization': 'COMPLETE O(n*k) algorithms',
                    'performance_stats': {},
                    'health_status': '🔴 CRITICAL - NO SPLATS'
                }
            
            # Collect comprehensive splat statistics
            splat_stats = []
            total_sample_efficiency = 0.0
            total_computation_savings = 0
            
            for splat in self.splats:
                try:
                    stats = splat.get_production_stats(epoch)
                    splat_stats.append(stats)
                    total_sample_efficiency += stats.get('sample_efficiency', 1.0)
                    total_computation_savings += stats.get('computation_savings', 0)
                except Exception as e:
                    logger.warning(f"Failed to get stats for splat {splat.id}: {e}")
                    splat_stats.append({
                        'layer_idx': self.layer_idx,
                        'is_healthy': False,
                        'sample_efficiency': 1.0,
                        'computation_savings': 0
                    })
            
            healthy_splats = sum(1 for s in splat_stats if s.get('is_healthy', False))
            
            # Aggregate statistics
            if len(splat_stats) > 0:
                avg_usefulness = np.mean([s.get('usefulness', 0) for s in splat_stats])
                avg_trajectory_influence = np.mean([s.get('avg_trajectory_influence', 0) for s in splat_stats])
                avg_sample_efficiency = total_sample_efficiency / len(splat_stats)
                avg_influence_trend = np.mean([s.get('influence_trend_slope', 0) for s in splat_stats])
            else:
                avg_usefulness = 0.0
                avg_trajectory_influence = 0.0
                avg_sample_efficiency = 1.0
                avg_influence_trend = 0.0
            
            # Coverage efficiency calculation
            coverage_efficiency = 0.0
            if self.coverage_positioner and self.splats:
                try:
                    coverage_efficiency = min(1.0, healthy_splats / max(len(self.splats), 1))
                except Exception:
                    coverage_efficiency = 0.0
            
            # COMPLETE O(n*k) birth management statistics
            birth_stats = {}
            optimization_type = "COMPLETE O(n*k) algorithms"
            if self.birth_manager:
                try:
                    birth_stats = self.birth_manager.get_birth_statistics()
                    optimization_type = f"COMPLETE {birth_stats.get('optimization', 'O(n*k) algorithms')}"
                except Exception as e:
                    logger.warning(f"Failed to get birth stats: {e}")
                    birth_stats = {'error': str(e)}
            
            # Performance statistics
            performance_stats = {
                'total_computations': self.total_computations,
                'total_savings': self.total_savings + total_computation_savings,
                'avg_sample_efficiency': avg_sample_efficiency,
                'attention_computation_times': {
                    'count': len(self.attention_computation_times),
                    'avg_time': np.mean(self.attention_computation_times) if self.attention_computation_times else 0.0,
                    'total_time': sum(self.attention_computation_times)
                }
            }
            
            # Enhanced health status with more sophisticated criteria
            if epoch < 3:
                if healthy_splats >= 1:
                    health_status = '🟢 HEALTHY (Early)'
                else:
                    health_status = '🟡 DEVELOPING'
            elif epoch < 8:
                if healthy_splats >= max(1, self.min_splats // 2):
                    health_status = '🟢 HEALTHY'
                elif healthy_splats >= max(1, self.min_splats // 3):
                    health_status = '🟡 WEAK'
                else:
                    health_status = '🔴 CRITICAL'
            else:
                # Consider trend for mature training
                if healthy_splats >= max(2, self.min_splats // 2) and avg_influence_trend >= -1e-8:
                    health_status = '🟢 HEALTHY'
                elif healthy_splats >= max(1, self.min_splats // 3):
                    health_status = '🟡 WEAK'
                else:
                    health_status = '🔴 CRITICAL'
            
            return {
                'layer_idx': self.layer_idx,
                'num_splats': len(self.splats),
                'healthy_splats': healthy_splats,
                'avg_usefulness': avg_usefulness,
                'avg_trajectory_influence': avg_trajectory_influence,
                'avg_influence_trend': avg_influence_trend,
                'trajectory_strength': torch.sigmoid(self.trajectory_strength).item(),
                'coverage_efficiency': coverage_efficiency,
                'birth_stats': birth_stats,
                'optimization': optimization_type,
                'performance_stats': performance_stats,
                'health_status': health_status
            }
        except Exception as e:
            logger.warning(f"Failed to get COMPLETE production stats for layer {self.layer_idx}: {e}")
            return {
                'layer_idx': self.layer_idx,
                'num_splats': len(self.splats) if self.splats else 0,
                'healthy_splats': 0,
                'avg_usefulness': 0.0,
                'avg_trajectory_influence': 0.0,
                'avg_influence_trend': 0.0,
                'trajectory_strength': 0.0,
                'coverage_efficiency': 0.0,
                'birth_stats': {'error': str(e)},
                'optimization': 'COMPLETE O(n*k) algorithms',
                'performance_stats': {'error': str(e)},
                'health_status': '🔴 ERROR'
            }


# Maintain backward compatibility with improved aliases
FixedProductionTrajectoryFlowSplat = OnkOptimizedTrajectoryFlowSplat
FixedProductionSplatFlowAttention = OnkOptimizedSplatFlowAttention


def get_quick_model_stats(model) -> Dict:
    """COMPLETE quick model statistics including all O(n*k) optimization metrics"""
    try:
        # Get trajectory flow statistics
        try:
            flow_stats = model.trajectory_flow.get_comprehensive_statistics()
        except Exception:
            flow_stats = {}
        
        # Get comprehensive splat health from all layers
        total_splats = 0
        healthy_splats = 0
        total_trajectory_influence = 0
        total_coverage_efficiency = 0.0
        total_births = 0
        total_birth_requests = 0
        onk_layers = 0
        total_computation_savings = 0
        total_sample_efficiency = 0.0
        
        for layer in model.layers:
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'get_production_stats'):
                try:
                    stats = layer.attention.get_production_stats()
                    total_splats += stats.get('num_splats', 0)
                    healthy_splats += stats.get('healthy_splats', 0)
                    total_trajectory_influence += stats.get('avg_trajectory_influence', 0)
                    total_coverage_efficiency += stats.get('coverage_efficiency', 0.0)
                    
                    # COMPLETE O(n*k) birth statistics
                    birth_stats = stats.get('birth_stats', {})
                    total_births += birth_stats.get('total_births', 0)
                    total_birth_requests += birth_stats.get('pending_requests', 0)
                    
                    # COMPLETE O(n*k) performance statistics
                    perf_stats = stats.get('performance_stats', {})
                    total_computation_savings += perf_stats.get('total_savings', 0)
                    total_sample_efficiency += perf_stats.get('avg_sample_efficiency', 1.0)
                    
                    # Check if COMPLETE O(n*k) algorithms are active
                    optimization = stats.get('optimization', '')
                    if 'O(n*k)' in optimization:
                        onk_layers += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to get stats from layer: {e}")
                    continue
        
        layer_count = len(model.layers) if model.layers else 1
        avg_trajectory_influence = total_trajectory_influence / layer_count
        avg_coverage_efficiency = total_coverage_efficiency / layer_count
        avg_sample_efficiency = total_sample_efficiency / max(onk_layers, 1)
        health_percentage = (healthy_splats / max(total_splats, 1)) * 100
        
        # Enhanced optimization status
        optimization_status = f"{onk_layers}/{layer_count} layers using COMPLETE O(n*k)"
        if onk_layers == layer_count and layer_count > 0:
            optimization_status += " ✅"
        elif onk_layers > 0:
            optimization_status += " 🔄"
        else:
            optimization_status += " ❌"
        
        return {
            'total_splats': total_splats,
            'healthy_splats': healthy_splats,
            'health_pct': health_percentage,
            'avg_traj_influence': avg_trajectory_influence,
            'avg_coverage_efficiency': avg_coverage_efficiency,
            'total_births': total_births,
            'pending_birth_requests': total_birth_requests,
            'flow_magnitude': flow_stats.get('max_flow_magnitude', 0),
            'cache_hit_rate': flow_stats.get('cache', {}).get('hit_rate', 0) * 100,
            'active_layers': flow_stats.get('total_layers_with_flow', 0),
            'onk_optimization': optimization_status,
            'computation_savings': total_computation_savings,
            'avg_sample_efficiency': avg_sample_efficiency,
            'optimization_complete': onk_layers == layer_count and layer_count > 0
        }
    except Exception as e:
        logger.warning(f"Failed to get COMPLETE model stats: {e}")
        return {
            'total_splats': 0,
            'healthy_splats': 0, 
            'health_pct': 0,
            'avg_traj_influence': 0,
            'avg_coverage_efficiency': 0.0,
            'total_births': 0,
            'pending_birth_requests': 0,
            'flow_magnitude': 0,
            'cache_hit_rate': 0,
            'active_layers': 0,
            'onk_optimization': '0/0 layers using COMPLETE O(n*k) ❌',
            'computation_savings': 0,
            'avg_sample_efficiency': 0.0,
            'optimization_complete': False
        }
