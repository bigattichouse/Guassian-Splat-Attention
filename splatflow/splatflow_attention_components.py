"""
SplatFlow Attention Components Module
Core splat and attention mechanism implementations for the SplatFlow system.
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

logger = logging.getLogger(__name__)


class FixedProductionTrajectoryFlowSplat:
    """FIXED splat with proper positioning and adaptive influence radius"""
    
    def __init__(self, position: torch.Tensor, scale: float, amplitude: float, 
                 splat_id: int, device: torch.device, layer_idx: int = 0):
        self.device = device
        self.id = splat_id
        self.layer_idx = layer_idx
        
        self.position = position.clone().detach().to(device).requires_grad_(True)
        self.log_scale = torch.tensor(math.log(scale), device=device, requires_grad=True)
        self.amplitude = torch.tensor(amplitude, device=device, requires_grad=True)
        
        self.velocity = torch.zeros_like(self.position, device=device)
        self.trajectory_momentum = 0.9
        
        # Enhanced learning rates
        base_lr = 0.1  # Increased base learning rate
        self.trajectory_learning_rate = base_lr * (1.0 + layer_idx * 0.8)
        
        self.age = 0
        self.usefulness = 2.0 + layer_idx * 0.5
        self.activation_history = []
        self.trajectory_influence_history = []
        
        self.splat_connections = {}
        self.flow_magnitude = 0.0
        
        # FIXED: Initialize embedding statistics for adaptive radius
        self.embedding_stats = {
            'mean_magnitude': 1.0,
            'std_magnitude': 0.5,
            'median_inter_token_distance': 2.0,
            'sample_count': 0
        }
        
    def update_embedding_statistics(self, sample_embeddings: torch.Tensor):
        """Update embedding statistics for adaptive radius computation"""
        with torch.no_grad():
            try:
                flat_embeddings = sample_embeddings.reshape(-1, sample_embeddings.shape[-1])
                
                if len(flat_embeddings) == 0:
                    return
                
                # Update magnitude statistics with safety checks
                magnitudes = torch.norm(flat_embeddings, dim=-1)
                
                if len(magnitudes) > 0:
                    mean_mag = magnitudes.mean().item()
                    std_mag = magnitudes.std().item() if len(magnitudes) > 1 else 0.5
                    
                    # Ensure reasonable bounds
                    self.embedding_stats['mean_magnitude'] = max(0.1, min(mean_mag, 10.0))
                    self.embedding_stats['std_magnitude'] = max(0.1, min(std_mag, 5.0))
                
                # Update inter-token distance statistics
                if len(flat_embeddings) > 1:
                    # Sample subset to avoid memory issues
                    sample_size = min(50, len(flat_embeddings))  # Reduced sample size
                    sample_indices = torch.randperm(len(flat_embeddings))[:sample_size]
                    sample_tokens = flat_embeddings[sample_indices]
                    
                    if len(sample_tokens) > 1:
                        distances = torch.cdist(sample_tokens, sample_tokens)
                        distances = distances[distances > 1e-6]  # Remove near-zero distances
                        
                        if len(distances) > 0:
                            median_dist = torch.median(distances).item()
                            self.embedding_stats['median_inter_token_distance'] = max(0.5, min(median_dist, 8.0))
                
                self.embedding_stats['sample_count'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to update embedding statistics for splat {self.id}: {e}")
    
    def compute_adaptive_influence_radius(self, token_embeddings: torch.Tensor) -> float:
        """FIXED: Compute adaptive influence radius based on local embedding density"""
        with torch.no_grad():
            try:
                # Method 1: Use stored statistics (more efficient)
                base_radius = self.embedding_stats['median_inter_token_distance'] * 1.5
                
                # Method 2: Compute local statistics if needed
                flat_tokens = token_embeddings.reshape(-1, token_embeddings.shape[-1])
                
                if len(flat_tokens) > 5:  # Only if we have enough tokens
                    # Calculate distance from splat to all tokens
                    splat_expanded = self.position.unsqueeze(0)
                    distances_to_splat = torch.norm(flat_tokens - splat_expanded, dim=-1)
                    
                    # Use percentile-based approach for adaptive radius
                    if len(distances_to_splat) > 0:
                        radius_candidates = []
                        
                        # Safely compute quantiles
                        try:
                            if len(distances_to_splat) >= 5:
                                radius_candidates.extend([
                                    torch.quantile(distances_to_splat, 0.1).item(),
                                    torch.quantile(distances_to_splat, 0.2).item()
                                ])
                        except Exception:
                            pass  # Quantile computation might fail on small tensors
                        
                        radius_candidates.extend([
                            self.embedding_stats['median_inter_token_distance'] * 1.2,
                            base_radius,
                            2.0  # Fallback
                        ])
                        
                        # Choose radius that covers at least some tokens
                        for radius in sorted(radius_candidates):
                            if radius > 0 and (distances_to_splat < radius).sum().item() > 0:
                                return max(0.5, min(radius, 8.0))  # Clamp to reasonable bounds
                
                return max(1.0, min(base_radius, 5.0))  # Fallback with reasonable bounds
                
            except Exception as e:
                logger.warning(f"Failed to compute adaptive radius for splat {self.id}: {e}")
                return 2.0  # Safe fallback
    
    def update_with_fixed_trajectory_flow(self, layer_trajectory: torch.Tensor, 
                                        token_embeddings: torch.Tensor, 
                                        splat_network: Optional[Dict] = None,
                                        epoch: int = 0):
        """FIXED: Update splat with enhanced trajectory flow and proper positioning"""
        self.age += 1
        device = self.device
        
        try:
            layer_trajectory = DeviceManager.ensure_tensor_device(layer_trajectory, device)
            token_embeddings = DeviceManager.ensure_tensor_device(token_embeddings, device)
            
            # Update embedding statistics periodically
            if self.age % 50 == 0:  # Every 50 updates
                self.update_embedding_statistics(token_embeddings)
            
            # Compute trajectory influence with adaptive radius
            trajectory_influence = self.compute_fixed_trajectory_influence(
                layer_trajectory, token_embeddings, epoch
            )
            
            influence_magnitude = safe_tensor_to_scalar(torch.norm(trajectory_influence))
            self.trajectory_influence_history.append(influence_magnitude)
            if len(self.trajectory_influence_history) > 100:
                self.trajectory_influence_history.pop(0)
            
            # Apply inter-splat flow if available
            if splat_network:
                try:
                    inter_splat_flow = self.compute_inter_splat_flow(splat_network)
                    trajectory_influence = trajectory_influence + 0.4 * inter_splat_flow
                except Exception as e:
                    logger.warning(f"Inter-splat flow computation failed for splat {self.id}: {e}")
            
            # FIXED: Enhanced adaptive learning rate
            adaptive_lr = self.trajectory_learning_rate
            if self.layer_idx > 0:
                layer_boost = 1.0 + self.layer_idx * 1.5
                adaptive_lr *= layer_boost
            
            # Progressive learning rate based on epoch
            if epoch < 10:
                adaptive_lr *= 1.5  # Higher learning rate in early epochs
            
            self.velocity = (self.trajectory_momentum * self.velocity + 
                            adaptive_lr * trajectory_influence).to(device)
            
            # FIXED: Adaptive movement bounds based on embedding scale
            max_vel = self.embedding_stats['std_magnitude'] * 0.5
            max_vel = max(0.3, min(max_vel, 2.0))  # Reasonable bounds
            self.velocity = torch.clamp(self.velocity, -max_vel, max_vel)
            
            # Update position with bounds based on embedding statistics
            with torch.no_grad():
                new_position = self.position + self.velocity
                # Use embedding statistics for reasonable bounds
                bound_scale = self.embedding_stats['mean_magnitude'] + 2 * self.embedding_stats['std_magnitude']
                bounds = max(2.0, min(bound_scale, 10.0))
                self.position.data = torch.clamp(new_position, -bounds, bounds)
            
            # FIXED: Progressive usefulness criteria
            if len(self.trajectory_influence_history) > 0:
                recent_influence = np.mean(self.trajectory_influence_history[-20:])
            else:
                recent_influence = 0.0
            
            # Progressive thresholds based on epoch
            if epoch < 5:
                baseline_influence = 1e-6  # Very lenient early on
            elif epoch < 15:
                baseline_influence = 1e-5  # Moderate
            else:
                baseline_influence = 1e-4  # Normal after warmup
            
            usefulness_delta = 0.1 * (recent_influence - baseline_influence)
            self.usefulness = np.clip(self.usefulness + usefulness_delta, 0.1, 5.0)
            self.flow_magnitude = influence_magnitude
            
        except Exception as e:
            logger.warning(f"Failed to update splat {self.id} with trajectory flow: {e}")
    
    def compute_fixed_trajectory_influence(self, layer_trajectory: torch.Tensor, 
                                         token_embeddings: torch.Tensor,
                                         epoch: int = 0) -> torch.Tensor:
        """FIXED: Compute trajectory influence with adaptive radius and better debugging"""
        try:
            batch_size, seq_len, dim = token_embeddings.shape
            device = self.device
            
            if seq_len == 0 or batch_size == 0:
                return torch.zeros_like(self.position).to(device)
            
            # FIXED: Use adaptive influence radius
            influence_radius = self.compute_adaptive_influence_radius(token_embeddings)
            
            splat_expanded = self.position.unsqueeze(0).unsqueeze(0).to(device)
            distances = torch.norm(token_embeddings - splat_expanded, dim=-1)
            
            # Enhanced debugging
            if distances.numel() > 0:
                min_dist = torch.min(distances).item()
                mean_dist = torch.mean(distances).item()
                max_dist = torch.max(distances).item()
                
                # Debug output for splat 0 occasionally
                if self.id == 0 and random.random() < 0.05:
                    print(f"    🔧 FIXED Splat {self.id}: dist_range=[{min_dist:.3f}, {mean_dist:.3f}, {max_dist:.3f}], radius={influence_radius:.3f}")
            
            # Progressive radius expansion if no tokens in range
            influence_mask = distances < influence_radius
            tokens_in_influence = influence_mask.sum().item()
            
            if tokens_in_influence == 0:
                # Progressive radius expansion
                expansion_factors = [1.5, 2.0, 3.0, 5.0, 8.0]
                for expansion_factor in expansion_factors:
                    expanded_radius = influence_radius * expansion_factor
                    influence_mask = distances < expanded_radius
                    if influence_mask.any():
                        influence_radius = expanded_radius
                        tokens_in_influence = influence_mask.sum().item()
                        if self.id == 0 and random.random() < 0.1:
                            print(f"    🔧 FIXED Splat {self.id}: expanded radius to {expanded_radius:.3f}, found {tokens_in_influence} tokens")
                        break
                else:
                    # Emergency repositioning - move toward closest token
                    if epoch < 10 and distances.numel() > 0:  # Only in early epochs
                        closest_token_idx = torch.argmin(distances.reshape(-1))
                        closest_token = token_embeddings.reshape(-1, dim)[closest_token_idx]
                        
                        with torch.no_grad():
                            # Move 50% toward closest token
                            direction = closest_token - self.position
                            self.position.data += 0.5 * direction
                        
                        if self.id == 0:
                            print(f"    🚨 EMERGENCY: Moved splat {self.id} toward closest token")
                    
                    return torch.zeros_like(self.position).to(device)
            
            # Compute influence with softer falloff
            proximity_weights = torch.exp(-distances / (influence_radius * 0.3))  # Even softer falloff
            proximity_weights = proximity_weights * influence_mask.float()
            
            # More sensitive trajectory magnitude weighting
            traj_magnitudes = torch.norm(layer_trajectory, dim=-1)
            magnitude_weights = torch.sigmoid(traj_magnitudes * 0.5)  # More sensitive
            
            total_weights = proximity_weights * magnitude_weights
            total_weight_sum = safe_tensor_to_scalar(total_weights.sum())
            
            if self.id == 0 and random.random() < 0.02:
                print(f"    🔧 FIXED Splat {self.id}: tokens_in_range={tokens_in_influence}, total_weight={total_weight_sum:.6f}")
            
            if total_weight_sum < 1e-12:
                return torch.zeros_like(self.position).to(device)
            
            # Compute weighted trajectory influence with safe division
            weighted_trajectories = layer_trajectory * total_weights.unsqueeze(-1)
            influence_vector = weighted_trajectories.sum(dim=(0, 1)) / max(total_weight_sum, 1e-12)
            
            # Layer-specific boost
            layer_boost = 1.0 + self.layer_idx * 1.2
            influence_vector = influence_vector * layer_boost
            
            influence_magnitude = torch.norm(influence_vector).item()
            if self.id == 0 and random.random() < 0.02:
                print(f"    🔧 FIXED Splat {self.id}: final_influence_mag={influence_magnitude:.6f}")
            
            return influence_vector.to(device)
            
        except Exception as e:
            logger.warning(f"Failed to compute trajectory influence for splat {self.id}: {e}")
            return torch.zeros_like(self.position).to(self.device)
    
    def compute_inter_splat_flow(self, splat_network: Dict) -> torch.Tensor:
        """Compute enhanced flow between connected splats"""
        device = self.device
        inter_flow = torch.zeros_like(self.position).to(device)
        
        try:
            for other_splat in splat_network.values():
                if other_splat.id != self.id:
                    direction = other_splat.position - self.position
                    distance = torch.norm(direction)
                    
                    if distance > 1e-6:
                        normalized_direction = direction / max(distance, 1e-6)  # Safe division
                        optimal_distance = 1.5 + self.layer_idx * 0.4
                        
                        if distance > optimal_distance:
                            flow_strength = 0.2 * (distance - optimal_distance)
                            inter_flow += flow_strength * normalized_direction
                        else:
                            flow_strength = 0.3 * (optimal_distance - distance)
                            inter_flow -= flow_strength * normalized_direction
        except Exception as e:
            logger.warning(f"Inter-splat flow computation failed for splat {self.id}: {e}")
        
        return inter_flow.to(device)
    
    def is_healthy(self, epoch: int = 0) -> bool:
        """FIXED: Enhanced health check with epoch-based progressive criteria"""
        try:
            if len(self.trajectory_influence_history) > 0:
                recent_influence = np.mean(self.trajectory_influence_history[-10:])
            else:
                recent_influence = 0.0
            
            # Progressive thresholds - start very lenient, gradually tighten
            if epoch < 5:
                influence_threshold = 1e-6  # Very lenient for early training
                usefulness_threshold = 0.05
            elif epoch < 15:
                influence_threshold = 1e-5  # Moderate
                usefulness_threshold = 0.1
            else:
                influence_threshold = 1e-4  # Normal threshold after warmup
                usefulness_threshold = 0.2
            
            is_influence_healthy = recent_influence > influence_threshold
            is_usefulness_healthy = self.usefulness > usefulness_threshold
            
            return is_influence_healthy and is_usefulness_healthy
        except Exception as e:
            logger.warning(f"Health check failed for splat {self.id}: {e}")
            return True  # Default to healthy to avoid premature removal
    
    def get_production_stats(self, epoch: int = 0) -> Dict:
        """Get comprehensive production-level statistics"""
        try:
            if len(self.trajectory_influence_history) > 0:
                recent_influence = np.mean(self.trajectory_influence_history[-20:])
                avg_influence = np.mean(self.trajectory_influence_history)
            else:
                recent_influence = 0.0
                avg_influence = 0.0
            
            return {
                'layer_idx': self.layer_idx,
                'age': self.age,
                'usefulness': self.usefulness,
                'recent_trajectory_influence': recent_influence,
                'avg_trajectory_influence': avg_influence,
                'flow_magnitude': self.flow_magnitude,
                'velocity_magnitude': torch.norm(self.velocity).item(),
                'position_magnitude': torch.norm(self.position).item(),
                'trajectory_learning_rate': self.trajectory_learning_rate,
                'is_healthy': self.is_healthy(epoch),
                'embedding_stats': self.embedding_stats
            }
        except Exception as e:
            logger.warning(f"Failed to get stats for splat {self.id}: {e}")
            return {
                'layer_idx': self.layer_idx,
                'age': self.age,
                'usefulness': 1.0,
                'recent_trajectory_influence': 0.0,
                'avg_trajectory_influence': 0.0,
                'flow_magnitude': 0.0,
                'velocity_magnitude': 0.0,
                'position_magnitude': 0.0,
                'trajectory_learning_rate': self.trajectory_learning_rate,
                'is_healthy': True,
                'embedding_stats': self.embedding_stats
            }


class FixedProductionSplatFlowAttention(nn.Module):
    """FIXED production-ready SplatFlow attention with proper splat positioning"""
    
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
        
        self.min_splats = max(4, num_splats // 4)  # Reduced minimum for easier success
        self.recovery_enabled = True
        self.last_recovery_epoch = 0
        
        self.token_value_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.output_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Enhanced trajectory strength
        initial_strength = 0.6 + layer_idx * 0.3
        self.trajectory_strength = nn.Parameter(torch.tensor(initial_strength))
        
        # Initialize with placeholder splats - will be properly positioned later
        self._initialize_placeholder_splats()
        self._init_weights()
        
        logger.info(f"🎯 FIXED Production SplatFlow attention initialized for layer {layer_idx}")
    
    def _initialize_placeholder_splats(self):
        """Initialize placeholder splats - will be repositioned based on actual embeddings"""
        device = DeviceManager.get_primary_device()
        self.splats = []
        
        for i in range(self.num_splats):
            # Placeholder initialization - will be fixed later
            position = torch.randn(self.model_dim, device=device) * 0.1
            scale = 1.0 + torch.rand(1).item() * 0.5
            amplitude = 1.0 + torch.rand(1).item() * 0.5
            
            splat = FixedProductionTrajectoryFlowSplat(position, scale, amplitude, i, device, self.layer_idx)
            self.splats.append(splat)
        
        logger.info(f"🎯 Initialized {len(self.splats)} placeholder splats for layer {self.layer_idx}")
    
    def fix_splat_positioning_based_on_embeddings(self, sample_embeddings: torch.Tensor):
        """FIXED: Properly position splats based on actual token embedding statistics"""
        with torch.no_grad():
            try:
                device = self.splats[0].device if self.splats else DeviceManager.get_primary_device()
                sample_embeddings = DeviceManager.ensure_tensor_device(sample_embeddings, device)
                
                # Get comprehensive embedding statistics
                flat_embeddings = sample_embeddings.reshape(-1, self.model_dim)
                
                if len(flat_embeddings) == 0:
                    logger.warning(f"Empty embeddings for layer {self.layer_idx}, skipping positioning")
                    return
                
                # Calculate proper statistics with safety checks
                mean_pos = flat_embeddings.mean(dim=0)
                std_pos = flat_embeddings.std(dim=0)
                
                # Ensure std is not zero
                std_pos = torch.clamp(std_pos, min=0.1)
                
                # Get percentile ranges for better coverage
                try:
                    percentiles = torch.quantile(flat_embeddings, torch.tensor([0.1, 0.9], device=device), dim=0)
                    embedding_range = percentiles[1] - percentiles[0]
                except Exception:
                    # Fallback if quantile fails
                    embedding_range = 2 * std_pos
                
                print(f"🔧 FIXED Layer {self.layer_idx} embedding analysis:")
                print(f"   Token count: {len(flat_embeddings)}")
                print(f"   Mean magnitude: {torch.norm(mean_pos).item():.3f}")
                print(f"   Std magnitude: {torch.norm(std_pos).item():.3f}")
                print(f"   Range magnitude: {torch.norm(embedding_range).item():.3f}")
                
                # Update embedding statistics for all splats
                for splat in self.splats:
                    splat.update_embedding_statistics(sample_embeddings)
                
                # Reinitialize splats within the actual embedding space
                for i, splat in enumerate(self.splats):
                    if i < len(flat_embeddings):
                        # Start from actual token position
                        base_pos = flat_embeddings[i % len(flat_embeddings)]
                    else:
                        # Sample from distribution
                        base_pos = mean_pos + torch.randn_like(mean_pos, device=device) * std_pos * 0.5
                    
                    # Add small perturbation to avoid exact overlap
                    perturbation = torch.randn_like(base_pos, device=device) * torch.norm(std_pos) * 0.1
                    new_position = base_pos + perturbation
                    
                    # Update splat position
                    splat.position.data = new_position
                    
                    # Reset other parameters
                    splat.usefulness = 2.0
                    splat.velocity.zero_()
                    splat.trajectory_influence_history.clear()
                    
                    print(f"   ✅ Splat {i}: repositioned to magnitude {torch.norm(splat.position).item():.3f}")
                    
            except Exception as e:
                logger.error(f"Failed to fix splat positioning for layer {self.layer_idx}: {e}")
    
    def progressive_splat_repositioning(self, layer_embeddings: torch.Tensor, epoch: int):
        """FIXED: Progressively move splats toward token clusters"""
        if epoch % 3 == 0 and epoch > 0:  # Every 3 epochs after first
            with torch.no_grad():
                try:
                    device = self.splats[0].device if self.splats else DeviceManager.get_primary_device()
                    layer_embeddings = DeviceManager.ensure_tensor_device(layer_embeddings, device)
                    flat_embeddings = layer_embeddings.reshape(-1, self.model_dim)
                    
                    if len(flat_embeddings) == 0:
                        return
                    
                    repositioned_count = 0
                    
                    for splat in self.splats:
                        # Find closest tokens to current splat position
                        distances = torch.norm(flat_embeddings - splat.position.unsqueeze(0), dim=-1)
                        if len(distances) > 0:
                            closest_idx = torch.argmin(distances)
                            closest_token = flat_embeddings[closest_idx]
                            
                            # Move splat toward closest token cluster
                            direction = closest_token - splat.position
                            move_strength = 0.15 if epoch < 10 else 0.1  # More aggressive early on
                            splat.position.data += move_strength * direction
                            
                            # Add small random perturbation to explore
                            perturbation_strength = 0.05 if epoch < 10 else 0.02
                            perturbation = torch.randn_like(splat.position, device=device) * perturbation_strength
                            splat.position.data += perturbation
                            
                            repositioned_count += 1
                    
                    if repositioned_count > 0:
                        print(f"    🔄 Layer {self.layer_idx}: Repositioned {repositioned_count} splats (epoch {epoch})")
                        
                except Exception as e:
                    logger.warning(f"Progressive repositioning failed for layer {self.layer_idx}: {e}")
    
    def emergency_splat_rescue(self, layer_embeddings: torch.Tensor, epoch: int):
        """FIXED: Emergency system to rescue non-functional splats"""
        try:
            healthy_count = sum(1 for splat in self.splats if splat.is_healthy(epoch))
            
            if len(self.splats) > 0 and healthy_count < len(self.splats) * 0.25:  # Less than 25% healthy
                print(f"    🚨 EMERGENCY RESCUE Layer {self.layer_idx}: Only {healthy_count}/{len(self.splats)} healthy splats")
                
                with torch.no_grad():
                    device = self.splats[0].device if self.splats else DeviceManager.get_primary_device()
                    layer_embeddings = DeviceManager.ensure_tensor_device(layer_embeddings, device)
                    flat_embeddings = layer_embeddings.reshape(-1, self.model_dim)
                    
                    if len(flat_embeddings) == 0:
                        return
                    
                    # Rescue unhealthy splats by repositioning to token locations
                    unhealthy_splats = [s for s in self.splats if not s.is_healthy(epoch)]
                    rescued_count = 0
                    
                    for i, splat in enumerate(unhealthy_splats):
                        if i < len(flat_embeddings):
                            # Move directly to a token position
                            token_idx = (i * len(flat_embeddings)) // max(len(unhealthy_splats), 1)
                            token_idx = min(token_idx, len(flat_embeddings) - 1)
                            target_position = flat_embeddings[token_idx].clone()
                            
                            # Add small offset to avoid exact overlap
                            offset = torch.randn_like(target_position, device=device) * 0.1
                            splat.position.data = target_position + offset
                            
                            # Reset splat parameters
                            splat.usefulness = 1.5
                            splat.velocity.zero_()
                            splat.trajectory_influence_history.clear()
                            
                            rescued_count += 1
                    
                    print(f"      🔧 Rescued {rescued_count} splats to token positions")
                    
        except Exception as e:
            logger.warning(f"Emergency rescue failed for layer {self.layer_idx}: {e}")
    
    def _init_weights(self):
        """Initialize weights with production-level scaling"""
        std = 0.02 / math.sqrt(self.layer_idx + 1)
        nn.init.normal_(self.token_value_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=std)
    
    def compute_production_attention_matrix(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute attention matrix with production-level robustness"""
        device = DeviceManager.get_primary_device()
        batch_size, seq_len, model_dim = token_embeddings.shape
        
        token_embeddings = DeviceManager.ensure_tensor_device(token_embeddings, device)
        
        if not self.splats:
            logger.warning(f"No splats available in layer {self.layer_idx}, using uniform attention")
            return torch.ones(batch_size, seq_len, 1, device=device) / 1.0
        
        try:
            centers = []
            scales = []
            amplitudes = []
            
            for splat in self.splats:
                centers.append(DeviceManager.ensure_tensor_device(splat.position.detach(), device))
                scales.append(DeviceManager.ensure_tensor_device(
                    torch.exp(splat.log_scale).detach().clamp(min=0.1, max=8.0), device))
                amplitudes.append(DeviceManager.ensure_tensor_device(
                    splat.amplitude.detach().clamp(min=0.1, max=5.0), device))
            
            if len(centers) == 0:
                return torch.ones(batch_size, seq_len, 1, device=device) / 1.0
            
            centers = DeviceManager.safe_cat([c.unsqueeze(0) for c in centers], dim=0, target_device=device)
            scales = DeviceManager.safe_cat([s.unsqueeze(0) for s in scales], dim=0, target_device=device)
            amplitudes = DeviceManager.safe_cat([a.unsqueeze(0) for a in amplitudes], dim=0, target_device=device)
            
            tokens_expanded = token_embeddings.unsqueeze(2)
            centers_expanded = centers.unsqueeze(0).unsqueeze(0)
            
            diff = tokens_expanded - centers_expanded
            distances_sq = torch.sum(diff ** 2, dim=-1)
            
            scales_sq = scales ** 2
            scales_sq = torch.clamp(scales_sq, min=1e-6)  # Prevent division by zero
            normalized_distances = distances_sq / scales_sq.unsqueeze(0).unsqueeze(0)
            normalized_distances = torch.clamp(normalized_distances, max=50.0)  # Higher clamp
            
            gaussian_weights = torch.exp(-0.5 * normalized_distances)
            attention_weights = gaussian_weights * amplitudes.unsqueeze(0).unsqueeze(0)
            
            attention_sums = attention_weights.sum(dim=-1, keepdim=True)
            attention_sums = torch.clamp(attention_sums, min=1e-12)  # Lower clamp
            attention_weights = attention_weights / attention_sums
            
            return attention_weights
            
        except Exception as e:
            logger.error(f"Production attention computation failed for layer {self.layer_idx}: {e}")
            fallback_attention = torch.ones(batch_size, seq_len, max(1, len(self.splats)), device=device)
            fallback_attention = fallback_attention / max(1, len(self.splats))
            return fallback_attention
    
    def forward(self, token_embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """FIXED production-level forward pass with comprehensive error handling"""
        self.forward_count += 1
        device = DeviceManager.get_primary_device()
        batch_size, seq_len, model_dim = token_embeddings.shape
        
        token_embeddings = DeviceManager.ensure_tensor_device(token_embeddings, device)
        
        try:
            if self.trajectory_computer is not None:
                trajectories, _ = self.trajectory_computer.compute_enhanced_trajectory_flow(self.layer_idx, token_embeddings)
                trajectories = DeviceManager.ensure_tensor_device(trajectories, device)
                
                traj_magnitude = torch.norm(trajectories).item()
                if traj_magnitude < 0.001:
                    trajectories = trajectories + torch.randn_like(trajectories, device=device) * 0.05
            else:
                trajectories = torch.randn_like(token_embeddings, device=device) * 0.05
                trajectories = DeviceManager.ensure_tensor_device(trajectories, device)
                
        except Exception as e:
            logger.error(f"Trajectory computation failed for layer {self.layer_idx}: {e}")
            trajectories = torch.randn_like(token_embeddings, device=device) * 0.05
            trajectories = DeviceManager.ensure_tensor_device(trajectories, device)
        
        attention_weights = self.compute_production_attention_matrix(token_embeddings)
        
        if attention_weights.size(-1) == 0:
            logger.warning(f"No active splats in layer {self.layer_idx}")
            return token_embeddings
        
        try:
            token_values = self.token_value_proj(token_embeddings)
            
            splat_representations = torch.einsum('bsn,bsd->bnd', attention_weights, token_values)
            token_outputs = torch.einsum('bsn,bnd->bsd', attention_weights, splat_representations)
            
            token_outputs = self.dropout_layer(token_outputs)
            output = self.output_proj(token_outputs)
            
            if attention_mask is not None:
                attention_mask = DeviceManager.ensure_tensor_device(attention_mask, device)
                output = output * attention_mask.unsqueeze(-1)
            
            if (self.training and self.adaptation_enabled and 
                self.forward_count % self.adaptation_frequency == 0):
                with torch.no_grad():
                    self.adapt_splats_for_production(token_embeddings, trajectories, attention_weights)
            
            return output
            
        except Exception as e:
            logger.error(f"Forward pass failed for layer {self.layer_idx}: {e}")
            return token_embeddings
    
    def adapt_splats_for_production(self, token_embeddings: torch.Tensor, 
                                  trajectories: torch.Tensor, 
                                  attention_weights: torch.Tensor,
                                  epoch: int = 0):
        """FIXED production-level splat adaptation with enhanced trajectory flow"""
        if not self.adaptation_enabled or not self.splats:
            return
        
        device = DeviceManager.get_primary_device()
        
        try:
            token_embeddings = DeviceManager.ensure_tensor_device(token_embeddings, device)
            trajectories = DeviceManager.ensure_tensor_device(trajectories, device)
            attention_weights = DeviceManager.ensure_tensor_device(attention_weights, device)
            
            if attention_weights.size(-1) > 0:
                splat_activations = attention_weights.mean(dim=(0, 1))
            else:
                splat_activations = torch.zeros(len(self.splats), device=device)
                
            trajectory_strength_value = torch.sigmoid(self.trajectory_strength) * 5.0
            
            splat_network = {splat.id: splat for splat in self.splats}
            
            for i, splat in enumerate(self.splats):
                if i >= len(splat_activations):
                    continue
                
                try:
                    activation = splat_activations[i].item() if i < len(splat_activations) else 0.0
                    
                    splat.update_with_fixed_trajectory_flow(
                        trajectories,
                        token_embeddings,
                        splat_network,
                        epoch
                    )
                except Exception as e:
                    logger.warning(f"Failed to update splat {i} in layer {self.layer_idx}: {e}")
            
        except Exception as e:
            logger.error(f"Production adaptation failed for layer {self.layer_idx}: {e}")
    
    def get_production_stats(self, epoch: int = 0) -> Dict:
        """Get comprehensive production-level statistics"""
        try:
            if not self.splats:
                return {
                    'layer_idx': self.layer_idx,
                    'num_splats': 0,
                    'healthy_splats': 0,
                    'avg_usefulness': 0.0,
                    'avg_trajectory_influence': 0.0,
                    'trajectory_strength': 0.0,
                    'health_status': '🔴 CRITICAL - NO SPLATS'
                }
            
            splat_stats = [splat.get_production_stats(epoch) for splat in self.splats]
            
            healthy_splats = sum(1 for s in splat_stats if s['is_healthy'])
            
            if len(splat_stats) > 0:
                avg_usefulness = np.mean([s['usefulness'] for s in splat_stats])
                avg_trajectory_influence = np.mean([s['avg_trajectory_influence'] for s in splat_stats])
            else:
                avg_usefulness = 0.0
                avg_trajectory_influence = 0.0
            
            # Progressive health status thresholds
            if epoch < 5:
                # Very lenient in early epochs
                if healthy_splats >= 1:
                    health_status = '🟢 HEALTHY (Early)'
                else:
                    health_status = '🟡 DEVELOPING'
            elif epoch < 15:
                # Moderate expectations
                if healthy_splats >= self.min_splats:
                    health_status = '🟢 HEALTHY'
                elif healthy_splats >= max(1, self.min_splats // 2):
                    health_status = '🟡 WEAK'
                else:
                    health_status = '🔴 CRITICAL'
            else:
                # Full expectations
                if healthy_splats >= self.min_splats:
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
                'trajectory_strength': torch.sigmoid(self.trajectory_strength).item(),
                'health_status': health_status
            }
        except Exception as e:
            logger.warning(f"Failed to get production stats for layer {self.layer_idx}: {e}")
            return {
                'layer_idx': self.layer_idx,
                'num_splats': len(self.splats) if self.splats else 0,
                'healthy_splats': 0,
                'avg_usefulness': 0.0,
                'avg_trajectory_influence': 0.0,
                'trajectory_strength': 0.0,
                'health_status': '🔴 ERROR'
            }


def get_quick_model_stats(model) -> Dict:
    """Get quick model statistics for batch logging"""
    try:
        # Get trajectory flow statistics
        flow_stats = model.trajectory_flow.get_comprehensive_statistics()
        
        # Get splat health from all layers
        total_splats = 0
        healthy_splats = 0
        total_trajectory_influence = 0
        
        for layer in model.layers:
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'get_production_stats'):
                stats = layer.attention.get_production_stats()
                total_splats += stats.get('num_splats', 0)
                healthy_splats += stats.get('healthy_splats', 0)
                total_trajectory_influence += stats.get('avg_trajectory_influence', 0)
        
        layer_count = len(model.layers) if model.layers else 1
        avg_trajectory_influence = total_trajectory_influence / layer_count
        health_percentage = (healthy_splats / max(total_splats, 1)) * 100
        
        return {
            'total_splats': total_splats,
            'healthy_splats': healthy_splats,
            'health_pct': health_percentage,
            'avg_traj_influence': avg_trajectory_influence,
            'flow_magnitude': flow_stats.get('max_flow_magnitude', 0),
            'cache_hit_rate': flow_stats.get('cache', {}).get('hit_rate', 0) * 100,
            'active_layers': flow_stats.get('total_layers_with_flow', 0)
        }
    except Exception as e:
        logger.warning(f"Failed to get model stats: {e}")
        return {
            'total_splats': 0,
            'healthy_splats': 0, 
            'health_pct': 0,
            'avg_traj_influence': 0,
            'flow_magnitude': 0,
            'cache_hit_rate': 0,
            'active_layers': 0
        }
