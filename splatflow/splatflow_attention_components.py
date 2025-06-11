"""
SplatFlow Attention Components Module - FIXED VERSION
Core splat and attention mechanism implementations with intelligent birth system.
FIXES: Variable naming, device management, positioning, birth system integration
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

# Import adaptive birth system
try:
    from .splatflow_adaptive_birth import (
        AdaptiveSplatBirthManager,
        CoverageAnalyzer,
        TrajectoryBirthAnalyzer,
        SplatBirthRequest
    )
    ADAPTIVE_BIRTH_AVAILABLE = True
except ImportError:
    # Fallback if adaptive birth module not available
    ADAPTIVE_BIRTH_AVAILABLE = False
    AdaptiveSplatBirthManager = None
    logger.warning("Adaptive birth system not available - using basic splat management")

# Import enhanced coverage algorithms
try:
    from .enhanced_coverage_algorithms import (
        IntelligentSplatPositioner, 
        CoverageAwareSplatAdapter,
        integrate_enhanced_coverage
    )
    ENHANCED_COVERAGE_AVAILABLE = True
except ImportError:
    # Fallback if enhanced coverage module not available
    ENHANCED_COVERAGE_AVAILABLE = False
    IntelligentSplatPositioner = None
    CoverageAwareSplatAdapter = None

logger = logging.getLogger(__name__)


class FixedProductionTrajectoryFlowSplat:
    """FIXED splat with adaptive birth integration and improved positioning"""
    
    def __init__(self, position: torch.Tensor, scale: float, amplitude: float, 
                 splat_id: int, device: torch.device, layer_idx: int = 0):
        self.device = device
        self.id = splat_id
        self.layer_idx = layer_idx
        
        # FIXED: Ensure all parameters are on correct device from initialization
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
        
        # FIXED: Improved embedding statistics initialization
        self.embedding_stats = {
            'mean_magnitude': 2.0,      # More realistic initial value
            'std_magnitude': 1.0,       # Better initial estimate
            'median_inter_token_distance': 4.0,  # Increased for better coverage
            'sample_count': 0
        }
        
        # Coverage tracking
        self.coverage_contribution = 0.0
        self.last_coverage_update = 0
        
        # FIXED: Adaptive radius with improved defaults
        self.max_healthy_radius = 10.0  # Increased from 8.0
        self.expansion_warning_threshold = 8.0  # Increased from 6.0
        self.birth_request_callback = None
        self.last_radius_expansion_attempt = 0
        self.birth_reason = None
        
    def update_embedding_statistics(self, sample_embeddings: torch.Tensor):
        """FIXED: Update embedding statistics with better error handling"""
        with torch.no_grad():
            try:
                # FIXED: Ensure embeddings are on correct device
                sample_embeddings = sample_embeddings.to(self.device)
                flat_embeddings = sample_embeddings.reshape(-1, sample_embeddings.shape[-1])
                
                if len(flat_embeddings) == 0:
                    return
                
                # Update magnitude statistics with safety checks
                magnitudes = torch.norm(flat_embeddings, dim=-1)
                
                if len(magnitudes) > 0:
                    mean_mag = magnitudes.mean().item()
                    std_mag = magnitudes.std().item() if len(magnitudes) > 1 else 1.0
                    
                    # FIXED: More realistic bounds based on actual embeddings
                    self.embedding_stats['mean_magnitude'] = max(0.5, min(mean_mag, 15.0))
                    self.embedding_stats['std_magnitude'] = max(0.5, min(std_mag, 8.0))
                
                # Update inter-token distance statistics
                if len(flat_embeddings) > 1:
                    # Sample subset to avoid memory issues
                    sample_size = min(30, len(flat_embeddings))  # Reduced for efficiency
                    sample_indices = torch.randperm(len(flat_embeddings), device=self.device)[:sample_size]
                    sample_tokens = flat_embeddings[sample_indices]
                    
                    if len(sample_tokens) > 1:
                        distances = torch.cdist(sample_tokens, sample_tokens)
                        distances = distances[distances > 1e-6]  # Remove near-zero distances
                        
                        if len(distances) > 0:
                            median_dist = torch.median(distances).item()
                            # FIXED: Better range for median distance
                            self.embedding_stats['median_inter_token_distance'] = max(1.0, min(median_dist, 12.0))
                
                self.embedding_stats['sample_count'] += 1
                
            except Exception as e:
                logger.warning(f"Failed to update embedding statistics for splat {self.id}: {e}")
    
    def compute_adaptive_influence_radius(self, token_embeddings: torch.Tensor) -> float:
        """FIXED: Compute radius with improved defaults and birth requests"""
        with torch.no_grad():
            try:
                # FIXED: Ensure embeddings are on correct device
                token_embeddings = token_embeddings.to(self.device)
                
                # FIXED: Better base radius calculation
                base_radius = self.embedding_stats['median_inter_token_distance'] * 2.0  # Increased multiplier
                
                flat_tokens = token_embeddings.reshape(-1, token_embeddings.shape[-1])
                
                if len(flat_tokens) > 5:
                    # Calculate distance from splat to all tokens
                    splat_expanded = self.position.unsqueeze(0)
                    distances_to_splat = torch.norm(flat_tokens - splat_expanded, dim=-1)
                    
                    if len(distances_to_splat) > 0:
                        radius_candidates = []
                        
                        # FIXED: More robust quantile computation
                        try:
                            if len(distances_to_splat) >= 10:
                                # Use percentile-based approach
                                sorted_distances, _ = torch.sort(distances_to_splat)
                                # Take 20th percentile for better coverage
                                percentile_20 = sorted_distances[len(sorted_distances) // 5].item()
                                radius_candidates.append(percentile_20)
                        except Exception:
                            pass
                        
                        # Add fallback candidates
                        radius_candidates.extend([
                            self.embedding_stats['median_inter_token_distance'] * 1.5,
                            base_radius,
                            3.0  # Minimum fallback
                        ])
                        
                        # Choose radius that covers at least some tokens
                        for radius in sorted(radius_candidates):
                            # FIXED: Apply hard radius limit with birth requests
                            if radius > self.max_healthy_radius:
                                self._request_birth_for_distant_tokens(flat_tokens, radius)
                                return min(self.max_healthy_radius, base_radius)
                            
                            if radius > 0 and (distances_to_splat < radius).sum().item() > 0:
                                # Check if we should warn about expansion
                                if radius > self.expansion_warning_threshold:
                                    self._consider_birth_for_expansion(flat_tokens, radius)
                                
                                return max(1.0, min(radius, self.max_healthy_radius))
                
                return max(2.0, min(base_radius, self.max_healthy_radius))  # FIXED: Better minimum
                
            except Exception as e:
                logger.warning(f"Failed to compute adaptive radius for splat {self.id}: {e}")
                return min(3.0, self.max_healthy_radius)  # Safe fallback
    
    def _request_birth_for_distant_tokens(self, flat_tokens: torch.Tensor, required_radius: float):
        """FIXED: Request birth of new splat for distant tokens"""
        try:
            if self.birth_request_callback is None:
                return
            
            # FIXED: Ensure tokens are on correct device
            flat_tokens = flat_tokens.to(self.device)
            
            # Find tokens that would require large radius
            splat_expanded = self.position.unsqueeze(0)
            distances_to_splat = torch.norm(flat_tokens - splat_expanded, dim=-1)
            
            # Find distant token cluster
            distant_mask = distances_to_splat > self.max_healthy_radius
            if distant_mask.any():
                distant_tokens = flat_tokens[distant_mask]
                
                # Calculate centroid of distant tokens
                if len(distant_tokens) > 0:
                    centroid = distant_tokens.mean(dim=0).to(self.device)
                    
                    # Calculate urgency based on how many tokens and how far
                    urgency = len(distant_tokens) / len(flat_tokens)  # Proportion of tokens
                    urgency *= (required_radius / self.max_healthy_radius)  # Distance factor
                    
                    # Make birth request
                    self.birth_request_callback(
                        position=centroid,
                        reason="radius_limit_reached",
                        urgency=urgency,
                        parent_splat_id=self.id,
                        token_cluster_size=len(distant_tokens)
                    )
                    
                    self.last_radius_expansion_attempt = self.age
                    
                    # FIXED: Reduced debug spam
                    if self.id == 0 and random.random() < 0.1:  # Only 10% of the time
                        print(f"    🐣 Splat {self.id}: Requested birth for {len(distant_tokens)} distant tokens "
                              f"(urgency={urgency:.3f})")
                              
        except Exception as e:
            logger.warning(f"Failed to request birth for splat {self.id}: {e}")
    
    def _consider_birth_for_expansion(self, flat_tokens: torch.Tensor, proposed_radius: float):
        """FIXED: Consider birth when approaching radius limits"""
        try:
            if self.birth_request_callback is None:
                return
            
            # Only consider if we haven't requested recently
            if self.age - self.last_radius_expansion_attempt < 10:
                return
            
            # FIXED: Ensure tokens are on correct device
            flat_tokens = flat_tokens.to(self.device)
            
            # If expansion is significant, consider birth
            current_scale = torch.exp(self.log_scale).item() if hasattr(self, 'log_scale') else 3.0
            if proposed_radius > current_scale * 1.5:
                
                # Look for token clusters that could benefit from new splat
                splat_expanded = self.position.unsqueeze(0)
                distances_to_splat = torch.norm(flat_tokens - splat_expanded, dim=-1)
                
                # Find moderately distant tokens
                moderate_mask = (distances_to_splat > current_scale * 1.2) & (distances_to_splat < self.max_healthy_radius)
                
                if moderate_mask.sum().item() >= 3:  # At least 3 tokens
                    moderate_tokens = flat_tokens[moderate_mask]
                    centroid = moderate_tokens.mean(dim=0).to(self.device)
                    
                    urgency = 0.3  # Lower urgency than hard limit
                    
                    self.birth_request_callback(
                        position=centroid,
                        reason="expansion_prevention",
                        urgency=urgency,
                        parent_splat_id=self.id,
                        token_cluster_size=len(moderate_tokens)
                    )
                    
                    self.last_radius_expansion_attempt = self.age
                    
        except Exception as e:
            logger.warning(f"Failed to consider birth for splat {self.id}: {e}")
    
    def update_with_fixed_trajectory_flow(self, layer_trajectory: torch.Tensor, 
                                        token_embeddings: torch.Tensor, 
                                        splat_network: Optional[Dict] = None,
                                        epoch: int = 0):
        """FIXED: Update splat with enhanced trajectory flow and proper device handling"""
        self.age += 1
        
        try:
            # FIXED: Ensure all tensors are on correct device
            layer_trajectory = layer_trajectory.to(self.device)
            token_embeddings = token_embeddings.to(self.device)
            
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
                            adaptive_lr * trajectory_influence).to(self.device)
            
            # FIXED: Improved movement bounds
            max_vel = self.embedding_stats['std_magnitude'] * 0.8  # Increased from 0.5
            max_vel = max(0.5, min(max_vel, 3.0))  # Better bounds
            self.velocity = torch.clamp(self.velocity, -max_vel, max_vel)
            
            # Update position with bounds based on embedding statistics
            with torch.no_grad():
                new_position = self.position + self.velocity
                # FIXED: More realistic bounds
                bound_scale = self.embedding_stats['mean_magnitude'] + 3 * self.embedding_stats['std_magnitude']
                bounds = max(5.0, min(bound_scale, 20.0))  # Increased bounds
                self.position.data = torch.clamp(new_position, -bounds, bounds)
            
            # FIXED: Progressive usefulness criteria
            if len(self.trajectory_influence_history) > 0:
                recent_influence = np.mean(self.trajectory_influence_history[-20:])
            else:
                recent_influence = 0.0
            
            # FIXED: More lenient progressive thresholds
            if epoch < 5:
                baseline_influence = 1e-7  # Very lenient early on
            elif epoch < 15:
                baseline_influence = 1e-6  # Moderate
            else:
                baseline_influence = 1e-5  # Normal after warmup (increased from 1e-4)
            
            usefulness_delta = 0.1 * (recent_influence - baseline_influence)
            self.usefulness = np.clip(self.usefulness + usefulness_delta, 0.1, 5.0)
            self.flow_magnitude = influence_magnitude
            
        except Exception as e:
            logger.warning(f"Failed to update splat {self.id} with trajectory flow: {e}")
    
    def compute_fixed_trajectory_influence(self, layer_trajectory: torch.Tensor, 
                                         token_embeddings: torch.Tensor,
                                         epoch: int = 0) -> torch.Tensor:
        """FIXED: Compute trajectory influence with improved radius and less emergency positioning"""
        try:
            batch_size, seq_len, dim = token_embeddings.shape
            
            if seq_len == 0 or batch_size == 0:
                return torch.zeros_like(self.position, device=self.device)
            
            # FIXED: Ensure all tensors on correct device
            layer_trajectory = layer_trajectory.to(self.device)
            token_embeddings = token_embeddings.to(self.device)
            
            # Use adaptive influence radius
            influence_radius = self.compute_adaptive_influence_radius(token_embeddings)
            
            splat_expanded = self.position.unsqueeze(0).unsqueeze(0)
            distances = torch.norm(token_embeddings - splat_expanded, dim=-1)
            
            # FIXED: Reduced debug spam and better emergency positioning logic
            if distances.numel() > 0:
                min_dist = torch.min(distances).item()
                mean_dist = torch.mean(distances).item()
                
                # Debug output less frequently
                if self.id == 0 and random.random() < 0.02:  # Only 2% of the time
                    print(f"    🔧 FIXED Splat {self.id}: min_dist={min_dist:.3f}, mean_dist={mean_dist:.3f}, radius={influence_radius:.3f}")
            
            # Progressive radius expansion if no tokens in range
            influence_mask = distances < influence_radius
            tokens_in_influence = influence_mask.sum().item()
            
            if tokens_in_influence == 0:
                # FIXED: More conservative expansion with better emergency positioning
                expansion_factors = [1.5, 2.0, 3.0, 4.0]  # Reduced from [1.5, 2.0, 3.0, 5.0, 8.0]
                for expansion_factor in expansion_factors:
                    expanded_radius = influence_radius * expansion_factor
                    influence_mask = distances < expanded_radius
                    if influence_mask.any():
                        influence_radius = expanded_radius
                        tokens_in_influence = influence_mask.sum().item()
                        if self.id == 0 and random.random() < 0.05:  # Less frequent logging
                            print(f"    🔧 FIXED Splat {self.id}: expanded radius to {expanded_radius:.3f}, found {tokens_in_influence} tokens")
                        break
                else:
                    # FIXED: More conservative emergency repositioning
                    if epoch < 5 and distances.numel() > 0:  # Only in very early epochs
                        closest_token_idx = torch.argmin(distances.reshape(-1))
                        closest_token = token_embeddings.reshape(-1, dim)[closest_token_idx]
                        
                        with torch.no_grad():
                            # Move 30% toward closest token (reduced from 50%)
                            direction = closest_token - self.position
                            self.position.data += 0.3 * direction
                        
                        if self.id == 0 and random.random() < 0.1:  # Less frequent emergency logging
                            print(f"    🚨 EMERGENCY: Moved splat {self.id} toward closest token")
                    
                    return torch.zeros_like(self.position, device=self.device)
            
            # FIXED: Improved influence computation
            proximity_weights = torch.exp(-distances / (influence_radius * 0.4))  # Slightly wider falloff
            proximity_weights = proximity_weights * influence_mask.float()
            
            # More sensitive trajectory magnitude weighting
            traj_magnitudes = torch.norm(layer_trajectory, dim=-1)
            magnitude_weights = torch.sigmoid(traj_magnitudes * 0.8)  # More sensitive
            
            total_weights = proximity_weights * magnitude_weights
            total_weight_sum = safe_tensor_to_scalar(total_weights.sum())
            
            if self.id == 0 and random.random() < 0.01:  # Very infrequent logging
                print(f"    🔧 FIXED Splat {self.id}: tokens_in_range={tokens_in_influence}, total_weight={total_weight_sum:.6f}")
            
            if total_weight_sum < 1e-12:
                return torch.zeros_like(self.position, device=self.device)
            
            # Compute weighted trajectory influence with safe division
            weighted_trajectories = layer_trajectory * total_weights.unsqueeze(-1)
            influence_vector = weighted_trajectories.sum(dim=(0, 1)) / max(total_weight_sum, 1e-12)
            
            # Layer-specific boost
            layer_boost = 1.0 + self.layer_idx * 1.2
            influence_vector = influence_vector * layer_boost
            
            return influence_vector.to(self.device)
            
        except Exception as e:
            logger.warning(f"Failed to compute trajectory influence for splat {self.id}: {e}")
            return torch.zeros_like(self.position, device=self.device)
    
    def compute_inter_splat_flow(self, splat_network: Dict) -> torch.Tensor:
        """Compute enhanced flow between connected splats"""
        inter_flow = torch.zeros_like(self.position, device=self.device)
        
        try:
            for other_splat in splat_network.values():
                if other_splat.id != self.id:
                    # FIXED: Ensure other splat position is on correct device
                    other_position = other_splat.position.to(self.device)
                    direction = other_position - self.position
                    distance = torch.norm(direction)
                    
                    if distance > 1e-6:
                        normalized_direction = direction / max(distance, 1e-6)
                        optimal_distance = 2.0 + self.layer_idx * 0.5  # Increased optimal distance
                        
                        if distance > optimal_distance:
                            flow_strength = 0.2 * (distance - optimal_distance)
                            inter_flow += flow_strength * normalized_direction
                        else:
                            flow_strength = 0.3 * (optimal_distance - distance)
                            inter_flow -= flow_strength * normalized_direction
        except Exception as e:
            logger.warning(f"Inter-splat flow computation failed for splat {self.id}: {e}")
        
        return inter_flow.to(self.device)
    
    def is_healthy(self, epoch: int = 0) -> bool:
        """FIXED: Enhanced health check with more lenient progressive criteria"""
        try:
            if len(self.trajectory_influence_history) > 0:
                recent_influence = np.mean(self.trajectory_influence_history[-10:])
            else:
                recent_influence = 0.0
            
            # FIXED: More lenient progressive thresholds
            if epoch < 5:
                influence_threshold = 1e-7  # Very lenient for early training
                usefulness_threshold = 0.01
            elif epoch < 15:
                influence_threshold = 1e-6  # Moderate
                usefulness_threshold = 0.05
            else:
                influence_threshold = 1e-5  # Normal threshold (increased from 1e-4)
                usefulness_threshold = 0.1
            
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
                'embedding_stats': self.embedding_stats,
                'coverage_contribution': self.coverage_contribution,
                'birth_reason': self.birth_reason,
                'max_radius': self.max_healthy_radius
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
                'embedding_stats': self.embedding_stats,
                'coverage_contribution': 0.0,
                'birth_reason': self.birth_reason,
                'max_radius': self.max_healthy_radius
            }


class FixedProductionSplatFlowAttention(nn.Module):
    """FIXED production-ready SplatFlow attention with adaptive birth system"""
    
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
        
        self.token_value_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.output_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Enhanced trajectory strength
        initial_strength = 0.6 + layer_idx * 0.3
        self.trajectory_strength = nn.Parameter(torch.tensor(initial_strength))
        
        # FIXED: Initialize birth management system with proper error handling
        if ADAPTIVE_BIRTH_AVAILABLE:
            try:
                self.birth_manager = AdaptiveSplatBirthManager(
                    max_splats=max_splats,
                    max_radius=10.0,  # Increased from 8.0
                    max_births_per_epoch=2,
                    birth_cooldown=3
                )
                logger.info(f"🐣 Adaptive birth system enabled for layer {layer_idx}")
            except Exception as e:
                logger.warning(f"Failed to initialize birth manager: {e}")
                self.birth_manager = None
        else:
            self.birth_manager = None
            logger.warning(f"Adaptive birth system not available for layer {layer_idx}")
        
        # Enhanced coverage components (existing)
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
        
        # Initialize with placeholder splats
        self._initialize_placeholder_splats()
        self._init_weights()
        
        logger.info(f"🎯 FIXED SplatFlow attention initialized for layer {layer_idx}")
    
    def _initialize_placeholder_splats(self):
        """Initialize placeholder splats with better positioning"""
        device = DeviceManager.get_primary_device()
        self.splats = []
        
        for i in range(self.num_splats):
            # FIXED: Better placeholder initialization
            position = torch.randn(self.model_dim, device=device) * 0.5  # Reduced variance
            scale = 2.0 + torch.rand(1).item() * 1.0  # Increased base scale
            amplitude = 1.0 + torch.rand(1).item() * 0.5
            
            splat = FixedProductionTrajectoryFlowSplat(position, scale, amplitude, i, device, self.layer_idx)
            
            # FIXED: Connect birth request callback with error handling
            if self.birth_manager:
                try:
                    splat.birth_request_callback = self.birth_manager.request_splat_birth
                except Exception as e:
                    logger.warning(f"Failed to set birth callback for splat {i}: {e}")
            
            self.splats.append(splat)
        
        logger.info(f"🎯 Initialized {len(self.splats)} placeholder splats for layer {self.layer_idx}")
    
    def fix_splat_positioning_based_on_embeddings(self, sample_embeddings: torch.Tensor):
        """FIXED: Properly position splats with better error handling"""
        with torch.no_grad():
            try:
                device = self.splats[0].device if self.splats else DeviceManager.get_primary_device()
                sample_embeddings = sample_embeddings.to(device)
                
                print(f"🎯 FIXED: Positioning splats for layer {self.layer_idx}...")
                
                # Use enhanced coverage positioning if available
                if self.coverage_positioner:
                    try:
                        # Generate intelligent positions using clustering
                        intelligent_positions = self.coverage_positioner.generate_intelligent_positions(
                            sample_embeddings, num_splats=len(self.splats)
                        )
                        
                        # Apply intelligent positions to splats
                        for i, splat in enumerate(self.splats):
                            if i < len(intelligent_positions):
                                splat.position.data = intelligent_positions[i].to(device)
                                print(f"   ✅ Splat {i}: intelligently positioned at magnitude {torch.norm(splat.position).item():.3f}")
                        
                        logger.info(f"✅ ENHANCED: Layer {self.layer_idx} splats positioned with intelligent clustering")
                        return
                        
                    except Exception as e:
                        logger.warning(f"Enhanced positioning failed: {e}, falling back to basic method")
                
                # Fallback to improved basic method
                self._fallback_positioning(sample_embeddings, device)
                
            except Exception as e:
                logger.error(f"Failed to fix splat positioning for layer {self.layer_idx}: {e}")
    
    def _fallback_positioning(self, sample_embeddings: torch.Tensor, device: torch.device):
        """FIXED: Improved fallback positioning method"""
        # Get comprehensive embedding statistics
        flat_embeddings = sample_embeddings.reshape(-1, self.model_dim)
        
        if len(flat_embeddings) == 0:
            logger.warning(f"Empty embeddings for layer {self.layer_idx}, skipping positioning")
            return
        
        # FIXED: Better statistics calculation
        mean_pos = flat_embeddings.mean(dim=0)
        std_pos = flat_embeddings.std(dim=0)
        
        # Ensure std is not zero
        std_pos = torch.clamp(std_pos, min=0.2)  # Increased minimum
        
        print(f"🔧 FIXED Layer {self.layer_idx} embedding analysis:")
        print(f"   Token count: {len(flat_embeddings)}")
        print(f"   Mean magnitude: {torch.norm(mean_pos).item():.3f}")
        print(f"   Std magnitude: {torch.norm(std_pos).item():.3f}")
        
        # Update embedding statistics for all splats
        for splat in self.splats:
            splat.update_embedding_statistics(sample_embeddings)
        
        # FIXED: Better positioning strategy
        for i, splat in enumerate(self.splats):
            try:
                if i < len(flat_embeddings):
                    # Start from actual token position
                    base_pos = flat_embeddings[i % len(flat_embeddings)]
                else:
                    # Sample from distribution with better spread
                    base_pos = mean_pos + torch.randn_like(mean_pos, device=device) * std_pos * 0.8
                
                # FIXED: Better perturbation to avoid exact overlap
                perturbation = torch.randn_like(base_pos, device=device) * torch.norm(std_pos) * 0.3
                new_position = base_pos + perturbation
                
                # Update splat position
                splat.position.data = new_position.to(device)
                
                # Reset other parameters
                splat.usefulness = 2.0
                splat.velocity.zero_()
                splat.trajectory_influence_history.clear()
                
                print(f"   ✅ Splat {i}: repositioned to magnitude {torch.norm(splat.position).item():.3f}")
            except Exception as e:
                logger.warning(f"Failed to reposition splat {i}: {e}")
    
    def progressive_splat_repositioning(self, layer_embeddings: torch.Tensor, epoch: int):
        """FIXED: Progressive splat repositioning with better error handling"""
        if epoch % 3 == 0 and epoch > 0:  # Every 3 epochs after first
            with torch.no_grad():
                try:
                    device = self.splats[0].device if self.splats else DeviceManager.get_primary_device()
                    layer_embeddings = layer_embeddings.to(device)
                    sample_size = min(50, layer_embeddings.size(1))
                    sample_embeddings = layer_embeddings[:, :sample_size].to(device)
                    
                    # Use enhanced coverage-aware repositioning if available
                    if self.coverage_adapter:
                        try:
                            coverage_stats = self.coverage_adapter.coverage_aware_update(
                                self.splats, sample_embeddings, epoch
                            )
                            
                            if coverage_stats['improvements_made'] > 0:
                                print(f"    🎯 Layer {self.layer_idx}: Coverage-aware repositioning improved {coverage_stats['improvements_made']} splats")
                        except Exception as e:
                            logger.warning(f"Coverage-aware repositioning failed: {e}")
                    
                    else:
                        # FIXED: Improved fallback repositioning
                        repositioned_count = 0
                        flat_embeddings = sample_embeddings.reshape(-1, self.model_dim)
                        
                        for splat in self.splats:
                            try:
                                # Find closest tokens to current splat position
                                distances = torch.norm(flat_embeddings - splat.position.unsqueeze(0), dim=-1)
                                if len(distances) > 0:
                                    # Get multiple close tokens for better positioning
                                    top_k = min(5, len(distances))
                                    _, closest_indices = torch.topk(distances, top_k, largest=False)
                                    closest_tokens = flat_embeddings[closest_indices]
                                    
                                    # Move toward centroid of closest tokens
                                    target_position = closest_tokens.mean(dim=0)
                                    direction = target_position - splat.position
                                    
                                    move_strength = 0.1 if epoch < 10 else 0.05  # More conservative
                                    splat.position.data += move_strength * direction
                                    
                                    # Add small random perturbation
                                    perturbation_strength = 0.02 if epoch < 10 else 0.01
                                    perturbation = torch.randn_like(splat.position, device=device) * perturbation_strength
                                    splat.position.data += perturbation
                                    
                                    repositioned_count += 1
                            except Exception as e:
                                logger.warning(f"Failed to reposition splat {splat.id}: {e}")
                        
                        if repositioned_count > 0:
                            print(f"    🔄 Layer {self.layer_idx}: Repositioned {repositioned_count} splats (epoch {epoch})")
                        
                except Exception as e:
                    logger.warning(f"Progressive repositioning failed for layer {self.layer_idx}: {e}")
    
    def emergency_splat_rescue(self, layer_embeddings: torch.Tensor, epoch: int):
        """FIXED: More conservative emergency system"""
        try:
            healthy_count = sum(1 for splat in self.splats if splat.is_healthy(epoch))
            
            # FIXED: More conservative rescue trigger (15% instead of 25%)
            if len(self.splats) > 0 and healthy_count < len(self.splats) * 0.15:
                print(f"    🚨 EMERGENCY RESCUE Layer {self.layer_idx}: Only {healthy_count}/{len(self.splats)} healthy splats")
                
                with torch.no_grad():
                    device = self.splats[0].device if self.splats else DeviceManager.get_primary_device()
                    layer_embeddings = layer_embeddings.to(device)
                    flat_embeddings = layer_embeddings.reshape(-1, self.model_dim)
                    
                    if len(flat_embeddings) == 0:
                        return
                    
                    # FIXED: More strategic rescue
                    unhealthy_splats = [s for s in self.splats if not s.is_healthy(epoch)]
                    rescued_count = 0
                    
                    for i, splat in enumerate(unhealthy_splats[:min(3, len(unhealthy_splats))]):  # Limit rescues
                        try:
                            # Move to a well-distributed position
                            if i < len(flat_embeddings):
                                # Use spread distribution instead of sequential
                                token_idx = (i * len(flat_embeddings)) // max(len(unhealthy_splats), 1)
                                token_idx = min(token_idx, len(flat_embeddings) - 1)
                                target_position = flat_embeddings[token_idx].clone()
                                
                                # Add strategic offset
                                offset = torch.randn_like(target_position, device=device) * 0.3
                                splat.position.data = target_position + offset
                                
                                # Reset splat parameters more conservatively
                                splat.usefulness = 1.0  # Reduced from 1.5
                                splat.velocity.zero_()
                                splat.trajectory_influence_history.clear()
                                
                                rescued_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to rescue splat {splat.id}: {e}")
                    
                    if rescued_count > 0:
                        print(f"      🔧 Rescued {rescued_count} splats to strategic positions")
                    
        except Exception as e:
            logger.warning(f"Emergency rescue failed for layer {self.layer_idx}: {e}")
    
    def _init_weights(self):
        """Initialize weights with production-level scaling"""
        std = 0.02 / math.sqrt(self.layer_idx + 1)
        nn.init.normal_(self.token_value_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=std)
    
    def compute_production_attention_matrix(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """FIXED: Compute attention matrix with better device handling"""
        device = DeviceManager.get_primary_device()
        batch_size, seq_len, model_dim = token_embeddings.shape
        
        token_embeddings = token_embeddings.to(device)
        
        if not self.splats:
            logger.warning(f"No splats available in layer {self.layer_idx}, using uniform attention")
            return torch.ones(batch_size, seq_len, 1, device=device) / 1.0
        
        try:
            centers = []
            scales = []
            amplitudes = []
            
            for splat in self.splats:
                try:
                    centers.append(splat.position.detach().to(device))
                    scales.append(torch.exp(splat.log_scale).detach().clamp(min=0.1, max=12.0).to(device))  # Increased max
                    amplitudes.append(splat.amplitude.detach().clamp(min=0.1, max=5.0).to(device))
                except Exception as e:
                    logger.warning(f"Failed to get parameters for splat {splat.id}: {e}")
                    # Skip this splat
                    continue
            
            if len(centers) == 0:
                return torch.ones(batch_size, seq_len, 1, device=device) / 1.0
            
            centers = torch.stack(centers, dim=0).to(device)
            scales = torch.stack(scales, dim=0).to(device)
            amplitudes = torch.stack(amplitudes, dim=0).to(device)
            
            tokens_expanded = token_embeddings.unsqueeze(2)
            centers_expanded = centers.unsqueeze(0).unsqueeze(0)
            
            diff = tokens_expanded - centers_expanded
            distances_sq = torch.sum(diff ** 2, dim=-1)
            
            scales_sq = scales ** 2
            scales_sq = torch.clamp(scales_sq, min=1e-6)
            normalized_distances = distances_sq / scales_sq.unsqueeze(0).unsqueeze(0)
            normalized_distances = torch.clamp(normalized_distances, max=50.0)
            
            gaussian_weights = torch.exp(-0.5 * normalized_distances)
            attention_weights = gaussian_weights * amplitudes.unsqueeze(0).unsqueeze(0)
            
            attention_sums = attention_weights.sum(dim=-1, keepdim=True)
            attention_sums = torch.clamp(attention_sums, min=1e-12)
            attention_weights = attention_weights / attention_sums
            
            return attention_weights
            
        except Exception as e:
            logger.error(f"Production attention computation failed for layer {self.layer_idx}: {e}")
            fallback_attention = torch.ones(batch_size, seq_len, max(1, len(self.splats)), device=device)
            fallback_attention = fallback_attention / max(1, len(self.splats))
            return fallback_attention
    
    def forward(self, token_embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                epoch: int = 0) -> torch.Tensor:
        """FIXED: Production-level forward pass with proper birth management"""
        self.forward_count += 1
        device = DeviceManager.get_primary_device()
        batch_size, seq_len, model_dim = token_embeddings.shape
        
        token_embeddings = token_embeddings.to(device)
        
        try:
            if self.trajectory_computer is not None:
                trajectories, _ = self.trajectory_computer.compute_enhanced_trajectory_flow(self.layer_idx, token_embeddings)
                trajectories = trajectories.to(device)
                
                traj_magnitude = torch.norm(trajectories).item()
                if traj_magnitude < 0.001:
                    trajectories = trajectories + torch.randn_like(trajectories, device=device) * 0.05
            else:
                trajectories = torch.randn_like(token_embeddings, device=device) * 0.05
                trajectories = trajectories.to(device)
                
        except Exception as e:
            logger.error(f"Trajectory computation failed for layer {self.layer_idx}: {e}")
            trajectories = torch.randn_like(token_embeddings, device=device) * 0.05
            trajectories = trajectories.to(device)
        
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
                attention_mask = attention_mask.to(device)
                output = output * attention_mask.unsqueeze(-1)
            
            if (self.training and self.adaptation_enabled and 
                self.forward_count % self.adaptation_frequency == 0):
                with torch.no_grad():
                    self.adapt_splats_for_production(token_embeddings, trajectories, attention_weights, epoch)
            
            return output
            
        except Exception as e:
            logger.error(f"Forward pass failed for layer {self.layer_idx}: {e}")
            return token_embeddings
    
    def adapt_splats_for_production(self, token_embeddings: torch.Tensor, 
                                  trajectories: torch.Tensor, 
                                  attention_weights: torch.Tensor,
                                  epoch: int = 0):
        """FIXED: Production-level splat adaptation with proper birth management"""
        if not self.adaptation_enabled or not self.splats:
            return
        
        device = DeviceManager.get_primary_device()
        
        try:
            # FIXED: Ensure all tensors are on correct device
            token_embeddings = token_embeddings.to(device)
            trajectories = trajectories.to(device)
            attention_weights = attention_weights.to(device)
            
            if attention_weights.size(-1) > 0:
                splat_activations = attention_weights.mean(dim=(0, 1))
            else:
                splat_activations = torch.zeros(len(self.splats), device=device)
                
            trajectory_strength_value = torch.sigmoid(self.trajectory_strength) * 5.0
            
            splat_network = {splat.id: splat for splat in self.splats}
            
            # FIXED: Process birth requests before updating existing splats
            if self.birth_manager:
                try:
                    new_splat_params = self.birth_manager.process_birth_requests(
                        current_splats=self.splats,
                        token_embeddings=token_embeddings,
                        trajectories=trajectories,
                        epoch=epoch
                    )
                    
                    # FIXED: Create actual splats from parameters (corrected variable name)
                    for splat_params in new_splat_params:
                        try:
                            # Create new splat from parameters
                            new_splat = FixedProductionTrajectoryFlowSplat(
                                position=splat_params['position'],
                                scale=splat_params['scale'],
                                amplitude=splat_params['amplitude'],
                                splat_id=splat_params['splat_id'],
                                device=splat_params['device'],
                                layer_idx=self.layer_idx
                            )
                            
                            # Initialize as newborn
                            new_splat.age = 0
                            new_splat.usefulness = 2.0
                            new_splat.velocity = torch.zeros_like(splat_params['position'])
                            new_splat.birth_reason = splat_params.get('birth_reason', 'unknown')
                            
                            # Set up birth request callback
                            if self.birth_manager:
                                new_splat.birth_request_callback = self.birth_manager.request_splat_birth
                            
                            self.splats.append(new_splat)
                            splat_network[new_splat.id] = new_splat
                            
                        except Exception as e:
                            logger.warning(f"Failed to create splat from parameters: {e}")
                    
                    if new_splat_params:  # FIXED: Use correct variable name
                        logger.info(f"🐣 Layer {self.layer_idx}: Added {len(new_splat_params)} new splats "
                                  f"(total: {len(self.splats)})")
                except Exception as e:
                    logger.warning(f"Birth request processing failed: {e}")
            
            # Update all splats (including new ones)
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
        """Get comprehensive production-level statistics with birth management metrics"""
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
                    'health_status': '🔴 CRITICAL - NO SPLATS'
                }
            
            splat_stats = []
            for splat in self.splats:
                try:
                    stats = splat.get_production_stats(epoch)
                    splat_stats.append(stats)
                except Exception as e:
                    logger.warning(f"Failed to get stats for splat {splat.id}: {e}")
                    # Add default stats for failed splat
                    splat_stats.append({
                        'layer_idx': self.layer_idx,
                        'num_splats': 0,
                        'healthy_splats': 0,
                        'is_healthy': False
                    })
            
            healthy_splats = sum(1 for s in splat_stats if s.get('is_healthy', False))
            
            if len(splat_stats) > 0:
                avg_usefulness = np.mean([s.get('usefulness', 0) for s in splat_stats])
                avg_trajectory_influence = np.mean([s.get('avg_trajectory_influence', 0) for s in splat_stats])
            else:
                avg_usefulness = 0.0
                avg_trajectory_influence = 0.0
            
            # Coverage efficiency calculation
            coverage_efficiency = 0.0
            if self.coverage_positioner and self.splats:
                try:
                    # Simplified coverage calculation
                    coverage_efficiency = min(1.0, healthy_splats / max(len(self.splats), 1))
                except Exception:
                    coverage_efficiency = 0.0
            
            # FIXED: Get birth management statistics with error handling
            birth_stats = {}
            if self.birth_manager:
                try:
                    birth_stats = self.birth_manager.get_birth_statistics()
                except Exception as e:
                    logger.warning(f"Failed to get birth stats: {e}")
                    birth_stats = {'error': str(e)}
            
            # FIXED: More lenient health status thresholds
            if epoch < 5:
                if healthy_splats >= 1:
                    health_status = '🟢 HEALTHY (Early)'
                else:
                    health_status = '🟡 DEVELOPING'
            elif epoch < 15:
                if healthy_splats >= max(1, self.min_splats // 2):  # More lenient
                    health_status = '🟢 HEALTHY'
                elif healthy_splats >= max(1, self.min_splats // 4):  # More lenient
                    health_status = '🟡 WEAK'
                else:
                    health_status = '🔴 CRITICAL'
            else:
                if healthy_splats >= max(1, self.min_splats // 2):  # More lenient
                    health_status = '🟢 HEALTHY'
                elif healthy_splats >= max(1, self.min_splats // 4):  # More lenient
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
                'coverage_efficiency': coverage_efficiency,
                'birth_stats': birth_stats,
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
                'coverage_efficiency': 0.0,
                'birth_stats': {'error': str(e)},
                'health_status': '🔴 ERROR'
            }


def get_quick_model_stats(model) -> Dict:
    """FIXED: Get quick model statistics for batch logging including birth stats"""
    try:
        # Get trajectory flow statistics
        try:
            flow_stats = model.trajectory_flow.get_comprehensive_statistics()
        except Exception:
            flow_stats = {}
        
        # Get splat health from all layers
        total_splats = 0
        healthy_splats = 0
        total_trajectory_influence = 0
        total_coverage_efficiency = 0.0
        total_births = 0
        total_birth_requests = 0
        
        for layer in model.layers:
            if hasattr(layer, 'attention') and hasattr(layer.attention, 'get_production_stats'):
                try:
                    stats = layer.attention.get_production_stats()
                    total_splats += stats.get('num_splats', 0)
                    healthy_splats += stats.get('healthy_splats', 0)
                    total_trajectory_influence += stats.get('avg_trajectory_influence', 0)
                    total_coverage_efficiency += stats.get('coverage_efficiency', 0.0)
                    
                    # FIXED: Safe birth statistics collection
                    birth_stats = stats.get('birth_stats', {})
                    total_births += birth_stats.get('total_births', 0)
                    total_birth_requests += birth_stats.get('pending_requests', 0)
                except Exception as e:
                    logger.warning(f"Failed to get stats from layer: {e}")
                    continue
        
        layer_count = len(model.layers) if model.layers else 1
        avg_trajectory_influence = total_trajectory_influence / layer_count
        avg_coverage_efficiency = total_coverage_efficiency / layer_count
        health_percentage = (healthy_splats / max(total_splats, 1)) * 100
        
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
            'active_layers': flow_stats.get('total_layers_with_flow', 0)
        }
    except Exception as e:
        logger.warning(f"Failed to get model stats: {e}")
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
            'active_layers': 0
        }
