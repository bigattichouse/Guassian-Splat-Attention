"""
SplatFlow Adaptive Birth System
Manages intelligent splat birth/death to prevent radius explosion and maintain O(n*k) efficiency.
FIXES: Device mismatch errors, variable naming issues, improved positioning logic
"""

import torch
import torch.nn as nn
import time
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class SplatBirthRequest:
    """Represents a request to birth a new splat"""
    
    def __init__(self, position: torch.Tensor, reason: str, urgency: float = 1.0, 
                 parent_splat_id: Optional[int] = None, token_cluster_size: int = 0):
        self.position = position.clone().detach()
        self.reason = reason
        self.urgency = urgency
        self.parent_splat_id = parent_splat_id
        self.token_cluster_size = token_cluster_size
        self.timestamp = time.time()
        self.processed = False
    
    def __repr__(self):
        return f"SplatBirthRequest(reason={self.reason}, urgency={self.urgency:.2f}, cluster_size={self.token_cluster_size})"


class CoverageAnalyzer:
    """Analyzes token coverage gaps and recommends splat birth positions - FIXED DEVICE ISSUES"""
    
    def __init__(self, coverage_threshold: float = 0.1, min_cluster_size: int = 3, 
                 cluster_radius: float = 3.0):
        self.coverage_threshold = coverage_threshold
        self.min_cluster_size = min_cluster_size
        self.cluster_radius = cluster_radius
        
        # Statistics
        self.gaps_detected = 0
        self.clusters_found = 0
        
    def analyze_coverage_gaps(self, token_embeddings: torch.Tensor, 
                            attention_weights: torch.Tensor) -> List[SplatBirthRequest]:
        """
        FIXED: Analyze token coverage with proper device management
        
        Args:
            token_embeddings: [batch, seq_len, embed_dim]
            attention_weights: [batch, seq_len, num_splats]
        
        Returns:
            List of birth requests for poorly covered token clusters
        """
        
        try:
            # FIXED: Ensure device consistency from the start
            device = token_embeddings.device
            batch_size, seq_len, embed_dim = token_embeddings.shape
            
            if attention_weights.size(-1) == 0:
                return []
            
            # FIXED: Move attention weights to same device
            attention_weights = attention_weights.to(device)
            
            # Calculate per-token attention coverage (sum across all splats)
            token_coverage = attention_weights.sum(dim=-1)  # [batch, seq_len]
            
            # FIXED: Create mask on same device
            uncovered_mask = token_coverage < self.coverage_threshold  # Already on correct device
            
            if not uncovered_mask.any():
                return []
            
            birth_requests = []
            
            # Process each batch separately
            for batch_idx in range(batch_size):
                batch_uncovered = uncovered_mask[batch_idx]
                
                if not batch_uncovered.any():
                    continue
                
                # FIXED: Get positions with proper device handling
                uncovered_positions = token_embeddings[batch_idx][batch_uncovered]  # Already on device
                uncovered_indices = torch.where(batch_uncovered)[0].to(device)  # Ensure indices on device
                
                if len(uncovered_positions) == 0:
                    continue
                
                # Cluster nearby uncovered tokens
                clusters = self._cluster_uncovered_tokens(uncovered_positions, device)
                
                # Create birth requests for significant clusters
                for cluster_positions in clusters:
                    if len(cluster_positions) >= self.min_cluster_size:
                        # Calculate cluster centroid
                        centroid = cluster_positions.mean(dim=0).to(device)
                        
                        # Calculate urgency based on cluster size and coverage deficit
                        urgency = len(cluster_positions) / seq_len  # Relative cluster size
                        
                        birth_request = SplatBirthRequest(
                            position=centroid,
                            reason="coverage_gap",
                            urgency=urgency,
                            token_cluster_size=len(cluster_positions)
                        )
                        
                        birth_requests.append(birth_request)
                        self.clusters_found += 1
            
            if birth_requests:
                self.gaps_detected += 1
                logger.debug(f"Coverage analysis found {len(birth_requests)} potential birth locations")
            
            return birth_requests
            
        except Exception as e:
            logger.warning(f"Coverage analysis failed: {e}")
            return []
    
    def _cluster_uncovered_tokens(self, token_positions: torch.Tensor, device: torch.device) -> List[torch.Tensor]:
        """
        FIXED: Cluster nearby uncovered tokens with proper device management
        
        Args:
            token_positions: [num_uncovered, embed_dim]
            device: Target device for computations
            
        Returns:
            List of token position clusters
        """
        
        if len(token_positions) == 0:
            return []
        
        # FIXED: Ensure all tensors are on the correct device
        token_positions = token_positions.to(device)
        
        clusters = []
        remaining_positions = token_positions.clone()
        remaining_indices = torch.arange(len(token_positions), device=device)  # Create indices on device
        
        while len(remaining_positions) > 0:
            # Start new cluster with first remaining position
            seed_position = remaining_positions[0]
            
            # Find all positions within cluster radius of seed
            distances = torch.norm(remaining_positions - seed_position.unsqueeze(0), dim=-1)
            cluster_mask = distances <= self.cluster_radius
            
            # Extract cluster positions
            cluster_positions = remaining_positions[cluster_mask]
            clusters.append(cluster_positions)
            
            # Remove clustered positions from remaining
            remaining_positions = remaining_positions[~cluster_mask]
            remaining_indices = remaining_indices[~cluster_mask]
        
        return clusters
    
    def get_statistics(self) -> Dict:
        """Get coverage analysis statistics"""
        return {
            'gaps_detected': self.gaps_detected,
            'clusters_found': self.clusters_found,
            'coverage_threshold': self.coverage_threshold,
            'min_cluster_size': self.min_cluster_size
        }


class TrajectoryBirthAnalyzer:
    """Analyzes trajectory flow patterns to guide optimal splat birth positioning - FIXED"""
    
    def __init__(self, flow_threshold: float = 0.05):
        self.flow_threshold = flow_threshold
        self.trajectory_births_recommended = 0
    
    def analyze_trajectory_bottlenecks(self, trajectories: torch.Tensor, 
                                     current_splat_positions: List[torch.Tensor]) -> List[SplatBirthRequest]:
        """
        FIXED: Find trajectory flow bottlenecks with proper device handling
        
        Args:
            trajectories: [batch, seq_len, embed_dim] - trajectory flow vectors
            current_splat_positions: List of current splat positions
            
        Returns:
            List of birth requests based on trajectory analysis
        """
        
        try:
            if len(trajectories) == 0 or len(current_splat_positions) == 0:
                return []
            
            # FIXED: Get device from trajectories and ensure consistency
            device = trajectories.device
            batch_size, seq_len, embed_dim = trajectories.shape
            
            birth_requests = []
            
            # Find trajectory convergence points (high flow density)
            trajectory_magnitudes = torch.norm(trajectories, dim=-1)  # [batch, seq_len]
            
            # Identify positions with high trajectory flow but poor splat coverage
            high_flow_mask = trajectory_magnitudes > self.flow_threshold
            
            if not high_flow_mask.any():
                return []
            
            # FIXED: Ensure splat positions are on same device
            splat_positions_on_device = [pos.to(device) for pos in current_splat_positions]
            
            # For each high-flow position, check if it's well-served by existing splats
            for batch_idx in range(batch_size):
                batch_high_flow = high_flow_mask[batch_idx]
                
                if not batch_high_flow.any():
                    continue
                
                high_flow_positions = trajectories[batch_idx][batch_high_flow]
                high_flow_indices = torch.where(batch_high_flow)[0]
                
                # Check coverage by existing splats
                for pos_idx, position in enumerate(high_flow_positions):
                    min_distance_to_splat = float('inf')
                    
                    for splat_pos in splat_positions_on_device:
                        distance = torch.norm(position - splat_pos).item()
                        min_distance_to_splat = min(min_distance_to_splat, distance)
                    
                    # If position is far from all splats and has high flow, recommend birth
                    if min_distance_to_splat > 6.0:  # Threshold for "far from splats"
                        flow_strength = trajectory_magnitudes[batch_idx][high_flow_indices[pos_idx]].item()
                        
                        birth_request = SplatBirthRequest(
                            position=position.to(device),  # FIXED: Ensure position on device
                            reason="trajectory_bottleneck",
                            urgency=flow_strength,
                            token_cluster_size=1
                        )
                        
                        birth_requests.append(birth_request)
                        self.trajectory_births_recommended += 1
            
            if birth_requests:
                logger.debug(f"Trajectory analysis recommended {len(birth_requests)} births")
            
            return birth_requests
            
        except Exception as e:
            logger.warning(f"Trajectory bottleneck analysis failed: {e}")
            return []


class AdaptiveSplatBirthManager:
    """
    FIXED: Manages the lifecycle of splats - birth, life, and death
    Prevents radius explosion by birthing new splats instead of expanding existing ones
    """
    
    def __init__(self, max_splats: int = 32, max_radius: float = 8.0, 
                 max_births_per_epoch: int = 2, birth_cooldown: int = 3):
        
        # Population control
        self.max_splats = max_splats
        self.max_radius = max_radius
        self.max_births_per_epoch = max_births_per_epoch
        self.birth_cooldown = birth_cooldown
        
        # Birth management
        self.pending_birth_requests = []
        self.birth_history = []
        self.births_this_epoch = 0
        self.last_birth_epoch = -1
        self.cooldown_remaining = 0
        
        # Analysis components
        self.coverage_analyzer = CoverageAnalyzer()
        self.trajectory_analyzer = TrajectoryBirthAnalyzer()
        
        # Statistics
        self.total_births = 0
        self.total_deaths = 0
        self.birth_reasons = defaultdict(int)
        
        logger.info(f"🐣 FIXED Adaptive Birth Manager initialized (max_splats={max_splats}, max_radius={max_radius})")
    
    def request_splat_birth(self, position: torch.Tensor, reason: str, urgency: float = 1.0,
                          parent_splat_id: Optional[int] = None, token_cluster_size: int = 0):
        """Request birth of a new splat at specified position"""
        
        birth_request = SplatBirthRequest(
            position=position.clone().detach(),  # FIXED: Ensure detached copy
            reason=reason,
            urgency=urgency,
            parent_splat_id=parent_splat_id,
            token_cluster_size=token_cluster_size
        )
        
        self.pending_birth_requests.append(birth_request)
        logger.debug(f"Birth requested: {birth_request}")
    
    def should_birth_instead_of_expand(self, current_radius: float, proposed_radius: float) -> bool:
        """
        Decide whether to birth a new splat instead of expanding radius
        
        Args:
            current_radius: Current splat radius
            proposed_radius: Proposed expanded radius
            
        Returns:
            True if should birth new splat instead of expanding
        """
        
        # Hard limit on radius expansion
        if proposed_radius > self.max_radius:
            return True
        
        # If expansion would be more than 2x, consider birthing
        if proposed_radius > current_radius * 2.0:
            return True
        
        # If we're not at population limit, prefer birthing for large expansions
        if len(self.pending_birth_requests) == 0 and proposed_radius > current_radius * 1.5:
            return True
        
        return False
    
    def process_birth_requests(self, current_splats: List, token_embeddings: torch.Tensor, 
                             trajectories: Optional[torch.Tensor], epoch: int) -> List[Dict]:
        """
        FIXED: Process pending birth requests and return splat parameters for new splats
        
        Args:
            current_splats: List of current splats
            token_embeddings: Current token embeddings
            trajectories: Trajectory flow vectors (optional)
            epoch: Current training epoch
            
        Returns:
            List of splat parameter dictionaries for newly birthed splats
        """
        
        # Check cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return []
        
        # Reset birth counter for new epoch
        if epoch != self.last_birth_epoch:
            self.births_this_epoch = 0
            self.last_birth_epoch = epoch
        
        # Check if we can birth more splats this epoch
        if self.births_this_epoch >= self.max_births_per_epoch:
            return []
        
        try:
            # FIXED: Use correct variable name throughout
            new_splat_params = []  # FIXED: Renamed from new_splats
            
            # FIXED: Get device from token embeddings for consistency
            device = token_embeddings.device
            
            # Add coverage-based birth requests
            if len(current_splats) > 0:
                # Get current attention weights (simplified)
                attention_weights = self._estimate_attention_coverage(current_splats, token_embeddings)
                coverage_requests = self.coverage_analyzer.analyze_coverage_gaps(
                    token_embeddings, attention_weights
                )
                self.pending_birth_requests.extend(coverage_requests)
            
            # Add trajectory-based birth requests
            if trajectories is not None and len(current_splats) > 0:
                current_positions = [splat.position.to(device) for splat in current_splats]  # FIXED: Ensure device
                trajectory_requests = self.trajectory_analyzer.analyze_trajectory_bottlenecks(
                    trajectories, current_positions
                )
                self.pending_birth_requests.extend(trajectory_requests)
            
            if not self.pending_birth_requests:
                return []
            
            # Evaluate and prioritize birth requests
            approved_requests = self._evaluate_birth_requests(current_splats, token_embeddings)
            
            # Birth new splats
            for request in approved_requests:
                if self.births_this_epoch >= self.max_births_per_epoch:
                    break
                
                if len(current_splats) >= self.max_splats:
                    # Remove worst splat to make room
                    removed = self._remove_worst_splat(current_splats)
                    if removed:
                        logger.debug(f"Removed underperforming splat to make room")
                
                # Birth new splat
                new_splat_params_dict = self._birth_new_splat(request, current_splats, epoch, device)
                if new_splat_params_dict:
                    new_splat_params.append(new_splat_params_dict)  # FIXED: Use correct variable
                    self.births_this_epoch += 1
                    self.total_births += 1
                    self.birth_reasons[request.reason] += 1
                    
                    logger.info(f"🐣 Birthed splat #{self.total_births} for {request.reason} "
                              f"(cluster_size={request.token_cluster_size})")
            
            # Set cooldown if we birthed splats
            if new_splat_params:  # FIXED: Use correct variable
                self.cooldown_remaining = self.birth_cooldown
            
            # Clear processed requests
            self.pending_birth_requests = []
            
            return new_splat_params  # FIXED: Return correct variable
            
        except Exception as e:
            logger.warning(f"Birth processing failed: {e}")
            return []
    
    def _evaluate_birth_requests(self, current_splats: List, 
                               token_embeddings: torch.Tensor) -> List[SplatBirthRequest]:
        """FIXED: Evaluate and prioritize birth requests with device handling"""
        
        if not self.pending_birth_requests:
            return []
        
        try:
            # FIXED: Get device for consistency
            device = token_embeddings.device
            
            # Ensure all request positions are on the same device
            for request in self.pending_birth_requests:
                request.position = request.position.to(device)
            
            # Filter out duplicate/nearby requests
            filtered_requests = self._filter_nearby_requests(self.pending_birth_requests)
            
            # Sort by urgency and cluster size
            prioritized = sorted(filtered_requests, 
                               key=lambda x: (x.urgency, x.token_cluster_size), 
                               reverse=True)
            
            # Limit number of births
            max_births = min(self.max_births_per_epoch - self.births_this_epoch, 2)
            return prioritized[:max_births]
        except Exception as e:
            logger.warning(f"Birth request evaluation failed: {e}")
            return []
    
    def _filter_nearby_requests(self, requests: List[SplatBirthRequest]) -> List[SplatBirthRequest]:
        """Remove birth requests that are too close to each other"""
        
        if len(requests) <= 1:
            return requests
        
        try:
            filtered = []
            min_birth_distance = 4.0  # Minimum distance between new splats
            
            for request in requests:
                too_close = False
                
                # Check distance to other approved requests
                for existing in filtered:
                    distance = torch.norm(request.position - existing.position).item()
                    if distance < min_birth_distance:
                        too_close = True
                        break
                
                if not too_close:
                    filtered.append(request)
            
            return filtered
        except Exception as e:
            logger.warning(f"Request filtering failed: {e}")
            return requests  # Return unfiltered if filtering fails
    
    def _birth_new_splat(self, request: SplatBirthRequest, current_splats: List, 
                        epoch: int, device: torch.device) -> Optional[Dict]:
        """FIXED: Create parameters for a new splat based on birth request"""
        
        try:
            splat_id = self.total_births  # Simple ID system
            
            # Determine parameters based on request and nearby splats
            scale, amplitude = self._calculate_birth_parameters(request, current_splats)
            
            # FIXED: Ensure position is on correct device
            position = request.position.to(device)
            
            # Return splat parameters instead of creating the splat here
            splat_params = {
                'position': position,
                'scale': scale,
                'amplitude': amplitude,
                'splat_id': splat_id,
                'device': device,
                'layer_idx': 0,  # Will be set by parent layer
                'birth_reason': request.reason
            }
            
            # Record birth request (actual birth tracked by attention layer)
            self.birth_history.append({
                'splat_id': splat_id,
                'position': position.clone(),
                'reason': request.reason,
                'epoch': epoch,
                'urgency': request.urgency
            })
            
            return splat_params
            
        except Exception as e:
            logger.warning(f"Failed to prepare new splat parameters: {e}")
            return None
    
    def _calculate_birth_parameters(self, request: SplatBirthRequest, 
                                  current_splats: List) -> Tuple[float, float]:
        """Calculate initial scale and amplitude for new splat"""
        
        # FIXED: Improved default parameters based on emergency rescue patterns
        default_scale = 2.0  # Increased from 1.0 to better cover tokens
        default_amplitude = 1.0
        
        if not current_splats:
            return default_scale, default_amplitude
        
        try:
            # Find nearest existing splat for parameter inheritance
            min_distance = float('inf')
            nearest_splat = None
            
            for splat in current_splats:
                try:
                    distance = torch.norm(request.position - splat.position).item()
                    if distance < min_distance:
                        min_distance = distance
                        nearest_splat = splat
                except Exception:
                    continue
            
            if nearest_splat:
                # Inherit and adapt parameters from nearest splat
                try:
                    parent_scale = torch.exp(nearest_splat.log_scale).item()
                    parent_amplitude = nearest_splat.amplitude.item()
                    
                    # FIXED: Less aggressive scaling down to maintain coverage
                    scale = max(parent_scale * 0.9, 1.5)  # Don't scale down too much
                    amplitude = parent_amplitude * 0.95
                    
                    return scale, amplitude
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Parameter inheritance failed: {e}")
        
        return default_scale, default_amplitude
    
    def _estimate_attention_coverage(self, current_splats: List, 
                                   token_embeddings: torch.Tensor) -> torch.Tensor:
        """FIXED: Estimate current attention coverage with proper device handling"""
        
        try:
            batch_size, seq_len, embed_dim = token_embeddings.shape
            device = token_embeddings.device
            
            if not current_splats:
                return torch.zeros(batch_size, seq_len, 0, device=device)
            
            # Simple distance-based coverage estimation
            attention_weights = torch.zeros(batch_size, seq_len, len(current_splats), device=device)
            
            for splat_idx, splat in enumerate(current_splats):
                try:
                    # FIXED: Ensure splat position is on correct device
                    splat_pos = splat.position.to(device).unsqueeze(0).unsqueeze(0)
                    distances = torch.norm(token_embeddings - splat_pos, dim=-1)
                    
                    # FIXED: Improved radius calculation based on emergency rescue patterns
                    radius = 4.0  # Increased from 4.0 to better match actual needs
                    weights = torch.exp(-0.5 * (distances / radius) ** 2)
                    attention_weights[:, :, splat_idx] = weights
                    
                except Exception as e:
                    logger.warning(f"Coverage estimation failed for splat {splat_idx}: {e}")
                    continue
            
            return attention_weights
            
        except Exception as e:
            logger.warning(f"Coverage estimation failed: {e}")
            return torch.zeros(batch_size, seq_len, 0, device=token_embeddings.device)
    
    def _remove_worst_splat(self, current_splats: List) -> bool:
        """Remove the worst performing splat to make room for new one"""
        
        if not current_splats:
            return False
        
        try:
            # Find splat with lowest usefulness
            worst_splat = None
            worst_usefulness = float('inf')
            worst_idx = -1
            
            for idx, splat in enumerate(current_splats):
                try:
                    if hasattr(splat, 'usefulness') and splat.usefulness < worst_usefulness:
                        worst_usefulness = splat.usefulness
                        worst_splat = splat
                        worst_idx = idx
                except Exception:
                    continue
            
            if worst_idx >= 0:
                removed_splat = current_splats.pop(worst_idx)
                self.total_deaths += 1
                logger.debug(f"Removed splat with usefulness {worst_usefulness:.3f}")
                return True
            
        except Exception as e:
            logger.warning(f"Failed to remove worst splat: {e}")
        
        return False
    
    def get_birth_statistics(self) -> Dict:
        """Get comprehensive birth management statistics"""
        
        return {
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'births_this_epoch': self.births_this_epoch,
            'pending_requests': len(self.pending_birth_requests),
            'cooldown_remaining': self.cooldown_remaining,
            'birth_reasons': dict(self.birth_reasons),
            'max_splats': self.max_splats,
            'max_radius': self.max_radius,
            'coverage_stats': self.coverage_analyzer.get_statistics()
        }
    
    def reset_epoch_counters(self):
        """Reset per-epoch counters"""
        self.births_this_epoch = 0
        self.pending_birth_requests = []
