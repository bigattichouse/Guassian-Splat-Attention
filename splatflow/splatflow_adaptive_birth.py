"""
SplatFlow Adaptive Birth System - O(n*k) Optimized
Manages intelligent splat birth/death to prevent radius explosion and maintain O(n*k) efficiency.
Uses smart sampling and splat-guided algorithms throughout to avoid O(n²) bottlenecks.
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


class OnkCoverageAnalyzer:
    """O(n*k) coverage analyzer - uses smart sampling instead of O(n²) token analysis"""
    
    def __init__(self, coverage_threshold: float = 0.1, min_cluster_size: int = 5, 
                 cluster_radius: float = 3.0, sample_ratio: float = 0.2):
        self.coverage_threshold = coverage_threshold
        self.min_cluster_size = min_cluster_size
        self.cluster_radius = cluster_radius
        self.sample_ratio = sample_ratio  # Sample subset for O(n*k) efficiency
        
        # Statistics
        self.gaps_detected = 0
        self.clusters_found = 0
        
    def analyze_coverage_gaps(self, token_embeddings: torch.Tensor, 
                            attention_weights: torch.Tensor, 
                            current_splats: List) -> List[SplatBirthRequest]:
        """
        O(n*k) coverage analysis using splat-based sampling
        
        Key optimization: Instead of O(n²) token-token analysis,
        use O(n*k) token-splat analysis with strategic sampling
        """
        
        try:
            device = token_embeddings.device
            batch_size, seq_len, embed_dim = token_embeddings.shape
            
            if attention_weights.size(-1) == 0 or len(current_splats) == 0:
                return []
            
            attention_weights = attention_weights.to(device)
            
            # OPTIMIZATION 1: Work with single batch to avoid batch overhead
            batch_idx = 0
            batch_tokens = token_embeddings[batch_idx]  # [seq_len, embed_dim]
            batch_attention = attention_weights[batch_idx]  # [seq_len, num_splats]
            
            # OPTIMIZATION 2: Sample tokens instead of analyzing all O(n*k) → O(sample_size * k)
            sample_size = max(32, int(seq_len * self.sample_ratio))
            sample_indices = torch.randperm(seq_len, device=device)[:sample_size]
            
            sampled_tokens = batch_tokens[sample_indices]
            sampled_attention = batch_attention[sample_indices]
            
            # OPTIMIZATION 3: Fast coverage calculation using existing attention weights
            token_coverage = sampled_attention.sum(dim=-1)  # Sum across splats
            uncovered_mask = token_coverage < self.coverage_threshold
            
            if not uncovered_mask.any():
                return []
            
            uncovered_tokens = sampled_tokens[uncovered_mask]
            uncovered_indices = sample_indices[uncovered_mask]
            
            if len(uncovered_tokens) < self.min_cluster_size:
                return []
            
            # OPTIMIZATION 4: O(k) splat-guided clustering instead of O(n²)
            birth_requests = self._onk_cluster_analysis(
                uncovered_tokens, uncovered_indices, current_splats, device
            )
            
            if birth_requests:
                self.gaps_detected += 1
                self.clusters_found += len(birth_requests)
                
                # Reduced logging frequency
                if self.gaps_detected % 20 == 1:  # Only every 20th detection
                    logger.debug(f"O(n*k) coverage analysis: {len(birth_requests)} birth candidates")
            
            return birth_requests
            
        except Exception as e:
            logger.warning(f"O(n*k) coverage analysis failed: {e}")
            return []
    
    def _onk_cluster_analysis(self, uncovered_tokens: torch.Tensor, 
                            uncovered_indices: torch.Tensor,
                            current_splats: List, device: torch.device) -> List[SplatBirthRequest]:
        """
        O(k) splat-guided clustering instead of O(n²) token clustering
        
        Strategy: Use existing splat positions as cluster seeds,
        then refine based on uncovered tokens
        """
        
        birth_requests = []
        
        try:
            if len(uncovered_tokens) == 0:
                return []
            
            if len(current_splats) == 0:
                # Fallback: simple clustering when no splats available
                if len(uncovered_tokens) >= self.min_cluster_size:
                    centroid = uncovered_tokens.mean(dim=0).to(device)
                    urgency = len(uncovered_tokens) / 100  # Rough estimate
                    
                    birth_request = SplatBirthRequest(
                        position=centroid,
                        reason="onk_coverage_gap_fallback",
                        urgency=urgency,
                        token_cluster_size=len(uncovered_tokens)
                    )
                    birth_requests.append(birth_request)
                
                return birth_requests
            
            # Get splat positions as clustering guides
            splat_positions = torch.stack([splat.position.detach() for splat in current_splats]).to(device)
            
            # Find coverage gaps relative to existing splats - O(uncovered_tokens * k)
            distances_to_splats = torch.cdist(
                uncovered_tokens.unsqueeze(0), 
                splat_positions.unsqueeze(0)
            ).squeeze(0)  # [num_uncovered, num_splats]
            
            # Find tokens that are far from ALL splats (true coverage gaps)
            min_distances, closest_splat_idx = torch.min(distances_to_splats, dim=1)
            gap_threshold = 6.0  # Distance threshold for "coverage gap"
            
            true_gaps = min_distances > gap_threshold
            if not true_gaps.any():
                return []
            
            gap_tokens = uncovered_tokens[true_gaps]
            gap_distances = min_distances[true_gaps]
            
            # Group gaps by proximity - O(gap_tokens * k) not O(gap_tokens²)
            gap_clusters = self._splat_guided_grouping(gap_tokens, gap_distances, splat_positions)
            
            # Create birth requests for significant clusters
            for cluster_center, cluster_size, urgency in gap_clusters:
                if cluster_size >= self.min_cluster_size:
                    birth_request = SplatBirthRequest(
                        position=cluster_center.to(device),
                        reason="onk_coverage_gap",
                        urgency=urgency,
                        token_cluster_size=cluster_size
                    )
                    birth_requests.append(birth_request)
            
            return birth_requests
            
        except Exception as e:
            logger.warning(f"O(n*k) cluster analysis failed: {e}")
            return []
    
    def _splat_guided_grouping(self, gap_tokens: torch.Tensor, gap_distances: torch.Tensor,
                             splat_positions: torch.Tensor) -> List[Tuple]:
        """
        Group gap tokens using splat positions as guides - O(k) not O(n²)
        """
        
        clusters = []
        
        try:
            if len(gap_tokens) == 0:
                return clusters
            
            # Strategy: For each major gap region, find centroid
            # Use distance-based weighting to avoid O(n²) clustering
            
            # Find tokens that are very far from existing splats
            very_far_mask = gap_distances > 8.0
            if very_far_mask.any():
                far_tokens = gap_tokens[very_far_mask]
                
                # Simple centroid-based clustering
                if len(far_tokens) >= self.min_cluster_size:
                    cluster_center = far_tokens.mean(dim=0)
                    urgency = len(far_tokens) / len(gap_tokens)  # Proportion of gaps
                    clusters.append((cluster_center, len(far_tokens), urgency))
            
            # Find moderately distant tokens for secondary clusters
            moderate_mask = (gap_distances > 6.0) & (gap_distances <= 8.0)
            if moderate_mask.any() and moderate_mask.sum().item() >= self.min_cluster_size:
                moderate_tokens = gap_tokens[moderate_mask]
                cluster_center = moderate_tokens.mean(dim=0)
                urgency = 0.5  # Lower urgency for moderate gaps
                clusters.append((cluster_center, len(moderate_tokens), urgency))
            
            return clusters
            
        except Exception as e:
            logger.warning(f"Splat-guided grouping failed: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get coverage analysis statistics"""
        return {
            'gaps_detected': self.gaps_detected,
            'clusters_found': self.clusters_found,
            'coverage_threshold': self.coverage_threshold,
            'min_cluster_size': self.min_cluster_size,
            'optimization': 'O(n*k) algorithms'
        }


class OnkTrajectoryAnalyzer:
    """O(n*k) trajectory analysis instead of O(n²) analysis"""
    
    def __init__(self, flow_threshold: float = 0.05, sample_ratio: float = 0.3):
        self.flow_threshold = flow_threshold
        self.sample_ratio = sample_ratio
        self.trajectory_births_recommended = 0
    
    def analyze_trajectory_bottlenecks(self, trajectories: torch.Tensor, 
                                     current_splat_positions: List[torch.Tensor]) -> List[SplatBirthRequest]:
        """
        O(n*k) trajectory bottleneck analysis using smart sampling
        
        Key optimization: Sample high-flow regions instead of analyzing all tokens
        """
        
        try:
            if len(trajectories) == 0 or len(current_splat_positions) == 0:
                return []
            
            device = trajectories.device
            batch_size, seq_len, embed_dim = trajectories.shape
            
            birth_requests = []
            
            # OPTIMIZATION 1: Sample high-flow tokens instead of all tokens
            trajectory_magnitudes = torch.norm(trajectories, dim=-1)  # [batch, seq_len]
            
            # Work with first batch
            batch_magnitudes = trajectory_magnitudes[0]
            batch_trajectories = trajectories[0]
            
            # Find high-flow tokens efficiently
            high_flow_threshold = max(self.flow_threshold, batch_magnitudes.quantile(0.8))
            high_flow_mask = batch_magnitudes > high_flow_threshold
            
            if not high_flow_mask.any():
                return []
            
            # OPTIMIZATION 2: Sample subset of high-flow tokens
            high_flow_indices = torch.where(high_flow_mask)[0]
            sample_size = max(16, int(len(high_flow_indices) * self.sample_ratio))
            
            if len(high_flow_indices) > sample_size:
                # Sample highest-flow tokens
                flow_values = batch_magnitudes[high_flow_indices]
                _, top_indices = torch.topk(flow_values, sample_size)
                sampled_indices = high_flow_indices[top_indices]
            else:
                sampled_indices = high_flow_indices
            
            sampled_positions = batch_trajectories[sampled_indices]
            
            # OPTIMIZATION 3: O(sample_size * k) distance computation
            splat_positions_tensor = torch.stack(current_splat_positions).to(device)
            
            # Compute distances only for sampled high-flow tokens
            distances_to_splats = torch.cdist(
                sampled_positions.unsqueeze(0),
                splat_positions_tensor.unsqueeze(0)
            ).squeeze(0)  # [sample_size, num_splats]
            
            min_distances, _ = torch.min(distances_to_splats, dim=1)
            
            # Find positions far from all splats
            far_threshold = 8.0
            far_mask = min_distances > far_threshold
            
            if far_mask.any():
                far_positions = sampled_positions[far_mask]
                far_flow_magnitudes = batch_magnitudes[sampled_indices[far_mask]]
                
                # Group nearby far positions using simple centroid approach
                if len(far_positions) >= 2:
                    # Simple centroid approach to avoid O(n²)
                    centroid = far_positions.mean(dim=0)
                    avg_flow_strength = far_flow_magnitudes.mean().item()
                    
                    birth_request = SplatBirthRequest(
                        position=centroid.to(device),
                        reason="onk_trajectory_bottleneck",
                        urgency=avg_flow_strength,
                        token_cluster_size=len(far_positions)
                    )
                    
                    birth_requests.append(birth_request)
                    self.trajectory_births_recommended += 1
            
            return birth_requests
            
        except Exception as e:
            logger.warning(f"O(n*k) trajectory analysis failed: {e}")
            return []


class OnkSplatRepositioner:
    """O(n*k) splat repositioning instead of O(n²)"""
    
    def __init__(self, sample_size: int = 64):
        self.sample_size = sample_size
    
    def progressive_reposition_onk(self, splats: List, token_embeddings: torch.Tensor) -> int:
        """O(n*k) progressive repositioning using sampling"""
        
        repositioned_count = 0
        
        try:
            device = token_embeddings.device
            batch_size, seq_len, embed_dim = token_embeddings.shape
            
            if seq_len == 0 or not splats:
                return 0
            
            # Sample tokens for analysis
            sample_size = min(self.sample_size, seq_len)
            sample_indices = torch.randperm(seq_len, device=device)[:sample_size]
            sampled_tokens = token_embeddings[0][sample_indices]  # First batch
            
            for splat in splats:
                try:
                    # Find closest sampled tokens to this splat
                    distances = torch.norm(sampled_tokens - splat.position.unsqueeze(0), dim=-1)
                    
                    # Get top-k closest tokens for better positioning
                    k_closest = min(8, len(distances))
                    _, closest_indices = torch.topk(distances, k_closest, largest=False)
                    closest_tokens = sampled_tokens[closest_indices]
                    
                    # Move toward centroid of closest tokens
                    target_position = closest_tokens.mean(dim=0)
                    direction = target_position - splat.position
                    
                    # Conservative movement
                    move_strength = 0.05
                    
                    with torch.no_grad():
                        splat.position.data += move_strength * direction
                        repositioned_count += 1
                
                except Exception as e:
                    logger.warning(f"Failed to reposition splat {splat.id}: {e}")
                    continue
            
            return repositioned_count
            
        except Exception as e:
            logger.warning(f"Progressive repositioning failed: {e}")
            return 0
    
    def emergency_reposition_onk(self, splats: List, token_embeddings: torch.Tensor) -> int:
        """O(k) emergency repositioning using smart sampling"""
        
        repositioned_count = 0
        
        try:
            device = token_embeddings.device
            batch_size, seq_len, embed_dim = token_embeddings.shape
            
            if seq_len == 0:
                return 0
            
            # Strategic sampling: mix of random + evenly spaced
            sample_size = min(self.sample_size, seq_len)
            if seq_len > sample_size:
                random_indices = torch.randperm(seq_len, device=device)[:sample_size//2]
                spaced_indices = torch.linspace(0, seq_len-1, sample_size//2, device=device).long()
                sample_indices = torch.cat([random_indices, spaced_indices]).unique()
            else:
                sample_indices = torch.arange(seq_len, device=device)
            
            sampled_tokens = token_embeddings[0][sample_indices]
            
            # Only reposition truly problematic splats
            for splat in splats:
                try:
                    if hasattr(splat, 'usefulness') and splat.usefulness < 0.1:
                        # Find closest token in sample
                        distances = torch.norm(sampled_tokens - splat.position.unsqueeze(0), dim=-1)
                        closest_idx = torch.argmin(distances)
                        closest_token = sampled_tokens[closest_idx]
                        
                        # Move splat toward closest token (conservative movement)
                        direction = closest_token - splat.position
                        move_strength = 0.2
                        
                        with torch.no_grad():
                            splat.position.data += move_strength * direction
                            
                            # Apply bounds
                            bounds = 15.0
                            splat.position.data = torch.clamp(splat.position.data, -bounds, bounds)
                        
                        repositioned_count += 1
                        
                except Exception as e:
                    logger.warning(f"Emergency repositioning failed for splat {splat.id}: {e}")
                    continue
            
            return repositioned_count
            
        except Exception as e:
            logger.warning(f"Emergency repositioning failed: {e}")
            return 0


class AdaptiveSplatBirthManager:
    """
    O(n*k) Birth manager - prevents radius explosion by birthing new splats
    Uses O(n*k) algorithms throughout to avoid O(n²) bottlenecks
    """
    
    def __init__(self, max_splats: int = 32, max_radius: float = 8.0, 
                 max_births_per_epoch: int = 2, birth_cooldown: int = 3):
        
        # Population control
        self.max_splats = max_splats
        self.max_radius = max_radius
        self.max_births_per_epoch = max_births_per_epoch
        self.birth_cooldown = birth_cooldown
        
        # O(n*k) analyzers
        self.coverage_analyzer = OnkCoverageAnalyzer()
        self.trajectory_analyzer = OnkTrajectoryAnalyzer()
        self.repositioner = OnkSplatRepositioner()
        
        # Birth management
        self.pending_birth_requests = []
        self.birth_history = []
        self.births_this_epoch = 0
        self.last_birth_epoch = -1
        self.cooldown_remaining = 0
        
        # Statistics
        self.total_births = 0
        self.total_deaths = 0
        self.birth_reasons = defaultdict(int)
        
        logger.info(f"🚀 O(n*k) Adaptive Birth Manager initialized")
    
    def request_splat_birth(self, position: torch.Tensor, reason: str, urgency: float = 1.0,
                          parent_splat_id: Optional[int] = None, token_cluster_size: int = 0):
        """Request birth of a new splat at specified position"""
        
        birth_request = SplatBirthRequest(
            position=position.clone().detach(),
            reason=reason,
            urgency=urgency,
            parent_splat_id=parent_splat_id,
            token_cluster_size=token_cluster_size
        )
        
        self.pending_birth_requests.append(birth_request)
        logger.debug(f"O(n*k) Birth requested: {birth_request}")
    
    def should_birth_instead_of_expand(self, current_radius: float, proposed_radius: float) -> bool:
        """Decide whether to birth a new splat instead of expanding radius"""
        
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
        """Process birth requests using O(n*k) analysis"""
        
        # Check cooldown and limits
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return []
        
        if epoch != self.last_birth_epoch:
            self.births_this_epoch = 0
            self.last_birth_epoch = epoch
        
        if self.births_this_epoch >= self.max_births_per_epoch:
            return []
        
        try:
            new_splat_params = []
            device = token_embeddings.device
            
            # O(n*k) coverage analysis
            if len(current_splats) > 0:
                attention_weights = self._estimate_attention_coverage_onk(current_splats, token_embeddings)
                coverage_requests = self.coverage_analyzer.analyze_coverage_gaps(
                    token_embeddings, attention_weights, current_splats
                )
                self.pending_birth_requests.extend(coverage_requests)
            
            # O(n*k) trajectory analysis
            if trajectories is not None and len(current_splats) > 0:
                current_positions = [splat.position.to(device) for splat in current_splats]
                trajectory_requests = self.trajectory_analyzer.analyze_trajectory_bottlenecks(
                    trajectories, current_positions
                )
                self.pending_birth_requests.extend(trajectory_requests)
            
            if not self.pending_birth_requests:
                return []
            
            # Process approved requests
            approved_requests = self._evaluate_birth_requests(current_splats, token_embeddings)
            
            for request in approved_requests:
                if self.births_this_epoch >= self.max_births_per_epoch:
                    break
                
                if len(current_splats) >= self.max_splats:
                    removed = self._remove_worst_splat(current_splats)
                    if removed:
                        logger.debug(f"Removed underperforming splat to make room")
                
                new_splat_params_dict = self._birth_new_splat(request, current_splats, epoch, device)
                if new_splat_params_dict:
                    new_splat_params.append(new_splat_params_dict)
                    self.births_this_epoch += 1
                    self.total_births += 1
                    self.birth_reasons[request.reason] += 1
                    
                    # Reduced logging frequency
                    if self.total_births % 5 == 1:  # Every 5th birth
                        logger.info(f"🚀 O(n*k) Birth #{self.total_births}: {request.reason} "
                                  f"(cluster_size={request.token_cluster_size})")
            
            if new_splat_params:
                self.cooldown_remaining = self.birth_cooldown
            
            self.pending_birth_requests = []
            return new_splat_params
            
        except Exception as e:
            logger.warning(f"O(n*k) birth processing failed: {e}")
            return []
    
    def _estimate_attention_coverage_onk(self, current_splats: List, 
                                       token_embeddings: torch.Tensor) -> torch.Tensor:
        """O(n*k) attention coverage estimation using sampling"""
        
        try:
            batch_size, seq_len, embed_dim = token_embeddings.shape
            device = token_embeddings.device
            
            if not current_splats:
                return torch.zeros(batch_size, seq_len, 0, device=device)
            
            # O(n*k) OPTIMIZATION: Sample tokens for efficient coverage estimation
            sample_size = min(128, seq_len)  # Limit sample size
            sample_indices = torch.randperm(seq_len, device=device)[:sample_size]
            sampled_tokens = token_embeddings[:, sample_indices, :]  # [batch, sample_size, embed_dim]
            
            attention_weights = torch.zeros(batch_size, sample_size, len(current_splats), device=device)
            
            for splat_idx, splat in enumerate(current_splats):
                try:
                    splat_pos = splat.position.to(device).unsqueeze(0).unsqueeze(0)
                    distances = torch.norm(sampled_tokens - splat_pos, dim=-1)
                    
                    radius = 4.0
                    weights = torch.exp(-0.5 * (distances / radius) ** 2)
                    attention_weights[:, :, splat_idx] = weights
                    
                except Exception:
                    continue
            
            # Expand back to full sequence (approximate using nearest neighbor)
            full_attention = torch.zeros(batch_size, seq_len, len(current_splats), device=device)
            full_attention[:, sample_indices, :] = attention_weights
            
            # Simple interpolation for non-sampled positions
            if sample_size < seq_len:
                for i in range(seq_len):
                    if i not in sample_indices:
                        nearest_idx = sample_indices[torch.argmin(torch.abs(sample_indices - i))]
                        full_attention[:, i, :] = full_attention[:, nearest_idx, :]
            
            return full_attention
            
        except Exception as e:
            logger.warning(f"O(n*k) coverage estimation failed: {e}")
            return torch.zeros(batch_size, seq_len, 0, device=token_embeddings.device)
    
    def apply_progressive_repositioning_onk(self, splats: List, token_embeddings: torch.Tensor) -> int:
        """Apply O(n*k) progressive repositioning"""
        return self.repositioner.progressive_reposition_onk(splats, token_embeddings)
    
    def apply_emergency_repositioning_onk(self, splats: List, token_embeddings: torch.Tensor) -> int:
        """Apply O(n*k) emergency repositioning"""
        return self.repositioner.emergency_reposition_onk(splats, token_embeddings)
    
    def _evaluate_birth_requests(self, current_splats: List, token_embeddings: torch.Tensor):
        """Evaluate and prioritize birth requests"""
        
        if not self.pending_birth_requests:
            return []
        
        try:
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
    
    def _filter_nearby_requests(self, requests: List) -> List:
        """Remove birth requests that are too close to each other"""
        
        if len(requests) <= 1:
            return requests
        
        try:
            filtered = []
            min_birth_distance = 4.0
            
            for request in requests:
                too_close = False
                
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
            return requests
    
    def _birth_new_splat(self, request, current_splats: List, epoch: int, device: torch.device):
        """Create parameters for a new splat based on birth request"""
        
        try:
            splat_id = self.total_births
            
            # Determine parameters based on request and nearby splats
            scale, amplitude = self._calculate_birth_parameters(request, current_splats)
            
            position = request.position.to(device)
            
            # Return splat parameters
            splat_params = {
                'position': position,
                'scale': scale,
                'amplitude': amplitude,
                'splat_id': splat_id,
                'device': device,
                'layer_idx': 0,  # Will be set by parent layer
                'birth_reason': request.reason
            }
            
            # Record birth request
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
    
    def _calculate_birth_parameters(self, request, current_splats: List) -> Tuple[float, float]:
        """Calculate initial scale and amplitude for new splat"""
        
        default_scale = 2.0
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
                try:
                    parent_scale = torch.exp(nearest_splat.log_scale).item()
                    parent_amplitude = nearest_splat.amplitude.item()
                    
                    scale = max(parent_scale * 0.9, 1.5)
                    amplitude = parent_amplitude * 0.95
                    
                    return scale, amplitude
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Parameter inheritance failed: {e}")
        
        return default_scale, default_amplitude
    
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
            'optimization': 'O(n*k) algorithms active'
        }
    
    def reset_epoch_counters(self):
        """Reset per-epoch counters"""
        self.births_this_epoch = 0
        self.pending_birth_requests = []
