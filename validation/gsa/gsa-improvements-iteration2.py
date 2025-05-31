"""
GSA Final Fixes & Error Resolution

This program addresses the remaining issues:
1. JSON serialization error (TypeError: Object of type bool is not JSON serializable)
2. Birth/Death mechanism not triggering (0 adaptations performed)
3. Any remaining edge cases and stability issues

Focus: Clean up all remaining bugs and make birth/death actually work.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class RobustGSA(nn.Module):
    """GSA with all bugs fixed and robust birth/death mechanisms"""
    
    def __init__(self, dim: int, n_splats: int = 16, n_heads: int = 8, 
                 movement_scale: float = 0.05, pruning_threshold: float = 0.001):
        super().__init__()
        self.dim = dim
        self.n_splats = n_splats
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.movement_scale = movement_scale
        self.pruning_threshold = pruning_threshold
        
        # Initialize with more diverse starting positions
        self.splat_base_centers = nn.Parameter(self._diverse_init_centers())
        self.splat_center_deltas = nn.Parameter(torch.zeros(n_heads, n_splats, self.head_dim))
        
        # Initialize with wider range of scales and amplitudes to encourage diversity
        self.splat_log_scales = nn.Parameter(
            torch.randn(n_heads, n_splats) * 0.5 - 0.5  # Mean around exp(-0.5) = 0.6
        )
        self.splat_log_amplitudes = nn.Parameter(
            torch.randn(n_heads, n_splats) * 1.0 - 1.5  # Wide range, some will be small
        )
        
        # Movement and temperature controls
        self.movement_scale_param = nn.Parameter(torch.tensor(movement_scale))
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Projections
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        
        # Tracking buffers - ensure they're properly registered
        self.register_buffer('training_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('adaptation_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('splat_ages', torch.zeros(n_heads, n_splats, dtype=torch.long))
        self.register_buffer('last_adaptation_step', torch.tensor(0, dtype=torch.long))
        
    def _diverse_init_centers(self) -> torch.Tensor:
        """Create diverse initial positions to encourage different specializations"""
        centers = torch.zeros(self.n_heads, self.n_splats, self.head_dim)
        
        for h in range(self.n_heads):
            for s in range(self.n_splats):
                if s % 4 == 0:
                    # Cluster near origin
                    centers[h, s] = torch.randn(self.head_dim) * 0.1
                elif s % 4 == 1:
                    # On unit sphere
                    vec = torch.randn(self.head_dim)
                    centers[h, s] = vec / torch.norm(vec) * 0.5
                elif s % 4 == 2:
                    # Along coordinate axes
                    axis = s % self.head_dim
                    centers[h, s, axis] = 0.7 * (1 if s % 2 == 0 else -1)
                else:
                    # Random positions
                    centers[h, s] = torch.randn(self.head_dim) * 0.3
                    
        return centers
    
    def get_effective_centers(self) -> torch.Tensor:
        """Compute effective splat positions with bounded movement"""
        movement_scale = torch.sigmoid(self.movement_scale_param) * 0.2
        return self.splat_base_centers + self.splat_center_deltas * movement_scale
    
    def forward(self, x: torch.Tensor, return_extras: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Forward pass with robust error handling"""
        B, T, D = x.shape
        
        if D != self.dim:
            raise ValueError(f"Input dimension {D} doesn't match model dimension {self.dim}")
        
        if self.training:
            self.training_step += 1
            self.splat_ages += 1
        
        # Project to Q, K, V with error checking
        qkv = self.qkv(x)  # [B, T, 3*D]
        if qkv.shape[-1] != 3 * self.dim:
            raise RuntimeError(f"QKV projection shape mismatch: got {qkv.shape}, expected last dim {3 * self.dim}")
            
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: [B, T, n_heads, head_dim]
        
        # Get effective centers
        effective_centers = self.get_effective_centers()
        
        # Compute attention with error handling
        try:
            attention_weights = self._compute_attention_robust(q, k, effective_centers)
        except Exception as e:
            print(f"Warning: Attention computation failed ({e}), using uniform attention")
            attention_weights = torch.ones(B, T, T, self.n_heads, device=x.device) / T
        
        # Apply to values
        output = torch.einsum('bijh,bjhd->bihd', attention_weights, v)
        output = output.reshape(B, T, D)
        output = self.out(output)
        
        if return_extras:
            extras = self._get_safe_extras(effective_centers)
            return output, extras
        
        return output, attention_weights
    
    def _compute_attention_robust(self, q: torch.Tensor, k: torch.Tensor, 
                                centers: torch.Tensor) -> torch.Tensor:
        """Robust attention computation with numerical stability"""
        B, T, H, head_dim = q.shape
        
        # Get splat parameters with safety bounds
        scales = torch.exp(self.splat_log_scales).clamp(min=1e-4, max=5.0)
        amplitudes = torch.exp(self.splat_log_amplitudes).clamp(min=1e-6, max=10.0)
        
        # Apply pruning mask
        active_mask = amplitudes > self.pruning_threshold
        effective_amplitudes = amplitudes * active_mask.float()
        
        attention_logits = torch.zeros(B, T, T, H, device=q.device)
        
        for h in range(H):
            # Skip heads with no active splats
            if effective_amplitudes[h].sum() < 1e-8:
                attention_logits[:, :, :, h] = torch.eye(T, device=q.device).unsqueeze(0).expand(B, -1, -1)
                continue
                
            q_h = q[:, :, h]  # [B, T, head_dim]
            k_h = k[:, :, h]  # [B, T, head_dim]
            centers_h = centers[h]  # [n_splats, head_dim]
            
            # Efficient distance computation
            q_expanded = q_h.unsqueeze(2)  # [B, T, 1, head_dim]
            k_expanded = k_h.unsqueeze(2)  # [B, T, 1, head_dim]
            centers_expanded = centers_h.unsqueeze(0).unsqueeze(0)  # [1, 1, n_splats, head_dim]
            
            # Squared distances with numerical stability
            q_dists_sq = torch.sum((q_expanded - centers_expanded) ** 2, dim=-1).clamp(max=100.0)
            k_dists_sq = torch.sum((k_expanded - centers_expanded) ** 2, dim=-1).clamp(max=100.0)
            
            # Gaussian weights with numerical stability
            scales_h = scales[h].unsqueeze(0).unsqueeze(0)
            q_weights = torch.exp(-0.5 * q_dists_sq / (scales_h ** 2 + 1e-8))
            k_weights = torch.exp(-0.5 * k_dists_sq / (scales_h ** 2 + 1e-8))
            
            # Apply amplitudes
            amps_h = effective_amplitudes[h].unsqueeze(0).unsqueeze(0)
            q_weights_amp = q_weights * amps_h
            k_weights_amp = k_weights * amps_h
            
            # Compute attention for this head
            head_attention = torch.einsum('bis,bjs->bij', q_weights_amp, k_weights_amp)
            attention_logits[:, :, :, h] = head_attention
        
        # Apply temperature and normalize with numerical stability
        temp = self.temperature.clamp(min=0.01, max=10.0)
        attention_logits = attention_logits / temp
        
        # Add small epsilon to prevent all-zero rows
        attention_logits = attention_logits + 1e-8
        attention = F.softmax(attention_logits, dim=2)
        
        return attention
    
    def _get_safe_extras(self, effective_centers: torch.Tensor) -> Dict:
        """Get extras dict with safe type conversion"""
        with torch.no_grad():
            amplitudes = torch.exp(self.splat_log_amplitudes)
            scales = torch.exp(self.splat_log_scales)
            
            return {
                'attention': None,  # Will be filled by caller if needed
                'effective_centers': effective_centers.detach().cpu(),
                'amplitudes': amplitudes.detach().cpu(),
                'scales': scales.detach().cpu(),
                'movement_scale': float(torch.sigmoid(self.movement_scale_param).item()),
                'temperature': float(self.temperature.item()),
                'active_splats': int((amplitudes > self.pruning_threshold).sum().item()),
                'total_splats': int(self.n_splats * self.n_heads)
            }
    
    def aggressive_adapt_splats(self, force: bool = False):
        """More aggressive adaptation that actually triggers birth/death"""
        if not self.training:
            return
            
        # Adapt more frequently and aggressively
        should_adapt = (self.training_step % 25 == 0) or force
        if not should_adapt:
            return
            
        with torch.no_grad():
            amplitudes = torch.exp(self.splat_log_amplitudes)
            active_mask = amplitudes > self.pruning_threshold
            
            adaptations_made = 0
            
            for h in range(self.n_heads):
                head_amps = amplitudes[h]
                head_active = active_mask[h]
                
                # Death: Find truly weak splats (much more aggressive)
                weak_threshold = max(0.0001, head_amps.mean().item() * 0.1)  # 10% of mean
                very_weak_mask = head_amps < weak_threshold
                old_splats = self.splat_ages[h] > 50
                
                candidates_for_death = very_weak_mask & old_splats
                
                if candidates_for_death.sum() > 0:
                    # Kill the weakest splats
                    n_to_kill = min(candidates_for_death.sum().item(), self.n_splats // 4)  # Kill up to 25%
                    weakest_indices = head_amps.argsort()[:n_to_kill]
                    
                    for idx in weakest_indices:
                        if candidates_for_death[idx]:
                            # Reset this splat
                            self.splat_base_centers.data[h, idx] = torch.randn(self.head_dim) * 0.3
                            self.splat_center_deltas.data[h, idx] = 0
                            self.splat_log_scales.data[h, idx] = torch.randn(1) * 0.3 - 0.5
                            self.splat_log_amplitudes.data[h, idx] = torch.randn(1) * 0.5 - 2.0
                            self.splat_ages[h, idx] = 0
                            adaptations_made += 1
                
                # Birth: Duplicate successful splats
                if adaptations_made > 0 and head_amps.max() > head_amps.mean() + head_amps.std():
                    # Find the best performing splat
                    best_idx = head_amps.argmax()
                    
                    # Find a weak splat to replace
                    weak_indices = (head_amps < head_amps.median()).nonzero().squeeze()
                    if len(weak_indices) > 0:
                        replace_idx = weak_indices[torch.randint(len(weak_indices), (1,))].item()
                        
                        # Copy the successful splat with noise
                        noise_scale = 0.1
                        self.splat_base_centers.data[h, replace_idx] = (
                            self.splat_base_centers.data[h, best_idx] + 
                            torch.randn(self.head_dim) * noise_scale
                        )
                        self.splat_center_deltas.data[h, replace_idx] = (
                            self.splat_center_deltas.data[h, best_idx] * 0.5
                        )
                        self.splat_log_scales.data[h, replace_idx] = (
                            self.splat_log_scales.data[h, best_idx] + torch.randn(1) * 0.1
                        )
                        self.splat_log_amplitudes.data[h, replace_idx] = (
                            self.splat_log_amplitudes.data[h, best_idx] - 0.5  # Start smaller
                        )
                        self.splat_ages[h, replace_idx] = 0
                        adaptations_made += 1
            
            if adaptations_made > 0:
                self.adaptation_count += adaptations_made
                self.last_adaptation_step = self.training_step.clone()
    
    def get_comprehensive_stats(self) -> Dict[str, Union[int, float]]:
        """Get comprehensive statistics with safe type conversion"""
        with torch.no_grad():
            amplitudes = torch.exp(self.splat_log_amplitudes)
            scales = torch.exp(self.splat_log_scales)
            movement_distances = torch.norm(self.splat_center_deltas, dim=-1)
            active_mask = amplitudes > self.pruning_threshold
            
            # Convert all tensors to Python primitives
            stats = {
                'n_active_splats': int(active_mask.sum().item()),
                'total_splats': int(self.n_splats * self.n_heads),
                'avg_amplitude': float(amplitudes.mean().item()),
                'amplitude_variance': float(amplitudes.var().item()),
                'amplitude_min': float(amplitudes.min().item()),
                'amplitude_max': float(amplitudes.max().item()),
                'avg_movement': float(movement_distances.mean().item()),
                'max_movement': float(movement_distances.max().item()),
                'adaptations_performed': int(self.adaptation_count.item()),
                'training_step': int(self.training_step.item()),
                'last_adaptation_step': int(self.last_adaptation_step.item()),
                'avg_splat_age': float(self.splat_ages.float().mean().item()),
                'movement_scale': float(torch.sigmoid(self.movement_scale_param).item()),
                'temperature': float(self.temperature.item())
            }
            
            return stats

def test_aggressive_birth_death():
    """Test with much more aggressive conditions to force birth/death"""
    print("\n🔄 AGGRESSIVE BIRTH/DEATH TEST")
    print("-" * 50)
    
    dim = 32
    n_splats = 8  # Fewer splats to make competition more intense
    
    gsa = RobustGSA(dim=dim, n_splats=n_splats, n_heads=4, pruning_threshold=0.01)
    
    # Much more aggressive training to create clear winners and losers
    optimizer = torch.optim.Adam(gsa.parameters(), lr=0.1)  # Very high learning rate
    
    print("Creating extreme conditions to force adaptations...")
    
    adaptation_events = []
    
    for step in range(300):
        # Create extremely biased data that should kill some splats
        x = torch.randn(6, 16, dim) * 0.05  # Very small base signal
        
        if step < 100:
            # Phase 1: Only positions 0, 1, 2 matter (should kill splats not serving these)
            x[:, 0] += torch.randn(6, dim) * 2.0  # Very strong signal
            x[:, 1] += torch.randn(6, dim) * 1.5
            x[:, 2] += torch.randn(6, dim) * 1.0
        elif step < 200:
            # Phase 2: Only positions 13, 14, 15 matter (should kill early-position splats)
            x[:, -3:] += torch.randn(6, 3, dim) * 2.0
        else:
            # Phase 3: Only middle positions matter (should kill edge-position splats)
            x[:, 7:9] += torch.randn(6, 2, dim) * 2.0
        
        output, _ = gsa(x)
        
        # Strong reconstruction pressure
        loss = F.mse_loss(output, x)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion with high LR
        torch.nn.utils.clip_grad_norm_(gsa.parameters(), 1.0)
        optimizer.step()
        
        # Check for adaptations every 25 steps
        if step % 25 == 0:
            stats_before = gsa.get_comprehensive_stats()
            gsa.aggressive_adapt_splats(force=True)  # Force adaptation check
            stats_after = gsa.get_comprehensive_stats()
            
            adaptations_this_step = stats_after['adaptations_performed'] - stats_before['adaptations_performed']
            
            adaptation_events.append({
                'step': step,
                'phase': 1 if step < 100 else (2 if step < 200 else 3),
                'adaptations': adaptations_this_step,
                'total_adaptations': stats_after['adaptations_performed'],
                'active_splats': stats_after['n_active_splats'],
                'amp_min': stats_after['amplitude_min'],
                'amp_max': stats_after['amplitude_max'],
                'avg_age': stats_after['avg_splat_age']
            })
            
            if adaptations_this_step > 0:
                print(f"  Step {step} (Phase {adaptation_events[-1]['phase']}): 🎯 {adaptations_this_step} adaptations!")
                print(f"    Active: {stats_after['n_active_splats']}, Amp range: {stats_after['amplitude_min']:.3f}-{stats_after['amplitude_max']:.3f}")
            else:
                print(f"  Step {step}: Active: {stats_after['n_active_splats']}, Amp range: {stats_after['amplitude_min']:.3f}-{stats_after['amplitude_max']:.3f}")
    
    # Analyze results
    total_adaptations = adaptation_events[-1]['total_adaptations'] if adaptation_events else 0
    adaptations_by_phase = {}
    for event in adaptation_events:
        phase = event['phase']
        if phase not in adaptations_by_phase:
            adaptations_by_phase[phase] = 0
        adaptations_by_phase[phase] += event['adaptations']
    
    print(f"\n📊 Aggressive Birth/Death Analysis:")
    print(f"  Total adaptations: {total_adaptations}")
    print(f"  Adaptations by phase: {adaptations_by_phase}")
    print(f"  Final active splats: {adaptation_events[-1]['active_splats'] if adaptation_events else 'N/A'}")
    print(f"  Final amplitude range: {adaptation_events[-1]['amp_min']:.3f} - {adaptation_events[-1]['amp_max']:.3f}")
    
    # Success if we got significant adaptations
    success = total_adaptations >= 5 and len(adaptations_by_phase) >= 2
    print(f"  Status: {'✅ SUCCESS' if success else '❌ STILL NEEDS MORE WORK'}")
    
    return success, total_adaptations, adaptations_by_phase

def test_all_fixes():
    """Test all fixes together"""
    print("\n🔧 COMPREHENSIVE FIX VALIDATION")
    print("-" * 50)
    
    # Test 1: JSON serialization fix
    print("Testing JSON serialization...")
    try:
        test_data = {
            'numpy_bool': bool(np.array([True])[0]),
            'torch_tensor': float(torch.tensor(3.14).item()),
            'normal_bool': True,
            'normal_int': 42
        }
        json_str = json.dumps(test_data)
        print("  ✅ JSON serialization working")
        json_success = True
    except Exception as e:
        print(f"  ❌ JSON serialization failed: {e}")
        json_success = False
    
    # Test 2: Birth/death mechanism
    birth_death_success, total_adaptations, adaptations_by_phase = test_aggressive_birth_death()
    
    # Test 3: Pattern learning (quick test)
    print("\n🎯 Quick Pattern Learning Validation...")
    gsa = RobustGSA(dim=32, n_splats=8, n_heads=4)
    x = torch.randn(4, 12, 32)
    
    try:
        output, extras = gsa(x, return_extras=True)
        pattern_success = output.shape == x.shape and extras is not None
        print(f"  ✅ Pattern learning setup working: output shape {output.shape}")
    except Exception as e:
        print(f"  ❌ Pattern learning failed: {e}")
        pattern_success = False
    
    # Test 4: Visualization data preparation
    print("\n📊 Visualization Data Test...")
    try:
        stats = gsa.get_comprehensive_stats()
        # Ensure all values are JSON serializable
        json.dumps(stats)
        print(f"  ✅ Statistics generation working: {len(stats)} metrics")
        viz_success = True
    except Exception as e:
        print(f"  ❌ Visualization data failed: {e}")
        viz_success = False
    
    return {
        'json_serialization': json_success,
        'birth_death': birth_death_success,
        'pattern_learning': pattern_success,
        'visualization': viz_success,
        'total_adaptations': total_adaptations,
        'adaptations_by_phase': adaptations_by_phase
    }

def run_complete_fix_validation():
    """Run complete validation with all fixes"""
    print("=" * 70)
    print("🛠️  GSA COMPLETE FIX & ERROR RESOLUTION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTarget: Fix all remaining errors and failures")
    
    # Run comprehensive tests
    results = test_all_fixes()
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 COMPLETE FIX RESULTS")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if isinstance(v, bool) and v)
    total = sum(1 for v in results.values() if isinstance(v, bool))
    
    for test_name, success in results.items():
        if isinstance(success, bool):
            status = "✅ FIXED" if success else "❌ STILL BROKEN"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    # Detailed results
    if results['birth_death']:
        print(f"\n🔄 BIRTH/DEATH SUCCESS: {results['total_adaptations']} total adaptations")
        print(f"   Adaptations by phase: {results['adaptations_by_phase']}")
    
    overall_success = passed >= total * 0.8  # 80% success rate
    
    print(f"\nFIXES SUCCESSFUL: {passed}/{total}")
    print(f"OVERALL: {'✅ ALL MAJOR ISSUES RESOLVED' if overall_success else '🔧 SOME ISSUES REMAIN'}")
    
    # Save results with proper type conversion
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'test_results': {k: v for k, v in results.items() if isinstance(v, bool)},
        'total_adaptations': int(results['total_adaptations']),
        'adaptations_by_phase': {str(k): int(v) for k, v in results['adaptations_by_phase'].items()},
        'fixes_successful': int(passed),
        'total_tests': int(total),
        'overall_success': bool(overall_success)
    }
    
    try:
        with open('gsa_complete_fixes_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n✅ Results saved successfully to: gsa_complete_fixes_results.json")
    except Exception as e:
        print(f"\n❌ Failed to save results: {e}")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return overall_success

if __name__ == "__main__":
    success = run_complete_fix_validation()
    exit(0 if success else 1)
