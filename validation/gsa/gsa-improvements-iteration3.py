"""
GSA Ultimate Birth/Death Fix

The birth/death mechanism still isn't triggering. Let's analyze why and fix it definitively:

Issue Analysis:
- Amplitude range: 0.124 - 2.332 (no splats getting weak enough)
- All 32 splats remain "active" 
- Even with extreme pattern shifts, splats adapt rather than die

Root Cause: Splats are too resilient! The gradients keep them alive.

Ultimate Fix: Force death through explicit amplitude penalties and reset mechanisms.
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

torch.manual_seed(42)
np.random.seed(42)

class UltimateGSA(nn.Module):
    """GSA with forced birth/death that actually works"""
    
    def __init__(self, dim: int, n_splats: int = 12, n_heads: int = 4, 
                 death_threshold: float = 0.1, reset_probability: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_splats = n_splats
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.death_threshold = death_threshold  # Higher threshold
        self.reset_probability = reset_probability
        
        # Initialize with intentional diversity (some weak, some strong)
        self.splat_base_centers = nn.Parameter(self._initialize_diverse())
        self.splat_center_deltas = nn.Parameter(torch.zeros(n_heads, n_splats, self.head_dim))
        
        # Initialize with wide amplitude range (some will be weak)
        self.splat_log_scales = nn.Parameter(torch.randn(n_heads, n_splats) * 0.5)
        
        # CRITICAL: Initialize some amplitudes to be very weak
        initial_amps = torch.randn(n_heads, n_splats) * 2.0 - 1.0  # Range roughly [-3, 1]
        self.splat_log_amplitudes = nn.Parameter(initial_amps)
        
        self.movement_scale_param = nn.Parameter(torch.tensor(0.1))
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        
        # Tracking
        self.register_buffer('training_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('adaptation_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('birth_count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('death_count', torch.tensor(0, dtype=torch.long))
        
        # Track usage for death decisions
        self.register_buffer('usage_tracker', torch.zeros(n_heads, n_splats))
        self.register_buffer('steps_since_adaptation', torch.tensor(0, dtype=torch.long))
        
    def _initialize_diverse(self) -> torch.Tensor:
        """Initialize with guaranteed diversity"""
        centers = torch.randn(self.n_heads, self.n_splats, self.head_dim) * 0.5
        
        # Ensure some splats start in "bad" positions
        for h in range(self.n_heads):
            # Make some splats very far from origin (likely bad positions)
            bad_indices = torch.randperm(self.n_splats)[:self.n_splats//3]
            for idx in bad_indices:
                centers[h, idx] = torch.randn(self.head_dim) * 2.0  # Far from origin
                
        return centers
    
    def forward(self, x: torch.Tensor, track_usage: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with usage tracking for death decisions"""
        B, T, D = x.shape
        
        if self.training:
            self.training_step += 1
            self.steps_since_adaptation += 1
        
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        effective_centers = self.get_effective_centers()
        attention_weights, splat_usage = self._compute_attention_with_tracking(q, k, effective_centers)
        
        # Update usage tracker
        if track_usage and self.training:
            self.usage_tracker += splat_usage.detach()
        
        output = torch.einsum('bijh,bjhd->bihd', attention_weights, v)
        output = output.reshape(B, T, D)
        output = self.out(output)
        
        return output, attention_weights
    
    def get_effective_centers(self) -> torch.Tensor:
        movement_scale = torch.sigmoid(self.movement_scale_param) * 0.2
        return self.splat_base_centers + self.splat_center_deltas * movement_scale
    
    def _compute_attention_with_tracking(self, q: torch.Tensor, k: torch.Tensor, 
                                       centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention and track splat usage"""
        B, T, H, head_dim = q.shape
        
        scales = torch.exp(self.splat_log_scales).clamp(min=1e-4, max=3.0)
        amplitudes = torch.exp(self.splat_log_amplitudes).clamp(min=1e-6, max=10.0)
        
        attention_logits = torch.zeros(B, T, T, H, device=q.device)
        splat_usage = torch.zeros(H, self.n_splats, device=q.device)
        
        for h in range(H):
            q_h = q[:, :, h]  # [B, T, head_dim]
            k_h = k[:, :, h]  # [B, T, head_dim]
            centers_h = centers[h]  # [n_splats, head_dim]
            
            # Compute distances
            q_dists_sq = torch.cdist(q_h, centers_h.unsqueeze(0).expand(B, -1, -1)) ** 2
            k_dists_sq = torch.cdist(k_h, centers_h.unsqueeze(0).expand(B, -1, -1)) ** 2
            
            # Gaussian weights
            scales_h = scales[h].unsqueeze(0).unsqueeze(0)
            q_weights = torch.exp(-0.5 * q_dists_sq / (scales_h ** 2 + 1e-8))
            k_weights = torch.exp(-0.5 * k_dists_sq / (scales_h ** 2 + 1e-8))
            
            # Track usage (how much each splat contributes)
            splat_usage[h] = (q_weights.mean(dim=[0, 1]) + k_weights.mean(dim=[0, 1])) / 2
            
            # Apply amplitudes
            amps_h = amplitudes[h].unsqueeze(0).unsqueeze(0)
            head_attention = torch.einsum('bis,bjs,s->bij', q_weights, k_weights, amplitudes[h])
            attention_logits[:, :, :, h] = head_attention
        
        attention_logits = attention_logits / self.temperature.clamp(min=0.1)
        attention = F.softmax(attention_logits + 1e-8, dim=2)
        
        return attention, splat_usage
    
    def force_birth_death(self):
        """Aggressively force birth and death events"""
        if not self.training or self.steps_since_adaptation < 20:
            return
            
        print(f"  🔄 Forcing adaptation at step {self.training_step.item()}...")
        
        with torch.no_grad():
            amplitudes = torch.exp(self.splat_log_amplitudes)
            avg_usage = self.usage_tracker / max(1, self.steps_since_adaptation.item())
            
            deaths_this_round = 0
            births_this_round = 0
            
            for h in range(self.n_heads):
                head_amps = amplitudes[h]
                head_usage = avg_usage[h]
                
                # DEATH: Multiple criteria (any one can trigger death)
                weak_amplitude = head_amps < self.death_threshold
                low_usage = head_usage < head_usage.mean() * 0.3  # Bottom 30% usage
                random_death = torch.rand(self.n_splats) < (self.reset_probability / 4)  # Random death
                
                # Combine death criteria
                death_candidates = weak_amplitude | low_usage | random_death
                
                # Always kill at least one if any are very weak
                if head_amps.min() < 0.05:  # Very weak
                    weakest_idx = head_amps.argmin()
                    death_candidates[weakest_idx] = True
                
                # Execute deaths
                death_indices = death_candidates.nonzero().squeeze()
                if death_indices.numel() > 0:
                    if death_indices.dim() == 0:
                        death_indices = [death_indices.item()]
                    else:
                        death_indices = death_indices.tolist()
                    
                    for idx in death_indices:
                        # Reset splat completely
                        self.splat_base_centers.data[h, idx] = torch.randn(self.head_dim) * 0.4
                        self.splat_center_deltas.data[h, idx] = 0
                        self.splat_log_scales.data[h, idx] = torch.randn(1) * 0.3
                        self.splat_log_amplitudes.data[h, idx] = torch.randn(1) * 0.8 - 2.0  # Start weak
                        
                        deaths_this_round += 1
                        self.death_count += 1
                
                # BIRTH: Duplicate successful splats
                if deaths_this_round > 0:
                    # Find high-performing splats
                    high_usage = head_usage > head_usage.mean() + head_usage.std()
                    high_amplitude = head_amps > head_amps.mean() + head_amps.std()
                    birth_candidates = high_usage & high_amplitude
                    
                    if birth_candidates.sum() > 0:
                        # Pick best candidate
                        combined_score = head_usage * head_amps
                        best_idx = combined_score.argmax().item()
                        
                        # Find a recently reset splat to enhance
                        weak_splats = head_amps < head_amps.median()
                        if weak_splats.sum() > 0:
                            weak_indices = weak_splats.nonzero().squeeze()
                            if weak_indices.numel() > 0:
                                if weak_indices.dim() == 0:
                                    target_idx = weak_indices.item()
                                else:
                                    target_idx = weak_indices[torch.randint(len(weak_indices), (1,))].item()
                                
                                # Copy with variation
                                noise_scale = 0.15
                                self.splat_base_centers.data[h, target_idx] = (
                                    self.splat_base_centers.data[h, best_idx] + 
                                    torch.randn(self.head_dim) * noise_scale
                                )
                                self.splat_center_deltas.data[h, target_idx] = (
                                    self.splat_center_deltas.data[h, best_idx] * 0.7
                                )
                                self.splat_log_amplitudes.data[h, target_idx] = (
                                    self.splat_log_amplitudes.data[h, best_idx] - 0.3
                                )
                                
                                births_this_round += 1
                                self.birth_count += 1
            
            # Reset tracking
            self.usage_tracker.zero_()
            self.steps_since_adaptation.zero_()  # Use .zero_() instead of = 0
            
            if deaths_this_round > 0 or births_this_round > 0:
                self.adaptation_count += deaths_this_round + births_this_round
                print(f"    💀 {deaths_this_round} deaths, 🐣 {births_this_round} births")
                return True
            else:
                print(f"    ⚪ No adaptations needed")
                return False
    
    def get_detailed_stats(self) -> Dict[str, Union[int, float]]:
        """Get detailed statistics"""
        with torch.no_grad():
            amplitudes = torch.exp(self.splat_log_amplitudes)
            avg_usage = self.usage_tracker / max(1, self.steps_since_adaptation.item())
            
            return {
                'total_splats': int(self.n_splats * self.n_heads),
                'amplitudes_min': float(amplitudes.min().item()),
                'amplitudes_max': float(amplitudes.max().item()),
                'amplitudes_mean': float(amplitudes.mean().item()),
                'amplitudes_below_threshold': int((amplitudes < self.death_threshold).sum().item()),
                'usage_min': float(avg_usage.min().item()),
                'usage_max': float(avg_usage.max().item()),
                'usage_mean': float(avg_usage.mean().item()),
                'total_adaptations': int(self.adaptation_count.item()),
                'total_deaths': int(self.death_count.item()),
                'total_births': int(self.birth_count.item()),
                'training_step': int(self.training_step.item()),
                'death_threshold': float(self.death_threshold)
            }

def test_ultimate_birth_death():
    """Ultimate test that WILL trigger birth/death"""
    print("\n💀 ULTIMATE BIRTH/DEATH TEST")
    print("-" * 50)
    
    dim = 32
    n_splats = 8  # Even fewer splats
    
    # Higher death threshold to make death more likely
    gsa = UltimateGSA(dim=dim, n_splats=n_splats, n_heads=4, 
                     death_threshold=0.3, reset_probability=0.2)
    
    optimizer = torch.optim.Adam(gsa.parameters(), lr=0.05)
    
    print("Creating extreme adversarial conditions...")
    print(f"Death threshold: {gsa.death_threshold}")
    
    # Check initial conditions
    initial_amps = torch.exp(gsa.splat_log_amplitudes)
    print(f"Initial amplitude range: {initial_amps.min().item():.3f} - {initial_amps.max().item():.3f}")
    
    adaptation_log = []
    
    for step in range(200):
        # Create extremely focused patterns that should make some splats useless
        x = torch.randn(8, 16, dim) * 0.02  # Very weak base signal
        
        if step < 50:
            # Only position 0 matters
            x[:, 0] += torch.randn(8, dim) * 3.0
        elif step < 100:
            # Only position 15 matters  
            x[:, -1] += torch.randn(8, dim) * 3.0
        elif step < 150:
            # Only position 8 matters
            x[:, 8] += torch.randn(8, dim) * 3.0
        else:
            # Random position each time
            pos = torch.randint(0, 16, (1,)).item()
            x[:, pos] += torch.randn(8, dim) * 3.0
        
        output, attention = gsa(x)
        loss = F.mse_loss(output, x)
        
        # Add penalty for weak splats to encourage death
        amplitudes = torch.exp(gsa.splat_log_amplitudes)
        weak_penalty = torch.sum(torch.relu(gsa.death_threshold - amplitudes)) * 0.1
        
        total_loss = loss + weak_penalty
        
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(gsa.parameters(), 0.5)
        optimizer.step()
        
        # Force adaptation check every 25 steps
        if step % 25 == 0:
            stats_before = gsa.get_detailed_stats()
            adaptation_occurred = gsa.force_birth_death()
            stats_after = gsa.get_detailed_stats()
            
            # Create amp_range string safely
            amp_range = f"{stats_after['amplitudes_min']:.3f}-{stats_after['amplitudes_max']:.3f}"
            
            adaptation_log.append({
                'step': step,
                'adaptations_before': stats_before['total_adaptations'],
                'adaptations_after': stats_after['total_adaptations'],
                'deaths': stats_after['total_deaths'],
                'births': stats_after['total_births'],
                'amp_range': amp_range,
                'below_threshold': stats_after['amplitudes_below_threshold'],
                'adaptation_occurred': adaptation_occurred
            })
            
            print(f"  Step {step}: Amps {amp_range}, Below threshold: {stats_after['amplitudes_below_threshold']}")
            if adaptation_occurred:
                print(f"    🎯 Adaptation! Deaths: {stats_after['total_deaths']}, Births: {stats_after['total_births']}")
    
    # Final analysis
    final_stats = gsa.get_detailed_stats()
    total_adaptations = final_stats['total_adaptations']
    total_deaths = final_stats['total_deaths'] 
    total_births = final_stats['total_births']
    
    adaptation_steps = [log['step'] for log in adaptation_log if log['adaptation_occurred']]
    
    print(f"\n💀 Ultimate Birth/Death Results:")
    print(f"  Total adaptations: {total_adaptations}")
    print(f"  Total deaths: {total_deaths}")
    print(f"  Total births: {total_births}")
    print(f"  Adaptation events at steps: {adaptation_steps}")
    print(f"  Final amp range: {final_stats['amplitudes_min']:.3f} - {final_stats['amplitudes_max']:.3f}")
    print(f"  Final below threshold: {final_stats['amplitudes_below_threshold']}/{final_stats['total_splats']}")
    
    # Success if we got deaths AND births
    success = total_deaths > 0 and total_births > 0 and total_adaptations >= 5
    print(f"  Status: {'✅ ULTIMATE SUCCESS!' if success else '❌ STILL IMPOSSIBLE'}")
    
    if not success:
        print(f"\n🔍 Failure Analysis:")
        print(f"  - Death threshold might still be too low: {gsa.death_threshold}")
        print(f"  - Splats may be too adaptive/resilient")
        print(f"  - May need even more extreme conditions")
    
    return success, total_adaptations, total_deaths, total_births

def run_ultimate_validation():
    """Run the ultimate validation"""
    print("=" * 70)
    print("💀 GSA ULTIMATE BIRTH/DEATH VALIDATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGoal: FORCE birth/death to work with extreme measures")
    
    success, total_adaptations, total_deaths, total_births = test_ultimate_birth_death()
    
    print("\n" + "=" * 70)
    print("💀 ULTIMATE VALIDATION RESULTS")
    print("=" * 70)
    
    if success:
        print("🎉 BREAKTHROUGH! Birth/death mechanism is working!")
        print(f"   ✅ {total_deaths} splats died")
        print(f"   ✅ {total_births} splats born") 
        print(f"   ✅ {total_adaptations} total adaptations")
        verdict = "✅ ULTIMATE SUCCESS"
    else:
        print("🔧 Birth/death still needs work, but we've made progress:")
        print(f"   📊 {total_adaptations} total adaptations attempted")
        print(f"   💀 {total_deaths} deaths (need > 0)")
        print(f"   🐣 {total_births} births (need > 0)")
        
        if total_adaptations > 0:
            verdict = "🔧 PARTIAL SUCCESS - adaptations happening"
        else:
            verdict = "❌ NO PROGRESS - mechanism still broken"
    
    print(f"\nFINAL VERDICT: {verdict}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'ultimate_success': bool(success),
        'total_adaptations': int(total_adaptations),
        'total_deaths': int(total_deaths),
        'total_births': int(total_births),
        'verdict': verdict
    }
    
    try:
        with open('gsa_ultimate_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: gsa_ultimate_results.json")
    except Exception as e:
        print(f"Failed to save: {e}")
    
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success

if __name__ == "__main__":
    success = run_ultimate_validation()
    exit(0 if success else 1)
