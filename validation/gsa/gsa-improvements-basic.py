"""
Focused GSA Repair & Validation Program

This program addresses the key weaknesses identified in the comprehensive validation:
1. Pattern learning degradation (attention patterns getting worse, not better)
2. Computational efficiency (35x slower than standard attention)
3. Visualization tensor shape issues
4. Birth/death mechanisms not actually triggering

Focus: Fix core issues and demonstrate clear improvements over baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EfficientGSA(nn.Module):
    """Efficiency-focused GSA with key repairs"""
    
    def __init__(self, dim: int, n_splats: int = 16, n_heads: int = 8, 
                 movement_scale: float = 0.05, pruning_threshold: float = 0.02):
        super().__init__()
        self.dim = dim
        self.n_splats = n_splats
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.movement_scale = movement_scale
        self.pruning_threshold = pruning_threshold
        
        # Fewer splats, more focused initialization
        self.splat_base_centers = nn.Parameter(self._smart_init_centers())
        self.splat_center_deltas = nn.Parameter(torch.zeros(n_heads, n_splats, self.head_dim))
        
        # Start with more conservative scales and amplitudes
        self.splat_log_scales = nn.Parameter(torch.ones(n_heads, n_splats) * -0.5)  # Smaller initial scales
        self.splat_log_amplitudes = nn.Parameter(torch.ones(n_heads, n_splats) * -1.0)  # Smaller initial amplitudes
        
        # Bounded movement control
        self.movement_scale_param = nn.Parameter(torch.tensor(movement_scale))
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Standard projections
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)  # No bias for efficiency
        self.out = nn.Linear(dim, dim, bias=False)
        
        # Track adaptation metrics
        self.register_buffer('training_step', torch.tensor(0))
        self.register_buffer('adaptation_count', torch.tensor(0))
        
    def _smart_init_centers(self) -> torch.Tensor:
        """Strategic initialization for faster convergence"""
        centers = torch.zeros(self.n_heads, self.n_splats, self.head_dim)
        
        for h in range(self.n_heads):
            # Initialize half on unit sphere, half near origin
            half = self.n_splats // 2
            
            # First half: distributed on unit sphere
            sphere_points = torch.randn(half, self.head_dim)
            sphere_points = sphere_points / torch.norm(sphere_points, dim=1, keepdim=True)
            centers[h, :half] = sphere_points * 0.5
            
            # Second half: near origin with small perturbations
            centers[h, half:] = torch.randn(self.n_splats - half, self.head_dim) * 0.1
            
        return centers
    
    def get_effective_centers(self) -> torch.Tensor:
        """Compute effective splat positions with bounded movement"""
        movement_scale = torch.sigmoid(self.movement_scale_param) * 0.1  # Smaller movement bound
        return self.splat_base_centers + self.splat_center_deltas * movement_scale
    
    def forward(self, x: torch.Tensor, return_extras: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Efficient forward pass with optimizations"""
        B, T, D = x.shape
        
        if self.training:
            self.training_step += 1
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Get effective centers
        effective_centers = self.get_effective_centers()
        
        # Efficient attention computation using batch operations
        attention_weights = self._compute_attention_efficient(q, k, effective_centers)
        
        # Apply to values
        output = torch.einsum('bijh,bjhd->bihd', attention_weights, v)
        output = output.reshape(B, T, D)
        output = self.out(output)
        
        if return_extras:
            extras = {
                'attention': attention_weights,
                'effective_centers': effective_centers,
                'amplitudes': torch.exp(self.splat_log_amplitudes),
                'scales': torch.exp(self.splat_log_scales),
                'movement_scale': torch.sigmoid(self.movement_scale_param).item(),
                'temperature': self.temperature.item()
            }
            return output, extras
        
        return output, attention_weights
    
    def _compute_attention_efficient(self, q: torch.Tensor, k: torch.Tensor, 
                                   centers: torch.Tensor) -> torch.Tensor:
        """Optimized attention computation"""
        B, T, H, head_dim = q.shape
        
        # Pre-compute splat parameters
        scales = torch.exp(self.splat_log_scales).clamp(min=1e-4, max=2.0)
        amplitudes = torch.exp(self.splat_log_amplitudes)
        
        # Apply pruning mask
        active_mask = amplitudes > self.pruning_threshold
        effective_amplitudes = amplitudes * active_mask.float()
        
        attention_logits = torch.zeros(B, T, T, H, device=q.device)
        
        # Vectorized computation per head
        for h in range(H):
            if effective_amplitudes[h].sum() < 1e-6:  # Skip heads with no active splats
                continue
                
            q_h = q[:, :, h]  # [B, T, head_dim]
            k_h = k[:, :, h]  # [B, T, head_dim]
            centers_h = centers[h]  # [n_splats, head_dim]
            
            # Batch distance computation
            q_expanded = q_h.unsqueeze(2)  # [B, T, 1, head_dim]
            k_expanded = k_h.unsqueeze(2)  # [B, T, 1, head_dim]
            centers_expanded = centers_h.unsqueeze(0).unsqueeze(0)  # [1, 1, n_splats, head_dim]
            
            # Compute squared distances
            q_dists_sq = torch.sum((q_expanded - centers_expanded) ** 2, dim=-1)  # [B, T, n_splats]
            k_dists_sq = torch.sum((k_expanded - centers_expanded) ** 2, dim=-1)  # [B, T, n_splats]
            
            # Gaussian weights
            scales_h = scales[h].unsqueeze(0).unsqueeze(0)  # [1, 1, n_splats]
            q_weights = torch.exp(-0.5 * q_dists_sq / (scales_h ** 2 + 1e-8))
            k_weights = torch.exp(-0.5 * k_dists_sq / (scales_h ** 2 + 1e-8))
            
            # Apply amplitudes
            amps_h = effective_amplitudes[h].unsqueeze(0).unsqueeze(0)  # [1, 1, n_splats]
            q_weights_amp = q_weights * amps_h
            k_weights_amp = k_weights * amps_h
            
            # Compute attention for this head
            head_attention = torch.einsum('bis,bjs->bij', q_weights_amp, k_weights_amp)
            attention_logits[:, :, :, h] = head_attention
        
        # Apply temperature and normalize
        attention_logits = attention_logits / self.temperature.clamp(min=0.1)
        attention = F.softmax(attention_logits, dim=2)
        
        return attention
    
    def adapt_splats_targeted(self):
        """More targeted adaptation focusing on clear improvements"""
        if not self.training or self.training_step % 100 != 0:
            return
            
        with torch.no_grad():
            amplitudes = torch.exp(self.splat_log_amplitudes)
            
            for h in range(self.n_heads):
                head_amps = amplitudes[h]
                
                # Only reset truly dead splats (very conservative)
                dead_mask = head_amps < 0.001  # Much lower threshold
                if dead_mask.sum() > 0:
                    n_dead = dead_mask.sum().item()
                    
                    # Reset to random positions
                    new_positions = torch.randn(n_dead, self.head_dim) * 0.2
                    self.splat_base_centers.data[h][dead_mask] = new_positions
                    self.splat_center_deltas.data[h][dead_mask] = 0
                    self.splat_log_amplitudes.data[h][dead_mask] = -2.0  # Small but not zero
                    
                    self.adaptation_count += 1
    
    def get_adaptation_stats(self) -> Dict:
        """Get key adaptation statistics"""
        with torch.no_grad():
            amplitudes = torch.exp(self.splat_log_amplitudes)
            scales = torch.exp(self.splat_log_scales)
            movement_distances = torch.norm(self.splat_center_deltas, dim=-1)
            
            return {
                'n_active_splats': (amplitudes > self.pruning_threshold).sum().item(),
                'total_splats': self.n_splats * self.n_heads,
                'avg_amplitude': amplitudes.mean().item(),
                'amplitude_variance': amplitudes.var().item(),
                'avg_movement': movement_distances.mean().item(),
                'max_movement': movement_distances.max().item(),
                'adaptations_performed': self.adaptation_count.item(),
                'training_step': self.training_step.item()
            }

class StandardAttention(nn.Module):
    """Optimized standard attention for fair comparison"""
    
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Efficient attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.reshape(B, T, D)
        
        return self.out(out), attention

def test_pattern_learning_focused():
    """Focused test on pattern learning with clearer success metrics"""
    print("\n🎯 FOCUSED PATTERN LEARNING TEST")
    print("-" * 50)
    
    dim = 64
    seq_len = 16
    n_heads = 8
    n_splats = 12  # Fewer splats for easier optimization
    
    gsa = EfficientGSA(dim=dim, n_splats=n_splats, n_heads=n_heads)
    
    # Create very clear pattern learning task
    def create_clear_pattern_task(batch_size=8):
        """Create a task where success is easy to measure"""
        x = torch.randn(batch_size, seq_len, dim) * 0.1
        
        # Pattern: first token should attend strongly to last token
        # We'll encourage this by making the last token a "summary" of useful info
        summary_info = torch.randn(batch_size, dim) * 0.5
        x[:, -1] = summary_info  # Last position contains summary
        x[:, 0] += 0.3 * summary_info  # First position should benefit from summary
        
        # Target attention: first position should attend to last position
        target_attention = torch.zeros(batch_size, seq_len, seq_len)
        target_attention[:, 0, -1] = 1.0  # Strong attention from first to last
        # Add some baseline attention
        for i in range(seq_len):
            target_attention[:, i, i] = 0.1  # Small self-attention
        
        # Normalize
        target_attention = target_attention / target_attention.sum(dim=-1, keepdim=True)
        
        return x, target_attention
    
    print("Training on focused pattern learning task...")
    optimizer = torch.optim.Adam(gsa.parameters(), lr=0.002)
    
    initial_pattern_errors = []
    final_pattern_errors = []
    
    for epoch in range(100):
        epoch_errors = []
        
        for step in range(10):  # 10 steps per epoch
            x, target_attention = create_clear_pattern_task()
            
            output, attention = gsa(x)
            
            # Focus on attention pattern for first token only
            first_token_attention = attention[:, 0, :, :].mean(dim=-1)  # Average across heads
            first_token_target = target_attention[:, 0, :]
            
            pattern_loss = F.mse_loss(first_token_attention, first_token_target)
            
            # Small reconstruction loss to maintain output quality
            recon_loss = F.mse_loss(output, x)
            
            total_loss = 5.0 * pattern_loss + 0.1 * recon_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(gsa.parameters(), 0.5)
            optimizer.step()
            
            epoch_errors.append(pattern_loss.item())
        
        avg_error = np.mean(epoch_errors)
        
        if epoch < 5:
            initial_pattern_errors.extend(epoch_errors)
        if epoch >= 95:
            final_pattern_errors.extend(epoch_errors)
        
        if epoch % 20 == 0:
            stats = gsa.get_adaptation_stats()
            print(f"  Epoch {epoch}: Pattern Error = {avg_error:.4f}")
            print(f"    Active splats: {stats['n_active_splats']}/{stats['total_splats']}")
            print(f"    Movement: {stats['avg_movement']:.3f}")
            
            # Trigger adaptation
            gsa.adapt_splats_targeted()
    
    # Analyze improvement
    initial_avg = np.mean(initial_pattern_errors)
    final_avg = np.mean(final_pattern_errors)
    improvement = initial_avg - final_avg
    improvement_pct = (improvement / initial_avg) * 100 if initial_avg > 0 else 0
    
    print(f"\n📊 Pattern Learning Results:")
    print(f"  Initial error: {initial_avg:.4f}")
    print(f"  Final error: {final_avg:.4f}")
    print(f"  Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")
    
    # Success criteria: clear improvement in pattern matching
    success = improvement > 0.01 and improvement_pct > 10
    print(f"  Status: {'✅ SUCCESS' if success else '❌ NEEDS WORK'}")
    
    return success, improvement_pct

def test_computational_efficiency():
    """Test computational efficiency improvements"""
    print("\n⚡ COMPUTATIONAL EFFICIENCY TEST")
    print("-" * 50)
    
    dim = 128
    n_heads = 8
    batch_size = 4
    
    # Test with different splat counts
    splat_counts = [8, 16, 24, 32]
    seq_lengths = [64, 128, 256]
    
    results = {}
    
    for n_splats in splat_counts:
        print(f"\nTesting with {n_splats} splats per head:")
        
        gsa = EfficientGSA(dim=dim, n_splats=n_splats, n_heads=n_heads)
        std_attn = StandardAttention(dim=dim, n_heads=n_heads)
        
        # Warmup
        dummy = torch.randn(batch_size, 32, dim)
        _ = gsa(dummy)
        _ = std_attn(dummy)
        
        seq_results = {}
        
        for seq_len in seq_lengths:
            test_input = torch.randn(batch_size, seq_len, dim)
            
            # Time GSA
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(5):
                _ = gsa(test_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            gsa_time = (time.time() - start) / 5
            
            # Time Standard
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start = time.time()
            for _ in range(5):
                _ = std_attn(test_input)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            std_time = (time.time() - start) / 5
            
            speedup = std_time / gsa_time
            overhead = gsa_time / std_time
            
            seq_results[seq_len] = {
                'gsa_time': gsa_time,
                'std_time': std_time,
                'overhead': overhead,
                'speedup': speedup
            }
            
            print(f"  Seq {seq_len}: {overhead:.1f}x overhead ({speedup:.3f}x speedup)")
        
        results[n_splats] = seq_results
    
    # Find best efficiency point
    best_overhead = float('inf')
    best_config = None
    
    for n_splats, seq_results in results.items():
        avg_overhead = np.mean([r['overhead'] for r in seq_results.values()])
        if avg_overhead < best_overhead:
            best_overhead = avg_overhead
            best_config = n_splats
    
    print(f"\n📊 Efficiency Summary:")
    print(f"  Best configuration: {best_config} splats per head")
    print(f"  Best average overhead: {best_overhead:.1f}x")
    print(f"  Improvement over baseline: {35.8/best_overhead:.1f}x better")
    
    # Success if we improved efficiency significantly
    success = best_overhead < 20  # Target: less than 20x overhead
    print(f"  Status: {'✅ SUCCESS' if success else '❌ NEEDS WORK'}")
    
    return success, best_overhead

def test_birth_death_mechanisms():
    """Test that birth/death actually happens"""
    print("\n🔄 BIRTH/DEATH MECHANISM TEST")
    print("-" * 50)
    
    dim = 32
    n_splats = 8  # Small number to see effects clearly
    
    gsa = EfficientGSA(dim=dim, n_splats=n_splats, n_heads=4, pruning_threshold=0.05)
    
    # Create scenario that should kill some splats
    optimizer = torch.optim.Adam(gsa.parameters(), lr=0.05)  # Higher LR to create instability
    
    print("Creating conditions that should trigger birth/death...")
    
    adaptation_history = []
    
    for step in range(500):
        # Create data that strongly favors only some patterns
        x = torch.randn(4, 12, dim) * 0.1
        
        if step < 250:
            # Phase 1: Only first few positions matter
            x[:, :3] += torch.randn(4, 3, dim) * 1.0
        else:
            # Phase 2: Switch to last few positions
            x[:, -3:] += torch.randn(4, 3, dim) * 1.0
        
        output, _ = gsa(x)
        
        # Reconstruction loss
        loss = F.mse_loss(output, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Force adaptation checks
        if step % 50 == 0:
            stats_before = gsa.get_adaptation_stats()
            gsa.adapt_splats_targeted()
            stats_after = gsa.get_adaptation_stats()
            
            adaptation_history.append({
                'step': step,
                'adaptations_before': stats_before['adaptations_performed'],
                'adaptations_after': stats_after['adaptations_performed'],
                'active_splats': stats_after['n_active_splats'],
                'avg_amplitude': stats_after['avg_amplitude']
            })
            
            if stats_after['adaptations_performed'] > stats_before['adaptations_performed']:
                print(f"  Step {step}: 🎯 Adaptation triggered! Active: {stats_after['n_active_splats']}")
            else:
                print(f"  Step {step}: Active: {stats_after['n_active_splats']}, Avg amp: {stats_after['avg_amplitude']:.3f}")
    
    # Analyze adaptation activity
    total_adaptations = adaptation_history[-1]['adaptations_after'] if adaptation_history else 0
    pattern_shift_step = 250
    adaptations_after_shift = sum(1 for h in adaptation_history 
                                if h['step'] >= pattern_shift_step and 
                                h['adaptations_after'] > h['adaptations_before'])
    
    print(f"\n📊 Birth/Death Analysis:")
    print(f"  Total adaptations performed: {total_adaptations}")
    print(f"  Adaptations after pattern shift: {adaptations_after_shift}")
    print(f"  Final active splats: {adaptation_history[-1]['active_splats'] if adaptation_history else 'N/A'}")
    
    # Success if adaptations actually happened
    success = total_adaptations > 0 and adaptations_after_shift > 0
    print(f"  Status: {'✅ SUCCESS' if success else '❌ NEEDS WORK'}")
    
    return success, total_adaptations

def create_fixed_visualization():
    """Create a working visualization without tensor shape issues"""
    print("\n📊 CREATING WORKING VISUALIZATION")
    print("-" * 50)
    
    try:
        # Simple, controlled setup
        dim = 32
        n_splats = 12
        gsa = EfficientGSA(dim=dim, n_splats=n_splats, n_heads=4)
        
        # Simple training
        optimizer = torch.optim.Adam(gsa.parameters(), lr=0.01)
        
        for step in range(100):
            x = torch.randn(4, 16, dim)
            output, _ = gsa(x)
            loss = F.mse_loss(output, x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        with torch.no_grad():
            stats = gsa.get_adaptation_stats()
            effective_centers = gsa.get_effective_centers()
            amplitudes = torch.exp(gsa.splat_log_amplitudes)
            scales = torch.exp(gsa.splat_log_scales)
            movement_distances = torch.norm(gsa.splat_center_deltas, dim=-1)
            
            # 1. Amplitude distribution
            ax = axes[0, 0]
            ax.hist(amplitudes.cpu().numpy().flatten(), bins=15, alpha=0.7, color='blue')
            ax.axvline(x=gsa.pruning_threshold, color='red', linestyle='--', label='Pruning Threshold')
            ax.set_title('Splat Amplitude Distribution')
            ax.set_xlabel('Amplitude')
            ax.legend()
            
            # 2. Movement distances
            ax = axes[0, 1]
            ax.hist(movement_distances.cpu().numpy().flatten(), bins=15, alpha=0.7, color='green')
            ax.set_title('Splat Movement Distances')
            ax.set_xlabel('Movement Distance')
            
            # 3. Scale distribution
            ax = axes[1, 0]
            ax.hist(scales.cpu().numpy().flatten(), bins=15, alpha=0.7, color='orange')
            ax.set_title('Splat Scale Distribution')
            ax.set_xlabel('Scale (σ)')
            
            # 4. Summary statistics
            ax = axes[1, 1]
            stats_text = f"""
            Active Splats: {stats['n_active_splats']}/{stats['total_splats']}
            Avg Amplitude: {stats['avg_amplitude']:.3f}
            Avg Movement: {stats['avg_movement']:.3f}
            Adaptations: {stats['adaptations_performed']}
            Training Steps: {stats['training_step']}
            """
            ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title('Training Statistics')
        
        plt.tight_layout()
        plt.savefig('repaired_gsa_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualization saved successfully: repaired_gsa_analysis.png")
        return True
        
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        return False

def run_focused_repair_validation():
    """Run focused validation targeting specific weaknesses"""
    print("=" * 70)
    print("🔧 FOCUSED GSA REPAIR & VALIDATION SUITE")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTarget: Fix pattern learning, efficiency, and birth/death issues")
    
    results = {}
    
    # Test 1: Pattern Learning (the main failure)
    pattern_success, pattern_improvement = test_pattern_learning_focused()
    results['pattern_learning'] = pattern_success
    
    # Test 2: Computational Efficiency
    efficiency_success, best_overhead = test_computational_efficiency()
    results['efficiency'] = efficiency_success
    
    # Test 3: Birth/Death Mechanisms
    lifecycle_success, total_adaptations = test_birth_death_mechanisms()
    results['birth_death'] = lifecycle_success
    
    # Test 4: Visualization
    viz_success = create_fixed_visualization()
    results['visualization'] = viz_success
    
    # Summary
    print("\n" + "=" * 70)
    print("🎯 FOCUSED REPAIR RESULTS")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ FIXED" if success else "❌ STILL BROKEN"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    if results['pattern_learning']:
        print(f"\n🎉 PATTERN LEARNING: Achieved {pattern_improvement:.1f}% improvement!")
    
    if results['efficiency']:
        print(f"⚡ EFFICIENCY: Reduced overhead to {best_overhead:.1f}x (was 35.8x)")
    
    if results['birth_death']:
        print(f"🔄 BIRTH/DEATH: {total_adaptations} adaptations performed")
    
    print(f"\nREPAIRS SUCCESSFUL: {passed}/{total}")
    overall_success = passed >= 3  # Need at least 3/4 working
    
    print(f"OVERALL: {'✅ MAJOR IMPROVEMENTS' if overall_success else '🔧 MORE WORK NEEDED'}")
    
    # Save results
    repair_results = {
        'timestamp': datetime.now().isoformat(),
        'test_results': {k: bool(v) for k, v in results.items()},  # Convert numpy bools to Python bools
        'pattern_improvement_pct': float(pattern_improvement) if 'pattern_improvement' in locals() else 0.0,
        'efficiency_overhead': float(best_overhead) if 'best_overhead' in locals() else 0.0,
        'adaptations_performed': int(total_adaptations) if 'total_adaptations' in locals() else 0,
        'repairs_successful': int(passed),
        'total_tests': int(total),
        'overall_success': bool(overall_success)
    }
    
    with open('gsa_repair_results.json', 'w') as f:
        json.dump(repair_results, f, indent=2)
    
    print(f"\nResults saved to: gsa_repair_results.json")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return overall_success

if __name__ == "__main__":
    success = run_focused_repair_validation()
    exit(0 if success else 1)
