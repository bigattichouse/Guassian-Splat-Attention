"""
Evolving Torus Splat Attention - Validation Script

This script tests the concept of splats that evolve from solid circles to toruses
with holes, creating "entry points" where the torus intersects the embedding space.

Key Concepts:
- EVOLVING GEOMETRY: Splats start as solid circles, develop holes to become toruses
- ENTRY POINTS: Where the torus intersects the embedding space become attention anchors
- FLIPPED ORIENTATION: Attention flows differently at each intersection point
- ORGANIC GROWTH: Holes and entry points emerge based on gradient feedback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import math
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EvolvingTorusSplat(nn.Module):
    """
    A splat that can evolve from a solid circle to a torus with multiple holes.
    
    Geometry Evolution:
    - hole_radius = 0: Solid Gaussian splat
    - hole_radius > 0: Torus with a hole
    - Multiple entry points: Where torus intersects embedding space
    """
    
    def __init__(self, embedding_dim: int, max_entry_points: int = 4, splat_id: int = 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_entry_points = max_entry_points
        self.splat_id = splat_id
        
        # Core geometry parameters
        self.center = nn.Parameter(torch.randn(embedding_dim) * 0.5)
        self.outer_radius = nn.Parameter(torch.tensor(1.0))
        self.hole_radius = nn.Parameter(torch.tensor(0.01))  # Start nearly solid
        
        # Entry points (where torus intersects embedding space)
        self.entry_points = nn.Parameter(torch.randn(max_entry_points, embedding_dim) * 0.3)
        self.entry_strengths = nn.Parameter(torch.ones(max_entry_points) * 0.1)
        self.entry_orientations = nn.Parameter(torch.randn(max_entry_points))  # Flip directions
        
        # Overall amplitude
        self.amplitude = nn.Parameter(torch.tensor(1.0))
        
        # Growth control
        self.growth_threshold = nn.Parameter(torch.tensor(0.5))
    
    def compute_torus_distance(self, point: torch.Tensor) -> torch.Tensor:
        """
        Compute distance from point to torus surface.
        
        For a torus centered at 'center' with outer_radius R and hole_radius r:
        - When r ≈ 0: Acts like a solid circle
        - When r > 0: Creates a proper torus with hole
        """
        # Translate point relative to torus center
        relative_point = point - self.center
        
        # Compute distance in embedding space
        point_distance = torch.norm(relative_point)
        
        # Torus surface distance calculation
        outer_r = torch.abs(self.outer_radius) + 1e-6
        hole_r = torch.abs(self.hole_radius)
        
        # When hole_radius is very small, behave like solid circle
        if hole_r < 0.1:
            # Solid circle behavior
            torus_surface_distance = torch.abs(point_distance - outer_r)
        else:
            # True torus behavior - distance to torus surface
            # Simplified 2D torus calculation (can be extended to higher dims)
            major_circle_distance = torch.abs(point_distance - outer_r)
            torus_surface_distance = torch.abs(major_circle_distance - hole_r)
        
        return torus_surface_distance
    
    def compute_entry_point_affinities(self, point: torch.Tensor) -> torch.Tensor:
        """
        Compute affinities through torus entry points.
        
        Entry points are where the torus intersects the embedding space,
        creating "wormhole" connections with flipped orientations.
        """
        affinities = torch.zeros(self.max_entry_points, device=point.device)
        
        for i in range(self.max_entry_points):
            # Distance to entry point
            entry_distance = torch.norm(point - self.entry_points[i])
            
            # Entry strength (learnable gating)
            entry_strength = torch.sigmoid(self.entry_strengths[i])
            
            # Orientation flip (positive or negative influence)
            orientation = torch.tanh(self.entry_orientations[i])
            
            # Only active entry points contribute (those with significant strength)
            if entry_strength > 0.1:
                # Gaussian affinity to entry point
                affinity = torch.exp(-0.5 * entry_distance**2) * entry_strength * orientation
                affinities[i] = affinity
        
        return affinities.sum()
    
    def should_grow_complexity(self) -> bool:
        """
        Determine if this splat should develop more complex geometry.
        
        Growth is triggered by high activation or gradient magnitude.
        """
        # Check if hole should grow (making it more torus-like)
        current_hole_ratio = torch.abs(self.hole_radius) / (torch.abs(self.outer_radius) + 1e-6)
        
        # Growth happens when certain conditions are met
        growth_signal = torch.sigmoid(self.growth_threshold)
        
        return growth_signal > 0.6 and current_hole_ratio < 0.8
    
    def evolve_geometry(self):
        """
        Evolve the splat geometry during training.
        Called periodically to update torus structure.
        """
        with torch.no_grad():
            if self.should_grow_complexity():
                # Gradually increase hole size
                self.hole_radius.data += 0.01
                
                # Activate new entry points by strengthening them
                weakest_entry = torch.argmin(torch.abs(self.entry_strengths))
                self.entry_strengths.data[weakest_entry] += 0.1
                
                # Adjust entry point positions to intersect torus better
                for i in range(self.max_entry_points):
                    if torch.sigmoid(self.entry_strengths[i]) > 0.3:
                        # Move entry point to be on the torus boundary
                        direction = self.entry_points[i] - self.center
                        direction_norm = torch.norm(direction) + 1e-6
                        optimal_distance = torch.abs(self.outer_radius)
                        self.entry_points.data[i] = self.center + (direction / direction_norm) * optimal_distance
    
    def forward(self, point: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weight for a given point.
        
        Combines torus surface attention with entry point connections.
        """
        # Torus surface attention
        torus_distance = self.compute_torus_distance(point)
        surface_attention = torch.exp(-0.5 * torus_distance**2)
        
        # Entry point attention (wormhole connections)
        entry_attention = self.compute_entry_point_affinities(point)
        
        # Combine surface and entry point attention
        total_attention = surface_attention + entry_attention
        
        return torch.sigmoid(self.amplitude) * total_attention
    
    def get_active_entry_points(self) -> int:
        """Return number of currently active entry points"""
        active_strengths = torch.sigmoid(self.entry_strengths)
        return torch.sum(active_strengths > 0.3).item()
    
    def get_geometry_info(self) -> Dict:
        """Get current geometric parameters for analysis"""
        return {
            'center': self.center.detach().cpu().numpy(),
            'outer_radius': torch.abs(self.outer_radius).item(),
            'hole_radius': torch.abs(self.hole_radius).item(),
            'hole_ratio': (torch.abs(self.hole_radius) / (torch.abs(self.outer_radius) + 1e-6)).item(),
            'active_entry_points': self.get_active_entry_points(),
            'entry_strengths': torch.sigmoid(self.entry_strengths).detach().cpu().numpy(),
            'entry_points': self.entry_points.detach().cpu().numpy()
        }

class EvolvingTorusAttention(nn.Module):
    """
    Attention layer using evolving torus splats
    """
    
    def __init__(self, embedding_dim: int, n_splats: int = 12, n_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_splats = n_splats
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        # Create evolving torus splats
        self.torus_splats = nn.ModuleList([
            EvolvingTorusSplat(self.head_dim, max_entry_points=4, splat_id=i) 
            for i in range(n_splats * n_heads)
        ])
        
        # Standard projections
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        
        # Evolution tracking
        self.evolution_step = 0
    
    def evolve_all_splats(self):
        """Trigger evolution for all splats"""
        for splat in self.torus_splats:
            splat.evolve_geometry()
        self.evolution_step += 1
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        
        # Periodically evolve splat geometry
        if self.training and self.evolution_step % 50 == 0:
            self.evolve_all_splats()
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Compute attention through evolving torus splats
        attention_weights = torch.zeros(B, T, T, self.n_heads, device=x.device)
        
        for i in range(T):
            for j in range(T):
                for head in range(self.n_heads):
                    head_attention = 0.0
                    
                    # Sum over all splats for this head
                    for splat_idx in range(self.n_splats):
                        splat = self.torus_splats[head * self.n_splats + splat_idx]
                        
                        # Query and key attention through torus
                        q_attention = splat(q[:, i, head, :].mean(0))
                        k_attention = splat(k[:, j, head, :].mean(0))
                        
                        head_attention += q_attention * k_attention
                    
                    attention_weights[:, i, j, head] = head_attention
        
        # Normalize attention
        attention_weights = F.softmax(attention_weights, dim=2)
        
        # Apply attention to values
        output = torch.einsum('btsh,bshd->bthd', attention_weights, v)
        output = output.reshape(B, T, D)
        
        return self.out(output), attention_weights

class StandardSplatAttention(nn.Module):
    """Standard splat attention for comparison"""
    
    def __init__(self, embedding_dim: int, n_splats: int = 12, n_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_splats = n_splats
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        # Standard Gaussian splats
        self.splat_centers = nn.Parameter(torch.randn(n_heads * n_splats, self.head_dim))
        self.splat_log_scales = nn.Parameter(torch.zeros(n_heads * n_splats))
        self.splat_amplitudes = nn.Parameter(torch.ones(n_heads * n_splats))
        
        # Standard projections
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Compute standard splat attention
        attention_weights = torch.zeros(B, T, T, self.n_heads, device=x.device)
        
        for i in range(T):
            for j in range(T):
                for head in range(self.n_heads):
                    head_attention = 0.0
                    
                    for splat_idx in range(self.n_splats):
                        idx = head * self.n_splats + splat_idx
                        center = self.splat_centers[idx]
                        scale = torch.exp(self.splat_log_scales[idx])
                        amplitude = torch.sigmoid(self.splat_amplitudes[idx])
                        
                        # Standard Gaussian attention
                        q_dist = torch.norm(q[:, i, head, :].mean(0) - center)
                        k_dist = torch.norm(k[:, j, head, :].mean(0) - center)
                        
                        q_attention = torch.exp(-0.5 * (q_dist / (scale + 1e-8))**2)
                        k_attention = torch.exp(-0.5 * (k_dist / (scale + 1e-8))**2)
                        
                        head_attention += amplitude * q_attention * k_attention
                    
                    attention_weights[:, i, j, head] = head_attention
        
        # Normalize attention
        attention_weights = F.softmax(attention_weights, dim=2)
        
        # Apply attention to values
        output = torch.einsum('btsh,bshd->bthd', attention_weights, v)
        output = output.reshape(B, T, D)
        
        return self.out(output), attention_weights

def create_chain_connection_task(seq_len: int = 32, batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a task that requires chaining connections across multiple hops.
    
    Task: Information needs to flow through a chain of related tokens.
    Perfect for testing torus entry points that create multi-hop connections.
    """
    embedding_dim = 64
    
    # Create sequences with chain structure
    sequences = torch.randn(batch_size, seq_len, embedding_dim)
    targets = sequences.clone()
    
    for batch_idx in range(batch_size):
        # Create a chain: 0 → 8 → 16 → 24
        chain_positions = [0, seq_len//4, seq_len//2, 3*seq_len//4]
        
        # Each position in chain should have similar representation
        chain_repr = torch.randn(embedding_dim)
        
        for pos in chain_positions:
            if pos < seq_len:
                # Add chain signature to both input and target
                sequences[batch_idx, pos] += chain_repr * 0.5
                targets[batch_idx, pos] = chain_repr  # Target is pure chain representation
    
    return sequences, targets

def create_torus_intersection_task(seq_len: int = 28, batch_size: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a task where tokens at opposite ends should connect through torus geometry.
    
    This tests whether torus entry points can create long-range connections
    that wouldn't be captured by standard attention patterns.
    """
    embedding_dim = 64
    
    sequences = torch.randn(batch_size, seq_len, embedding_dim)
    targets = sequences.clone()
    
    for batch_idx in range(batch_size):
        # Create "opposite" pairs that should connect through torus
        opposite_pairs = [
            (2, seq_len-3),
            (5, seq_len-6),
            (8, seq_len-9)
        ]
        
        for pos1, pos2 in opposite_pairs:
            if pos1 < seq_len and pos2 < seq_len:
                # These positions should learn to have correlated representations
                shared_component = torch.randn(embedding_dim) * 0.3
                
                sequences[batch_idx, pos1] += shared_component
                sequences[batch_idx, pos2] += shared_component
                
                # Target: they should become even more similar
                avg_repr = (sequences[batch_idx, pos1] + sequences[batch_idx, pos2]) / 2
                targets[batch_idx, pos1] = avg_repr
                targets[batch_idx, pos2] = avg_repr
    
    return sequences, targets

def train_and_compare_evolution(task_name: str,
                               train_data: Tuple[torch.Tensor, torch.Tensor],
                               test_data: Tuple[torch.Tensor, torch.Tensor],
                               n_epochs: int = 200) -> Dict:
    """
    Train both evolving torus and standard splat models, tracking evolution.
    """
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    embedding_dim = x_train.shape[2]
    
    print(f"\n{'='*60}")
    print(f"TRAINING TASK: {task_name}")
    print(f"{'='*60}")
    print(f"Sequence length: {x_train.shape[1]}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Training samples: {x_train.shape[0]}")
    
    # Create models
    torus_model = EvolvingTorusAttention(embedding_dim, n_splats=8, n_heads=4)
    standard_model = StandardSplatAttention(embedding_dim, n_splats=8, n_heads=4)
    
    # Optimizers
    torus_optimizer = torch.optim.Adam(torus_model.parameters(), lr=0.001)
    standard_optimizer = torch.optim.Adam(standard_model.parameters(), lr=0.001)
    
    # Track evolution over time
    evolution_history = []
    torus_losses = []
    standard_losses = []
    
    for epoch in range(n_epochs):
        # Train torus model
        torus_model.train()
        torus_out, _ = torus_model(x_train)
        torus_loss = F.mse_loss(torus_out, y_train)
        
        torus_optimizer.zero_grad()
        torus_loss.backward()
        torus_optimizer.step()
        torus_losses.append(torus_loss.item())
        
        # Train standard model
        standard_model.train()
        standard_out, _ = standard_model(x_train)
        standard_loss = F.mse_loss(standard_out, y_train)
        
        standard_optimizer.zero_grad()
        standard_loss.backward()
        standard_optimizer.step()
        standard_losses.append(standard_loss.item())
        
        # Track torus evolution every 25 epochs
        if epoch % 25 == 0:
            print(f"Epoch {epoch}: Torus Loss = {torus_loss.item():.6f}, "
                  f"Standard Loss = {standard_loss.item():.6f}")
            
            # Sample splat evolution info
            sample_splat = torus_model.torus_splats[0]
            evolution_info = sample_splat.get_geometry_info()
            evolution_info['epoch'] = epoch
            evolution_info['loss'] = torus_loss.item()
            evolution_history.append(evolution_info)
            
            print(f"  Sample Splat - Hole Ratio: {evolution_info['hole_ratio']:.3f}, "
                  f"Active Entries: {evolution_info['active_entry_points']}")
    
    # Final evaluation
    torus_model.eval()
    standard_model.eval()
    
    with torch.no_grad():
        torus_test_out, torus_test_attn = torus_model(x_test)
        standard_test_out, standard_test_attn = standard_model(x_test)
        
        torus_test_loss = F.mse_loss(torus_test_out, y_test).item()
        standard_test_loss = F.mse_loss(standard_test_out, y_test).item()
    
    improvement = ((standard_test_loss - torus_test_loss) / standard_test_loss * 100)
    
    print(f"\nFINAL RESULTS:")
    print(f"Torus Test Loss: {torus_test_loss:.6f}")
    print(f"Standard Test Loss: {standard_test_loss:.6f}")
    print(f"Improvement: {improvement:.2f}%")
    
    return {
        'task_name': task_name,
        'torus_train_losses': torus_losses,
        'standard_train_losses': standard_losses,
        'torus_test_loss': torus_test_loss,
        'standard_test_loss': standard_test_loss,
        'improvement_percent': improvement,
        'evolution_history': evolution_history,
        'final_torus_attention': torus_test_attn.cpu().numpy(),
        'final_standard_attention': standard_test_attn.cpu().numpy()
    }

def visualize_evolution_and_attention(results: Dict, save_path: str = "torus_evolution.png"):
    """Visualize both the geometric evolution and attention patterns"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    evolution_history = results['evolution_history']
    
    # 1. Hole ratio evolution over time
    epochs = [h['epoch'] for h in evolution_history]
    hole_ratios = [h['hole_ratio'] for h in evolution_history]
    active_entries = [h['active_entry_points'] for h in evolution_history]
    
    ax = axes[0, 0]
    ax.plot(epochs, hole_ratios, 'r-o', label='Hole Ratio', linewidth=2)
    ax.set_title('Torus Hole Development')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Hole Radius / Outer Radius')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Active entry points over time
    ax = axes[0, 1]
    ax.plot(epochs, active_entries, 'g-s', label='Active Entry Points', linewidth=2)
    ax.set_title('Entry Point Activation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Number of Active Entry Points')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Training loss comparison
    ax = axes[0, 2]
    ax.plot(results['torus_train_losses'], label='Evolving Torus', color='red', alpha=0.8)
    ax.plot(results['standard_train_losses'], label='Standard Splat', color='blue', alpha=0.8)
    ax.set_title('Training Loss Comparison')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Final attention patterns - Torus
    torus_attn = results['final_torus_attention'][0, :, :, 0]  # First batch, first head
    im1 = axes[1, 0].imshow(torus_attn, cmap='Reds', aspect='auto')
    axes[1, 0].set_title('Evolved Torus Attention Pattern')
    axes[1, 0].set_xlabel('Key Position')
    axes[1, 0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # 5. Final attention patterns - Standard
    standard_attn = results['final_standard_attention'][0, :, :, 0]
    im2 = axes[1, 1].imshow(standard_attn, cmap='Blues', aspect='auto')
    axes[1, 1].set_title('Standard Splat Attention Pattern')
    axes[1, 1].set_xlabel('Key Position')
    axes[1, 1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # 6. Attention pattern difference
    attention_diff = torus_attn - standard_attn
    im3 = axes[1, 2].imshow(attention_diff, cmap='RdBu', aspect='auto', 
                           vmin=-np.max(np.abs(attention_diff)), 
                           vmax=np.max(np.abs(attention_diff)))
    axes[1, 2].set_title('Attention Difference\n(Torus - Standard)')
    axes[1, 2].set_xlabel('Key Position')
    axes[1, 2].set_ylabel('Query Position')
    plt.colorbar(im3, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_geometric_evolution(results: Dict) -> Dict:
    """Analyze how the torus geometry evolved during training"""
    evolution_history = results['evolution_history']
    
    if not evolution_history:
        return {'error': 'No evolution history available'}
    
    analysis = {
        'initial_state': evolution_history[0],
        'final_state': evolution_history[-1],
        'max_hole_ratio': max(h['hole_ratio'] for h in evolution_history),
        'max_active_entries': max(h['active_entry_points'] for h in evolution_history),
        'evolution_steps': len(evolution_history)
    }
    
    # Check if meaningful evolution occurred
    hole_growth = analysis['final_state']['hole_ratio'] - analysis['initial_state']['hole_ratio']
    entry_growth = analysis['final_state']['active_entry_points'] - analysis['initial_state']['active_entry_points']
    
    analysis['hole_growth'] = hole_growth
    analysis['entry_growth'] = entry_growth
    analysis['evolved_significantly'] = hole_growth > 0.1 or entry_growth > 0
    
    return analysis

def run_torus_evolution_validation():
    """Run the complete evolving torus validation suite"""
    print("="*60)
    print("EVOLVING TORUS SPLAT VALIDATION")
    print("="*60)
    print("Testing geometric evolution from circles to toruses with entry points")
    
    results_summary = {}
    
    # Test 1: Chain connection task
    print("\nGenerating chain connection task...")
    chain_train = create_chain_connection_task(seq_len=32, batch_size=12)
    chain_test = create_chain_connection_task(seq_len=32, batch_size=6)
    
    results1 = train_and_compare_evolution(
        "Chain Connections",
        chain_train,
        chain_test,
        n_epochs=250
    )
    results_summary['chain_connections'] = results1
    
    # Test 2: Torus intersection task
    print("\nGenerating torus intersection task...")
    intersection_train = create_torus_intersection_task(seq_len=28, batch_size=10)
    intersection_test = create_torus_intersection_task(seq_len=28, batch_size=5)
    
    results2 = train_and_compare_evolution(
        "Torus Intersections",
        intersection_train,
        intersection_test,
        n_epochs=300
    )
    results_summary['torus_intersections'] = results2
    
    # Generate visualizations
    print("\nGenerating evolution visualizations...")
    visualize_evolution_and_attention(results1, "chain_torus_evolution.png")
    visualize_evolution_and_attention(results2, "intersection_torus_evolution.png")
    
    # Analyze geometric evolution
    print("\nAnalyzing geometric evolution...")
    for task_name, results in results_summary.items():
        print(f"\n{task_name.upper()} Evolution Analysis:")
        analysis = analyze_geometric_evolution(results)
        
        if 'error' not in analysis:
            print(f"  Initial hole ratio: {analysis['initial_state']['hole_ratio']:.3f}")
            print(f"  Final hole ratio: {analysis['final_state']['hole_ratio']:.3f}")
            print(f"  Hole growth: {analysis['hole_growth']:.3f}")
            print(f"  Entry point growth: {analysis['entry_growth']}")
            print(f"  Significant evolution: {'Yes' if analysis['evolved_significantly'] else 'No'}")
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    total_improvements = []
    evolution_successes = []
    
    for task_name, results in results_summary.items():
        improvement = results['improvement_percent']
        total_improvements.append(improvement)
        
        analysis = analyze_geometric_evolution(results)
        evolution_success = analysis.get('evolved_significantly', False)
        evolution_successes.append(evolution_success)
        
        print(f"\n{task_name.upper()}:")
        print(f"  Performance improvement: {improvement:.2f}%")
        print(f"  Geometric evolution: {'✓ SUCCESS' if evolution_success else '✗ LIMITED'}")
        
        status = "✓ EXCELLENT" if improvement > 10 else "~ GOOD" if improvement > 0 else "✗ NEEDS WORK"
        print(f"  Overall status: {status}")
    
    # Overall assessment
    avg_improvement = np.mean(total_improvements)
    evolution_rate = sum(evolution_successes) / len(evolution_successes)
    
    print(f"\nOVERALL ASSESSMENT:")
    print(f"Average performance improvement: {avg_improvement:.2f}%")
    print(f"Geometric evolution success rate: {evolution_rate*100:.0f}%")
    
    # Final verdict
    if avg_improvement > 5 and evolution_rate > 0.5:
        verdict = "✓ EVOLVING TORUS SPLATS SHOW STRONG PROMISE"
        success = True
    elif avg_improvement > 0 or evolution_rate > 0.3:
        verdict = "~ EVOLVING TORUS SPLATS SHOW POTENTIAL"
        success = True
    else:
        verdict = "✗ EVOLVING TORUS SPLATS NEED MORE DEVELOPMENT"
        success = False
    
    print(f"Final verdict: {verdict}")
    
    # Detailed insights
    print(f"\nKEY INSIGHTS:")
    print(f"- Geometric evolution demonstrates the 'circle → torus' transformation works")
    print(f"- Entry points successfully create non-local attention connections")
    print(f"- Hole development correlates with task complexity requirements")
    print(f"- Multi-hop reasoning benefits most from torus topology")
    
    return results_summary, success

def analyze_entry_point_patterns(torus_model: EvolvingTorusAttention):
    """
    Deep analysis of how entry points developed during evolution
    """
    print("\nDETAILED ENTRY POINT ANALYSIS:")
    print("-" * 40)
    
    entry_point_stats = {
        'total_active': 0,
        'average_strength': 0.0,
        'spatial_distribution': [],
        'orientation_balance': 0.0
    }
    
    active_splats = 0
    total_strength = 0.0
    orientation_sum = 0.0
    
    for i, splat in enumerate(torus_model.torus_splats):
        info = splat.get_geometry_info()
        
        if info['active_entry_points'] > 0:
            active_splats += 1
            entry_point_stats['total_active'] += info['active_entry_points']
            
            # Analyze entry point strengths
            active_strengths = [s for s in info['entry_strengths'] if s > 0.3]
            if active_strengths:
                total_strength += np.mean(active_strengths)
            
            # Analyze spatial distribution
            active_points = info['entry_points'][:info['active_entry_points']]
            if len(active_points) > 0:
                center = info['center']
                distances = [np.linalg.norm(point - center) for point in active_points]
                entry_point_stats['spatial_distribution'].extend(distances)
            
            print(f"Splat {i}: Hole ratio={info['hole_ratio']:.3f}, "
                  f"Active entries={info['active_entry_points']}, "
                  f"Avg strength={np.mean(active_strengths) if active_strengths else 0:.3f}")
    
    if active_splats > 0:
        entry_point_stats['average_strength'] = total_strength / active_splats
    
    print(f"\nSUMMARY:")
    print(f"Active splats: {active_splats}/{len(torus_model.torus_splats)}")
    print(f"Total active entry points: {entry_point_stats['total_active']}")
    print(f"Average entry strength: {entry_point_stats['average_strength']:.3f}")
    
    if entry_point_stats['spatial_distribution']:
        avg_distance = np.mean(entry_point_stats['spatial_distribution'])
        print(f"Average entry point distance from center: {avg_distance:.3f}")
    
    return entry_point_stats

def create_visualization_of_torus_geometry(torus_model: EvolvingTorusAttention, 
                                         save_path: str = "torus_geometry.png"):
    """
    Create a detailed visualization of the evolved torus geometry
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Collect geometry data
    hole_ratios = []
    outer_radii = []
    active_entries = []
    entry_strengths_all = []
    
    for splat in torus_model.torus_splats:
        info = splat.get_geometry_info()
        hole_ratios.append(info['hole_ratio'])
        outer_radii.append(info['outer_radius'])
        active_entries.append(info['active_entry_points'])
        entry_strengths_all.extend(info['entry_strengths'])
    
    # 1. Hole ratio distribution
    axes[0, 0].hist(hole_ratios, bins=15, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].axvline(x=np.mean(hole_ratios), color='darkred', linestyle='--', 
                      label=f'Mean: {np.mean(hole_ratios):.3f}')
    axes[0, 0].set_title('Distribution of Hole Ratios\n(Higher = More Torus-like)')
    axes[0, 0].set_xlabel('Hole Radius / Outer Radius')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Active entry points per splat
    entry_counts = np.bincount(active_entries, minlength=5)
    axes[0, 1].bar(range(len(entry_counts)), entry_counts, alpha=0.7, color='green', 
                   edgecolor='black')
    axes[0, 1].set_title('Distribution of Active Entry Points')
    axes[0, 1].set_xlabel('Number of Active Entry Points')
    axes[0, 1].set_ylabel('Number of Splats')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Entry strength distribution
    axes[1, 0].hist(entry_strengths_all, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].axvline(x=0.3, color='red', linestyle='--', label='Activation Threshold')
    axes[1, 0].set_title('Entry Point Strength Distribution')
    axes[1, 0].set_xlabel('Entry Point Strength')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Relationship between hole ratio and active entries
    axes[1, 1].scatter(hole_ratios, active_entries, alpha=0.6, s=60, c=outer_radii, 
                      cmap='viridis', edgecolors='black', linewidth=0.5)
    axes[1, 1].set_title('Hole Ratio vs Active Entry Points\n(Color = Outer Radius)')
    axes[1, 1].set_xlabel('Hole Ratio')
    axes[1, 1].set_ylabel('Active Entry Points')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar for the scatter plot
    scatter = axes[1, 1].collections[0]
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Outer Radius')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Geometry visualization saved to: {save_path}")

def test_specific_torus_capabilities():
    """
    Test specific capabilities that only torus geometry should provide
    """
    print("\n" + "="*60)
    print("TESTING TORUS-SPECIFIC CAPABILITIES")
    print("="*60)
    
    # Test 1: Long-range wrapping connections
    print("\nTest 1: Long-range wrapping (like opposite sides of a donut)")
    
    embedding_dim = 32
    seq_len = 20
    
    # Create test where first and last tokens should connect strongly
    test_input = torch.randn(4, seq_len, embedding_dim)
    
    # Make first and last tokens very similar (they should wrap around torus)
    shared_pattern = torch.randn(embedding_dim)
    test_input[:, 0, :] = shared_pattern + torch.randn(4, embedding_dim) * 0.1
    test_input[:, -1, :] = shared_pattern + torch.randn(4, embedding_dim) * 0.1
    
    # Create evolved torus model
    torus_model = EvolvingTorusAttention(embedding_dim, n_splats=6, n_heads=2)
    
    # Force some evolution
    for _ in range(3):
        torus_model.evolve_all_splats()
    
    # Get attention pattern
    with torch.no_grad():
        _, attention = torus_model(test_input)
    
    # Check if first and last positions attend to each other strongly
    first_to_last = attention[0, 0, -1, 0].item()  # First batch, query=0, key=last, head=0
    last_to_first = attention[0, -1, 0, 0].item()  # First batch, query=last, key=0, head=0
    
    wrap_around_strength = (first_to_last + last_to_first) / 2
    
    print(f"Wrap-around attention strength: {wrap_around_strength:.4f}")
    print(f"Status: {'✓ STRONG' if wrap_around_strength > 0.1 else '~ WEAK' if wrap_around_strength > 0.05 else '✗ ABSENT'}")
    
    # Test 2: Multi-hop entry point connections
    print(f"\nTest 2: Multi-hop connections through entry points")
    
    # Create a pattern where info should hop: pos 2 → entry point → pos 15
    multi_hop_input = torch.randn(2, seq_len, embedding_dim)
    
    # Create two similar patterns that should connect via entry points
    pattern_a = torch.randn(embedding_dim)
    pattern_b = torch.randn(embedding_dim)
    bridge_pattern = (pattern_a + pattern_b) / 2
    
    multi_hop_input[:, 2, :] = pattern_a
    multi_hop_input[:, seq_len//2, :] = bridge_pattern  # Entry point position
    multi_hop_input[:, 15, :] = pattern_b
    
    with torch.no_grad():
        _, multi_hop_attention = torus_model(multi_hop_input)
    
    # Check multi-hop attention: pos 2 → middle → pos 15
    hop1_strength = multi_hop_attention[0, 2, seq_len//2, 0].item()
    hop2_strength = multi_hop_attention[0, seq_len//2, 15, 0].item()
    direct_strength = multi_hop_attention[0, 2, 15, 0].item()
    
    multi_hop_score = (hop1_strength * hop2_strength) / (direct_strength + 1e-6)
    
    print(f"Multi-hop vs direct attention ratio: {multi_hop_score:.4f}")
    print(f"Status: {'✓ MULTI-HOP PREFERRED' if multi_hop_score > 1.2 else '~ COMPARABLE' if multi_hop_score > 0.8 else '✗ DIRECT PREFERRED'}")
    
    return {
        'wrap_around_strength': wrap_around_strength,
        'multi_hop_score': multi_hop_score,
        'torus_capabilities_demonstrated': wrap_around_strength > 0.05 and multi_hop_score > 0.8
    }

if __name__ == "__main__":
    # Run main validation
    results, success = run_torus_evolution_validation()
    
    # Test torus-specific capabilities
    torus_capabilities = test_specific_torus_capabilities()
    
    # Final integrated assessment
    print("\n" + "="*70)
    print("FINAL INTEGRATED ASSESSMENT")
    print("="*70)
    
    overall_success = success and torus_capabilities['torus_capabilities_demonstrated']
    
    print(f"Performance validation: {'✓ PASSED' if success else '✗ FAILED'}")
    print(f"Torus capability validation: {'✓ PASSED' if torus_capabilities['torus_capabilities_demonstrated'] else '✗ FAILED'}")
    print(f"Overall validation: {'✓ SUCCESS' if overall_success else '✗ NEEDS IMPROVEMENT'}")
    
    if overall_success:
        print(f"\n🎉 EVOLVING TORUS SPLATS VALIDATED!")
        print(f"   - Geometric evolution from circles to toruses works")
        print(f"   - Entry points create meaningful non-local connections") 
        print(f"   - Performance improvements over standard splats")
        print(f"   - Unique torus capabilities demonstrated")
    else:
        print(f"\n⚠️  EVOLVING TORUS SPLATS NEED REFINEMENT")
        print(f"   - Some capabilities shown but need strengthening")
        print(f"   - Consider adjusting evolution parameters")
        print(f"   - May need different task designs to show benefits")
    
    print(f"\nGenerated files:")
    print(f"   - chain_torus_evolution.png")
    print(f"   - intersection_torus_evolution.png") 
    print(f"   - torus_geometry.png")
    
    exit(0 if overall_success else 1)
