"""
Toroidal Splat Attention - Validation Script

This script tests whether toroidal splats with "wormhole" connections
can outperform standard Gaussian splats for attention mechanisms.

Key Concepts:
- TOROIDAL SPLAT: A splat that wraps around in embedding space, creating
  a torus topology that can connect distant regions
- WORMHOLE: A learned connection that links two points on the torus,
  enabling non-local attention patterns
- VALIDATION: Test on tasks requiring long-range dependencies
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

class ToroidalSplat(nn.Module):
    """
    A single toroidal splat with wormhole connections.
    
    The torus is parameterized by:
    - center: Primary position in embedding space
    - wormhole_exit: Secondary position (the "other side" of the torus)
    - major_radius: Size of the main torus
    - minor_radius: Thickness of the torus tube
    - wormhole_strength: How much attention flows through the wormhole
    """
    
    def __init__(self, embedding_dim: int, splat_id: int = 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.splat_id = splat_id
        
        # Primary torus parameters
        self.center = nn.Parameter(torch.randn(embedding_dim))
        self.major_radius = nn.Parameter(torch.tensor(1.0))  # Main torus size
        self.minor_radius = nn.Parameter(torch.tensor(0.5))  # Tube thickness
        
        # Wormhole parameters
        self.wormhole_exit = nn.Parameter(torch.randn(embedding_dim))
        self.wormhole_strength = nn.Parameter(torch.tensor(0.5))
        
        # Attention amplitude
        self.amplitude = nn.Parameter(torch.tensor(1.0))
        
        # Learnable mixing weights
        self.torus_weight = nn.Parameter(torch.tensor(0.7))
        self.wormhole_weight = nn.Parameter(torch.tensor(0.3))
    
    def compute_torus_distance(self, point: torch.Tensor) -> torch.Tensor:
        """
        Compute distance to the torus surface.
        
        For a torus centered at 'center' with major radius R and minor radius r,
        the distance from a point to the torus surface is computed using
        toroidal coordinates.
        """
        # Translate point relative to torus center
        relative_point = point - self.center
        
        # For simplicity, we'll use a 2D projection for the torus calculation
        # In practice, you might want to use full 3D toroidal coordinates
        
        # Project to first two dimensions for torus calculation
        if self.embedding_dim >= 2:
            x = relative_point[0]
            y = relative_point[1]
            
            # Distance from origin in XY plane
            xy_dist = torch.sqrt(x**2 + y**2 + 1e-8)
            
            # Distance to major radius circle
            major_circle_dist = torch.abs(xy_dist - torch.abs(self.major_radius))
            
            # Add contribution from other dimensions (treating as minor radius variations)
            if self.embedding_dim > 2:
                z_contribution = torch.norm(relative_point[2:])
                minor_dist = torch.sqrt(major_circle_dist**2 + z_contribution**2 + 1e-8)
            else:
                minor_dist = major_circle_dist
            
            # Distance to torus surface
            torus_dist = torch.abs(minor_dist - torch.abs(self.minor_radius))
        else:
            # Fallback for 1D case
            torus_dist = torch.abs(torch.norm(relative_point) - torch.abs(self.major_radius))
        
        return torus_dist
    
    def compute_wormhole_affinity(self, point: torch.Tensor) -> torch.Tensor:
        """
        Compute affinity through the wormhole connection.
        
        This creates a "tunnel" between the main torus and the wormhole exit.
        """
        # Distance to main center
        center_dist = torch.norm(point - self.center)
        center_affinity = torch.exp(-0.5 * center_dist**2)
        
        # Distance to wormhole exit
        wormhole_dist = torch.norm(point - self.wormhole_exit)
        wormhole_affinity = torch.exp(-0.5 * wormhole_dist**2)
        
        # Wormhole creates a connection between these two regions
        wormhole_connection = center_affinity * wormhole_affinity * torch.sigmoid(self.wormhole_strength)
        
        return wormhole_connection
    
    def forward(self, point: torch.Tensor) -> torch.Tensor:
        """
        Compute the attention weight for a given point.
        
        Combines torus-based attention with wormhole connections.
        """
        # Torus-based attention (inverse distance to torus surface)
        torus_dist = self.compute_torus_distance(point)
        torus_attention = torch.exp(-torus_dist**2 / (2 * torch.abs(self.minor_radius)**2 + 1e-8))
        
        # Wormhole-based attention
        wormhole_attention = self.compute_wormhole_affinity(point)
        
        # Combine torus and wormhole attention
        combined_attention = (
            torch.sigmoid(self.torus_weight) * torus_attention +
            torch.sigmoid(self.wormhole_weight) * wormhole_attention
        )
        
        return torch.sigmoid(self.amplitude) * combined_attention

class StandardGaussianSplat(nn.Module):
    """Standard Gaussian splat for comparison"""
    
    def __init__(self, embedding_dim: int, splat_id: int = 0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.splat_id = splat_id
        
        self.center = nn.Parameter(torch.randn(embedding_dim))
        self.log_scale = nn.Parameter(torch.tensor(0.0))
        self.amplitude = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, point: torch.Tensor) -> torch.Tensor:
        distance = torch.norm(point - self.center)
        scale = torch.exp(self.log_scale)
        return torch.sigmoid(self.amplitude) * torch.exp(-0.5 * (distance / (scale + 1e-8))**2)

class ToroidalAttentionLayer(nn.Module):
    """
    Attention layer using toroidal splats
    """
    
    def __init__(self, embedding_dim: int, n_splats: int = 16, n_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_splats = n_splats
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        # Create toroidal splats
        self.toroidal_splats = nn.ModuleList([
            ToroidalSplat(self.head_dim, i) for i in range(n_splats * n_heads)
        ])
        
        # Standard projections
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Compute attention through toroidal splats
        attention_weights = torch.zeros(B, T, T, self.n_heads, device=x.device)
        
        for i in range(T):  # Query positions
            for j in range(T):  # Key positions
                for head in range(self.n_heads):
                    head_attention = 0.0
                    
                    # Sum over all splats for this head
                    for splat_idx in range(self.n_splats):
                        splat = self.toroidal_splats[head * self.n_splats + splat_idx]
                        
                        # Query attention through splat
                        q_attention = splat(q[:, i, head, :].mean(0))  # Average over batch
                        # Key attention through splat
                        k_attention = splat(k[:, j, head, :].mean(0))  # Average over batch
                        
                        # Combined attention
                        head_attention += q_attention * k_attention
                    
                    attention_weights[:, i, j, head] = head_attention
        
        # Normalize attention
        attention_weights = F.softmax(attention_weights, dim=2)
        
        # Apply attention to values
        output = torch.einsum('btsh,bshd->bthd', attention_weights, v)
        output = output.reshape(B, T, D)
        
        return self.out(output), attention_weights

class StandardSplatAttentionLayer(nn.Module):
    """Standard splat attention for comparison"""
    
    def __init__(self, embedding_dim: int, n_splats: int = 16, n_heads: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_splats = n_splats
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        
        # Create standard Gaussian splats
        self.splats = nn.ModuleList([
            StandardGaussianSplat(self.head_dim, i) for i in range(n_splats * n_heads)
        ])
        
        # Standard projections
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim)
        self.out = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Compute attention through standard splats
        attention_weights = torch.zeros(B, T, T, self.n_heads, device=x.device)
        
        for i in range(T):
            for j in range(T):
                for head in range(self.n_heads):
                    head_attention = 0.0
                    
                    for splat_idx in range(self.n_splats):
                        splat = self.splats[head * self.n_splats + splat_idx]
                        
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

def create_long_range_dependency_task(seq_len: int = 32, batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a task that requires long-range dependencies.
    
    Task: Copy tokens from the beginning to the end, with noise in between.
    This tests whether the attention mechanism can create "wormhole" connections
    across long sequences.
    """
    # Create sequences with special tokens at start and end positions
    sequences = torch.randint(1, 10, (batch_size, seq_len))
    targets = sequences.clone()
    
    # Put special patterns at the beginning
    sequences[:, 0] = 42  # Special start token
    sequences[:, 1] = torch.arange(batch_size) + 100  # Unique identifier per sequence
    
    # The task is to copy the identifier to the end
    targets[:, -1] = sequences[:, 1]
    
    return sequences.float(), targets

def create_wormhole_pattern_task(seq_len: int = 24) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a task specifically designed to test wormhole connections.
    
    Task: Connect specific positions that should attend to each other
    despite being far apart (like opposite sides of a torus).
    """
    batch_size = 4
    sequences = torch.randn(batch_size, seq_len, 32)  # 32-dim embeddings
    
    # Create "wormhole pairs" - positions that should attend to each other
    wormhole_pairs = [(2, seq_len-3), (5, seq_len-6), (8, seq_len-9)]
    
    targets = sequences.clone()
    
    # The target is that wormhole pairs should have similar representations
    for pos1, pos2 in wormhole_pairs:
        # Target: average of the pair
        avg_repr = (sequences[:, pos1] + sequences[:, pos2]) / 2
        targets[:, pos1] = avg_repr
        targets[:, pos2] = avg_repr
    
    return sequences, targets

def train_and_compare_models(task_name: str, 
                           train_data: Tuple[torch.Tensor, torch.Tensor],
                           test_data: Tuple[torch.Tensor, torch.Tensor],
                           n_epochs: int = 100) -> Dict:
    """
    Train both toroidal and standard splat models and compare performance.
    """
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    seq_len = x_train.shape[1]
    embedding_dim = x_train.shape[2] if len(x_train.shape) > 2 else 64
    
    print(f"\n{'='*50}")
    print(f"TRAINING TASK: {task_name}")
    print(f"{'='*50}")
    print(f"Sequence length: {seq_len}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Training samples: {x_train.shape[0]}")
    
    # Create models
    toroidal_model = ToroidalAttentionLayer(embedding_dim, n_splats=8, n_heads=4)
    standard_model = StandardSplatAttentionLayer(embedding_dim, n_splats=8, n_heads=4)
    
    # Optimizers
    toroidal_optimizer = torch.optim.Adam(toroidal_model.parameters(), lr=0.001)
    standard_optimizer = torch.optim.Adam(standard_model.parameters(), lr=0.001)
    
    # Training loop
    toroidal_losses = []
    standard_losses = []
    
    for epoch in range(n_epochs):
        # Train toroidal model
        toroidal_model.train()
        toroidal_out, _ = toroidal_model(x_train)
        toroidal_loss = F.mse_loss(toroidal_out, y_train)
        
        toroidal_optimizer.zero_grad()
        toroidal_loss.backward()
        toroidal_optimizer.step()
        toroidal_losses.append(toroidal_loss.item())
        
        # Train standard model
        standard_model.train()
        standard_out, _ = standard_model(x_train)
        standard_loss = F.mse_loss(standard_out, y_train)
        
        standard_optimizer.zero_grad()
        standard_loss.backward()
        standard_optimizer.step()
        standard_losses.append(standard_loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Toroidal Loss = {toroidal_loss.item():.6f}, "
                  f"Standard Loss = {standard_loss.item():.6f}")
    
    # Evaluate on test data
    toroidal_model.eval()
    standard_model.eval()
    
    with torch.no_grad():
        toroidal_test_out, toroidal_test_attn = toroidal_model(x_test)
        standard_test_out, standard_test_attn = standard_model(x_test)
        
        toroidal_test_loss = F.mse_loss(toroidal_test_out, y_test).item()
        standard_test_loss = F.mse_loss(standard_test_out, y_test).item()
    
    print(f"\nFINAL TEST RESULTS:")
    print(f"Toroidal Test Loss: {toroidal_test_loss:.6f}")
    print(f"Standard Test Loss: {standard_test_loss:.6f}")
    print(f"Improvement: {((standard_test_loss - toroidal_test_loss) / standard_test_loss * 100):.2f}%")
    
    return {
        'task_name': task_name,
        'toroidal_train_losses': toroidal_losses,
        'standard_train_losses': standard_losses,
        'toroidal_test_loss': toroidal_test_loss,
        'standard_test_loss': standard_test_loss,
        'toroidal_attention': toroidal_test_attn.cpu().numpy(),
        'standard_attention': standard_test_attn.cpu().numpy(),
        'improvement_percent': (standard_test_loss - toroidal_test_loss) / standard_test_loss * 100
    }

def visualize_attention_patterns(results: Dict, save_path: str = "toroidal_attention_comparison.png"):
    """Visualize the attention patterns learned by both models"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    toroidal_attn = results['toroidal_attention'][0, :, :, 0]  # First batch, first head
    standard_attn = results['standard_attention'][0, :, :, 0]
    
    # Plot attention matrices
    im1 = axes[0, 0].imshow(toroidal_attn, cmap='Blues', aspect='auto')
    axes[0, 0].set_title('Toroidal Attention Pattern')
    axes[0, 0].set_xlabel('Key Position')
    axes[0, 0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[1, 0].imshow(standard_attn, cmap='Blues', aspect='auto')
    axes[1, 0].set_title('Standard Splat Attention Pattern')
    axes[1, 0].set_xlabel('Key Position')
    axes[1, 0].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1, 0])
    
    # Plot training losses
    axes[0, 1].plot(results['toroidal_train_losses'], label='Toroidal', color='red')
    axes[0, 1].plot(results['standard_train_losses'], label='Standard', color='blue')
    axes[0, 1].set_title('Training Loss Comparison')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # Plot difference in attention patterns
    attention_diff = toroidal_attn - standard_attn
    im3 = axes[1, 1].imshow(attention_diff, cmap='RdBu', aspect='auto', vmin=-0.1, vmax=0.1)
    axes[1, 1].set_title('Attention Pattern Difference\n(Toroidal - Standard)')
    axes[1, 1].set_xlabel('Key Position')
    axes[1, 1].set_ylabel('Query Position')
    plt.colorbar(im3, ax=axes[1, 1])
    
    # Performance comparison bar chart
    test_losses = [results['toroidal_test_loss'], results['standard_test_loss']]
    model_names = ['Toroidal', 'Standard']
    bars = axes[0, 2].bar(model_names, test_losses, color=['red', 'blue'], alpha=0.7)
    axes[0, 2].set_title('Test Loss Comparison')
    axes[0, 2].set_ylabel('Test Loss')
    
    # Add improvement percentage text
    improvement = results['improvement_percent']
    axes[1, 2].text(0.5, 0.5, f"Improvement:\n{improvement:.2f}%", 
                    transform=axes[1, 2].transAxes, fontsize=16, 
                    ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightgreen' if improvement > 0 else 'lightcoral'))
    axes[1, 2].set_title('Performance Improvement')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def analyze_wormhole_effectiveness(toroidal_model: ToroidalAttentionLayer) -> Dict:
    """
    Analyze how well the wormholes are working in the toroidal model.
    """
    results = {}
    
    # Extract splat parameters
    with torch.no_grad():
        for i, splat in enumerate(toroidal_model.toroidal_splats):
            results[f'splat_{i}'] = {
                'center': splat.center.cpu().numpy(),
                'wormhole_exit': splat.wormhole_exit.cpu().numpy(),
                'wormhole_strength': torch.sigmoid(splat.wormhole_strength).item(),
                'major_radius': torch.abs(splat.major_radius).item(),
                'minor_radius': torch.abs(splat.minor_radius).item(),
                'torus_weight': torch.sigmoid(splat.torus_weight).item(),
                'wormhole_weight': torch.sigmoid(splat.wormhole_weight).item()
            }
    
    # Compute statistics
    wormhole_strengths = [results[k]['wormhole_strength'] for k in results.keys()]
    torus_weights = [results[k]['torus_weight'] for k in results.keys()]
    
    print(f"\nWORMHOLE ANALYSIS:")
    print(f"Average wormhole strength: {np.mean(wormhole_strengths):.3f} ± {np.std(wormhole_strengths):.3f}")
    print(f"Average torus weight: {np.mean(torus_weights):.3f} ± {np.std(torus_weights):.3f}")
    print(f"Number of strong wormholes (>0.5): {sum(1 for s in wormhole_strengths if s > 0.5)}")
    
    return results

def run_validation_suite():
    """
    Run the complete toroidal splat validation suite.
    """
    print("="*60)
    print("TOROIDAL SPLAT ATTENTION VALIDATION")
    print("="*60)
    
    results_summary = {}
    
    # Test 1: Long-range dependency task
    print("\nGenerating long-range dependency task...")
    long_range_train = create_long_range_dependency_task(seq_len=32, batch_size=16)
    long_range_test = create_long_range_dependency_task(seq_len=32, batch_size=8)
    
    # Convert to proper embedding format
    x_train, y_train = long_range_train
    x_test, y_test = long_range_test
    
    # Embed into higher dimensional space
    embedding_dim = 64
    x_train_embed = torch.randn(x_train.shape[0], x_train.shape[1], embedding_dim)
    y_train_embed = torch.randn(y_train.shape[0], y_train.shape[1], embedding_dim)
    x_test_embed = torch.randn(x_test.shape[0], x_test.shape[1], embedding_dim)
    y_test_embed = torch.randn(y_test.shape[0], y_test.shape[1], embedding_dim)
    
    # Encode the original sequences into the embeddings
    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[1]):
            x_train_embed[i, j] *= x_train[i, j] / 10.0  # Scale by token value
            y_train_embed[i, j] *= y_train[i, j] / 10.0
    
    for i in range(x_test.shape[0]):
        for j in range(x_test.shape[1]):
            x_test_embed[i, j] *= x_test[i, j] / 10.0
            y_test_embed[i, j] *= y_test[i, j] / 10.0
    
    results1 = train_and_compare_models(
        "Long-Range Dependencies",
        (x_train_embed, y_train_embed),
        (x_test_embed, y_test_embed),
        n_epochs=150
    )
    results_summary['long_range'] = results1
    
    # Test 2: Wormhole pattern task
    print("\nGenerating wormhole pattern task...")
    wormhole_train_data = [create_wormhole_pattern_task(seq_len=24) for _ in range(8)]
    wormhole_test_data = [create_wormhole_pattern_task(seq_len=24) for _ in range(4)]
    
    # Combine batches
    x_wh_train = torch.cat([d[0] for d in wormhole_train_data], dim=0)
    y_wh_train = torch.cat([d[1] for d in wormhole_train_data], dim=0)
    x_wh_test = torch.cat([d[0] for d in wormhole_test_data], dim=0)
    y_wh_test = torch.cat([d[1] for d in wormhole_test_data], dim=0)
    
    results2 = train_and_compare_models(
        "Wormhole Connections",
        (x_wh_train, y_wh_train),
        (x_wh_test, y_wh_test),
        n_epochs=200
    )
    results_summary['wormhole'] = results2
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_attention_patterns(results1, "long_range_attention_comparison.png")
    visualize_attention_patterns(results2, "wormhole_attention_comparison.png")
    
    # Analyze the best performing toroidal model
    if results1['improvement_percent'] > results2['improvement_percent']:
        best_model_data = (x_train_embed, y_train_embed)
        best_task = "Long-Range Dependencies"
    else:
        best_model_data = (x_wh_train, y_wh_train)
        best_task = "Wormhole Connections"
    
    # Retrain the best model for analysis
    print(f"\nRetraining best model ({best_task}) for detailed analysis...")
    toroidal_model = ToroidalAttentionLayer(best_model_data[0].shape[2], n_splats=8, n_heads=4)
    optimizer = torch.optim.Adam(toroidal_model.parameters(), lr=0.001)
    
    for epoch in range(100):
        optimizer.zero_grad()
        out, _ = toroidal_model(best_model_data[0])
        loss = F.mse_loss(out, best_model_data[1])
        loss.backward()
        optimizer.step()
    
    wormhole_analysis = analyze_wormhole_effectiveness(toroidal_model)
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for task_name, results in results_summary.items():
        improvement = results['improvement_percent']
        print(f"\n{task_name.upper()}:")
        print(f"  Toroidal Test Loss: {results['toroidal_test_loss']:.6f}")
        print(f"  Standard Test Loss: {results['standard_test_loss']:.6f}")
        print(f"  Improvement: {improvement:.2f}%")
        print(f"  Status: {'✓ BETTER' if improvement > 5 else '~ SIMILAR' if improvement > -5 else '✗ WORSE'}")
    
    # Overall assessment
    avg_improvement = np.mean([r['improvement_percent'] for r in results_summary.values()])
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Average improvement: {avg_improvement:.2f}%")
    
    if avg_improvement > 10:
        verdict = "✓ TOROIDAL SPLATS SHOW SIGNIFICANT PROMISE"
    elif avg_improvement > 0:
        verdict = "~ TOROIDAL SPLATS SHOW MODEST IMPROVEMENT"
    else:
        verdict = "✗ TOROIDAL SPLATS NEED MORE DEVELOPMENT"
    
    print(f"Verdict: {verdict}")
    
    return results_summary, avg_improvement > 0

if __name__ == "__main__":
    success = run_validation_suite()
    print(f"\nValidation {'PASSED' if success[1] else 'NEEDS WORK'}")
