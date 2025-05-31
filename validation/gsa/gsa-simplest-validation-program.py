"""
Gaussian Splat Attention - Realistic Validation Program

This program validates GSA with tests that align with how attention actually works:
1. Can GSA learn meaningful attention patterns from data?
2. Does GSA work as a drop-in replacement for standard attention?
3. Can GSA discover structure in sequences?

Run this program to validate the GSA mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
from sklearn.decomposition import PCA

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MinimalGSA(nn.Module):
    """Minimal Gaussian Splat Attention implementation"""
    
    def __init__(self, dim: int, n_splats: int = 32, n_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.n_splats = n_splats
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Core GSA parameters with careful initialization
        self.splat_centers = nn.Parameter(torch.randn(n_heads, n_splats, self.head_dim) * 0.5)
        self.splat_log_scales = nn.Parameter(torch.zeros(n_heads, n_splats))
        self.splat_log_amplitudes = nn.Parameter(torch.zeros(n_heads, n_splats))
        
        # Standard attention projections
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        
        # Optional temperature for attention sharpness
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional splat weight return"""
        B, T, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # Each: [B, T, H, head_dim]
        
        # Initialize distance tensors
        q_dists = torch.zeros(B, T, self.n_heads, self.n_splats, device=x.device)
        k_dists = torch.zeros(B, T, self.n_heads, self.n_splats, device=x.device)
        
        # Compute distances per head
        for h in range(self.n_heads):
            q_h = q[:, :, h, :].reshape(B * T, self.head_dim)
            k_h = k[:, :, h, :].reshape(B * T, self.head_dim)
            centers_h = self.splat_centers[h]
            
            q_dists_h = torch.cdist(q_h.unsqueeze(1), centers_h.unsqueeze(0)).squeeze(1)
            k_dists_h = torch.cdist(k_h.unsqueeze(1), centers_h.unsqueeze(0)).squeeze(1)
            
            q_dists[:, :, h, :] = q_dists_h.reshape(B, T, self.n_splats)
            k_dists[:, :, h, :] = k_dists_h.reshape(B, T, self.n_splats)
        
        # Convert to Gaussian weights
        scales = torch.exp(self.splat_log_scales).unsqueeze(0).unsqueeze(0)
        q_weights = torch.exp(-0.5 * (q_dists / (scales + 1e-6)) ** 2)
        k_weights = torch.exp(-0.5 * (k_dists / (scales + 1e-6)) ** 2)
        
        # Store weights for analysis if requested
        if return_weights:
            splat_weights = (q_weights, k_weights)
        
        # Compute attention through splats
        amplitudes = torch.exp(self.splat_log_amplitudes)
        attention_logits = torch.einsum('bihs,bjhs,hs->bijh', 
                                       q_weights, k_weights, amplitudes)
        
        # Apply temperature and softmax
        attention_logits = attention_logits / self.temperature
        attention = F.softmax(attention_logits, dim=2)
        
        # Apply attention to values
        out = torch.einsum('bijh,bjhd->bihd', attention, v)
        out = out.reshape(B, T, D)
        
        if return_weights:
            return self.out(out), attention, splat_weights
        return self.out(out), attention

class StandardAttention(nn.Module):
    """Standard multi-head attention for comparison"""
    
    def __init__(self, dim: int, n_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Standard scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.reshape(B, T, D)
        
        return self.out(out), attention

def create_synthetic_sequence_data(n_samples: int = 100, seq_len: int = 16, dim: int = 32):
    """Create synthetic sequences with learnable patterns"""
    data = []
    labels = []
    
    for _ in range(n_samples):
        # Pattern 1: Copy task - output should copy a specific position
        if np.random.random() < 0.5:
            seq = torch.randn(seq_len, dim) * 0.5
            copy_pos = np.random.randint(0, seq_len)
            seq[copy_pos] += torch.randn(dim)  # Make one position distinctive
            label = copy_pos
        # Pattern 2: Average task - output should average specific positions
        else:
            seq = torch.randn(seq_len, dim) * 0.5
            important_positions = np.random.choice(seq_len, size=3, replace=False)
            for pos in important_positions:
                seq[pos] += torch.randn(dim) * 0.5
            label = important_positions[0]  # Simplified: just predict first important position
            
        data.append(seq)
        labels.append(label)
    
    return torch.stack(data), torch.tensor(labels)

def test_sequence_modeling():
    """Test if GSA can learn to solve sequence tasks"""
    print("\nTEST 1: Sequence Modeling Task")
    print("-"*40)
    
    dim = 32
    seq_len = 16
    n_splats = 16
    
    # Create models
    gsa_model = nn.Sequential(
        MinimalGSA(dim=dim, n_splats=n_splats, n_heads=4),
        nn.Linear(dim, seq_len)  # Predict which position to attend to
    )
    
    std_model = nn.Sequential(
        StandardAttention(dim=dim, n_heads=4),
        nn.Linear(dim, seq_len)
    )
    
    # Create data
    train_data, train_labels = create_synthetic_sequence_data(200, seq_len, dim)
    test_data, test_labels = create_synthetic_sequence_data(50, seq_len, dim)
    
    # Train both models
    models = {'GSA': gsa_model, 'Standard': std_model}
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(50):
            # Forward pass
            if isinstance(model[0], MinimalGSA):
                hidden, _ = model[0](train_data)
            else:
                hidden, _ = model[0](train_data)
            
            # Pool over sequence dimension
            pooled = hidden.mean(dim=1)
            logits = model[1](pooled)
            
            # Compute loss
            loss = F.cross_entropy(logits, train_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            if isinstance(model[0], MinimalGSA):
                hidden, _ = model[0](test_data)
            else:
                hidden, _ = model[0](test_data)
            
            pooled = hidden.mean(dim=1)
            logits = model[1](pooled)
            
            predictions = logits.argmax(dim=-1)
            accuracy = (predictions == test_labels).float().mean().item()
            
        results[name] = accuracy
        print(f"  Test Accuracy: {accuracy:.4f}")
    
    return results['GSA'] > 0.7  # Success if GSA achieves reasonable accuracy

def test_attention_consistency():
    """Test if GSA produces consistent attention patterns"""
    print("\nTEST 2: Attention Consistency")
    print("-"*40)
    
    dim = 64
    seq_len = 8
    n_splats = 16
    
    gsa = MinimalGSA(dim=dim, n_splats=n_splats, n_heads=4)
    
    # Test 1: Same input should produce same attention
    x = torch.randn(1, seq_len, dim)
    _, attn1 = gsa(x)
    _, attn2 = gsa(x)
    
    consistency = torch.allclose(attn1, attn2, rtol=1e-5)
    print(f"  Deterministic: {consistency}")
    
    # Test 2: Similar inputs should produce similar attention
    x1 = torch.randn(1, seq_len, dim)
    x2 = x1 + torch.randn_like(x1) * 0.1  # Small perturbation
    
    _, attn1 = gsa(x1)
    _, attn2 = gsa(x2)
    
    similarity = F.cosine_similarity(
        attn1.flatten().unsqueeze(0),
        attn2.flatten().unsqueeze(0)
    ).item()
    
    print(f"  Stability (cosine similarity): {similarity:.4f}")
    
    return consistency and similarity > 0.9

def test_splat_specialization():
    """Test if splats specialize during training"""
    print("\nTEST 3: Splat Specialization")
    print("-"*40)
    
    dim = 32
    seq_len = 16
    n_splats = 8
    
    # Create a simple task where first half and second half need different attention
    gsa = MinimalGSA(dim=dim, n_splats=n_splats, n_heads=2)
    optimizer = torch.optim.Adam(gsa.parameters(), lr=0.01)
    
    # Track splat amplitudes over training
    amplitude_history = []
    
    for step in range(200):
        # Create data where first half attends locally, second half globally
        x = torch.randn(4, seq_len, dim)
        
        # Create target: reconstruct input with specific attention pattern
        target = x.clone()
        
        output, attention = gsa(x)
        loss = F.mse_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record amplitudes
        if step % 50 == 0:
            amplitudes = torch.exp(gsa.splat_log_amplitudes).detach()
            amplitude_history.append(amplitudes)
            print(f"  Step {step}: Loss = {loss.item():.4f}")
    
    # Analyze specialization
    initial_amplitudes = amplitude_history[0]
    final_amplitudes = amplitude_history[-1]
    
    # Check if some splats have become more important than others
    initial_variance = initial_amplitudes.var().item()
    final_variance = final_amplitudes.var().item()
    
    print(f"\n  Initial amplitude variance: {initial_variance:.6f}")
    print(f"  Final amplitude variance: {final_variance:.6f}")
    
    # Check if any splats have been "pruned" (very low amplitude)
    pruned_splats = (final_amplitudes < 0.01).sum().item()
    print(f"  Splats with amplitude < 0.01: {pruned_splats}/{n_splats * 2}")
    
    return final_variance > initial_variance * 2  # Specialization increases variance

def visualize_splat_analysis(gsa: MinimalGSA, save_path: str = "splat_analysis.png"):
    """Comprehensive visualization of learned splats"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Get parameters
    centers = gsa.splat_centers.detach().cpu().numpy()
    scales = torch.exp(gsa.splat_log_scales).detach().cpu().numpy()
    amplitudes = torch.exp(gsa.splat_log_amplitudes).detach().cpu().numpy()
    
    # 1. PCA projection of splat centers
    ax = axes[0, 0]
    all_centers = centers.reshape(-1, centers.shape[-1])
    if all_centers.shape[1] > 2:
        pca = PCA(n_components=2)
        centers_2d = pca.fit_transform(all_centers)
    else:
        centers_2d = all_centers[:, :2]
    
    for h in range(min(4, gsa.n_heads)):
        start = h * gsa.n_splats
        end = (h + 1) * gsa.n_splats
        ax.scatter(centers_2d[start:end, 0], centers_2d[start:end, 1],
                  s=amplitudes[h] * 200, alpha=0.6, label=f'Head {h}')
    ax.set_title('Splat Centers (PCA)')
    ax.legend()
    
    # 2. Amplitude distribution
    ax = axes[0, 1]
    ax.hist(amplitudes.flatten(), bins=20, alpha=0.7, color='blue')
    ax.axvline(x=amplitudes.mean(), color='red', linestyle='--', label='Mean')
    ax.set_title('Amplitude Distribution')
    ax.set_xlabel('Amplitude')
    ax.legend()
    
    # 3. Scale distribution
    ax = axes[0, 2]
    ax.hist(scales.flatten(), bins=20, alpha=0.7, color='green')
    ax.set_title('Scale Distribution')
    ax.set_xlabel('Scale (σ)')
    
    # 4. Amplitude vs Scale
    ax = axes[1, 0]
    ax.scatter(scales.flatten(), amplitudes.flatten(), alpha=0.5)
    ax.set_xlabel('Scale (σ)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Amplitude vs Scale')
    
    # 5. Splat utilization by head
    ax = axes[1, 1]
    head_labels = [f'Head {h}' for h in range(gsa.n_heads)]
    ax.bar(head_labels[:4], amplitudes[:4].mean(axis=1))
    ax.set_title('Average Amplitude by Head')
    ax.set_ylabel('Average Amplitude')
    
    # 6. Temperature value
    ax = axes[1, 2]
    temp_value = gsa.temperature.item()
    ax.text(0.5, 0.5, f'Temperature: {temp_value:.3f}', 
            ha='center', va='center', fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Learned Temperature')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_validation_suite():
    """Run the complete validation suite"""
    print("="*60)
    print("GSA VALIDATION SUITE (Realistic Tests)")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test sequence modeling
    sequence_passed = test_sequence_modeling()
    
    # Test attention consistency
    consistency_passed = test_attention_consistency()
    
    # Test splat specialization
    specialization_passed = test_splat_specialization()
    
    # Create a sample GSA for visualization
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    # Train a GSA on a simple task for visualization
    dim = 64
    gsa = MinimalGSA(dim=dim, n_splats=32, n_heads=8)
    
    # Simple training to encourage splat differentiation
    optimizer = torch.optim.Adam(gsa.parameters(), lr=0.01)
    print("Training GSA for visualization...")
    
    for step in range(100):
        x = torch.randn(8, 16, dim)
        output, _ = gsa(x)
        loss = F.mse_loss(output, x)  # Reconstruction task
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    visualize_splat_analysis(gsa)
    print("Visualization saved to: splat_analysis.png")
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    results = {
        'sequence_modeling': sequence_passed,
        'attention_consistency': consistency_passed,
        'splat_specialization': specialization_passed
    }
    
    for test, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOVERALL: {'✓ VALIDATION PASSED' if all_passed else '✗ VALIDATION FAILED'}")
    
    # Save results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'overall_success': all_passed
    }
    
    with open('gsa_validation_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    return all_passed

if __name__ == "__main__":
    success = run_validation_suite()
    exit(0 if success else 1)
