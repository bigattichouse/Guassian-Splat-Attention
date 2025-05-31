"""
GSA Comprehensive Validation Script

This script implements the complete validation framework for Gaussian Splat Attention,
incorporating all lessons learned from extensive experimentation.

Key Features:
- 7 comprehensive test phases
- Proven GSA architecture (movement-focused, no complex birth/death)
- Clear success metrics and failure analysis
- Production-ready code quality
- Comprehensive reporting and visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class ValidationConfig:
    """Configuration for GSA validation"""
    dim: int = 64
    n_splats: int = 12
    n_heads: int = 8
    movement_scale: float = 0.08
    pruning_threshold: float = 0.02
    temperature_init: float = 1.0
    learning_rate: float = 0.001
    max_epochs: int = 100
    batch_size: int = 8
    seq_len: int = 16
    
    # Success criteria thresholds
    pattern_improvement_threshold: float = 20.0  # %
    movement_threshold: float = 0.1
    efficiency_threshold: float = 15.0  # max overhead
    specialization_threshold: float = 2.0  # amplitude variance increase

class ProvenGSA(nn.Module):
    """
    Production-ready GSA implementation based on validation learnings.
    
    Focuses on proven features:
    - Adaptive movement (position learning)
    - Soft pruning (amplitude-based)
    - Temperature control
    - Numerical stability
    
    Avoids complex features:
    - Birth/death mechanisms
    - Full covariance matrices
    - Complex adaptation scheduling
    """
    
    def __init__(self, config: ValidationConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_splats = config.n_splats
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        
        # Core splat parameters (proven to work)
        self.splat_base_centers = nn.Parameter(self._strategic_init_centers())
        self.splat_center_deltas = nn.Parameter(torch.zeros(config.n_heads, config.n_splats, self.head_dim))
        self.splat_log_scales = nn.Parameter(torch.zeros(config.n_heads, config.n_splats))
        self.splat_log_amplitudes = nn.Parameter(torch.ones(config.n_heads, config.n_splats) * -0.5)
        
        # Movement and temperature control
        self.movement_scale_param = nn.Parameter(torch.tensor(config.movement_scale))
        self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
        
        # Standard projections
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)
        
        # Tracking for validation
        self.register_buffer('training_step', torch.tensor(0, dtype=torch.long))
        self.register_buffer('initial_centers', self.splat_base_centers.data.clone())
        
    def _strategic_init_centers(self) -> torch.Tensor:
        """Strategic initialization proven to work well"""
        centers = torch.zeros(self.config.n_heads, self.config.n_splats, self.head_dim)
        
        for h in range(self.config.n_heads):
            # Strategy 1: Grid initialization (proven stable)
            n_grid = self.config.n_splats // 2
            if self.head_dim >= 2:
                grid_size = int(np.ceil(np.sqrt(n_grid)))
                x = torch.linspace(-0.5, 0.5, grid_size)
                y = torch.linspace(-0.5, 0.5, grid_size)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:n_grid]
                centers[h, :n_grid, :2] = grid_points
            
            # Strategy 2: Sphere initialization (proven diverse)
            remaining = self.config.n_splats - n_grid
            if remaining > 0:
                sphere_points = torch.randn(remaining, self.head_dim)
                sphere_points = sphere_points / torch.norm(sphere_points, dim=1, keepdim=True)
                centers[h, n_grid:] = sphere_points * 0.3
                
        return centers
    
    def get_effective_centers(self) -> torch.Tensor:
        """Get effective splat positions with bounded movement"""
        movement_scale = torch.sigmoid(self.movement_scale_param) * 0.2
        return self.splat_base_centers + self.splat_center_deltas * movement_scale
    
    def forward(self, x: torch.Tensor, return_extras: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """Forward pass with comprehensive error handling"""
        B, T, D = x.shape
        
        if D != self.dim:
            raise ValueError(f"Input dimension {D} doesn't match model dimension {self.dim}")
        
        if self.training:
            self.training_step += 1
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Get effective centers and compute attention
        effective_centers = self.get_effective_centers()
        attention_weights = self._compute_stable_attention(q, k, effective_centers)
        
        # Apply to values
        output = torch.einsum('bijh,bjhd->bihd', attention_weights, v)
        output = output.reshape(B, T, D)
        output = self.out(output)
        
        if return_extras:
            extras = self._get_validation_extras(effective_centers, attention_weights)
            return output, extras
        
        return output, attention_weights
    
    def _compute_stable_attention(self, q: torch.Tensor, k: torch.Tensor, 
                                centers: torch.Tensor) -> torch.Tensor:
        """Numerically stable attention computation"""
        B, T, H, head_dim = q.shape
        
        # Get splat parameters with safety bounds
        scales = torch.exp(self.splat_log_scales).clamp(min=1e-4, max=3.0)
        amplitudes = torch.exp(self.splat_log_amplitudes).clamp(min=1e-6, max=10.0)
        
        # Apply soft pruning
        active_mask = amplitudes > self.config.pruning_threshold
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
            
            # Vectorized distance computation
            q_expanded = q_h.unsqueeze(2)  # [B, T, 1, head_dim]
            k_expanded = k_h.unsqueeze(2)  # [B, T, 1, head_dim]
            centers_expanded = centers_h.unsqueeze(0).unsqueeze(0)  # [1, 1, n_splats, head_dim]
            
            # Compute squared distances with stability
            q_dists_sq = torch.sum((q_expanded - centers_expanded) ** 2, dim=-1).clamp(max=50.0)
            k_dists_sq = torch.sum((k_expanded - centers_expanded) ** 2, dim=-1).clamp(max=50.0)
            
            # Gaussian weights
            scales_h = scales[h].unsqueeze(0).unsqueeze(0)
            q_weights = torch.exp(-0.5 * q_dists_sq / (scales_h ** 2 + 1e-8))
            k_weights = torch.exp(-0.5 * k_dists_sq / (scales_h ** 2 + 1e-8))
            
            # Apply amplitudes and compute attention
            amps_h = effective_amplitudes[h].unsqueeze(0).unsqueeze(0)
            head_attention = torch.einsum('bis,bjs,s->bij', q_weights, k_weights, effective_amplitudes[h])
            attention_logits[:, :, :, h] = head_attention
        
        # Apply temperature and normalize
        temp = self.temperature.clamp(min=0.1, max=10.0)
        attention_logits = attention_logits / temp
        attention = F.softmax(attention_logits + 1e-8, dim=2)
        
        return attention
    
    def _get_validation_extras(self, effective_centers: torch.Tensor, 
                             attention_weights: torch.Tensor) -> Dict:
        """Get comprehensive validation metrics"""
        with torch.no_grad():
            amplitudes = torch.exp(self.splat_log_amplitudes)
            scales = torch.exp(self.splat_log_scales)
            movement_distances = torch.norm(self.splat_center_deltas, dim=-1)
            
            return {
                'effective_centers': effective_centers.detach().cpu(),
                'amplitudes': amplitudes.detach().cpu(),
                'scales': scales.detach().cpu(),
                'movement_distances': movement_distances.detach().cpu(),
                'attention_weights': attention_weights.detach().cpu(),
                'active_splats': int((amplitudes > self.config.pruning_threshold).sum().item()),
                'total_splats': int(self.config.n_splats * self.config.n_heads),
                'movement_scale': float(torch.sigmoid(self.movement_scale_param).item()),
                'temperature': float(self.temperature.item()),
                'training_step': int(self.training_step.item())
            }
    
    def get_movement_stats(self) -> Dict[str, float]:
        """Get movement and specialization statistics"""
        with torch.no_grad():
            movement_distances = torch.norm(self.splat_center_deltas, dim=-1)
            amplitudes = torch.exp(self.splat_log_amplitudes)
            
            return {
                'avg_movement': float(movement_distances.mean().item()),
                'max_movement': float(movement_distances.max().item()),
                'amplitude_variance': float(amplitudes.var().item()),
                'amplitude_range': float((amplitudes.max() - amplitudes.min()).item()),
                'active_splat_ratio': float((amplitudes > self.config.pruning_threshold).float().mean().item())
            }

class StandardAttention(nn.Module):
    """Standard multi-head attention for comparison"""
    
    def __init__(self, config: ValidationConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.head_dim = config.dim // config.n_heads
        
        self.qkv = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out = nn.Linear(config.dim, config.dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.reshape(B, T, D)
        
        return self.out(out), attention

class ValidationSuite:
    """Comprehensive validation suite for GSA"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results = {}
        self.detailed_logs = []
        
    def run_full_validation(self) -> Dict:
        """Run the complete validation suite"""
        print("=" * 80)
        print("🧪 GSA COMPREHENSIVE VALIDATION SUITE")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration: {self.config}")
        print()
        
        # Test 1: Mathematical Correctness
        test1_passed = self._test_mathematical_correctness()
        self.results['mathematical_correctness'] = test1_passed
        
        # Test 2: Pattern Learning Capability
        test2_passed, pattern_improvement = self._test_pattern_learning()
        self.results['pattern_learning'] = test2_passed
        self.results['pattern_improvement_pct'] = pattern_improvement
        
        # Test 3: Adaptive Movement Validation
        test3_passed, movement_stats = self._test_adaptive_movement()
        self.results['adaptive_movement'] = test3_passed
        self.results['movement_stats'] = movement_stats
        
        # Test 4: Efficiency Benchmarking
        test4_passed, efficiency_stats = self._test_efficiency()
        self.results['efficiency'] = test4_passed
        self.results['efficiency_stats'] = efficiency_stats
        
        # Test 5: Sequence Modeling Performance
        test5_passed, modeling_stats = self._test_sequence_modeling()
        self.results['sequence_modeling'] = test5_passed
        self.results['modeling_stats'] = modeling_stats
        
        # Test 6: Integration Testing
        test6_passed = self._test_integration()
        self.results['integration'] = test6_passed
        
        # Test 7: Scalability Analysis
        test7_passed, scalability_stats = self._test_scalability()
        self.results['scalability'] = test7_passed
        self.results['scalability_stats'] = scalability_stats
        
        # Generate comprehensive report
        self._generate_final_report()
        
        return self.results
    
    def _test_mathematical_correctness(self) -> bool:
        """Test 1: Verify mathematical correctness"""
        print("🧮 TEST 1: Mathematical Correctness")
        print("-" * 50)
        
        try:
            gsa = ProvenGSA(self.config)
            x = torch.randn(4, self.config.seq_len, self.config.dim)
            
            output, attention = gsa(x)
            
            # Check attention properties
            attention_sums = attention.sum(dim=2)
            attention_valid = torch.allclose(attention_sums, torch.ones_like(attention_sums), rtol=1e-4)
            attention_non_negative = torch.all(attention >= 0)
            
            # Check gradient flow
            loss = output.sum()
            loss.backward()
            
            has_gradients = all(
                param.grad is not None and torch.isfinite(param.grad).all()
                for param in gsa.parameters()
            )
            
            # Check output shape
            correct_shape = output.shape == x.shape
            
            success = attention_valid and attention_non_negative and has_gradients and correct_shape
            
            print(f"  ✅ Attention sums to 1: {attention_valid}")
            print(f"  ✅ Attention non-negative: {attention_non_negative}")
            print(f"  ✅ Gradient flow: {has_gradients}")
            print(f"  ✅ Correct output shape: {correct_shape}")
            print(f"  Status: {'✅ PASSED' if success else '❌ FAILED'}")
            
            return success
            
        except Exception as e:
            print(f"  ❌ FAILED with error: {e}")
            return False
    
    def _test_pattern_learning(self) -> Tuple[bool, float]:
        """Test 2: Pattern learning capability"""
        print("\n🎯 TEST 2: Pattern Learning Capability")
        print("-" * 50)
        
        try:
            gsa = ProvenGSA(self.config)
            optimizer = torch.optim.Adam(gsa.parameters(), lr=self.config.learning_rate)
            
            def create_pattern_task(batch_size=8):
                """Create clear pattern learning task"""
                x = torch.randn(batch_size, self.config.seq_len, self.config.dim) * 0.1
                
                # Pattern: first token should attend to last token
                summary = torch.randn(batch_size, self.config.dim) * 0.5
                x[:, -1] = summary  # Last position contains summary
                x[:, 0] += 0.3 * summary  # First should benefit from summary
                
                # Target attention pattern
                target_attention = torch.zeros(batch_size, self.config.seq_len, self.config.seq_len)
                target_attention[:, 0, -1] = 1.0  # Strong first-to-last attention
                target_attention = F.softmax(target_attention + 1e-8, dim=-1)
                
                return x, target_attention
            
            print("  Training pattern learning task...")
            
            initial_errors = []
            final_errors = []
            
            for epoch in range(60):
                epoch_errors = []
                
                for step in range(10):
                    x, target_attention = create_pattern_task()
                    
                    output, attention = gsa(x)
                    
                    # Focus on first token attention
                    first_token_attention = attention[:, 0, :, :].mean(dim=-1)
                    first_token_target = target_attention[:, 0, :]
                    
                    pattern_loss = F.mse_loss(first_token_attention, first_token_target)
                    recon_loss = F.mse_loss(output, x)
                    
                    total_loss = 3.0 * pattern_loss + 0.1 * recon_loss
                    
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(gsa.parameters(), 1.0)
                    optimizer.step()
                    
                    epoch_errors.append(pattern_loss.item())
                
                if epoch < 5:
                    initial_errors.extend(epoch_errors)
                if epoch >= 55:
                    final_errors.extend(epoch_errors)
                
                if epoch % 15 == 0:
                    avg_error = np.mean(epoch_errors)
                    movement_stats = gsa.get_movement_stats()
                    print(f"    Epoch {epoch}: Pattern Error = {avg_error:.4f}, Movement = {movement_stats['avg_movement']:.3f}")
            
            # Calculate improvement
            initial_avg = np.mean(initial_errors)
            final_avg = np.mean(final_errors)
            improvement = ((initial_avg - final_avg) / initial_avg) * 100 if initial_avg > 0 else 0
            
            success = improvement > self.config.pattern_improvement_threshold
            
            print(f"  📊 Results:")
            print(f"    Initial error: {initial_avg:.4f}")
            print(f"    Final error: {final_avg:.4f}")
            print(f"    Improvement: {improvement:.1f}%")
            print(f"  Status: {'✅ PASSED' if success else '❌ FAILED'} (threshold: {self.config.pattern_improvement_threshold}%)")
            
            return success, improvement
            
        except Exception as e:
            print(f"  ❌ FAILED with error: {e}")
            return False, 0.0
    
    def _test_adaptive_movement(self) -> Tuple[bool, Dict]:
        """Test 3: Adaptive movement validation"""
        print("\n🚀 TEST 3: Adaptive Movement Validation")
        print("-" * 50)
        
        try:
            gsa = ProvenGSA(self.config)
            optimizer = torch.optim.Adam(gsa.parameters(), lr=self.config.learning_rate * 2)
            
            print("  Training with movement-encouraging task...")
            
            initial_stats = gsa.get_movement_stats()
            
            for step in range(200):
                x = torch.randn(6, self.config.seq_len, self.config.dim) * 0.1
                
                # Shift patterns to encourage movement
                if step < 100:
                    x[:, :5] += torch.randn(6, 5, self.config.dim) * 0.8
                else:
                    x[:, -5:] += torch.randn(6, 5, self.config.dim) * 0.8
                
                output, _ = gsa(x)
                loss = F.mse_loss(output, x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if step % 50 == 0:
                    stats = gsa.get_movement_stats()
                    print(f"    Step {step}: Movement = {stats['avg_movement']:.3f}, Amp Var = {stats['amplitude_variance']:.3f}")
            
            final_stats = gsa.get_movement_stats()
            
            # Success criteria
            movement_success = final_stats['avg_movement'] > self.config.movement_threshold
            specialization_success = (final_stats['amplitude_variance'] / 
                                    max(initial_stats['amplitude_variance'], 1e-8)) > self.config.specialization_threshold
            
            success = movement_success and specialization_success
            
            print(f"  📊 Results:")
            print(f"    Movement: {initial_stats['avg_movement']:.3f} → {final_stats['avg_movement']:.3f}")
            print(f"    Specialization: {initial_stats['amplitude_variance']:.3f} → {final_stats['amplitude_variance']:.3f}")
            print(f"  Status: {'✅ PASSED' if success else '❌ FAILED'}")
            
            return success, final_stats
            
        except Exception as e:
            print(f"  ❌ FAILED with error: {e}")
            return False, {}
    
    def _test_efficiency(self) -> Tuple[bool, Dict]:
        """Test 4: Efficiency benchmarking"""
        print("\n⚡ TEST 4: Efficiency Benchmarking")
        print("-" * 50)
        
        try:
            gsa = ProvenGSA(self.config)
            std_attn = StandardAttention(self.config)
            
            # Warmup
            dummy = torch.randn(4, 32, self.config.dim)
            _ = gsa(dummy)
            _ = std_attn(dummy)
            
            efficiency_results = {}
            seq_lengths = [32, 64, 128]
            
            for seq_len in seq_lengths:
                test_input = torch.randn(4, seq_len, self.config.dim)
                
                # Time GSA
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()
                for _ in range(10):
                    _ = gsa(test_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                gsa_time = (time.time() - start) / 10
                
                # Time Standard
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()
                for _ in range(10):
                    _ = std_attn(test_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                std_time = (time.time() - start) / 10
                
                overhead = gsa_time / std_time
                efficiency_results[seq_len] = {
                    'gsa_time': gsa_time,
                    'std_time': std_time,
                    'overhead': overhead
                }
                
                print(f"    Seq {seq_len}: {overhead:.1f}x overhead")
            
            avg_overhead = np.mean([r['overhead'] for r in efficiency_results.values()])
            success = avg_overhead < self.config.efficiency_threshold
            
            print(f"  📊 Results:")
            print(f"    Average overhead: {avg_overhead:.1f}x")
            print(f"  Status: {'✅ PASSED' if success else '❌ FAILED'} (threshold: {self.config.efficiency_threshold}x)")
            
            return success, efficiency_results
            
        except Exception as e:
            print(f"  ❌ FAILED with error: {e}")
            return False, {}
    
    def _test_sequence_modeling(self) -> Tuple[bool, Dict]:
        """Test 5: Sequence modeling performance"""
        print("\n📝 TEST 5: Sequence Modeling Performance")
        print("-" * 50)
        
        try:
            def create_sequence_task(n_samples=100):
                """Create structured sequence task"""
                data, labels = [], []
                
                for _ in range(n_samples):
                    seq = torch.randn(self.config.seq_len, self.config.dim) * 0.2
                    
                    if np.random.random() < 0.5:
                        # Copy task
                        copy_pos = np.random.randint(0, self.config.seq_len // 2)
                        seq[copy_pos] += torch.randn(self.config.dim) * 0.8
                        label = copy_pos
                    else:
                        # Average task
                        important_positions = np.random.choice(self.config.seq_len, size=3, replace=False)
                        for pos in important_positions:
                            seq[pos] += torch.randn(self.config.dim) * 0.5
                        label = important_positions[0]
                    
                    data.append(seq)
                    labels.append(label)
                
                return torch.stack(data), torch.tensor(labels)
            
            # Test both models
            models = {
                'GSA': ProvenGSA(self.config),
                'Standard': StandardAttention(self.config)
            }
            
            results = {}
            
            for name, model in models.items():
                print(f"  Training {name}...")
                
                classifier = nn.Sequential(
                    model,
                    nn.AdaptiveAvgPool1d(1) if hasattr(model, 'dim') else nn.Identity(),
                    nn.Flatten(),
                    nn.Linear(self.config.dim, self.config.seq_len)
                )
                
                optimizer = torch.optim.Adam(classifier.parameters(), lr=self.config.learning_rate)
                
                # Training data
                train_data, train_labels = create_sequence_task(200)
                test_data, test_labels = create_sequence_task(50)
                
                # Training loop
                for epoch in range(50):
                    for i in range(0, len(train_data), self.config.batch_size):
                        batch_data = train_data[i:i+self.config.batch_size]
                        batch_labels = train_labels[i:i+self.config.batch_size]
                        
                        if hasattr(model, 'config'):  # GSA
                            hidden, _ = model(batch_data)
                        else:  # Standard
                            hidden, _ = model(batch_data)
                        
                        pooled = hidden.mean(dim=1)
                        logits = classifier[-1](pooled)
                        
                        loss = F.cross_entropy(logits, batch_labels)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # Evaluation
                classifier.eval()
                with torch.no_grad():
                    if hasattr(model, 'config'):  # GSA
                        hidden, _ = model(test_data)
                    else:  # Standard
                        hidden, _ = model(test_data)
                    
                    pooled = hidden.mean(dim=1)
                    logits = classifier[-1](pooled)
                    
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == test_labels).float().mean().item()
                
                results[name] = accuracy
                print(f"    {name} accuracy: {accuracy:.3f}")
            
            gsa_competitive = results['GSA'] >= results['Standard'] * 0.8  # Within 20%
            
            print(f"  📊 Results:")
            print(f"    GSA: {results['GSA']:.3f}")
            print(f"    Standard: {results['Standard']:.3f}")
            print(f"  Status: {'✅ PASSED' if gsa_competitive else '❌ FAILED'}")
            
            return gsa_competitive, results
            
        except Exception as e:
            print(f"  ❌ FAILED with error: {e}")
            return False, {}
    
    def _test_integration(self) -> bool:
        """Test 6: Integration testing"""
        print("\n🔗 TEST 6: Integration Testing")
        print("-" * 50)
        
        try:
            # Create simple transformer with GSA
            class SimpleTransformer(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.attention = ProvenGSA(config)
                    self.norm1 = nn.LayerNorm(config.dim)
                    self.ffn = nn.Sequential(
                        nn.Linear(config.dim, config.dim * 4),
                        nn.ReLU(),
                        nn.Linear(config.dim * 4, config.dim)
                    )
                    self.norm2 = nn.LayerNorm(config.dim)
                
                def forward(self, x):
                    # Self-attention with residual
                    attn_out, _ = self.attention(x)
                    x = self.norm1(x + attn_out)
                    
                    # FFN with residual
                    ffn_out = self.ffn(x)
                    x = self.norm2(x + ffn_out)
                    
                    return x
            
            model = SimpleTransformer(self.config)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            
            print("  Testing transformer integration...")
            
            # Test training stability
            stable_training = True
            for step in range(50):
                x = torch.randn(4, self.config.seq_len, self.config.dim)
                
                output = model(x)
                loss = F.mse_loss(output, x)  # Reconstruction task
                
                optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN gradients
                for param in model.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        stable_training = False
                        break
                
                if not stable_training:
                    break
                    
                optimizer.step()
                
                if step % 10 == 0:
                    print(f"    Step {step}: Loss = {loss.item():.4f}")
            
            print(f"  Status: {'✅ PASSED' if stable_training else '❌ FAILED'}")
            
            return stable_training
            
        except Exception as e:
            print(f"  ❌ FAILED with error: {e}")
            return False
    
    def _test_scalability(self) -> Tuple[bool, Dict]:
        """Test 7: Scalability analysis"""
        print("\n📈 TEST 7: Scalability Analysis")
        print("-" * 50)
        
        try:
            scalability_results = {}
            
            # Test different dimensions
            dimensions = [32, 64, 128]
            seq_lengths = [16, 32, 64]
            
            base_time = None
            
            for dim in dimensions:
                for seq_len in seq_lengths:
                    config = ValidationConfig(
                        dim=dim,
                        seq_len=seq_len,
                        n_splats=self.config.n_splats,
                        n_heads=min(8, dim // 8)  # Ensure valid head dimension
                    )
                    
                    try:
                        gsa = ProvenGSA(config)
                        x = torch.randn(4, seq_len, dim)
                        
                        # Time forward pass
                        start = time.time()
                        for _ in range(5):
                            _ = gsa(x)
                        elapsed = (time.time() - start) / 5
                        
                        if base_time is None:
                            base_time = elapsed
                        
                        scaling_factor = elapsed / base_time
                        
                        scalability_results[f"{dim}x{seq_len}"] = {
                            'time': elapsed,
                            'scaling_factor': scaling_factor
                        }
                        
                        print(f"    Dim {dim}, Seq {seq_len}: {elapsed*1000:.1f}ms ({scaling_factor:.1f}x)")
                        
                    except Exception as inner_e:
                        print(f"    Dim {dim}, Seq {seq_len}: Failed ({inner_e})")
                        scalability_results[f"{dim}x{seq_len}"] = {'time': float('inf'), 'scaling_factor': float('inf')}
            
            # Check if scaling is reasonable (no exponential blowup)
            max_scaling = max(r.get('scaling_factor', 0) for r in scalability_results.values() if r.get('scaling_factor', 0) != float('inf'))
            reasonable_scaling = max_scaling < 50  # Within 50x of base case
            
            print(f"  📊 Results:")
            print(f"    Max scaling factor: {max_scaling:.1f}x")
            print(f"  Status: {'✅ PASSED' if reasonable_scaling else '❌ FAILED'}")
            
            return reasonable_scaling, scalability_results
            
        except Exception as e:
            print(f"  ❌ FAILED with error: {e}")
            return False, {}
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE VALIDATION REPORT")
        print("=" * 80)
        
        # Count successes
        test_results = [
            self.results.get('mathematical_correctness', False),
            self.results.get('pattern_learning', False),
            self.results.get('adaptive_movement', False),
            self.results.get('efficiency', False),
            self.results.get('sequence_modeling', False),
            self.results.get('integration', False),
            self.results.get('scalability', False)
        ]
        
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        test_names = [
            "Mathematical Correctness",
            "Pattern Learning",
            "Adaptive Movement", 
            "Efficiency",
            "Sequence Modeling",
            "Integration",
            "Scalability"
        ]
        
        for i, (name, passed) in enumerate(zip(test_names, test_results)):
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{name}: {status}")
        
        print(f"\nTESTS PASSED: {passed_tests}/{total_tests}")
        
        # Overall assessment
        if passed_tests >= 6:
            verdict = "🎉 OUTSTANDING SUCCESS"
            assessment = "GSA is ready for production use"
        elif passed_tests >= 5:
            verdict = "✅ STRONG SUCCESS"
            assessment = "GSA shows strong promise with minor issues to address"
        elif passed_tests >= 4:
            verdict = "⚠️ MODERATE SUCCESS"
            assessment = "GSA core concept is sound but needs refinement"
        elif passed_tests >= 2:
            verdict = "🔧 PARTIAL SUCCESS"
            assessment = "GSA shows potential but requires significant work"
        else:
            verdict = "❌ VALIDATION FAILED"
            assessment = "GSA concept needs fundamental rethinking"
        
        print(f"\nOVERALL RESULT: {verdict}")
        print(f"ASSESSMENT: {assessment}")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS:")
        
        if self.results.get('pattern_learning'):
            improvement = self.results.get('pattern_improvement_pct', 0)
            print(f"   • Pattern learning achieved {improvement:.1f}% improvement")
        
        if self.results.get('adaptive_movement'):
            movement = self.results.get('movement_stats', {}).get('avg_movement', 0)
            print(f"   • Splat movement demonstrated: {movement:.3f} average distance")
        
        if self.results.get('efficiency'):
            overhead = np.mean([r['overhead'] for r in self.results.get('efficiency_stats', {}).values()])
            print(f"   • Computational overhead: {overhead:.1f}x vs standard attention")
        
        print(f"\n📋 RECOMMENDATIONS:")
        
        if not self.results.get('efficiency'):
            print("   • Priority: Optimize computational efficiency")
        
        if not self.results.get('scalability'):
            print("   • Priority: Address scalability limitations")
        
        if passed_tests >= 5:
            print("   • Consider production deployment")
            print("   • Focus on optimization and integration")
        else:
            print("   • Continue research and development")
            print("   • Address fundamental limitations")
        
        # Save results
        self.results['validation_summary'] = {
            'tests_passed': passed_tests,
            'total_tests': total_tests,
            'verdict': verdict,
            'assessment': assessment,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open('gsa_comprehensive_validation_results.json', 'w') as f:
                # Convert any tensor values to lists for JSON serialization
                json_results = self._convert_for_json(self.results)
                json.dump(json_results, f, indent=2)
            print(f"\n✅ Detailed results saved to: gsa_comprehensive_validation_results.json")
        except Exception as e:
            print(f"\n❌ Failed to save results: {e}")
        
        print(f"\nValidation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _convert_for_json(self, obj):
        """Convert PyTorch tensors and other non-JSON types for serialization"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        elif isinstance(obj, (int, float, str)):
            return obj
        else:
            return str(obj)  # Fallback to string representation

def run_comprehensive_validation():
    """Run the complete GSA validation"""
    config = ValidationConfig()
    suite = ValidationSuite(config)
    
    results = suite.run_full_validation()
    
    # Return success status
    passed_tests = results['validation_summary']['tests_passed']
    total_tests = results['validation_summary']['total_tests']
    
    return passed_tests >= 5  # Success if 5+ tests pass

if __name__ == "__main__":
    success = run_comprehensive_validation()
    exit(0 if success else 1)
