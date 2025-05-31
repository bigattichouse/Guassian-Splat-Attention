"""
Simplified Gaussian Splat Attention (GSA) - Debug Validation Script
Addresses suspicious results: zero variance, no specialization, degrading baseline.
Tests gradient flow, parameter updates, and training loop integrity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import time
from dataclasses import dataclass
import json
import warnings
from collections import defaultdict

# Configuration
@dataclass
class ModelConfig:
    vocab_size: int = 1000
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_splats_per_head: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Debug utilities
class DebugLogger:
    """Centralized debug logging for tracking parameter evolution and gradients."""
    
    def __init__(self):
        self.logs = defaultdict(list)
        self.gradient_logs = defaultdict(list)
        self.parameter_snapshots = defaultdict(list)
        
    def log(self, category: str, value: Any, epoch: int = None):
        """Log a value under a category."""
        entry = {'value': value, 'epoch': epoch}
        self.logs[category].append(entry)
    
    def log_gradients(self, model_name: str, model: nn.Module, epoch: int):
        """Log gradient statistics for all parameters."""
        grad_stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                # Handle single-element tensors for std()
                grad_std = grad.std().item() if grad.numel() > 1 else 0.0
                grad_stats[name] = {
                    'mean': grad.mean().item(),
                    'std': grad_std,
                    'max': grad.max().item(),
                    'min': grad.min().item(),
                    'norm': grad.norm().item(),
                    'has_nan': torch.isnan(grad).any().item(),
                    'has_inf': torch.isinf(grad).any().item(),
                    'numel': grad.numel()
                }
            else:
                grad_stats[name] = {'error': 'No gradient'}
        
        self.gradient_logs[model_name].append({
            'epoch': epoch,
            'gradients': grad_stats
        })
    
    def log_parameters(self, model_name: str, model: nn.Module, epoch: int):
        """Log parameter values and statistics."""
        param_stats = {}
        for name, param in model.named_parameters():
            param_data = param.data
            # Handle single-element tensors for std()
            std_val = param_data.std().item() if param_data.numel() > 1 else 0.0
            param_stats[name] = {
                'mean': param_data.mean().item(),
                'std': std_val,
                'max': param_data.max().item(),
                'min': param_data.min().item(),
                'norm': param_data.norm().item(),
                'requires_grad': param.requires_grad,
                'shape': list(param.shape),
                'numel': param_data.numel()
            }
        
        self.parameter_snapshots[model_name].append({
            'epoch': epoch,
            'parameters': param_stats
        })
    
    def check_parameter_evolution(self, model_name: str) -> Dict:
        """Check if parameters are actually changing over time."""
        snapshots = self.parameter_snapshots[model_name]
        if len(snapshots) < 2:
            return {'error': 'Not enough snapshots'}
        
        evolution = {}
        first_params = snapshots[0]['parameters']
        last_params = snapshots[-1]['parameters']
        
        for param_name in first_params:
            if param_name in last_params:
                first_val = first_params[param_name]
                last_val = last_params[param_name]
                
                evolution[param_name] = {
                    'mean_change': abs(last_val['mean'] - first_val['mean']),
                    'std_change': abs(last_val['std'] - first_val['std']),
                    'norm_change': abs(last_val['norm'] - first_val['norm']),
                    'is_frozen': abs(last_val['mean'] - first_val['mean']) < 1e-6
                }
        
        return evolution
    
    def get_splat_statistics(self, model_name: str) -> Dict:
        """Extract splat-specific parameter statistics."""
        snapshots = self.parameter_snapshots[model_name]
        if not snapshots:
            return {'error': 'No snapshots available'}
        
        splat_stats = []
        for snapshot in snapshots:
            epoch_stats = {'epoch': snapshot['epoch']}
            params = snapshot['parameters']
            
            # Extract splat parameters
            centers = [v for k, v in params.items() if 'center' in k]
            scales = [v for k, v in params.items() if 'log_scale' in k]
            amplitudes = [v for k, v in params.items() if 'amplitude' in k]
            
            if centers:
                epoch_stats['center_variance'] = np.var([c['mean'] for c in centers])
                epoch_stats['center_norms'] = [c['norm'] for c in centers]
            
            if scales:
                epoch_stats['scale_values'] = [np.exp(s['mean']) for s in scales]
                epoch_stats['scale_variance'] = np.var([s['mean'] for s in scales])
            
            if amplitudes:
                epoch_stats['amplitude_values'] = [a['mean'] for a in amplitudes]
                epoch_stats['amplitude_variance'] = np.var([a['mean'] for a in amplitudes])
                epoch_stats['utilization'] = sum(1 for a in amplitudes if a['mean'] > 0.1) / len(amplitudes)
            
            splat_stats.append(epoch_stats)
        
        return splat_stats

debug_logger = DebugLogger()

class SimplifiedGSASplat(nn.Module):
    """
    DEBUG VERSION: Simple Gaussian splat with extensive logging and verification.
    """
    
    def __init__(self, dim: int, splat_id: str = None):
        super().__init__()
        self.dim = dim
        self.splat_id = splat_id or f"splat_{id(self)}"
        
        # Core parameters with different initialization to encourage diversity
        self.center = nn.Parameter(torch.randn(dim) * 0.5)  # Larger initialization
        self.log_scale = nn.Parameter(torch.randn(1) * 0.5)  # Random initial scales
        self.amplitude = nn.Parameter(torch.rand(1) * 2.0)   # Random amplitudes
        
        # Debug tracking
        self.forward_count = 0
        self.grad_norm_history = []
        
    def verify_gradients(self):
        """Debug method to check gradient flow."""
        grad_info = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_info[name] = {
                    'has_grad': True,
                    'grad_norm': param.grad.norm().item(),
                    'param_norm': param.norm().item(),
                    'grad_max': param.grad.abs().max().item()
                }
            else:
                grad_info[name] = {'has_grad': False}
        return grad_info
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Forward pass with guaranteed gradient flow to all parameters."""
        self.forward_count += 1
        
        B, T, D = queries.shape
        
        # Use raw parameters directly (more gradient-friendly)
        scale = torch.exp(self.log_scale).clamp(min=1e-4, max=10.0)  # Bound scale
        
        # Compute attention using scaled dot-product approach
        # This is more gradient-stable than distance-based computation
        
        # Project queries and keys through splat center
        q_centered = queries - self.center.unsqueeze(0).unsqueeze(0)  # [B, T, D]
        k_centered = keys - self.center.unsqueeze(0).unsqueeze(0)     # [B, T, D]
        
        # Scaled attention computation (similar to standard attention)
        # Scale by splat scale parameter
        q_scaled = q_centered / scale  # [B, T, D]
        k_scaled = k_centered / scale  # [B, T, D]
        
        # Compute attention matrix using scaled dot product
        attention_scores = torch.matmul(q_scaled, k_scaled.transpose(-2, -1))  # [B, T, T]
        
        # Apply splat amplitude
        attention = self.amplitude * attention_scores
        
        # Add small identity to ensure gradients flow to center parameter
        identity_component = 0.01 * self.amplitude * torch.eye(T, device=queries.device).unsqueeze(0)
        attention = attention + identity_component
        
        return attention

    
    def log_gradient_stats(self):
        """Log gradient statistics after backward pass."""
        stats = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.grad_norm_history.append(grad_norm)
                # Handle single-element tensors for std()
                grad_std = param.grad.std().item() if param.grad.numel() > 1 else 0.0
                stats[name] = {
                    'grad_norm': grad_norm,
                    'grad_mean': param.grad.mean().item(),
                    'grad_std': grad_std
                }
            else:
                stats[name] = {'error': 'No gradient'}
        return stats

class SimplifiedGSAAttention(nn.Module):
    """
    DEBUG VERSION: Multi-head attention with splat parameter tracking.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.dim // config.n_heads
        
        # Create splats with unique IDs for tracking
        self.splats = nn.ModuleList([
            SimplifiedGSASplat(self.head_dim, f"L{id(self)}_H{h}_S{s}")
            for h in range(config.n_heads)
            for s in range(config.n_splats_per_head)
        ])
        
        # Standard projections
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.out_proj = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(self.head_dim)
        
        # Debug tracking
        self.attention_stats = []
    
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply layer norm
        q = self.norm(q)
        k = self.norm(k)
        
        # COMPLETELY NEW APPROACH: Parallel splat processing
        all_head_outputs = []
        all_attention_weights = []
        
        for h in range(self.config.n_heads):
            # Get splats for this head
            head_splats = []
            for s in range(self.config.n_splats_per_head):
                splat_idx = h * self.config.n_splats_per_head + s
                head_splats.append(self.splats[splat_idx])
            
            # Compute attention for each splat independently
            splat_attentions = []
            splat_outputs = []
            
            for splat in head_splats:
                # Each splat computes its own attention
                splat_attention = splat(q[:, h], k[:, h])  # [B, T, T]
                
                # Apply softmax to THIS splat's attention
                splat_attention_norm = F.softmax(splat_attention, dim=-1)
                
                # Apply to values to get this splat's output
                splat_output = torch.matmul(splat_attention_norm, v[:, h])  # [B, T, head_dim]
                
                splat_attentions.append(splat_attention_norm)
                splat_outputs.append(splat_output)
            
            # Combine splat outputs with learnable weights
            # This ensures ALL splats contribute to the final loss
            splat_outputs_stacked = torch.stack(splat_outputs, dim=0)  # [n_splats, B, T, head_dim]
            
            # Compute splat importance weights (this creates competition)
            splat_importance = torch.softmax(torch.stack([
                torch.mean(splat.amplitude) for splat in head_splats
            ]), dim=0)
            
            # Weighted combination of splat outputs
            head_output = torch.zeros_like(splat_outputs[0])
            for s, (output, weight) in enumerate(zip(splat_outputs, splat_importance)):
                head_output += weight * output
            
            # Average attention weights for visualization
            head_attention = torch.stack(splat_attentions, dim=0).mean(dim=0)
            
            all_head_outputs.append(head_output)
            all_attention_weights.append(head_attention)
        
        # Combine all heads
        combined_output = torch.stack(all_head_outputs, dim=1)  # [B, n_heads, T, head_dim]
        combined_output = combined_output.transpose(1, 2).contiguous().view(B, T, D)
        
        # Final projection
        output = self.out_proj(combined_output)
        
        if return_attention:
            attention_weights = torch.stack(all_attention_weights, dim=1)  # [B, n_heads, T, T]
            return output, attention_weights
        else:
            return output, None
        
class SimplifiedGSATransformerBlock(nn.Module):
    """DEBUG VERSION: Transformer block with parameter tracking."""
    
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attention = SimplifiedGSAAttention(config)
        
        # Standard MLP
        mlp_dim = int(config.dim * 4)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(mlp_dim, config.dim),
            nn.Dropout(config.dropout)
        )
        
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # Attention block
        attn_out, attn_weights = self.attention(self.norm1(x), return_attention)
        x = x + self.dropout(attn_out)
        
        # MLP block
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x, attn_weights if return_attention else None

class SimplifiedGSATransformer(nn.Module):
    """DEBUG VERSION: Complete transformer with extensive parameter tracking."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.dim))
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimplifiedGSATransformerBlock(config, layer_id=i)
            for i in range(config.n_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie embedding and output weights
        self.output.weight = self.token_emb.weight
        
        # Initialize weights
        self._init_weights()
        
        # Debug: track initialization
        self.initialized = True
        self.parameter_count = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _init_weights(self):
        """Initialize with diversity to avoid identical parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, SimplifiedGSASplat):
                # Special initialization for splats to ensure diversity
                with torch.no_grad():
                    # Different centers for each splat
                    module.center.data = torch.randn_like(module.center) * (0.1 + torch.rand(1).item() * 0.4)
                    # Different scales
                    module.log_scale.data = torch.randn_like(module.log_scale) * 0.5
                    # Different amplitudes
                    module.amplitude.data = torch.rand_like(module.amplitude) * 2.0
    
    def verify_parameters_in_optimizer(self, optimizer):
        """Verify all parameters are in optimizer."""
        opt_params = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                opt_params.add(id(p))
        
        model_params = set(id(p) for p in self.parameters())
        
        missing = model_params - opt_params
        extra = opt_params - model_params
        
        return {
            'all_included': len(missing) == 0,
            'missing_count': len(missing),
            'extra_count': len(extra),
            'model_param_count': len(model_params),
            'optimizer_param_count': len(opt_params)
        }
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, T = x.shape
        
        # Embeddings
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.dropout(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        attentions = []
        for block in self.blocks:
            x, attn = block(x, return_attention)
            if return_attention and attn is not None:
                attentions.append(attn)
        
        # Output projection
        x = self.norm(x)
        logits = self.output(x)
        
        return logits, attentions if return_attention else None

class StandardTransformerBaseline(nn.Module):
    """Standard transformer baseline with debug tracking."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.dim))
        self.dropout = nn.Dropout(config.dropout)
        
        # Standard transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.dim,
                nhead=config.n_heads,
                dim_feedforward=config.dim * 4,
                dropout=config.dropout,
                activation='gelu',
                batch_first=True
            )
            for _ in range(config.n_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(config.dim)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.output.weight = self.token_emb.weight
        
        self._init_weights()
        
        # Debug tracking
        self.parameter_count = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        B, T = x.shape
        
        # Embeddings
        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb[:, :T, :]
        x = self.dropout(tok_emb + pos_emb)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output
        x = self.norm(x)
        logits = self.output(x)
        
        return logits, None

# Validation tasks
def create_debug_task(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    """Create a simple debug task for gradient verification."""
    # Simple pattern: predict next token in sequence
    data = torch.arange(0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1) % vocab_size
    return data

def create_copy_task(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
    """Create copy task."""
    data = torch.randint(1, vocab_size // 2, (batch_size, seq_len), device=device)
    copy_len = 10
    data[:, -copy_len:] = data[:, :copy_len]
    return data

def debug_training_step(model: nn.Module, data: torch.Tensor, optimizer: torch.optim.Optimizer, 
                       model_name: str, step: int, config: ModelConfig) -> Dict:
    """Single training step with extensive debugging."""
    model.train()
    
    # Forward pass
    logits, _ = model(data[:, :-1])
    loss = F.cross_entropy(
        logits.reshape(-1, config.vocab_size),
        data[:, 1:].reshape(-1)
    )
    
    # Pre-backward checks
    pre_backward_params = {}
    for name, param in model.named_parameters():
        pre_backward_params[name] = param.data.clone()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check specific splat gradients
    if step % 10 == 0 and model_name == 'gsa':
        print(f"\nStep {step} Splat Gradient Check:")
        for block_idx, block in enumerate(model.blocks):
            splat = block.attention.splats[0]  # Check first splat
            grad_info = splat.verify_gradients()
            for param_name, info in grad_info.items():
                if info['has_grad']:
                    print(f"  {param_name}: norm={info['grad_norm']:.6f}")
                else:
                    print(f"  {param_name}: NO GRADIENT!")
    
    # Check gradients
    grad_info = {}
    zero_grad_params = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            # Handle single-element tensors for std()
            grad_std = param.grad.std().item() if param.grad.numel() > 1 else 0.0
            grad_info[name] = {
                'norm': grad_norm,
                'mean': param.grad.mean().item(),
                'std': grad_std,
                'has_nan': torch.isnan(param.grad).any().item(),
                'has_inf': torch.isinf(param.grad).any().item()
            }
            if grad_norm == 0:
                zero_grad_params.append(name)
        else:
            grad_info[name] = {'error': 'No gradient'}
            zero_grad_params.append(name)
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step
    optimizer.step()
    
    # Check parameter updates
    param_updates = {}
    frozen_params = []
    for name, param in model.named_parameters():
        if name in pre_backward_params:
            change = (param.data - pre_backward_params[name]).abs().max().item()
            param_updates[name] = change
            if change < 1e-8:
                frozen_params.append(name)
    
    # Log splat-specific gradients (only for GSA model)
    if hasattr(model, 'blocks') and model_name == 'gsa':
        for block_idx, block in enumerate(model.blocks):
            if hasattr(block.attention, 'splats'):
                for splat_idx, splat in enumerate(block.attention.splats):
                    splat_grad_stats = splat.log_gradient_stats()
                    if step % 10 == 0:  # Log every 10 steps
                        debug_logger.log(f'splat_grads_L{block_idx}_S{splat_idx}', splat_grad_stats, step)
    
    return {
        'loss': loss.item(),
        'grad_info': grad_info,
        'zero_grad_params': zero_grad_params,
        'param_updates': param_updates,
        'frozen_params': frozen_params,
        'total_params': len(list(model.parameters())),
        'params_with_gradients': len([p for p in model.parameters() if p.grad is not None])
    }

def run_debug_validation():
    """Run comprehensive debug validation."""
    print("="*80)
    print("SIMPLIFIED GSA DEBUG VALIDATION")
    print("="*80)
    print("Investigating suspicious results: zero variance, no specialization, degrading baseline")
    
    # Configuration
    config = ModelConfig(
        vocab_size=1000,
        dim=128,  # Smaller for faster debugging
        n_layers=2,
        n_heads=4,
        n_splats_per_head=4,
        max_seq_len=64
    )
    
    batch_size = 16
    n_debug_steps = 50
    
    print(f"\nDebug Configuration:")
    print(f"  Model dimension: {config.dim}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    print(f"  Splats per head: {config.n_splats_per_head}")
    print(f"  Debug steps: {n_debug_steps}")
    
    # Create models
    print("\nCreating models...")
    gsa_model = SimplifiedGSATransformer(config).to(device)
    std_model = StandardTransformerBaseline(config).to(device)
    
    print(f"  GSA parameters: {gsa_model.parameter_count:,}")
    print(f"  GSA trainable: {gsa_model.trainable_params:,}")
    print(f"  STD parameters: {std_model.parameter_count:,}")
    print(f"  STD trainable: {std_model.trainable_params:,}")
    
    # Create optimizers
    gsa_optimizer = torch.optim.AdamW(gsa_model.parameters(), lr=0.001, weight_decay=0.01)
    std_optimizer = torch.optim.AdamW(std_model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Verify optimizer setup
    print("\nVerifying optimizer setup...")
    gsa_opt_check = gsa_model.verify_parameters_in_optimizer(gsa_optimizer)
    print(f"  GSA optimizer check: {gsa_opt_check}")
    
    if not gsa_opt_check['all_included']:
        print("  ❌ WARNING: Not all GSA parameters in optimizer!")
    else:
        print("  ✓ All GSA parameters included in optimizer")
    
    # Log initial parameters
    debug_logger.log_parameters('gsa', gsa_model, 0)
    debug_logger.log_parameters('std', std_model, 0)
    
    # Debug training loop
    print("\nStarting debug training...")
    print("-"*60)
    
    # Create debug data
    debug_data = create_debug_task(batch_size, config.max_seq_len, config.vocab_size, device)
    copy_data = create_copy_task(batch_size, config.max_seq_len, config.vocab_size, device)
    
    gsa_debug_results = []
    std_debug_results = []
    
    for step in range(n_debug_steps):
        # Use alternating tasks
        data = debug_data if step % 2 == 0 else copy_data
        
        # Debug GSA training step
        gsa_result = debug_training_step(gsa_model, data, gsa_optimizer, 'gsa', step, config)
        gsa_debug_results.append(gsa_result)
        
        # Debug standard training step
        std_result = debug_training_step(std_model, data, std_optimizer, 'std', step, config)
        std_debug_results.append(std_result)
        
        # Log parameters and gradients periodically
        if step % 10 == 0:
            debug_logger.log_parameters('gsa', gsa_model, step)
            debug_logger.log_parameters('std', std_model, step)
            debug_logger.log_gradients('gsa', gsa_model, step)
            debug_logger.log_gradients('std', std_model, step)
            
            print(f"\nStep {step}:")
            print(f"  GSA Loss: {gsa_result['loss']:.4f}")
            print(f"  STD Loss: {std_result['loss']:.4f}")
            print(f"  GSA zero gradients: {len(gsa_result['zero_grad_params'])}")
            print(f"  GSA frozen params: {len(gsa_result['frozen_params'])}")
            
            # Check for specific splat parameters
            splat_grads = [name for name in gsa_result['grad_info'] 
                           if 'splat' in name.lower() or 'center' in name or 'scale' in name or 'amplitude' in name]
            if splat_grads:
                print(f"  Splat parameter gradients found: {len(splat_grads)}")
                # Sample a few
                for name in splat_grads[:3]:
                    grad_data = gsa_result['grad_info'][name]
                    if 'norm' in grad_data:
                        print(f"    {name}: norm={grad_data['norm']:.6f}")
    
    # Analyze parameter evolution
    print("\n" + "="*80)
    print("PARAMETER EVOLUTION ANALYSIS")
    print("="*80)
    
    gsa_evolution = debug_logger.check_parameter_evolution('gsa')
    std_evolution = debug_logger.check_parameter_evolution('std')
    
    print("\nGSA Parameter Evolution:")
    splat_params = {k: v for k, v in gsa_evolution.items() 
                   if 'center' in k or 'scale' in k or 'amplitude' in k}
    
    for param_name, changes in list(splat_params.items())[:10]:  # Show first 10
        print(f"  {param_name}:")
        print(f"    Mean change: {changes['mean_change']:.6f}")
        print(f"    Frozen: {changes['is_frozen']}")
    
    # Get splat statistics
    splat_stats = debug_logger.get_splat_statistics('gsa')
    
    print("\nSplat Statistics Over Time:")
    if splat_stats and len(splat_stats) > 1:
        first_epoch = splat_stats[0]
        last_epoch = splat_stats[-1]
        
        print(f"  Initial (Step 0):")
        if 'scale_variance' in first_epoch:
            print(f"    Scale variance: {first_epoch['scale_variance']:.6f}")
        if 'amplitude_variance' in first_epoch:
            print(f"    Amplitude variance: {first_epoch['amplitude_variance']:.6f}")
        if 'utilization' in first_epoch:
            print(f"    Utilization: {first_epoch['utilization']:.2f}")
        
        print(f"  Final (Step {last_epoch['epoch']}):")
        if 'scale_variance' in last_epoch:
            print(f"    Scale variance: {last_epoch['scale_variance']:.6f}")
        if 'amplitude_variance' in last_epoch:
            print(f"    Amplitude variance: {last_epoch['amplitude_variance']:.6f}")
        if 'utilization' in last_epoch:
            print(f"    Utilization: {last_epoch['utilization']:.2f}")
    
    # Gradient flow analysis
    print("\n" + "="*80)
    print("GRADIENT FLOW ANALYSIS")
    print("="*80)
    
    # Check gradient statistics
    grad_logs = debug_logger.gradient_logs['gsa']
    if grad_logs:
        last_grads = grad_logs[-1]['gradients']
        
        # Categorize parameters
        splat_grads = {}
        other_grads = {}
        
        for param_name, grad_info in last_grads.items():
            if any(key in param_name for key in ['center', 'log_scale', 'amplitude']):
                splat_grads[param_name] = grad_info
            else:
                other_grads[param_name] = grad_info
        
        print(f"\nSplat Parameter Gradients (Last Step):")
        for name, info in list(splat_grads.items())[:10]:
            if 'norm' in info:
                print(f"  {name}: norm={info['norm']:.6f}, mean={info['mean']:.6f}")
            else:
                print(f"  {name}: {info}")
    
    # Identify issues
    print("\n" + "="*80)
    print("ISSUE IDENTIFICATION")
    print("="*80)
    
    issues = []
    
    # Check 1: Frozen parameters
    frozen_splat_params = [k for k, v in gsa_evolution.items() 
                          if v['is_frozen'] and any(x in k for x in ['center', 'scale', 'amplitude'])]
    if frozen_splat_params:
        issues.append(f"FROZEN PARAMETERS: {len(frozen_splat_params)} splat parameters not updating")
        print(f"\n❌ Issue: {len(frozen_splat_params)} frozen splat parameters detected")
        for param in frozen_splat_params[:5]:
            print(f"   - {param}")
    
    # Check 2: Zero gradients
    total_zero_grads = sum(len(r['zero_grad_params']) for r in gsa_debug_results)
    if total_zero_grads > 0:
        issues.append(f"ZERO GRADIENTS: {total_zero_grads} instances of zero gradients")
        print(f"\n❌ Issue: Zero gradients detected in {total_zero_grads} cases")
    
    # Check 3: Low parameter variance
    if splat_stats and len(splat_stats) > 1:
        final_stats = splat_stats[-1]
        if 'scale_variance' in final_stats and final_stats['scale_variance'] < 0.001:
            issues.append("LOW VARIANCE: Scale parameters have near-zero variance")
            print(f"\n❌ Issue: Scale variance too low ({final_stats['scale_variance']:.6f})")
        if 'amplitude_variance' in final_stats and final_stats['amplitude_variance'] < 0.001:
            issues.append("LOW VARIANCE: Amplitude parameters have near-zero variance")
            print(f"\n❌ Issue: Amplitude variance too low ({final_stats['amplitude_variance']:.6f})")
    
    # Check 4: Baseline degradation
    std_losses = [r['loss'] for r in std_debug_results]
    if len(std_losses) > 10 and std_losses[-1] > std_losses[0] * 1.1:
        issues.append("BASELINE DEGRADATION: Standard model loss increasing")
        print(f"\n❌ Issue: Standard baseline degrading ({std_losses[0]:.4f} → {std_losses[-1]:.4f})")
    
    # Visualization of debug results
    print("\n" + "="*80)
    print("GENERATING DEBUG VISUALIZATIONS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Loss comparison
    ax = axes[0, 0]
    gsa_losses = [r['loss'] for r in gsa_debug_results]
    std_losses = [r['loss'] for r in std_debug_results]
    ax.plot(gsa_losses, label='GSA', color='green', alpha=0.8)
    ax.plot(std_losses, label='Standard', color='blue', alpha=0.8)
    ax.set_title('Training Loss Comparison')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Parameter update magnitudes
    ax = axes[0, 1]
    if gsa_debug_results:
        update_magnitudes = []
        for result in gsa_debug_results:
            if 'param_updates' in result:
                splat_updates = [v for k, v in result['param_updates'].items() 
                               if any(x in k for x in ['center', 'scale', 'amplitude'])]
                if splat_updates:
                    update_magnitudes.append(np.mean(splat_updates))
        
        if update_magnitudes:
            ax.plot(update_magnitudes, color='orange', alpha=0.8)
            ax.set_title('Splat Parameter Update Magnitudes')
            ax.set_xlabel('Step')
            ax.set_ylabel('Average Update Magnitude')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
    
    # 3. Gradient norms
    ax = axes[0, 2]
    if grad_logs:
        center_norms = []
        scale_norms = []
        amplitude_norms = []
        
        for log in grad_logs:
            grads = log['gradients']
            centers = [g['norm'] for k, g in grads.items() if 'center' in k and 'norm' in g]
            scales = [g['norm'] for k, g in grads.items() if 'log_scale' in k and 'norm' in g]
            amplitudes = [g['norm'] for k, g in grads.items() if 'amplitude' in k and 'norm' in g]
            
            if centers:
                center_norms.append(np.mean(centers))
            if scales:
                scale_norms.append(np.mean(scales))
            if amplitudes:
                amplitude_norms.append(np.mean(amplitudes))
        
        steps = [log['epoch'] for log in grad_logs]
        if center_norms:
            ax.plot(steps, center_norms, label='Centers', alpha=0.8)
        if scale_norms:
            ax.plot(steps, scale_norms, label='Scales', alpha=0.8)
        if amplitude_norms:
            ax.plot(steps, amplitude_norms, label='Amplitudes', alpha=0.8)
        
        ax.set_title('Splat Gradient Norms')
        ax.set_xlabel('Step')
        ax.set_ylabel('Gradient Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Parameter variance evolution
    ax = axes[1, 0]
    if splat_stats and len(splat_stats) > 1:
        epochs = [s['epoch'] for s in splat_stats]
        scale_vars = [s.get('scale_variance', 0) for s in splat_stats]
        amp_vars = [s.get('amplitude_variance', 0) for s in splat_stats]
        
        ax.plot(epochs, scale_vars, label='Scale Variance', color='purple', alpha=0.8)
        ax.plot(epochs, amp_vars, label='Amplitude Variance', color='red', alpha=0.8)
        ax.set_title('Parameter Variance Evolution')
        ax.set_xlabel('Step')
        ax.set_ylabel('Variance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Utilization over time
    ax = axes[1, 1]
    if splat_stats and len(splat_stats) > 1:
        epochs = [s['epoch'] for s in splat_stats]
        utilizations = [s.get('utilization', 1.0) for s in splat_stats]
        
        ax.plot(epochs, utilizations, color='teal', alpha=0.8, linewidth=2)
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Splat Utilization (Amplitude > 0.1)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Utilization')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    # 6. Debug summary
    ax = axes[1, 2]
    ax.text(0.1, 0.9, 'DEBUG SUMMARY', fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    summary_text = [
        f'Total Parameters: {gsa_model.parameter_count:,}',
        f'Trainable Parameters: {gsa_model.trainable_params:,}',
        f'Parameters in Optimizer: {gsa_opt_check["optimizer_param_count"]}',
        '',
        'Issues Detected:',
    ]
    
    if issues:
        for issue in issues[:5]:  # Show first 5 issues
            summary_text.append(f'• {issue}')
    else:
        summary_text.append('• No major issues detected')
    
    summary_text.extend([
        '',
        'Gradient Flow:',
        f'• Zero gradients: {sum(len(r["zero_grad_params"]) for r in gsa_debug_results[-10:])} (last 10 steps)',
        f'• Frozen params: {len(frozen_splat_params)}',
        '',
        'Recommendations:',
    ])
    
    # Add recommendations based on findings
    if frozen_splat_params:
        summary_text.append('• Check optimizer includes all params')
    if total_zero_grads > 0:
        summary_text.append('• Verify gradient computation')
    if splat_stats and splat_stats[-1].get('scale_variance', 1) < 0.001:
        summary_text.append('• Increase initialization diversity')
    
    ax.text(0.05, 0.85, '\n'.join(summary_text), transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('gsa_debug_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional detailed analysis
    print("\n" + "="*80)
    print("DETAILED PARAMETER ANALYSIS")
    print("="*80)
    
    # Sample some specific splat parameters
    for block_idx, block in enumerate(gsa_model.blocks):
        print(f"\nBlock {block_idx} Splat Analysis:")
        for splat_idx, splat in enumerate(block.attention.splats[:4]):  # First 4 splats
            print(f"  Splat {splat_idx}:")
            print(f"    Center norm: {splat.center.norm().item():.4f}")
            print(f"    Scale: {torch.exp(splat.log_scale).item():.4f}")
            print(f"    Amplitude: {splat.amplitude.item():.4f}")
            print(f"    Forward count: {splat.forward_count}")
            if splat.grad_norm_history:
                print(f"    Avg gradient norm: {np.mean(splat.grad_norm_history[-10:]):.6f}")
    
    # Save detailed debug report
    debug_report = {
        'config': {
            'dim': config.dim,
            'n_layers': config.n_layers,
            'n_heads': config.n_heads,
            'n_splats_per_head': config.n_splats_per_head
        },
        'optimizer_check': gsa_opt_check,
        'issues_detected': issues,
        'parameter_evolution': {
            'initial_variance': splat_stats[0] if splat_stats else None,
            'final_variance': splat_stats[-1] if splat_stats else None
        },
        'gradient_summary': {
            'total_zero_gradients': total_zero_grads,
            'frozen_parameters': len(frozen_splat_params)
        },
        'final_losses': {
            'gsa': gsa_losses[-1] if gsa_losses else None,
            'std': std_losses[-1] if std_losses else None
        }
    }
    
    with open('gsa_debug_report.json', 'w') as f:
        json.dump(debug_report, f, indent=2)
    
    # Final diagnosis
    print("\n" + "="*80)
    print("FINAL DIAGNOSIS")
    print("="*80)
    
    if not issues:
        print("✓ No major issues detected. Parameters are updating and gradients are flowing.")
    else:
        print("❌ Issues detected that explain suspicious results:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\nRoot Causes:")
        if frozen_splat_params:
            print("  • Parameters not updating → Check optimizer setup")
            print("  • Possible detached gradients → Check forward pass")
        if total_zero_grads > 0:
            print("  • Zero gradients → Check loss computation")
            print("  • Possible numerical issues → Check for NaN/Inf")
        if splat_stats and splat_stats[-1].get('scale_variance', 1) < 0.001:
            print("  • Low variance → Initialization not diverse enough")
            print("  • Possible symmetric gradients → Break symmetry")
    
    print("\nDebug artifacts saved:")
    print("  • gsa_debug_analysis.png - Visual analysis")
    print("  • gsa_debug_report.json - Detailed findings")
    
    return {
        'issues': issues,
        'debug_report': debug_report,
        'success': len(issues) == 0
    }

if __name__ == "__main__":
    print("🔍 Starting GSA Debug Validation...")
    print("This will identify why parameters show zero variance and no specialization.")
    
    results = run_debug_validation()
    
    print("\n" + "="*80)
    print("DEBUG VALIDATION COMPLETE")
    print("="*80)
    
    if results['success']:
        print("✅ No major issues found - implementation appears correct")
        print("   Parameters are updating and gradients are flowing properly")
    else:
        print("❌ Issues identified that explain suspicious results:")
        for issue in results['issues']:
            print(f"   • {issue}")
        print("\nRefer to debug artifacts for detailed analysis and recommendations.")
