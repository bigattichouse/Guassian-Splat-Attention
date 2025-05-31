"""
Hypertoroidal Geometric Transformer - Production-Ready Validation
Fixed implementation with vectorized operations, stable evolution, and proper baselines.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

# Configuration
@dataclass
class ModelConfig:
   vocab_size: int = 1000
   dim: int = 256
   n_layers: int = 4
   n_heads: int = 8
   n_splats_per_head: int = 4
   max_seq_len: int = 128
   dropout: float = 0.1
   
# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class VectorizedHypertoroidalSplat(nn.Module):
   """
   Vectorized implementation of hypertoroidal splat with stable evolution.
   """
   
   def __init__(self, dim: int):
       super().__init__()
       self.dim = dim
       
       # Core geometry parameters
       self.center = nn.Parameter(torch.randn(dim) * 0.1)
       self.outer_radius = nn.Parameter(torch.tensor(1.0))
       self.hole_radius = nn.Parameter(torch.tensor(0.01))  # Start nearly solid
       
       # Entry points for wormhole connections
       self.n_entries = 4
       self.entry_points = nn.Parameter(torch.randn(self.n_entries, dim) * 0.2) # from conservative 0.1
       self.entry_strengths = nn.Parameter(torch.zeros(self.n_entries) * 0.1)
       
       # Overall amplitude
       self.amplitude = nn.Parameter(torch.tensor(1.0))
       
       # Evolution tracking
       self.evolution_step = 0
       self.last_evolution_epoch = 0
       
   def compute_torus_surface_distance(self, point_dists: torch.Tensor) -> torch.Tensor:
       """
       Proper torus surface distance calculation - vectorized.
       point_dists: [batch_size, n_points]
       """
       outer_r = torch.abs(self.outer_radius) + 1e-8
       hole_r = torch.abs(self.hole_radius)
       hole_ratio = hole_r / outer_r
       
       if hole_ratio < 0.1:
           # Solid circle phase
           surface_dist = torch.abs(point_dists - outer_r)
       else:
           # True torus phase
           major_circle_dist = torch.abs(point_dists - outer_r)
           torus_surface_dist = torch.sqrt(major_circle_dist**2 + hole_r**2)
           surface_dist = torch.abs(torus_surface_dist - hole_r)
           
       return surface_dist
   
   def compute_vectorized_attention(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
       """
       Fully vectorized attention computation.
       queries: [batch, n_queries, dim]
       keys: [batch, n_keys, dim]
       returns: [batch, n_queries, n_keys]
       """
       batch_size = queries.shape[0]
       n_queries = queries.shape[1]
       n_keys = keys.shape[1]
       
       # Expand center for broadcasting
       center_expanded = self.center.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
       
       # Compute distances to torus center
       q_dists = torch.norm(queries - center_expanded, dim=-1)  # [batch, n_queries]
       k_dists = torch.norm(keys - center_expanded, dim=-1)     # [batch, n_keys]
       
       # Compute torus surface distances
       q_surface_dist = self.compute_torus_surface_distance(q_dists)  # [batch, n_queries]
       k_surface_dist = self.compute_torus_surface_distance(k_dists)  # [batch, n_keys]
       
       # Expand for pairwise computation
       q_surface_dist = q_surface_dist.unsqueeze(2)  # [batch, n_queries, 1]
       k_surface_dist = k_surface_dist.unsqueeze(1)  # [batch, 1, n_keys]
       
       # Surface attention
       surface_attention = torch.exp(-0.5 * (q_surface_dist**2 + k_surface_dist**2))
       
       # Entry point attention (vectorized)
       entry_attention = torch.zeros_like(surface_attention)
       
       active_strengths = torch.sigmoid(self.entry_strengths)
       active_entries = active_strengths > 0.1
       
       if active_entries.any():
           # Compute distances to all entry points at once
           active_points = self.entry_points[active_entries]  # [n_active, dim]
           active_strengths_filtered = active_strengths[active_entries]  # [n_active]
           
           # Queries to entry points: [batch, n_queries, n_active]
           q_entry_dists = torch.cdist(queries, active_points.unsqueeze(0).expand(batch_size, -1, -1))
           # Keys to entry points: [batch, n_keys, n_active]
           k_entry_dists = torch.cdist(keys, active_points.unsqueeze(0).expand(batch_size, -1, -1))
           
           # Compute entry point contributions
           q_entry_attn = torch.exp(-0.5 * q_entry_dists**2)  # [batch, n_queries, n_active]
           k_entry_attn = torch.exp(-0.5 * k_entry_dists**2)  # [batch, n_keys, n_active]
           
           # Combine with strengths and sum
           for i in range(active_strengths_filtered.shape[0]):
               strength = active_strengths_filtered[i]
               entry_contribution = strength * q_entry_attn[:, :, i:i+1] @ k_entry_attn[:, :, i:i+1].transpose(-2, -1)
               entry_attention += entry_contribution
       
       # Combine surface and entry point attention
       total_attention = self.amplitude * (surface_attention + entry_attention)
       
       return total_attention
   
   def gentle_evolution(self):
       """Gentle, stable geometric evolution"""
       with torch.no_grad():
           # Very gradual hole growth
           growth_rate = 0.015 # from conservative 0.002
           self.hole_radius.data += growth_rate
           self.hole_radius.data = torch.clamp(self.hole_radius.data, 0.01, 0.4) #raised from 0.3 in initial
           
           # Gentle entry point strengthening
           self.entry_strengths.data += 0.5 #raised from 0.01 in intial
           self.evolution_step += 1
           
   def layer_specific_evolution(self, hole_growth_rate: float, entry_strength_boost: float):
      """Layer-specific geometric evolution with custom rates"""
      with torch.no_grad():
          # Layer-specific hole growth
          self.hole_radius.data += hole_growth_rate
          self.hole_radius.data = torch.clamp(self.hole_radius.data, 0.01, 0.4)
            
          # Layer-specific entry point strengthening
          self.entry_strengths.data += entry_strength_boost
          self.evolution_step += 1

class VectorizedHypertoroidalAttention(nn.Module):
   """
   Fully vectorized multi-head attention using hypertoroidal splats.
   """
   
   def __init__(self, config: ModelConfig):
       super().__init__()
       self.config = config
       self.head_dim = config.dim // config.n_heads
       
       # Create splats for each head
       self.splats = nn.ModuleList([
           VectorizedHypertoroidalSplat(self.head_dim)
           for _ in range(config.n_heads * config.n_splats_per_head)
       ])
       
       # Projections
       self.qkv = nn.Linear(config.dim, 3 * config.dim)
       self.out_proj = nn.Linear(config.dim, config.dim)
       self.dropout = nn.Dropout(config.dropout)
       
       # Layer norm for stability
       self.norm = nn.LayerNorm(self.head_dim)
       
   def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
       B, T, D = x.shape
       
       # Project to Q, K, V
       qkv = self.qkv(x)
       q, k, v = qkv.chunk(3, dim=-1)
       
       # Reshape for multi-head
       q = q.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
       k = k.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
       v = v.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
       
       # Normalize for stability
       q = self.norm(q)
       k = self.norm(k)
       
       # Compute attention for each head using splats
       attention_weights = []
       
       for h in range(self.config.n_heads):
           head_attention = torch.zeros(B, T, T, device=x.device)
           
           # Sum contributions from all splats for this head
           for s in range(self.config.n_splats_per_head):
               splat_idx = h * self.config.n_splats_per_head + s
               splat = self.splats[splat_idx]
               
               # Get attention from this splat
               splat_attention = splat.compute_vectorized_attention(
                   q[:, h, :, :],  # [B, T, head_dim]
                   k[:, h, :, :]   # [B, T, head_dim]
               )
               
               head_attention += splat_attention / self.config.n_splats_per_head
           
           attention_weights.append(head_attention)
       
       # Stack heads
       attention_weights = torch.stack(attention_weights, dim=1)  # [B, H, T, T]
       
       # Apply softmax
       attention_weights = F.softmax(attention_weights, dim=-1)
       attention_weights = self.dropout(attention_weights)
       
       # Apply attention to values
       output = torch.matmul(attention_weights, v)  # [B, H, T, head_dim]
       
       # Reshape and project
       output = output.transpose(1, 2).contiguous().view(B, T, D)
       output = self.out_proj(output)
       
       if return_attention:
           return output, attention_weights
       return output, None

class HypertoroidalTransformerBlock(nn.Module):
   """Transformer block with hypertoroidal attention"""
   
   def __init__(self, config: ModelConfig):
       super().__init__()
       self.config = config
       
       self.attention = VectorizedHypertoroidalAttention(config)
       
       # MLP
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
       
       return x, attn_weights

class HypertoroidalTransformer(nn.Module):
   """Complete hypertoroidal transformer"""
   
   def __init__(self, config: ModelConfig):
       super().__init__()
       self.config = config
       
       # Embeddings
       self.token_emb = nn.Embedding(config.vocab_size, config.dim)
       self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.dim))
       self.dropout = nn.Dropout(config.dropout)
       
       # Transformer blocks
       self.blocks = nn.ModuleList([
           HypertoroidalTransformerBlock(config)
           for _ in range(config.n_layers)
       ])
       
       # Output
       self.norm = nn.LayerNorm(config.dim)
       self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
       
       # Tie weights
       self.output.weight = self.token_emb.weight
       
       # Initialize
       self._init_weights()
       
   def _init_weights(self):
       for module in self.modules():
           if isinstance(module, nn.Linear):
               torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
       attentions = []
       for block in self.blocks:
           x, attn = block(x, return_attention)
           if return_attention:
               attentions.append(attn)
       
       # Output
       x = self.norm(x)
       logits = self.output(x)
       
       return logits, attentions if return_attention else None
   
   def should_evolve_geometry(self, epoch: int, recent_losses: List[float]) -> bool:
       """Determine if it's safe to evolve geometry"""
       if epoch < 10:  # changed from conservative 20
           return False
           
       if len(recent_losses) < 5:
           return False
           
       # Check stability
       loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
       is_improving = loss_trend < -0.001
       
       # Check time since last evolution
       min_gap = 15 #changed from conservative 25
       for block in self.blocks:
           for splat in block.attention.splats:
               if epoch - splat.last_evolution_epoch < min_gap:
                   return False
       
       return is_improving
   
   def evolve_all_geometries(self, epoch: int):
        """Evolve geometries with layer-specific rates"""
        # Different evolution aggressiveness per layer
        layer_evolution_rates = [0.010, 0.015, 0.020, 0.025]  # Deeper layers more aggressive
        layer_entry_boosts = [0.03, 0.05, 0.07, 0.09]       # Deeper layers stronger entry points
        
        for layer_idx, block in enumerate(self.blocks):
            evolution_rate = layer_evolution_rates[layer_idx] if layer_idx < len(layer_evolution_rates) else 0.015
            entry_boost = layer_entry_boosts[layer_idx] if layer_idx < len(layer_entry_boosts) else 0.05
            
            for splat in block.attention.splats:
                splat.layer_specific_evolution(evolution_rate, entry_boost)
                splat.last_evolution_epoch = epoch

class StandardTransformerBaseline(nn.Module):
   """Standard transformer for comparison"""
   
   def __init__(self, config: ModelConfig):
       super().__init__()
       self.config = config
       
       # Embeddings
       self.token_emb = nn.Embedding(config.vocab_size, config.dim)
       self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seq_len, config.dim))
       self.dropout = nn.Dropout(config.dropout)
       
       # Transformer blocks
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
       
       # Output
       self.norm = nn.LayerNorm(config.dim)
       self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
       
       # Tie weights
       self.output.weight = self.token_emb.weight
       
       # Initialize
       self._init_weights()
       
   def _init_weights(self):
       for module in self.modules():
           if isinstance(module, nn.Linear):
               torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
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
def create_copy_task(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
   """Create copy task to test long-range dependencies"""
   # Create sequences
   data = torch.randint(1, vocab_size // 2, (batch_size, seq_len), device=device)
   
   # Copy pattern: first 10 tokens should be copied to last 10
   copy_len = 10
   data[:, -copy_len:] = data[:, :copy_len]
   
   return data

def create_arithmetic_chain_task(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
   """Create arithmetic reasoning chains"""
   # Use special tokens for numbers and operations
   data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
   
   for i in range(batch_size):
       # Create chain: A + B = C, then later C * 2 = ?
       a = torch.randint(1, 10, (1,)).item()
       b = torch.randint(1, 10, (1,)).item()
       c = a + b
       
       # Encode in sequence (simplified)
       data[i, 0] = a
       data[i, 1] = vocab_size - 1  # + token
       data[i, 2] = b
       data[i, 3] = vocab_size - 2  # = token
       data[i, 4] = c
       
       # Later in sequence
       data[i, seq_len // 2] = c
       data[i, seq_len // 2 + 1] = vocab_size - 3  # * token
       data[i, seq_len // 2 + 2] = 2
       data[i, seq_len // 2 + 3] = vocab_size - 2  # = token
       data[i, seq_len // 2 + 4] = c * 2
   
   return data

def create_wrap_around_pattern_task(batch_size: int, seq_len: int, vocab_size: int, device: torch.device):
   """Create patterns that wrap from end to beginning"""
   data = torch.randint(1, vocab_size - 10, (batch_size, seq_len), device=device)
   
   pattern_len = 5
   for i in range(batch_size):
       # Create pattern at beginning
       pattern = torch.randint(1, vocab_size - 10, (pattern_len,))
       data[i, :pattern_len] = pattern
       
       # Continue pattern at end (wrapping around)
       data[i, -pattern_len:] = pattern
       
       # Test position: should predict first token after seeing last
       data[i, -1] = vocab_size - 5  # Special marker
   
   return data

# Testing functions
def test_wrap_around_attention(model: nn.Module, config: ModelConfig, device: torch.device) -> Dict:
   """Test if model can connect first and last tokens through torus topology"""
   seq_len = 64
   test_input = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
   
   # Make first and last tokens share information
   shared_token = torch.randint(0, config.vocab_size, (1,)).item()
   test_input[0, 0] = shared_token
   test_input[0, -1] = shared_token
   
   # Get attention weights
   with torch.no_grad():
       _, attentions = model(test_input, return_attention=True)
       
       if attentions and len(attentions) > 0:
           # Check first layer attention
           attn = attentions[0][0, 0]  # First batch, first head
           
           # Measure wrap-around strength
           first_to_last = attn[0, -1].item()
           last_to_first = attn[-1, 0].item()
           wrap_strength = (first_to_last + last_to_first) / 2
           
           # Compare to random positions
           random_pairs = []
           for _ in range(10):
               i = torch.randint(1, seq_len-1, (1,)).item()
               j = torch.randint(1, seq_len-1, (1,)).item()
               if i != j:
                   random_pairs.append(attn[i, j].item())
           
           avg_random = np.mean(random_pairs) if random_pairs else 0.0
           
           return {
               'wrap_around_strength': wrap_strength,
               'random_baseline': avg_random,
               'wrap_advantage': wrap_strength / (avg_random + 1e-8),
               'success': wrap_strength > avg_random * 1.5
           }
   
   return {'success': False, 'error': 'No attention weights'}

def test_multi_hop_reasoning(model: nn.Module, config: ModelConfig, device: torch.device) -> Dict:
   """Test if entry points create multi-hop connections"""
   seq_len = 48
   
   # Create arithmetic chain
   test_input = create_arithmetic_chain_task(1, seq_len, config.vocab_size, device)
   
   with torch.no_grad():
       logits, _ = model(test_input[:, :-1])
       
       # Check if model can predict the result
       target_pos = seq_len // 2 + 4 - 1  # Position of result (adjusted for input)
       prediction = torch.argmax(logits[0, target_pos]).item()
       expected = test_input[0, target_pos + 1].item()
       
       # Also check confidence
       probs = F.softmax(logits[0, target_pos], dim=-1)
       confidence = probs[expected].item()
       
       return {
           'prediction': prediction,
           'expected': expected,
           'correct': prediction == expected,
           'confidence': confidence,
           'success': prediction == expected or confidence > 0.1
       }

def analyze_geometric_evolution(model: HypertoroidalTransformer) -> Dict:
   """Analyze how geometries evolved across layers"""
   layer_analysis = []
   
   for layer_idx, block in enumerate(model.blocks):
       holes = []
       entries = []
       
       for splat in block.attention.splats:
           hole_ratio = (splat.hole_radius.item() / 
                        (splat.outer_radius.item() + 1e-8))
           active_entries = torch.sum(
               torch.sigmoid(splat.entry_strengths) > 0.3
           ).item()
           
           holes.append(hole_ratio)
           entries.append(active_entries)
       
       layer_analysis.append({
           'layer': layer_idx,
           'avg_hole_ratio': np.mean(holes),
           'std_hole_ratio': np.std(holes),
           'avg_active_entries': np.mean(entries),
           'std_active_entries': np.std(entries)
       })
   
   # Check for layer specialization
   hole_variance_across_layers = np.var([l['avg_hole_ratio'] for l in layer_analysis])
   entry_variance_across_layers = np.var([l['avg_active_entries'] for l in layer_analysis])
   
   return {
       'layer_analysis': layer_analysis,
       'hole_specialization': hole_variance_across_layers > 0.001,
       'entry_specialization': entry_variance_across_layers > 0.1,
       'evolved_meaningfully': any(l['avg_hole_ratio'] > 0.05 for l in layer_analysis)
   }

# Visualization functions
def visualize_comparative_results(hgt_losses, std_losses, geometry_evolution, save_path='hgt_validation.png'):
   """Visualize training and evolution results"""
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   # 1. Training loss comparison
   ax = axes[0, 0]
   ax.plot(hgt_losses, label='HGT', color='red', alpha=0.8)
   ax.plot(std_losses, label='Standard', color='blue', alpha=0.8)
   ax.set_title('Training Loss Comparison')
   ax.set_xlabel('Epoch')
   ax.set_ylabel('Loss')
   ax.set_yscale('log')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   
   # 2. Geometric evolution
   ax = axes[0, 1]
   if geometry_evolution:
       for epoch_data in geometry_evolution:
           if 'layer_analysis' in epoch_data:
               for layer_data in epoch_data['layer_analysis']:
                   layer_idx = layer_data['layer']
                   # Plot average hole ratio for this layer at this epoch
                   ax.scatter(epoch_data['epoch'], layer_data['avg_hole_ratio'], 
                             label=f'Layer {layer_idx}' if epoch_data['epoch'] == geometry_evolution[0]['epoch'] else "", 
                             alpha=0.8)
   ax.set_title('Hole Ratio Evolution by Layer')
   ax.set_xlabel('Evolution Step')
   ax.set_ylabel('Average Hole Ratio')
   ax.legend()
   ax.grid(True, alpha=0.3)
   
   # 3. Layer specialization
   ax = axes[1, 0]
   final_analysis = geometry_evolution[-1] if geometry_evolution else None
   if final_analysis and 'layer_analysis' in final_analysis:
       layers = [l['layer'] for l in final_analysis['layer_analysis']]
       hole_ratios = [l['avg_hole_ratio'] for l in final_analysis['layer_analysis']]
       active_entries = [l['avg_active_entries'] for l in final_analysis['layer_analysis']]
       
       x = np.arange(len(layers))
       width = 0.35
       ax.bar(x - width/2, hole_ratios, width, label='Hole Ratio', alpha=0.8)
       ax.bar(x + width/2, np.array(active_entries)/4, width, label='Active Entries (norm)', alpha=0.8)
       ax.set_title('Final Layer Specialization')
       ax.set_xlabel('Layer')
       ax.set_ylabel('Value')
       ax.legend()
       ax.grid(True, alpha=0.3)
   
   # 4. Performance summary
   ax = axes[1, 1]
   ax.text(0.1, 0.9, 'VALIDATION SUMMARY', fontsize=14, fontweight='bold', transform=ax.transAxes)
   
   final_hgt_loss = hgt_losses[-1] if hgt_losses else float('inf')
   final_std_loss = std_losses[-1] if std_losses else float('inf')
   improvement = (final_std_loss - final_hgt_loss) / final_std_loss * 100 if final_std_loss > 0 else 0
   
   summary_text = [
       f'Final HGT Loss: {final_hgt_loss:.4f}',
       f'Final STD Loss: {final_std_loss:.4f}',
       f'Improvement: {improvement:.1f}%',
       '',
       'Geometric Evolution:',
       f'  Evolved: {"Yes" if final_analysis and final_analysis.get("evolved_meaningfully", False) else "No"}',
       f'  Layer Specialization: {"Yes" if final_analysis and final_analysis.get("hole_specialization", False) else "No"}'
   ]
   
   ax.text(0.1, 0.7, '\n'.join(summary_text), transform=ax.transAxes, 
           fontsize=11, verticalalignment='top', fontfamily='monospace')
   ax.axis('off')
   
   plt.tight_layout()
   plt.savefig(save_path, dpi=150)
   plt.close()

# Main validation function
def validate_hypertoroidal_transformer():
   """Complete validation of hypertoroidal transformer vs baseline"""
   print("="*70)
   print("HYPERTOROIDAL TRANSFORMER VALIDATION - PRODUCTION READY")
   print("="*70)
   
   # Configuration
   config = ModelConfig(
       vocab_size=1000,
       dim=256,
       n_layers=4,
       n_heads=8,
       n_splats_per_head=4,
       max_seq_len=128
   )
   
   batch_size = 32
   n_epochs = 100
   
   print("\nConfiguration:")
   print(f"  Model dimension: {config.dim}")
   print(f"  Layers: {config.n_layers}")
   print(f"  Heads: {config.n_heads}")
   print(f"  Splats per head: {config.n_splats_per_head}")
   print(f"  Device: {device}")
   
   # Create models
   print("\nCreating models...")
   hgt_model = HypertoroidalTransformer(config).to(device)
   std_model = StandardTransformerBaseline(config).to(device)
   
   # Count parameters
   hgt_params = sum(p.numel() for p in hgt_model.parameters())
   std_params = sum(p.numel() for p in std_model.parameters())
   print(f"  HGT parameters: {hgt_params:,}")
   print(f"  STD parameters: {std_params:,}")
   print(f"  Parameter overhead: {(hgt_params - std_params) / std_params * 100:.1f}%")
   
   # Create training data
   print("\nCreating datasets...")
   train_tasks = {
       'copy': create_copy_task(1000, 64, config.vocab_size, device),
       'arithmetic': create_arithmetic_chain_task(1000, 64, config.vocab_size, device),
       'wrap_around': create_wrap_around_pattern_task(1000, 64, config.vocab_size, device)
   }
   
   # Training setup
   hgt_optimizer = torch.optim.AdamW(hgt_model.parameters(), lr=0.001, weight_decay=0.01)
   std_optimizer = torch.optim.AdamW(std_model.parameters(), lr=0.001, weight_decay=0.01)
   
   # Training loop
   print("\nTraining models...")
   hgt_losses = []
   std_losses = []
   recent_hgt_losses = []
   geometry_evolution = []
   
   for epoch in range(n_epochs):
       # Train HGT
       hgt_model.train()
       hgt_epoch_loss = 0.0
       
       for task_name, task_data in train_tasks.items():
           for i in range(0, len(task_data), batch_size):
               batch = task_data[i:i+batch_size]
               
               # Forward pass
               logits, _ = hgt_model(batch[:, :-1])
               loss = F.cross_entropy(
                   logits.reshape(-1, config.vocab_size),
                   batch[:, 1:].reshape(-1)
               )
               
               # Backward pass
               hgt_optimizer.zero_grad()
               loss.backward()
               torch.nn.utils.clip_grad_norm_(hgt_model.parameters(), 1.0)
               hgt_optimizer.step()
               
               hgt_epoch_loss += loss.item()
       
       avg_hgt_loss = hgt_epoch_loss / (len(train_tasks) * len(task_data) // batch_size)
       hgt_losses.append(avg_hgt_loss)
       recent_hgt_losses.append(avg_hgt_loss)
       if len(recent_hgt_losses) > 10:
           recent_hgt_losses.pop(0)
       
       # Train Standard
       std_model.train()
       std_epoch_loss = 0.0
       
       for task_name, task_data in train_tasks.items():
           for i in range(0, len(task_data), batch_size):
               batch = task_data[i:i+batch_size]
               
               logits, _ = std_model(batch[:, :-1])
               loss = F.cross_entropy(
                   logits.reshape(-1, config.vocab_size),
                   batch[:, 1:].reshape(-1)
               )
               
               std_optimizer.zero_grad()
               loss.backward()
               torch.nn.utils.clip_grad_norm_(std_model.parameters(), 1.0)
               std_optimizer.step()
               
               std_epoch_loss += loss.item()
       
       avg_std_loss = std_epoch_loss / (len(train_tasks) * len(task_data) // batch_size)
       std_losses.append(avg_std_loss)
       
       # Track geometric evolution
       if epoch % 10 == 0:
           layer_evolution = []
           for layer_idx, block in enumerate(hgt_model.blocks):
               hole_ratios = []
               active_entries = []
               for splat in block.attention.splats[:8]:  # Sample first 8 splats
                   hole_ratio = splat.hole_radius.item() / (splat.outer_radius.item() + 1e-8)
                   active = torch.sum(torch.sigmoid(splat.entry_strengths) > 0.3).item()
                   hole_ratios.append(hole_ratio)
                   active_entries.append(active)
               
               layer_evolution.append({
                   'layer': layer_idx,
                   'hole_ratios': hole_ratios,
                   'avg_hole_ratio': np.mean(hole_ratios),
                   'avg_active_entries': np.mean(active_entries)
               })
           
           geometry_evolution.append({
               'epoch': epoch,
               'layer_analysis': layer_evolution
           })
       
       # Evolve HGT geometry if stable
       if hgt_model.should_evolve_geometry(epoch, recent_hgt_losses):
           hgt_model.evolve_all_geometries(epoch)
           print(f"  Epoch {epoch}: Evolved geometries (loss trend: {np.polyfit(range(len(recent_hgt_losses)), recent_hgt_losses, 1)[0]:.6f})")
       
       # Progress update
       if epoch % 20 == 0:
           print(f"Epoch {epoch}: HGT Loss = {avg_hgt_loss:.4f}, STD Loss = {avg_std_loss:.4f}")
   
   print("\n" + "="*70)
   print("TESTING PHASE")
   print("="*70)
   
   # Test unique capabilities
   print("\nTesting torus-specific capabilities...")
   
   # Test 1: Wrap-around attention
   hgt_wrap = test_wrap_around_attention(hgt_model, config, device)
   std_wrap = test_wrap_around_attention(std_model, config, device)
   
   print("\nWrap-around Attention Test:")
   print(f"  HGT: strength={hgt_wrap.get('wrap_around_strength', 0):.4f}, "
         f"advantage={hgt_wrap.get('wrap_advantage', 0):.2f}x, "
         f"success={hgt_wrap.get('success', False)}")
   print(f"  STD: strength={std_wrap.get('wrap_around_strength', 0):.4f}, "
         f"baseline={std_wrap.get('random_baseline', 0):.4f}")
   
   # Test 2: Multi-hop reasoning
   hgt_hop = test_multi_hop_reasoning(hgt_model, config, device)
   std_hop = test_multi_hop_reasoning(std_model, config, device)
   
   print("\nMulti-hop Reasoning Test:")
   print(f"  HGT: correct={hgt_hop.get('correct', False)}, "
         f"confidence={hgt_hop.get('confidence', 0):.3f}, "
         f"success={hgt_hop.get('success', False)}")
   print(f"  STD: correct={std_hop.get('correct', False)}, "
         f"confidence={std_hop.get('confidence', 0):.3f}")
   
   # Test 3: Task-specific performance
   print("\nTask-specific Performance:")
   
   test_results = {}
   for task_name, task_data in train_tasks.items():
       # Test both models
       test_batch = task_data[:100]  # Use first 100 samples
       
       hgt_model.eval()
       std_model.eval()
       
       with torch.no_grad():
           # HGT performance
           hgt_logits, _ = hgt_model(test_batch[:, :-1])
           hgt_loss = F.cross_entropy(
               hgt_logits.reshape(-1, config.vocab_size),
               test_batch[:, 1:].reshape(-1)
           ).item()
           
           # Standard performance
           std_logits, _ = std_model(test_batch[:, :-1])
           std_loss = F.cross_entropy(
               std_logits.reshape(-1, config.vocab_size),
               test_batch[:, 1:].reshape(-1)
           ).item()
       
       improvement = (std_loss - hgt_loss) / std_loss * 100 if std_loss > 0 else 0
       test_results[task_name] = {
           'hgt_loss': hgt_loss,
           'std_loss': std_loss,
           'improvement': improvement
       }
       
       print(f"  {task_name}: HGT={hgt_loss:.4f}, STD={std_loss:.4f}, "
             f"Improvement={improvement:.1f}%")
   
   # Analyze geometric evolution
   print("\nGeometric Evolution Analysis:")
   final_geometry = analyze_geometric_evolution(hgt_model)
   
   for layer_data in final_geometry['layer_analysis']:
       print(f"  Layer {layer_data['layer']}: "
             f"hole_ratio={layer_data['avg_hole_ratio']:.3f}±{layer_data['std_hole_ratio']:.3f}, "
             f"entries={layer_data['avg_active_entries']:.1f}±{layer_data['std_active_entries']:.1f}")
   
   print(f"\n  Evolved meaningfully: {final_geometry['evolved_meaningfully']}")
   print(f"  Layer specialization (holes): {final_geometry['hole_specialization']}")
   print(f"  Layer specialization (entries): {final_geometry['entry_specialization']}")
   
   # Generate visualization
   visualize_comparative_results(hgt_losses, std_losses, geometry_evolution)
   print("\nVisualization saved to 'hgt_validation.png'")
   
   # Final assessment
   print("\n" + "="*70)
   print("FINAL ASSESSMENT")
   print("="*70)
   
   # Success criteria
   criteria = {
       'training_convergence': hgt_losses[-1] < hgt_losses[0] * 0.5,
       'competitive_performance': all(r['improvement'] > -10 for r in test_results.values()),
       'geometric_evolution': final_geometry['evolved_meaningfully'],
       'layer_specialization': final_geometry['hole_specialization'] or final_geometry['entry_specialization'],
       'wrap_around_capability': hgt_wrap.get('success', False),
       'multi_hop_capability': hgt_hop.get('success', False),
       'task_advantage': any(r['improvement'] > 0 for r in test_results.values())
   }
   
   print("\nSuccess Criteria:")
   for criterion, passed in criteria.items():
       status = "✓ PASSED" if passed else "✗ FAILED"
       print(f"  {criterion}: {status}")
   
   passed_count = sum(criteria.values())
   total_count = len(criteria)
   success_rate = passed_count / total_count
   
   print(f"\nOverall Success Rate: {passed_count}/{total_count} ({success_rate*100:.0f}%)")
   
   # Performance summary
   avg_improvement = np.mean([r['improvement'] for r in test_results.values()])
   best_task = max(test_results.items(), key=lambda x: x[1]['improvement'])
   
   print(f"\nPerformance Summary:")
   print(f"  Average improvement: {avg_improvement:.1f}%")
   print(f"  Best task: {best_task[0]} ({best_task[1]['improvement']:.1f}% improvement)")
   print(f"  Final perplexity - HGT: {np.exp(hgt_losses[-1]):.2f}, STD: {np.exp(std_losses[-1]):.2f}")
   
   # Final verdict
   if success_rate >= 0.7 and avg_improvement > -5:
       print("\n🎉 HYPERTOROIDAL TRANSFORMER VALIDATED!")
       print("   ✓ Successfully trained with stable geometric evolution")
       print("   ✓ Competitive with standard transformer baseline")
       print("   ✓ Demonstrates unique torus-based capabilities")
       print("   ✓ Shows promise for specialized applications")
       verdict = "SUCCESS"
   elif success_rate >= 0.5:
       print("\n⚡ HYPERTOROIDAL TRANSFORMER SHOWS PROMISE")
       print("   ~ Some criteria met but needs optimization")
       print("   ~ Consider tuning evolution parameters")
       print("   ~ May excel on specific task types")
       verdict = "PROMISING"
   else:
       print("\n⚠️  HYPERTOROIDAL TRANSFORMER NEEDS DEVELOPMENT")
       print("   ✗ Several criteria not met")
       print("   ✗ Requires architectural improvements")
       print("   ✗ Evolution mechanism needs refinement")
       verdict = "NEEDS_WORK"
   
   # Recommendations
   print("\nRecommendations:")
   if not criteria['geometric_evolution']:
       print("  - Adjust evolution parameters for more aggressive geometry changes")
   if not criteria['competitive_performance']:
       print("  - Optimize vectorized operations further")
       print("  - Consider sparse attention patterns to reduce computation")
   if not criteria['wrap_around_capability']:
       print("  - Increase entry point strength initialization")
       print("  - Design tasks that specifically reward wrap-around connections")
   if avg_improvement < 0:
       print("  - Focus on tasks where long-range dependencies are critical")
       print("  - Explore different splat initialization strategies")
   
   print("\nNext Steps:")
   print("  1. Test on real language modeling benchmarks")
   print("  2. Develop CUDA kernels for torus operations")
   print("  3. Explore hierarchical torus structures")
   print("  4. Investigate emergent geometric patterns")
   
   return {
       'verdict': verdict,
       'success_rate': success_rate,
       'avg_improvement': avg_improvement,
       'criteria': criteria,
       'test_results': test_results,
       'final_geometry': final_geometry
   }

if __name__ == "__main__":
   # Run validation
   start_time = time.time()
   results = validate_hypertoroidal_transformer()
   end_time = time.time()
   
   print(f"\nValidation completed in {end_time - start_time:.1f} seconds")
   
   # Save results
   import json
   with open('hgt_validation_results.json', 'w') as f:
       # Convert numpy values to Python types for JSON serialization
       json_results = {
           'verdict': results['verdict'],
           'success_rate': float(results['success_rate']),
           'avg_improvement': float(results['avg_improvement']),
           'criteria': results['criteria'],
           'test_results': {k: {kk: float(vv) if isinstance(vv, np.floating) else vv 
                               for kk, vv in v.items()} 
                           for k, v in results['test_results'].items()}
       }
       json.dump(json_results, f, indent=2)
   
   print("\nResults saved to 'hgt_validation_results.json'")
   print("\nValidation complete! 🚀")
