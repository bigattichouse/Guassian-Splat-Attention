# Simplified GSA Implementation - Reversion Prompt

## Context Summary

I'm working on **Gaussian Splat Attention (GSA)** and need to revert from a complex hypertoroidal implementation back to a **simplified, efficient version**. The hypertoroidal approach showed promise but is overengineered - the core benefits likely come from learned splat positioning, not complex torus topology.

## Current Status
- ✅ **Hypertoroidal version works**: 2-5% improvements over standard attention
- ⚠️ **Too complex**: Torus geometry, entry points, evolution mechanisms
- ⚠️ **Too slow**: 60 seconds/epoch vs 6 seconds for standard attention
- 🎯 **Need**: Simplified version that captures benefits with 10x better efficiency

## Core Insight

The performance gains likely come from:
1. **Learned splat positions** creating better attention patterns
2. **Multiple splats per head** providing richer representations
3. **Amplitude learning** for automatic pruning

**NOT from**:
- Complex torus surface calculations
- Entry point wormholes  
- Geometric evolution mechanisms

## Implementation Requirements

### Simplified GSA Architecture

Replace the complex `VectorizedHypertoroidalSplat` with:

```python
class SimplifiedGSASplat(nn.Module):
    """
    Simple, efficient Gaussian splat with scalar variance.
    3 parameters per splat: center, scale, amplitude
    """
    def __init__(self, dim: int):
        super().__init__()
        self.center = nn.Parameter(torch.randn(dim) * 0.1)          # Position in embedding space
        self.log_scale = nn.Parameter(torch.zeros(1))               # Log of scalar variance
        self.amplitude = nn.Parameter(torch.tensor(1.0))            # Contribution strength
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute attention through simple Gaussian splat.
        queries: [batch, seq_len, dim]
        keys: [batch, seq_len, dim]
        returns: [batch, seq_len, seq_len]
        """
        scale = torch.exp(self.log_scale) + 1e-6  # Ensure positive
        
        # Vectorized distance computation
        q_dists = torch.cdist(queries, self.center.unsqueeze(0).unsqueeze(0))  # [B, T, 1]
        k_dists = torch.cdist(keys, self.center.unsqueeze(0).unsqueeze(0))     # [B, T, 1]
        
        # Gaussian weights
        q_weights = torch.exp(-0.5 * (q_dists / scale) ** 2)  # [B, T, 1]
        k_weights = torch.exp(-0.5 * (k_dists / scale) ** 2)  # [B, T, 1]
        
        # Attention matrix through splat
        attention = self.amplitude * (q_weights @ k_weights.transpose(-2, -1))  # [B, T, T]
        return attention.squeeze(-1).squeeze(-1)  # [B, T, T]
```

### Simplified GSA Attention Layer

Replace `VectorizedHypertoroidalAttention` with:

```python
class SimplifiedGSAAttention(nn.Module):
    """
    Multi-head attention using simple Gaussian splats.
    Much faster than hypertoroidal version.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.dim // config.n_heads
        
        # Create simple splats for each head
        self.splats = nn.ModuleList([
            SimplifiedGSASplat(self.head_dim)
            for _ in range(config.n_heads * config.n_splats_per_head)
        ])
        
        # Standard projections
        self.qkv = nn.Linear(config.dim, 3 * config.dim)
        self.out_proj = nn.Linear(config.dim, config.dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = x.shape
        
        # Project to Q, K, V and reshape for multi-head
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        k = k.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention for each head using splats
        attention_weights = torch.zeros(B, self.config.n_heads, T, T, device=x.device)
        
        for h in range(self.config.n_heads):
            head_attention = torch.zeros(B, T, T, device=x.device)
            
            # Sum contributions from all splats for this head
            for s in range(self.config.n_splats_per_head):
                splat_idx = h * self.config.n_splats_per_head + s
                splat = self.splats[splat_idx]
                
                # Simple splat attention - much faster than torus computation
                splat_attention = splat(q[:, h], k[:, h])  # [B, T, T]
                head_attention += splat_attention / self.config.n_splats_per_head
            
            attention_weights[:, h] = head_attention
        
        # Apply softmax and dropout
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, v)  # [B, H, T, D/H]
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        
        return self.out_proj(output), attention_weights
```

### Simplified Transformer Block

Replace `HypertoroidalTransformerBlock` with:

```python
class SimplifiedGSATransformerBlock(nn.Module):
    """Transformer block using simplified GSA - no evolution mechanisms"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
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
        attn_out, attn_weights = self.attention(self.norm1(x))
        x = x + self.dropout(attn_out)
        
        # MLP block  
        x = x + self.dropout(self.mlp(self.norm2(x)))
        
        return x, attn_weights if return_attention else None
```

## Key Changes to Make

### 1. **Remove All Hypertoroidal Complexity**
- No `hole_radius`, `entry_points`, `entry_strengths`
- No geometric evolution mechanisms
- No torus distance calculations
- No `evolve_all_geometries()` methods

### 2. **Keep the Validation Framework**
- Same test tasks (copy, arithmetic, wrap_around)
- Same baseline comparison with `StandardTransformerBaseline`
- Same performance metrics and visualization
- Same overall validation structure

### 3. **Update Class Names**
Replace in the main transformer:
- `HypertoroidalTransformerBlock` → `SimplifiedGSATransformerBlock`
- `VectorizedHypertoroidalAttention` → `SimplifiedGSAAttention`
- Remove all evolution-related code

### 4. **Expected Performance**
- **Speed**: 5-10x faster training (6-12 seconds per epoch instead of 60)
- **Performance**: Should match or exceed hypertoroidal results (2-5% improvement)
- **Memory**: Lower memory usage due to fewer parameters
- **Simplicity**: Much easier to debug and optimize

## Validation Goals

The simplified version should demonstrate:
1. **Competitive performance** with hypertoroidal version
2. **Much faster training** (sub-15 second epochs)
3. **Same or better task improvements** (2-5% over standard attention)
4. **Cleaner, more interpretable** attention patterns

## Success Criteria

**If simplified GSA achieves similar performance with 5-10x speedup, it proves:**
- Torus complexity was unnecessary
- Core benefits come from learned splat positioning
- Simple approach is more practical for real applications

## File Structure

When implementing:
1. **Keep same config and validation framework**
2. **Replace hypertoroidal classes with simplified versions**
3. **Remove all evolution mechanisms**
4. **Maintain same experimental rigor**

The goal is to prove that **simpler is better** - that we can get the benefits of splat attention without the geometric complexity.

---

**Implementation Strategy**: Strip out all hypertoroidal complexity, implement simple Gaussian splats with scalar variances, and prove the core concept works better with minimal complexity.

**Expected Outcome**: Faster, simpler, equally effective GSA that's actually practical for real applications.
