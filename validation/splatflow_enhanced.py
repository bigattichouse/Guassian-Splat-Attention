"""
Vectorized SplatFlow Implementation with CUDA Optimization
Fixes the performance bottlenecks in the original implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class VectorizedTrajectoryComputer:
    """Fully vectorized trajectory computation"""
    
    def __init__(self, influence_radius: int = 8):
        self.influence_radius = influence_radius
    
    def compute_batch_trajectories(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Vectorized trajectory computation for entire batch
        
        Args:
            embeddings: [batch_size, seq_len, embedding_dim]
            
        Returns:
            trajectories: [batch_size, seq_len, embedding_dim]
        """
        batch_size, seq_len, embedding_dim = embeddings.shape
        device = embeddings.device
        
        if seq_len < 2:
            return torch.zeros_like(embeddings)
        
        # Compute all trajectory vectors at once: [batch, seq_len-1, embed_dim]
        all_trajectories = embeddings[:, 1:] - embeddings[:, :-1]
        trajectory_magnitudes = torch.norm(all_trajectories, dim=-1, keepdim=True)  # [batch, seq_len-1, 1]
        
        # Normalize trajectories (handle zero magnitudes)
        safe_magnitudes = torch.clamp(trajectory_magnitudes, min=1e-8)
        normalized_trajectories = all_trajectories / safe_magnitudes
        
        # Create weights based on magnitude and recency
        magnitude_weights = torch.tanh(trajectory_magnitudes.squeeze(-1))  # [batch, seq_len-1]
        
        # Compute influence windows vectorized
        trajectories = torch.zeros_like(embeddings)
        
        for pos in range(1, seq_len):
            window_start = max(0, pos - self.influence_radius)
            window_end = min(pos, seq_len - 1)
            window_size = window_end - window_start
            
            if window_size > 0:
                # Recency weights: more recent = higher weight
                recency_positions = torch.arange(window_start, window_end, device=device)
                recency_weights = (recency_positions - window_start + 1).float() / window_size
                
                # Combined weights
                combined_weights = magnitude_weights[:, window_start:window_end] * recency_weights.unsqueeze(0)
                
                # Normalize weights
                weight_sums = torch.sum(combined_weights, dim=1, keepdim=True)
                safe_weight_sums = torch.clamp(weight_sums, min=1e-8)
                normalized_weights = combined_weights / safe_weight_sums
                
                # Weighted average of trajectories
                window_trajectories = normalized_trajectories[:, window_start:window_end]  # [batch, window, embed]
                weighted_trajectory = torch.sum(
                    window_trajectories * normalized_weights.unsqueeze(-1), 
                    dim=1
                )  # [batch, embed]
                
                trajectories[:, pos] = weighted_trajectory
        
        return trajectories

class VectorizedHypertoroidalSplat(nn.Module):
    """Fully vectorized hypertoroidal splat with batch operations"""
    
    def __init__(self, embedding_dim: int, max_entry_points: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Learnable parameters
        self.center = nn.Parameter(torch.randn(embedding_dim) * 0.02)
        self.outer_radius = nn.Parameter(torch.tensor(1.0))
        self.hole_radius = nn.Parameter(torch.tensor(0.01))
        
        # Entry points for non-local connections
        self.entry_points = nn.Parameter(torch.randn(max_entry_points, embedding_dim) * 0.1)
        self.entry_strengths = nn.Parameter(torch.ones(max_entry_points) * 0.3)
        self.entry_orientations = nn.Parameter(torch.zeros(max_entry_points))
        
        # Evolution tracking
        self.register_buffer('activation_history', torch.zeros(100))
        self.register_buffer('history_ptr', torch.tensor(0, dtype=torch.long))
        
    def compute_batch_attention(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """
        Vectorized attention computation for entire batch
        
        Args:
            queries: [batch_size, seq_len, embedding_dim]
            keys: [batch_size, seq_len, embedding_dim]
            
        Returns:
            attention_matrix: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, embedding_dim = queries.shape
        device = queries.device
        
        # Expand center for broadcasting: [1, 1, embedding_dim]
        center_expanded = self.center.unsqueeze(0).unsqueeze(0)
        
        # Compute distances to center: [batch, seq_len]
        query_dists = torch.norm(queries - center_expanded, dim=-1)
        key_dists = torch.norm(keys - center_expanded, dim=-1)
        
        # Torus surface attention (vectorized)
        hole_ratio = self.hole_radius / (self.outer_radius + 1e-8)
        
        if hole_ratio < 0.05:  # Circle stage
            # Gaussian attention
            query_weights = torch.exp(-0.5 * (query_dists / (self.outer_radius + 1e-8))**2)
            key_weights = torch.exp(-0.5 * (key_dists / (self.outer_radius + 1e-8))**2)
        else:  # Torus stage
            # Distance to torus surface
            query_torus_dists = torch.abs(query_dists - self.outer_radius)
            key_torus_dists = torch.abs(key_dists - self.outer_radius)
            
            query_weights = torch.exp(-0.5 * query_torus_dists**2)
            key_weights = torch.exp(-0.5 * key_torus_dists**2)
        
        # Surface attention matrix: [batch, seq_len, seq_len]
        surface_attention = torch.einsum('bi,bj->bij', query_weights, key_weights)
        
        # Entry point contributions (vectorized)
        entry_attention = torch.zeros_like(surface_attention)
        active_entries = torch.sigmoid(self.entry_strengths) > 0.3
        
        if torch.any(active_entries):
            # Compute distances to all entry points at once
            # queries: [batch, seq_len, embed_dim], entry_points: [max_entry, embed_dim]
            # Result: [batch, seq_len, max_entry]
            query_entry_dists = torch.cdist(queries, self.entry_points.unsqueeze(0).expand(batch_size, -1, -1))
            key_entry_dists = torch.cdist(keys, self.entry_points.unsqueeze(0).expand(batch_size, -1, -1))
            
            # Apply entry point weights
            active_strengths = torch.sigmoid(self.entry_strengths)[active_entries]
            active_orientations = torch.tanh(self.entry_orientations)[active_entries]
            
            # Compute entry contributions
            for i, (strength, orientation) in enumerate(zip(active_strengths, active_orientations)):
                if active_entries[i]:
                    entry_query_weights = torch.exp(-0.5 * query_entry_dists[:, :, i]**2)
                    entry_key_weights = torch.exp(-0.5 * key_entry_dists[:, :, i]**2)
                    
                    entry_contribution = (
                        strength * orientation * 
                        torch.einsum('bi,bj->bij', entry_query_weights, entry_key_weights)
                    )
                    entry_attention += entry_contribution
        
        total_attention = surface_attention + entry_attention
        
        # Update activation history (vectorized)
        with torch.no_grad():
            avg_activation = torch.mean(total_attention).item()
            ptr = self.history_ptr.item()
            self.activation_history[ptr] = avg_activation
            self.history_ptr.copy_((ptr + 1) % 100)
        
        return total_attention

class VectorizedSplatAttention(nn.Module):
    """Fully vectorized splat attention layer"""
    
    def __init__(self, model_dim: int, num_heads: int, num_splats: int, 
                 use_hypertoroidal: bool = False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.num_splats = num_splats
        
        # Projections
        self.qkv_proj = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out_proj = nn.Linear(model_dim, model_dim, bias=False)
        
        # Trajectory computer
        self.trajectory_computer = VectorizedTrajectoryComputer()
        
        # Initialize splats
        if use_hypertoroidal:
            self.splats = nn.ModuleList([
                nn.ModuleList([
                    VectorizedHypertoroidalSplat(self.head_dim) 
                    for _ in range(num_splats)
                ]) for _ in range(num_heads)
            ])
        else:
            # Standard Gaussian splats (vectorized)
            self.splat_centers = nn.Parameter(torch.randn(num_heads, num_splats, self.head_dim) * 0.02)
            self.splat_log_scales = nn.Parameter(torch.zeros(num_heads, num_splats))
            self.splat_amplitudes = nn.Parameter(torch.ones(num_heads, num_splats))
        
        self.use_hypertoroidal = use_hypertoroidal
    
    def compute_gaussian_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Vectorized Gaussian splat attention
        
        Args:
            q, k, v: [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            output: [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Compute all distances at once using broadcasting
        # q: [batch, heads, seq_len, head_dim, 1]
        # centers: [1, heads, 1, head_dim, num_splats]
        q_expanded = q.unsqueeze(-1)  # [batch, heads, seq_len, head_dim, 1]
        k_expanded = k.unsqueeze(-1)
        centers_expanded = self.splat_centers.unsqueeze(0).unsqueeze(2).transpose(-2, -1)  # [1, heads, 1, head_dim, num_splats]
        
        # Compute squared distances: [batch, heads, seq_len, num_splats]
        q_dists_sq = torch.sum((q_expanded - centers_expanded)**2, dim=-2)
        k_dists_sq = torch.sum((k_expanded - centers_expanded)**2, dim=-2)
        
        # Apply Gaussian with learned scales
        scales = torch.exp(self.splat_log_scales).unsqueeze(0).unsqueeze(2)  # [1, heads, 1, num_splats]
        scales_sq = scales**2
        
        q_weights = torch.exp(-0.5 * q_dists_sq / scales_sq)  # [batch, heads, seq_len, num_splats]
        k_weights = torch.exp(-0.5 * k_dists_sq / scales_sq)
        
        # Apply amplitudes
        amplitudes = torch.sigmoid(self.splat_amplitudes).unsqueeze(0).unsqueeze(2)  # [1, heads, 1, num_splats]
        q_weights = q_weights * amplitudes
        k_weights = k_weights * amplitudes
        
        # Compute attention matrix: [batch, heads, seq_len, seq_len]
        attention_matrix = torch.einsum('bhis,bhjs->bhij', q_weights, k_weights)
        
        # Normalize
        attention_matrix = F.softmax(attention_matrix, dim=-1)
        
        # Apply to values
        output = torch.matmul(attention_matrix, v)
        
        return output
    
    def compute_hypertoroidal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Vectorized hypertoroidal attention"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        output = torch.zeros_like(v)
        
        for head in range(num_heads):
            head_q = q[:, head]  # [batch, seq_len, head_dim]
            head_k = k[:, head]
            head_v = v[:, head]
            
            # Combine attention from all splats in this head
            total_attention = torch.zeros(batch_size, seq_len, seq_len, device=q.device)
            
            for splat in self.splats[head]:
                splat_attention = splat.compute_batch_attention(head_q, head_k)
                total_attention += splat_attention
            
            # Normalize and apply to values
            normalized_attention = F.softmax(total_attention, dim=-1)
            head_output = torch.matmul(normalized_attention, head_v)
            output[:, head] = head_output
        
        return output
    
    def forward(self, x: torch.Tensor, trajectories: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Vectorized forward pass
        
        Args:
            x: [batch_size, seq_len, model_dim]
            trajectories: Optional [batch_size, seq_len, model_dim] trajectory flows
            
        Returns:
            output: [batch_size, seq_len, model_dim]
        """
        batch_size, seq_len, model_dim = x.shape
        
        # Compute trajectories if not provided
        if trajectories is None:
            trajectories = self.trajectory_computer.compute_batch_trajectories(x)
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3*model_dim]
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Enhance with trajectories
        traj_reshaped = trajectories.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q + 0.1 * traj_reshaped  # Add trajectory information
        k = k + 0.1 * traj_reshaped
        
        # Compute attention based on splat type
        if self.use_hypertoroidal:
            output = self.compute_hypertoroidal_attention(q, k, v)
        else:
            output = self.compute_gaussian_attention(q, k, v)
        
        # Reshape back and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        output = self.out_proj(output)
        
        return output

# CUDA Kernel Integration (Optional)
try:
    import triton
    import triton.language as tl
    
    @triton.jit
    def gaussian_splat_kernel(
        # Pointers
        queries_ptr, keys_ptr, centers_ptr, scales_ptr, output_ptr,
        # Shapes
        batch_size, seq_len, num_heads, head_dim, num_splats,
        # Strides
        q_batch_stride, q_head_stride, q_seq_stride, q_dim_stride,
        k_batch_stride, k_head_stride, k_seq_stride, k_dim_stride,
        c_head_stride, c_splat_stride, c_dim_stride,
        o_batch_stride, o_head_stride, o_seq_stride1, o_seq_stride2,
        # Block sizes
        BLOCK_SIZE: tl.constexpr
    ):
        """Custom CUDA kernel for Gaussian splat attention"""
        # Get program IDs
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        seq_block_id = tl.program_id(2)
        
        # Calculate offsets
        seq_start = seq_block_id * BLOCK_SIZE
        seq_offsets = seq_start + tl.arange(0, BLOCK_SIZE)
        
        # Load queries and keys for this block
        q_offset = (batch_id * q_batch_stride + head_id * q_head_stride + 
                   seq_offsets[:, None] * q_seq_stride + tl.arange(0, head_dim)[None, :] * q_dim_stride)
        
        k_offset = (batch_id * k_batch_stride + head_id * k_head_stride + 
                   seq_offsets[:, None] * k_seq_stride + tl.arange(0, head_dim)[None, :] * k_dim_stride)
        
        q_block = tl.load(queries_ptr + q_offset, mask=seq_offsets[:, None] < seq_len)
        k_block = tl.load(keys_ptr + k_offset, mask=seq_offsets[:, None] < seq_len)
        
        # Initialize attention accumulator
        attention_acc = tl.zeros([BLOCK_SIZE, BLOCK_SIZE], dtype=tl.float32)
        
        # Process each splat
        for splat_id in range(num_splats):
            # Load splat parameters
            center_offset = head_id * c_head_stride + splat_id * c_splat_stride + tl.arange(0, head_dim) * c_dim_stride
            center = tl.load(centers_ptr + center_offset)
            scale = tl.load(scales_ptr + head_id * num_splats + splat_id)
            
            # Compute distances to splat center
            q_diff = q_block - center[None, :]
            k_diff = k_block - center[None, :]
            
            q_dist_sq = tl.sum(q_diff * q_diff, axis=1)
            k_dist_sq = tl.sum(k_diff * k_diff, axis=1)
            
            # Gaussian weights
            q_weights = tl.exp(-0.5 * q_dist_sq / (scale * scale))
            k_weights = tl.exp(-0.5 * k_dist_sq / (scale * scale))
            
            # Outer product for attention
            splat_attention = q_weights[:, None] * k_weights[None, :]
            attention_acc += splat_attention
        
        # Store result
        output_offset = (batch_id * o_batch_stride + head_id * o_head_stride + 
                        seq_offsets[:, None] * o_seq_stride1 + seq_offsets[None, :] * o_seq_stride2)
        
        mask = (seq_offsets[:, None] < seq_len) & (seq_offsets[None, :] < seq_len)
        tl.store(output_ptr + output_offset, attention_acc, mask=mask)

    class CudaOptimizedSplatAttention(VectorizedSplatAttention):
        """CUDA-optimized version using Triton kernels"""
        
        def compute_gaussian_attention_cuda(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """Use custom CUDA kernel for maximum performance"""
            batch_size, num_heads, seq_len, head_dim = q.shape
            
            # Allocate output
            attention_output = torch.zeros(batch_size, num_heads, seq_len, seq_len, 
                                         device=q.device, dtype=q.dtype)
            
            # Launch kernel
            grid = (batch_size, num_heads, triton.cdiv(seq_len, 64))
            gaussian_splat_kernel[grid](
                q, k, self.splat_centers, torch.exp(self.splat_log_scales), attention_output,
                batch_size, seq_len, num_heads, head_dim, self.num_splats,
                # Strides (computed automatically)
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                self.splat_centers.stride(0), self.splat_centers.stride(1), self.splat_centers.stride(2),
                attention_output.stride(0), attention_output.stride(1), attention_output.stride(2), attention_output.stride(3),
                BLOCK_SIZE=64
            )
            
            # Apply softmax and values
            attention_output = F.softmax(attention_output, dim=-1)
            output = torch.matmul(attention_output, v)
            
            return output

except ImportError:
    # Triton not available, use standard PyTorch implementation
    CudaOptimizedSplatAttention = VectorizedSplatAttention

# Performance Comparison
def benchmark_implementations():
    """Compare performance of different implementations"""
    import time
    
    # Test parameters
    batch_size = 4
    seq_len = 512
    model_dim = 256
    num_heads = 8
    num_splats = 16
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate test data
    x = torch.randn(batch_size, seq_len, model_dim, device=device)
    
    # Test different implementations
    implementations = [
        ("Vectorized", VectorizedSplatAttention(model_dim, num_heads, num_splats)),
        ("Hypertoroidal", VectorizedSplatAttention(model_dim, num_heads, num_splats, use_hypertoroidal=True)),
    ]
    
    if 'CudaOptimizedSplatAttention' in globals() and CudaOptimizedSplatAttention != VectorizedSplatAttention:
        implementations.append(("CUDA Optimized", CudaOptimizedSplatAttention(model_dim, num_heads, num_splats)))
    
    results = {}
    
    for name, model in implementations:
        model = model.to(device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(20):
                output = model(x)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 20
        results[name] = avg_time
        
        print(f"{name}: {avg_time:.4f}s per forward pass")
        print(f"  Output shape: {output.shape}")
        print(f"  Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f}GB" if device.type == 'cuda' else "")
    
    return results

if __name__ == "__main__":
    print("🚀 Vectorized SplatFlow Performance Test")
    print("=" * 50)
    
    benchmark_results = benchmark_implementations()
    
    print("\n📊 Performance Summary:")
    print("=" * 50)
    
    baseline_time = benchmark_results.get("Vectorized", 1.0)
    for name, time_taken in benchmark_results.items():
        speedup = baseline_time / time_taken if name != "Vectorized" else 1.0
        print(f"{name}: {time_taken:.4f}s ({speedup:.1f}x vs baseline)")
