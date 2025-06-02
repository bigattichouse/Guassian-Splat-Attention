"""
SplatFlow Minimal Proof-of-Concept
Start here to validate the core concept before building the full system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

class MinimalSplat(nn.Module):
    """Simplified splat for initial testing"""
    
    def __init__(self, embedding_dim: int = 64, model_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        
        # Learnable position in embedding space
        self.position = nn.Parameter(torch.randn(embedding_dim) * 0.5)
        self.scale = nn.Parameter(torch.ones(1))
        
        # Simple information transformation
        self.transform = nn.Linear(model_dim, model_dim)
        
    def forward(self, token_embeddings: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Process information through this splat"""
        # Compute distance from splat to each token
        distances = torch.norm(token_positions - self.position.unsqueeze(0), dim=-1)  # [seq_len]
        
        # Gaussian influence
        influence = torch.exp(-0.5 * (distances / self.scale) ** 2)  # [seq_len]
        
        # Gather weighted information
        weighted_info = token_embeddings * influence.unsqueeze(-1)  # [seq_len, model_dim]
        gathered = torch.sum(weighted_info, dim=0, keepdim=True)  # [1, model_dim]
        
        # Transform and redistribute
        transformed = self.transform(gathered)  # [1, model_dim]
        redistributed = transformed.expand_as(token_embeddings) * influence.unsqueeze(-1)
        
        return redistributed


class MinimalSplatFlow(nn.Module):
    """Minimal SplatFlow layer for testing core concepts"""
    
    def __init__(self, model_dim: int = 128, embedding_dim: int = 64, num_splats: int = 8):
        super().__init__()
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        
        # Token position encoder
        self.position_encoder = nn.Linear(model_dim, embedding_dim)
        
        # Splats
        self.splats = nn.ModuleList([
            MinimalSplat(embedding_dim, model_dim) for _ in range(num_splats)
        ])
        
        # Output processing
        self.output_weight = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [seq_len, model_dim] - simplified to single sequence
        Returns:
            output: [seq_len, model_dim]
        """
        # Map tokens to embedding space positions
        token_positions = torch.tanh(self.position_encoder(x))  # [seq_len, embedding_dim]
        
        # Process through each splat
        splat_outputs = []
        for splat in self.splats:
            splat_out = splat(x, token_positions)
            splat_outputs.append(splat_out)
        
        # Combine splat outputs
        combined_output = torch.stack(splat_outputs, dim=0).sum(dim=0)  # [seq_len, model_dim]
        
        # Mix with original (residual-like connection)
        output = x + self.output_weight * combined_output
        
        return output
    
    def visualize_splat_field(self, x: torch.Tensor, save_path: str = "splat_field.png"):
        """Visualize how splats are positioned relative to tokens"""
        with torch.no_grad():
            # Get token positions
            token_positions = torch.tanh(self.position_encoder(x))  # [seq_len, embedding_dim]
            
            # For 2D visualization, use first 2 dimensions
            if self.embedding_dim >= 2:
                token_pos_2d = token_positions[:, :2].cpu().numpy()
                splat_pos_2d = torch.stack([s.position[:2] for s in self.splats]).cpu().numpy()
                
                plt.figure(figsize=(10, 8))
                
                # Plot tokens
                plt.scatter(token_pos_2d[:, 0], token_pos_2d[:, 1], 
                           c='blue', s=100, alpha=0.7, label='Tokens')
                
                # Plot splats with influence circles
                for i, (splat_pos, splat) in enumerate(zip(splat_pos_2d, self.splats)):
                    plt.scatter(splat_pos[0], splat_pos[1], 
                               c='red', s=200, alpha=0.8, marker='*')
                    
                    # Draw influence circle
                    circle = plt.Circle(splat_pos, splat.scale.item(), 
                                      fill=False, color='red', alpha=0.3)
                    plt.gca().add_patch(circle)
                    
                    plt.text(splat_pos[0], splat_pos[1] + 0.1, f'S{i}', 
                            ha='center', va='bottom', fontsize=8)
                
                # Annotate tokens
                for i, pos in enumerate(token_pos_2d):
                    plt.text(pos[0], pos[1] + 0.05, f'T{i}', 
                            ha='center', va='bottom', fontsize=8)
                
                plt.legend()
                plt.title('SplatFlow: Splat Field Visualization')
                plt.xlabel('Embedding Dimension 1')
                plt.ylabel('Embedding Dimension 2')
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.show()


def test_splatflow_on_simple_task():
    """Test SplatFlow on a simple pattern learning task"""
    
    # Create simple patterns to learn
    # Pattern: tokens that are similar should attend to each other
    seq_len = 16
    model_dim = 128
    
    # Create test data - alternating pattern
    test_sequences = []
    for i in range(100):  # 100 training examples
        # Create sequence with pattern: [A, B, A, B, A, B, ...]
        sequence = torch.zeros(seq_len, model_dim)
        for j in range(seq_len):
            if j % 2 == 0:
                # Pattern A
                sequence[j, :model_dim//2] = 1.0 + torch.randn(model_dim//2) * 0.1
            else:
                # Pattern B  
                sequence[j, model_dim//2:] = 1.0 + torch.randn(model_dim//2) * 0.1
        
        test_sequences.append(sequence)
    
    # Create SplatFlow model
    model = MinimalSplatFlow(model_dim=model_dim, embedding_dim=32, num_splats=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training SplatFlow on pattern recognition task...")
    
    # Training loop
    for epoch in range(200):
        total_loss = 0
        
        for seq in test_sequences:
            # Forward pass
            output = model(seq)
            
            # Loss: similar tokens should produce similar outputs
            # Tokens at even positions should be similar to each other
            # Tokens at odd positions should be similar to each other
            loss = 0
            
            # Even positions should be similar
            even_positions = output[0::2]  # [seq_len//2, model_dim]
            if len(even_positions) > 1:
                even_target = even_positions.mean(dim=0, keepdim=True).expand_as(even_positions)
                loss += F.mse_loss(even_positions, even_target)
            
            # Odd positions should be similar
            odd_positions = output[1::2]  # [seq_len//2, model_dim]
            if len(odd_positions) > 1:
                odd_target = odd_positions.mean(dim=0, keepdim=True).expand_as(odd_positions)
                loss += F.mse_loss(odd_positions, odd_target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 50 == 0:
            avg_loss = total_loss / len(test_sequences)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.6f}")
    
    # Test the trained model
    print("\nTesting trained model...")
    model.eval()
    
    with torch.no_grad():
        test_seq = test_sequences[0]
        output = model(test_seq)
        
        print("Input pattern analysis:")
        print("Even positions (should be similar):")
        even_inputs = test_seq[0::2]
        print(f"  Similarity: {F.cosine_similarity(even_inputs[0], even_inputs[1], dim=0).item():.3f}")
        
        print("Odd positions (should be similar):")
        odd_inputs = test_seq[1::2]
        print(f"  Similarity: {F.cosine_similarity(odd_inputs[0], odd_inputs[1], dim=0).item():.3f}")
        
        print("\nOutput pattern analysis:")
        print("Even positions (should be similar):")
        even_outputs = output[0::2]
        print(f"  Similarity: {F.cosine_similarity(even_outputs[0], even_outputs[1], dim=0).item():.3f}")
        
        print("Odd positions (should be similar):")
        odd_outputs = output[1::2]
        print(f"  Similarity: {F.cosine_similarity(odd_outputs[0], odd_outputs[1], dim=0).item():.3f}")
        
        # Visualize the splat field
        model.visualize_splat_field(test_seq)
        
        # Analyze where splats positioned themselves
        print("\nSplat Analysis:")
        token_positions = torch.tanh(model.position_encoder(test_seq))
        for i, splat in enumerate(model.splats):
            pos = splat.position[:2]  # First 2 dims for analysis
            scale = splat.scale.item()
            print(f"  Splat {i}: position=({pos[0].item():.2f}, {pos[1].item():.2f}), scale={scale:.3f}")


def compare_with_standard_attention():
    """Compare SplatFlow with standard attention on the same task"""
    
    class StandardAttention(nn.Module):
        def __init__(self, model_dim: int):
            super().__init__()
            self.model_dim = model_dim
            self.attention = nn.MultiheadAttention(model_dim, num_heads=4, batch_first=False)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [seq_len, model_dim]
            attn_out, _ = self.attention(x, x, x)
            return x + attn_out
    
    print("\n" + "="*50)
    print("COMPARISON: SplatFlow vs Standard Attention")
    print("="*50)
    
    # Same test setup as before
    seq_len = 16
    model_dim = 128
    
    # Test both models
    splatflow_model = MinimalSplatFlow(model_dim=model_dim, num_splats=4)
    standard_model = StandardAttention(model_dim=model_dim)
    
    # Create test sequence
    test_seq = torch.zeros(seq_len, model_dim)
    for j in range(seq_len):
        if j % 2 == 0:
            test_seq[j, :model_dim//2] = 1.0
        else:
            test_seq[j, model_dim//2:] = 1.0
    
    print(f"Input sequence shape: {test_seq.shape}")
    print(f"Pattern: Alternating A/B pattern")
    
    # Test SplatFlow
    with torch.no_grad():
        splatflow_out = splatflow_model(test_seq)
        print(f"\nSplatFlow output shape: {splatflow_out.shape}")
        
        # Count parameters
        splatflow_params = sum(p.numel() for p in splatflow_model.parameters())
        standard_params = sum(p.numel() for p in standard_model.parameters())
        
        print(f"SplatFlow parameters: {splatflow_params:,}")
        print(f"Standard attention parameters: {standard_params:,}")
        print(f"Parameter ratio: {splatflow_params/standard_params:.2f}x")
    
    # Test Standard Attention
    with torch.no_grad():
        standard_out = standard_model(test_seq)
        print(f"Standard attention output shape: {standard_out.shape}")
    
    print("\nBoth models successfully processed the sequence!")
    print("SplatFlow provides a learnable, interpretable alternative to standard attention.")


if __name__ == "__main__":
    print("SplatFlow Minimal Proof-of-Concept")
    print("=" * 40)
    
    # Test core functionality
    test_splatflow_on_simple_task()
    
    # Compare with standard approach
    compare_with_standard_attention()
    
    print("\n✅ Proof-of-concept complete!")
    print("Next steps:")
    print("1. Test on real language modeling tasks")
    print("2. Add adaptive splat birth/death")
    print("3. Scale up to full transformer replacement")
    print("4. Optimize for long sequences")
