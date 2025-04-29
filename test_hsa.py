
import numpy as np
import torch
from hsa.core import create_hsa

def test_basic_hsa():
    # Create an HSA instance with default configuration
    hsa = create_hsa()
    
    # Display initial state
    print(hsa)
    
    # Create some dummy token embeddings
    tokens = np.random.randn(16, 64)  # 16 tokens, 64-dim embeddings
    
    # Initialize splats
    hsa.initialize(tokens)
    
    # Display state after initialization
    print(hsa)
    
    # Compute attention matrix
    attention_matrix = hsa.compute_attention(tokens)
    print(f"Attention matrix shape: {attention_matrix.shape}")
    
    # Get statistics
    stats = hsa.get_stats()
    print(f"Splat counts: {stats['splat_counts']}")
    print(f"Level contributions: {stats['level_contributions']}")
    
    print("Test successful!")

if __name__ == "__main__":
    test_basic_hsa()
