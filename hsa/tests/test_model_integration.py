import sys
import os
import unittest
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Set, Optional, Union, Any
 
from hsa.data_structures import Splat, Hierarchy, SplatRegistry
from hsa.attention import AttentionComputer, create_attention_computer, HSAMultiheadAttention
from hsa.model_integration import (
    HSAAttention, 
    HSATransformerLayer, 
    replace_attention_with_hsa,
    HSAModelAdapter
)

class TestHSAAttention(unittest.TestCase):
    """Test cases for the HSAAttention class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Define a basic hierarchy configuration
        self.hierarchy_config = {
            "levels": ["Token", "Phrase", "Document"],
            "init_splats_per_level": [20, 10, 5],
            "level_weights": [0.5, 0.3, 0.2]
        }
        
        # Define dimensions for testing
        self.dim = 64
        self.num_heads = 4
        self.head_dim = 16
        self.seq_len = 16
        self.batch_size = 2
        
        # Create a test instance
        self.hsa_attention = HSAAttention(
            dim=self.dim,
            hierarchy_config=self.hierarchy_config,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=0.0,  # No dropout for testing
            sparse_topk=4,  # Small topk for testing
            init_splats=False  # Don't initialize splats yet
        )
        
        # Create test inputs
        self.hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.dim
        )
        self.attention_mask = torch.ones(
            self.batch_size, self.seq_len
        )
        
        # Manually initialize splats for testing
        self.initialize_test_splats()
    
    def initialize_test_splats(self):
        """Initialize test splats manually."""
        # Create a simple splat registry
        hierarchy = Hierarchy(
            levels=self.hierarchy_config["levels"],
            init_splats_per_level=self.hierarchy_config["init_splats_per_level"],
            level_weights=self.hierarchy_config["level_weights"]
        )
        registry = SplatRegistry(hierarchy)
        
        # Create some basic splats
        for level_idx, level_name in enumerate(hierarchy.levels):
            n_splats = hierarchy.get_init_splats_count(level_name)
            for i in range(n_splats):
                # Create a simple splat with random position
                position = np.random.randn(self.head_dim)
                covariance = np.eye(self.head_dim) * 0.1  # Small identity covariance
                
                splat = Splat(
                    position=position,
                    covariance=covariance,
                    amplitude=1.0,
                    level=level_name
                )
                
                registry.register(splat)
        
        self.hsa_attention.splat_registry = registry
        self.hsa_attention.is_initialized = True
    
    def test_initialization(self):
        """Test if the attention layer initializes correctly."""
        # Check dimensions
        self.assertEqual(self.hsa_attention.dim, self.dim)
        self.assertEqual(self.hsa_attention.num_heads, self.num_heads)
        self.assertEqual(self.hsa_attention.head_dim, self.head_dim)
        
        # Check projections
        self.assertEqual(self.hsa_attention.q_proj.in_features, self.dim)
        self.assertEqual(self.hsa_attention.q_proj.out_features, self.num_heads * self.head_dim)
        
        # Check hierarchy
        self.assertEqual(self.hsa_attention.hierarchy.levels, self.hierarchy_config["levels"])
    
    def test_reshape_for_heads(self):
        """Test the reshaping for multi-head attention."""
        # Create a test input
        x = torch.randn(self.batch_size, self.seq_len, self.num_heads * self.head_dim)
        
        # Apply reshaping
        reshaped = self.hsa_attention._reshape_for_heads(x)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self.assertEqual(reshaped.size(), expected_shape)
    
    def test_compute_hsa_attention(self):
        """Test the HSA attention computation."""
        # Create test inputs (already reshaped for multi-head attention)
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        v = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        
        # Compute attention
        output = self.hsa_attention._compute_hsa_attention(q, k, v)
        
        # Check output shape
        expected_shape = (self.batch_size, self.num_heads, self.seq_len, self.head_dim)
        self.assertEqual(output.size(), expected_shape)
    
    def test_forward(self):
        """Test the forward pass."""
        # Run a forward pass
        output = self.hsa_attention(self.hidden_states, self.attention_mask)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.dim)
        self.assertEqual(output.size(), expected_shape)
    
    def test_automatic_initialization(self):
        """Test automatic splat initialization."""
        # Create a new instance with initialization enabled
        hsa_with_init = HSAAttention(
            dim=self.dim,
            hierarchy_config=self.hierarchy_config,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=0.0,
            sparse_topk=4,
            init_splats=False,  # Will initialize on first forward pass
        )
        
        # Run a forward pass to trigger initialization
        output = hsa_with_init(self.hidden_states)
        
        # Check that initialization happened
        self.assertTrue(hsa_with_init.is_initialized)
        self.assertIsNotNone(hsa_with_init.splat_registry)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.dim)
        self.assertEqual(output.size(), expected_shape)
    
    def test_update_splats(self):
        """Test updating the splat registry."""
        # Create a new splat registry
        hierarchy = Hierarchy(
            levels=self.hierarchy_config["levels"],
            init_splats_per_level=self.hierarchy_config["init_splats_per_level"],
            level_weights=self.hierarchy_config["level_weights"]
        )
        new_registry = SplatRegistry(hierarchy)
        
        # Add a single splat
        position = np.random.randn(self.head_dim)
        covariance = np.eye(self.head_dim) * 0.1
        splat = Splat(
            position=position,
            covariance=covariance,
            amplitude=1.0,
            level=hierarchy.levels[0]
        )
        new_registry.register(splat)
        
        # Update the splat registry
        self.hsa_attention.update_splats(new_registry)
        
        # Check that the update worked
        self.assertEqual(len(self.hsa_attention.splat_registry.splats), 1)
        self.assertTrue(self.hsa_attention.is_initialized)


class TestHSATransformerLayer(unittest.TestCase):
    """Test cases for the HSATransformerLayer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Define a basic hierarchy configuration
        self.hierarchy_config = {
            "levels": ["Token", "Phrase", "Document"],
            "init_splats_per_level": [20, 10, 5],
            "level_weights": [0.5, 0.3, 0.2]
        }
        
        # Define dimensions for testing
        self.dim = 64
        self.num_heads = 4
        self.ffn_dim = 128
        self.seq_len = 16
        self.batch_size = 2
        
        # Create a test instance
        self.hsa_layer = HSATransformerLayer(
            dim=self.dim,
            hierarchy_config=self.hierarchy_config,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            dropout=0.0,  # No dropout for testing
            sparse_topk=4  # Small topk for testing
        )
        
        # Create test inputs
        self.hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.dim
        )
        self.attention_mask = torch.ones(
            self.batch_size, self.seq_len
        )
        
        # Manually initialize splats for testing
        self.initialize_test_splats()
    
    def initialize_test_splats(self):
        """Initialize test splats for the HSA attention in the layer."""
        # Create a simple splat registry
        hierarchy = Hierarchy(
            levels=self.hierarchy_config["levels"],
            init_splats_per_level=self.hierarchy_config["init_splats_per_level"],
            level_weights=self.hierarchy_config["level_weights"]
        )
        registry = SplatRegistry(hierarchy)
        
        # Create some basic splats
        for level_idx, level_name in enumerate(hierarchy.levels):
            n_splats = hierarchy.get_init_splats_count(level_name)
            for i in range(n_splats):
                # Create a simple splat with random position
                position = np.random.randn(self.dim // self.num_heads)
                covariance = np.eye(self.dim // self.num_heads) * 0.1
                
                splat = Splat(
                    position=position,
                    covariance=covariance,
                    amplitude=1.0,
                    level=level_name
                )
                
                registry.register(splat)
        
        self.hsa_layer.attention.splat_registry = registry
        self.hsa_layer.attention.is_initialized = True
    
    def test_initialization(self):
        """Test if the transformer layer initializes correctly."""
        # Check dimensions of layer norm
        self.assertEqual(self.hsa_layer.layer_norm1.normalized_shape[0], self.dim)
        
        # Check FFN dimensions
        self.assertEqual(self.hsa_layer.ffn[0].in_features, self.dim)
        self.assertEqual(self.hsa_layer.ffn[0].out_features, self.ffn_dim)
        self.assertEqual(self.hsa_layer.ffn[3].in_features, self.ffn_dim)
        self.assertEqual(self.hsa_layer.ffn[3].out_features, self.dim)
    
    def test_forward(self):
        """Test the forward pass of the transformer layer."""
        # Run a forward pass
        output = self.hsa_layer(self.hidden_states, self.attention_mask)
        
        # Check output shape
        expected_shape = (self.batch_size, self.seq_len, self.dim)
        self.assertEqual(output.size(), expected_shape)
        
        # Check that output is different from input (transformation happened)
        self.assertFalse(torch.allclose(output, self.hidden_states))


class MockTransformerWithAttention(nn.Module):
    """Mock transformer model for testing attention replacement."""
    
    def __init__(self, dim=64, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Create a simple self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(dim)
        
        # Another module with "attention" in the name but not an attention module
        self.attention_output = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        # Apply self-attention
        attn_output, _ = self.self_attention(x, x, x, attn_mask=mask)
        x = x + attn_output
        x = self.layer_norm(x)
        x = self.attention_output(x)
        return x


class TestAttentionReplacement(unittest.TestCase):
    """Test cases for replacing standard attention with HSA attention."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Define a basic hierarchy configuration
        self.hierarchy_config = {
            "levels": ["Token", "Phrase", "Document"],
            "init_splats_per_level": [20, 10, 5],
            "level_weights": [0.5, 0.3, 0.2]
        }
        
        # HSA configuration
        self.hsa_config = {
            "hierarchy": self.hierarchy_config,
            "attention": {
                "sparse_topk": 4
            }
        }
        
        # Create a mock transformer model
        self.mock_model = MockTransformerWithAttention(dim=64, num_heads=4)
        
        # Dimensions for testing
        self.seq_len = 16
        self.batch_size = 2
        self.dim = 64
        
        # Test inputs
        self.test_input = torch.randn(self.batch_size, self.seq_len, self.dim)
    
    def test_replace_attention(self):
        """Test replacing standard attention with HSA attention."""
        # Get the original model's output before replacement
        original_output = self.mock_model(self.test_input)
        
        # Replace attention with HSA
        modified_model = replace_attention_with_hsa(
            model=self.mock_model,
            hsa_config=self.hsa_config,
            attention_layer_pattern="self_attention",
            replace_in_place=True
        )
        
        # Check that the attention module was replaced
        self.assertIsInstance(modified_model.self_attention, HSAMultiheadAttention)
        
        # Check that other modules with "attention" in the name were not replaced
        self.assertNotIsInstance(modified_model.attention_output, HSAAttention)
        
        # Test forward pass with the modified model
        modified_output = modified_model(self.test_input)
        
        # Check that the output shape is maintained
        self.assertEqual(modified_output.shape, original_output.shape)
    
    def test_model_adapter(self):
        """Test the HSAModelAdapter class."""
        # Create an adapter for a specific model type
        adapter = HSAModelAdapter(model_type="bert")
        
        # Adapt the mock model
        adapted_model = adapter.adapt(self.mock_model, self.hsa_config)
        
        # Check that the adaptation worked
        self.assertIsInstance(adapted_model.self_attention, HSAMultiheadAttention)
        
        # Test creating a transformer layer
        transformer_layer = adapter.create_transformer_layer(
            dim=self.dim,
            hsa_config=self.hsa_config,
            num_heads=4
        )
        
        # Check that the layer was created correctly
        self.assertIsInstance(transformer_layer, HSATransformerLayer)
        self.assertEqual(transformer_layer.attention.dim, self.dim)


if __name__ == "__main__":
    unittest.main()
