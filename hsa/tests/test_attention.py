import sys
import os
import unittest
import numpy as np 

from hsa.data_structures import (
    Splat, 
    Hierarchy, 
    SplatRegistry, 
    ensure_positive_definite, 
    sample_covariance_matrix
)
from hsa.attention import AttentionComputer, SplatAttentionMetrics, create_attention_computer
from hsa.attention.implementations import DenseAttentionComputer, SparseAttentionComputer, SpatialAttentionComputer

class TestAttentionComputer(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        # Define a simple hierarchy for testing
        self.hierarchy = Hierarchy(
            levels=["fine", "medium", "coarse"],
            init_splats_per_level=[10, 5, 2],
            level_weights=[0.5, 0.3, 0.2]
        )
        
        # Create a test splat registry
        self.registry = SplatRegistry(self.hierarchy)
        
        # Create some splats for testing
        self.embed_dim = 4
        
        # Create one splat for each level
        self.fine_splat = Splat(
            position=np.array([0.1, 0.2, 0.3, 0.4]),
            covariance=np.eye(self.embed_dim) * 0.5,
            amplitude=1.0,
            level="fine",
            splat_id="test_fine_splat"
        )
        
        self.medium_splat = Splat(
            position=np.array([0.5, 0.4, 0.3, 0.2]),
            covariance=np.eye(self.embed_dim) * 1.0,
            amplitude=0.8,
            level="medium",
            splat_id="test_medium_splat"
        )
        
        self.coarse_splat = Splat(
            position=np.array([0.9, 0.8, 0.7, 0.6]),
            covariance=np.eye(self.embed_dim) * 2.0,
            amplitude=0.6,
            level="coarse",
            splat_id="test_coarse_splat"
        )
        
        # Register the splats
        self.registry.register(self.fine_splat)
        self.registry.register(self.medium_splat)
        self.registry.register(self.coarse_splat)
        
        # Create parent-child relationships
        self.coarse_splat.add_child(self.medium_splat)
        self.medium_splat.add_child(self.fine_splat)
        
        # Create token embeddings for testing
        self.tokens = np.array([
            [0.1, 0.1, 0.1, 0.1],  # token 0
            [0.2, 0.2, 0.2, 0.2],  # token 1
            [0.3, 0.3, 0.3, 0.3],  # token 2
            [0.4, 0.4, 0.4, 0.4],  # token 3
            [0.5, 0.5, 0.5, 0.5],  # token 4
        ])
        
        # Create attention computers
        self.dense_computer = DenseAttentionComputer(self.hierarchy, sparse_topk=2)
        self.sparse_computer = SparseAttentionComputer(self.hierarchy, sparse_topk=2)
        
    def test_compute_attention(self):
        """Test computation of the full attention matrix."""
        attention_matrix = self.dense_computer.compute_attention(
            self.tokens, self.registry
        )
        
        # Should return a matrix of the correct shape
        self.assertEqual(attention_matrix.shape, (5, 5))
        
        # All values should be non-negative
        self.assertTrue(np.all(attention_matrix >= 0))
        
    def test_sparse_vs_dense_compute_attention(self):
        """Test that sparse and dense implementations produce similar results."""
        # Compute attention with both implementations
        attention_dense = self.dense_computer.compute_attention(
            self.tokens, self.registry
        )
        
        attention_sparse = self.sparse_computer.compute_attention(
            self.tokens, self.registry
        )
        
        # Check for similar properties
        
        # 1. Check shapes are the same
        self.assertEqual(attention_sparse.shape, attention_dense.shape)
        
        # 2. Both should have non-negative values
        self.assertTrue(np.all(attention_sparse >= 0))
        self.assertTrue(np.all(attention_dense >= 0))
        
        # 3. Check sparsity patterns: both should have zeros in similar positions
        # but we don't expect exact matches due to implementation differences
        zero_sparse = (attention_sparse == 0)
        zero_dense = (attention_dense == 0)
        
        # Calculate overlap in zero positions - if both matrices have zeros,
        # we expect some overlap but not perfect alignment
        if np.any(zero_sparse) and np.any(zero_dense):
            overlap = np.logical_and(zero_sparse, zero_dense)
            # At least 30% of zero positions should match
            # (this is a heuristic; adjust based on your implementations)
            zero_match_ratio = np.sum(overlap) / max(1, np.sum(zero_sparse))
            self.assertGreaterEqual(zero_match_ratio, 0.3, 
                                   "Zero positions in matrices don't have sufficient overlap")
        
        # 4. Check that both implementations produce values in similar ranges
        if np.count_nonzero(attention_sparse) > 0 and np.count_nonzero(attention_dense) > 0:
            max_sparse = np.max(attention_sparse)
            max_dense = np.max(attention_dense)
            
            # Check if the max values are within an order of magnitude
            ratio = max_sparse / max_dense if max_dense > 0 else float('inf')
            self.assertTrue(0.1 <= ratio <= 10, 
                           f"Max values too different: {max_sparse} vs {max_dense}")
            
            # Check overall magnitude of non-zero values
            nonzero_sparse = attention_sparse[attention_sparse > 0]
            nonzero_dense = attention_dense[attention_dense > 0]
            
            mean_sparse = np.mean(nonzero_sparse)
            mean_dense = np.mean(nonzero_dense)
            
            # Check if means are within an order of magnitude
            ratio = mean_sparse / mean_dense if mean_dense > 0 else float('inf')
            self.assertTrue(0.1 <= ratio <= 10, f"Mean values too different: {mean_sparse} vs {mean_dense}")
                       
    def test_compute_splat_attention_map(self):
        """Test computation of attention map for a single splat."""
        attention_map = self.dense_computer.compute_splat_attention_map(
            self.tokens, self.fine_splat
        )
        
        # Should return a matrix of the correct shape
        self.assertEqual(attention_map.shape, (5, 5))
        
        # All values should be non-negative
        self.assertTrue(np.all(attention_map >= 0))
        
        # Verify a specific value
        token_i = self.tokens[0]
        token_j = self.tokens[1]
        diff = (token_i - token_j) - self.fine_splat.position
        distance = np.sqrt(diff @ self.fine_splat.covariance_inverse @ diff)
        expected_value = self.fine_splat.amplitude * np.exp(-distance**2)
        
        self.assertAlmostEqual(attention_map[0, 1], expected_value)


class TestSpatialAttentionComputer(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        # Define a simple hierarchy
        self.hierarchy = Hierarchy(
            levels=["fine", "coarse"],
            init_splats_per_level=[5, 2],
            level_weights=[0.7, 0.3]
        )
        
        # Create attention computers
        self.sparse_computer = SparseAttentionComputer(
            self.hierarchy, sparse_topk=3
        )
        
        self.spatial_computer = SpatialAttentionComputer(
            self.hierarchy, sparse_topk=3
        )
        
        # Create tokens
        self.embed_dim = 4
        self.tokens = np.random.randn(8, self.embed_dim)  # 8 tokens
        
        # Create registry
        self.registry = SplatRegistry(self.hierarchy)
        
        # Create and register splats
        self.fine_splat1 = Splat(
            position=np.random.randn(self.embed_dim),
            covariance=np.eye(self.embed_dim) * 0.5,
            amplitude=1.0,
            level="fine"
        )
        
        self.fine_splat2 = Splat(
            position=np.random.randn(self.embed_dim),
            covariance=np.eye(self.embed_dim) * 0.5,
            amplitude=0.8,
            level="fine"
        )
        
        self.coarse_splat = Splat(
            position=np.random.randn(self.embed_dim),
            covariance=np.eye(self.embed_dim) * 1.5,
            amplitude=0.6,
            level="coarse"
        )
        
        self.registry.register(self.fine_splat1)
        self.registry.register(self.fine_splat2)
        self.registry.register(self.coarse_splat)
        
    def test_spatial_vs_standard_compute_attention(self):
        """Test the spatial attention computer implementation against standard."""
        # Compute attention with both implementations
        attention_standard = self.sparse_computer.compute_attention(
            self.tokens, self.registry
        )
        
        attention_spatial = self.spatial_computer.compute_attention(
            self.tokens, self.registry
        )
        
        # Check properties:
        
        # 1. Both matrices should have the same shape
        self.assertEqual(attention_standard.shape, attention_spatial.shape)
        
        # 2. Both should have only non-negative values
        self.assertTrue(np.all(attention_standard >= 0))
        self.assertTrue(np.all(attention_spatial >= 0))
        
        # 3. Check sparsity patterns - this is less strict than exact comparison
        # For small sequences, we'll check if both matrices have similar sparsity
        if np.count_nonzero(attention_standard == 0) > 0 and np.count_nonzero(attention_spatial == 0) > 0:
            sparsity_standard = np.count_nonzero(attention_standard == 0) / attention_standard.size
            sparsity_spatial = np.count_nonzero(attention_spatial == 0) / attention_spatial.size
            
            # Sparsity levels should be roughly similar (within 20%)
            ratio = abs(sparsity_standard - sparsity_spatial) / max(sparsity_standard, sparsity_spatial)
            self.assertLessEqual(ratio, 0.2, 
                                "Sparsity levels between implementations too different")
            
    def test_build_token_index(self):
        """Test building the spatial index for tokens."""
        # Reset token index
        self.spatial_computer.token_index = None
        
        # Build index
        self.spatial_computer._build_token_index(self.tokens)
        
        # Check that index was created
        self.assertIsNotNone(self.spatial_computer.token_index)
        
    def test_build_splat_index(self):
        """Test building the spatial index for splats."""
        # Get the fine splats
        fine_splats = list(self.registry.get_splats_at_level("fine"))
        
        # Build index
        self.spatial_computer._build_splat_index(fine_splats, "fine")
        
        # Check that index was created
        self.assertIn("fine", self.spatial_computer.splat_indices)
        self.assertIsNotNone(self.spatial_computer.splat_indices["fine"])


class TestSplatAttentionMetrics(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        # Create a metrics tracker
        self.metrics = SplatAttentionMetrics()
        
        # Create some splats for testing
        self.embed_dim = 4
        self.splat1 = Splat(
            position=np.array([0.1, 0.2, 0.3, 0.4]),
            covariance=np.eye(self.embed_dim) * 0.5,
            amplitude=1.0,
            level="test",
            splat_id="splat1"
        )
        
        self.splat2 = Splat(
            position=np.array([0.5, 0.6, 0.7, 0.8]),
            covariance=np.eye(self.embed_dim) * 1.0,
            amplitude=0.8,
            level="test",
            splat_id="splat2"
        )
        
        # Create token embeddings
        self.tokens = np.array([
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3],
        ])
        
        # Create attention matrices for testing
        self.current_attention = np.array([
            [0.5, 0.3, 0.2],
            [0.3, 0.5, 0.2],
            [0.2, 0.2, 0.6]
        ])
        
        self.target_attention = np.array([
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6]
        ])
        
    def test_compute_splat_activation(self):
        """Test computation of splat activation level."""
        activation = self.metrics.compute_splat_activation(
            self.splat1, self.tokens, self.current_attention
        )
        
        # Activation should be positive
        self.assertGreater(activation, 0)
        
        # Should be stored in the metrics
        self.assertIn("splat1", self.metrics.splat_activations)
        self.assertEqual(self.metrics.splat_activations["splat1"], activation)
        
    def test_compute_error_contribution(self):
        """Test computation of splat error contribution."""
        error_contrib = self.metrics.compute_splat_error_contribution(
            self.splat1, self.tokens, self.target_attention, self.current_attention
        )
        
        # Error contribution should be non-negative (since we take absolute value)
        self.assertGreaterEqual(error_contrib, 0)
        
        # Should be stored in the metrics
        self.assertIn("splat1", self.metrics.splat_error_contributions)
        self.assertEqual(self.metrics.splat_error_contributions["splat1"], error_contrib)
        
    def test_get_splat_metrics(self):
        """Test retrieval of metrics for a specific splat."""
        # Compute some metrics first
        activation = self.metrics.compute_splat_activation(
            self.splat1, self.tokens, self.current_attention
        )
        
        error_contrib = self.metrics.compute_splat_error_contribution(
            self.splat1, self.tokens, self.target_attention, self.current_attention
        )
        
        # Get the combined metrics
        metrics = self.metrics.get_splat_metrics("splat1")
        
        # Should have both metrics
        self.assertIn("activation", metrics)
        self.assertIn("error_contribution", metrics)
        
        # Values should match what was computed
        self.assertEqual(metrics["activation"], activation)
        self.assertEqual(metrics["error_contribution"], error_contrib)
        
    def test_get_top_active_splats(self):
        """Test retrieval of top active splats."""
        # Compute activations for both splats
        self.metrics.compute_splat_activation(
            self.splat1, self.tokens, self.current_attention
        )
        
        self.metrics.compute_splat_activation(
            self.splat2, self.tokens, self.current_attention
        )
        
        # Get top active splat
        top_splats = self.metrics.get_top_active_splats(n=1)
        
        # Should return a list with one tuple
        self.assertEqual(len(top_splats), 1)
        self.assertEqual(len(top_splats[0]), 2)  # (splat_id, activation)
        
        # Get both splats
        both_splats = self.metrics.get_top_active_splats(n=2)
        self.assertEqual(len(both_splats), 2)
        
    def test_get_top_error_contributors(self):
        """Test retrieval of top error contributors."""
        # Compute error contributions for both splats
        self.metrics.compute_splat_error_contribution(
            self.splat1, self.tokens, self.target_attention, self.current_attention
        )
        
        self.metrics.compute_splat_error_contribution(
            self.splat2, self.tokens, self.target_attention, self.current_attention
        )
        
        # Get top error contributor
        top_contributors = self.metrics.get_top_error_contributors(n=1)
        
        # Should return a list with one tuple
        self.assertEqual(len(top_contributors), 1)
        self.assertEqual(len(top_contributors[0]), 2)  # (splat_id, error_contribution)
        
    def test_reset_metrics(self):
        """Test resetting all metrics."""
        # Compute some metrics
        self.metrics.compute_splat_activation(
            self.splat1, self.tokens, self.current_attention
        )
        
        self.metrics.compute_splat_error_contribution(
            self.splat1, self.tokens, self.target_attention, self.current_attention
        )
        
        # Reset all metrics
        self.metrics.reset()
        
        # All dictionaries should be empty
        self.assertEqual(len(self.metrics.splat_activations), 0)
        self.assertEqual(len(self.metrics.splat_error_contributions), 0)


class TestAttentionFactoryFunction(unittest.TestCase):
    def test_create_attention_computer(self):
        """Test the factory function for creating attention computers."""
        hierarchy = Hierarchy(
            levels=["fine", "coarse"],
            init_splats_per_level=[5, 2],
            level_weights=[0.7, 0.3]
        )
        
        # Create standard attention computer (inefficient/dense)
        attention_standard = create_attention_computer(
            hierarchy, sparse_topk=4, efficient=False
        )
        self.assertIsInstance(attention_standard, AttentionComputer)
        self.assertIsInstance(attention_standard, DenseAttentionComputer)
        self.assertNotIsInstance(attention_standard, SpatialAttentionComputer)
        
        # Create efficient attention computer using the spatial indexing
        attention_spatial = create_attention_computer(
            hierarchy, sparse_topk=4, efficient=True, use_spatial=True
        )
        self.assertIsInstance(attention_spatial, AttentionComputer)
        self.assertIsInstance(attention_spatial, SpatialAttentionComputer)
        
        # Create regular efficient computer (sparse but not spatial)
        attention_efficient = create_attention_computer(
            hierarchy, sparse_topk=4, efficient=True, use_spatial=False
        )
        self.assertIsInstance(attention_efficient, AttentionComputer)
        self.assertIsInstance(attention_efficient, SparseAttentionComputer)
        self.assertNotIsInstance(attention_efficient, SpatialAttentionComputer)


if __name__ == "__main__":
    unittest.main()
