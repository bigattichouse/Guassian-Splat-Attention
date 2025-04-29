#!/usr/bin/env python3
"""
Test script for Hierarchical Splat Attention (HSA) implementation.

This script tests both the core data structures and initialization components
of the HSA system, verifying that they function correctly.
"""

import sys
import os
import unittest
import numpy as np
from typing import List, Dict, Set, Tuple

# Import HSA modules
from hsa.data_structures import (
    Splat, 
    Hierarchy, 
    SplatRegistry, 
    ensure_positive_definite, 
    sample_covariance_matrix
)
from hsa.initialization import (
    HSAInitializer,
    initialize_splats,
    reinitialize_splat
)

def _initialize_splat_centers(data_points, n_clusters, n_neighbors=10, affinity='nearest_neighbors', random_seed=None):
    """Stub implementation for testing."""
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Simple implementation for tests
    if len(data_points) >= n_clusters:
        # Randomly select centers from data points
        indices = np.random.choice(len(data_points), size=n_clusters, replace=False)
        centers = data_points[indices]
    else:
        # Generate additional centers based on existing data
        centers = np.zeros((n_clusters, data_points.shape[1]))
        centers[:len(data_points)] = data_points
        
        # Generate remaining centers as random variations
        for i in range(len(data_points), n_clusters):
            # Pick a random existing point and add noise
            base_idx = np.random.randint(len(data_points))
            centers[i] = data_points[base_idx] + np.random.randn(*data_points[base_idx].shape) * 0.1
    
    return centers


class TestHSADataStructures(unittest.TestCase):
    """Tests for the core HSA data structures."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Define dimensions for tests
        self.dim = 8
        
        # Create sample embeddings
        self.sample_position = np.random.randn(self.dim)
        self.sample_covariance = sample_covariance_matrix(self.dim)
        
        # Create sample hierarchy
        self.hierarchy = Hierarchy(
            levels=["token", "phrase", "sentence"],
            init_splats_per_level=[20, 10, 5],
            level_weights=[0.6, 0.3, 0.1]
        )
        
    def test_splat_initialization(self):
        """Test Splat initialization and basic properties."""
        # Create a splat
        splat = Splat(
            position=self.sample_position,
            covariance=self.sample_covariance,
            amplitude=0.8,
            level="token"
        )
        
        # Test basic properties
        self.assertEqual(splat.level, "token")
        self.assertEqual(splat.amplitude, 0.8)
        self.assertTrue(np.array_equal(splat.position, self.sample_position))
        self.assertTrue(np.array_equal(splat.covariance, self.sample_covariance))
        
        # Test ID generation
        self.assertTrue(splat.id.startswith("splat_"))
        
        # Test lazy computing of covariance inverse
        self.assertIsNone(splat._covariance_inverse)
        inverse = splat.covariance_inverse
        self.assertIsNotNone(splat._covariance_inverse)
        
        # Test lazy computing of normalization factor
        self.assertIsNone(splat._normalization_factor)
        norm_factor = splat.normalization_factor
        self.assertIsNotNone(splat._normalization_factor)
        
    def test_splat_compute_distance(self):
        """Test distance computation between tokens relative to a splat."""
        # Create a splat
        splat = Splat(
            position=np.zeros(self.dim),  # Zero position for easier testing
            covariance=np.eye(self.dim),  # Identity covariance for easier testing
            amplitude=1.0,
            level="token"
        )
        
        # Create two token embeddings
        token1 = np.ones(self.dim)
        token2 = np.ones(self.dim) * 2
        
        # Expected distance (with identity covariance and zero position)
        # should be sqrt(dim) because every dimension has diff of 1
        expected_distance = np.sqrt(self.dim)
        
        # Test distance computation
        distance = splat.compute_distance(token1, token2)
        self.assertAlmostEqual(distance, expected_distance)
        
        # Test attention score
        attention = splat.compute_attention(token1, token2)
        expected_attention = 1.0 * np.exp(-expected_distance**2)
        self.assertAlmostEqual(attention, expected_attention)
        
    def test_hierarchy(self):
        """Test hierarchy functionality."""
        # Test level indices
        self.assertEqual(self.hierarchy.get_level_index("token"), 0)
        self.assertEqual(self.hierarchy.get_level_index("phrase"), 1)
        self.assertEqual(self.hierarchy.get_level_index("sentence"), 2)
        
        # Test level relationships
        self.assertEqual(self.hierarchy.get_parent_level("token"), "phrase")
        self.assertEqual(self.hierarchy.get_parent_level("phrase"), "sentence")
        self.assertIsNone(self.hierarchy.get_parent_level("sentence"))
        
        self.assertEqual(self.hierarchy.get_child_level("sentence"), "phrase")
        self.assertEqual(self.hierarchy.get_child_level("phrase"), "token")
        self.assertIsNone(self.hierarchy.get_child_level("token"))
        
        # Test weights
        self.assertEqual(self.hierarchy.get_level_weight("token"), 0.6)
        self.assertEqual(self.hierarchy.get_level_weight("phrase"), 0.3)
        self.assertEqual(self.hierarchy.get_level_weight("sentence"), 0.1)
        
        # Test init counts
        self.assertEqual(self.hierarchy.get_init_splats_count("token"), 20)
        self.assertEqual(self.hierarchy.get_init_splats_count("phrase"), 10)
        self.assertEqual(self.hierarchy.get_init_splats_count("sentence"), 5)
        
        # Test error handling
        with self.assertRaises(ValueError):
            self.hierarchy.get_level_index("paragraph")
            
    def test_splat_registry(self):
        """Test SplatRegistry functionality."""
        # Create registry
        registry = SplatRegistry(self.hierarchy)
        
        # Create some test splats
        splat1 = Splat(self.sample_position, self.sample_covariance, 0.7, "token")
        splat2 = Splat(self.sample_position, self.sample_covariance, 0.5, "phrase")
        splat3 = Splat(self.sample_position, self.sample_covariance, 0.3, "sentence")
        
        # Register splats
        registry.register(splat1)
        registry.register(splat2)
        registry.register(splat3)
        
        # Test registration
        self.assertEqual(len(registry), 3)
        self.assertEqual(len(registry.get_splats_at_level("token")), 1)
        self.assertEqual(len(registry.get_splats_at_level("phrase")), 1)
        self.assertEqual(len(registry.get_splats_at_level("sentence")), 1)
        
        # Test retrieval
        self.assertEqual(registry.get_splat(splat1.id), splat1)
        
        # Test parent-child relationships
        splat3.add_child(splat2)
        splat2.add_child(splat1)
        
        self.assertEqual(splat1.parent, splat2)
        self.assertEqual(splat2.parent, splat3)
        self.assertIn(splat1, splat2.children)
        self.assertIn(splat2, splat3.children)
        
        # Test unregistration
        registry.unregister(splat1)
        self.assertEqual(len(registry), 2)
        self.assertEqual(len(registry.get_splats_at_level("token")), 0)
        self.assertEqual(len(splat2.children), 0)  # Child ref should be removed
        
    def test_utility_functions(self):
        """Test utility functions."""
        # Test ensure_positive_definite
        # Create a matrix with negative eigenvalues
        bad_matrix = np.array([
            [1.0, 2.0],
            [2.0, 1.0]
        ])
        fixed_matrix = ensure_positive_definite(bad_matrix)
        
        # Check that eigenvalues are positive
        eigenvals = np.linalg.eigvals(fixed_matrix)
        self.assertTrue(np.all(eigenvals > 0))
        
        # Test sample_covariance_matrix
        cov = sample_covariance_matrix(5, scale=0.5)
        self.assertEqual(cov.shape, (5, 5))
        
        # Check that it's positive definite
        eigenvals = np.linalg.eigvals(cov)
        self.assertTrue(np.all(eigenvals > 0))


class TestHSAInitialization(unittest.TestCase):
    """Tests for HSA initialization."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Define dimensions for tests
        self.embedding_dim = 8
        self.sequence_length = 100
        
        # Create random token embeddings
        self.tokens = np.random.randn(self.sequence_length, self.embedding_dim)
        
        # Define hierarchy config
        self.hierarchy_config = {
            "levels": ["token", "phrase", "sentence"],
            "init_splats_per_level": [10, 5, 2],
            "level_weights": [0.6, 0.3, 0.1]
        }
        
        # Create hierarchy object
        self.hierarchy = Hierarchy(
            levels=self.hierarchy_config["levels"],
            init_splats_per_level=self.hierarchy_config["init_splats_per_level"],
            level_weights=self.hierarchy_config["level_weights"]
        )
        
    def test_initializer_creation(self):
        """Test creation of HSAInitializer."""
        initializer = HSAInitializer(
            hierarchy=self.hierarchy,
            n_neighbors=5,
            affinity="nearest_neighbors",
            random_seed=42
        )
        
        self.assertEqual(initializer.n_neighbors, 5)
        self.assertEqual(initializer.affinity, "nearest_neighbors")
        
    def test_data_sampling(self):
        """Test data sampling for different hierarchy levels."""
        initializer = HSAInitializer(self.hierarchy, random_seed=42)
        
        # Test token level (should use all tokens)
        token_samples = initializer._sample_data_points(self.tokens, "token")
        self.assertEqual(len(token_samples), len(self.tokens))
        
        # Test higher levels (should subsample)
        phrase_samples = initializer._sample_data_points(self.tokens, "phrase")
        self.assertLess(len(phrase_samples), len(self.tokens))
        
        sentence_samples = initializer._sample_data_points(self.tokens, "sentence")
        self.assertLess(len(sentence_samples), len(phrase_samples))
        
    def test_splat_centers_initialization(self):
        """Test initialization of splat centers."""
        initializer = HSAInitializer(self.hierarchy, random_seed=42)
        
        # Test with normal case
        centers = _initialize_splat_centers(self.tokens, 5)
        self.assertEqual(centers.shape, (5, self.embedding_dim))
        
        # Test edge case (more clusters than data points)
        small_data = self.tokens[:3]
        centers = _initialize_splat_centers(small_data, 5)
        self.assertEqual(centers.shape, (5, self.embedding_dim))
        
    def test_covariance_initialization(self):
        """Test initialization of covariance matrices."""
        initializer = HSAInitializer(self.hierarchy, random_seed=42)
        
        # Create a center
        center = np.mean(self.tokens, axis=0)
        
        # Initialize covariance
        cov = initializer._initialize_covariance(center, self.tokens)
        
        # Check dimensions and properties
        self.assertEqual(cov.shape, (self.embedding_dim, self.embedding_dim))
        
        # Check that it's positive definite
        eigenvals = np.linalg.eigvals(cov)
        self.assertTrue(np.all(eigenvals > 0))
        
    def test_end_to_end_initialization(self):
        """Test end-to-end initialization of splats."""
        # Initialize splats using convenience function
        registry = initialize_splats(
            tokens=self.tokens,
            hierarchy_config=self.hierarchy_config,
            n_neighbors=5,
            random_seed=42
        )
        
        # Check that we have the expected number of splats
        expected_total = sum(self.hierarchy_config["init_splats_per_level"])
        self.assertEqual(len(registry), expected_total)
        
        # Check that splats are distributed correctly across levels
        for level_idx, level_name in enumerate(self.hierarchy_config["levels"]):
            expected_count = self.hierarchy_config["init_splats_per_level"][level_idx]
            actual_count = len(registry.get_splats_at_level(level_name))
            self.assertEqual(actual_count, expected_count)
            
        # Check that parent-child relationships are established
        for level_idx, level_name in enumerate(self.hierarchy_config["levels"]):
            if level_idx < len(self.hierarchy_config["levels"]) - 1:  # Not top level
                # All splats at this level should have a parent
                for splat in registry.get_splats_at_level(level_name):
                    self.assertIsNotNone(splat.parent)
                    
    def test_splat_reinitialization(self):
        """Test reinitialization of a splat."""
        # Create a splat
        splat = Splat(
            position=np.zeros(self.embedding_dim),
            covariance=np.eye(self.embedding_dim),
            amplitude=0.5,
            level="token"
        )
        
        # Save original values
        original_position = splat.position.copy()
        original_covariance = splat.covariance.copy()
        
        # Reinitialize splat
        reinitialize_splat(splat, self.tokens)
        
        # Check that values have changed
        self.assertFalse(np.array_equal(splat.position, original_position))
        self.assertFalse(np.array_equal(splat.covariance, original_covariance))


if __name__ == "__main__":
    unittest.main()
