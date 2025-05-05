import unittest
import numpy as np
import scipy.sparse as sp
from hsa.sparse_attention_utils import (
    compute_token_relevance, initialize_spatial_indexes, update_spatial_indexes,
    create_sparse_matrix_from_values, apply_sparse_topk, get_sparsity_ratio,
    find_relevant_splats_for_token, create_causal_mask, normalize_attention_rows,
    compute_pairwise_splat_distances
)
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy
from hsa.splat import Splat


class TestSparseAttentionUtils(unittest.TestCase):
    """Tests for sparse attention utility functions."""

    def setUp(self):
        """Set up test data for utility function tests."""
        # Create a simple hierarchy and registry
        self.hierarchy = Hierarchy(levels=["token", "sentence", "document"])
        self.registry = SplatRegistry(hierarchy=self.hierarchy, embedding_dim=2)
        
        # Create some test splats
        self.splats = []
        for i in range(5):
            # Create splats at different positions
            x = i - 2  # Positions: -2, -1, 0, 1, 2
            y = i % 3 - 1  # Positions: -1, 0, 1, -1, 0
            position = np.array([x, y], dtype=float)
            
            # Create different covariance matrices
            if i % 2 == 0:
                # Diagonal covariance
                covariance = np.diag([0.5, 0.2]) * (i + 1)
            else:
                # Non-diagonal covariance
                covariance = np.array([[0.5, 0.1], [0.1, 0.2]]) * (i + 1)
            
            splat = Splat(
                dim=2,
                position=position,
                covariance=covariance,
                amplitude=1.0,
                level="token" if i < 3 else "sentence",
                id=f"splat_{i}"
            )
            self.splats.append(splat)
            self.registry.register(splat)
        
        # Create test tokens
        self.tokens = np.array([
            [-1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.5, 0.5]
        ])

    def test_compute_token_relevance(self):
        """Test computing token relevance to a splat."""
        # Test with a splat
        splat = self.splats[0]
        
        # Compute relevance for all tokens
        relevance = compute_token_relevance(self.tokens, splat)
        
        # Check result shape
        self.assertEqual(relevance.shape, (len(self.tokens),))
        
        # All relevance values should be between 0 and 1
        self.assertTrue(np.all(relevance >= 0))
        self.assertTrue(np.all(relevance <= 1))
        
        # Tokens closer to the splat position should have higher relevance
        distances = np.linalg.norm(self.tokens - splat.position, axis=1)
        for i in range(1, len(distances)):
            if distances[i-1] < distances[i]:
                self.assertGreater(relevance[i-1], relevance[i])

    def test_initialize_spatial_indexes(self):
        """Test initializing spatial indexes for a registry."""
        # Initialize spatial indexes
        indexes = initialize_spatial_indexes(self.registry)
        
        # Should create one index per level that has splats
        levels_with_splats = set(splat.level for splat in self.splats)
        self.assertEqual(set(indexes.keys()), levels_with_splats)
        
        # Each index should have the right splats
        for level, index in indexes.items():
            # Get splats at this level
            level_splats = list(self.registry.get_splats_at_level(level))
            
            # Index should contain all these splats
            all_splats_in_index = index.get_all_splats()
            for splat in level_splats:
                self.assertIn(splat, all_splats_in_index)

    def test_update_spatial_indexes(self):
        """Test updating spatial indexes for a registry."""
        # Initialize spatial indexes
        indexes = initialize_spatial_indexes(self.registry)
        
        # Remember initial counts
        initial_counts = {level: len(index.get_all_splats()) for level, index in indexes.items()}
        
        # Add a new splat
        new_splat = Splat(
            dim=2,
            position=np.array([3.0, 3.0]),
            covariance=np.eye(2) * 0.1,
            amplitude=1.0,
            level="token",
            id="new_splat"
        )
        self.registry.register(new_splat)
        
        # Update indexes
        update_spatial_indexes(self.registry, indexes)
        
        # Verify the token level index has been updated
        self.assertEqual(len(indexes["token"].get_all_splats()), initial_counts["token"] + 1)
        
        # The new splat should be in the index
        self.assertIn(new_splat, indexes["token"].get_all_splats())
        
        # Remove a splat
        self.registry.unregister(self.splats[0])
        
        # Update indexes
        update_spatial_indexes(self.registry, indexes)
        
        # Verify the token level index has been updated
        self.assertEqual(len(indexes["token"].get_all_splats()), initial_counts["token"])
        
        # The removed splat should not be in the index
        self.assertNotIn(self.splats[0], indexes["token"].get_all_splats())

    def test_create_sparse_matrix_from_values(self):
        """Test creating a sparse matrix from values."""
        # Create data for sparse matrix
        data = [1.0, 2.0, 3.0]
        row_ind = [0, 1, 2]
        col_ind = [1, 0, 2]
        shape = (3, 3)
        
        # Create sparse matrix
        matrix = create_sparse_matrix_from_values(data, row_ind, col_ind, shape)
        
        # Check shape
        self.assertEqual(matrix.shape, shape)
        
        # Check values
        expected = np.zeros(shape)
        for d, r, c in zip(data, row_ind, col_ind):
            expected[r, c] = d
        
        np.testing.assert_array_equal(matrix, expected)

    def test_apply_sparse_topk(self):
        """Test applying top-k sparsity to a matrix."""
        # Create test matrix
        matrix = np.array([
            [0.1, 0.5, 0.2, 0.8],
            [0.7, 0.3, 0.9, 0.1],
            [0.2, 0.6, 0.4, 0.3]
        ])
        
        # Apply top-2 sparsity
        k = 2
        result = apply_sparse_topk(matrix, k)
        
        # Check shape
        self.assertEqual(result.shape, matrix.shape)
        
        # Each row should have exactly k non-zero values
        for i in range(result.shape[0]):
            non_zero = np.count_nonzero(result[i])
            self.assertEqual(non_zero, k)
            
            # The non-zero values should be the largest k values in the original row
            top_k_indices = np.argsort(matrix[i])[-k:]
            for j in range(matrix.shape[1]):
                if j in top_k_indices:
                    self.assertEqual(result[i, j], matrix[i, j])
                else:
                    self.assertEqual(result[i, j], 0.0)

    def test_get_sparsity_ratio(self):
        """Test calculating sparsity ratio of a matrix."""
        # Create test matrices with different sparsity
        # 1. Dense matrix (no zeros)
        dense = np.ones((3, 3))
        # 2. Half-sparse matrix (half zeros)
        half_sparse = np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0]
        ])
        # 3. Very sparse matrix (mostly zeros)
        very_sparse = np.zeros((3, 3))
        very_sparse[0, 0] = 1.0

        # Calculate sparsity ratios
        dense_ratio = get_sparsity_ratio(dense)
        half_ratio = get_sparsity_ratio(half_sparse)
        sparse_ratio = get_sparsity_ratio(very_sparse)

        # Check ratios - using the actual mathematical values
        self.assertEqual(dense_ratio, 0.0)  # No zeros
        # Calculate actual zeros in half_sparse (5 zeros out of 9 elements)
        expected_half_ratio = 5/9
        self.assertAlmostEqual(half_ratio, expected_half_ratio, places=6)  # 5/9 zeros
        self.assertAlmostEqual(sparse_ratio, 8/9, places=6)  # 8/9 zeros

    def test_find_relevant_splats_for_token(self):
        """Test finding relevant splats for a token."""
        # Initialize a spatial index for tokens
        indexes = initialize_spatial_indexes(self.registry)
        token_index = indexes["token"]
        
        # Test with a token position
        token = np.array([0.0, 0.0])
        
        # Find relevant splats
        relevant_splats = find_relevant_splats_for_token(
            token, token_index, relevance_threshold=0.1, max_splats=5
        )
        
        # Check result structure
        for splat, relevance in relevant_splats:
            self.assertIsInstance(splat, Splat)
            self.assertGreaterEqual(relevance, 0.1)
            self.assertLessEqual(relevance, 1.0)
        
        # Results should be sorted by relevance (highest first)
        for i in range(1, len(relevant_splats)):
            self.assertGreaterEqual(relevant_splats[i-1][1], relevant_splats[i][1])

    def test_create_causal_mask(self):
        """Test creating a causal (lower triangular) mask."""
        # Create causal mask for different sequence lengths
        seq_lengths = [1, 3, 5]
        
        for seq_len in seq_lengths:
            mask = create_causal_mask(seq_len)
            
            # Check shape
            self.assertEqual(mask.shape, (seq_len, seq_len))
            
            # Check values - should be lower triangular
            for i in range(seq_len):
                for j in range(seq_len):
                    if i >= j:
                        # Lower triangle: should be 1
                        self.assertEqual(mask[i, j], 1.0)
                    else:
                        # Upper triangle: should be 0
                        self.assertEqual(mask[i, j], 0.0)

    def test_normalize_attention_rows(self):
        """Test normalizing rows of attention matrix."""
        # Create test attention matrix
        attention = np.array([
            [1.0, 2.0, 3.0],
            [0.5, 1.5, 3.0],
            [0.0, 0.0, 0.0]  # Zero row
        ])
        
        # Normalize rows
        normalized = normalize_attention_rows(attention)
        
        # Check shape
        self.assertEqual(normalized.shape, attention.shape)
        
        # Each row should sum to 1, except zero rows which stay 0
        for i in range(normalized.shape[0]):
            row_sum = np.sum(normalized[i])
            if np.sum(attention[i]) > 0:
                self.assertAlmostEqual(row_sum, 1.0)
            else:
                self.assertAlmostEqual(row_sum, 0.0)
        
        # Relative proportions should be preserved
        for i in range(normalized.shape[0]):
            if np.sum(attention[i]) > 0:
                for j in range(normalized.shape[1] - 1):
                    if attention[i, j] > 0 and attention[i, j+1] > 0:
                        orig_ratio = attention[i, j] / attention[i, j+1]
                        norm_ratio = normalized[i, j] / normalized[i, j+1]
                        self.assertAlmostEqual(orig_ratio, norm_ratio)

    def test_compute_pairwise_splat_distances(self):
        """Test computing pairwise distances between token sets through a splat."""
        # Create a test splat
        splat = self.splats[0]
        
        # Create token sets
        tokens_i = np.array([
            [0.0, 0.0],
            [1.0, 0.0]
        ])
        tokens_j = np.array([
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0]
        ])
        
        # Compute distances
        distances = compute_pairwise_splat_distances(tokens_i, tokens_j, splat)
        
        # Check shape
        self.assertEqual(distances.shape, (len(tokens_i), len(tokens_j)))
        
        # Check specific values
        for i in range(len(tokens_i)):
            for j in range(len(tokens_j)):
                # Compute distance directly
                direct_distance = splat.compute_distance(tokens_i[i], tokens_j[j])
                # Should match the value in the matrix
                self.assertAlmostEqual(distances[i, j], direct_distance)

    def test_apply_sparse_topk_edge_cases(self):
        """Test apply_sparse_topk with edge cases."""
        # Test with k larger than number of columns
        matrix = np.array([
            [0.1, 0.5, 0.2],
            [0.7, 0.3, 0.9]
        ])
        
        # k > number of columns
        k = 5
        result = apply_sparse_topk(matrix, k)
        
        # Should return a copy of the original matrix
        np.testing.assert_array_equal(result, matrix)
        
        # Test with k = 1
        k = 1
        result = apply_sparse_topk(matrix, k)
        
        # Each row should have exactly one non-zero value (the maximum)
        for i in range(result.shape[0]):
            non_zero_indices = np.nonzero(result[i])[0]
            self.assertEqual(len(non_zero_indices), 1)
            max_idx = np.argmax(matrix[i])
            self.assertEqual(non_zero_indices[0], max_idx)

    def test_compute_token_relevance_with_special_cases(self):
        """Test compute_token_relevance with special cases."""
        # Create a splat without covariance_inverse
        position = np.array([0.0, 0.0])
        covariance = np.eye(2) * 0.1
        splat = Splat(
            dim=2,
            position=position,
            covariance=covariance,
            amplitude=2.0,  # Non-default amplitude
            level="token",
            id="special_splat"
        )
        
        # Delete covariance_inverse attribute
        if hasattr(splat, 'covariance_inverse'):
            delattr(splat, 'covariance_inverse')
        
        # Compute relevance
        relevance = compute_token_relevance(self.tokens, splat)
        
        # Check result shape
        self.assertEqual(relevance.shape, (len(self.tokens),))
        
        # All relevance values should be between 0 and amplitude
        self.assertTrue(np.all(relevance >= 0))
        self.assertTrue(np.all(relevance <= splat.amplitude))
        
        # Tokens closer to the splat position should have higher relevance
        distances = np.linalg.norm(self.tokens - splat.position, axis=1)
        sorted_indices = np.argsort(distances)
        self.assertGreaterEqual(relevance[sorted_indices[0]], relevance[sorted_indices[-1]])


if __name__ == "__main__":
    unittest.main()
