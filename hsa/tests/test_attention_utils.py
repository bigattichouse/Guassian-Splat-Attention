import pytest
import numpy as np

from hsa.attention_utils import (
    apply_causal_mask,
    normalize_rows,
    apply_topk_mask,
    vectorized_pairwise_distances,
    batch_attention_computation
)
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy


class TestApplyCausalMask:
    def test_apply_causal_mask(self):
        # Create a test matrix
        attention_matrix = np.ones((3, 3))
        
        # Apply causal mask
        masked_matrix = apply_causal_mask(attention_matrix)
        
        # Expected result is a lower triangular matrix
        expected = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0]
        ])
        
        # Check if the result matches the expected matrix
        np.testing.assert_array_equal(masked_matrix, expected)
    
    def test_apply_causal_mask_rectangular(self):
        # Create a rectangular test matrix
        attention_matrix = np.ones((2, 3))
        
        # Apply causal mask
        masked_matrix = apply_causal_mask(attention_matrix)
        
        # Expected result - first row should only have the first element as 1
        # and second row should have first and second elements as 1
        expected = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        
        # Check if the result matches the expected matrix
        np.testing.assert_array_equal(masked_matrix, expected)


class TestNormalizeRows:
    def test_normalize_rows(self):
        # Create a test matrix
        attention_matrix = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0]
        ])
        
        # Normalize rows
        normalized_matrix = normalize_rows(attention_matrix)
        
        # Expected result - each row should sum to 1
        row_sums = np.sum(normalized_matrix, axis=1)
        expected_sums = np.ones(2)
        
        # Check if row sums are all 1
        np.testing.assert_allclose(row_sums, expected_sums)
    
    def test_normalize_rows_with_zeros(self):
        # Create a test matrix with a zero row
        attention_matrix = np.array([
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0]
        ])
        
        # Normalize rows
        normalized_matrix = normalize_rows(attention_matrix)
        
        # Expected result - first row should sum to 1, second row should be all 0s
        row_sums = np.sum(normalized_matrix, axis=1)
        expected_sums = np.array([1.0, 0.0])
        
        # Check if row sums match expected values
        np.testing.assert_allclose(row_sums, expected_sums, atol=1e-8)


class TestApplyTopkMask:
    def test_apply_topk_mask_k1(self):
        # Create a test matrix
        attention_matrix = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        
        # Apply top-1 mask
        masked_matrix = apply_topk_mask(attention_matrix, k=1)
        
        # Expected result - only the maximum value in each row should be kept
        expected = np.array([
            [0.0, 0.0, 0.3],
            [0.0, 0.0, 0.6]
        ])
        
        # Check if the result matches the expected matrix
        np.testing.assert_array_equal(masked_matrix, expected)
    
    def test_apply_topk_mask_k2(self):
        # Create a test matrix
        attention_matrix = np.array([
            [0.1, 0.2, 0.3],
            [0.6, 0.5, 0.4]
        ])
        
        # Apply top-2 mask
        masked_matrix = apply_topk_mask(attention_matrix, k=2)
        
        # Expected result - top 2 values in each row should be kept
        expected = np.array([
            [0.0, 0.2, 0.3],
            [0.6, 0.5, 0.0]
        ])
        
        # Check if the result matches the expected matrix
        np.testing.assert_array_equal(masked_matrix, expected)
    
    def test_apply_topk_mask_large_k(self):
        # Create a test matrix
        attention_matrix = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        
        # Apply top-5 mask (larger than matrix width)
        masked_matrix = apply_topk_mask(attention_matrix, k=5)
        
        # Expected result - all values should be kept
        expected = attention_matrix.copy()
        
        # Check if the result matches the expected matrix
        np.testing.assert_array_equal(masked_matrix, expected)


class TestVectorizedPairwiseDistances:
    def test_vectorized_pairwise_distances(self):
        # Create test tokens
        tokens = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        
        # Create splat position and covariance inverse
        splat_position = np.array([0.0, 0.0])
        splat_cov_inv = np.eye(2)  # Identity matrix for simplicity
        
        # Calculate distances
        distances = vectorized_pairwise_distances(tokens, splat_position, splat_cov_inv)
        
        # Expected distances (with identity covariance, these are Euclidean distances)
        # Distance from token to center, squared
        d1 = 1.0  # sqrt(1^2 + 0^2) = 1
        d2 = 1.0  # sqrt(0^2 + 1^2) = 1
        d3 = np.sqrt(2.0)  # sqrt(1^2 + 1^2) = sqrt(2)
        
        # Outer product of distances
        expected = np.array([
            [d1*d1, d1*d2, d1*d3],
            [d2*d1, d2*d2, d2*d3],
            [d3*d1, d3*d2, d3*d3]
        ])
        
        # Check if distances match expected values
        np.testing.assert_allclose(distances, expected)


class TestBatchAttentionComputation:
    def test_batch_attention_computation(self):
        # Create test tokens
        tokens = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0]
        ])
        
        # Create test splats
        splat1 = Splat(
            dim=2,
            position=np.array([0.0, 0.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
            amplitude=1.0,
            level="token",
            id="splat1"
        )
        
        splat2 = Splat(
            dim=2,
            position=np.array([1.0, 1.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
            amplitude=1.0,
            level="token",
            id="splat2"
        )
        
        # Compute batch attention
        attention = batch_attention_computation(tokens, [splat1, splat2])
        
        # Ensure the result has the correct shape
        assert attention.shape == (3, 3)
        
        # Verify diagonal elements are non-zero (self-attention)
        for i in range(3):
            assert attention[i, i] > 0
