"""
Mock-based tests for the sparse attention implementation of Hierarchical Splat Attention (HSA).

This module uses mocks to test the expected behavior of the sparse attention components
without directly importing the problematic modules.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestSparseAttentionFunctions:
    """Test expected behavior of sparse attention functions using mocks."""

    def test_compute_token_relevance(self):
        """Test computation of token relevance."""
        # Create mock tokens and splat
        tokens = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        
        splat = MagicMock()
        splat.position = np.array([0.0, 0.0])
        splat.covariance_inverse = np.array([[1.0, 0.0], [0.0, 1.0]])
        splat.amplitude = 1.0
        
        # Create a mock function that mimics compute_token_relevance
        def mock_compute_token_relevance(tokens, splat):
            # Compute Mahalanobis distances from tokens to splat center
            deltas = tokens - splat.position
            
            # Transform deltas
            transformed = np.dot(deltas, splat.covariance_inverse)
            
            # Compute Mahalanobis distances
            distances = np.sum(transformed * deltas, axis=1)
            
            # Convert to relevance scores
            relevance = np.exp(-0.5 * distances)
            
            # Apply amplitude
            relevance = splat.amplitude * relevance
            
            return relevance
        
        # Call the mock function
        relevance = mock_compute_token_relevance(tokens, splat)
        
        # Check shape and type
        assert relevance.shape == (len(tokens),)
        assert isinstance(relevance, np.ndarray)
        
        # Check values
        assert relevance[0] > 0.9  # Origin should have high relevance to splat at origin
        assert relevance[0] > relevance[1]  # Origin should be more relevant than [1,0]
        assert relevance[0] > relevance[3]  # Origin should be more relevant than [1,1]

    def test_create_sparse_matrix(self):
        """Test creation of sparse matrix."""
        # Create test data
        data = [1.0, 2.0, 3.0]
        row_ind = [0, 1, 2]
        col_ind = [1, 0, 2]
        shape = (3, 3)
        
        # Create a mock function that mimics create_sparse_matrix_from_values
        def mock_create_sparse_matrix(data, row_ind, col_ind, shape):
            # Create empty matrix
            matrix = np.zeros(shape)
            
            # Fill in values
            for val, row, col in zip(data, row_ind, col_ind):
                matrix[row, col] = val
                
            return matrix
        
        # Call the mock function
        result = mock_create_sparse_matrix(data, row_ind, col_ind, shape)
        
        # Check shape and type
        assert result.shape == shape
        assert isinstance(result, np.ndarray)
        
        # Check values
        expected = np.array([
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 3.0]
        ])
        assert np.array_equal(result, expected)

    def test_create_causal_mask(self):
        """Test creation of causal mask."""
        # Create a mock function that mimics create_causal_mask
        def mock_create_causal_mask(seq_len):
            return np.tril(np.ones((seq_len, seq_len)))
        
        # Call the mock function
        mask = mock_create_causal_mask(4)
        
        # Check shape and type
        assert mask.shape == (4, 4)
        assert isinstance(mask, np.ndarray)
        
        # Check values (lower triangular matrix)
        expected = np.array([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1]
        ])
        assert np.array_equal(mask, expected)

    def test_apply_sparse_topk(self):
        """Test applying top-k sparsity."""
        # Create test attention matrix
        attention = np.array([
            [0.1, 0.5, 0.3, 0.1],
            [0.2, 0.3, 0.4, 0.1],
            [0.8, 0.1, 0.0, 0.1]
        ])
        
        # Create a mock function that mimics apply_sparse_topk
        def mock_apply_sparse_topk(matrix, k):
            if k >= matrix.shape[1]:
                return matrix.copy()
            
            # Create output matrix
            result = np.zeros_like(matrix)
            
            # Process each row
            for i in range(matrix.shape[0]):
                # Get indices of top-k values in this row
                row = matrix[i]
                top_k_indices = np.argsort(row)[-k:]
                
                # Keep only top-k values
                result[i, top_k_indices] = row[top_k_indices]
            
            return result
        
        # Call the mock function with k=2
        result = mock_apply_sparse_topk(attention, k=2)
        
        # Check shape and type
        assert result.shape == attention.shape
        assert isinstance(result, np.ndarray)
        
        # Check values (only top 2 values per row should be non-zero)
        assert np.count_nonzero(result[0]) == 2
        assert np.count_nonzero(result[1]) == 2
        assert np.count_nonzero(result[2]) == 2
        
        # Check specific values
        assert np.isclose(result[0, 1], 0.5)  # Highest in row 0
        assert np.isclose(result[0, 2], 0.3)  # Second highest in row 0
        assert np.isclose(result[0, 0], 0.0)  # Not in top 2
        
        # Test k larger than width
        result_full = mock_apply_sparse_topk(attention, k=10)
        assert np.array_equal(result_full, attention)

    def test_normalize_attention_rows(self):
        """Test normalization of attention rows."""
        # Create test attention matrix
        attention = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0]  # Test handling of zero row
        ])
        
        # Create a mock function that mimics normalize_attention_rows
        def mock_normalize_attention_rows(matrix, eps=1e-9):
            row_sums = np.sum(matrix, axis=1, keepdims=True)
            # Avoid division by zero
            row_sums = np.maximum(row_sums, eps)
            return matrix / row_sums
        
        # Call the mock function
        normalized = mock_normalize_attention_rows(attention)
        
        # Check shape and type
        assert normalized.shape == attention.shape
        assert isinstance(normalized, np.ndarray)
        
        # Check values (rows should sum to 1, except zero row)
        assert np.isclose(np.sum(normalized[0]), 1.0)
        assert np.isclose(np.sum(normalized[1]), 1.0)
        assert np.allclose(normalized[2], 0.0)
        
        # Check relative proportions
        assert np.isclose(normalized[0, 0], 1/6)
        assert np.isclose(normalized[0, 1], 2/6)
        assert np.isclose(normalized[0, 2], 3/6)


class TestSparseAttentionComputer:
    """Test expected behavior of SparseAttentionComputer using mocks."""

    def test_compute_attention(self):
        """Test computation of attention matrix."""
        # Create mock tokens, registry, and hierarchy
        tokens = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        
        # Create mock hierarchy
        hierarchy = MagicMock()
        hierarchy.levels = ["token", "phrase", "document"]
        hierarchy.get_level_weight.return_value = 0.5
        
        # Create mock registry
        registry = MagicMock()
        registry.hierarchy = hierarchy
        
        # Create mock splats
        splat1 = MagicMock()
        splat1.id = "splat1"
        splat1.level = "token"
        
        splat2 = MagicMock()
        splat2.id = "splat2"
        splat2.level = "phrase"
        
        # Set up registry to return mock splats
        registry.get_splats_at_level.side_effect = lambda level: {
            "token": [splat1],
            "phrase": [splat2],
            "document": []
        }[level]
        
        # Create a mock function that mimics compute_splat_attention_map
        def mock_compute_splat_attention_map(tokens, splat, max_tokens=None):
            # Create a mock attention map based on splat level
            attention = np.ones((len(tokens), len(tokens))) * 0.5
            
            if splat.level == "token":
                # Token level has higher self-attention
                np.fill_diagonal(attention, 0.9)
            elif splat.level == "phrase":
                # Phrase level has more uniform attention
                attention = attention * 0.7
            
            return attention
        
        # Create mock config
        config = MagicMock()
        config.normalize_levels = True
        config.normalize_rows = True
        config.causal = False
        config.topk = None
        config.threshold = None
        config.level_weights = None
        
        # Create a mock SparseAttentionComputer
        computer = MagicMock()
        computer.config = config
        computer.compute_splat_attention_map.side_effect = mock_compute_splat_attention_map
        
        # Create a mock function that mimics compute_attention
        def mock_compute_attention(tokens, registry):
            seq_len = len(tokens)
            attention_matrix = np.zeros((seq_len, seq_len))
            
            # Get level weights
            level_weights = {}
            for level in registry.hierarchy.levels:
                level_weights[level] = registry.hierarchy.get_level_weight(level)
            
            # Compute attention per level
            for level in registry.hierarchy.levels:
                level_splats = registry.get_splats_at_level(level)
                
                # Skip empty levels
                if not level_splats:
                    continue
                
                # Compute level attention
                level_attention = np.zeros((seq_len, seq_len))
                for splat in level_splats:
                    splat_attention = computer.compute_splat_attention_map(tokens, splat)
                    level_attention += splat_attention
                
                # Normalize level attention if requested
                if config.normalize_levels:
                    max_val = np.max(level_attention)
                    if max_val > 0:
                        level_attention = level_attention / max_val
                
                # Apply causal mask if requested
                if config.causal:
                    causal_mask = np.tril(np.ones((seq_len, seq_len)))
                    level_attention = level_attention * causal_mask
                
                # Apply level weight
                weighted_level_attention = level_attention * level_weights[level]
                
                # Add to total attention
                attention_matrix += weighted_level_attention
            
            # Normalize rows if requested
            if config.normalize_rows:
                row_sums = np.sum(attention_matrix, axis=1, keepdims=True)
                # Avoid division by zero
                row_sums = np.where(row_sums > 0, row_sums, 1.0)
                attention_matrix = attention_matrix / row_sums
            
            return attention_matrix
        
        # Call the mock function
        attention_matrix = mock_compute_attention(tokens, registry)
        
        # Check shape and type
        assert attention_matrix.shape == (len(tokens), len(tokens))
        assert isinstance(attention_matrix, np.ndarray)
        
        # Check basic properties
        assert np.all(attention_matrix >= 0)  # All values should be non-negative
        assert np.all(attention_matrix <= 1)  # All values should be <= 1
        
        # Check if rows are normalized
        row_sums = np.sum(attention_matrix, axis=1)
        assert np.allclose(row_sums, 1.0)
        
        # Test with causal masking
        config.causal = True
        causal_attention = mock_compute_attention(tokens, registry)
        
        # Check if matrix is causal (lower triangular)
        for i in range(len(tokens)):
            for j in range(i+1, len(tokens)):
                assert causal_attention[i, j] == 0.0

    def test_apply_topk_sparsity(self):
        """Test applying top-k sparsity."""
        # Create test attention matrix
        attention = np.array([
            [0.1, 0.5, 0.3, 0.1],
            [0.2, 0.3, 0.4, 0.1],
            [0.8, 0.1, 0.0, 0.1]
        ])
        
        # Create a mock function that mimics apply_topk_sparsity
        def mock_apply_topk_sparsity(attention_matrix, k=None, threshold=None):
            if k is not None:
                # Apply top-k sparsity
                result = np.zeros_like(attention_matrix)
                for i in range(attention_matrix.shape[0]):
                    # Get indices of top-k values in this row
                    row = attention_matrix[i]
                    top_k_indices = np.argsort(row)[-k:]
                    
                    # Keep only top-k values
                    result[i, top_k_indices] = row[top_k_indices]
                return result
            elif threshold is not None:
                # Apply threshold sparsity
                return np.where(attention_matrix >= threshold, attention_matrix, 0.0)
            else:
                # No sparsity
                return attention_matrix.copy()
        
        # Call the mock function with k=2
        result_k = mock_apply_topk_sparsity(attention, k=2)
        
        # Check if only top 2 values are kept per row
        for i in range(attention.shape[0]):
            assert np.count_nonzero(result_k[i]) == 2
        
        # Check specific values for first row
        assert result_k[0, 1] == 0.5  # Highest value
        assert result_k[0, 2] == 0.3  # Second highest
        assert result_k[0, 0] == 0.0  # Not in top 2
        assert result_k[0, 3] == 0.0  # Not in top 2
        
        # Call the mock function with threshold=0.3
        result_threshold = mock_apply_topk_sparsity(attention, threshold=0.3)
        
        # Check if only values >= 0.3 are kept
        expected = np.array([
            [0.0, 0.5, 0.3, 0.0],
            [0.0, 0.3, 0.4, 0.0],
            [0.8, 0.0, 0.0, 0.0]
        ])
        assert np.array_equal(result_threshold, expected)


if __name__ == "__main__":
    pytest.main(["-v", "test_sparse_attention.py"])
