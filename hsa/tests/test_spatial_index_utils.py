"""
Tests for the spatial_index_utils module in Hierarchical Splat Attention (HSA).

This module provides test coverage for the utility functions in spatial_index_utils.py.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import the module to test
from hsa.spatial_index_utils import (
    compute_bounding_box,
    compute_centroid,
    compute_distance_matrix,
    find_closest_pairs,
    find_farthest_pairs
)

# Import dependencies for testing
from hsa.splat import Splat


class TestSpatialIndexUtils:
    """Test suite for spatial_index_utils.py functions."""

    def setup_method(self):
        """Set up test data before each test."""
        # Create some test splats with known positions
        self.splat1 = Mock(spec=Splat)
        self.splat1.position = np.array([0.0, 0.0])
        self.splat1.id = "splat1"

        self.splat2 = Mock(spec=Splat)
        self.splat2.position = np.array([1.0, 0.0])
        self.splat2.id = "splat2"

        self.splat3 = Mock(spec=Splat)
        self.splat3.position = np.array([0.0, 1.0])
        self.splat3.id = "splat3"

        self.splat4 = Mock(spec=Splat)
        self.splat4.position = np.array([1.0, 1.0])
        self.splat4.id = "splat4"

        self.splat5 = Mock(spec=Splat)
        self.splat5.position = np.array([-1.0, -1.0])
        self.splat5.id = "splat5"

        # Create array of test splats
        self.test_splats = [self.splat1, self.splat2, self.splat3, self.splat4, self.splat5]

    def test_compute_bounding_box_with_splats(self):
        """Test computation of bounding box with multiple splats."""
        min_coords, max_coords = compute_bounding_box(self.test_splats)

        # Check expected bounds
        np.testing.assert_array_equal(min_coords, np.array([-1.0, -1.0]))
        np.testing.assert_array_equal(max_coords, np.array([1.0, 1.0]))

    def test_compute_bounding_box_empty(self):
        """Test computation of bounding box with empty list."""
        result = compute_bounding_box([])
        
        # Should return None, None for empty list
        assert result == (None, None)

    def test_compute_bounding_box_single_splat(self):
        """Test computation of bounding box with a single splat."""
        min_coords, max_coords = compute_bounding_box([self.splat1])

        # Min and max should be the same for a single point
        np.testing.assert_array_equal(min_coords, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(max_coords, np.array([0.0, 0.0]))

    def test_compute_centroid_with_splats(self):
        """Test computation of centroid with multiple splats."""
        centroid = compute_centroid(self.test_splats)

        # Expected centroid is average of all positions
        expected_centroid = np.array([0.2, 0.2])  # (0+1+0+1-1)/5, (0+0+1+1-1)/5
        np.testing.assert_array_almost_equal(centroid, expected_centroid)

    def test_compute_centroid_empty(self):
        """Test computation of centroid with empty list."""
        centroid = compute_centroid([])
        
        # Should return None for empty list
        assert centroid is None

    def test_compute_centroid_single_splat(self):
        """Test computation of centroid with a single splat."""
        centroid = compute_centroid([self.splat1])
        
        # Centroid should be the position of the single splat
        np.testing.assert_array_equal(centroid, np.array([0.0, 0.0]))

    def test_compute_distance_matrix_with_splats(self):
        """Test computation of distance matrix with multiple splats."""
        distance_matrix = compute_distance_matrix(self.test_splats)
        
        # Check dimensions
        assert distance_matrix.shape == (5, 5)
        
        # Check diagonal elements (should be 0)
        for i in range(5):
            assert distance_matrix[i, i] == 0.0
            
        # Check specific known distances
        assert distance_matrix[0, 1] == 1.0  # Distance between splat1 and splat2
        assert distance_matrix[0, 2] == 1.0  # Distance between splat1 and splat3
        assert distance_matrix[0, 4] == np.sqrt(2)  # Distance between splat1 and splat5
        
        # Check symmetry
        for i in range(5):
            for j in range(5):
                assert distance_matrix[i, j] == distance_matrix[j, i]

    def test_compute_distance_matrix_empty(self):
        """Test computation of distance matrix with empty list."""
        distance_matrix = compute_distance_matrix([])
        
        # Should return empty array
        assert distance_matrix.size == 0
        assert isinstance(distance_matrix, np.ndarray)

    def test_compute_distance_matrix_single_splat(self):
        """Test computation of distance matrix with a single splat."""
        distance_matrix = compute_distance_matrix([self.splat1])
        
        # Should return 1x1 matrix with 0
        assert distance_matrix.shape == (1, 1)
        assert distance_matrix[0, 0] == 0.0

    def test_find_closest_pairs_with_splats(self):
        """Test finding closest pairs of splats."""
        closest_pairs = find_closest_pairs(self.test_splats, k=3)
        
        # Should return 3 pairs
        assert len(closest_pairs) == 3
        
        # Each pair should be a tuple of (index_i, index_j, distance)
        for pair in closest_pairs:
            assert len(pair) == 3
            assert isinstance(pair[0], int)
            assert isinstance(pair[1], int)
            assert isinstance(pair[2], float)
            
        # Pairs should be sorted by distance (ascending)
        assert closest_pairs[0][2] <= closest_pairs[1][2] <= closest_pairs[2][2]
        
        # Check specific pairs (assuming specific order based on how splats were created)
        # The closest pairs should include adjacent splats (distance = 1.0)
        assert closest_pairs[0][2] == 1.0  # First pair has distance 1.0

    def test_find_closest_pairs_empty(self):
        """Test finding closest pairs with empty list."""
        closest_pairs = find_closest_pairs([])
        
        # Should return empty list
        assert closest_pairs == []

    def test_find_closest_pairs_single_splat(self):
        """Test finding closest pairs with a single splat."""
        closest_pairs = find_closest_pairs([self.splat1])
        
        # Should return empty list (need at least 2 splats for a pair)
        assert closest_pairs == []

    def test_find_closest_pairs_k_too_large(self):
        """Test finding closest pairs with k larger than number of possible pairs."""
        # With 5 splats, there are 10 possible pairs (5 choose 2)
        closest_pairs = find_closest_pairs(self.test_splats, k=20)
        
        # Should return all 10 pairs
        assert len(closest_pairs) == 10

    def test_find_farthest_pairs_with_splats(self):
        """Test finding farthest pairs of splats."""
        farthest_pairs = find_farthest_pairs(self.test_splats, k=3)
        
        # Should return 3 pairs
        assert len(farthest_pairs) == 3
        
        # Each pair should be a tuple of (index_i, index_j, distance)
        for pair in farthest_pairs:
            assert len(pair) == 3
            assert isinstance(pair[0], int)
            assert isinstance(pair[1], int)
            assert isinstance(pair[2], float)
            
        # Pairs should be sorted by distance (descending)
        assert farthest_pairs[0][2] >= farthest_pairs[1][2] >= farthest_pairs[2][2]
        
        # Check specific pairs
        # The farthest pairs should include diagonal splats (distance = sqrt(8))
        # e.g., splat4 (1,1) and splat5 (-1,-1)
        assert farthest_pairs[0][2] == pytest.approx(2.0 * np.sqrt(2))  # Distance is 2*sqrt(2)

    def test_find_farthest_pairs_empty(self):
        """Test finding farthest pairs with empty list."""
        farthest_pairs = find_farthest_pairs([])
        
        # Should return empty list
        assert farthest_pairs == []

    def test_find_farthest_pairs_single_splat(self):
        """Test finding farthest pairs with a single splat."""
        farthest_pairs = find_farthest_pairs([self.splat1])
        
        # Should return empty list (need at least 2 splats for a pair)
        assert farthest_pairs == []

    def test_find_farthest_pairs_k_too_large(self):
        """Test finding farthest pairs with k larger than number of possible pairs."""
        # With 5 splats, there are 10 possible pairs (5 choose 2)
        farthest_pairs = find_farthest_pairs(self.test_splats, k=20)
        
        # Should return all 10 pairs
        assert len(farthest_pairs) == 10
