"""
Enhanced tests for the Splat class in the HSA implementation with focus on stability.
"""

import pytest
import numpy as np
import logging
from unittest.mock import patch, MagicMock

from hsa.splat import Splat, RingBuffer


class TestRingBuffer:
    """Tests for the RingBuffer class."""
    
    def test_init(self):
        """Test initialization."""
        rb = RingBuffer(5)
        assert rb.capacity == 5
        assert rb.size == 0
        assert rb.index == 0
        assert rb.buffer == [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Test with invalid capacity
        rb = RingBuffer(0)  # Should be corrected to 1
        assert rb.capacity == 1
        
        rb = RingBuffer(-3)  # Should be corrected to 1
        assert rb.capacity == 1
    
    def test_add(self):
        """Test adding values."""
        rb = RingBuffer(3)
        
        # Add first element
        rb.add(1.0)
        assert rb.size == 1
        assert rb.index == 1
        assert rb.buffer == [1.0, 0.0, 0.0]
        
        # Add second element
        rb.add(2.0)
        assert rb.size == 2
        assert rb.index == 2
        assert rb.buffer == [1.0, 2.0, 0.0]
        
        # Add third element
        rb.add(3.0)
        assert rb.size == 3
        assert rb.index == 0
        assert rb.buffer == [1.0, 2.0, 3.0]
        
        # Add fourth element (wrapping)
        rb.add(4.0)
        assert rb.size == 3
        assert rb.index == 1
        assert rb.buffer == [4.0, 2.0, 3.0]
    
    def test_add_non_finite(self):
        """Test adding non-finite values."""
        rb = RingBuffer(3)
        
        # These should be replaced with 0.0
        rb.add(float('nan'))
        rb.add(float('inf'))
        rb.add(float('-inf'))
        
        assert rb.size == 3
        for value in rb.buffer:
            assert value == 0.0
    
    def test_get_values(self):
        """Test getting values."""
        rb = RingBuffer(3)
        
        # Empty buffer
        assert rb.get_values() == []
        
        # Partially filled buffer
        rb.add(1.0)
        rb.add(2.0)
        assert rb.get_values() == [1.0, 2.0]
        
        # Full buffer
        rb.add(3.0)
        assert rb.get_values() == [1.0, 2.0, 3.0]
        
        # Wrapped buffer
        rb.add(4.0)
        assert rb.get_values() == [4.0, 2.0, 3.0]
    
    def test_average(self):
        """Test computing average."""
        rb = RingBuffer(3)
        
        # Empty buffer
        assert rb.average() == 0.0
        
        # Single element
        rb.add(3.0)
        assert rb.average() == 3.0
        
        # Multiple elements
        rb.add(7.0)
        assert rb.average() == 5.0
        
        # Full buffer
        rb.add(2.0)
        assert rb.average() == 4.0
        
        # Wrapped buffer
        rb.add(9.0)
        assert pytest.approx(rb.average()) == (9.0 + 7.0 + 2.0) / 3


class TestSplat:
    """Tests for the Splat class."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        splat = Splat(dim=3)
        
        assert splat.dim == 3
        assert splat.id is not None
        assert isinstance(splat.id, str)
        assert np.array_equal(splat.position, np.zeros(3))
        assert np.array_equal(splat.covariance, np.eye(3))
        assert splat.amplitude == 1.0
        assert splat.level == "token"
        assert splat.parent is None
        assert len(splat.children) == 0
        assert splat.info_contribution == 0.0
        assert splat.lifetime == 0
    
    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        position = np.array([1.0, 2.0, 3.0])
        covariance = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0]
        ])
        
        splat = Splat(
            dim=3,
            position=position,
            covariance=covariance,
            amplitude=2.0,
            level="phrase",
            id="test_splat"
        )
        
        assert splat.dim == 3
        assert splat.id == "test_splat"
        assert np.array_equal(splat.position, position)
        assert np.allclose(splat.covariance, covariance)  # Should be approximately equal due to stability adjustments
        assert splat.amplitude == 2.0
        assert splat.level == "phrase"
    
    def test_init_validation(self):
        """Test validation during initialization."""
        # Invalid dimension
        with pytest.raises(ValueError):
            Splat(dim=0)
        
        with pytest.raises(ValueError):
            Splat(dim=-3)
        
        # Invalid position shape
        with pytest.raises(ValueError):
            Splat(dim=3, position=np.array([1.0, 2.0]))
        
        # Invalid covariance shape
        with pytest.raises(ValueError):
            Splat(dim=3, covariance=np.array([[1.0, 2.0], [3.0, 4.0]]))
    
    def test_parent_child_relationship(self):
        """Test parent-child relationship between splats."""
        parent = Splat(dim=3, level="phrase")
        child = Splat(dim=3, level="token", parent=parent)
        
        assert child.parent == parent
        assert child in parent.children
        assert len(parent.children) == 1
        
        # Add another child
        another_child = Splat(dim=3, level="token", parent=parent)
        assert len(parent.children) == 2
        assert another_child in parent.children
        
    def test_update_parameters(self):
        """Test updating splat parameters."""
        splat = Splat(dim=3)
        
        new_position = np.array([1.0, 2.0, 3.0])
        new_covariance = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0]
        ])
        
        splat.update_parameters(
            position=new_position,
            covariance=new_covariance,
            amplitude=2.0
        )
        
        assert np.array_equal(splat.position, new_position)
        assert np.allclose(splat.covariance, new_covariance)  # Approximately equal due to stability adjustments
        assert splat.amplitude == 2.0
        assert splat.lifetime == 1
        
        # Update just one parameter
        newer_position = np.array([4.0, 5.0, 6.0])
        splat.update_parameters(position=newer_position)
        
        assert np.array_equal(splat.position, newer_position)
        assert np.allclose(splat.covariance, new_covariance)  # Unchanged
        assert splat.amplitude == 2.0  # Unchanged
        assert splat.lifetime == 2
    
    def test_update_parameters_validation(self):
        """Test validation during parameter updates."""
        splat = Splat(dim=3)
        
        # Invalid position shape
        with pytest.raises(ValueError):
            splat.update_parameters(position=np.array([1.0, 2.0]))
        
        # Invalid covariance shape
        with pytest.raises(ValueError):
            splat.update_parameters(covariance=np.array([[1.0, 2.0], [3.0, 4.0]]))
        
        # Invalid amplitude (should log warning but not raise)
        with patch('hsa.splat.logger') as mock_logger:
            splat.update_parameters(amplitude=-1.0)
            mock_logger.warning.assert_called_once()
            assert splat.amplitude == 1.0  # Unchanged
    
    def test_stabilize_covariance(self):
        """Test covariance matrix stabilization."""
        splat = Splat(dim=3)
        
        # Create a poorly conditioned covariance matrix
        bad_covariance = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1e-10, 0.0],
            [0.0, 0.0, 1e10]
        ])
        
        # Stabilize it
        stable_covariance = splat._stabilize_covariance(bad_covariance)
        
        # Check eigenvalues are within bounds
        eigenvalues = np.linalg.eigvalsh(stable_covariance)
        assert np.min(eigenvalues) >= splat.MIN_COVARIANCE_EIGENVALUE
        assert np.max(eigenvalues) <= splat.MAX_COVARIANCE_EIGENVALUE
        
        # Matrix should be symmetric
        assert np.allclose(stable_covariance, stable_covariance.T)
        
        # Test with non-symmetric input (due to numerical errors)
        almost_symmetric = np.array([
            [1.0, 0.1, 0.2],
            [0.1 + 1e-10, 2.0, 0.3],
            [0.2, 0.3 + 1e-10, 3.0]
        ])
        
        stable_covariance = splat._stabilize_covariance(almost_symmetric)
        assert np.allclose(stable_covariance, stable_covariance.T)
    
    def test_stabilize_covariance_fallback(self):
        """Test fallback for covariance stabilization."""
        splat = Splat(dim=3)
        
        # Create a covariance that will fail eigendecomposition
        bad_covariance = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.nan, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Should use fallback method
        with patch('hsa.splat.logger') as mock_logger:
            stable_covariance = splat._stabilize_covariance(bad_covariance)
            mock_logger.warning.assert_called_once()
        
        # Result should be usable
        assert np.all(np.isfinite(stable_covariance))
        assert np.allclose(stable_covariance, stable_covariance.T)
    
    def test_update_cached_values(self):
        """Test updating cached computation values."""
        splat = Splat(dim=2)
        
        # Save original cached values
        original_inverse = splat.covariance_inverse.copy()
        original_normalization = splat.normalization_factor
        
        # Modify covariance and update
        new_covariance = np.array([
            [2.0, 0.5],
            [0.5, 3.0]
        ])
        splat.covariance = new_covariance.copy()
        splat._update_cached_values()
        
        # Check cached values changed
        assert not np.array_equal(splat.covariance_inverse, original_inverse)
        assert splat.normalization_factor != original_normalization
        
        # Check inverse is correct
        expected_inverse = np.linalg.inv(new_covariance)
        assert np.allclose(splat.covariance_inverse, expected_inverse)
        
        # Test with bad covariance (not positive definite)
        bad_covariance = np.array([
            [1.0, 2.0],
            [2.0, 1.0]
        ])
        
        with patch('hsa.splat.logger') as mock_logger:
            splat.covariance = bad_covariance.copy()
            splat._update_cached_values()
            # Should log a warning
            assert mock_logger.warning.called
        
        # Should still have valid cached values
        assert np.all(np.isfinite(splat.covariance_inverse))
        assert np.isfinite(splat.normalization_factor)
    
    def test_compute_distance(self):
        """Test computing distance between tokens through a splat."""
        # Create a splat with identity covariance
        splat = Splat(dim=2)
        
        # For identity covariance, Mahalanobis distance is Euclidean distance
        token_a = np.array([1.0, 0.0])
        token_b = np.array([0.0, 1.0])
        
        # Distance should be |token_a - center| + |token_b - center| = 1 + 1 = 2
        distance = splat.compute_distance(token_a, token_b)
        assert pytest.approx(distance) == 2.0
        
        # Try with off-center splat
        splat.update_parameters(position=np.array([1.0, 1.0]))
        
        # Distance should be |token_a - center| + |token_b - center| = 1 + 1 = 2
        distance = splat.compute_distance(token_a, token_b)
        assert pytest.approx(distance) == 2.0
        
        # Try with non-identity covariance
        covariance = np.array([
            [2.0, 0.0],
            [0.0, 0.5]
        ])
        splat.update_parameters(covariance=covariance)
        
        # For this covariance, distances are scaled differently in each dimension
        distance = splat.compute_distance(token_a, token_b)
        
        # Expected values: 
        # token_a to center: sqrt((0-1)^2/2 + (0-1)^2/0.5) = sqrt(0.5 + 2) = sqrt(2.5)
        # token_b to center: sqrt((1-1)^2/2 + (1-1)^2/0.5) = sqrt(0 + 0) = 0
        expected = np.sqrt(2.5) + 0
        assert pytest.approx(distance) == expected
    
    def test_compute_distance_validation(self):
        """Test validation in distance computation."""
        splat = Splat(dim=2)
        
        # Invalid token shapes
        with pytest.raises(ValueError):
            splat.compute_distance(np.array([1.0]), np.array([2.0]))
        
        with pytest.raises(ValueError):
            splat.compute_distance(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
            
        # Test fallback to Euclidean distance on error
        with patch.object(splat, 'covariance_inverse', 
                         new=np.array([[np.nan, 0], [0, 1]])):
            with patch('hsa.splat.logger') as mock_logger:
                # Should use fallback
                distance = splat.compute_distance(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
                # Should log a warning
                assert mock_logger.warning.called
                # Should still return a finite value
                assert np.isfinite(distance)
    
    def test_compute_attention(self):
        """Test computing attention between tokens through a splat."""
        # Create a splat with identity covariance
        splat = Splat(dim=2)
        
        # For identity covariance, same token should have high attention
        token = np.array([0.0, 0.0])  # At center
        attention = splat.compute_attention(token, token)
        
        # Should be maximum
        assert pytest.approx(attention) == 1.0
        
        # Different tokens should have lower attention
        token_a = np.array([1.0, 0.0])
        token_b = np.array([0.0, 1.0])
        attention = splat.compute_attention(token_a, token_b)
        
        # Should be less than 1.0
        assert attention < 1.0
        
        # Test with higher amplitude
        splat.update_parameters(amplitude=2.0)
        attention = splat.compute_attention(token_a, token_b)
        
        # Should be higher with higher amplitude, but capped at 1.0
        assert attention > 0.0
        assert attention <= 1.0
    
    def test_compute_attention_validation(self):
        """Test validation in attention computation."""
        splat = Splat(dim=2)
        
        # Invalid token shapes
        with pytest.raises(ValueError):
            splat.compute_attention(np.array([1.0]), np.array([2.0]))
        
        with pytest.raises(ValueError):
            splat.compute_attention(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))
        
        # Test with error in computation
        with patch.object(splat, 'covariance_inverse', 
                         new=np.array([[np.nan, 0], [0, 1]])):
            with patch('hsa.splat.logger') as mock_logger:
                # Should handle error
                attention = splat.compute_attention(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
                # Should log a warning
                assert mock_logger.warning.called
                # Should return 0.0 for safety
                assert attention == 0.0
                # Should still update activation history
                assert splat.activation_history.size > 0
    
    def test_compute_attention_numerical_bounds(self):
        """Test numerical bounds in attention computation."""
        splat = Splat(dim=2)
        
        # Test with very large Mahalanobis distances
        far_token_a = np.array([100.0, 0.0])
        far_token_b = np.array([0.0, 100.0])
        
        attention = splat.compute_attention(far_token_a, far_token_b)
        
        # Should be very close to zero but finite
        assert attention >= 0.0
        assert attention < 1e-10
        assert np.isfinite(attention)
        
        # Test with very large amplitude
        splat.update_parameters(amplitude=1e6)
        
        attention = splat.compute_attention(np.array([0.1, 0.1]), np.array([0.1, 0.1]))
        
        # Should be capped at 1.0
        assert attention <= 1.0
    
    def test_get_average_activation(self):
        """Test getting average activation from history."""
        splat = Splat(dim=2)
        
        # Initially should be zero
        assert splat.get_average_activation() == 0.0
        
        # Add some activations by computing attention
        token_a = np.array([0.0, 0.0])
        token_b = np.array([1.0, 0.0])
        
        _ = splat.compute_attention(token_a, token_a)  # High activation
        _ = splat.compute_attention(token_a, token_b)  # Lower activation
        
        # Average should be between the two
        avg = splat.get_average_activation()
        assert 0.0 < avg < 1.0
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Create a splat with very small covariance
        small_cov = np.array([
            [1e-10, 0.0],
            [0.0, 1e-10]
        ])
        splat = Splat(dim=2, covariance=small_cov)
        
        # Covariance should be regularized to avoid numerical issues
        min_eig = splat.MIN_COVARIANCE_EIGENVALUE
        eigenvalues = np.linalg.eigvalsh(splat.covariance)
        assert np.min(eigenvalues) >= min_eig
        
        # Far token should not cause numerical issues
        far_token = np.array([1e6, 1e6])
        attention = splat.compute_attention(far_token, far_token)
        
        # Should be very close to zero but finite
        assert attention >= 0.0
        assert np.isfinite(attention)
        
        # Very close tokens should not cause numerical issues
        close_tokens = np.array([1e-10, 1e-10])
        attention = splat.compute_attention(close_tokens, close_tokens)
        
        # Should be positive and finite
        assert attention > 0.0
        assert np.isfinite(attention)
    
    def test_clone(self):
        """Test cloning a splat."""
        # Create a splat with custom parameters
        original = Splat(
            dim=2,
            position=np.array([1.0, 2.0]),
            covariance=np.array([[2.0, 0.5], [0.5, 3.0]]),
            amplitude=2.0,
            level="phrase",
            id="original"
        )
        
        # Add some activation history
        token = np.array([1.0, 2.0])
        original.compute_attention(token, token)
        
        # Clone it
        clone = original.clone()
        
        # Check that parameters match
        assert clone.dim == original.dim
        assert clone.id != original.id  # Should have a new ID
        assert np.array_equal(clone.position, original.position)
        assert np.array_equal(clone.covariance, original.covariance)
        assert clone.amplitude == original.amplitude
        assert clone.level == original.level
        assert clone.parent == original.parent
        
        # Check that history was copied
        assert clone.activation_history.size == original.activation_history.size
        assert clone.get_average_activation() == original.get_average_activation()
        
        # Check that metrics were copied
        assert clone.info_contribution == original.info_contribution
        assert clone.lifetime == original.lifetime
        
        # Clone with specified ID
        clone2 = original.clone(new_id="custom_id")
        assert clone2.id == "custom_id"
        
        # Modify original - clone should remain unchanged
        original.update_parameters(position=np.array([3.0, 4.0]))
        assert not np.array_equal(clone.position, original.position)
    
    def test_repr(self):
        """Test string representation."""
        splat = Splat(dim=2, id="test_splat", level="token", amplitude=0.5)
        splat.lifetime = 10
        
        # Should contain key information
        repr_str = repr(splat)
        assert "Splat" in repr_str
        assert "id=test_splat" in repr_str
        assert "level=token" in repr_str
        assert "amplitude=0.500" in repr_str
        assert "lifetime=10" in repr_str
