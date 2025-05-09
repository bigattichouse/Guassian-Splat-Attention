"""
Tests for the Splat implementation.
"""

import pytest
import torch
import sys
import numpy as np
from gsa.splat import Splat
from gsa.numeric_utils import generate_random_covariance


@pytest.fixture
def embedding_dim():
    return 8


@pytest.fixture
def sample_splat(embedding_dim):
    position = torch.randn(embedding_dim)
    covariance = generate_random_covariance(embedding_dim)
    amplitude = 0.8
    return Splat(position, covariance, amplitude)


def test_splat_initialization(embedding_dim):
    """Test that a splat can be initialized with valid parameters."""
    position = torch.randn(embedding_dim)
    covariance = generate_random_covariance(embedding_dim)
    amplitude = 0.8
    
    splat = Splat(position, covariance, amplitude)
    
    assert splat.embedding_dim == embedding_dim
    assert torch.allclose(splat.position, position)
    assert splat.amplitude == amplitude
    assert splat.id is not None
    assert len(splat.activation_history) == 0


def test_splat_initialization_with_custom_id():
    """Test that a splat can be initialized with a custom ID."""
    embedding_dim = 4
    position = torch.randn(embedding_dim)
    covariance = generate_random_covariance(embedding_dim)
    custom_id = "test_splat_123"
    
    splat = Splat(position, covariance, splat_id=custom_id)
    
    assert splat.id == custom_id


def test_splat_initialization_with_invalid_position():
    """Test that initialization fails with invalid position shape."""
    embedding_dim = 8
    position = torch.randn(embedding_dim, 2)  # Wrong shape
    covariance = generate_random_covariance(embedding_dim)
    
    with pytest.raises(ValueError):
        Splat(position, covariance)


def test_splat_initialization_with_invalid_covariance():
    """Test that initialization fails with invalid covariance shape."""
    embedding_dim = 8
    position = torch.randn(embedding_dim)
    covariance = generate_random_covariance(embedding_dim - 1)  # Wrong shape
    
    with pytest.raises(ValueError):
        Splat(position, covariance)


def test_normalization_factor_extreme_eigenvalues():
    """Test normalization factor computation with extreme eigenvalues."""
    embedding_dim = 3
    position = torch.zeros(embedding_dim)
    
    # Create a covariance matrix with extremely small eigenvalues
    small_eig_cov = torch.eye(embedding_dim) * 1e-10
    
    # Create a covariance matrix with extremely large eigenvalues
    large_eig_cov = torch.eye(embedding_dim) * 1e10
    
    # Both should create valid splats with finite normalization factors
    splat_small = Splat(position, small_eig_cov)
    splat_large = Splat(position, large_eig_cov)
    
    assert torch.isfinite(splat_small.normalization_factor)
    assert torch.isfinite(splat_large.normalization_factor)
    assert splat_small.normalization_factor > 0
    assert splat_large.normalization_factor > 0


def test_compute_normalization_factor():
    """Test computation of normalization factor."""
    embedding_dim = 3
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    
    splat = Splat(position, covariance)
    
    # For identity covariance, normalization should be (2π)^(-d/2)
    expected = (2 * np.pi) ** (-embedding_dim / 2)
    assert splat.normalization_factor.item() == pytest.approx(expected, rel=1e-5)
    
    # Test with non-positive determinant
    covariance_bad = torch.ones((embedding_dim, embedding_dim))  # Rank-1 matrix
    
    # Patch slogdet to simulate negative determinant
    orig_slogdet = torch.linalg.slogdet
    
    def mock_slogdet(*args, **kwargs):
        return torch.tensor(-1.0), torch.tensor(0.0)
    
    torch.linalg.slogdet = mock_slogdet
    
    try:
        splat = Splat(position, covariance_bad)
        assert torch.isfinite(splat.normalization_factor)
        assert splat.normalization_factor > 0
    finally:
        torch.linalg.slogdet = orig_slogdet
    
    # Test extreme logdet values
    def mock_slogdet_extreme_small(*args, **kwargs):
        return torch.tensor(1.0), torch.tensor(-100.0)
    
    torch.linalg.slogdet = mock_slogdet_extreme_small
    
    try:
        splat = Splat(position, covariance)
        assert torch.isfinite(splat.normalization_factor)
        assert splat.normalization_factor > 0
    finally:
        torch.linalg.slogdet = orig_slogdet
    
    def mock_slogdet_extreme_large(*args, **kwargs):
        return torch.tensor(1.0), torch.tensor(100.0)  # Very large value
    
    torch.linalg.slogdet = mock_slogdet_extreme_large
    
    try:
        splat = Splat(position, covariance)
        assert torch.isfinite(splat.normalization_factor)
        assert splat.normalization_factor > 0
    finally:
        torch.linalg.slogdet = orig_slogdet
    
    # Test with error in slogdet
    def mock_slogdet_error(*args, **kwargs):
        raise RuntimeError("Simulated slogdet error")
    
    torch.linalg.slogdet = mock_slogdet_error
    
    # We need to mock torch.pi and torch.sqrt for the fallback path
    orig_pi = torch.pi
    orig_sqrt = torch.sqrt
    
    # Force torch.pi to be a tensor
    torch.pi = torch.tensor(np.pi)
    
    try:
        splat = Splat(position, covariance)
        assert torch.isfinite(splat.normalization_factor)
        assert splat.normalization_factor > 0
    finally:
        torch.linalg.slogdet = orig_slogdet
        torch.pi = orig_pi


def test_non_finite_normalization_factor():
    """Test handling of non-finite normalization factor."""
    embedding_dim = 3
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance)
    
    # Create a patch to force normalization factor to be non-finite
    orig_exp = torch.exp
    
    def mock_exp(x):
        if isinstance(x, torch.Tensor) and x.numel() == 1:
            return torch.tensor(float('nan'), device=x.device)
        return orig_exp(x)
    
    # Apply the patch
    torch.exp = mock_exp
    
    try:
        # Force recomputation of normalization factor with non-finite result
        splat._compute_normalization_factor()
        # This should trigger the fallback code on lines 120-121
        assert torch.isfinite(splat.normalization_factor)
        assert splat.normalization_factor > 0
    finally:
        # Restore original function
        torch.exp = orig_exp


def test_compute_attention(sample_splat, embedding_dim):
    """Test attention computation between two tokens."""
    token_a = torch.randn(embedding_dim)
    token_b = torch.randn(embedding_dim)
    
    attention = sample_splat.compute_attention(token_a, token_b)
    
    assert isinstance(attention, torch.Tensor)
    assert attention.shape == torch.Size([])  # Scalar output
    assert attention >= 0.0  # Attention should be non-negative


def test_compute_attention_special_cases():
    """Test attention computation for special cases."""
    # Test the 2D special case
    embedding_dim = 2
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance)
    
    zero_token = torch.zeros(embedding_dim)
    
    attention = splat.compute_attention(zero_token, zero_token)
    assert attention == 1.0
    
    # Test with error in gaussian computation
    embedding_dim = 4
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance)
    
    # Create tokens with NaN to trigger error
    token_a = torch.tensor([float('nan'), 0.0, 0.0, 0.0])
    token_b = torch.zeros(embedding_dim)
    
    attention = splat.compute_attention(token_a, token_b)
    assert attention == 0.0
    
    # Test with invalid precision matrix
    position = torch.zeros(embedding_dim)
    covariance = torch.ones((embedding_dim, embedding_dim))  # Singular matrix
    splat = Splat(position, covariance)  # Constructor should fix it
    
    # Replace precision with invalid matrix
    splat.precision = torch.ones((embedding_dim, embedding_dim)) * float('nan')
    
    attention = splat.compute_attention(token_b, token_b)
    assert attention == 0.0


def test_compute_attention_batch(sample_splat, embedding_dim):
    """Test batch attention computation."""
    batch_size = 5
    tokens = torch.randn(batch_size, embedding_dim)
    
    attention_matrix = sample_splat.compute_attention_batch(tokens)
    
    assert isinstance(attention_matrix, torch.Tensor)
    assert attention_matrix.shape == (batch_size, batch_size)
    assert (attention_matrix >= 0.0).all()  # All attention values should be non-negative


def test_compute_attention_batch_special_cases():
    """Test batch attention computation for special cases."""
    # Test the 2D special case
    embedding_dim = 2
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance)
    
    batch_size = 3
    tokens = torch.zeros(batch_size, embedding_dim)
    
    attention_matrix = splat.compute_attention_batch(tokens)
    assert torch.allclose(attention_matrix, torch.ones(batch_size, batch_size))
    
    # Test with small batch size (should use pairwise computation)
    embedding_dim = 4
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance)
    
    small_batch = torch.randn(3, embedding_dim)
    attention_matrix = splat.compute_attention_batch(small_batch)
    assert attention_matrix.shape == (3, 3)
    
    # Test pairwise computation with error
    # Create a mock to raise error during compute_attention
    orig_compute_attention = splat.compute_attention
    
    def mock_compute_attention(*args, **kwargs):
        raise RuntimeError("Simulated compute_attention error")
    
    splat.compute_attention = mock_compute_attention
    
    try:
        attention_matrix = splat.compute_attention_batch(small_batch)
        assert torch.allclose(attention_matrix, torch.zeros(3, 3))
    finally:
        splat.compute_attention = orig_compute_attention
    
    # Test large batch vectorized computation
    large_batch = torch.randn(15, embedding_dim)
    attention_matrix = splat.compute_attention_batch(large_batch)
    assert attention_matrix.shape == (15, 15)
    
    # Test vectorized computation with error
    def mock_matmul(*args, **kwargs):
        raise RuntimeError("Simulated matmul error")
    
    orig_matmul = torch.matmul
    torch.matmul = mock_matmul
    
    try:
        attention_matrix = splat.compute_attention_batch(large_batch)
        assert torch.allclose(attention_matrix, torch.zeros(15, 15))
    finally:
        torch.matmul = orig_matmul


def test_compute_attention_batch_with_errors():
    """Test batch attention computation with various error conditions."""
    embedding_dim = 4
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance)
    
    # Test vectorized computation with error in sum
    orig_sum = torch.sum
    
    def mock_sum(*args, **kwargs):
        if len(args) > 0 and torch.is_tensor(args[0]):
            if args[0].dim() > 1:  # Only intercept the weighted_deltas * deltas sum
                raise RuntimeError("Simulated sum error")
        return orig_sum(*args, **kwargs)
    
    torch.sum = mock_sum
    
    try:
        batch = torch.randn(15, embedding_dim)  # Large enough for vectorized path
        attention_matrix = splat.compute_attention_batch(batch)
        assert attention_matrix.shape == (15, 15)
        assert torch.allclose(attention_matrix, torch.zeros(15, 15))  # Should return zeros on error
    finally:
        torch.sum = orig_sum


def test_update_activation(sample_splat):
    """Test updating activation history."""
    sample_splat.update_activation(0.5)
    sample_splat.update_activation(0.7)
    
    assert len(sample_splat.activation_history) == 2
    assert sample_splat.get_average_activation() == pytest.approx(0.6)


def test_update_activation_non_finite():
    """Test handling of non-finite activation values."""
    embedding_dim = 4
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance)
    
    splat.update_activation(float('nan'))
    splat.update_activation(float('inf'))
    
    assert len(splat.activation_history) == 2
    assert splat.get_average_activation() == 0.0  # Both values should be replaced with 0.0


def test_activation_non_finite_tracking():
    """Test handling of non-finite activation values in activation_history."""
    embedding_dim = 4
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance)
    
    # Create an activation value that is not a finite number
    non_finite_value = float('nan')
    
    # This should use the error handling in update_activation
    splat.update_activation(non_finite_value)
    
    # Get the activation history and confirm it contains a finite value
    values = splat.activation_history.get_values()
    assert len(values) == 1
    assert np.isfinite(values[0])
    assert values[0] == 0.0  # Non-finite values should be replaced with 0.0


def test_get_average_activation_empty():
    """Test average activation with empty history."""
    embedding_dim = 8
    position = torch.randn(embedding_dim)
    covariance = generate_random_covariance(embedding_dim)
    splat = Splat(position, covariance)
    
    assert splat.get_average_activation() == 0.0


def test_update_parameters(sample_splat, embedding_dim):
    """Test updating splat parameters."""
    original_position = sample_splat.position.clone()
    original_covariance = sample_splat.covariance.clone()
    original_amplitude = sample_splat.amplitude
    
    position_delta = torch.randn(embedding_dim) * 0.1
    covariance_delta = torch.randn(embedding_dim, embedding_dim) * 0.1
    amplitude_delta = 0.1
    
    sample_splat.update_parameters(position_delta, covariance_delta, amplitude_delta)
    
    # Check that parameters were updated
    assert not torch.allclose(sample_splat.position, original_position)
    assert not torch.allclose(sample_splat.covariance, original_covariance)
    assert sample_splat.amplitude != original_amplitude
    
    # Check that the covariance is still positive definite
    eigenvalues = torch.linalg.eigvalsh(sample_splat.covariance)
    assert (eigenvalues > 0).all()


def test_update_parameters_direct():
    """Test updating parameters directly rather than via deltas."""
    embedding_dim = 4
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance, amplitude=1.0)
    
    new_position = torch.ones(embedding_dim)
    new_covariance = torch.eye(embedding_dim) * 2.0
    new_amplitude = 0.5
    
    splat.update_parameters(
        position=new_position,
        covariance=new_covariance,
        amplitude=new_amplitude
    )
    
    assert torch.allclose(splat.position, new_position)
    assert torch.allclose(splat.covariance, new_covariance)
    assert splat.amplitude == new_amplitude


def test_update_parameters_error_handling():
    """Test error handling in parameter updates."""
    embedding_dim = 4
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance, amplitude=1.0)
    
    # The update_parameters method catches ValueErrors internally and logs them
    # Let's patch the _ensure_positive_definite method to simulate an error
    orig_ensure_pd = splat._compute_normalization_factor
    
    def mock_ensure_pd(*args, **kwargs):
        raise ValueError("Simulated positive definite error")
    
    splat._compute_normalization_factor = mock_ensure_pd
    
    # Test with invalid position shape - should be handled without raising exception
    invalid_position = torch.zeros(embedding_dim + 1)
    splat.update_parameters(position=invalid_position)
    
    # Position should remain unchanged
    assert torch.allclose(splat.position, position)
    
    # Test with invalid covariance shape
    invalid_covariance = torch.eye(embedding_dim + 1)
    splat.update_parameters(covariance=invalid_covariance)
    
    # Covariance should remain unchanged
    assert torch.allclose(splat.covariance, covariance)
    
    # Test with negative amplitude
    splat.update_parameters(amplitude=-0.5)
    assert splat.amplitude >= 0.0  # Should be clamped to 0 or a small positive value
    
    # Restore original method
    splat._compute_normalization_factor = orig_ensure_pd


def test_update_parameters_invalid_combinations():
    """Test parameter updates with invalid combinations of inputs."""
    embedding_dim = 4
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance)
    
    # Test with completely invalid covariance that can't be fixed
    # This should trigger the error path where ensure_positive_definite also fails
    invalid_cov = torch.ones((embedding_dim, embedding_dim)) * float('nan')
    
    # Should not raise an exception but handle gracefully
    splat.update_parameters(covariance=invalid_cov)
    
    # The covariance should remain valid
    assert torch.isfinite(splat.covariance).all()
    assert torch.isfinite(splat.precision).all()


def test_compute_distance_to(embedding_dim):
    """Test computing distance between two splats."""
    position1 = torch.zeros(embedding_dim)
    position2 = torch.ones(embedding_dim)
    covariance = torch.eye(embedding_dim)
    
    splat1 = Splat(position1, covariance)
    splat2 = Splat(position2, covariance)
    
    distance = splat1.compute_distance_to(splat2)
    
    assert distance > 0.0
    # For identity covariance, this should be the Euclidean distance
    assert distance == pytest.approx(np.sqrt(embedding_dim))


def test_compute_distance_to_error_handling():
    """Test error handling in distance computation."""
    embedding_dim = 4
    splat1 = Splat(torch.zeros(embedding_dim), torch.eye(embedding_dim))
    splat2 = Splat(torch.ones(embedding_dim), torch.eye(embedding_dim))
    
    # Create a patched tensor.mv that raises an error
    orig_mv = torch.mv
    
    def mock_mv(*args, **kwargs):
        raise RuntimeError("Simulated mv error")
    
    torch.mv = mock_mv
    
    try:
        # Should fallback to Euclidean distance
        distance = splat1.compute_distance_to(splat2)
        assert distance == pytest.approx(np.sqrt(embedding_dim))
    finally:
        torch.mv = orig_mv


def test_clone(sample_splat):
    """Test creating a clone of a splat."""
    # Clone with default ID
    clone = sample_splat.clone()
    
    assert clone.id != sample_splat.id  # New ID should be assigned
    assert torch.allclose(clone.position, sample_splat.position)
    assert torch.allclose(clone.covariance, sample_splat.covariance)
    assert clone.amplitude == sample_splat.amplitude
    assert clone.get_average_activation() == sample_splat.get_average_activation()
    
    # Clone with custom ID
    custom_id = "custom_clone_id"
    clone = sample_splat.clone(new_id=custom_id)
    
    assert clone.id == custom_id


def test_trend_analysis(sample_splat):
    """Test activation trend analysis."""
    # Add some increasing activations
    sample_splat.update_activation(0.1)
    sample_splat.update_activation(0.2)
    sample_splat.update_activation(0.3)
    
    trend = sample_splat.trend_analysis()
    
    assert isinstance(trend, dict)
    assert 'increasing' in trend
    assert 'decreasing' in trend
    assert 'stable' in trend
    assert 'slope' in trend
    
    assert trend['increasing'] is True
    assert trend['slope'] > 0


def test_to_dict_and_from_dict(sample_splat):
    """Test serialization and deserialization of splats."""
    data = sample_splat.to_dict()
    
    assert isinstance(data, dict)
    assert 'id' in data
    assert 'position' in data
    assert 'covariance' in data
    assert 'amplitude' in data
    
    # Create a new splat from the dictionary
    new_splat = Splat.from_dict(data)
    
    assert new_splat.id == sample_splat.id
    assert torch.allclose(new_splat.position, sample_splat.position)
    assert torch.allclose(new_splat.covariance, sample_splat.covariance)
    assert new_splat.amplitude == sample_splat.amplitude


def test_from_dict_error_handling():
    """Test error handling in from_dict method."""
    # Test with missing position
    data = {
        'id': 'test_splat',
        'covariance': [[1.0, 0.0], [0.0, 1.0]],
        'amplitude': 1.0
    }
    
    # Should create a fallback splat
    splat = Splat.from_dict(data)
    assert splat.id == 'test_splat'
    assert splat.position.shape == (1,)  # Default dim is derived from position length
    
    # Test with complete error
    # Mock the from_dict error case by patching the Splat constructor
    orig_init = Splat.__init__
    
    def mock_init(self, *args, **kwargs):
        orig_init(self, torch.zeros(2), torch.eye(2), splat_id=kwargs.get('splat_id', None))
    
    Splat.__init__ = mock_init
    
    try:
        splat = Splat.from_dict({'id': 'test_splat', 'position': [1.0, 2.0, 3.0]})
        assert splat.id == 'test_splat'
        assert torch.allclose(splat.position, torch.zeros(2))  # Should create fallback
    finally:
        Splat.__init__ = orig_init


def test_from_dict_complex_error():
    """Test error handling in from_dict when tensor conversion fails."""
    # Create a dict with valid structure but will cause tensor conversion to fail
    data = {
        'id': 'test_splat',
        'position': ['not', 'a', 'number'],  # This will fail tensor conversion
        'covariance': [[1.0, 0.0], [0.0, 1.0]],
        'amplitude': 1.0
    }
    
    # Force tensor conversion to raise an error that triggers line 388
    orig_tensor = torch.tensor
    
    def mock_tensor(*args, **kwargs):
        if isinstance(args[0], list) and not isinstance(args[0][0], (int, float)):
            raise ValueError("Cannot convert strings to tensor")
        return orig_tensor(*args, **kwargs)
    
    torch.tensor = mock_tensor
    
    try:
        # This should create a fallback splat
        splat = Splat.from_dict(data)
        
        # Verify we got a fallback splat
        assert splat is not None
        assert splat.id == 'test_splat'
        assert splat.position.shape[0] > 0  # Should create some position
        assert splat.covariance.shape[0] == splat.position.shape[0]  # Matching dims
    finally:
        torch.tensor = orig_tensor


def test_splat_dict_serialization_round_trip():
    """Test a complete round-trip serialization to dict and back."""
    embedding_dim = 5
    position = torch.randn(embedding_dim)
    covariance = generate_random_covariance(embedding_dim)
    amplitude = 0.7
    splat_id = "test_round_trip"
    
    original = Splat(position, covariance, amplitude, splat_id=splat_id)
    
    # Add some activation history
    for i in range(5):
        original.update_activation(0.1 * i)
    
    # Convert to dict
    data = original.to_dict()
    
    # Convert back to splat
    reconstructed = Splat.from_dict(data)
    
    # Verify key properties match
    assert reconstructed.id == original.id
    assert torch.allclose(reconstructed.position, original.position)
    assert torch.allclose(reconstructed.covariance, original.covariance)
    assert reconstructed.amplitude == original.amplitude
    assert reconstructed.get_average_activation() == 0.0  # Activation history doesn't round-trip


def test_repr(sample_splat):
    """Test string representation of splat."""
    # Update the activation to make it predictable
    sample_splat.update_activation(0.5)
    repr_str = repr(sample_splat)
    
    assert isinstance(repr_str, str)
    assert sample_splat.id in repr_str
    assert str(round(sample_splat.amplitude, 3)).replace('.000', '') in repr_str.replace('.000', '')
    
    # Check for activation value in representation
    # Account for different rounding display formats
    activation_str = str(round(sample_splat.get_average_activation(), 3))
    # Remove trailing zeros if present
    if activation_str.endswith('.0'):
        activation_str = activation_str[:-2]
    elif activation_str.endswith('.000'):
        activation_str = activation_str[:-4]
    
    assert activation_str in repr_str.replace('.000', '').replace('.0', '')
    
def test_compute_attention_general_exception():
    """Test the exception handler in compute_attention (lines 173-176)."""
    embedding_dim = 4
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance)
    
    # Create tokens that will trigger calculation but not special cases
    token_a = torch.ones(embedding_dim)
    token_b = torch.ones(embedding_dim) * 2.0
    
    # Instead of trying to patch functions, let's corrupt the internal state
    # Set precision matrix to NaN to force computation to fail
    splat.precision = torch.tensor(float('nan')) * splat.precision
    
    # This should now trigger the exception handler in compute_attention
    attention = splat.compute_attention(token_a, token_b)
    
    # Verify the error handling behavior
    assert attention == 0.0
    assert splat.activation_history[-1] == 0.0
            
def test_clone_copies_activation_history():
    """Test that clone properly copies activation history."""
    embedding_dim = 4
    position = torch.zeros(embedding_dim)
    covariance = torch.eye(embedding_dim)
    splat = Splat(position, covariance)
    
    # Add some activation values
    activation_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    for value in activation_values:
        splat.update_activation(value)
    
    # Verify original has the correct activation history
    assert len(splat.activation_history) == len(activation_values)
    assert splat.activation_history.get_values() == activation_values
    
    # Clone the splat
    clone = splat.clone()
    
    # Verify the clone has the same activation history
    # This will only pass if line 388 is executed
    assert len(clone.activation_history) == len(activation_values)
    assert clone.activation_history.get_values() == activation_values
    
    # Change activation in original and verify it doesn't affect clone
    splat.update_activation(0.9)
    assert len(splat.activation_history) == len(activation_values) + 1
    assert len(clone.activation_history) == len(activation_values)
