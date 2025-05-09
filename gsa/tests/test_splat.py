"""
Tests for the Splat implementation.
"""

import pytest
import torch
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


def test_compute_attention(sample_splat, embedding_dim):
    """Test attention computation between two tokens."""
    token_a = torch.randn(embedding_dim)
    token_b = torch.randn(embedding_dim)
    
    attention = sample_splat.compute_attention(token_a, token_b)
    
    assert isinstance(attention, torch.Tensor)
    assert attention.shape == torch.Size([])  # Scalar output
    assert attention >= 0.0  # Attention should be non-negative


def test_compute_attention_batch(sample_splat, embedding_dim):
    """Test batch attention computation."""
    batch_size = 5
    tokens = torch.randn(batch_size, embedding_dim)
    
    attention_matrix = sample_splat.compute_attention_batch(tokens)
    
    assert isinstance(attention_matrix, torch.Tensor)
    assert attention_matrix.shape == (batch_size, batch_size)
    assert (attention_matrix >= 0.0).all()  # All attention values should be non-negative


def test_update_activation(sample_splat):
    """Test updating activation history."""
    sample_splat.update_activation(0.5)
    sample_splat.update_activation(0.7)
    
    assert len(sample_splat.activation_history) == 2
    assert sample_splat.get_average_activation() == pytest.approx(0.6)


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
