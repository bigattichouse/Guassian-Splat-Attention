"""
Tests for the numeric utilities module.
"""

import pytest
import torch
import numpy as np
from gsa.numeric_utils import (
    ensure_positive_definite,
    stable_matrix_inverse,
    compute_gaussian,
    bounded_amplitude,
    stable_softmax,
    generate_random_covariance,
    compute_matrix_decomposition,
    compute_condition_number,
    is_positive_definite,
    matrix_to_scalar_variance,
    compute_mahalanobis_distance,
    diag_loading,
    normalize_matrix
)


@pytest.fixture
def sample_matrix():
    """Create a sample matrix for testing."""
    return torch.tensor([
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 0.5],
        [0.0, 0.5, 2.0]
    ])


@pytest.fixture
def non_positive_definite_matrix():
    """Create a non-positive definite matrix for testing."""
    return torch.tensor([
        [1.0, 2.0, 3.0],
        [2.0, 1.0, 4.0],
        [3.0, 4.0, -10.0]
    ])


def test_ensure_positive_definite(sample_matrix, non_positive_definite_matrix):
    """Test making a matrix positive definite."""
    # Already positive definite matrix should not change much
    result = ensure_positive_definite(sample_matrix)
    assert torch.allclose(result, sample_matrix, rtol=1e-5)
    
    # Non-positive definite matrix should be corrected
    result = ensure_positive_definite(non_positive_definite_matrix)
    eigenvalues = torch.linalg.eigvalsh(result)
    assert (eigenvalues > 0).all()


def test_stable_matrix_inverse(sample_matrix):
    """Test stable matrix inversion."""
    inverse = stable_matrix_inverse(sample_matrix)
    
    # Check that the inverse is indeed the inverse
    identity = torch.matmul(sample_matrix, inverse)
    assert torch.allclose(identity, torch.eye(3), rtol=1e-5, atol=1e-7)


def test_compute_gaussian():
    """Test Gaussian computation."""
    # Simple case: identity covariance, zero mean
    dim = 3
    x = torch.ones(dim)
    mean = torch.zeros(dim)
    precision = torch.eye(dim)
    
    result = compute_gaussian(x, mean, precision)
    
    # For the given parameters, should be exp(-dim/2)
    expected = torch.exp(torch.tensor(-dim/2.0))
    assert torch.isclose(result, expected)
    
    # Test with different mean and precision
    x = torch.tensor([1.0, 2.0, 3.0])
    mean = torch.tensor([0.0, 1.0, 2.0])
    covariance = torch.tensor([
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.5]
    ])
    precision = stable_matrix_inverse(covariance)
    
    result = compute_gaussian(x, mean, precision)
    assert result > 0.0  # Should be positive


def test_bounded_amplitude():
    """Test bounding amplitude values."""
    # Normal values should not change
    assert bounded_amplitude(0.5) == 0.5
    
    # Values below min should be clamped
    assert bounded_amplitude(-1.0) > 0.0
    
    # Values above max should be clamped
    assert bounded_amplitude(100.0) <= 10.0


def test_stable_softmax():
    """Test stable softmax implementation."""
    # Simple case
    x = torch.tensor([1.0, 2.0, 3.0])
    result = stable_softmax(x)
    
    # Should sum to 1
    assert torch.isclose(result.sum(), torch.tensor(1.0))
    
    # Largest value should have highest probability
    assert torch.argmax(result) == 2
    
    # Test with large values which would cause numerical issues with naive softmax
    x = torch.tensor([1000.0, 0.0, 0.0])
    result = stable_softmax(x)
    
    # Should still sum to 1 without overflow
    assert torch.isclose(result.sum(), torch.tensor(1.0))
    assert result[0] > 0.99  # First value should dominate


def test_generate_random_covariance():
    """Test random covariance matrix generation."""
    dim = 5
    cov = generate_random_covariance(dim)
    
    # Check shape
    assert cov.shape == (dim, dim)
    
    # Should be symmetric
    assert torch.allclose(cov, cov.t())
    
    # Should be positive definite
    eigenvalues = torch.linalg.eigvalsh(cov)
    assert (eigenvalues > 0).all()


def test_compute_matrix_decomposition(sample_matrix):
    """Test matrix decomposition."""
    eigenvalues, eigenvectors = compute_matrix_decomposition(sample_matrix)
    
    # Check that eigenvectors are orthogonal
    identity = torch.matmul(eigenvectors.t(), eigenvectors)
    assert torch.allclose(identity, torch.eye(3), rtol=1e-5, atol=1e-7)
    
    # Check that eigenvalues are correct
    for i in range(3):
        v = eigenvectors[:, i]
        lambda_v = eigenvalues[i]
        Av = torch.matmul(sample_matrix, v)
        lambda_v_times_v = lambda_v * v
        assert torch.allclose(Av, lambda_v_times_v, rtol=1e-5)


def test_compute_condition_number(sample_matrix, non_positive_definite_matrix):
    """Test condition number computation."""
    # Test with well-conditioned matrix
    cond = compute_condition_number(sample_matrix)
    assert cond > 1.0  # Condition number is always >= 1
    
    # Test with ill-conditioned matrix
    fixed_matrix = ensure_positive_definite(non_positive_definite_matrix)
    cond = compute_condition_number(fixed_matrix)
    assert cond > 1.0


def test_is_positive_definite(sample_matrix, non_positive_definite_matrix):
    """Test checking if a matrix is positive definite."""
    assert is_positive_definite(sample_matrix)
    assert not is_positive_definite(non_positive_definite_matrix)


def test_matrix_to_scalar_variance(sample_matrix):
    """Test converting covariance matrix to scalar variance."""
    variance = matrix_to_scalar_variance(sample_matrix)
    eigenvalues = torch.linalg.eigvalsh(sample_matrix)
    expected = eigenvalues.mean().item()
    assert variance == expected


def test_compute_mahalanobis_distance():
    """Test Mahalanobis distance computation."""
    x = torch.tensor([3.0, 2.0, 1.0])
    mean = torch.tensor([0.0, 0.0, 0.0])
    covariance = torch.eye(3)
    
    distance = compute_mahalanobis_distance(x, mean, covariance)
    
    # For identity covariance, Mahalanobis distance equals Euclidean distance
    expected = torch.sqrt(torch.sum((x - mean) ** 2)).item()
    assert distance == pytest.approx(expected)


def test_diag_loading(sample_matrix):
    """Test diagonal loading."""
    loaded = diag_loading(sample_matrix, alpha=0.1)
    
    # Diagonal elements should be increased
    for i in range(3):
        assert loaded[i, i] > sample_matrix[i, i]
    
    # Off-diagonal elements should remain the same
    for i in range(3):
        for j in range(3):
            if i != j:
                assert loaded[i, j] == sample_matrix[i, j]


def test_normalize_matrix_frobenius(sample_matrix):
    """Test matrix normalization with Frobenius norm."""
    normalized = normalize_matrix(sample_matrix, norm_type='frobenius')
    
    # Compute Frobenius norm of the result
    norm = torch.norm(normalized, p='fro')
    assert torch.isclose(norm, torch.tensor(1.0))


def test_normalize_matrix_spectral(sample_matrix):
    """Test matrix normalization with spectral norm."""
    normalized = normalize_matrix(sample_matrix, norm_type='spectral')
    
    # Compute spectral norm (largest singular value)
    _, s, _ = torch.linalg.svd(normalized)
    assert torch.isclose(s[0], torch.tensor(1.0))


def test_normalize_matrix_trace(sample_matrix):
    """Test matrix normalization with trace."""
    normalized = normalize_matrix(sample_matrix, norm_type='trace')
    
    # Compute trace
    trace = torch.trace(normalized)
    assert torch.isclose(trace, torch.tensor(1.0))


def test_normalize_matrix_invalid_type(sample_matrix):
    """Test that invalid normalization type raises an error."""
    with pytest.raises(ValueError):
        normalize_matrix(sample_matrix, norm_type='invalid')
