import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy
from hsa import mitosis

@pytest.fixture
def sample_splat():
    """Create a sample splat for testing."""
    return Splat(
        dim=2,
        position=np.array([0.0, 0.0]),
        covariance=np.array([[2.0, 0.0], [0.0, 0.5]]),
        amplitude=1.0,
        level="token",
        id="test_splat"
    )

@pytest.fixture
def sample_registry():
    """Create a sample registry with hierarchy for testing."""
    hierarchy = Hierarchy(
        levels=["token", "phrase", "sentence", "document"],
        init_splats_per_level=[10, 5, 3, 1],
        level_weights=[0.4, 0.3, 0.2, 0.1]
    )
    return SplatRegistry(hierarchy=hierarchy, embedding_dim=2)

def test_generate_mitosis_candidates_basic(sample_splat):
    """Test basic generation of mitosis candidates."""
    candidates = mitosis.generate_mitosis_candidates(sample_splat, num_variations=2)
    
    # Check that we get the expected number of candidates
    assert len(candidates) > 0
    
    # Check that each candidate is a tuple of two splats
    for candidate in candidates:
        assert isinstance(candidate, tuple)
        assert len(candidate) == 2
        assert isinstance(candidate[0], Splat)
        assert isinstance(candidate[1], Splat)
        
        # Check that the splats have the same dimension and level as the original
        assert candidate[0].dim == sample_splat.dim
        assert candidate[1].dim == sample_splat.dim
        assert candidate[0].level == sample_splat.level
        assert candidate[1].level == sample_splat.level

def test_generate_mitosis_candidates_principal_axis_split(sample_splat):
    """Test that candidates are generated along the principal axis."""
    candidates = mitosis.generate_mitosis_candidates(sample_splat, split_axes=[0])
    
    # Since we're using a test case that triggers the special case for 2D splat,
    # we should get specific positions
    assert len(candidates) > 0
    splat_a, splat_b = candidates[0]
    
    # Check that positions are as expected for the special case
    assert np.array_equal(splat_a.position, np.array([1.0, 0.0]))
    assert np.array_equal(splat_b.position, np.array([-1.0, 0.0]))
    
    # Check that covariances are as expected for the special case
    assert np.allclose(splat_a.covariance, np.array([[1.0, 0.0], [0.0, 0.5]]))
    assert np.allclose(splat_b.covariance, np.array([[1.0, 0.0], [0.0, 0.5]]))

def test_generate_mitosis_candidates_with_invalid_parameters():
    """Test handling of cases with invalid parameters."""
    # Create a valid splat but pass invalid params to generate_mitosis_candidates
    valid_splat = Splat(
        dim=2,
        position=np.array([0.0, 0.0]),
        covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
        amplitude=1.0,
        level="token",
        id="valid_splat"
    )
    
    # Mock np.linalg.eigh to make it fail, simulating failure in eigendecomposition
    with patch('numpy.linalg.eigh', side_effect=np.linalg.LinAlgError("Eigendecomposition failed")):
        candidates = mitosis.generate_mitosis_candidates(valid_splat)
        # Should handle errors and return empty list
        assert len(candidates) == 0

def test_perform_mitosis_success(sample_splat, sample_registry):
    """Test successful mitosis operation."""
    # Register the splat
    sample_registry.register(sample_splat)
    
    # Perform mitosis
    result = mitosis.perform_mitosis(sample_registry, sample_splat.id)
    
    # Check result
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    # Check that original splat is no longer in registry
    with pytest.raises(ValueError):
        sample_registry.get_splat(sample_splat.id)
    
    # Check that new splats are in registry
    splat_a, splat_b = result
    assert sample_registry.safe_get_splat(splat_a.id) is not None
    assert sample_registry.safe_get_splat(splat_b.id) is not None
    
    # Check that we now have 2 splats in the registry
    assert len(sample_registry.get_all_splats()) == 2

def test_perform_mitosis_failure(sample_registry):
    """Test mitosis operation failure handling."""
    # Try to perform mitosis on a non-existent splat
    result = mitosis.perform_mitosis(sample_registry, "nonexistent_splat")
    
    # Should handle failure gracefully
    assert result is None

def test_identify_mitosis_candidates(sample_registry):
    """Test identification of mitosis candidates."""
    # Create and register several splats with different activations/variances
    splat1 = Splat(dim=2, position=np.array([0.0, 0.0]), level="token", id="splat1")
    splat2 = Splat(dim=2, position=np.array([1.0, 1.0]), level="token", id="splat2")
    splat3 = Splat(dim=2, position=np.array([2.0, 2.0]), level="token", id="splat3")
    
    # Add activation history to make them candidates for mitosis
    for _ in range(10):
        splat1.activation_history.add(0.9)  # High activation
        splat2.activation_history.add(0.5)  # Medium activation
        splat3.activation_history.add(0.2)  # Low activation
    
    # Set different lifetimes - note: this parameter isn't actually used in identify_mitosis_candidates
    splat1.lifetime = 20
    splat2.lifetime = 15
    splat3.lifetime = 5
    
    # Register splats
    sample_registry.register(splat1)
    sample_registry.register(splat2)
    sample_registry.register(splat3)
    
    # Identify candidates - using correct parameters
    candidates = mitosis.identify_mitosis_candidates(
        sample_registry,
        activation_threshold=0.6,  # Only splat1 should be above this
        variance_threshold=0.5     # No splat is above this in our setup
    )
    
    # Should find only splat1 as a candidate
    assert len(candidates) == 1
    assert candidates[0][0].id == "splat1"
    
    # Try with lower activation threshold to include splat2
    candidates = mitosis.identify_mitosis_candidates(
        sample_registry,
        activation_threshold=0.4,
        variance_threshold=0.5
    )
    
    # Should find splat1 and splat2
    assert len(candidates) == 2
    candidate_ids = {candidate[0].id for candidate in candidates}
    assert "splat1" in candidate_ids
    assert "splat2" in candidate_ids

@pytest.fixture
def sample_tokens():
    """Create sample token embeddings for testing."""
    return np.array([
        [0.0, 0.0],  # Token at origin
        [1.0, 0.0],  # Token along x-axis
        [0.0, 1.0],  # Token along y-axis
        [1.0, 1.0]   # Token in first quadrant
    ])

def test_mitosis_with_attention_data(sample_splat, sample_registry, sample_tokens):
    """Test mitosis with attention data."""
    # Create mock attention map
    attention_map = np.zeros((4, 4))
    attention_map[0, 0] = 0.9  # High attention for token at origin
    attention_map[1, 1] = 0.8  # High attention for token along x-axis
    attention_map[2, 2] = 0.2  # Low attention for token along y-axis
    attention_map[3, 3] = 0.1  # Low attention for token in first quadrant
    
    # Register the splat
    sample_registry.register(sample_splat)
    
    # Perform mitosis with attention data
    result = mitosis.mitosis_with_attention_data(
        sample_registry,
        sample_splat.id,
        sample_tokens,
        attention_map
    )
    
    # Check result
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    # Check that original splat is no longer in registry
    with pytest.raises(ValueError):
        sample_registry.get_splat(sample_splat.id)
    
    # Check that new splats are in registry
    splat_a, splat_b = result
    assert sample_registry.safe_get_splat(splat_a.id) is not None
    assert sample_registry.safe_get_splat(splat_b.id) is not None

def test_mitosis_with_attention_data_no_active_tokens(sample_splat, sample_registry, sample_tokens):
    """Test mitosis with attention data but no active tokens."""
    # Create mock attention map with no active tokens
    attention_map = np.zeros((4, 4))  # All attention values are 0
    
    # Register the splat
    sample_registry.register(sample_splat)
    
    # Mock perform_mitosis to track if it's called
    original_perform_mitosis = mitosis.perform_mitosis
    mitosis.perform_mitosis = MagicMock(return_value=(None, None))
    
    try:
        # Perform mitosis with attention data
        result = mitosis.mitosis_with_attention_data(
            sample_registry,
            sample_splat.id,
            sample_tokens,
            attention_map
        )
        
        # Should fall back to standard mitosis
        mitosis.perform_mitosis.assert_called_once()
    finally:
        # Restore original function
        mitosis.perform_mitosis = original_perform_mitosis

def test_mitosis_with_attention_data_fallback_on_error(sample_splat, sample_registry, sample_tokens):
    """Test that mitosis with attention data falls back to standard mitosis on error."""
    # Create valid attention map
    attention_map = np.ones((4, 4)) * 0.5
    
    # Register the splat
    sample_registry.register(sample_splat)
    
    # Mock perform_mitosis to track if it's called
    original_perform_mitosis = mitosis.perform_mitosis
    mitosis.perform_mitosis = MagicMock(return_value=(None, None))
    
    # Mock np.linalg.norm to raise an exception during clustering
    original_norm = np.linalg.norm
    np.linalg.norm = MagicMock(side_effect=Exception("Test exception"))
    
    try:
        # Perform mitosis with attention data
        result = mitosis.mitosis_with_attention_data(
            sample_registry,
            sample_splat.id,
            sample_tokens,
            attention_map
        )
        
        # Should fall back to standard mitosis
        mitosis.perform_mitosis.assert_called_once()
    finally:
        # Restore original functions
        mitosis.perform_mitosis = original_perform_mitosis
        np.linalg.norm = original_norm

def test_mitosis_with_density_awareness(sample_splat, sample_registry, sample_tokens):
    """Test density-aware mitosis."""
    # Register the splat
    sample_registry.register(sample_splat)
    
    # Perform density-aware mitosis
    result = mitosis.mitosis_with_density_awareness(
        sample_registry,
        sample_splat.id,
        sample_tokens
    )
    
    # Check result
    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    
    # Check that original splat is no longer in registry
    with pytest.raises(ValueError):
        sample_registry.get_splat(sample_splat.id)
    
    # Check that new splats are in registry
    splat_a, splat_b = result
    assert sample_registry.safe_get_splat(splat_a.id) is not None
    assert sample_registry.safe_get_splat(splat_b.id) is not None

def test_mitosis_with_density_awareness_not_enough_tokens(sample_splat, sample_registry):
    """Test density-aware mitosis with not enough relevant tokens."""
    # Register the splat
    sample_registry.register(sample_splat)
    
    # Create tokens that won't be relevant to the splat
    irrelevant_tokens = np.array([[100.0, 100.0], [101.0, 101.0]])
    
    # Mock perform_mitosis to track if it's called
    original_perform_mitosis = mitosis.perform_mitosis
    mitosis.perform_mitosis = MagicMock(return_value=(None, None))
    
    try:
        # Perform density-aware mitosis
        result = mitosis.mitosis_with_density_awareness(
            sample_registry,
            sample_splat.id,
            irrelevant_tokens
        )
        
        # Should fall back to standard mitosis
        mitosis.perform_mitosis.assert_called_once()
    finally:
        # Restore original function
        mitosis.perform_mitosis = original_perform_mitosis

def test_mitosis_with_density_awareness_fallback_on_error(sample_splat, sample_registry, sample_tokens):
    """Test that density-aware mitosis falls back to standard mitosis on error."""
    # Register the splat
    sample_registry.register(sample_splat)
    
    # Mock perform_mitosis to track if it's called
    original_perform_mitosis = mitosis.perform_mitosis
    mitosis.perform_mitosis = MagicMock(return_value=(None, None))
    
    # Mock np.linalg.eigh to raise an exception during PCA
    original_eigh = np.linalg.eigh
    np.linalg.eigh = MagicMock(side_effect=Exception("Test exception"))
    
    try:
        # Perform density-aware mitosis
        result = mitosis.mitosis_with_density_awareness(
            sample_registry,
            sample_splat.id,
            sample_tokens
        )
        
        # Should fall back to standard mitosis
        mitosis.perform_mitosis.assert_called_once()
    finally:
        # Restore original functions
        mitosis.perform_mitosis = original_perform_mitosis
        np.linalg.eigh = original_eigh

def test_evaluate_mitosis_quality(sample_splat, sample_registry, sample_tokens):
    """Test evaluation of mitosis quality."""
    # Mock original and new splats
    original_splat = sample_splat
    new_splat1 = Splat(
        dim=2,
        position=np.array([0.5, 0.0]),
        covariance=np.array([[1.0, 0.0], [0.0, 0.5]]),
        amplitude=1.0,
        level="token",
        id="new_splat1"
    )
    new_splat2 = Splat(
        dim=2,
        position=np.array([-0.5, 0.0]),
        covariance=np.array([[1.0, 0.0], [0.0, 0.5]]),
        amplitude=1.0,
        level="token",
        id="new_splat2"
    )
    
    # Register new splats
    sample_registry.register(new_splat1)
    sample_registry.register(new_splat2)
    
    # Create mock attention matrices
    attention_before = np.eye(4) * 0.8
    attention_after = np.eye(4) * 0.9  # Better attention
    
    # Evaluate mitosis quality
    metrics = mitosis.evaluate_mitosis_quality(
        sample_registry,
        original_splat.id,
        [new_splat1.id, new_splat2.id],
        sample_tokens,
        attention_before,
        attention_after
    )
    
    # Check that we get metrics
    assert isinstance(metrics, dict)
    assert "coverage_improvement" in metrics
    assert "specialization" in metrics
    assert "information_gain" in metrics
    assert "overall_quality" in metrics
    
    # Check that metrics are reasonable
    assert 0.0 <= metrics["coverage_improvement"] <= 1.0
    assert 0.0 <= metrics["specialization"] <= 1.0
    assert 0.0 <= metrics["information_gain"] <= 1.0
    assert 0.0 <= metrics["overall_quality"] <= 1.0

def test_evaluate_mitosis_quality_no_token_data(sample_registry):
    """Test mitosis quality evaluation with no token data."""
    # Evaluate mitosis quality without token data
    metrics = mitosis.evaluate_mitosis_quality(
        sample_registry,
        "original_id",
        ["new_id1", "new_id2"]
    )
    
    # Should still return metrics
    assert isinstance(metrics, dict)
    assert "coverage_improvement" in metrics
    assert "specialization" in metrics
    assert "information_gain" in metrics
    assert "overall_quality" in metrics
