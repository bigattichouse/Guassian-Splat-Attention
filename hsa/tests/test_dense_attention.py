"""
Tests for the DenseAttentionComputer class in the HSA implementation.
"""

import pytest
import numpy as np

from hsa.splat import Splat
from hsa.hierarchy import Hierarchy
from hsa.registry import SplatRegistry
from hsa.attention_interface import AttentionConfig, AttentionResult
from hsa.dense_attention import DenseAttentionComputer


class TestDenseAttentionComputer:
    """Tests for the DenseAttentionComputer class."""
    
    @pytest.fixture
    def hierarchy(self) -> Hierarchy:
        """Create a simple hierarchy for testing."""
        return Hierarchy(
            levels=["token", "phrase"],
            init_splats_per_level=[5, 2],
            level_weights=[0.7, 0.3]
        )
    
    @pytest.fixture
    def registry(self, hierarchy) -> SplatRegistry:
        """Create a registry with some splats for testing."""
        registry = SplatRegistry(hierarchy=hierarchy, embedding_dim=2)
        
        # Create token-level splats
        token_splat1 = Splat(
            dim=2,
            position=np.array([0.0, 0.0]),
            level="token",
            id="token_1"
        )
        
        token_splat2 = Splat(
            dim=2,
            position=np.array([1.0, 0.0]),
            level="token",
            id="token_2"
        )
        
        # Create phrase-level splat
        phrase_splat = Splat(
            dim=2,
            position=np.array([0.5, 0.5]),
            covariance=np.array([[2.0, 0.0], [0.0, 2.0]]),  # Broader coverage
            level="phrase",
            id="phrase_1"
        )
        
        # Register splats
        registry.register(token_splat1)
        registry.register(token_splat2)
        registry.register(phrase_splat)
        
        return registry
    
    @pytest.fixture
    def tokens(self) -> np.ndarray:
        """Create token embeddings for testing."""
        return np.array([
            [0.0, 0.0],  # Close to token_splat1
            [1.0, 0.0],  # Close to token_splat2
            [0.5, 0.5]   # Close to phrase_splat
        ])
    
    def test_init(self):
        """Test initialization."""
        computer = DenseAttentionComputer()
        assert isinstance(computer.config, AttentionConfig)
        
        custom_config = AttentionConfig(topk=10, causal=True)
        computer = DenseAttentionComputer(config=custom_config)
        assert computer.config.topk == 10
        assert computer.config.causal is True
    
    def test_compute_splat_attention_map(self, tokens, registry):
        """Test computing attention map for a single splat."""
        computer = DenseAttentionComputer()
        
        # Get token-level splat
        token_splat = registry.get_splat("token_1")
        
        # Compute attention map
        attention_map = computer.compute_splat_attention_map(tokens, token_splat)
        
        # Check dimensions
        assert attention_map.shape == (3, 3)
        
        # Should have high attention for tokens close to the splat
        assert attention_map[0, 0] > 0.5  # Diagonal should be high (self-attention)
        assert attention_map[1, 1] < attention_map[0, 0]  # token_2 is further from token_1
        assert attention_map[2, 2] < attention_map[0, 0]  # phrase token is further from token_1
        
        # Test with max_tokens limit
        limited_map = computer.compute_splat_attention_map(tokens, token_splat, max_tokens=2)
        assert limited_map.shape == (3, 3)  # Still full size output
    
    def test_apply_topk_sparsity(self):
        """Test applying top-k sparsity to attention matrix."""
        computer = DenseAttentionComputer()
        
        # Create attention matrix
        attention = np.array([
            [0.9, 0.5, 0.1],
            [0.2, 0.8, 0.3],
            [0.4, 0.2, 0.7]
        ])
        
        # Apply top-2 sparsity
        sparse = computer.apply_topk_sparsity(attention, k=2)
        
        # Check each row has exactly 2 non-zero entries
        for i in range(3):
            non_zeros = np.count_nonzero(sparse[i])
            assert non_zeros == 2
        
        # Highest values should be kept
        assert sparse[0, 0] == 0.9
        assert sparse[0, 1] == 0.5
        assert sparse[0, 2] == 0.0
        
        # Apply threshold sparsity
        sparse = computer.apply_topk_sparsity(attention, threshold=0.4)
        
        # Check values below threshold are zeroed
        assert sparse[0, 0] == 0.9
        assert sparse[0, 1] == 0.5
        assert sparse[0, 2] == 0.0
        
        assert sparse[1, 0] == 0.0
        assert sparse[1, 1] == 0.8
        assert sparse[1, 2] == 0.0
        
        assert sparse[2, 0] == 0.4
        assert sparse[2, 1] == 0.0
        assert sparse[2, 2] == 0.7
        
        # Test with custom threshold
        # Create an attention matrix with some values below 0.05
        attention_with_small_values = np.array([
            [0.9, 0.5, 0.04],
            [0.02, 0.8, 0.3],
            [0.4, 0.03, 0.7]
        ])
        
        # Configure computer with a higher default threshold
        config = AttentionConfig(default_threshold=0.05)
        computer_with_custom_threshold = DenseAttentionComputer(config)
        
        # Apply default threshold
        sparse = computer_with_custom_threshold.apply_topk_sparsity(attention_with_small_values)
        
        # Check values below threshold are zeroed
        assert np.count_nonzero(sparse) < np.count_nonzero(attention_with_small_values)
        assert sparse[0, 2] == 0.0  # 0.04 < 0.05
        assert sparse[1, 0] == 0.0  # 0.02 < 0.05
        assert sparse[2, 1] == 0.0  # 0.03 < 0.05

    def test_compute_attention(self, registry, tokens):
        """Test computing full attention matrix."""
        computer = DenseAttentionComputer()
        
        # Compute attention
        attention = computer.compute_attention(tokens, registry)
        
        # Check dimensions
        assert attention.shape == (3, 3)
        
        # Diagonal should have strong attention
        for i in range(3):
            assert attention[i, i] > 0.0
        
        # Rows should sum to 1 (default normalization)
        for i in range(3):
            assert pytest.approx(np.sum(attention[i])) == 1.0
        
        # Test with custom config - disabling row normalization
        config = AttentionConfig(
            normalize_rows=False,
            causal=True,
            topk=2
        )
        computer = DenseAttentionComputer(config=config)
        
        # Confirm the config is correctly set
        assert computer.config.normalize_rows is False
        
        # Compute attention with custom config
        attention = computer.compute_attention(tokens, registry)
        
        # Check causal mask (upper triangle should be zero)
        for i in range(3):
            for j in range(i + 1, 3):
                assert attention[i, j] == 0.0
        
        # Rows should not necessarily sum to 1 when normalize_rows=False
        # Look at non-zero rows (some rows might be all zeros due to causal + topk)
        non_normalized_found = False
        for i in range(3):
            row_sum = np.sum(attention[i])
            if row_sum > 0:  # Only check non-zero rows
                # If normalize_rows=False, at least some rows should NOT sum to 1
                if abs(row_sum - 1.0) > 1e-10:
                    non_normalized_found = True
                    break
        
        # At least one row should not be normalized
        assert non_normalized_found, "No unnormalized rows found even with normalize_rows=False"
        
    def test_compute_attention_with_details(self, registry, tokens):
        """Test computing attention with detailed contributions."""
        computer = DenseAttentionComputer()
        
        # Compute attention with details
        result = computer.compute_attention_with_details(tokens, registry)
        
        # Should be an AttentionResult object
        assert isinstance(result, AttentionResult)
        
        # Check basic properties
        assert result.attention_matrix.shape == (3, 3)
        assert "token" in result.level_contributions
        assert "phrase" in result.level_contributions
        
        # Should have splat contributions
        assert len(result.splat_contributions) == 3  # One per splat
        assert "token_1" in result.splat_contributions
        assert "token_2" in result.splat_contributions
        assert "phrase_1" in result.splat_contributions
        
        # Should have active splats
        assert len(result.active_splats) > 0
        
        # Test with custom config
        config = AttentionConfig(causal=True)
        computer = DenseAttentionComputer(config=config)
        
        result = computer.compute_attention_with_details(tokens, registry)
        
        # Check causal mask (upper triangle should be zero)
        for i in range(3):
            for j in range(i + 1, 3):
                assert result.attention_matrix[i, j] == 0.0
                
        # Level contributions should also respect causality
        for level, contribution in result.level_contributions.items():
            for i in range(3):
                for j in range(i + 1, 3):
                    assert contribution[i, j] == 0.0
