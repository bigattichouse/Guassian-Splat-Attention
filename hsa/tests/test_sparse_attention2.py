import pytest
import numpy as np

from hsa.sparse_attention import SparseAttentionComputer
from hsa.attention_interface import AttentionConfig
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy
from hsa.sparse_attention_utils import compute_token_relevance, get_sparsity_ratio


class TestSparseAttentionComputer:
    @pytest.fixture
    def registry(self):
        """Create a simple registry with a few splats for testing."""
        hierarchy = Hierarchy(
            levels=["token", "phrase", "sentence"],
            init_splats_per_level=[2, 1, 1],
            level_weights=[0.5, 0.3, 0.2]
        )
        
        registry = SplatRegistry(hierarchy=hierarchy, embedding_dim=2)
        
        # Token level splats
        token_splat1 = Splat(
            dim=2,
            position=np.array([0.0, 0.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
            amplitude=1.0,
            level="token",
            id="token_1"
        )
        
        token_splat2 = Splat(
            dim=2,
            position=np.array([2.0, 2.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
            amplitude=1.0,
            level="token",
            id="token_2"
        )
        
        # Phrase level splat
        phrase_splat = Splat(
            dim=2,
            position=np.array([1.0, 1.0]),
            covariance=np.array([[2.0, 0.0], [0.0, 2.0]]),
            amplitude=1.0,
            level="phrase",
            id="phrase_1"
        )
        
        # Sentence level splat
        sentence_splat = Splat(
            dim=2,
            position=np.array([0.0, 0.0]),
            covariance=np.array([[3.0, 0.0], [0.0, 3.0]]),
            amplitude=1.0,
            level="sentence",
            id="sentence_1"
        )
        
        # Add parent-child relationships
        token_splat1.parent = phrase_splat
        phrase_splat.children.add(token_splat1)
        
        token_splat2.parent = phrase_splat
        phrase_splat.children.add(token_splat2)
        
        phrase_splat.parent = sentence_splat
        sentence_splat.children.add(phrase_splat)
        
        # Register all splats
        registry.register(token_splat1)
        registry.register(token_splat2)
        registry.register(phrase_splat)
        registry.register(sentence_splat)
        
        return registry
    
    @pytest.fixture
    def tokens(self):
        """Create simple test tokens."""
        return np.array([
            [0.0, 0.0],  # Close to token_1 and sentence_1
            [1.0, 1.0],  # Close to phrase_1
            [2.0, 2.0]   # Close to token_2
        ])
    
    @pytest.fixture
    def sparse_attention_computer(self):
        """Create a sparse attention computer with default config."""
        config = AttentionConfig(
            topk=None,
            threshold=0.01,
            normalize_levels=True,
            normalize_rows=True,
            causal=False
        )
        return SparseAttentionComputer(
            config=config,
            sparsity_threshold=0.01,
            use_spatial_index=False
        )
    
    def test_compute_splat_attention_map_small_sequence(self, sparse_attention_computer, registry, tokens):
        """Test compute_splat_attention_map with a small sequence."""
        # Get a splat from the registry
        splat = registry.get_splat("token_1")
        
        # Compute attention map
        attention_map = sparse_attention_computer.compute_splat_attention_map(tokens, splat)
        
        # Check shape
        assert attention_map.shape == (3, 3)
        
        # Check that attention is highest near splat center
        # For tokens at [0,0], [1,1], and [2,2], the attention at [0,0] should be >= attention at [1,1]
        # since splat token_1 is at [0,0]
        assert attention_map[0, 0] >= attention_map[1, 1]
        assert attention_map[0, 0] >= attention_map[2, 2]
    
    def test_compute_attention(self, sparse_attention_computer, registry, tokens):
        """Test compute_attention with a simple registry and tokens."""
        # Compute attention matrix
        attention_matrix = sparse_attention_computer.compute_attention(tokens, registry)
        
        # Check shape
        assert attention_matrix.shape == (3, 3)
        
        # Check that some attention values are non-zero
        assert np.sum(attention_matrix > 0) > 0
        
        # Since normalize_rows=True, each row should sum to 1
        row_sums = np.sum(attention_matrix, axis=1)
        np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-6)
    
    def test_compute_attention_with_details(self, sparse_attention_computer, registry, tokens):
        """Test compute_attention_with_details."""
        # Compute attention with details
        attention_result = sparse_attention_computer.compute_attention_with_details(tokens, registry)
        
        # Check that attention matrix exists and has correct shape
        assert attention_result.attention_matrix.shape == (3, 3)
        
        # Check that level contributions are included
        assert "token" in attention_result.level_contributions
        assert "phrase" in attention_result.level_contributions
        assert "sentence" in attention_result.level_contributions
        
        # Check that splat contributions are included
        assert "token_1" in attention_result.splat_contributions
        assert "token_2" in attention_result.splat_contributions
        assert "phrase_1" in attention_result.splat_contributions
        assert "sentence_1" in attention_result.splat_contributions
        
        # Check that active splats list is non-empty
        assert len(attention_result.active_splats) > 0
    
    def test_apply_topk_sparsity_with_k(self, sparse_attention_computer):
        """Test apply_topk_sparsity with k parameter."""
        # Create a test matrix
        attention_matrix = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ])
        
        # Apply top-1 sparsity
        sparse_matrix = sparse_attention_computer.apply_topk_sparsity(attention_matrix, k=1)
        
        # Check that only one non-zero element per row
        assert np.sum(sparse_matrix[0] > 0) == 1
        assert np.sum(sparse_matrix[1] > 0) == 1
        assert np.sum(sparse_matrix[2] > 0) == 1
        
        # The max value in each row should be retained
        assert sparse_matrix[0, 2] == 0.3
        assert sparse_matrix[1, 2] == 0.6
        assert sparse_matrix[2, 2] == 0.9
    
    def test_apply_topk_sparsity_with_threshold(self, sparse_attention_computer):
        """Test apply_topk_sparsity with threshold parameter."""
        # Create a test matrix
        attention_matrix = np.array([
            [0.01, 0.02, 0.03],
            [0.04, 0.05, 0.06],
            [0.07, 0.08, 0.09]
        ])
        
        # Apply threshold sparsity
        sparse_matrix = sparse_attention_computer.apply_topk_sparsity(
            attention_matrix, threshold=0.05
        )
        
        # Check that values below threshold are zeroed out
        expected = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.05, 0.06],
            [0.07, 0.08, 0.09]
        ])
        np.testing.assert_array_equal(sparse_matrix, expected)
    
    def test_causal_attention(self):
        """Test that causal masking works correctly."""
        # Create a causal attention computer
        config = AttentionConfig(
            topk=None,
            threshold=0.01,
            normalize_levels=True,
            normalize_rows=True,
            causal=True  # Enable causal masking
        )
        
        causal_computer = SparseAttentionComputer(
            config=config,
            sparsity_threshold=0.01,
            use_spatial_index=False
        )
        
        # Create a simple registry
        hierarchy = Hierarchy(
            levels=["token"],
            init_splats_per_level=[1],
            level_weights=[1.0]
        )
        
        registry = SplatRegistry(hierarchy=hierarchy, embedding_dim=2)
        
        # Add a single splat
        splat = Splat(
            dim=2,
            position=np.array([0.0, 0.0]),
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
            amplitude=1.0,
            level="token",
            id="token_1"
        )
        
        registry.register(splat)
        
        # Create test tokens
        tokens = np.array([
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2]
        ])
        
        # Compute attention
        attention_matrix = causal_computer.compute_attention(tokens, registry)
        
        # Check that the matrix is lower triangular
        assert attention_matrix[0, 1] == 0
        assert attention_matrix[0, 2] == 0
        assert attention_matrix[1, 2] == 0
        
        # Lower triangular elements should be non-zero
        assert attention_matrix[1, 0] > 0
        assert attention_matrix[2, 0] > 0
        assert attention_matrix[2, 1] > 0
    
    def test_compute_token_relevance(self, registry, tokens):
        """Test the compute_token_relevance function from sparse_attention_utils."""
        # Get a splat
        splat = registry.get_splat("token_1")
        
        # Compute token relevance
        relevance = compute_token_relevance(tokens, splat)
        
        # Check shape
        assert relevance.shape == (3,)
        
        # Tokens closer to the splat should have higher relevance
        assert relevance[0] > relevance[2]
    
    def test_get_sparsity_ratio(self):
        """Test the get_sparsity_ratio function from sparse_attention_utils."""
        # Create a test matrix with 50% sparsity
        matrix = np.array([
            [0.1, 0.0, 0.3],
            [0.0, 0.5, 0.0],
            [0.7, 0.0, 0.9]
        ])
        
        # Compute sparsity ratio
        ratio = get_sparsity_ratio(matrix)
        
        # Check that ratio is correct - 4 out of 9 elements are zero
        assert ratio == 4/9
    
    def test_large_sequence_optimization(self, sparse_attention_computer, registry):
        """Test that sparse attention works with larger sequences."""
        # Create a larger sequence of tokens
        large_tokens = np.random.normal(0, 1, (300, 2))
        
        # Ensure that sparse attention can handle it without error
        try:
            attention_matrix = sparse_attention_computer.compute_attention(large_tokens, registry)
            assert attention_matrix.shape == (300, 300)
        except Exception as e:
            pytest.fail(f"Failed to compute attention with large sequence: {e}")
