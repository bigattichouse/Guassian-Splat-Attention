"""
Unit tests for the information metrics module.

These tests verify the correctness of the information-theoretic metrics
that enhance the HSA adaptation mechanism.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from hsa.information_metrics import (
    compute_attention_entropy,
    compute_kl_divergence,
    compute_js_divergence,
    compute_mutual_information,
    compute_splat_information_contribution,
    compute_splat_attention_entropy,
    compute_information_gradient,
    compute_splat_attention_map,
    compute_level_information_flow,
    InformationMetricsTracker
)

# Import required dependencies for testing
from hsa.data_structures import Splat, Hierarchy, SplatRegistry, ensure_positive_definite

class TestAttentionEntropy(unittest.TestCase):
    """Test the attention entropy computation."""
    
    def test_uniform_distribution(self):
        """Test entropy of a uniform distribution."""
        # Create a uniform attention matrix (maximum entropy)
        n = 4
        uniform_attn = np.ones((n, n)) / n
        
        # Compute entropy
        entropy = compute_attention_entropy(uniform_attn)
        
        # For a uniform distribution over n elements, entropy should be log2(n)
        expected_entropy = np.log2(n)
        self.assertAlmostEqual(entropy, expected_entropy, places=5)
    
    def test_deterministic_distribution(self):
        """Test entropy of a deterministic distribution (zero entropy)."""
        # Create a deterministic attention matrix (one 1.0 per row, rest zeros)
        n = 4
        deterministic_attn = np.zeros((n, n))
        for i in range(n):
            deterministic_attn[i, i] = 1.0
        
        # Compute entropy
        entropy = compute_attention_entropy(deterministic_attn)
        
        # Deterministic distribution should have zero entropy
        self.assertAlmostEqual(entropy, 0.0, places=5)
    
    def test_mixed_distribution(self):
        """Test entropy of a mixed distribution."""
        # Create an attention matrix with mixed entropy
        mixed_attn = np.array([
            [0.5, 0.5, 0.0, 0.0],  # Entropy = 1
            [0.25, 0.25, 0.25, 0.25],  # Entropy = 2
            [1.0, 0.0, 0.0, 0.0],  # Entropy = 0
            [0.7, 0.1, 0.1, 0.1]   # Entropy ~= 1.36
        ])
        
        # Compute entropy
        entropy = compute_attention_entropy(mixed_attn)
        
        # Expected entropy is the average of row entropies
        expected_entropy = (1.0 + 2.0 + 0.0 + 1.36) / 4
        self.assertAlmostEqual(entropy, expected_entropy, places=1)

class TestKLDivergence(unittest.TestCase):
    """Test the KL divergence computation."""
    
    def test_identical_distributions(self):
        """Test KL divergence of identical distributions (should be zero)."""
        # Create an attention matrix
        attn = np.array([
            [0.5, 0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 0.1, 0.1, 0.1]
        ])
        
        # Compute KL divergence with itself
        kl_div = compute_kl_divergence(attn, attn)
        
        # KL divergence of identical distributions should be zero
        self.assertAlmostEqual(kl_div, 0.0, places=5)
    
    def test_different_distributions(self):
        """Test KL divergence of different distributions."""
        # Create two different attention matrices
        p = np.array([
            [0.5, 0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 0.1, 0.1, 0.1]
        ])
        
        q = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.1, 0.0, 0.0],
            [0.4, 0.2, 0.2, 0.2]
        ])
        
        # Compute KL divergence
        kl_div = compute_kl_divergence(p, q)
        
        # KL divergence should be positive
        self.assertGreater(kl_div, 0.0)
    
    def test_asymmetry(self):
        """Test that KL divergence is asymmetric."""
        # Create two different attention matrices
        p = np.array([
            [0.5, 0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 0.1, 0.1, 0.1]
        ])
        
        q = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.1, 0.0, 0.0],
            [0.4, 0.2, 0.2, 0.2]
        ])
        
        # Compute KL divergence in both directions
        kl_p_q = compute_kl_divergence(p, q)
        kl_q_p = compute_kl_divergence(q, p)
        
        # KL divergence should be asymmetric
        self.assertNotAlmostEqual(kl_p_q, kl_q_p, places=3)

class TestJSDivergence(unittest.TestCase):
    """Test the JS divergence computation."""
    
    def test_identical_distributions(self):
        """Test JS divergence of identical distributions (should be zero)."""
        # Create an attention matrix
        attn = np.array([
            [0.5, 0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 0.1, 0.1, 0.1]
        ])
        
        # Compute JS divergence with itself
        js_div = compute_js_divergence(attn, attn)
        
        # JS divergence of identical distributions should be zero
        self.assertAlmostEqual(js_div, 0.0, places=5)
    
    def test_different_distributions(self):
        """Test JS divergence of different distributions."""
        # Create two different attention matrices
        p = np.array([
            [0.5, 0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 0.1, 0.1, 0.1]
        ])
        
        q = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.1, 0.0, 0.0],
            [0.4, 0.2, 0.2, 0.2]
        ])
        
        # Compute JS divergence
        js_div = compute_js_divergence(p, q)
        
        # JS divergence should be positive
        self.assertGreater(js_div, 0.0)
    
    def test_symmetry(self):
        """Test that JS divergence is symmetric."""
        # Create two different attention matrices
        p = np.array([
            [0.5, 0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [1.0, 0.0, 0.0, 0.0],
            [0.7, 0.1, 0.1, 0.1]
        ])
        
        q = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.1, 0.0, 0.0],
            [0.4, 0.2, 0.2, 0.2]
        ])
        
        # Compute JS divergence in both directions
        js_p_q = compute_js_divergence(p, q)
        js_q_p = compute_js_divergence(q, p)
        
        # JS divergence should be symmetric
        self.assertAlmostEqual(js_p_q, js_q_p, places=5)

class TestMutualInformation(unittest.TestCase):
    """Test the mutual information computation."""
    
    def test_independent_variables(self):
        """Test mutual information of independent variables (should be zero)."""
        # Create a joint distribution of independent variables
        p_x = np.array([0.3, 0.7])
        p_y = np.array([0.4, 0.6])
        
        # Independent joint distribution is the outer product
        joint = np.outer(p_x, p_y)
        
        # Compute mutual information
        mi = compute_mutual_information(joint)
        
        # MI of independent variables should be zero
        self.assertAlmostEqual(mi, 0.0, places=5)
    
    def test_dependent_variables(self):
        """Test mutual information of dependent variables."""
        # Create a joint distribution with dependence
        joint = np.array([
            [0.4, 0.1],
            [0.1, 0.4]
        ])
        
        # Compute mutual information
        mi = compute_mutual_information(joint)
        
        # MI should be positive for dependent variables
        self.assertGreater(mi, 0.0)
    
    def test_deterministic_relationship(self):
        """Test mutual information of a deterministic relationship (maximum MI)."""
        # Create a joint distribution with perfect correlation
        joint = np.array([
            [0.5, 0.0],
            [0.0, 0.5]
        ])
        
        # Compute mutual information
        mi = compute_mutual_information(joint)
        
        # For a 2x2 distribution with perfect correlation, MI should be 1 bit
        expected_mi = 1.0
        self.assertAlmostEqual(mi, expected_mi, places=5)

class TestSplatInformationContribution(unittest.TestCase):
    """Test the splat information contribution computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock hierarchy
        self.hierarchy = Hierarchy(
            levels=["Token", "Phrase"],
            init_splats_per_level=[10, 5],
            level_weights=[0.6, 0.4]
        )
        
        # Create mock splat registry
        self.splat_registry = SplatRegistry(self.hierarchy)
        
        # Create mock tokens
        self.tokens = np.random.randn(10, 4)  # 10 tokens, 4-dim embeddings
        
        # Create mock splats
        self.splat1 = self._create_mock_splat(
            position=np.array([0.1, 0.2, 0.3, 0.4]),
            level="Token",
            amplitude=1.0
        )
        
        self.splat2 = self._create_mock_splat(
            position=np.array([0.5, 0.6, 0.7, 0.8]),
            level="Token",
            amplitude=0.8
        )
        
        self.splat3 = self._create_mock_splat(
            position=np.array([0.9, 0.8, 0.7, 0.6]),
            level="Phrase",
            amplitude=1.2
        )
        
        # Register splats
        self.splat_registry.register(self.splat1)
        self.splat_registry.register(self.splat2)
        self.splat_registry.register(self.splat3)
        
        # Create mock attention matrix
        self.attention_matrix = np.random.rand(10, 10)
    
    def _create_mock_splat(self, position, level, amplitude):
        """Helper to create a mock splat."""
        # Create a positive definite covariance matrix
        cov = np.eye(position.shape[0]) * 0.1
        
        return Splat(
            position=position,
            covariance=cov,
            amplitude=amplitude,
            level=level
        )
    
    @patch('hsa.information_metrics.create_attention_computer')
    def test_information_contribution(self, mock_create_attention):
        """Test computation of splat information contribution."""
        # Create mock attention computer
        mock_computer = MagicMock()
        mock_create_attention.return_value = mock_computer
        
        # Set up mock return values
        # 1. Full attention (already have self.attention_matrix)
        # 2. Attention without the splat
        attention_without_splat = np.random.rand(10, 10)
        mock_computer.compute_attention.return_value = attention_without_splat
        
        # Compute information contribution for splat1
        info_contribution = compute_splat_information_contribution(
            self.splat1, self.tokens, self.attention_matrix, self.splat_registry
        )
        
        # Verify the attention computer was created and called correctly
        mock_create_attention.assert_called_once()
        mock_computer.compute_attention.assert_called_once()
        
        # Verify the result is a non-negative float
        self.assertIsInstance(info_contribution, float)
        self.assertGreaterEqual(info_contribution, 0.0)

class TestSplatAttentionEntropy(unittest.TestCase):
    """Test the splat attention entropy computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock tokens
        self.tokens = np.random.randn(5, 4)  # 5 tokens, 4-dim embeddings
        
        # Create splats with different characteristics
        # Focused splat (low entropy)
        self.focused_splat = Splat(
            position=np.array([0.1, 0.2, 0.3, 0.4]),
            covariance=np.eye(4) * 0.01,  # Very small covariance
            amplitude=1.0,
            level="Token"
        )
        
        # Diffuse splat (high entropy)
        self.diffuse_splat = Splat(
            position=np.array([0.5, 0.6, 0.7, 0.8]),
            covariance=np.eye(4) * 1.0,   # Large covariance
            amplitude=1.0,
            level="Token"
        )
    
    def test_entropy_comparison(self):
        """Test that focused splat has lower entropy than diffuse splat."""
        # Compute entropies
        focused_entropy = compute_splat_attention_entropy(self.focused_splat, self.tokens)
        diffuse_entropy = compute_splat_attention_entropy(self.diffuse_splat, self.tokens)
        
        # Verify that diffuse splat has higher entropy
        self.assertLess(focused_entropy, diffuse_entropy)
    
    def test_entropy_value_range(self):
        """Test that entropy values are in expected range."""
        # Compute entropy
        entropy = compute_splat_attention_entropy(self.focused_splat, self.tokens)
        
        # For a 5x5 attention matrix, entropy should be between 0 and log2(5) ~= 2.32
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, np.log2(5) + 0.1)  # Add small tolerance

class TestInformationGradient(unittest.TestCase):
    """Test the information gradient computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock tokens
        self.tokens = np.random.randn(6, 3)  # 6 tokens, 3-dim embeddings
        
        # Create a mock splat
        self.splat = Splat(
            position=np.array([0.1, 0.2, 0.3]),
            covariance=np.eye(3) * 0.1,
            amplitude=1.0,
            level="Token"
        )
        
        # Create mock attention matrix
        self.attention_matrix = np.random.rand(6, 6)
    
    def test_gradient_shape(self):
        """Test that gradient has correct shape."""
        # Compute gradient
        gradient = compute_information_gradient(
            self.splat, self.tokens, self.attention_matrix
        )
        
        # Verify shape matches splat position
        self.assertEqual(gradient.shape, self.splat.position.shape)
    
    def test_gradient_normalization(self):
        """Test that gradient is normalized."""
        # Compute gradient
        gradient = compute_information_gradient(
            self.splat, self.tokens, self.attention_matrix
        )
        
        # Verify gradient is a unit vector (approximately)
        norm = np.linalg.norm(gradient)
        self.assertAlmostEqual(norm, 1.0, places=5)

class TestSplatAttentionMap(unittest.TestCase):
    """Test the splat attention map computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock tokens
        self.tokens = np.random.randn(4, 2)  # 4 tokens, 2-dim embeddings
        
        # Create a mock splat
        self.splat = Splat(
            position=np.array([0.1, 0.2]),
            covariance=np.eye(2) * 0.1,
            amplitude=1.0,
            level="Token"
        )
    
    def test_attention_map_shape(self):
        """Test attention map has correct shape."""
        # Compute attention map
        attention_map = compute_splat_attention_map(self.splat, self.tokens)
        
        # Verify shape matches token sequence length
        expected_shape = (self.tokens.shape[0], self.tokens.shape[0])
        self.assertEqual(attention_map.shape, expected_shape)
    
    def test_attention_map_values(self):
        """Test attention map has correct values."""
        # Compute attention map
        attention_map = compute_splat_attention_map(self.splat, self.tokens)
        
        # Verify all values are between 0 and the splat's amplitude
        self.assertTrue(np.all(attention_map >= 0.0))
        self.assertTrue(np.all(attention_map <= self.splat.amplitude + 1e-10))
        
        # Verify diagonal values are the highest (tokens attend to themselves most)
        for i in range(attention_map.shape[0]):
            row_max = np.max(attention_map[i, :])
            self.assertAlmostEqual(attention_map[i, i], row_max, places=5)

class TestLevelInformationFlow(unittest.TestCase):
    """Test the level information flow computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock hierarchy
        self.hierarchy = Hierarchy(
            levels=["Token", "Phrase", "Document"],
            init_splats_per_level=[10, 5, 2],
            level_weights=[0.5, 0.3, 0.2]
        )
        
        # Create mock splat registry
        self.splat_registry = SplatRegistry(self.hierarchy)
        
        # Create mock tokens
        self.tokens = np.random.randn(8, 4)  # 8 tokens, 4-dim embeddings
        
        # Create and register splats at different levels
        for i in range(3):  # 3 token-level splats
            splat = Splat(
                position=np.random.randn(4),
                covariance=np.eye(4) * 0.1,
                amplitude=1.0,
                level="Token"
            )
            self.splat_registry.register(splat)
        
        for i in range(2):  # 2 phrase-level splats
            splat = Splat(
                position=np.random.randn(4),
                covariance=np.eye(4) * 0.2,
                amplitude=0.8,
                level="Phrase"
            )
            self.splat_registry.register(splat)
        
        for i in range(1):  # 1 document-level splat
            splat = Splat(
                position=np.random.randn(4),
                covariance=np.eye(4) * 0.3,
                amplitude=0.5,
                level="Document"
            )
            self.splat_registry.register(splat)
    
    @patch('hsa.information_metrics.create_attention_computer')
    def test_information_flow(self, mock_create_attention):
        """Test computation of information flow between levels."""
        # Create mock attention computer
        mock_computer = MagicMock()
        mock_create_attention.return_value = mock_computer
        
        # Set up mock return values for attention matrices
        token_attention = np.random.rand(8, 8)
        phrase_attention = np.random.rand(8, 8)
        mock_computer.compute_attention.side_effect = [token_attention, phrase_attention]
        
        # Compute information flow from Token to Phrase
        info_flow = compute_level_information_flow(
            self.splat_registry, self.tokens, "Token", "Phrase"
        )
        
        # Verify the attention computer was created and called correctly
        mock_create_attention.assert_called_once()
        self.assertEqual(mock_computer.compute_attention.call_count, 2)
        
        # Verify the result is a non-negative float
        self.assertIsInstance(info_flow, float)
        self.assertGreaterEqual(info_flow, 0.0)

class TestInformationMetricsTracker(unittest.TestCase):
    """Test the information metrics tracker."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock hierarchy
        self.hierarchy = Hierarchy(
            levels=["Token", "Phrase"],
            init_splats_per_level=[5, 2],
            level_weights=[0.7, 0.3]
        )
        
        # Create mock splat registry
        self.splat_registry = SplatRegistry(self.hierarchy)
        
        # Create mock tokens
        self.tokens = np.random.randn(6, 3)  # 6 tokens, 3-dim embeddings
        
        # Create and register splats
        for i in range(3):  # 3 token-level splats
            splat = Splat(
                position=np.random.randn(3),
                covariance=np.eye(3) * 0.1,
                amplitude=1.0,
                level="Token"
            )
            self.splat_registry.register(splat)
        
        for i in range(2):  # 2 phrase-level splats
            splat = Splat(
                position=np.random.randn(3),
                covariance=np.eye(3) * 0.2,
                amplitude=0.8,
                level="Phrase"
            )
            self.splat_registry.register(splat)
        
        # Create mock attention matrix
        self.attention_matrix = np.random.rand(6, 6)
        
        # Create the tracker
        self.tracker = InformationMetricsTracker()
    
    @patch('hsa.information_metrics.compute_splat_information_contribution')
    @patch('hsa.information_metrics.compute_splat_attention_entropy')
    @patch('hsa.information_metrics.compute_level_information_flow')
    def test_compute_all_metrics(self, mock_flow, mock_entropy, mock_info_contribution):
        """Test computation of all metrics."""
        # Set up mock return values
        mock_info_contribution.return_value = 0.5
        mock_entropy.return_value = 1.2
        mock_flow.return_value = 0.3
        
        # Compute all metrics
        self.tracker.compute_all_metrics(
            self.splat_registry, self.tokens, self.attention_matrix
        )
        
        # Verify correct number of calls (5 splats total)
        self.assertEqual(mock_info_contribution.call_count, 5)
        self.assertEqual(mock_entropy.call_count, 5)
        
        # Verify level flow calls (2 levels, so 2 possible flows)
        expected_flow_calls = 2
        self.assertEqual(mock_flow.call_count, expected_flow_calls)
        
        # Verify metrics were stored
        self.assertEqual(len(self.tracker.splat_info_contributions), 5)
        self.assertEqual(len(self.tracker.splat_entropies), 5)
        self.assertEqual(len(self.tracker.level_info_flows), expected_flow_calls)
    
    def test_get_splat_metrics(self):
        """Test retrieving metrics for a specific splat."""
        # Set up test metrics
        splat_id = list(self.splat_registry.splats.keys())[0]
        self.tracker.splat_info_contributions[splat_id] = 0.7
        self.tracker.splat_entropies[splat_id] = 1.5
        
        # Get metrics for this splat
        metrics = self.tracker.get_splat_metrics(splat_id)
        
        # Verify metrics match what we set
        self.assertEqual(metrics["info_contribution"], 0.7)
        self.assertEqual(metrics["entropy"], 1.5)
    
    def test_get_top_info_contributors(self):
        """Test retrieving top information contributors."""
        # Set up test metrics for all splats
        splat_ids = list(self.splat_registry.splats.keys())
        contributions = [0.3, 0.7, 0.2, 0.5, 0.4]
        
        for i, splat_id in enumerate(splat_ids):
            self.tracker.splat_info_contributions[splat_id] = contributions[i]
        
        # Get top 3 contributors
        top3 = self.tracker.get_top_info_contributors(3)
        
        # Verify we get 3 results
        self.assertEqual(len(top3), 3)
        
        # Verify they're sorted in descending order
        self.assertEqual(top3[0][1], 0.7)  # Highest contribution
        self.assertEqual(top3[1][1], 0.5)  # Second highest
        self.assertEqual(top3[2][1], 0.4)  # Third highest
    
    def test_reset(self):
        """Test resetting all metrics."""
        # Set up some metrics
        splat_id = list(self.splat_registry.splats.keys())[0]
        self.tracker.splat_info_contributions[splat_id] = 0.7
        self.tracker.splat_entropies[splat_id] = 1.5
        self.tracker.level_info_flows["Token_to_Phrase"] = 0.3
        
        # Reset metrics
        self.tracker.reset()
        
        # Verify all metrics were cleared
        self.assertEqual(len(self.tracker.splat_info_contributions), 0)
        self.assertEqual(len(self.tracker.splat_entropies), 0)
        self.assertEqual(len(self.tracker.level_info_flows), 0)

if __name__ == '__main__':
    unittest.main()
