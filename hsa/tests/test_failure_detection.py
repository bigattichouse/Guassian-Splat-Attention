"""
Unit tests for failure detection in Hierarchical Splat Attention (HSA).

This module contains tests for the failure detection components, including
the FailureDetector class and specialized analyzers.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy
from hsa.failure_detection import FailureDetector, detect_pathological_configurations
from hsa.failure_detection_types import FailureType
from hsa.failure_detection_analyzers import AttentionMatrixAnalyzer, analyze_splat_configuration


class TestFailureDetectionTypes(unittest.TestCase):
    """Tests for failure type enumerations."""
    
    def test_failure_types_exist(self):
        """Test that all expected failure types are defined."""
        failure_types = [
            FailureType.NUMERICAL_INSTABILITY,
            FailureType.EMPTY_LEVEL,
            FailureType.ORPHANED_SPLAT,
            FailureType.ADAPTATION_STAGNATION,
            FailureType.PATHOLOGICAL_CONFIGURATION,
            FailureType.ATTENTION_COLLAPSE,
            FailureType.INFORMATION_BOTTLENECK,
            FailureType.MEMORY_OVERFLOW,
            FailureType.GRADIENT_INSTABILITY
        ]
        
        # Verify all types are accessible
        for failure_type in failure_types:
            self.assertIsInstance(failure_type, FailureType)


class TestFailureDetector(unittest.TestCase):
    """Tests for the FailureDetector class."""
    
    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a simple hierarchy for testing
        self.hierarchy = Hierarchy(
            levels=["token", "sentence", "document"],
            init_splats_per_level=[5, 3, 1],
            level_weights=[0.6, 0.3, 0.1]
        )
        
        # Create a registry
        self.registry = SplatRegistry(hierarchy=self.hierarchy, embedding_dim=2)
        
        # Create a detector
        self.detector = FailureDetector()
    
    def test_init(self):
        """Test initialization of the detector."""
        detector = FailureDetector(
            sensitivity=0.5,
            check_parent_child=False,
            check_covariance=False
        )
        
        self.assertEqual(detector.sensitivity, 0.5)
        self.assertFalse(detector.check_parent_child)
        self.assertFalse(detector.check_covariance)
        self.assertTrue(detector.check_attention)  # Default is True
        
        # Check default values
        default_detector = FailureDetector()
        self.assertEqual(default_detector.sensitivity, 1.0)
        self.assertTrue(default_detector.check_parent_child)
    
    def test_detect_empty_levels(self):
        """Test detection of empty levels."""
        # Initially all levels should be empty
        empty_levels = self.detector._detect_empty_levels(self.registry)
        self.assertEqual(set(empty_levels), {"token", "sentence", "document"})
        
        # Add a splat to one level
        token_splat = Splat(dim=2, level="token")
        self.registry.register(token_splat)
        
        empty_levels = self.detector._detect_empty_levels(self.registry)
        self.assertEqual(set(empty_levels), {"sentence", "document"})
    
    def test_detect_orphaned_splats(self):
        """Test detection of orphaned splats."""
        # Create splats at different levels
        token_splat = Splat(dim=2, level="token")
        sentence_splat = Splat(dim=2, level="sentence")
        doc_splat = Splat(dim=2, level="document")
        
        # Register splats
        self.registry.register(token_splat)
        self.registry.register(sentence_splat)
        self.registry.register(doc_splat)
        
        # Token splat should be orphaned (no parent)
        orphaned_ids = self.detector._detect_orphaned_splats(self.registry)
        self.assertEqual(orphaned_ids, [token_splat.id])
        
        # Set parent for token splat
        token_splat.parent = sentence_splat
        sentence_splat.children.add(token_splat)
        
        # No splats should be orphaned now
        orphaned_ids = self.detector._detect_orphaned_splats(self.registry)
        self.assertEqual(orphaned_ids, [])
        
        # Create a problematic parent-child relationship
        token_splat.parent = token_splat  # Setting parent to self (invalid)
        orphaned_ids = self.detector._detect_orphaned_splats(self.registry)
        self.assertEqual(orphaned_ids, [token_splat.id])
    
    def test_detect_covariance_instabilities(self):
        """Test detection of covariance instabilities."""
        # Create a splat with stable covariance
        stable_splat = Splat(dim=2, level="token", 
                            position=np.array([0.0, 0.0]),
                            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.registry.register(stable_splat)
        
        # Create a splat with unstable covariance (near-zero eigenvalue)
        unstable_splat = Splat(dim=2, level="token",
                              position=np.array([1.0, 1.0]),
                              covariance=np.array([[1.0, 0.99], [0.99, 1.0]]))
        self.registry.register(unstable_splat)
        
        # Create a splat with NaN in covariance
        nan_splat = Splat(dim=2, level="token")
        nan_splat.covariance = np.array([[1.0, 0.0], [0.0, np.nan]])
        self.registry.register(nan_splat)
        
        # Detect instabilities
        unstable_splats = self.detector._detect_covariance_instabilities(self.registry)
        
        # We should find the unstable and NaN splats
        self.assertEqual(len(unstable_splats), 2)
        
        # Verify the right splats were identified
        unstable_ids = [id for id, _ in unstable_splats]
        self.assertIn(unstable_splat.id, unstable_ids)
        self.assertIn(nan_splat.id, unstable_ids)
        self.assertNotIn(stable_splat.id, unstable_ids)
    
    def test_detect_pathological_configurations(self):
        """Test the main detection method."""
        # Create a problematic configuration
        token_splat = Splat(dim=2, level="token")
        self.registry.register(token_splat)
        
        # Detect issues
        failures = self.detector.detect_pathological_configurations(self.registry)
        
        # We should find empty levels and orphaned splat
        self.assertGreaterEqual(len(failures), 3)  # 2 empty levels + 1 orphaned splat
        
        # Verify failure types
        failure_types = [f[0] for f in failures]
        self.assertIn(FailureType.EMPTY_LEVEL, failure_types)
        self.assertIn(FailureType.ORPHANED_SPLAT, failure_types)
        
        # Check failure history
        self.assertEqual(self.detector.detection_count, 1)
        self.assertTrue(any(self.detector.failure_history[FailureType.EMPTY_LEVEL]))
    
    def test_analyze_failure_trends(self):
        """Test analysis of failure trends."""
        # Simulate multiple detections with failures
        for _ in range(20):
            self.detector.detection_count += 1
            self.detector.failure_history[FailureType.EMPTY_LEVEL].append(
                (self.detector.detection_count, {"level": "sentence"})
            )
        
        # Add some other types for variety
        self.detector.failure_history[FailureType.ORPHANED_SPLAT].append(
            (5, {"splat_id": "test_id"})
        )
        
        # Analyze trends
        trends = self.detector.analyze_failure_trends()
        
        # Verify analysis
        self.assertIn(FailureType.EMPTY_LEVEL.name, trends)
        self.assertIn("frequency", trends[FailureType.EMPTY_LEVEL.name])
        self.assertIn("trend", trends[FailureType.EMPTY_LEVEL.name])
        self.assertIn("count", trends[FailureType.EMPTY_LEVEL.name])
        
        # Check frequency calculation
        expected_frequency = 20 / 20  # 20 occurrences in 20 detections
        self.assertAlmostEqual(trends[FailureType.EMPTY_LEVEL.name]["frequency"], expected_frequency)
    
    def test_categorize_registry_health(self):
        """Test health categorization of registry."""
        # Empty registry should have issues
        health = self.detector.categorize_registry_health(self.registry)
        
        # Verify health assessment
        self.assertIn("health_score", health)
        self.assertIn("category", health)
        self.assertIn("issue_count", health)
        self.assertIn("issues_by_type", health)
        self.assertIn("needs_repair", health)
        
        # Test with a healthier registry
        mock_registry = MagicMock()
        # Setup the mock to have no issues
        with patch.object(self.detector, 'detect_pathological_configurations', return_value=[]):
            health = self.detector.categorize_registry_health(mock_registry)
            self.assertEqual(health["health_score"], 1.0)
            self.assertEqual(health["category"], "Excellent")
            self.assertFalse(health["needs_repair"])


class TestAttentionMatrixAnalyzer(unittest.TestCase):
    """Tests for the AttentionMatrixAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = AttentionMatrixAnalyzer()
    
    def test_analyze_balanced_attention(self):
        """Test analysis of a balanced attention matrix."""
        # Create a balanced attention matrix
        attention = np.ones((5, 5)) / 5  # Uniform attention
        
        results = self.analyzer.analyze_attention_matrix(attention)
        
        # Verify results
        self.assertEqual(results["shape"], (5, 5))
        self.assertFalse(results["has_nan"])
        self.assertFalse(results["has_inf"])
        self.assertAlmostEqual(results["sparsity"], 0.0)
        self.assertFalse(results["too_sparse"])
        
        # Check Gini coefficient for equality
        self.assertAlmostEqual(results["attention_gini"], 0.0)
        self.assertFalse(results["attention_collapse"])
        
        # Check entropy (should be high for uniform distribution)
        self.assertAlmostEqual(results["attention_entropy"], 1.0)
        self.assertFalse(results["low_entropy"])
    
    def test_analyze_collapsed_attention(self):
        """Test analysis of a collapsed attention matrix."""
        # Create a collapsed attention matrix (all attention to one token)
        attention = np.zeros((5, 5))
        attention[:, 0] = 1.0  # All attention to first token
        
        results = self.analyzer.analyze_attention_matrix(attention)
        
        # Verify results
        self.assertTrue(results["attention_collapse"])
        self.assertTrue(results["low_entropy"])
        self.assertGreater(results["attention_gini"], 0.8)
    
    def test_analyze_diagonal_attention(self):
        """Test analysis of a diagonal-dominant attention matrix."""
        # Create a diagonal-dominant matrix
        attention = np.eye(5) * 0.9  # 90% self-attention
        
        # Add small off-diagonal values
        attention += np.ones((5, 5)) * 0.02
        
        results = self.analyzer.analyze_attention_matrix(attention)
        
        # Verify results
        self.assertTrue(results["diagonal_dominant"])
        self.assertGreater(results["diagonal_ratio"], 0.8)
    
    def test_analyze_invalid_attention(self):
        """Test analysis of an invalid attention matrix."""
        # Create an attention matrix with NaN and Inf
        attention = np.ones((3, 3))
        attention[0, 0] = np.nan
        attention[1, 1] = np.inf
        
        results = self.analyzer.analyze_attention_matrix(attention)
        
        # Verify results
        self.assertTrue(results["has_nan"])
        self.assertTrue(results["has_inf"])
    
    def test_gini_coefficient(self):
        """Test Gini coefficient calculation."""
        # Equal distribution
        equal = np.ones(10)
        gini = self.analyzer._calculate_gini(equal)
        self.assertAlmostEqual(gini, 0.0)
        
        # Completely unequal distribution
        unequal = np.zeros(10)
        unequal[0] = 10
        gini = self.analyzer._calculate_gini(unequal)
        self.assertAlmostEqual(gini, 0.9)
        
        # Edge case - all zeros
        zeros = np.zeros(5)
        gini = self.analyzer._calculate_gini(zeros)
        self.assertEqual(gini, 0.0)
    
    def test_entropy_calculation(self):
        """Test entropy calculation."""
        # Uniform distribution has maximum entropy
        uniform = np.ones((4, 4)) / 16
        entropy = self.analyzer._calculate_entropy(uniform)
        self.assertAlmostEqual(entropy, 1.0)
        
        # Single value has minimum entropy
        single = np.zeros((4, 4))
        single[0, 0] = 1.0
        entropy = self.analyzer._calculate_entropy(single)
        self.assertAlmostEqual(entropy, 0.0)
        
        # Edge case - all zeros
        zeros = np.zeros((3, 3))
        entropy = self.analyzer._calculate_entropy(zeros)
        self.assertEqual(entropy, 0.0)


class TestSplatConfigurationAnalysis(unittest.TestCase):
    """Tests for splat configuration analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hierarchy = Hierarchy(
            levels=["token", "sentence", "document"],
            init_splats_per_level=[5, 3, 1],
            level_weights=[0.6, 0.3, 0.1]
        )
        self.registry = SplatRegistry(hierarchy=self.hierarchy, embedding_dim=2)
    
    def test_analyze_empty_registry(self):
        """Test analysis of an empty registry."""
        issues = analyze_splat_configuration(self.registry)
        
        # Should detect empty levels
        self.assertEqual(len(issues), 3)  # One for each empty level
        for issue in issues:
            self.assertEqual(issue["type"], "empty_level")
    
    def test_analyze_imbalanced_levels(self):
        """Test analysis of registry with imbalanced levels."""
        # Add many splats to one level and few to another
        for i in range(10):
            splat = Splat(dim=2, level="token")
            self.registry.register(splat)
        
        splat = Splat(dim=2, level="sentence")
        self.registry.register(splat)
        
        issues = analyze_splat_configuration(self.registry)
        
        # Should flag the sparse level
        sparse_issues = [issue for issue in issues if issue["type"] == "sparse_level"]
        self.assertGreaterEqual(len(sparse_issues), 1)
        
        # Should still have one empty level
        empty_issues = [issue for issue in issues if issue["type"] == "empty_level"]
        self.assertEqual(len(empty_issues), 1)
        self.assertEqual(empty_issues[0]["level"], "document")
    
    def test_analyze_skewed_covariance(self):
        """Test detection of skewed covariance matrices."""
        # Create a splat with highly skewed covariance
        skewed_splat = Splat(dim=2, level="token")
        skewed_splat.covariance = np.array([[1000.0, 0.0], [0.0, 0.1]])
        self.registry.register(skewed_splat)
        
        issues = analyze_splat_configuration(self.registry)
        
        # Should detect the skewed covariance
        skewed_issues = [issue for issue in issues if issue["type"] == "skewed_covariance"]
        self.assertEqual(len(skewed_issues), 1)
        self.assertEqual(skewed_issues[0]["splat_id"], skewed_splat.id)
    
    def test_analyze_invalid_covariance(self):
        """Test detection of invalid covariance matrices."""
        # Create a splat with invalid covariance (non-positive definite)
        invalid_splat = Splat(dim=2, level="token")
        invalid_splat.covariance = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive definite
        self.registry.register(invalid_splat)
        
        issues = analyze_splat_configuration(self.registry)
        
        # Should detect the invalid covariance
        invalid_issues = [issue for issue in issues if issue["type"] == "invalid_covariance"]
        self.assertEqual(len(invalid_issues), 1)
        self.assertEqual(invalid_issues[0]["splat_id"], invalid_splat.id)
    
    def test_analyze_overlapping_splats(self):
        """Test detection of overlapping splats."""
        # Create two splats with very close positions
        splat_a = Splat(dim=2, level="token", 
                       position=np.array([0.0, 0.0]),
                       covariance=np.array([[1.0, 0.0], [0.0, 1.0]]))
        
        splat_b = Splat(dim=2, level="token", 
                       position=np.array([0.1, 0.1]),
                       covariance=np.array([[1.0, 0.0], [0.0, 1.0]]))
        
        self.registry.register(splat_a)
        self.registry.register(splat_b)
        
        issues = analyze_splat_configuration(self.registry)
        
        # Should detect overlapping splats
        overlap_issues = [issue for issue in issues if issue["type"] == "overlapping_splats"]
        self.assertEqual(len(overlap_issues), 1)
        self.assertIn(splat_a.id, overlap_issues[0]["splat_ids"])
        self.assertIn(splat_b.id, overlap_issues[0]["splat_ids"])


class TestHelperFunctions(unittest.TestCase):
    """Tests for helper functions in failure detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hierarchy = Hierarchy(
            levels=["token", "sentence"],
            init_splats_per_level=[5, 2],
            level_weights=[0.7, 0.3]
        )
        self.registry = SplatRegistry(hierarchy=self.hierarchy, embedding_dim=2)
    
    def test_detect_pathological_configurations(self):
        """Test the helper function for detecting pathological configurations."""
        with patch('hsa.failure_detection_analyzers.analyze_splat_configuration', 
                   return_value=[{"type": "test_issue", "message": "Test"}]):
            issues = detect_pathological_configurations(self.registry, sensitivity=0.5)
            
            # Should get issues from both detector and analyzer
            self.assertGreater(len(issues), 1)
            
            # Verify issue structure
            self.assertIn("type", issues[0])
            self.assertIn("message", issues[0])
            self.assertIn("severity", issues[0])


if __name__ == '__main__':
    unittest.main()
