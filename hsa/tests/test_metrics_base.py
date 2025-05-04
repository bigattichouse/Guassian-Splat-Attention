import unittest
import numpy as np
from unittest.mock import Mock, patch

from hsa.adaptation_metrics_base import AdaptationMetricsComputer, SplatCandidateEvaluator, AdaptationMetricsAggregator
from hsa.adaptation_types import AdaptationMetrics
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy


class TestAdaptationMetricsBase(unittest.TestCase):
    """Tests for the base adaptation metrics interfaces."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a simple hierarchy
        self.hierarchy = Hierarchy(
            levels=["token", "sentence", "document"],
            init_splats_per_level=[10, 5, 2],
            level_weights=[0.5, 0.3, 0.2]
        )
        
        # Create a registry
        self.registry = SplatRegistry(hierarchy=self.hierarchy, embedding_dim=2)
        
        # Create a few test splats
        self.splats = []
        for i in range(3):
            splat = Splat(
                dim=2,
                position=np.array([i * 0.5, i * 0.2]),
                covariance=np.eye(2) * (1.0 + i * 0.1),
                amplitude=1.0,
                level="token",
                id=f"test_splat_{i}"
            )
            self.registry.register(splat)
            self.splats.append(splat)
        
        # Mock metrics computer for testing
        self.mock_metrics_computer = Mock(spec=AdaptationMetricsComputer)
        
        # Set up return values for the mock
        self.mock_metrics = AdaptationMetrics(
            activation_mean=0.5,
            activation_trend=0.1,
            information_contribution=0.3,
            coverage_uniformity=0.7,
            variance=0.2,
            similarity_to_others={"test_splat_1": 0.8}
        )
        
        self.mock_metrics_computer.compute_metrics.return_value = self.mock_metrics
        self.mock_metrics_computer.compute_splat_activation.return_value = 0.5
        self.mock_metrics_computer.compute_activation_trend.return_value = 0.1
        self.mock_metrics_computer.compute_splat_variance.return_value = 0.2
        self.mock_metrics_computer.compute_similarity.return_value = 0.8
        self.mock_metrics_computer.compute_coverage_uniformity.return_value = 0.7
        self.mock_metrics_computer.compute_information_contribution.return_value = 0.3
    
    def test_metrics_aggregator_compute_all_metrics(self):
        """Test computing metrics for all splats."""
        # Create aggregator with mock computer
        aggregator = AdaptationMetricsAggregator(self.mock_metrics_computer)
        
        # Compute metrics for all splats
        all_metrics = aggregator.compute_all_metrics(self.registry)
        
        # Should have metrics for all splats
        self.assertEqual(len(all_metrics), len(self.splats))
        
        # Mock should be called for each splat
        self.assertEqual(self.mock_metrics_computer.compute_metrics.call_count, len(self.splats))
    
    def test_metrics_aggregator_compute_level_metrics(self):
        """Test computing metrics for splats at a specific level."""
        # Create aggregator with mock computer
        aggregator = AdaptationMetricsAggregator(self.mock_metrics_computer)
        
        # Compute metrics for token level
        level_metrics = aggregator.compute_level_metrics("token", self.registry)
        
        # Should have metrics for all token splats
        token_splats = list(self.registry.get_splats_at_level("token"))
        self.assertEqual(len(level_metrics), len(token_splats))
    
    def test_metrics_aggregator_compute_similarity_matrix(self):
        """Test computing similarity matrix."""
        # Create aggregator with mock computer
        aggregator = AdaptationMetricsAggregator(self.mock_metrics_computer)
        
        # Compute similarity matrix
        similarity_matrix = aggregator.compute_similarity_matrix(self.splats)
        
        # Should be a square matrix with size equal to number of splats
        self.assertEqual(similarity_matrix.shape, (len(self.splats), len(self.splats)))
        
        # Diagonal should be 1.0 (self-similarity)
        for i in range(len(self.splats)):
            self.assertEqual(similarity_matrix[i, i], 1.0)
    
    def test_metrics_aggregator_find_similar_splats(self):
        """Test finding similar splats."""
        # Create aggregator with mock computer
        aggregator = AdaptationMetricsAggregator(self.mock_metrics_computer)
        
        # Find similar splats
        similar_pairs = aggregator.find_similar_splats(self.registry, threshold=0.7)
        
        # Since our mock returns 0.8 for similarity, all pairs should be found
        expected_pairs = len(self.splats) * (len(self.splats) - 1) // 2
        self.assertEqual(len(similar_pairs), expected_pairs)
        
        # Each pair should have similarity 0.8
        for _, _, similarity in similar_pairs:
            self.assertEqual(similarity, 0.8)
  
    def test_metrics_aggregator_find_splats_for_death(self):
        """Test finding splats for death."""
        # Create aggregator with mock computer
        aggregator = AdaptationMetricsAggregator(self.mock_metrics_computer)

        # Mock returns 0.5 for activation, threshold is 0.6
        candidates = aggregator.find_splats_for_death(
            self.registry, activation_threshold=0.6, min_lifetime=0
        )

        # All splats should be found (activation 0.5 <= threshold 0.6)
        self.assertEqual(len(candidates), len(self.splats))

        # Try with lower threshold
        candidates = aggregator.find_splats_for_death(
            self.registry, activation_threshold=0.4, min_lifetime=0
        )
        
        # No splats should be found (activation 0.5 > threshold 0.4)
        self.assertEqual(len(candidates), 0)
      
    def test_metrics_aggregator_find_splats_for_mitosis(self):
        """Test finding splats for mitosis."""
        # Create aggregator with mock computer
        aggregator = AdaptationMetricsAggregator(self.mock_metrics_computer)
        
        # Find splats for mitosis
        candidates = aggregator.find_splats_for_mitosis(
            self.registry,
            activation_threshold=0.4,
            variance_threshold=0.1,
            min_lifetime=0
        )
        
        # All splats should be found
        self.assertEqual(len(candidates), len(self.splats))
        
        # Try with higher thresholds
        candidates = aggregator.find_splats_for_mitosis(
            self.registry,
            activation_threshold=0.6,
            variance_threshold=0.3,
            min_lifetime=0
        )
        
        # No splats should be found
        self.assertEqual(len(candidates), 0)


if __name__ == "__main__":
    unittest.main()
