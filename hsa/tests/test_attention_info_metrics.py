import unittest
import numpy as np
from unittest.mock import Mock, patch

from hsa.attention_info_metrics import InfoTheoreticMetricsComputer, InfoTheoreticCandidateEvaluator
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy


class TestAttentionInfoMetrics(unittest.TestCase):
    """Tests for the attention_info_metrics module."""
    
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
        
        # Create a test splat
        self.splat = Splat(
            dim=2,
            position=np.array([0.0, 0.0]),
            covariance=np.eye(2),
            amplitude=1.0,
            level="token",
            id="test_splat"
        )
        self.registry.register(self.splat)
        
        # Create test tokens
        self.tokens = np.array([
            [0.0, 0.0],
            [0.5, 0.2],
            [1.0, 0.4]
        ])
    
    def test_info_theoretic_metrics_computer_creation(self):
        """Test creating an InfoTheoreticMetricsComputer."""
        metrics_computer = InfoTheoreticMetricsComputer()
        self.assertIsNotNone(metrics_computer)
        self.assertIsInstance(metrics_computer, InfoTheoreticMetricsComputer)
    
    def test_info_theoretic_candidate_evaluator_creation(self):
        """Test creating an InfoTheoreticCandidateEvaluator."""
        metrics_computer = InfoTheoreticMetricsComputer()
        evaluator = InfoTheoreticCandidateEvaluator(metrics_computer)
        self.assertIsNotNone(evaluator)
        self.assertIsInstance(evaluator, InfoTheoreticCandidateEvaluator)
    
    def test_integration(self):
        """Test integration between metrics computer and candidate evaluator."""
        # Create metrics computer
        metrics_computer = InfoTheoreticMetricsComputer()
        
        # Compute metrics
        metrics = metrics_computer.compute_metrics(self.splat, self.registry, self.tokens)
        
        # Create candidate evaluator
        evaluator = InfoTheoreticCandidateEvaluator(metrics_computer)
        
        # Create some candidates for adjustment
        adjust_candidates = []
        for i in range(3):
            splat = Splat(
                dim=2,
                position=np.array([0.1, 0.05]) * i,
                covariance=np.eye(2) * (1.0 - i * 0.1),
                amplitude=1.0 + i * 0.1,
                level="token",
                id=f"adjust_{i}"
            )
            adjust_candidates.append(splat)
        
        # Evaluate candidates
        best_splat = evaluator.evaluate_adjust_candidates(
            self.splat, adjust_candidates, self.registry, self.tokens
        )
        
        # Should select one of the candidates
        self.assertIsInstance(best_splat, Splat)
        found = False
        for candidate in adjust_candidates:
            if candidate.id == best_splat.id:
                found = True
                break
        self.assertTrue(found, "Selected splat not found in candidates")


if __name__ == "__main__":
    unittest.main()
