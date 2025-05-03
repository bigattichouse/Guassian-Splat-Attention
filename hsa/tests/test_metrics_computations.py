import unittest
import numpy as np
from unittest.mock import Mock, patch

from hsa.candidate_evaluation import InfoTheoreticCandidateEvaluator
from hsa.metrics_computation import InfoTheoreticMetricsComputer
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy


class TestInfoTheoreticCandidateEvaluator(unittest.TestCase):
    """Tests for the InfoTheoreticCandidateEvaluator class."""
    
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
        
        # Create test splats
        self.original_splat = Splat(
            dim=2,
            position=np.array([0.0, 0.0]),
            covariance=np.eye(2),
            amplitude=1.0,
            level="token",
            id="original_splat"
        )
        self.registry.register(self.original_splat)
        
        # Create test tokens
        self.tokens = np.array([
            [0.0, 0.0],
            [0.5, 0.2],
            [1.0, 0.4]
        ])
        
        # Create metrics computer
        self.metrics_computer = InfoTheoreticMetricsComputer()
        
        # Create candidate evaluator
        self.evaluator = InfoTheoreticCandidateEvaluator(self.metrics_computer)
        
        # Create candidate splats for various operations
        
        # Mitosis candidates
        self.mitosis_candidates = []
        for i in range(3):
            splat_a = Splat(
                dim=2,
                position=np.array([-0.2, -0.1]) * (i + 1),
                covariance=np.eye(2) * 0.8,
                amplitude=1.0,
                level="token",
                id=f"mitosis_a_{i}"
            )
            
            splat_b = Splat(
                dim=2,
                position=np.array([0.2, 0.1]) * (i + 1),
                covariance=np.eye(2) * 0.8,
                amplitude=1.0,
                level="token",
                id=f"mitosis_b_{i}"
            )
            
            self.mitosis_candidates.append((splat_a, splat_b))
        
        # Merge candidates
        self.splat_a = Splat(
            dim=2,
            position=np.array([-0.2, -0.1]),
            covariance=np.eye(2) * 0.8,
            amplitude=1.0,
            level="token",
            id="splat_a"
        )
        
        self.splat_b = Splat(
            dim=2,
            position=np.array([0.2, 0.1]),
            covariance=np.eye(2) * 0.8,
            amplitude=1.0,
            level="token",
            id="splat_b"
        )
        
        self.merge_candidates = []
        for i in range(3):
            splat = Splat(
                dim=2,
                position=np.array([0.0, 0.0]) + np.array([0.05, 0.02]) * i,
                covariance=np.eye(2) * (1.0 + i * 0.1),
                amplitude=1.0,
                level="token",
                id=f"merge_{i}"
            )
            
            self.merge_candidates.append(splat)
        
        # Birth candidates
        self.birth_candidates = []
        for i in range(3):
            splat = Splat(
                dim=2,
                position=np.array([0.5, 0.2]) * (i + 1),
                covariance=np.eye(2),
                amplitude=1.0,
                level="token",
                id=f"birth_{i}"
            )
            
            self.birth_candidates.append(splat)
        
        # Adjust candidates
        self.adjust_candidates = []
        for i in range(3):
            splat = Splat(
                dim=2,
                position=np.array([0.1, 0.05]) * i,
                covariance=np.eye(2) * (1.0 - i * 0.1),
                amplitude=1.0 + i * 0.1,
                level="token",
                id=f"adjust_{i}"
            )
            
            self.adjust_candidates.append(splat)
    
    def test_evaluate_mitosis_candidates(self):
        """Test evaluating mitosis candidates."""
        # Evaluate candidates
        best_pair = self.evaluator.evaluate_mitosis_candidates(
            self.original_splat, self.mitosis_candidates, self.registry, self.tokens
        )
        
        # Should return a pair of splats
        self.assertIsInstance(best_pair, tuple)
        self.assertEqual(len(best_pair), 2)
        self.assertIsInstance(best_pair[0], Splat)
        self.assertIsInstance(best_pair[1], Splat)
        
        # Should select one of the candidate pairs
        found = False
        for pair in self.mitosis_candidates:
            if pair[0].id == best_pair[0].id and pair[1].id == best_pair[1].id:
                found = True
                break
        self.assertTrue(found, "Selected pair not found in candidates")
    
    def test_evaluate_merge_candidates(self):
        """Test evaluating merge candidates."""
        # Evaluate candidates
        best_splat = self.evaluator.evaluate_merge_candidates(
            self.splat_a, self.splat_b, self.merge_candidates, self.registry, self.tokens
        )
        
        # Should return a single splat
        self.assertIsInstance(best_splat, Splat)
        
        # Should select one of the candidates
        found = False
        for candidate in self.merge_candidates:
            if candidate.id == best_splat.id:
                found = True
                break
        self.assertTrue(found, "Selected splat not found in candidates")
    
    def test_evaluate_birth_candidates(self):
        """Test evaluating birth candidates."""
        # Evaluate candidates
        best_splat = self.evaluator.evaluate_birth_candidates(
            self.birth_candidates, self.registry, self.tokens
        )
        
        # Should return a single splat
        self.assertIsInstance(best_splat, Splat)
        
        # Should select one of the candidates
        found = False
        for candidate in self.birth_candidates:
            if candidate.id == best_splat.id:
                found = True
                break
        self.assertTrue(found, "Selected splat not found in candidates")
    
    def test_evaluate_adjust_candidates(self):
        """Test evaluating adjust candidates."""
        # Evaluate candidates
        best_splat = self.evaluator.evaluate_adjust_candidates(
            self.original_splat, self.adjust_candidates, self.registry, self.tokens
        )
        
        # Should return a single splat
        self.assertIsInstance(best_splat, Splat)
        
        # Should select one of the candidates
        found = False
        for candidate in self.adjust_candidates:
            if candidate.id == best_splat.id:
                found = True
                break
        self.assertTrue(found, "Selected splat not found in candidates")
    
    def test_compute_coverage_score(self):
        """Test computing coverage score for mitosis candidates."""
        # This tests a private method, but it's important functionality
        score = self.evaluator._compute_coverage_score(
            self.original_splat, 
            self.mitosis_candidates[0][0], 
            self.mitosis_candidates[0][1],
            self.tokens
        )
        
        # Score should be a float between 0 and 1
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test without tokens
        score = self.evaluator._compute_coverage_score(
            self.original_splat, 
            self.mitosis_candidates[0][0], 
            self.mitosis_candidates[0][1],
            None
        )
        
        # Score should still be a float between 0 and 1
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
