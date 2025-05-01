"""
Tests for the adaptation metrics base interfaces in the HSA implementation.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import Dict, List, Set, Tuple

from hsa.splat import Splat
from hsa.hierarchy import Hierarchy
from hsa.registry import SplatRegistry
from hsa.adaptation_types import AdaptationMetrics
from hsa.adaptation_metrics_base import (
    AdaptationMetricsComputer,
    SplatCandidateEvaluator,
    AdaptationMetricsAggregator
)


# Create a concrete implementation of the abstract class for testing
class MockMetricsComputer(AdaptationMetricsComputer):
    """Mock implementation of AdaptationMetricsComputer for testing."""
    
    def compute_metrics(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: np.ndarray = None
    ) -> AdaptationMetrics:
        """Compute mock metrics."""
        return AdaptationMetrics(
            activation_mean=self.compute_splat_activation(splat, tokens),
            activation_trend=self.compute_activation_trend(splat),
            information_contribution=self.compute_information_contribution(splat, registry, tokens),
            coverage_uniformity=self.compute_coverage_uniformity(splat, registry, tokens),
            variance=self.compute_splat_variance(splat, tokens),
            similarity_to_others={"other_splat": 0.5}
        )
    
    def compute_splat_activation(
        self,
        splat: Splat,
        tokens: np.ndarray = None
    ) -> float:
        """Compute mock activation."""
        return splat.get_average_activation()
    
    def compute_activation_trend(
        self,
        splat: Splat
    ) -> float:
        """Compute mock activation trend."""
        # Simple mock implementation
        history = splat.activation_history.get_values()
        if len(history) < 2:
            return 0.0
        return history[-1] - history[0]
    
    def compute_splat_variance(
        self,
        splat: Splat,
        tokens: np.ndarray = None
    ) -> float:
        """Compute mock variance."""
        # Just return based on covariance determinant
        det = np.linalg.det(splat.covariance)
        return min(1.0, max(0.0, np.log(1 + det) / 10))
    
    def compute_similarity(
        self,
        splat_a: Splat,
        splat_b: Splat
    ) -> float:
        """Compute mock similarity."""
        # Simple distance-based similarity
        dist = np.linalg.norm(splat_a.position - splat_b.position)
        return max(0.0, 1.0 - dist / 5.0)
    
    def compute_coverage_uniformity(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: np.ndarray = None
    ) -> float:
        """Compute mock coverage uniformity."""
        # Simple mock implementation
        return 0.7
    
    def compute_information_contribution(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: np.ndarray = None
    ) -> float:
        """Compute mock information contribution."""
        # Simple mock implementation
        return 0.3 * splat.get_average_activation()


class MockCandidateEvaluator(SplatCandidateEvaluator):
    """Mock implementation of SplatCandidateEvaluator for testing."""
    
    def evaluate_mitosis_candidates(
        self,
        original_splat: Splat,
        candidates: List[Tuple[Splat, Splat]],
        registry: SplatRegistry,
        tokens: np.ndarray = None
    ) -> Tuple[Splat, Splat]:
        """Evaluate mitosis candidates."""
        # Just return the first candidate
        return candidates[0]
    
    def evaluate_merge_candidates(
        self,
        splat_a: Splat,
        splat_b: Splat,
        merge_candidates: List[Splat],
        registry: SplatRegistry,
        tokens: np.ndarray = None
    ) -> Splat:
        """Evaluate merge candidates."""
        # Just return the first candidate
        return merge_candidates[0]
    
    def evaluate_birth_candidates(
        self,
        candidates: List[Splat],
        registry: SplatRegistry,
        tokens: np.ndarray = None
    ) -> Splat:
        """Evaluate birth candidates."""
        # Just return the first candidate
        return candidates[0]
    
    def evaluate_adjust_candidates(
        self,
        original_splat: Splat,
        candidates: List[Splat],
        registry: SplatRegistry,
        tokens: np.ndarray = None
    ) -> Splat:
        """Evaluate adjust candidates."""
        # Just return the first candidate
        return candidates[0]


class TestAdaptationMetricsComputer:
    """Tests for the AdaptationMetricsComputer interface."""
    
    @pytest.fixture
    def metrics_computer(self) -> AdaptationMetricsComputer:
        """Create a mock metrics computer for testing."""
        return MockMetricsComputer()
    
    @pytest.fixture
    def splat(self) -> Splat:
        """Create a splat for testing."""
        splat = Splat(dim=2, position=np.array([1.0, 2.0]), id="test_splat")
        
        # Add some activation history
        token1 = np.array([1.0, 2.0])
        token2 = np.array([2.0, 3.0])
        splat.compute_attention(token1, token1)  # High activation
        splat.compute_attention(token1, token2)  # Lower activation
        
        return splat
    
    @pytest.fixture
    def registry(self) -> SplatRegistry:
        """Create a registry for testing."""
        hierarchy = Hierarchy(levels=["token", "phrase"])
        registry = SplatRegistry(hierarchy=hierarchy, embedding_dim=2)
        return registry
    
    def test_compute_metrics(self, metrics_computer, splat, registry):
        """Test computing metrics."""
        metrics = metrics_computer.compute_metrics(splat, registry)
        
        assert isinstance(metrics, AdaptationMetrics)
        assert 0.0 <= metrics.activation_mean <= 1.0
        assert -1.0 <= metrics.activation_trend <= 1.0
        assert 0.0 <= metrics.information_contribution <= 1.0
        assert 0.0 <= metrics.coverage_uniformity <= 1.0
        assert 0.0 <= metrics.variance <= 1.0
        assert "other_splat" in metrics.similarity_to_others
    
    def test_compute_splat_activation(self, metrics_computer, splat):
        """Test computing splat activation."""
        activation = metrics_computer.compute_splat_activation(splat)
        
        assert 0.0 <= activation <= 1.0
        assert activation == splat.get_average_activation()
    
    def test_compute_activation_trend(self, metrics_computer, splat):
        """Test computing activation trend."""
        trend = metrics_computer.compute_activation_trend(splat)
        
        assert -1.0 <= trend <= 1.0
    
    def test_compute_splat_variance(self, metrics_computer, splat):
        """Test computing splat variance."""
        variance = metrics_computer.compute_splat_variance(splat)
        
        assert 0.0 <= variance <= 1.0
    
    def test_compute_similarity(self, metrics_computer, splat):
        """Test computing similarity."""
        splat_b = Splat(dim=2, position=np.array([2.0, 3.0]), id="other_splat")
        
        similarity = metrics_computer.compute_similarity(splat, splat_b)
        
        assert 0.0 <= similarity <= 1.0
        
        # Similarity to self should be 1.0
        self_similarity = metrics_computer.compute_similarity(splat, splat)
        assert pytest.approx(self_similarity) == 1.0
    
    def test_compute_coverage_uniformity(self, metrics_computer, splat, registry):
        """Test computing coverage uniformity."""
        uniformity = metrics_computer.compute_coverage_uniformity(splat, registry)
        
        assert 0.0 <= uniformity <= 1.0
    
    def test_compute_information_contribution(self, metrics_computer, splat, registry):
        """Test computing information contribution."""
        contribution = metrics_computer.compute_information_contribution(splat, registry)
        
        assert 0.0 <= contribution <= 1.0


class TestSplatCandidateEvaluator:
    """Tests for the SplatCandidateEvaluator interface."""
    
    @pytest.fixture
    def evaluator(self) -> SplatCandidateEvaluator:
        """Create a mock evaluator for testing."""
        return MockCandidateEvaluator()
    
    @pytest.fixture
    def splat(self) -> Splat:
        """Create a splat for testing."""
        return Splat(dim=2, position=np.array([1.0, 2.0]), id="test_splat")
    
    @pytest.fixture
    def registry(self) -> SplatRegistry:
        """Create a registry for testing."""
        hierarchy = Hierarchy(levels=["token", "phrase"])
        registry = SplatRegistry(hierarchy=hierarchy, embedding_dim=2)
        return registry
    
    def test_evaluate_mitosis_candidates(self, evaluator, splat, registry):
        """Test evaluating mitosis candidates."""
        candidates = [
            (
                Splat(dim=2, position=np.array([0.5, 1.5]), id="candidate_1a"),
                Splat(dim=2, position=np.array([1.5, 2.5]), id="candidate_1b")
            ),
            (
                Splat(dim=2, position=np.array([0.0, 1.0]), id="candidate_2a"),
                Splat(dim=2, position=np.array([2.0, 3.0]), id="candidate_2b")
            )
        ]
        
        best = evaluator.evaluate_mitosis_candidates(splat, candidates, registry)
        
        assert isinstance(best, tuple)
        assert len(best) == 2
        assert best[0].id == "candidate_1a"
        assert best[1].id == "candidate_1b"
    
    def test_evaluate_merge_candidates(self, evaluator, splat, registry):
        """Test evaluating merge candidates."""
