"""
Pytest tests for Hierarchical Splat Attention (HSA) adaptation mechanisms.

This module tests the adaptation controller and the four core adaptation operations:
birth, death, mitosis, and merge.
"""

import pytest
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

# Import HSA modules
from hsa.splat import Splat
from hsa.hierarchy import Hierarchy
from hsa.registry import SplatRegistry
from hsa.adaptation_types import AdaptationType, AdaptationConfig, AdaptationMetrics
from hsa.adaptation_metrics_base import AdaptationMetricsComputer, SplatCandidateEvaluator
from hsa.adaptation_controller import AdaptationController
from hsa import mitosis, birth, death, merge


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleMetricsComputer(AdaptationMetricsComputer):
    """Simple implementation of the AdaptationMetricsComputer interface."""
    
    def compute_metrics(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> AdaptationMetrics:
        """Compute adaptation metrics for a splat."""
        activation = self.compute_splat_activation(splat, tokens)
        activation_trend = self.compute_activation_trend(splat)
        variance = self.compute_splat_variance(splat, tokens)
        coverage = self.compute_coverage_uniformity(splat, registry, tokens)
        info_contribution = self.compute_information_contribution(splat, registry, tokens)
        
        # Calculate similarities to other splats
        similarities = {}
        for other_splat in registry.get_all_splats():
            if other_splat.id != splat.id:
                similarities[other_splat.id] = self.compute_similarity(splat, other_splat)
        
        return AdaptationMetrics(
            activation_mean=activation,
            activation_trend=activation_trend,
            information_contribution=info_contribution,
            coverage_uniformity=coverage,
            variance=variance,
            similarity_to_others=similarities
        )
    
    def compute_splat_activation(
        self,
        splat: Splat,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        """Compute activation metric for a splat."""
        # Use average activation from history
        return splat.get_average_activation()
    
    def compute_activation_trend(
        self,
        splat: Splat
    ) -> float:
        """Compute activation trend over time."""
        # Simple approximation: constant trend
        return 0.0
    
    def compute_splat_variance(
        self,
        splat: Splat,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        """Compute internal variance of a splat."""
        # Use trace of covariance matrix as variance measure
        return np.trace(splat.covariance) / splat.dim
    
    def compute_similarity(
        self,
        splat_a: Splat,
        splat_b: Splat
    ) -> float:
        """Compute similarity between two splats."""
        # Simple position-based similarity
        distance = np.linalg.norm(splat_a.position - splat_b.position)
        similarity = np.exp(-distance)
        return similarity
    
    def compute_coverage_uniformity(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        """Compute how uniformly a splat covers its region."""
        # Simple approximation: constant value
        return 0.5
    
    def compute_information_contribution(
        self,
        splat: Splat,
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> float:
        """Compute information-theoretic contribution of a splat."""
        # Simple approximation based on activation
        return splat.get_average_activation()


class SimpleCandidateEvaluator(SplatCandidateEvaluator):
    """Simple implementation of the SplatCandidateEvaluator interface."""
    
    def evaluate_mitosis_candidates(
        self,
        original_splat: Splat,
        candidates: List[Tuple[Splat, Splat]],
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> Tuple[Splat, Splat]:
        """Evaluate candidate splat pairs for mitosis operation."""
        # Simple strategy: pick the candidate with greatest distance between splats
        best_distance = -1
        best_candidate = candidates[0]
        
        for candidate_pair in candidates:
            splat1, splat2 = candidate_pair
            distance = np.linalg.norm(splat1.position - splat2.position)
            
            if distance > best_distance:
                best_distance = distance
                best_candidate = candidate_pair
        
        return best_candidate
    
    def evaluate_merge_candidates(
        self,
        splat_a: Splat,
        splat_b: Splat,
        merge_candidates: List[Splat],
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> Splat:
        """Evaluate candidate splats for merge operation."""
        # Simple strategy: pick the first candidate
        return merge_candidates[0]
    
    def evaluate_birth_candidates(
        self,
        candidates: List[Splat],
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> Splat:
        """Evaluate candidate splats for birth operation."""
        # Simple strategy: pick the candidate with greatest distance from existing splats
        best_min_distance = -1
        best_candidate = candidates[0]
        
        for candidate in candidates:
            min_distance = float('inf')
            
            for existing_splat in registry.get_all_splats():
                distance = np.linalg.norm(candidate.position - existing_splat.position)
                min_distance = min(min_distance, distance)
            
            if min_distance > best_min_distance:
                best_min_distance = min_distance
                best_candidate = candidate
        
        return best_candidate
    
    def evaluate_adjust_candidates(
        self,
        original_splat: Splat,
        candidates: List[Splat],
        registry: SplatRegistry,
        tokens: Optional[np.ndarray] = None
    ) -> Splat:
        """Evaluate candidate parameter adjustments."""
        # Simple strategy: pick the first candidate
        return candidates[0]


def simulate_attention(
    registry: SplatRegistry,
    tokens: np.ndarray,
    n_steps: int = 20
) -> None:
    """Simulate attention computation to generate activation data for splats."""
    # This is a simplified simulation to generate activation data
    for _ in range(n_steps):
        # Randomly sample some tokens
        indices = np.random.choice(tokens.shape[0], min(10, tokens.shape[0]), replace=False)
        sampled_tokens = tokens[indices]
        
        # Compute "attention" through each splat
        for splat in registry.get_all_splats():
            for i in range(len(sampled_tokens)):
                for j in range(len(sampled_tokens)):
                    _ = splat.compute_attention(sampled_tokens[i], sampled_tokens[j])


def generate_test_tokens():
    """Generate test token embeddings."""
    # Create tokens with two clusters
    dim = 8
    n_tokens = 100
    
    tokens = np.zeros((n_tokens, dim))
    
    # First cluster
    cluster1_size = n_tokens // 2
    cluster1_center = np.random.normal(0, 1, dim)
    tokens[:cluster1_size] = cluster1_center + np.random.normal(0, 0.5, (cluster1_size, dim))
    
    # Second cluster
    cluster2_size = n_tokens - cluster1_size
    cluster2_center = np.random.normal(0, 1, dim)
    cluster2_center += 3 * (cluster2_center - cluster1_center) / np.linalg.norm(cluster2_center - cluster1_center)
    tokens[cluster1_size:] = cluster2_center + np.random.normal(0, 0.5, (cluster2_size, dim))
    
    return tokens


def initialize_test_environment() -> Tuple[SplatRegistry, AdaptationController, np.ndarray]:
    """Initialize test environment with registry, controller, and tokens."""
    # Create hierarchy
    hierarchy = Hierarchy(
        levels=["token", "phrase", "sentence"],
        init_splats_per_level=[10, 5, 2],
        level_weights=[0.5, 0.3, 0.2]
    )
    
    # Create embedding dimension
    dim = 8
    
    # Create registry
    registry = SplatRegistry(hierarchy, dim)
    
    # Generate test tokens
    tokens = generate_test_tokens()
    
    # Initialize with random splats
    registry.initialize_splats(tokens)
    
    # Create adaptation config
    config = AdaptationConfig(
        low_activation_threshold=0.1,
        high_activation_threshold=0.7,
        high_variance_threshold=0.5,
        high_similarity_threshold=0.8,
        adaptation_frequency=5,  # Set to small value for testing
        max_adaptations_per_cycle=3
    )
    
    # Create metrics computer and candidate evaluator
    metrics_computer = SimpleMetricsComputer()
    candidate_evaluator = SimpleCandidateEvaluator()
    
    # Create adaptation controller
    controller = AdaptationController(
        registry=registry,
        metrics_computer=metrics_computer,
        candidate_evaluator=candidate_evaluator,
        config=config
    )
    
    return registry, controller, tokens


def test_adaptation_cycle():
    """Test a full adaptation cycle."""
    # Initialize test environment
    registry, controller, tokens = initialize_test_environment()
    
    # Simulate attention computation to generate activation data
    simulate_attention(registry, tokens)
    
    # Get initial splat count
    initial_count = len(registry.get_all_splats())
    
    # Run adaptation cycle
    results = controller.execute_adaptation_cycle(tokens)
    
    # Check that adaptations were performed
    assert len(results) > 0, "No adaptations were performed"
    
    # Check that overall system integrity is maintained
    assert registry.verify_integrity(), "Registry integrity was broken by adaptation"
    
    # Log adaptation results
    for result in results:
        logger.info(
            f"Adaptation: {result.adaptation_type.name}, " +
            f"Success: {result.success}, " +
            f"Message: {result.message}"
        )
    
    # Get final splat count
    final_count = len(registry.get_all_splats())
    logger.info(f"Initial splat count: {initial_count}, Final splat count: {final_count}")


def test_specific_adaptation():
    """Test each adaptation type individually."""
    # Initialize test environment
    registry, _, tokens = initialize_test_environment()
    
    # Get initial splat count
    initial_count = len(registry.get_all_splats())
    
    # Test mitosis
    all_splats = registry.get_all_splats()
    if all_splats:
        target_splat = all_splats[0]
        logger.info(f"Testing mitosis on splat {target_splat.id}")
        mitosis_result = mitosis.perform_mitosis(registry, target_splat.id)
        assert mitosis_result is not None, "Mitosis failed"
        assert len(mitosis_result) == 2, "Mitosis should produce exactly 2 new splats"
        assert registry.verify_integrity(), "Registry integrity was broken by mitosis"
    
    # Test birth
    logger.info("Testing birth operation")
    birth_result = birth.perform_birth(
        registry, 
        "token", 
        position=np.random.normal(0, 1, registry.embedding_dim)
    )
    assert birth_result is not None, "Birth failed"
    assert registry.verify_integrity(), "Registry integrity was broken by birth"
    
    # Ensure we have enough splats for remaining tests
    assert len(registry.get_all_splats()) >= 2, "Not enough splats for testing"
    
    # Test merge
    all_splats = registry.get_all_splats()
    splat_a = all_splats[0]
    splat_b = all_splats[1]
    logger.info(f"Testing merge on splats {splat_a.id} and {splat_b.id}")
    merge_result = merge.perform_merge(registry, splat_a.id, splat_b.id)
    assert merge_result is not None, "Merge failed"
    assert registry.verify_integrity(), "Registry integrity was broken by merge"
    
    # Test death
    all_splats = registry.get_all_splats()
    if all_splats:
        target_splat = all_splats[-1]
        logger.info(f"Testing death on splat {target_splat.id}")
        death_result = death.perform_death(registry, target_splat.id)
        assert death_result is True, "Death failed"
        assert registry.verify_integrity(), "Registry integrity was broken by death"
    
    # Get final splat count
    final_count = len(registry.get_all_splats())
    logger.info(f"Initial splat count: {initial_count}, Final splat count: {final_count}")


def test_mitosis_operation():
    """Test mitosis operation in detail."""
    # Initialize test environment
    registry, _, tokens = initialize_test_environment()
    
    # Simulate attention computation to generate activation data
    simulate_attention(registry, tokens)
    
    # Get all splats
    all_splats = registry.get_all_splats()
    if not all_splats:
        pytest.skip("No splats available for testing")
    
    # Choose a splat for mitosis
    target_splat = all_splats[0]
    
    # Generate mitosis candidates
    candidates = mitosis.generate_mitosis_candidates(target_splat)
    assert len(candidates) > 0, "No mitosis candidates generated"
    
    # Perform mitosis
    result = mitosis.perform_mitosis(registry, target_splat.id)
    assert result is not None, "Mitosis failed"
    assert len(result) == 2, "Mitosis should produce exactly 2 new splats"
    
    # Check that original splat was removed
    assert registry.safe_get_splat(target_splat.id) is None, "Original splat was not removed"
    
    # Check that new splats were added
    for splat in result:
        assert registry.safe_get_splat(splat.id) is not None, "New splat was not added"
    
    # Check that registry integrity is maintained
    assert registry.verify_integrity(), "Registry integrity was broken by mitosis"


def test_birth_operation():
    """Test birth operation in detail."""
    # Initialize test environment
    registry, _, tokens = initialize_test_environment()
    
    # Find empty region
    empty_regions = birth.identify_empty_regions(registry, tokens)
    assert len(empty_regions) > 0, "No empty regions identified"
    
    # Generate birth candidates
    candidates = birth.generate_birth_candidates(
        registry,
        "token",
        position=empty_regions[0],
        tokens=tokens
    )
    assert len(candidates) > 0, "No birth candidates generated"
    
    # Perform birth
    result = birth.perform_birth(
        registry,
        "token",
        position=empty_regions[0]
    )
    assert result is not None, "Birth failed"
    
    # Check that new splat was added
    assert registry.safe_get_splat(result.id) is not None, "New splat was not added"
    
    # Check that registry integrity is maintained
    assert registry.verify_integrity(), "Registry integrity was broken by birth"


def test_death_operation():
    """Test death operation in detail."""
    # Initialize test environment
    registry, _, tokens = initialize_test_environment()
    
    # Simulate attention computation to generate activation data
    simulate_attention(registry, tokens)
    
    # Get all splats
    all_splats = registry.get_all_splats()
    if not all_splats:
        pytest.skip("No splats available for testing")
    
    # Choose a splat for death
    target_splat = all_splats[-1]
    
    # Identify death candidates
    candidates = death.identify_death_candidates(registry)
    
    # Perform death
    result = death.perform_death(registry, target_splat.id)
    assert result is True, "Death failed"
    
    # Check that splat was removed
    assert registry.safe_get_splat(target_splat.id) is None, "Splat was not removed"
    
    # Check that registry integrity is maintained
    assert registry.verify_integrity(), "Registry integrity was broken by death"


def test_merge_operation():
    """Test merge operation in detail."""
    # Initialize test environment
    registry, _, tokens = initialize_test_environment()
    
    # Get all splats
    all_splats = registry.get_all_splats()
    if len(all_splats) < 2:
        pytest.skip("Not enough splats available for testing")
    
    # Choose two splats for merging
    splat_a = all_splats[0]
    splat_b = all_splats[1]
    
    # Generate merge candidates
    candidates = merge.generate_merge_candidates(splat_a, splat_b)
    assert len(candidates) > 0, "No merge candidates generated"
    
    # Perform merge
    result = merge.perform_merge(registry, splat_a.id, splat_b.id)
    assert result is not None, "Merge failed"
    
    # Check that original splats were removed
    assert registry.safe_get_splat(splat_a.id) is None, "Original splat A was not removed"
    assert registry.safe_get_splat(splat_b.id) is None, "Original splat B was not removed"
    
    # Check that new splat was added
    assert registry.safe_get_splat(result.id) is not None, "New merged splat was not added"
    
    # Check that registry integrity is maintained
    assert registry.verify_integrity(), "Registry integrity was broken by merge"


def test_controller_initialization():
    """Test initialization of the adaptation controller."""
    # Initialize test environment
    registry, _, _ = initialize_test_environment()
    
    # Create basic components
    metrics_computer = SimpleMetricsComputer()
    candidate_evaluator = SimpleCandidateEvaluator()
    
    # Create with default config
    controller1 = AdaptationController(
        registry=registry,
        metrics_computer=metrics_computer,
        candidate_evaluator=candidate_evaluator
    )
    assert controller1.config is not None, "Default config not created"
    
    # Create with custom config
    custom_config = AdaptationConfig(
        low_activation_threshold=0.05,
        high_activation_threshold=0.8,
        adaptation_frequency=10
    )
    controller2 = AdaptationController(
        registry=registry,
        metrics_computer=metrics_computer,
        candidate_evaluator=candidate_evaluator,
        config=custom_config
    )
    assert controller2.config is custom_config, "Custom config not used"
    assert controller2.config.low_activation_threshold == 0.05, "Config value not set correctly"


def test_controller_step_function():
    """Test the step function of the adaptation controller."""
    # Initialize test environment
    registry, controller, tokens = initialize_test_environment()
    
    # Initial state
    initial_step = controller.step_counter
    initial_last_adaptation = controller.last_adaptation_step
    
    # Take a step (should not perform adaptation yet due to frequency)
    results = controller.step(tokens)
    
    # Check that step counter was incremented
    assert controller.step_counter == initial_step + 1, "Step counter not incremented"
    
    # Should not have performed adaptation yet
    assert len(results) == 0, "Adaptation was performed too soon"
    assert controller.last_adaptation_step == initial_last_adaptation, "Last adaptation step updated incorrectly"
    
    # Now set step counter to trigger adaptation
    controller.step_counter = initial_step + controller.config.adaptation_frequency - 1
    
    # Simulate attention to generate activation data
    simulate_attention(registry, tokens)
    
    # Take another step (should perform adaptation now)
    results = controller.step(tokens)
    
    # Check that adaptation was performed
    assert len(results) > 0, "Adaptation not performed when expected"
    assert controller.last_adaptation_step == controller.step_counter, "Last adaptation step not updated"


if __name__ == "__main__":
    # Run the tests directly
    test_adaptation_cycle()
    test_specific_adaptation()
    test_mitosis_operation()
    test_birth_operation()
    test_death_operation()
    test_merge_operation()
    test_controller_initialization()
    test_controller_step_function()
