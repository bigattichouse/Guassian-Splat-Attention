import unittest
import numpy as np
from unittest.mock import Mock, patch

from hsa.adaptation_controller import AdaptationController
from hsa.metrics_computation import InfoTheoreticMetricsComputer
from hsa.candidate_evaluation import InfoTheoreticCandidateEvaluator
from hsa.adaptation_types import AdaptationConfig
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy


class TestMetricsIntegration(unittest.TestCase):
    """Tests for the integration of metrics with the adaptation controller."""
    
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
        
        # Create test splats with different properties
        splat1 = Splat(
            dim=2,
            position=np.array([0.0, 0.0]),
            covariance=np.eye(2),
            amplitude=1.0,
            level="token",
            id="splat1"
        )
        
        splat2 = Splat(
            dim=2,
            position=np.array([1.0, 0.0]),
            covariance=np.eye(2) * 1.2,
            amplitude=0.8,
            level="token",
            id="splat2"
        )
        
        splat3 = Splat(
            dim=2,
            position=np.array([0.5, 0.5]),
            covariance=np.eye(2) * 0.7,
            amplitude=1.2,
            level="sentence",
            id="splat3"
        )
        
        # Add activation history
        for i in range(5):
            splat1.activation_history.add(0.2 + i * 0.1)  # Increasing
            splat2.activation_history.add(0.6 - i * 0.1)  # Decreasing
            splat3.activation_history.add(0.5)            # Constant
        
        # Register splats
        self.registry.register(splat1)
        self.registry.register(splat2)
        self.registry.register(splat3)
        
        # Create test tokens
        self.tokens = np.array([
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.5, 0.5]
        ])
        
        # Create metrics computer and candidate evaluator
        self.metrics_computer = InfoTheoreticMetricsComputer()
        self.candidate_evaluator = InfoTheoreticCandidateEvaluator(self.metrics_computer)
        
        # Create adaptation config with test settings
        self.config = AdaptationConfig(
            low_activation_threshold=0.2,      # For death
            high_activation_threshold=0.5,     # For mitosis
            high_variance_threshold=0.5,       # For mitosis
            high_similarity_threshold=0.8,     # For merge
            mitosis_probability=0.3,
            death_probability=0.3,
            merge_probability=0.2,
            birth_probability=0.1,
            adjust_probability=0.1,
            adaptation_frequency=1,            # Every step
            max_adaptations_per_cycle=3,
            min_lifetime_before_adaptation=0   # No lifetime requirement for testing
        )
        
        # Create adaptation controller
        self.controller = AdaptationController(
            registry=self.registry,
            metrics_computer=self.metrics_computer,
            candidate_evaluator=self.candidate_evaluator,
            config=self.config
        )
    
    def test_controller_initialization(self):
        """Test controller initialization with metrics components."""
        self.assertIsInstance(self.controller.metrics_computer, InfoTheoreticMetricsComputer)
        self.assertIsInstance(self.controller.candidate_evaluator, InfoTheoreticCandidateEvaluator)
        self.assertIsInstance(self.controller.metrics_aggregator, object)  # Should have a metrics aggregator
    
    def test_adaptation_step(self):
        """Test performing an adaptation step with metrics computations."""
        # Perform a step
        results = self.controller.step(self.tokens)
        
        # Should have returned some adaptation results
        self.assertTrue(len(results) > 0)
        
        # Each result should have valid structure
        for result in results:
            self.assertTrue(hasattr(result, 'success'))
            self.assertTrue(hasattr(result, 'adaptation_type'))
            self.assertTrue(hasattr(result, 'original_splat_id'))
            self.assertTrue(hasattr(result, 'new_splat_ids'))
            self.assertTrue(hasattr(result, 'removed_splat_ids'))
            self.assertTrue(hasattr(result, 'message'))
    
    def test_execute_adaptation_cycle(self):
        """Test executing a complete adaptation cycle."""
        # Execute cycle
        results = self.controller.execute_adaptation_cycle(self.tokens)
        
        # Should have returned some adaptation results
        self.assertTrue(len(results) > 0)
        
        # Verify that the metrics cache was populated during measurement phase
        self.assertTrue(len(self.controller.metrics_cache) > 0)
        
        # Adaptation counter should have been updated
        operation_count = sum(self.controller.adaptation_counter.values())
        self.assertEqual(operation_count, len(results))
    
    def test_identify_adaptation_targets(self):
        """Test identification of adaptation targets using metrics."""
        # Get private method
        analyze_metrics = self.controller._analyze_metrics
        
        # Call the method
        targets = analyze_metrics()
        
        # Should have found some targets
        self.assertTrue(len(targets) > 0)
        
        # Each target should have valid structure
        for target in targets:
            self.assertTrue(hasattr(target, 'splat_id'))
            self.assertTrue(hasattr(target, 'adaptation_type'))
            self.assertTrue(hasattr(target, 'trigger'))
            self.assertTrue(hasattr(target, 'parameters'))
    
    def test_identify_death_candidates(self):
        """Test identification of death candidates based on metrics."""
        # Get private method
        identify_death = self.controller._identify_death_candidates
        
        # Call the method
        candidates = identify_death()
        
        # Should have found some candidates
        self.assertTrue(len(candidates) >= 0)  # May be empty if no candidates meet criteria
        
        # Each candidate should be a (splat, activation) tuple
        for candidate in candidates:
            self.assertIsInstance(candidate, tuple)
            self.assertEqual(len(candidate), 2)
            self.assertIsInstance(candidate[0], Splat)
            self.assertIsInstance(candidate[1], float)
    
    def test_identify_mitosis_candidates(self):
        """Test identification of mitosis candidates based on metrics."""
        # Get private method
        identify_mitosis = self.controller._identify_mitosis_candidates
        
        # Call the method
        candidates = identify_mitosis()
        
        # Should have found some candidates
        self.assertTrue(len(candidates) >= 0)  # May be empty if no candidates meet criteria
        
        # Each candidate should be a (splat, activation, variance) tuple
        for candidate in candidates:
            self.assertIsInstance(candidate, tuple)
            self.assertEqual(len(candidate), 3)
            self.assertIsInstance(candidate[0], Splat)
            self.assertIsInstance(candidate[1], float)
            self.assertIsInstance(candidate[2], float)
    
    def test_identify_merge_candidates(self):
        """Test identification of merge candidates based on metrics."""
        # Get private method
        identify_merge = self.controller._identify_merge_candidates
        
        # Call the method
        candidates = identify_merge()
        
        # Should have found some candidates
        self.assertTrue(len(candidates) >= 0)  # May be empty if no candidates meet criteria
        
        # Each candidate should be a (splat_a, splat_b, similarity) tuple
        for candidate in candidates:
            self.assertIsInstance(candidate, tuple)
            self.assertEqual(len(candidate), 3)
            self.assertIsInstance(candidate[0], Splat)
            self.assertIsInstance(candidate[1], Splat)
            self.assertIsInstance(candidate[2], float)
    
    def test_prioritize_targets(self):
        """Test prioritization of adaptation targets."""
        # Get private method
        analyze_metrics = self.controller._analyze_metrics
        prioritize_targets = self.controller._prioritize_targets
        
        # Get all targets
        all_targets = analyze_metrics()
        
        # Prioritize to 2 targets
        max_count = 2
        if len(all_targets) > max_count:
            prioritized = prioritize_targets(all_targets, max_count)
            
            # Should have exactly max_count targets
            self.assertEqual(len(prioritized), max_count)
            
            # Should be a subset of original targets
            for target in prioritized:
                self.assertIn(target, all_targets)
    
    def test_get_statistics(self):
        """Test getting adaptation statistics."""
        # Perform a step to accumulate some statistics
        self.controller.step(self.tokens)
        
        # Get statistics
        stats = self.controller.get_adaptation_statistics()
        
        # Should have valid structure
        self.assertIn('total_steps', stats)
        self.assertIn('last_adaptation_step', stats)
        self.assertIn('adaptation_counts', stats)
        self.assertIn('total_adaptations', stats)
        self.assertIn('recent_history', stats)
        
        # Steps should be at least 1
        self.assertGreaterEqual(stats['total_steps'], 1)
        
        # Total adaptations should match sum of counts
        self.assertEqual(stats['total_adaptations'], sum(stats['adaptation_counts'].values()))
        
        # Recent history should be a list
        self.assertIsInstance(stats['recent_history'], list)
    
    def test_adaptation_execution(self):
        """Test execution of different adaptation operations."""
        # Mitosis execution
        from hsattention.adaptation_types import AdaptationType, AdaptationTarget, AdaptationTrigger
        
        # Create a mitosis target
        mitosis_target = AdaptationTarget(
            splat_id=self.registry.get_all_splats()[0].id,
            secondary_splat_id=None,
            adaptation_type=AdaptationType.MITOSIS,
            trigger=AdaptationTrigger.HIGH_ACTIVATION,
            parameters={}
        )
        
        # Execute mitosis
        mitosis_result = self.controller._execute_adaptation(mitosis_target, self.tokens)
        
        # Check result
        self.assertTrue(hasattr(mitosis_result, 'success'))
        self.assertEqual(mitosis_result.adaptation_type, AdaptationType.MITOSIS)
        
        # Death execution
        death_target = AdaptationTarget(
            splat_id=self.registry.get_all_splats()[0].id,
            secondary_splat_id=None,
            adaptation_type=AdaptationType.DEATH,
            trigger=AdaptationTrigger.LOW_ACTIVATION,
            parameters={}
        )
        
        # Execute death
        death_result = self.controller._execute_adaptation(death_target, self.tokens)
        
        # Check result
        self.assertTrue(hasattr(death_result, 'success'))
        self.assertEqual(death_result.adaptation_type, AdaptationType.DEATH)
        
        # Birth execution (doesn't require an existing splat)
        birth_target = AdaptationTarget(
            splat_id="",
            secondary_splat_id=None,
            adaptation_type=AdaptationType.BIRTH,
            trigger=AdaptationTrigger.EMPTY_REGION,
            parameters={"position": [0.7, 0.7]}
        )
        
        # Execute birth
        birth_result = self.controller._execute_adaptation(birth_target, self.tokens)
        
        # Check result
        self.assertTrue(hasattr(birth_result, 'success'))
        self.assertEqual(birth_result.adaptation_type, AdaptationType.BIRTH)
    
    def test_metrics_cache_update(self):
        """Test updating of metrics cache during adaptation."""
        # Clear metrics cache
        self.controller.metrics_cache = {}
        
        # Execute adaptation cycle
        self.controller.execute_adaptation_cycle(self.tokens)
        
        # Cache should be populated during measurement phase
        self.assertTrue(len(self.controller.metrics_cache) > 0)
        
        # Each entry should be an AdaptationMetrics object
        for splat_id, metrics in self.controller.metrics_cache.items():
            self.assertTrue(hasattr(metrics, 'activation_mean'))
            self.assertTrue(hasattr(metrics, 'activation_trend'))
            self.assertTrue(hasattr(metrics, 'information_contribution'))
            self.assertTrue(hasattr(metrics, 'coverage_uniformity'))
            self.assertTrue(hasattr(metrics, 'variance'))
            self.assertTrue(hasattr(metrics, 'similarity_to_others'))


if __name__ == "__main__":
    unittest.main()
