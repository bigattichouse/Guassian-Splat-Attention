import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Import the modules we need to test
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy
from hsa import merge


class TestMergeAdaptation(unittest.TestCase):
    """Tests for merge adaptation operations in HSA."""

    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a test hierarchy
        self.hierarchy = Hierarchy(
            levels=["token", "phrase", "document"],
            init_splats_per_level=[5, 3, 1],
            level_weights=[0.5, 0.3, 0.2]
        )
        
        # Create a test registry
        self.registry = SplatRegistry(self.hierarchy, embedding_dim=2)
        
        # Create test splats
        # Two similar splats
        self.splat_a = Splat(
            dim=2, 
            position=np.array([1.0, 0.0]), 
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]), 
            amplitude=1.0,
            level="token",
            id="splat_a"
        )
        
        self.splat_b = Splat(
            dim=2, 
            position=np.array([1.2, 0.1]), 
            covariance=np.array([[1.0, 0.0], [0.0, 1.0]]), 
            amplitude=0.9,
            level="token",
            id="splat_b"
        )
        
        # A dissimilar splat
        self.splat_c = Splat(
            dim=2, 
            position=np.array([-2.0, 3.0]), 
            covariance=np.array([[0.5, 0.0], [0.0, 0.5]]), 
            amplitude=1.0,
            level="token",
            id="splat_c"
        )
        
        # A splat at a different level
        self.splat_d = Splat(
            dim=2, 
            position=np.array([1.1, 0.0]), 
            covariance=np.array([[2.0, 0.0], [0.0, 2.0]]), 
            amplitude=1.0,
            level="phrase",
            id="splat_d"
        )
        
        # Add splats to registry
        self.registry.register(self.splat_a)
        self.registry.register(self.splat_b)
        self.registry.register(self.splat_c)
        self.registry.register(self.splat_d)
        
        # Create child splats for testing hierarchy preservation
        self.child_a = Splat(
            dim=2, 
            position=np.array([0.9, 0.1]), 
            covariance=np.array([[0.5, 0.0], [0.0, 0.5]]), 
            amplitude=1.0,
            level="token",
            parent=self.splat_a,
            id="child_a"
        )
        
        self.child_b = Splat(
            dim=2, 
            position=np.array([1.3, 0.0]), 
            covariance=np.array([[0.5, 0.0], [0.0, 0.5]]), 
            amplitude=1.0,
            level="token",
            parent=self.splat_b,
            id="child_b"
        )
        
        # Update parent-child relationships
        self.splat_a.children.add(self.child_a)
        self.splat_b.children.add(self.child_b)
        
        # Register child splats
        self.registry.register(self.child_a)
        self.registry.register(self.child_b)

    def test_generate_merge_candidates(self):
        """Test generating merge candidates for two splats."""
        # Test with default parameters (3 candidates)
        candidates = merge.generate_merge_candidates(self.splat_a, self.splat_b)
        self.assertEqual(len(candidates), 3)
        
        # Verify all candidates have correct properties
        for candidate in candidates:
            self.assertEqual(candidate.dim, 2)
            self.assertEqual(candidate.level, "token")
            self.assertEqual(candidate.parent, self.splat_a.parent)
            
            # Position should be between the two original splats
            pos_a_dist = np.linalg.norm(candidate.position - self.splat_a.position)
            pos_b_dist = np.linalg.norm(candidate.position - self.splat_b.position)
            dist_between = np.linalg.norm(self.splat_a.position - self.splat_b.position)
            
            # Candidate should be in between or very close to original splats
            self.assertLessEqual(pos_a_dist + pos_b_dist, dist_between * 1.01)
        
        # Test with different number of candidates
        candidates = merge.generate_merge_candidates(self.splat_a, self.splat_b, num_candidates=5)
        self.assertEqual(len(candidates), 5)
        
        # Test with different level splats
        candidates = merge.generate_merge_candidates(self.splat_a, self.splat_d)
        self.assertEqual(len(candidates), 3)
        
        # Verify level selection (should use higher level - phrase)
        self.assertEqual(candidates[0].level, "phrase")
        
        # Test with different dimensions (should return empty list)
        splat_wrong_dim = Splat(
            dim=3, 
            position=np.array([1.0, 0.0, 0.0]), 
            covariance=np.eye(3), 
            amplitude=1.0,
            level="token"
        )
        candidates = merge.generate_merge_candidates(self.splat_a, splat_wrong_dim)
        self.assertEqual(len(candidates), 0)

    def test_calculate_similarity(self):
        """Test calculating similarity between splats."""
        # Test similarity between similar splats
        sim_ab = merge.calculate_similarity(self.splat_a, self.splat_b)
        self.assertGreater(sim_ab, 0.7)  # Should be high similarity
        
        # Test similarity between dissimilar splats
        sim_ac = merge.calculate_similarity(self.splat_a, self.splat_c)
        self.assertLess(sim_ac, 0.3)  # Should be low similarity
        
        # Test identity similarity (should be very high)
        sim_aa = merge.calculate_similarity(self.splat_a, self.splat_a)
        self.assertGreater(sim_aa, 0.9)  # Should be almost 1.0
        
        # Test similarity with different levels
        sim_ad = merge.calculate_similarity(self.splat_a, self.splat_d)
        self.assertLess(sim_ad, sim_ab)  # Should be lower due to level difference
        self.assertGreater(sim_ad, sim_ac)  # But still higher than dissimilar splats
        
        # Test with different dimensions (should return 0)
        splat_wrong_dim = Splat(
            dim=3, 
            position=np.array([1.0, 0.0, 0.0]), 
            covariance=np.eye(3), 
            amplitude=1.0,
            level="token"
        )
        sim_wrong = merge.calculate_similarity(self.splat_a, splat_wrong_dim)
        self.assertEqual(sim_wrong, 0.0)

    def test_find_merge_candidates(self):
        """Test finding candidates for merging."""
        # Test with default threshold
        candidates = merge.find_merge_candidates(self.registry)
        
        # Should find at least the pair (splat_a, splat_b)
        self.assertGreaterEqual(len(candidates), 1)
        
        # Check if the first pair is (splat_a, splat_b)
        found_ab = False
        for splat_1, splat_2, _ in candidates:
            if ((splat_1.id == "splat_a" and splat_2.id == "splat_b") or 
                (splat_1.id == "splat_b" and splat_2.id == "splat_a")):
                found_ab = True
                break
        
        self.assertTrue(found_ab)
        
        # Test with very high threshold (should return empty list)
        high_threshold_candidates = merge.find_merge_candidates(
            self.registry, similarity_threshold=0.99)
        self.assertEqual(len(high_threshold_candidates), 0)
        
        # Test with same_level_only=False (should include cross-level pairs)
        cross_level_candidates = merge.find_merge_candidates(
            self.registry, similarity_threshold=0.5, same_level_only=False)
        
        found_cross_level = False
        for splat_1, splat_2, _ in cross_level_candidates:
            if splat_1.level != splat_2.level:
                found_cross_level = True
                break
        
        self.assertTrue(found_cross_level)
        
        # Test max_candidates parameter
        limited_candidates = merge.find_merge_candidates(
            self.registry, similarity_threshold=0.1, max_candidates=1)
        self.assertEqual(len(limited_candidates), 1)

    def test_perform_merge(self):
        """Test performing a merge operation."""
        # Count splats before merge
        initial_count = len(self.registry.get_all_splats())
        
        # Perform merge
        merged_splat = merge.perform_merge(self.registry, "splat_a", "splat_b")
        
        # Verify merged splat was created
        self.assertIsNotNone(merged_splat)
        
        # Verify original splats were removed
        with self.assertRaises(ValueError):
            self.registry.get_splat("splat_a")
        
        with self.assertRaises(ValueError):
            self.registry.get_splat("splat_b")
        
        # Verify the number of splats decreased by 1 (2 removed, 1 added)
        new_count = len(self.registry.get_all_splats())
        self.assertEqual(new_count, initial_count - 1)
        
        # Verify children were transferred to merged splat
        self.assertEqual(len(merged_splat.children), 2)
        
        # Verify child-parent relationships were updated
        for child_id in ["child_a", "child_b"]:
            child = self.registry.get_splat(child_id)
            self.assertEqual(child.parent, merged_splat)
        
        # Test with non-existent splat (should fail)
        with self.assertLogs(level='ERROR'):
            failed_merge = merge.perform_merge(self.registry, "nonexistent", "splat_c")
            self.assertIsNone(failed_merge)

    def test_get_level_index(self):
        """Test getting level index for hierarchy."""
        # Test known levels
        self.assertEqual(merge.get_level_index("token"), 0)
        self.assertEqual(merge.get_level_index("word"), 1)
        self.assertEqual(merge.get_level_index("phrase"), 2)
        self.assertEqual(merge.get_level_index("sentence"), 3)
        self.assertEqual(merge.get_level_index("document"), 5)
        
        # Test unknown level (should return 0)
        self.assertEqual(merge.get_level_index("unknown_level"), 0)

    def test_calculate_merge_suitability(self):
        """Test calculating merge suitability."""
        # Test suitability for all levels
        suitability = merge.calculate_merge_suitability(self.registry)
        self.assertGreaterEqual(suitability, 0.0)
        self.assertLessEqual(suitability, 1.0)
        
        # Test suitability for specific level
        token_suitability = merge.calculate_merge_suitability(self.registry, "token")
        self.assertGreaterEqual(token_suitability, 0.0)
        self.assertLessEqual(token_suitability, 1.0)
        
        # Test level with no splats
        empty_registry = SplatRegistry(self.hierarchy, embedding_dim=2)
        empty_suitability = merge.calculate_merge_suitability(empty_registry)
        self.assertEqual(empty_suitability, 0.0)

    def test_get_best_merge_strategy(self):
        """Test getting best merge strategy."""
        # Get merge strategies
        strategies = merge.get_best_merge_strategy(self.registry)
        
        # Should have an entry for each level
        self.assertEqual(len(strategies), len(self.hierarchy.levels))
        
        # All values should be between 0 and 1
        for level, score in strategies.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        
        # Token level should have highest suitability
        token_score = strategies["token"]
        for level, score in strategies.items():
            if level != "token":
                self.assertGreaterEqual(token_score, score)


if __name__ == '__main__':
    unittest.main()
