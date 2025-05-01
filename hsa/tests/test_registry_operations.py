"""
Tests for registry operations in Hierarchical Splat Attention (HSA).

This module tests the operations defined in registry_operations.py.
"""

import unittest
import numpy as np
import random
from typing import List, Set, Dict

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Import the modules to test
from hsa.hierarchy import Hierarchy
from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.registry_operations import RegistryOperations


class TestRegistryOperations(unittest.TestCase):
    """Test suite for RegistryOperations."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a test hierarchy
        self.hierarchy = Hierarchy(
            levels=["token", "phrase", "sentence"],
            init_splats_per_level=[10, 5, 2],
            level_weights=[0.5, 0.3, 0.2]
        )
        
        # Create a test registry
        self.registry = SplatRegistry(hierarchy=self.hierarchy, embedding_dim=4)
        
        # Create some test splats
        self.splats = []
        self.create_test_splats()
        
        # Register the splats
        for splat in self.splats:
            self.registry.register(splat)
    
    def create_test_splats(self):
        """Create test splats for use in tests."""
        # Create token-level splats
        for i in range(10):
            position = np.random.randn(4)
            covariance = np.eye(4) + 0.1 * np.random.randn(4, 4)
            covariance = 0.5 * (covariance + covariance.T)  # Make symmetric
            
            splat = Splat(
                dim=4,
                position=position,
                covariance=covariance,
                amplitude=0.5 + 0.5 * np.random.random(),
                level="token",
                id=f"token_{i}"
            )
            
            # Set some activation history
            for _ in range(5):
                splat.activation_history.add(np.random.random())
            
            self.splats.append(splat)
        
        # Create phrase-level splats
        for i in range(5):
            position = np.random.randn(4)
            covariance = np.eye(4) + 0.2 * np.random.randn(4, 4)
            covariance = 0.5 * (covariance + covariance.T)  # Make symmetric
            
            splat = Splat(
                dim=4,
                position=position,
                covariance=covariance,
                amplitude=0.5 + 0.5 * np.random.random(),
                level="phrase",
                id=f"phrase_{i}"
            )
            
            # Set some activation history
            for _ in range(5):
                splat.activation_history.add(np.random.random())
            
            self.splats.append(splat)
        
        # Create sentence-level splats
        for i in range(2):
            position = np.random.randn(4)
            covariance = np.eye(4) + 0.3 * np.random.randn(4, 4)
            covariance = 0.5 * (covariance + covariance.T)  # Make symmetric
            
            splat = Splat(
                dim=4,
                position=position,
                covariance=covariance,
                amplitude=0.5 + 0.5 * np.random.random(),
                level="sentence",
                id=f"sentence_{i}"
            )
            
            # Set some activation history
            for _ in range(5):
                splat.activation_history.add(np.random.random())
            
            self.splats.append(splat)
        
        # Establish some parent-child relationships
        # Make phrase splats parents of token splats
        for i, token_splat in enumerate([s for s in self.splats if s.level == "token"]):
            phrase_idx = i % 5
            phrase_splat = [s for s in self.splats if s.level == "phrase"][phrase_idx]
            token_splat.parent = phrase_splat
            phrase_splat.children.add(token_splat)
        
        # Make sentence splats parents of phrase splats
        for i, phrase_splat in enumerate([s for s in self.splats if s.level == "phrase"]):
            sentence_idx = i % 2
            sentence_splat = [s for s in self.splats if s.level == "sentence"][sentence_idx]
            phrase_splat.parent = sentence_splat
            sentence_splat.children.add(phrase_splat)
    
    def test_batch_register(self):
        """Test batch registration of splats."""
        # Create a new registry
        registry = SplatRegistry(hierarchy=self.hierarchy, embedding_dim=4)
        
        # Create test splats
        test_splats = []
        for i in range(5):
            splat = Splat(
                dim=4,
                position=np.random.randn(4),
                covariance=np.eye(4),
                level="token",
                id=f"test_batch_{i}"
            )
            test_splats.append(splat)
        
        # Add one invalid splat
        invalid_splat = Splat(
            dim=4,
            position=np.random.randn(4),
            covariance=np.eye(4),
            level="invalid_level",
            id="invalid_splat"
        )
        test_splats.append(invalid_splat)
        
        # Batch register
        success_count = RegistryOperations.batch_register(registry, test_splats)
        
        # Check results
        self.assertEqual(success_count, 5)  # Should register 5 valid splats
        self.assertEqual(registry.count_splats(), 5)
        self.assertEqual(registry.count_splats("token"), 5)
    
    def test_batch_unregister(self):
        """Test batch unregistration of splats."""
        # Get token-level splats
        token_splats = [s for s in self.splats if s.level == "token"][:5]
        
        # Add one invalid ID
        splat_ids = [s.id for s in token_splats]
        splat_ids.append("non_existent_splat")
        
        # Batch unregister
        success_count = RegistryOperations.batch_unregister(self.registry, splat_ids)
        
        # Check results
        self.assertEqual(success_count, 5)  # Should unregister 5 valid splats
        self.assertEqual(self.registry.count_splats("token"), 5)  # 5 of 10 should remain
    
    def test_filter_splats_by_level(self):
        """Test filtering splats by level."""
        # Filter by level
        token_splats = RegistryOperations.filter_splats_by_level(self.registry, "token")
        phrase_splats = RegistryOperations.filter_splats_by_level(self.registry, "phrase")
        sentence_splats = RegistryOperations.filter_splats_by_level(self.registry, "sentence")
        
        # Check results
        self.assertEqual(len(token_splats), 10)
        self.assertEqual(len(phrase_splats), 5)
        self.assertEqual(len(sentence_splats), 2)
        
        # Check splat levels
        for splat in token_splats:
            self.assertEqual(splat.level, "token")
        for splat in phrase_splats:
            self.assertEqual(splat.level, "phrase")
        for splat in sentence_splats:
            self.assertEqual(splat.level, "sentence")
    
    def test_filter_splats_by_activation(self):
        """Test filtering splats by activation."""
        # Find the activation range for better testing
        min_act = min(s.get_average_activation() for s in self.splats)
        max_act = max(s.get_average_activation() for s in self.splats)
        mid_act = (min_act + max_act) / 2
        
        # Filter by activation
        low_activated = RegistryOperations.filter_splats_by_activation(
            self.registry, min_activation=0, max_activation=mid_act
        )
        high_activated = RegistryOperations.filter_splats_by_activation(
            self.registry, min_activation=mid_act, max_activation=1.0
        )
        
        # Check results
        self.assertGreater(len(low_activated), 0)
        self.assertGreater(len(high_activated), 0)
        self.assertEqual(len(low_activated) + len(high_activated), len(self.splats))
        
        # Check activations
        for splat in low_activated:
            self.assertLessEqual(splat.get_average_activation(), mid_act)
        for splat in high_activated:
            self.assertGreaterEqual(splat.get_average_activation(), mid_act)
    
    def test_filter_splats_by_lifetime(self):
        """Test filtering splats by lifetime."""
        # Update some splat lifetimes
        for i, splat in enumerate(self.splats):
            splat.lifetime = i % 5
        
        # Filter by lifetime
        young_splats = RegistryOperations.filter_splats_by_lifetime(
            self.registry, min_lifetime=0, max_lifetime=2
        )
        old_splats = RegistryOperations.filter_splats_by_lifetime(
            self.registry, min_lifetime=3
        )
        
        # Check results
        self.assertEqual(len(young_splats) + len(old_splats), len(self.splats))
        
        # Check lifetimes
        for splat in young_splats:
            self.assertLessEqual(splat.lifetime, 2)
        for splat in old_splats:
            self.assertGreaterEqual(splat.lifetime, 3)
    
    def test_filter_splats_by_position(self):
        """Test filtering splats by position."""
        # Choose a center position near some splats
        center = np.zeros(4)
        
        # Filter by position with small radius
        small_radius = 1.0
        nearby_splats = RegistryOperations.filter_splats_by_position(
            self.registry, center, small_radius
        )
        
        # Filter by position with large radius
        large_radius = 10.0
        all_splats = RegistryOperations.filter_splats_by_position(
            self.registry, center, large_radius
        )
        
        # Check results
        self.assertLessEqual(len(nearby_splats), len(all_splats))
        self.assertEqual(len(all_splats), len(self.splats))
        
        # Check distances
        for splat in nearby_splats:
            distance = np.linalg.norm(splat.position - center)
            self.assertLessEqual(distance, small_radius)
    
    def test_filter_splats_by_custom(self):
        """Test filtering splats by custom predicate."""
        # Define a custom predicate function
        def is_even_id(splat):
            if splat.id.startswith("token_"):
                num = int(splat.id.split("_")[1])
                return num % 2 == 0
            return False
        
        # Filter by custom predicate
        even_token_splats = RegistryOperations.filter_splats_by_custom(
            self.registry, is_even_id
        )
        
        # Check results
        self.assertEqual(len(even_token_splats), 5)  # token_0, token_2, token_4, token_6, token_8
        
        # Check IDs
        for splat in even_token_splats:
            self.assertTrue(splat.id.startswith("token_"))
            num = int(splat.id.split("_")[1])
            self.assertEqual(num % 2, 0)
    
    def test_select_random_splats(self):
        """Test random selection of splats."""
        # Select random splats
        count = 3
        random_splats = RegistryOperations.select_random_splats(self.registry, count)
        
        # Check count
        self.assertEqual(len(random_splats), count)
        
        # Check uniqueness
        unique_ids = set(splat.id for splat in random_splats)
        self.assertEqual(len(unique_ids), count)
        
        # Select random splats from a specific level
        random_token_splats = RegistryOperations.select_random_splats(
            self.registry, count, level="token"
        )
        
        # Check level
        for splat in random_token_splats:
            self.assertEqual(splat.level, "token")
    
    def test_select_top_splats_by_activation(self):
        """Test selection of top splats by activation."""
        # Select top splats
        count = 3
        top_splats = RegistryOperations.select_top_splats_by_activation(
            self.registry, count
        )
        
        # Check count
        self.assertEqual(len(top_splats), count)
        
        # Check ordering
        for i in range(1, len(top_splats)):
            self.assertGreaterEqual(
                top_splats[i-1].get_average_activation(),
                top_splats[i].get_average_activation()
            )
        
        # Select top splats from a specific level
        top_token_splats = RegistryOperations.select_top_splats_by_activation(
            self.registry, count, level="token"
        )
        
        # Check level
        for splat in top_token_splats:
            self.assertEqual(splat.level, "token")
    
    def test_select_splats_near_position(self):
        """Test selection of splats near a position."""
        # Choose a center position
        center = np.zeros(4)
        
        # Select nearest splats
        count = 5
        nearest_splats = RegistryOperations.select_splats_near_position(
            self.registry, center, count
        )
        
        # Check count
        self.assertEqual(len(nearest_splats), count)
        
        # Check ordering
        for i in range(1, len(nearest_splats)):
            dist_prev = np.linalg.norm(nearest_splats[i-1].position - center)
            dist_curr = np.linalg.norm(nearest_splats[i].position - center)
            self.assertLessEqual(dist_prev, dist_curr)
        
        # Select nearest splats from a specific level
        nearest_token_splats = RegistryOperations.select_splats_near_position(
            self.registry, center, count, level="token"
        )
        
        # Check level
        for splat in nearest_token_splats:
            self.assertEqual(splat.level, "token")
    
    def test_reorganize_hierarchy(self):
        """Test reorganizing registry with a new hierarchy."""
        # Create a new hierarchy with different levels
        new_hierarchy = Hierarchy(
            levels=["word", "phrase", "paragraph"],
            init_splats_per_level=[15, 7, 3],
            level_weights=[0.4, 0.4, 0.2]
        )
        
        # Get a count of splats at each level before reorganization
        original_token_count = self.registry.count_splats("token")
        original_phrase_count = self.registry.count_splats("phrase")
        original_sentence_count = self.registry.count_splats("sentence")
        original_total = original_token_count + original_phrase_count + original_sentence_count
        
        # Reorganize registry
        success = RegistryOperations.reorganize_hierarchy(self.registry, new_hierarchy)
        
        # Check result
        self.assertTrue(success)
        self.assertEqual(self.registry.hierarchy, new_hierarchy)
        
        # Check that levels were properly mapped
        # token -> word
        word_count = self.registry.count_splats("word")
        self.assertEqual(word_count, original_token_count)
        
        # phrase -> phrase
        new_phrase_count = self.registry.count_splats("phrase")
        self.assertEqual(new_phrase_count, original_phrase_count)
        
        # sentence -> paragraph
        paragraph_count = self.registry.count_splats("paragraph")
        self.assertEqual(paragraph_count, original_sentence_count)
        
        # Check total count remains the same
        new_total = word_count + new_phrase_count + paragraph_count
        self.assertEqual(new_total, original_total)
        
        # Verify that all splats have valid levels
        for splat in self.registry.get_all_splats():
            self.assertIn(splat.level, new_hierarchy.levels)
    
    def test_redistribution_strategy_evenly(self):
        """Test even redistribution strategy."""
        # Get all splats
        all_splats = self.registry.get_all_splats()
        
        # Define target levels and counts
        target_levels = ["token", "phrase", "sentence"]
        target_counts = [8, 6, 3]
        
        # Apply redistribution strategy
        distribution = RegistryOperations.redistribution_strategy_evenly(
            self.registry, all_splats, target_levels, target_counts
        )
        
        # Check level counts
        for level, count in zip(target_levels, target_counts):
            self.assertEqual(len(distribution[level]), count)
        
        # Check total count
        total_count = sum(len(splats) for splats in distribution.values())
        self.assertEqual(total_count, len(all_splats))
    
    def test_redistribute_splats(self):
        """Test redistribution of splats across levels."""
        # Get all splats
        all_splats = self.registry.get_all_splats()
        
        # Define target levels and counts
        target_levels = ["token", "phrase", "sentence"]
        target_counts = [8, 6, 3]
        
        # Redistribute splats
        success = RegistryOperations.redistribute_splats(
            self.registry, all_splats, target_levels, target_counts
        )
        
        # Check result
        self.assertTrue(success)
        
        # Check level counts
        token_count = self.registry.count_splats("token")
        phrase_count = self.registry.count_splats("phrase")
        sentence_count = self.registry.count_splats("sentence")
        
        self.assertEqual(token_count, 8)
        self.assertEqual(phrase_count, 6)
        self.assertEqual(sentence_count, 3)
    
    def test_redistribute_by_position_clustering(self):
        """Test position-based redistribution."""
        # Define target levels and counts
        target_levels = ["token", "phrase", "sentence"]
        target_counts = [8, 6, 3]
        
        # Redistribute by position clustering
        success = RegistryOperations.redistribute_by_position_clustering(
            self.registry, target_levels, target_counts
        )
        
        # Check result
        self.assertTrue(success)
        
        # Check level counts
        token_count = self.registry.count_splats("token")
        phrase_count = self.registry.count_splats("phrase")
        sentence_count = self.registry.count_splats("sentence")
        
        self.assertEqual(token_count, 8)
        self.assertEqual(phrase_count, 6)
        self.assertEqual(sentence_count, 3)
    
    def test_balance_levels(self):
        """Test balancing levels according to hierarchy settings."""
        # Create an imbalanced registry
        imbalanced_registry = SplatRegistry(hierarchy=self.hierarchy, embedding_dim=4)
        
        # Add more splats to token level than specified
        for i in range(15):  # 15 > 10 (init_splats_per_level[0])
            splat = Splat(
                dim=4,
                position=np.random.randn(4),
                covariance=np.eye(4),
                level="token",
                id=f"imbalanced_token_{i}"
            )
            imbalanced_registry.register(splat)
        
        # Add fewer splats to phrase level than specified
        for i in range(2):  # 2 < 5 (init_splats_per_level[1])
            splat = Splat(
                dim=4,
                position=np.random.randn(4),
                covariance=np.eye(4),
                level="phrase",
                id=f"imbalanced_phrase_{i}"
            )
            imbalanced_registry.register(splat)
        
        # Balance levels
        success = RegistryOperations.balance_levels(imbalanced_registry)
        
        # Check result
        self.assertTrue(success)
        
        # Check level counts - should be closer to init_splats_per_level
        token_count = imbalanced_registry.count_splats("token")
        phrase_count = imbalanced_registry.count_splats("phrase")
        sentence_count = imbalanced_registry.count_splats("sentence")
        
        # We expect some rebalancing but not exact matches
        self.assertLessEqual(token_count, 15)  # Should decrease
        self.assertGreaterEqual(phrase_count, 2)  # Should increase or stay the same
    
    def test_merge_registries(self):
        """Test merging two registries."""
        # Create a second registry with different hierarchy
        hierarchy2 = Hierarchy(
            levels=["character", "word", "line"],
            init_splats_per_level=[8, 4, 2],
            level_weights=[0.4, 0.4, 0.2]
        )
        registry2 = SplatRegistry(hierarchy=hierarchy2, embedding_dim=4)
        
        # Add some splats
        for i in range(5):
            splat = Splat(
                dim=4,
                position=np.random.randn(4),
                covariance=np.eye(4),
                level="character",
                id=f"character_{i}"
            )
            registry2.register(splat)
        
        # Merge registries
        merged_registry = RegistryOperations.merge_registries(
            self.registry, registry2, strategy="keep_a"
        )
        
        # Check merged hierarchy
        merged_levels = merged_registry.hierarchy.levels
        for level in self.hierarchy.levels:
            self.assertIn(level, merged_levels)
        for level in hierarchy2.levels:
            self.assertIn(level, merged_levels)
        
        # Check splat count
        orig_count = self.registry.count_splats() + registry2.count_splats()
        merged_count = merged_registry.count_splats()
        self.assertEqual(merged_count, orig_count)
    
    def test_export_import(self):
        """Test export and import of registry."""
        # Export to dictionary
        exported = RegistryOperations.export_to_dict(self.registry)
        
        # Check exported data
        self.assertEqual(exported["embedding_dim"], 4)
        self.assertIn("hierarchy", exported)
        self.assertIn("splats", exported)
        self.assertEqual(len(exported["splats"]), len(self.splats))
        
        # Import from dictionary
        imported_registry = RegistryOperations.import_from_dict(exported)
        
        # Check imported registry
        self.assertEqual(imported_registry.embedding_dim, self.registry.embedding_dim)
        self.assertEqual(imported_registry.count_splats(), self.registry.count_splats())
        
        # Check hierarchy
        for level in self.hierarchy.levels:
            self.assertIn(level, imported_registry.hierarchy.levels)
            count1 = self.registry.count_splats(level)
            count2 = imported_registry.count_splats(level)
            self.assertEqual(count1, count2)
        
        # Check parent-child relationships
        for splat in imported_registry.get_all_splats():
            if splat.parent is not None:
                # Check if parent contains this splat in its children set
                self.assertIn(splat, splat.parent.children)


if __name__ == '__main__':
    unittest.main()
