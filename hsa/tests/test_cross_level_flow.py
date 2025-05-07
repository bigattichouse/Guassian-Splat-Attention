"""
Unit tests for cross-level information flow in Hierarchical Splat Attention (HSA).

This module tests the functionality of the CrossLevelInfoFlow class, which manages
information flow between hierarchy levels in HSA.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch

from hsa.splat import Splat
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy
from hsa.cross_level_flow import CrossLevelInfoFlow


class TestCrossLevelInfoFlow(unittest.TestCase):
    """Test cases for CrossLevelInfoFlow class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a hierarchy with three levels
        self.hierarchy = Hierarchy(
            levels=["token", "sentence", "document"],
            init_splats_per_level=[10, 5, 2],
            level_weights=[0.5, 0.3, 0.2]
        )
        
        # Create registry
        self.registry = SplatRegistry(
            hierarchy=self.hierarchy, 
            embedding_dim=2
        )
        
        # Create some test splats
        # Document level splats (highest level)
        self.doc_splat1 = Splat(dim=2, position=np.array([0.0, 0.0]), level="document", id="doc1")
        self.doc_splat2 = Splat(dim=2, position=np.array([2.0, 2.0]), level="document", id="doc2")
        
        # Sentence level splats (mid level)
        self.sent_splat1 = Splat(dim=2, position=np.array([0.5, 0.0]), level="sentence", 
                                parent=self.doc_splat1, id="sent1")
        self.sent_splat2 = Splat(dim=2, position=np.array([-0.5, 0.0]), level="sentence", 
                                parent=self.doc_splat1, id="sent2")
        self.sent_splat3 = Splat(dim=2, position=np.array([2.0, 2.5]), level="sentence", 
                                parent=self.doc_splat2, id="sent3")
        
        # Token level splats (lowest level)
        # Ensure token positions are different from sentence positions for test clarity
        self.token_splat1 = Splat(dim=2, position=np.array([0.8, 0.3]), level="token", 
                                parent=self.sent_splat1, id="token1")
        self.token_splat2 = Splat(dim=2, position=np.array([0.2, -0.3]), level="token", 
                                parent=self.sent_splat1, id="token2")
        self.token_splat3 = Splat(dim=2, position=np.array([-0.7, 0.1]), level="token", 
                                parent=self.sent_splat2, id="token3")
        
        # Add splats to registry
        self.registry.register(self.doc_splat1)
        self.registry.register(self.doc_splat2)
        self.registry.register(self.sent_splat1)
        self.registry.register(self.sent_splat2)
        self.registry.register(self.sent_splat3)
        self.registry.register(self.token_splat1)
        self.registry.register(self.token_splat2)
        self.registry.register(self.token_splat3)
        
        # Add to parent's children sets
        self.doc_splat1.children.add(self.sent_splat1)
        self.doc_splat1.children.add(self.sent_splat2)
        self.doc_splat2.children.add(self.sent_splat3)
        self.sent_splat1.children.add(self.token_splat1)
        self.sent_splat1.children.add(self.token_splat2)
        self.sent_splat2.children.add(self.token_splat3)
        
        # Create the CrossLevelInfoFlow instance with slightly stronger flow
        # to make movements more detectable in tests
        self.flow_manager = CrossLevelInfoFlow(
            registry=self.registry,
            top_down_strength=0.8,  # Increased from 0.5
            bottom_up_strength=0.6   # Increased from 0.3
        )
        
        # Create token embeddings for testing
        self.tokens = np.array([
            [0.5, 0.1],
            [0.3, -0.1],
            [1.0, 1.0],
            [2.0, 2.0]
        ])
    
    def test_init(self):
        """Test initialization of CrossLevelInfoFlow."""
        self.assertEqual(self.flow_manager.registry, self.registry)
        self.assertEqual(self.flow_manager.top_down_strength, 0.8)
        self.assertEqual(self.flow_manager.bottom_up_strength, 0.6)
        self.assertIsInstance(self.flow_manager.flow_stats, dict)
    
    def test_enable_top_down_flow(self):
        """Test enabling top-down information flow."""
        result = self.flow_manager.enable_top_down_flow(strength=0.7)
        
        # Check that the method returns the registry
        self.assertEqual(result, self.registry)
        
        # Check that flow stats were updated
        self.assertTrue(self.flow_manager.flow_stats.get("top_down_enabled", False))
        self.assertEqual(self.flow_manager.flow_stats.get("top_down_strength", 0), 0.7)
        
        # Test with default strength
        self.flow_manager.flow_stats.clear()  # Reset stats
        result = self.flow_manager.enable_top_down_flow()
        
        # Check that the method uses the default strength
        self.assertEqual(self.flow_manager.flow_stats.get("top_down_strength", 0), 0.8)
    
    def test_enable_bottom_up_flow(self):
        """Test enabling bottom-up information flow."""
        result = self.flow_manager.enable_bottom_up_flow(strength=0.7)
        
        # Check that the method returns the registry
        self.assertEqual(result, self.registry)
        
        # Check that flow stats were updated
        self.assertTrue(self.flow_manager.flow_stats.get("bottom_up_enabled", False))
        self.assertEqual(self.flow_manager.flow_stats.get("bottom_up_strength", 0), 0.7)
        
        # Test with default strength
        self.flow_manager.flow_stats.clear()  # Reset stats
        result = self.flow_manager.enable_bottom_up_flow()
        
        # Check that the method uses the default strength
        self.assertEqual(self.flow_manager.flow_stats.get("bottom_up_strength", 0), 0.6)
    
    def test_propagate_top_down(self):
        """Test top-down propagation of information."""
        # Store original positions for verification
        original_pos_token1 = self.token_splat1.position.copy()
        original_pos_token2 = self.token_splat2.position.copy()
        
        # Execute top-down propagation
        stats = self.flow_manager.propagate_top_down(tokens=self.tokens)
        
        # Check stats
        self.assertGreater(stats["flows_performed"], 0)
        self.assertGreater(stats["splats_affected"], 0)
        self.assertIn("token", stats["levels_updated"])
        self.assertIn("sentence", stats["levels_updated"])
        
        # Check that token positions were updated
        self.assertFalse(np.array_equal(self.token_splat1.position, original_pos_token1))
        self.assertFalse(np.array_equal(self.token_splat2.position, original_pos_token2))
        
        # Check flow stats were updated
        self.assertEqual(self.flow_manager.flow_stats["top_down_flows"], 1)
    
    def test_propagate_bottom_up(self):
        """Test bottom-up propagation of information."""
        # Store original positions for verification
        original_pos_sent1 = self.sent_splat1.position.copy()
        original_pos_doc1 = self.doc_splat1.position.copy()
        
        # Execute bottom-up propagation
        stats = self.flow_manager.propagate_bottom_up(tokens=self.tokens)
        
        # Check stats
        self.assertGreater(stats["flows_performed"], 0)
        self.assertGreater(stats["splats_affected"], 0)
        self.assertIn("token", stats["levels_updated"])
        self.assertIn("sentence", stats["levels_updated"])
        
        # Check that at least one parent position was updated
        # We use any() to check if at least one position changed
        sent_position_changed = not np.array_equal(self.sent_splat1.position, original_pos_sent1)
        doc_position_changed = not np.array_equal(self.doc_splat1.position, original_pos_doc1)
        
        self.assertTrue(sent_position_changed or doc_position_changed,
                       "Neither the sentence nor document position was updated")
        
        # Check flow stats were updated
        self.assertEqual(self.flow_manager.flow_stats["bottom_up_flows"], 1)
    
    def test_update_children(self):
        """Test the _update_children method."""
        # Store original positions for verification
        original_pos_token1 = self.token_splat1.position.copy()
        original_pos_token2 = self.token_splat2.position.copy()
        
        # Call _update_children directly
        self.flow_manager._update_children(
            parent=self.sent_splat1,
            children=[self.token_splat1, self.token_splat2]
        )
        
        # Check that token positions were updated
        self.assertFalse(np.array_equal(self.token_splat1.position, original_pos_token1))
        self.assertFalse(np.array_equal(self.token_splat2.position, original_pos_token2))
        
        # Check that tokens moved towards the parent
        token1_delta = self.token_splat1.position - original_pos_token1
        token2_delta = self.token_splat2.position - original_pos_token2
        
        # Tokens should move towards parent
        parent_to_token1 = self.sent_splat1.position - original_pos_token1
        parent_to_token2 = self.sent_splat1.position - original_pos_token2
        
        # Check direction of movement (sign of dot product should be positive)
        self.assertGreater(np.dot(token1_delta, parent_to_token1), 0)
        self.assertGreater(np.dot(token2_delta, parent_to_token2), 0)
    
    def test_update_parent(self):
        """Test the _update_parent method."""
        # Store original position for verification
        original_pos_sent1 = self.sent_splat1.position.copy()
        
        # Create token positions that are significantly different from the parent
        # so movement is more detectable
        test_token1 = Splat(dim=2, position=np.array([1.5, 0.7]), level="token", id="test_token1")
        test_token2 = Splat(dim=2, position=np.array([1.7, 0.5]), level="token", id="test_token2")
        
        # Call _update_parent directly with the test tokens
        self.flow_manager._update_parent(
            parent=self.sent_splat1,
            children=[test_token1, test_token2]
        )
        
        # Check that parent position was updated
        self.assertFalse(np.array_equal(self.sent_splat1.position, original_pos_sent1),
                        f"Parent position not updated: {self.sent_splat1.position} vs {original_pos_sent1}")
        
        # Check that parent moved towards the average position of children
        avg_child_pos = (test_token1.position + test_token2.position) / 2
        sent_delta = self.sent_splat1.position - original_pos_sent1
        
        # Parent should move towards average child position
        avg_to_parent = avg_child_pos - original_pos_sent1
        
        # Check direction of movement (sign of dot product should be positive)
        dot_product = np.dot(sent_delta, avg_to_parent)
        self.assertGreater(dot_product, 0, 
                          f"Parent not moving towards children. Dot product: {dot_product}")
    
    def test_reinforce_information_pathways(self):
        """Test reinforcing information pathways."""
        # Create mock attention history
        attention_history = [np.eye(4) for _ in range(3)]
        
        # Call the method
        result = self.flow_manager.reinforce_information_pathways(attention_history)
        
        # Check that the method returns the registry
        self.assertEqual(result, self.registry)
    
    def test_balance_level_contributions(self):
        """Test balancing level contributions."""
        # Store original weights
        original_weights = self.registry.hierarchy.level_weights.copy()
        
        # Update activation history for some splats to create imbalance
        for _ in range(5):
            self.token_splat1.activation_history.add(0.9)
            self.token_splat2.activation_history.add(0.8)
            self.sent_splat1.activation_history.add(0.2)
        
        # Call the method
        result = self.flow_manager.balance_level_contributions()
        
        # Check that the method returns the registry
        self.assertEqual(result, self.registry)
        
        # Check that weights were updated
        new_weights = self.registry.hierarchy.level_weights
        self.assertNotEqual(original_weights, new_weights)
        
        # Check that weights still sum to 1
        self.assertAlmostEqual(sum(new_weights), 1.0, places=6)
    
    def test_compute_cross_level_attention(self):
        """Test computing cross-level attention."""
        # Call the method
        result = self.flow_manager.compute_cross_level_attention(self.tokens)
        
        # Check return shape
        self.assertEqual(result.shape, (len(self.tokens), len(self.tokens)))
    
    def test_visualize_information_flow(self):
        """Test visualizing information flow."""
        # Call the method
        viz_data = self.flow_manager.visualize_information_flow(self.tokens)
        
        # Check basic structure
        self.assertIn("levels", viz_data)
        self.assertIn("splats_per_level", viz_data)
        self.assertIn("top_down_flows", viz_data)
        self.assertIn("bottom_up_flows", viz_data)
        
        # Check level counts
        self.assertEqual(viz_data["splats_per_level"]["token"], 3)  # Updated to match setup
        self.assertEqual(viz_data["splats_per_level"]["sentence"], 3)
        self.assertEqual(viz_data["splats_per_level"]["document"], 2)
    
    def test_analyze_bottlenecks(self):
        """Test analyzing information flow bottlenecks."""
        # Call the method
        report = self.flow_manager.analyze_bottlenecks()
        
        # Check basic structure
        self.assertIn("bottlenecks", report)
        self.assertIn("severity", report)
        self.assertIn("recommendations", report)
        
        # Create orphan splat to test orphan detection
        orphan_splat = Splat(dim=2, position=np.array([1.0, 1.0]), level="token", id="orphan")
        self.registry.register(orphan_splat)
        
        # Analyze again
        report = self.flow_manager.analyze_bottlenecks()
        
        # Should detect the orphan
        orphan_detected = False
        for bottleneck in report["bottlenecks"]:
            if bottleneck["type"] == "orphaned_splats":
                orphan_detected = True
                self.assertGreater(bottleneck["count"], 0)
                
        self.assertTrue(orphan_detected)
    
    def test_find_orphaned_splats(self):
        """Test finding orphaned splats."""
        # Initially should be no orphans
        orphans = self.flow_manager._find_orphaned_splats()
        self.assertEqual(len(orphans), 0)
        
        # Create orphan splat
        orphan_splat = Splat(dim=2, position=np.array([1.0, 1.0]), level="token", id="orphan")
        self.registry.register(orphan_splat)
        
        # Should now find the orphan
        orphans = self.flow_manager._find_orphaned_splats()
        self.assertEqual(len(orphans), 1)
        self.assertEqual(orphans[0].id, "orphan")
    
    def test_adapt_flow_strengths(self):
        """Test adapting flow strengths for different task types."""
        # Test classification task
        result = self.flow_manager.adapt_flow_strengths("classification")
        self.assertEqual(result, self.registry)
        self.assertEqual(self.flow_manager.top_down_strength, 0.3)
        self.assertEqual(self.flow_manager.bottom_up_strength, 0.7)
        
        # Test generation task
        result = self.flow_manager.adapt_flow_strengths("generation")
        self.assertEqual(result, self.registry)
        self.assertEqual(self.flow_manager.top_down_strength, 0.7)
        self.assertEqual(self.flow_manager.bottom_up_strength, 0.3)
        
        # Test question answering task
        result = self.flow_manager.adapt_flow_strengths("question_answering")
        self.assertEqual(result, self.registry)
        self.assertEqual(self.flow_manager.top_down_strength, 0.5)
        self.assertEqual(self.flow_manager.bottom_up_strength, 0.5)
        
        # Test unknown task type
        result = self.flow_manager.adapt_flow_strengths("unknown_task")
        self.assertEqual(result, self.registry)
        self.assertEqual(self.flow_manager.top_down_strength, 0.5)
        self.assertEqual(self.flow_manager.bottom_up_strength, 0.5)
    
    def test_error_handling(self):
        """Test error handling in propagation methods."""
        # Create a mock registry that raises exceptions
        mock_registry = Mock(spec=SplatRegistry)
        mock_registry.hierarchy = self.hierarchy
        mock_registry.get_splats_at_level.side_effect = Exception("Test exception")
        
        # Create flow manager with mock registry
        flow_manager = CrossLevelInfoFlow(registry=mock_registry)
        
        # Test error handling in top-down propagation
        stats = flow_manager.propagate_top_down()
        self.assertEqual(stats["flows_performed"], 0)
        self.assertEqual(stats["splats_affected"], 0)
        
        # Test error handling in bottom-up propagation
        stats = flow_manager.propagate_bottom_up()
        self.assertEqual(stats["flows_performed"], 0)
        self.assertEqual(stats["splats_affected"], 0)
    
    def test_single_level_hierarchy(self):
        """Test cross-level flow with a single-level hierarchy."""
        # Create single-level hierarchy
        single_hierarchy = Hierarchy(
            levels=["token"],
            init_splats_per_level=[10],
            level_weights=[1.0]
        )
        
        # Create registry with single level
        single_registry = SplatRegistry(
            hierarchy=single_hierarchy, 
            embedding_dim=2
        )
        
        # Create flow manager
        single_flow = CrossLevelInfoFlow(registry=single_registry)
        
        # Should handle single level gracefully
        top_down_stats = single_flow.propagate_top_down()
        bottom_up_stats = single_flow.propagate_bottom_up()
        
        # No flows should occur
        self.assertEqual(top_down_stats["flows_performed"], 0)
        self.assertEqual(bottom_up_stats["flows_performed"], 0)


if __name__ == "__main__":
    unittest.main()
