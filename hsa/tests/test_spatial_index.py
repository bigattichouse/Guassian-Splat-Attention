import unittest
import numpy as np
from hsa.spatial_index import SpatialIndex, _Node
from hsa.splat import Splat


class TestSpatialIndex(unittest.TestCase):
    """Tests for the tree-based spatial indexing implementation."""

    def setUp(self):
        """Set up test data for spatial index tests."""
        # Create a spatial index for 2D space
        self.dim = 2
        self.max_leaf_size = 3
        self.max_depth = 5
        self.index = SpatialIndex(
            dim=self.dim,
            max_leaf_size=self.max_leaf_size,
            max_depth=self.max_depth
        )
        
        # Create some test splats
        self.splats = []
        for i in range(10):
            # Create splats at different positions
            angle = 2 * np.pi * i / 10
            radius = 1.0 + i % 3  # Radii: 1, 2, 3, 1, 2, ...
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            position = np.array([x, y], dtype=float)
            covariance = np.eye(self.dim) * (0.1 + i * 0.05)
            
            splat = Splat(
                dim=self.dim,
                position=position,
                covariance=covariance,
                amplitude=1.0,
                level="token",
                id=f"splat_{i}"
            )
            self.splats.append(splat)

    def test_initialization(self):
        """Test spatial index initialization."""
        # Verify initial state
        self.assertEqual(self.index.dim, self.dim)
        self.assertEqual(self.index.max_leaf_size, self.max_leaf_size)
        self.assertEqual(self.index.max_depth, self.max_depth)
        self.assertEqual(self.index.num_splats, 0)
        self.assertEqual(self.index.num_nodes, 0)
        self.assertEqual(self.index.max_current_depth, 0)
        self.assertEqual(self.index.rebuild_count, 0)
        self.assertIsNone(self.index.root)
        self.assertEqual(len(self.index.splat_to_node), 0)

    def test_insert(self):
        """Test inserting splats into the index."""
        # Insert all test splats
        for splat in self.splats:
            self.index.insert(splat)
        
        # Verify counts
        self.assertEqual(self.index.num_splats, len(self.splats))
        self.assertGreater(self.index.num_nodes, 1)  # Should have created at least one node
        self.assertGreaterEqual(self.index.max_current_depth, 1)  # Should have at least depth 1
        
        # Verify splat_to_node mapping
        for splat in self.splats:
            self.assertIn(splat.id, self.index.splat_to_node)
            
            # The node should contain the splat
            node = self.index.splat_to_node[splat.id]
            if node.is_leaf():
                self.assertIn(splat, node.splats)

    def test_insert_incompatible_dimension(self):
        """Test inserting a splat with incompatible dimension."""
        # Create a 3D splat
        position = np.array([0.0, 0.0, 0.0])
        covariance = np.eye(3)
        splat = Splat(
            dim=3,
            position=position,
            covariance=covariance,
            amplitude=1.0,
            level="token",
            id="3d_splat"
        )
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.index.insert(splat)

    def test_remove(self):
        """Test removing splats from the index."""
        # Insert all splats
        for splat in self.splats:
            self.index.insert(splat)
        
        initial_count = self.index.num_splats
        
        # Remove a splat
        result = self.index.remove(self.splats[0].id)
        
        # Verify removal succeeded
        self.assertTrue(result)
        self.assertEqual(self.index.num_splats, initial_count - 1)
        self.assertNotIn(self.splats[0].id, self.index.splat_to_node)
        
        # Verify the splat is no longer in the tree
        all_splats = self.index.get_all_splats()
        self.assertNotIn(self.splats[0], all_splats)
        
        # Try to remove a non-existent splat
        result = self.index.remove("nonexistent_splat")
        
        # Verify removal failed
        self.assertFalse(result)
        self.assertEqual(self.index.num_splats, initial_count - 1)

    def test_update(self):
        """Test updating a splat's position in the index."""
        # Insert all splats
        for splat in self.splats:
            self.index.insert(splat)
        
        # Get a splat to update
        splat = self.splats[0]
        original_node = self.index.splat_to_node[splat.id]
        
        # Update its position significantly
        new_position = splat.position + np.array([10.0, 10.0])
        original_position = splat.position.copy()
        splat.update_parameters(position=new_position)
        
        # Update the index
        self.index.update(splat)
        
        # Verify the splat has been moved to a new node
        new_node = self.index.splat_to_node[splat.id]
        
        # Verify the splat is in the tree with its new position
        all_splats = self.index.get_all_splats()
        found_splat = None
        for s in all_splats:
            if s.id == splat.id:
                found_splat = s
                break
        
        self.assertIsNotNone(found_splat)
        np.testing.assert_array_equal(found_splat.position, new_position)
        
        # Reset the position for other tests
        splat.update_parameters(position=original_position)

    def test_update_nonexistent_splat(self):
        """Test updating a non-existent splat."""
        # Create a splat not in the index
        position = np.array([0.0, 0.0])
        covariance = np.eye(self.dim)
        splat = Splat(
            dim=self.dim,
            position=position,
            covariance=covariance,
            amplitude=1.0,
            level="token",
            id="nonexistent_splat"
        )
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.index.update(splat)

    def test_find_nearest(self):
        """Test finding the nearest splats to a position."""
        # Insert all splats
        for splat in self.splats:
            self.index.insert(splat)
        
        # Test position
        position = np.array([0.0, 0.0])
        
        # Find nearest splat
        k = 1
        nearest = self.index.find_nearest(position, k)
        
        # Should return one result
        self.assertEqual(len(nearest), k)
        
        # Result should be a (splat, distance) tuple
        self.assertEqual(len(nearest[0]), 2)
        self.assertIsInstance(nearest[0][0], Splat)
        self.assertIsInstance(nearest[0][1], float)
        
        # Find multiple nearest splats
        k = 3
        nearest = self.index.find_nearest(position, k)
        
        # Should return k results
        self.assertEqual(len(nearest), k)
        
        # Results should be sorted by distance (nearest first)
        for i in range(1, len(nearest)):
            self.assertLessEqual(nearest[i-1][1], nearest[i][1])
        
        # Verify correct splats are returned
        actual_distances = []
        for splat in self.splats:
            dist = np.linalg.norm(splat.position - position)
            actual_distances.append((splat, dist))
        
        actual_distances.sort(key=lambda x: x[1])
        expected_nearest = actual_distances[:k]
        
        # Compare splat IDs and distances
        for i in range(k):
            expected_splat, expected_dist = expected_nearest[i]
            actual_splat, actual_dist = nearest[i]
            self.assertEqual(actual_splat.id, expected_splat.id)
            self.assertAlmostEqual(actual_dist, expected_dist)

    def test_find_nearest_with_empty_index(self):
        """Test finding nearest splats with an empty index."""
        # Test position
        position = np.array([0.0, 0.0])
        
        # Find nearest splat
        k = 1
        nearest = self.index.find_nearest(position, k)
        
        # Should return empty list
        self.assertEqual(len(nearest), 0)

    def test_find_nearest_incompatible_dimension(self):
        """Test find_nearest with incompatible dimension."""
        # Test with 3D position
        position = np.array([0.0, 0.0, 0.0])
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.index.find_nearest(position, 1)

    def test_range_query(self):
        """Test finding all splats within a given radius."""
        # Insert all splats
        for splat in self.splats:
            self.index.insert(splat)
        
        # Test center and different radii
        center = np.array([0.0, 0.0])
        radii = [0.5, 1.5, 2.5]
        
        for radius in radii:
            # Find splats within radius
            results = self.index.range_query(center, radius)
            
            # Verify results
            for splat in results:
                # Distance should be less than or equal to radius
                distance = np.linalg.norm(splat.position - center)
                self.assertLessEqual(distance, radius)
            
            # Verify all matching splats are included
            for splat in self.splats:
                distance = np.linalg.norm(splat.position - center)
                if distance <= radius:
                    self.assertIn(splat, results)
                else:
                    self.assertNotIn(splat, results)

    def test_range_query_with_empty_index(self):
        """Test range query with an empty index."""
        # Test center and radius
        center = np.array([0.0, 0.0])
        radius = 1.5
        
        # Find splats within radius
        results = self.index.range_query(center, radius)
        
        # Should return empty list
        self.assertEqual(len(results), 0)

    def test_range_query_incompatible_dimension(self):
        """Test range_query with incompatible dimension."""
        # Test with 3D position
        center = np.array([0.0, 0.0, 0.0])
        radius = 1.5
        
        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.index.range_query(center, radius)

    def test_find_by_level(self):
        """Test finding nearest splats at a specific level."""
        # Create splats at different levels
        level_splats = []
        for i in range(5):
            for level in ["token", "sentence", "document"]:
                angle = 2 * np.pi * i / 5
                radius = 1.0
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                position = np.array([x, y], dtype=float)
                covariance = np.eye(self.dim) * 0.1
                splat = Splat(
                    dim=self.dim,
                    position=position,
                    covariance=covariance,
                    amplitude=1.0,
                    level=level,
                    id=f"splat_{level}_{i}"
                )
                level_splats.append(splat)
        
        # Insert all splats
        for splat in level_splats:
            self.index.insert(splat)
        
        # Test position
        position = np.array([0.0, 0.0])
        
        # Find nearest splats at 'token' level
        results = self.index.find_by_level("token", position, k=3)
        
        # Should return up to k results
        self.assertLessEqual(len(results), 3)
        
        # All results should be at 'token' level
        for splat, _ in results:
            self.assertEqual(splat.level, "token")
            
        # Results should be sorted by distance
        for i in range(1, len(results)):
            self.assertLessEqual(results[i-1][1], results[i][1])

    def test_find_by_level_nonexistent_level(self):
        """Test find_by_level with a non-existent level."""
        # Insert all splats
        for splat in self.splats:
            self.index.insert(splat)
        
        # Test position
        position = np.array([0.0, 0.0])
        
        # Find nearest splats at non-existent level
        results = self.index.find_by_level("nonexistent_level", position, k=2)
        
        # Should return empty list
        self.assertEqual(len(results), 0)

    def test_get_all_splats(self):
        """Test getting all splats from the index."""
        # Insert all splats
        for splat in self.splats:
            self.index.insert(splat)
        
        # Get all splats
        all_splats = self.index.get_all_splats()
        
        # Should return all inserted splats
        self.assertEqual(len(all_splats), len(self.splats))
        
        # Should contain all original splats
        for splat in self.splats:
            self.assertIn(splat, all_splats)

    def test_empty_get_all_splats(self):
        """Test get_all_splats with an empty index."""
        # Get all splats
        all_splats = self.index.get_all_splats()
        
        # Should return empty list
        self.assertEqual(len(all_splats), 0)

    def test_node(self):
        """Test the internal _Node class functionality."""
        # Create a root node
        root = _Node(dim=2, depth=0, max_leaf_size=3, max_depth=5)
        
        # Check initial state
        self.assertTrue(root.is_leaf())
        self.assertEqual(root.depth, 0)
        self.assertEqual(len(root.splats), 0)
        self.assertIsNone(root.split_axis)
        self.assertIsNone(root.split_value)
        self.assertIsNone(root.children)
        
        # Insert splats
        for i in range(5):
            splat = self.splats[i]
            node = root.insert(splat)
            
            # First max_leaf_size splats should be inserted into root
            if i < root.max_leaf_size:
                self.assertIs(node, root)
                self.assertIn(splat, root.splats)
            else:
                # After max_leaf_size, root should split
                self.assertFalse(root.is_leaf())
                self.assertIsNotNone(root.split_axis)
                self.assertIsNotNone(root.split_value)
                self.assertIsNotNone(root.children)
                self.assertEqual(len(root.children), 2)
                
                # Splat should be in one of the children
                self.assertTrue(splat in node.splats)
                
        # Test removing splats
        for i in range(5):
            # Should return True for successful removal
            result = root.remove(self.splats[i].id)
            self.assertTrue(result)
            
            # Splat should no longer be in the tree
            all_splats = root.get_all_splats()
            self.assertNotIn(self.splats[i], all_splats)
        
        # Test removing non-existent splat
        result = root.remove("nonexistent_splat")
        self.assertFalse(result)

    def test_rebuild(self):
        """Test rebuilding the index."""
        # Insert many splats to force tree to grow deep
        many_splats = []
        for i in range(30):
            position = np.random.normal(0, 1, self.dim)
            covariance = np.eye(self.dim) * 0.1
            splat = Splat(
                dim=self.dim,
                position=position,
                covariance=covariance,
                amplitude=1.0,
                level="token",
                id=f"many_splat_{i}"
            )
            many_splats.append(splat)
            self.index.insert(splat)
        
        # Check depth before rebuild
        depth_before = self.index.max_current_depth
        
        # Force rebuild
        self.index._rebuild()
        
        # Check rebuild count
        self.assertEqual(self.index.rebuild_count, 1)
        
        # Check that all splats are still in the index
        self.assertEqual(self.index.num_splats, len(many_splats))
        
        # Get all splats
        all_splats = self.index.get_all_splats()
        self.assertEqual(len(all_splats), len(many_splats))
        
        # Check that all original splats are in the rebuilt index
        for splat in many_splats:
            self.assertIn(splat, all_splats)


class TestSpatialIndexFactory(unittest.TestCase):
    """Tests for the SpatialIndexFactory."""

    def test_create_index(self):
        """Test creating an index via factory."""
        from hsa.spatial_index import SpatialIndexFactory
        
        # Create splats
        dim = 2
        splats = []
        for i in range(5):
            position = np.array([i - 2.0, 0.0], dtype=float)
            covariance = np.eye(dim) * 0.1
            splat = Splat(
                dim=dim,
                position=position,
                covariance=covariance,
                amplitude=1.0,
                level="token",
                id=f"factory_splat_{i}"
            )
            splats.append(splat)
        
        # Create index via factory
        index = SpatialIndexFactory.create_index(
            dim=dim,
            splats=splats,
            index_type="tree"
        )
        
        # Verify index type and contents
        self.assertIsInstance(index, SpatialIndex)
        self.assertEqual(index.num_splats, len(splats))
        
        # Get all splats
        all_splats = index.get_all_splats()
        
        # Verify all splats are in the index
        for splat in splats:
            self.assertIn(splat, all_splats)


if __name__ == "__main__":
    unittest.main()
