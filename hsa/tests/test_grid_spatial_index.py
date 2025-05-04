import unittest
import numpy as np
from hsa.grid_spatial_index import GridSpatialIndex
from hsa.splat import Splat


class TestGridSpatialIndex(unittest.TestCase):
    """Tests for the grid-based spatial indexing implementation."""

    def setUp(self):
        """Set up test data for grid spatial index tests."""
        # Create a grid index for 2D space
        self.dim = 2
        self.cell_size = 1.0
        self.min_coord = -10.0
        self.max_coord = 10.0
        self.index = GridSpatialIndex(
            dim=self.dim,
            cell_size=self.cell_size,
            min_coord=self.min_coord,
            max_coord=self.max_coord
        )
        
        # Create some test splats
        self.splats = []
        for i in range(5):
            # Create splats at different positions
            x = i - 2  # Positions: -2, -1, 0, 1, 2
            y = i % 3 - 1  # Positions: -1, 0, 1, -1, 0
            position = np.array([x, y], dtype=float)
            covariance = np.eye(self.dim) * (i + 1) * 0.1
            
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
        """Test grid spatial index initialization."""
        # Verify initial state
        self.assertEqual(self.index.dim, self.dim)
        self.assertEqual(self.index.cell_size, self.cell_size)
        self.assertEqual(self.index.min_coord, self.min_coord)
        self.assertEqual(self.index.max_coord, self.max_coord)
        self.assertEqual(self.index.num_splats, 0)
        
        # Calculate expected grid size
        expected_grid_size = int((self.max_coord - self.min_coord) / self.cell_size) + 1
        self.assertEqual(self.index.grid_size, expected_grid_size)

    def test_get_cell_index(self):
        """Test _get_cell_index method."""
        # Test with various positions
        positions = [
            (np.array([0.0, 0.0]), (10, 10)),  # Origin is at grid center
            (np.array([-10.0, -10.0]), (0, 0)),  # Lower bounds
            (np.array([10.0, 10.0]), (20, 20)),  # Upper bounds
            (np.array([5.5, -3.2]), (15, 6)),  # Random position
            (np.array([-15.0, 20.0]), (0, 20))  # Position outside grid bounds
        ]
        
        for position, expected in positions:
            cell = self.index._get_cell_index(position)
            self.assertEqual(cell, expected)

    def test_insert(self):
        """Test inserting splats into the index."""
        # Insert all test splats
        for splat in self.splats:
            self.index.insert(splat)
        
        # Verify counts
        self.assertEqual(self.index.num_splats, len(self.splats))
        self.assertGreaterEqual(self.index.num_cells, 1)
        
        # Verify splat_to_cell mapping
        for splat in self.splats:
            self.assertIn(splat.id, self.index.splat_to_cell)
            
            # Get the cell for this splat
            cell = self.index.splat_to_cell[splat.id]
            
            # Verify the splat is in that cell
            self.assertIn(splat, self.index.grid[cell])

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
        
        # Remove the first splat
        result = self.index.remove(self.splats[0].id)
        
        # Verify removal succeeded
        self.assertTrue(result)
        self.assertEqual(self.index.num_splats, initial_count - 1)
        self.assertNotIn(self.splats[0].id, self.index.splat_to_cell)
        
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
        
        # Remember its original cell
        original_cell = self.index.splat_to_cell[splat.id]
        
        # Update its position to move it to a different cell
        new_position = splat.position + np.array([5.0, 5.0])
        splat.update_parameters(position=new_position)
        
        # Update the index
        self.index.update(splat)
        
        # Verify the splat has been moved to a new cell
        new_cell = self.index.splat_to_cell[splat.id]
        self.assertNotEqual(new_cell, original_cell)
        
        # Verify the splat is in the correct cell
        self.assertIn(splat, self.index.grid[new_cell])
        
        # Verify it's no longer in the old cell
        if original_cell in self.index.grid:
            self.assertNotIn(splat, self.index.grid[original_cell])

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
        
        # Test center and radius
        center = np.array([0.0, 0.0])
        radius = 1.5
        
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
        for i in range(3):
            for level in ["token", "sentence", "document"]:
                position = np.array([i - 1.0, 0.0])
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
        results = self.index.find_by_level("token", position, k=2)
        
        # Should return up to k results
        self.assertLessEqual(len(results), 2)
        
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


if __name__ == "__main__":
    unittest.main()
