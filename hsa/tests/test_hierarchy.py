"""
Tests for the Hierarchy class in the HSA implementation.
"""

import pytest
import json
from hsa.hierarchy import Hierarchy


class TestHierarchy:
    """Tests for the Hierarchy class."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        hierarchy = Hierarchy()
        
        # Check default levels
        assert hierarchy.levels == ["token", "phrase", "sentence", "document"]
        
        # Check default initial splats per level
        assert hierarchy.init_splats_per_level == [100, 50, 20, 5]
        
        # Check default level weights (should sum to 1)
        assert len(hierarchy.level_weights) == 4
        assert pytest.approx(sum(hierarchy.level_weights)) == 1.0
        
        # Weights should be in decreasing order
        for i in range(1, len(hierarchy.level_weights)):
            assert hierarchy.level_weights[i-1] >= hierarchy.level_weights[i]
    
    def test_init_custom(self):
        """Test initialization with custom values."""
        levels = ["char", "word", "line", "paragraph"]
        init_splats = [200, 100, 50, 10]
        weights = [0.4, 0.3, 0.2, 0.1]
        
        hierarchy = Hierarchy(
            levels=levels,
            init_splats_per_level=init_splats,
            level_weights=weights
        )
        
        assert hierarchy.levels == levels
        assert hierarchy.init_splats_per_level == init_splats
        
        # Weights should be normalized to sum to 1
        assert pytest.approx(sum(hierarchy.level_weights)) == 1.0
        for i in range(len(weights)):
            assert pytest.approx(hierarchy.level_weights[i]) == weights[i] / sum(weights)
    
    def test_init_validation(self):
        """Test validation during initialization."""
        # Test with mismatched dimensions
        with pytest.raises(ValueError):
            Hierarchy(
                levels=["token", "phrase", "sentence"],
                init_splats_per_level=[100, 50]  # Wrong length
            )
        
        with pytest.raises(ValueError):
            Hierarchy(
                levels=["token", "phrase"],
                level_weights=[0.4, 0.3, 0.2]  # Wrong length
            )
    
    def test_get_level_index(self):
        """Test getting level index."""
        hierarchy = Hierarchy(levels=["a", "b", "c"])
        
        assert hierarchy.get_level_index("a") == 0
        assert hierarchy.get_level_index("b") == 1
        assert hierarchy.get_level_index("c") == 2
        
        # Test invalid level
        with pytest.raises(ValueError):
            hierarchy.get_level_index("d")
    
    def test_get_level_weight(self):
        """Test getting level weight."""
        hierarchy = Hierarchy(
            levels=["a", "b", "c"],
            level_weights=[0.5, 0.3, 0.2]
        )
        
        assert pytest.approx(hierarchy.get_level_weight("a")) == 0.5
        assert pytest.approx(hierarchy.get_level_weight("b")) == 0.3
        assert pytest.approx(hierarchy.get_level_weight("c")) == 0.2
        
        # Test invalid level
        with pytest.raises(ValueError):
            hierarchy.get_level_weight("d")
    
    def test_get_parent_level(self):
        """Test getting parent level."""
        hierarchy = Hierarchy(levels=["a", "b", "c"])
        
        assert hierarchy.get_parent_level("a") == "b"
        assert hierarchy.get_parent_level("b") == "c"
        assert hierarchy.get_parent_level("c") is None  # Top level has no parent
        
        # Test invalid level
        with pytest.raises(ValueError):
            hierarchy.get_parent_level("d")
    
    def test_get_child_level(self):
        """Test getting child level."""
        hierarchy = Hierarchy(levels=["a", "b", "c"])
        
        assert hierarchy.get_child_level("a") is None  # Bottom level has no child
        assert hierarchy.get_child_level("b") == "a"
        assert hierarchy.get_child_level("c") == "b"
        
        # Test invalid level
        with pytest.raises(ValueError):
            hierarchy.get_child_level("d")
    
    def test_is_valid_level(self):
        """Test checking if a level is valid."""
        hierarchy = Hierarchy(levels=["a", "b", "c"])
        
        assert hierarchy.is_valid_level("a") is True
        assert hierarchy.is_valid_level("b") is True
        assert hierarchy.is_valid_level("c") is True
        assert hierarchy.is_valid_level("d") is False
    
    def test_get_num_init_splats(self):
        """Test getting initial number of splats for a level."""
        hierarchy = Hierarchy(
            levels=["a", "b", "c"],
            init_splats_per_level=[100, 50, 20]
        )
        
        assert hierarchy.get_num_init_splats("a") == 100
        assert hierarchy.get_num_init_splats("b") == 50
        assert hierarchy.get_num_init_splats("c") == 20
        
        # Test invalid level
        with pytest.raises(ValueError):
            hierarchy.get_num_init_splats("d")
    
    def test_adjust_level_weights(self):
        """Test adjusting level weights."""
        hierarchy = Hierarchy(levels=["a", "b", "c"])
        
        new_weights = [0.6, 0.3, 0.1]
        hierarchy.adjust_level_weights(new_weights)
        
        # Weights should match after normalization
        for i, level in enumerate(hierarchy.levels):
            assert pytest.approx(hierarchy.get_level_weight(level)) == new_weights[i]
        
        # Sum should be 1
        assert pytest.approx(sum(hierarchy.level_weights)) == 1.0
        
        # Test with invalid dimension
        with pytest.raises(ValueError):
            hierarchy.adjust_level_weights([0.5, 0.5])  # Wrong length
    
    def test_add_level(self):
        """Test adding a new level."""
        hierarchy = Hierarchy(levels=["a", "b", "c"])
        
        # Add a level at the beginning (finest)
        hierarchy.add_level("aa", position=0, init_splats=200, weight=0.4)
        
        assert hierarchy.levels == ["aa", "a", "b", "c"]
        assert hierarchy.init_splats_per_level[0] == 200
        
        # Weights should be normalized
        assert pytest.approx(sum(hierarchy.level_weights)) == 1.0
        
        # Add a level in the middle
        hierarchy.add_level("bb", position=2, init_splats=75, weight=0.2)
        
        assert hierarchy.levels == ["aa", "a", "bb", "b", "c"]
        assert hierarchy.init_splats_per_level[2] == 75
        
        # Weights should still be normalized
        assert pytest.approx(sum(hierarchy.level_weights)) == 1.0
        
        # Test adding duplicate level
        with pytest.raises(ValueError):
            hierarchy.add_level("a", position=0, init_splats=100, weight=0.5)
        
        # Test invalid position
        with pytest.raises(ValueError):
            hierarchy.add_level("d", position=10, init_splats=10, weight=0.1)
    
    def test_remove_level(self):
        """Test removing a level."""
        hierarchy = Hierarchy(levels=["a", "b", "c", "d"])
        
        # Remove a middle level
        hierarchy.remove_level("b")
        
        assert hierarchy.levels == ["a", "c", "d"]
        
        # Weights should be normalized
        assert pytest.approx(sum(hierarchy.level_weights)) == 1.0
        
        # Can't remove only level
        with pytest.raises(ValueError):
            hierarchy = Hierarchy(levels=["a"])
            hierarchy.remove_level("a")
        
        # Test removing non-existent level
        with pytest.raises(ValueError):
            hierarchy.remove_level("x")
    
    def test_serialization(self):
        """Test serialization to/from dict and JSON."""
        hierarchy = Hierarchy(
            levels=["a", "b", "c"],
            init_splats_per_level=[100, 50, 20],
            level_weights=[0.6, 0.3, 0.1]
        )
        
        # Test to_dict
        data = hierarchy.to_dict()
        assert data["levels"] == ["a", "b", "c"]
        assert data["init_splats_per_level"] == [100, 50, 20]
        assert pytest.approx(data["level_weights"]) == [0.6, 0.3, 0.1]
        
        # Test from_dict
        new_hierarchy = Hierarchy.from_dict(data)
        assert new_hierarchy.levels == hierarchy.levels
        assert new_hierarchy.init_splats_per_level == hierarchy.init_splats_per_level
        assert pytest.approx(new_hierarchy.level_weights) == hierarchy.level_weights
        
        # Test to_json
        json_str = hierarchy.to_json()
        assert isinstance(json_str, str)
        
        # Test from_json
        new_hierarchy2 = Hierarchy.from_json(json_str)
        assert new_hierarchy2.levels == hierarchy.levels
        assert new_hierarchy2.init_splats_per_level == hierarchy.init_splats_per_level
        assert pytest.approx(new_hierarchy2.level_weights) == hierarchy.level_weights
    
    def test_repr(self):
        """Test string representation."""
        hierarchy = Hierarchy(levels=["a", "b", "c"])
        
        # Should contain level names
        repr_str = repr(hierarchy)
        assert "Hierarchy" in repr_str
        assert "a" in repr_str
        assert "b" in repr_str
        assert "c" in repr_str
        assert "num_levels=3" in repr_str
