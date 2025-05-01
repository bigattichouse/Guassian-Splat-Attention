"""
Hierarchy class implementation for Hierarchical Splat Attention (HSA).

The Hierarchy defines the structure and relationships between different levels
of splats, from fine-grained (token-level) to coarse-grained (document-level).
"""

from typing import List, Dict, Optional, Union
import json


class Hierarchy:
    """
    Hierarchy defines the hierarchical structure for organizing splats into 
    different levels, from token-level to document-level representations.
    """
    
    def __init__(
        self,
        levels: Optional[List[str]] = None,
        init_splats_per_level: Optional[List[int]] = None,
        level_weights: Optional[List[float]] = None
    ):
        """Initialize a hierarchy structure.
        
        Args:
            levels: Names of hierarchy levels from lowest to highest (fine to coarse-grained)
            init_splats_per_level: Initial number of splats per level
            level_weights: Weight of each level's contribution to attention
        """
        # Default levels if none provided
        self.levels = levels if levels is not None else ["token", "phrase", "sentence", "document"]
        
        # Default initial splats per level if none provided
        default_splats = [100, 50, 20, 5]
        if init_splats_per_level is not None:
            self.init_splats_per_level = init_splats_per_level
        else:
            # Use as many defaults as we have levels
            self.init_splats_per_level = default_splats[:len(self.levels)]
            # If we need more than we have defaults for, append some reasonable values
            if len(self.levels) > len(default_splats):
                for _ in range(len(self.levels) - len(default_splats)):
                    self.init_splats_per_level.append(5)  # Use a small default
        
        # Default level weights
        default_weights = [0.4, 0.3, 0.2, 0.1]
        if level_weights is not None:
            # Normalize provided weights
            total = sum(level_weights)
            self.level_weights = [w / total for w in level_weights]
        else:
            # Use as many defaults as we have levels
            raw_weights = default_weights[:len(self.levels)]
            # If we need more than we have defaults for, distribute remaining weight equally
            if len(self.levels) > len(default_weights):
                remaining = 1.0 - sum(raw_weights)
                remaining_per_level = remaining / (len(self.levels) - len(default_weights))
                raw_weights.extend([remaining_per_level] * (len(self.levels) - len(default_weights)))
            # Normalize weights
            total = sum(raw_weights)
            self.level_weights = [w / total for w in raw_weights]
            
        # Validate dimensions match
        if len(self.init_splats_per_level) != len(self.levels):
            raise ValueError(
                f"Number of init_splats_per_level ({len(self.init_splats_per_level)}) " +
                f"must match number of levels ({len(self.levels)})"
            )
            
        if len(self.level_weights) != len(self.levels):
            raise ValueError(
                f"Number of level_weights ({len(self.level_weights)}) " +
                f"must match number of levels ({len(self.levels)})"
            )
    
    def get_level_index(self, level_name: str) -> int:
        """Get the index of a level by name.
        
        Args:
            level_name: Name of the level
            
        Returns:
            Index of the level
            
        Raises:
            ValueError: If level name is not found
        """
        try:
            return self.levels.index(level_name)
        except ValueError:
            raise ValueError(f"Level '{level_name}' not found in hierarchy")
    
    def get_level_weight(self, level_name: str) -> float:
        """Get the weight of a level by name.
        
        Args:
            level_name: Name of the level
            
        Returns:
            Weight of the level
        """
        idx = self.get_level_index(level_name)
        return self.level_weights[idx]
    
    def get_parent_level(self, level_name: str) -> Optional[str]:
        """Get the parent level name for a given level.
        
        Args:
            level_name: Name of the current level
            
        Returns:
            Name of parent level, or None if at top level
        """
        idx = self.get_level_index(level_name)
        if idx + 1 < len(self.levels):
            return self.levels[idx + 1]
        return None
    
    def get_child_level(self, level_name: str) -> Optional[str]:
        """Get the child level name for a given level.
        
        Args:
            level_name: Name of the current level
            
        Returns:
            Name of child level, or None if at bottom level
        """
        idx = self.get_level_index(level_name)
        if idx > 0:
            return self.levels[idx - 1]
        return None
    
    def is_valid_level(self, level_name: str) -> bool:
        """Check if a level name is valid in this hierarchy.
        
        Args:
            level_name: Name to check
            
        Returns:
            True if valid, False otherwise
        """
        return level_name in self.levels
    
    def get_num_init_splats(self, level_name: str) -> int:
        """Get the initial number of splats for a given level.
        
        Args:
            level_name: Name of the level
            
        Returns:
            Initial number of splats for this level
        """
        idx = self.get_level_index(level_name)
        return self.init_splats_per_level[idx]
    
    def adjust_level_weights(self, new_weights: List[float]):
        """Adjust the weights of each level.
        
        Args:
            new_weights: New weights for each level (will be normalized)
            
        Raises:
            ValueError: If number of weights doesn't match number of levels
        """
        if len(new_weights) != len(self.levels):
            raise ValueError(
                f"Number of weights ({len(new_weights)}) " +
                f"must match number of levels ({len(self.levels)})"
            )
            
        # Normalize weights to sum to 1
        total = sum(new_weights)
        self.level_weights = [w / total for w in new_weights]
    
    def add_level(self, level_name: str, position: int, init_splats: int, weight: float):
        """Add a new level to the hierarchy.
        
        Args:
            level_name: Name of the new level
            position: Position to insert (0 is lowest/finest level)
            init_splats: Initial number of splats for this level
            weight: Weight for this level's contribution
            
        Raises:
            ValueError: If level name already exists or position is invalid
        """
        if level_name in self.levels:
            raise ValueError(f"Level '{level_name}' already exists in hierarchy")
            
        if position < 0 or position > len(self.levels):
            raise ValueError(f"Invalid position {position} for hierarchy with {len(self.levels)} levels")
            
        # Insert the new level
        self.levels.insert(position, level_name)
        self.init_splats_per_level.insert(position, init_splats)
        
        # Recalculate weights to maintain sum = 1
        new_weights = self.level_weights.copy()
        new_weights.insert(position, weight)
        total = sum(new_weights)
        self.level_weights = [w / total for w in new_weights]
    
    def remove_level(self, level_name: str):
        """Remove a level from the hierarchy.
        
        Args:
            level_name: Name of the level to remove
            
        Raises:
            ValueError: If level doesn't exist or is the only level
        """
        if len(self.levels) <= 1:
            raise ValueError("Cannot remove the only level in hierarchy")
            
        idx = self.get_level_index(level_name)
        
        # Remove the level
        self.levels.pop(idx)
        self.init_splats_per_level.pop(idx)
        
        # Recalculate weights to maintain sum = 1
        new_weights = self.level_weights.copy()
        new_weights.pop(idx)
        total = sum(new_weights)
        self.level_weights = [w / total for w in new_weights]
    
    def to_dict(self) -> Dict:
        """Convert hierarchy to dictionary for serialization.
        
        Returns:
            Dictionary representation of hierarchy
        """
        return {
            "levels": self.levels,
            "init_splats_per_level": self.init_splats_per_level,
            "level_weights": self.level_weights
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Hierarchy":
        """Create hierarchy from dictionary.
        
        Args:
            data: Dictionary representation of hierarchy
            
        Returns:
            New Hierarchy instance
        """
        return cls(
            levels=data["levels"],
            init_splats_per_level=data["init_splats_per_level"],
            level_weights=data["level_weights"]
        )
    
    def to_json(self) -> str:
        """Convert hierarchy to JSON string.
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "Hierarchy":
        """Create hierarchy from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            New Hierarchy instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        """String representation of the hierarchy.
        
        Returns:
            String representation
        """
        levels_str = ", ".join(self.levels)
        return f"Hierarchy(levels=[{levels_str}], num_levels={len(self.levels)})"
