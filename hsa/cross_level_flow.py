"""
Cross-level information flow for Hierarchical Splat Attention (HSA).

This module provides functionality for managing information flow between different
hierarchy levels, enabling top-down and bottom-up propagation of attention patterns
and adaptive mechanisms that maintain coherence between levels.
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np
import logging

from .splat import Splat
from .registry import SplatRegistry
from .hierarchy import Hierarchy

# Configure logging
logger = logging.getLogger(__name__)


class CrossLevelInfoFlow:
    """
    Manages information flow between hierarchy levels in HSA.
    
    This class provides mechanisms for transferring information between different
    hierarchical levels, including top-down (coarse to fine) and bottom-up 
    (fine to coarse) propagation of signal, as well as balancing level contributions.
    """
    
    def __init__(
        self,
        registry: SplatRegistry,
        top_down_strength: float = 0.5,
        bottom_up_strength: float = 0.3
    ):
        """Initialize cross-level information flow manager.
        
        Args:
            registry: SplatRegistry containing the splats
            top_down_strength: Strength of top-down (coarse to fine) information flow
            bottom_up_strength: Strength of bottom-up (fine to coarse) information flow
        """
        self.registry = registry
        self.top_down_strength = top_down_strength
        self.bottom_up_strength = bottom_up_strength
        
        # Cache for keeping track of flow statistics
        self.flow_stats = {
            "top_down_flows": 0,
            "bottom_up_flows": 0,
            "last_flow_timestamp": 0.0,
            "bottlenecks": []
        }
    
    def enable_top_down_flow(
        self,
        strength: Optional[float] = None
    ) -> SplatRegistry:
        """Enable top-down (coarse to fine) information flow.
        
        This updates the registry with adjusted strength parameters and 
        prepares for top-down propagation of attention patterns.
        
        Args:
            strength: Optional override for top-down flow strength
            
        Returns:
            Updated SplatRegistry
        """
        # Use provided strength or instance default
        flow_strength = strength if strength is not None else self.top_down_strength
        
        # Update registry with flow strength information
        # In a real implementation, this would modify registry metadata
        # or add markers to relevant splats
        
        # Track flow enablement
        self.flow_stats["top_down_enabled"] = True
        self.flow_stats["top_down_strength"] = flow_strength
        
        # Return updated registry for chaining calls
        return self.registry
    
    def enable_bottom_up_flow(
        self,
        strength: Optional[float] = None
    ) -> SplatRegistry:
        """Enable bottom-up (fine to coarse) information flow.
        
        This updates the registry with adjusted strength parameters and
        prepares for bottom-up aggregation of attention patterns.
        
        Args:
            strength: Optional override for bottom-up flow strength
            
        Returns:
            Updated SplatRegistry
        """
        # Use provided strength or instance default
        flow_strength = strength if strength is not None else self.bottom_up_strength
        
        # Update registry with flow strength information
        # In a real implementation, this would modify registry metadata
        # or add markers to relevant splats
        
        # Track flow enablement
        self.flow_stats["bottom_up_enabled"] = True
        self.flow_stats["bottom_up_strength"] = flow_strength
        
        # Return updated registry for chaining calls
        return self.registry
    
    def propagate_top_down(
        self,
        tokens: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Propagate information from higher to lower levels.
        
        This implements a top-down flow where higher-level (coarse) splats
        influence the parameters of lower-level (fine) splats.
        
        Args:
            tokens: Optional token embeddings for context-aware propagation
            
        Returns:
            Dictionary with propagation statistics
        """
        stats = {
            "flows_performed": 0,
            "splats_affected": 0,
            "levels_updated": set()
        }
        
        try:
            # Get hierarchy levels from high to low
            hierarchy = self.registry.hierarchy
            levels = list(hierarchy.levels)
            levels.reverse()  # Reverse to go from high to low
            
            # Skip if only one level
            if len(levels) <= 1:
                return stats
            
            # Process each level except the lowest
            for i in range(len(levels) - 1):
                parent_level = levels[i]
                child_level = levels[i + 1]
                
                # Get splats at these levels
                parent_splats = list(self.registry.get_splats_at_level(parent_level))
                
                # Skip if no parent splats
                if not parent_splats:
                    continue
                
                # Track level as updated
                stats["levels_updated"].add(parent_level)
                stats["levels_updated"].add(child_level)
                
                # Process each parent splat
                for parent in parent_splats:
                    # Get all children
                    children = list(parent.children)
                    
                    # Skip if no children
                    if not children:
                        continue
                    
                    # Update children based on parent's parameters
                    self._update_children(parent, children)
                    
                    # Update statistics
                    stats["flows_performed"] += 1
                    stats["splats_affected"] += len(children)
                    
            # Update flow statistics
            self.flow_stats["top_down_flows"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during top-down propagation: {e}")
            return stats
    
    def _update_children(
        self,
        parent: Splat,
        children: List[Splat]
    ) -> None:
        """Update children splats based on parent's parameters.
        
        Args:
            parent: Parent splat
            children: List of child splats to update
        """
        # This method implements the core logic of how parent splats
        # influence their children. In a real implementation, this would
        # involve more sophisticated parameter updates.
        
        # Simple implementation: subtly adjust child positions towards parent
        for child in children:
            # Calculate vector from child to parent
            delta = parent.position - child.position
            
            # Scale by flow strength to avoid too much influence
            delta *= self.top_down_strength * 0.1
            
            # Move child slightly towards parent
            new_position = child.position + delta
            
            # Update child parameters
            child.update_parameters(position=new_position)
    
    def propagate_bottom_up(
        self,
        tokens: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Propagate information from lower to higher levels.
        
        This implements a bottom-up flow where lower-level (fine) splats
        influence the parameters of higher-level (coarse) splats.
        
        Args:
            tokens: Optional token embeddings for context-aware propagation
            
        Returns:
            Dictionary with propagation statistics
        """
        stats = {
            "flows_performed": 0,
            "splats_affected": 0,
            "levels_updated": set()
        }
        
        try:
            # Get hierarchy levels from low to high
            hierarchy = self.registry.hierarchy
            levels = list(hierarchy.levels)
            
            # Skip if only one level
            if len(levels) <= 1:
                return stats
            
            # Process each level except the highest
            for i in range(len(levels) - 1):
                child_level = levels[i]
                parent_level = levels[i + 1]
                
                # Get splats at the parent level
                parent_splats = list(self.registry.get_splats_at_level(parent_level))
                
                # Skip if no parent splats
                if not parent_splats:
                    continue
                
                # Track level as updated
                stats["levels_updated"].add(parent_level)
                stats["levels_updated"].add(child_level)
                
                # Process each parent splat
                for parent in parent_splats:
                    # Get all children
                    children = list(parent.children)
                    
                    # Skip if no children
                    if not children:
                        continue
                    
                    # Update parent based on children's parameters
                    self._update_parent(parent, children)
                    
                    # Update statistics
                    stats["flows_performed"] += 1
                    stats["splats_affected"] += 1  # Count parent
                    
            # Update flow statistics
            self.flow_stats["bottom_up_flows"] += 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during bottom-up propagation: {e}")
            return stats
    
    def _update_parent(
        self,
        parent: Splat,
        children: List[Splat]
    ) -> None:
        """Update parent splat based on children's parameters.
        
        Args:
            parent: Parent splat to update
            children: List of child splats
        """
        # This method implements the core logic of how child splats
        # influence their parent. In a real implementation, this would
        # involve more sophisticated parameter updates.
        
        if not children:
            return
        
        # Calculate average position of children
        child_positions = np.array([child.position for child in children])
        avg_position = np.mean(child_positions, axis=0)
        
        # Calculate weighted vector from parent to average child position
        delta = avg_position - parent.position
        
        # Scale by flow strength
        delta *= self.bottom_up_strength
        
        # Move parent slightly towards average child position
        new_position = parent.position + delta
        
        # Update parent parameters
        parent.update_parameters(position=new_position)
    
    def reinforce_information_pathways(
        self,
        attention_history: List[np.ndarray]
    ) -> SplatRegistry:
        """Reinforce information pathways based on attention history.
        
        This strengthens connections along paths that are frequently used
        in attention computations.
        
        Args:
            attention_history: History of attention matrices
            
        Returns:
            Updated SplatRegistry
        """
        # This is a placeholder for a more sophisticated implementation
        # that would analyze attention patterns and strengthen relevant pathways
        
        # Update registry - in a real implementation, this would modify
        # the strengths of connections between splats at different levels
        
        # Return updated registry for chaining calls
        return self.registry
    
    def balance_level_contributions(self) -> SplatRegistry:
        """Balance the contributions of different hierarchy levels.
        
        This adjusts level weights to ensure each level contributes appropriately
        to the overall attention computation.
        
        Returns:
            Updated SplatRegistry with balanced level weights
        """
        # Get current level weights
        hierarchy = self.registry.hierarchy
        current_weights = [hierarchy.get_level_weight(level) for level in hierarchy.levels]
        
        # Calculate activity per level
        level_activity = {}
        
        for level in hierarchy.levels:
            splats = list(self.registry.get_splats_at_level(level))
            if not splats:
                level_activity[level] = 0.0
                continue
            
            # Calculate average activation for this level
            activations = [splat.get_average_activation() for splat in splats]
            level_activity[level] = sum(activations) / len(activations)
        
        # Adjust weights based on activity
        new_weights = []
        for i, level in enumerate(hierarchy.levels):
            activity = level_activity[level]
            
            # Basic rebalancing: increase weight for less active levels
            # and decrease for more active levels
            adjustment = 0.1 * (0.5 - activity)  # Center around 0.5 activity
            new_weight = current_weights[i] + adjustment
            
            # Ensure weight stays positive
            new_weight = max(0.1, new_weight)
            
            new_weights.append(new_weight)
        
        # Normalize weights to sum to 1
        total = sum(new_weights)
        if total > 0:
            new_weights = [w / total for w in new_weights]
        
        # Update hierarchy weights
        hierarchy.adjust_level_weights(new_weights)
        
        return self.registry
    
    def compute_cross_level_attention(
        self,
        tokens: np.ndarray
    ) -> np.ndarray:
        """Compute attention that incorporates cross-level interactions.
        
        This extends standard attention computation to include explicit
        cross-level information flow.
        
        Args:
            tokens: Token embeddings of shape [seq_len, embedding_dim]
            
        Returns:
            Attention matrix of shape [seq_len, seq_len]
        """
        # This is a placeholder for a more sophisticated implementation
        # that would compute attention with explicit cross-level flows
        
        # In a real implementation, this would compute attention that
        # explicitly incorporates information flowing across levels
        
        # For now, return a dummy attention matrix
        seq_len = tokens.shape[0]
        return np.eye(seq_len)
    
    def visualize_information_flow(
        self,
        tokens: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Visualize cross-level information flow.
        
        This creates a visualization of how information flows between
        hierarchy levels.
        
        Args:
            tokens: Optional token embeddings for context
            
        Returns:
            Dictionary with visualization data
        """
        # This is a placeholder for a more sophisticated visualization
        # that would show information flow between hierarchy levels
        
        # In a real implementation, this would generate data for
        # visualizing flow patterns, pathways, and strengths
        
        # Get hierarchy and splats for visualization
        hierarchy = self.registry.hierarchy
        levels = hierarchy.levels
        
        # Initialize visualization data
        viz_data = {
            "levels": levels,
            "splats_per_level": {},
            "top_down_flows": [],
            "bottom_up_flows": [],
            "bottlenecks": []
        }
        
        # Count splats per level
        for level in levels:
            viz_data["splats_per_level"][level] = self.registry.count_splats(level)
        
        # In a real implementation, generate flow data
        # For now, return dummy structure
        
        return viz_data
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze information flow bottlenecks in the hierarchy.
        
        This identifies paths where information flow is restricted,
        which could limit model performance.
        
        Returns:
            Report of bottlenecks and their severity
        """
        # This is a placeholder for a more sophisticated analysis
        # that would identify bottlenecks in information flow
        
        report = {
            "bottlenecks": [],
            "severity": 0.0,
            "recommendations": []
        }
        
        # Get hierarchy and levels
        hierarchy = self.registry.hierarchy
        levels = hierarchy.levels
        
        # Skip if only one level
        if len(levels) <= 1:
            return report
        
        # Check for empty levels (most common bottleneck)
        for i, level in enumerate(levels):
            splat_count = self.registry.count_splats(level)
            
            if splat_count == 0:
                bottleneck = {
                    "type": "empty_level",
                    "level": level,
                    "severity": 1.0,  # Critical severity
                    "recommendation": f"Add splats to empty level '{level}'"
                }
                
                report["bottlenecks"].append(bottleneck)
                report["recommendations"].append(bottleneck["recommendation"])
        
        # Check for disconnected splats
        orphaned = self._find_orphaned_splats()
        if orphaned:
            bottleneck = {
                "type": "orphaned_splats",
                "count": len(orphaned),
                "severity": 0.7,  # High severity
                "recommendation": "Create parent-child connections for orphaned splats"
            }
            
            report["bottlenecks"].append(bottleneck)
            report["recommendations"].append(bottleneck["recommendation"])
        
        # Calculate overall severity (maximum of all bottlenecks)
        if report["bottlenecks"]:
            report["severity"] = max(b["severity"] for b in report["bottlenecks"])
        
        return report
    
    def _find_orphaned_splats(self) -> List[Splat]:
        """Find splats that should have parents but don't.
        
        Returns:
            List of orphaned splats
        """
        # Get all levels except the highest
        hierarchy = self.registry.hierarchy
        levels = hierarchy.levels[:-1]  # All but the highest level
        
        orphaned = []
        
        for level in levels:
            splats = list(self.registry.get_splats_at_level(level))
            
            for splat in splats:
                # Check if this splat should have a parent but doesn't
                if splat.parent is None:
                    orphaned.append(splat)
        
        return orphaned
    
    def adapt_flow_strengths(
        self,
        task_type: str
    ) -> SplatRegistry:
        """Adapt flow strengths based on task type.
        
        This adjusts information flow parameters to optimize for
        specific types of tasks.
        
        Args:
            task_type: Type of task ("classification", "generation", etc.)
            
        Returns:
            Updated SplatRegistry
        """
        # Adjust flow strengths based on task type
        if task_type == "classification":
            # Classification benefits from stronger bottom-up flow
            self.bottom_up_strength = 0.7
            self.top_down_strength = 0.3
            
        elif task_type == "generation":
            # Generation benefits from stronger top-down flow
            self.top_down_strength = 0.7
            self.bottom_up_strength = 0.3
            
        elif task_type == "question_answering":
            # Q&A benefits from balanced flow
            self.top_down_strength = 0.5
            self.bottom_up_strength = 0.5
            
        else:
            # Default: balanced
            self.top_down_strength = 0.5
            self.bottom_up_strength = 0.5
        
        logger.info(f"Adapted flow strengths for task type '{task_type}': " +
                   f"top-down={self.top_down_strength}, " +
                   f"bottom-up={self.bottom_up_strength}")
        
        # Update registry
        self.enable_top_down_flow(self.top_down_strength)
        self.enable_bottom_up_flow(self.bottom_up_strength)
        
        return self.registry
