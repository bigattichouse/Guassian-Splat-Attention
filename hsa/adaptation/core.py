"""
Core adaptation module for Hierarchical Splat Attention (HSA).

This module defines the fundamental structures and interfaces for HSA adaptation:
- AdaptationType enum for categorizing adaptation operations
- AdaptationMonitor for tracking adaptation history
- AdaptationResult for detailed reporting of adaptation outcomes
- Core interfaces and types for adaptation mechanisms
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any
from enum import Enum
import logging
import time
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import core data structures
from hsa.data_structures import Splat, SplatRegistry


class AdaptationType(Enum):
    """Types of adaptation that can occur to splats."""
    MITOSIS = "mitosis"  # Splat division
    BIRTH = "birth"      # New splat creation
    DEATH = "death"      # Splat removal
    ADJUST = "adjust"    # Parameter adjustment without structural change
    MERGE = "merge"      # Merge similar splats


@dataclass
class SplatChange:
    """Detailed information about a change to a single splat."""
    splat_id: str
    level: str
    change_type: AdaptationType
    # For parameter changes
    position_before: Optional[np.ndarray] = None
    position_after: Optional[np.ndarray] = None
    amplitude_before: Optional[float] = None
    amplitude_after: Optional[float] = None
    # For structural changes
    related_splat_ids: List[str] = None
    parent_id: Optional[str] = None
    created_splat_ids: List[str] = None
    metrics_before: Dict[str, float] = None
    
    def __post_init__(self):
        if self.related_splat_ids is None:
            self.related_splat_ids = []
        if self.created_splat_ids is None:
            self.created_splat_ids = []
        if self.metrics_before is None:
            self.metrics_before = {}


class AdaptationResult:
    """
    Comprehensive result of an adaptation cycle, including detailed changes.
    """
    
    def __init__(self):
        """Initialize the adaptation result tracking."""
        self.changes: List[SplatChange] = []
        self.mitosis_count: int = 0
        self.birth_count: int = 0
        self.death_count: int = 0
        self.merge_count: int = 0
        self.adjust_count: int = 0
        self.start_time: float = time.time()
        self.duration: float = 0.0
        self.splats_before: int = 0
        self.splats_after: int = 0
        self.splats_by_level_before: Dict[str, int] = {}
        self.splats_by_level_after: Dict[str, int] = {}
        
    def add_change(self, change: SplatChange) -> None:
        """
        Add a splat change to the result.
        
        Args:
            change: The change to record
        """
        self.changes.append(change)
        
        # Update counts
        if change.change_type == AdaptationType.MITOSIS:
            self.mitosis_count += 1
        elif change.change_type == AdaptationType.BIRTH:
            self.birth_count += 1
        elif change.change_type == AdaptationType.DEATH:
            self.death_count += 1
        elif change.change_type == AdaptationType.MERGE:
            self.merge_count += 1
        elif change.change_type == AdaptationType.ADJUST:
            self.adjust_count += 1
    
    def set_registry_stats(self, registry_before: SplatRegistry, registry_after: SplatRegistry) -> None:
        """
        Record statistics about the splat registry before and after adaptation.
        
        Args:
            registry_before: Registry before adaptation
            registry_after: Registry after adaptation
        """
        self.splats_before = len(registry_before.splats)
        self.splats_after = len(registry_after.splats)
        
        # Count splats by level
        for level in registry_before.hierarchy.levels:
            self.splats_by_level_before[level] = len(registry_before.get_splats_at_level(level))
            self.splats_by_level_after[level] = len(registry_after.get_splats_at_level(level))
    
    def finish(self) -> None:
        """Mark adaptation as complete and record duration."""
        self.duration = time.time() - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the adaptation results.
        
        Returns:
            Dictionary with adaptation summary
        """
        return {
            "mitosis_count": self.mitosis_count,
            "birth_count": self.birth_count,
            "death_count": self.death_count,
            "merge_count": self.merge_count,
            "adjust_count": self.adjust_count,
            "total_changes": len(self.changes),
            "splats_before": self.splats_before,
            "splats_after": self.splats_after,
            "splats_by_level_before": self.splats_by_level_before,
            "splats_by_level_after": self.splats_by_level_after,
            "duration": self.duration
        }
    
    def __str__(self) -> str:
        """
        Get a string representation of the adaptation results.
        
        Returns:
            String summary of the adaptation
        """
        summary = self.get_summary()
        return (
            f"Adaptation completed in {summary['duration']:.2f}s: "
            f"{summary['mitosis_count']} mitosis, "
            f"{summary['birth_count']} births, "
            f"{summary['death_count']} deaths, "
            f"{summary['merge_count']} merges, "
            f"{summary['adjust_count']} adjusts. "
            f"Splats: {summary['splats_before']} → {summary['splats_after']}"
        )


class AdaptationMonitor:
    """
    Monitors and tracks adaptation-related metrics for splats over time.
    
    This class maintains history of splat activations and other metrics
    to make more informed adaptation decisions.
    """
    
    def __init__(self, consecutive_threshold: int = 3):
        """
        Initialize the adaptation monitor.
        
        Args:
            consecutive_threshold: Number of consecutive low activations before death
        """
        self.consecutive_threshold = consecutive_threshold
        self.low_activation_counts = {}  # Track consecutive low activations
        self.low_info_contribution_counts = {}  # Track consecutive low information contribution
        self.splat_lifetimes = {}  # Track how long each splat has existed
        self.adaptation_history = []  # Track all adaptations
        self.cycle_count = 0  # Count of adaptation cycles
        self.result_history: List[AdaptationResult] = []  # Detailed results history
    
    def update_lifetimes(self, splat_registry: SplatRegistry) -> None:
        """
        Update lifetime counters for all splats.
        
        Args:
            splat_registry: The registry containing all splats
        """
        # Add new splats to tracking
        for splat_id in splat_registry.splats:
            if splat_id not in self.splat_lifetimes:
                self.splat_lifetimes[splat_id] = 0
            else:
                self.splat_lifetimes[splat_id] += 1
        
        # Remove tracking for splats that no longer exist
        for splat_id in list(self.splat_lifetimes.keys()):
            if splat_id not in splat_registry.splats:
                del self.splat_lifetimes[splat_id]
                if splat_id in self.low_activation_counts:
                    del self.low_activation_counts[splat_id]
                if splat_id in self.low_info_contribution_counts:
                    del self.low_info_contribution_counts[splat_id]
    
    def record_adaptation(self, adaptation_type: AdaptationType, splat_id: str) -> None:
        """
        Record a single adaptation event.
        
        Args:
            adaptation_type: Type of adaptation that occurred
            splat_id: ID of the adapted splat
        """
        self.adaptation_history.append((adaptation_type, splat_id, time.time()))
    
    def add_result(self, result: AdaptationResult) -> None:
        """
        Record a complete adaptation result.
        
        Args:
            result: The detailed adaptation result to record
        """
        self.result_history.append(result)
        self.cycle_count += 1
    
    def reset_for_splat(self, splat_id: str) -> None:
        """
        Reset tracking counters for a specific splat.
        
        Args:
            splat_id: ID of the splat to reset
        """
        if splat_id in self.low_activation_counts:
            self.low_activation_counts[splat_id] = 0
        if splat_id in self.low_info_contribution_counts:
            self.low_info_contribution_counts[splat_id] = 0
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about adaptations across all history.
        
        Returns:
            Dictionary with adaptation statistics
        """
        stats = {
            "total_cycles": self.cycle_count,
            "total_adaptations": len(self.adaptation_history),
            "mitosis_count": sum(1 for a in self.adaptation_history if a[0] == AdaptationType.MITOSIS),
            "birth_count": sum(1 for a in self.adaptation_history if a[0] == AdaptationType.BIRTH),
            "death_count": sum(1 for a in self.adaptation_history if a[0] == AdaptationType.DEATH),
            "adjust_count": sum(1 for a in self.adaptation_history if a[0] == AdaptationType.ADJUST),
            "merge_count": sum(1 for a in self.adaptation_history if a[0] == AdaptationType.MERGE)
        }
        
        # Calculate rates per cycle if there have been cycles
        if self.cycle_count > 0:
            stats["adaptations_per_cycle"] = len(self.adaptation_history) / self.cycle_count
            stats["mitosis_per_cycle"] = stats["mitosis_count"] / self.cycle_count
            stats["birth_per_cycle"] = stats["birth_count"] / self.cycle_count
            stats["death_per_cycle"] = stats["death_count"] / self.cycle_count
            stats["adjust_per_cycle"] = stats["adjust_count"] / self.cycle_count
            stats["merge_per_cycle"] = stats["merge_count"] / self.cycle_count
        
        return stats
    
    def get_splat_stats(self, splat_id: str) -> Dict[str, Any]:
        """
        Get statistics about a specific splat.
        
        Args:
            splat_id: ID of the splat to get stats for
            
        Returns:
            Dictionary with splat statistics
        """
        stats = {
            "lifetime": self.splat_lifetimes.get(splat_id, 0),
            "low_activation_count": self.low_activation_counts.get(splat_id, 0),
            "low_info_contribution_count": self.low_info_contribution_counts.get(splat_id, 0)
        }
        
        # Count adaptations involving this splat
        splat_adaptations = [a for a in self.adaptation_history if a[1] == splat_id]
        stats["total_adaptations"] = len(splat_adaptations)
        
        # Count by type
        stats["mitosis_count"] = sum(1 for a in splat_adaptations if a[0] == AdaptationType.MITOSIS)
        stats["birth_count"] = sum(1 for a in splat_adaptations if a[0] == AdaptationType.BIRTH)
        stats["death_count"] = sum(1 for a in splat_adaptations if a[0] == AdaptationType.DEATH)
        stats["adjust_count"] = sum(1 for a in splat_adaptations if a[0] == AdaptationType.ADJUST)
        stats["merge_count"] = sum(1 for a in splat_adaptations if a[0] == AdaptationType.MERGE)
        
        return stats
    
    def get_recent_results(self, count: int = 5) -> List[AdaptationResult]:
        """
        Get the most recent adaptation results.
        
        Args:
            count: Number of recent results to return
            
        Returns:
            List of recent adaptation results
        """
        return self.result_history[-count:] if self.result_history else []


# Create default global monitor
default_monitor = AdaptationMonitor()


def create_adaptation_result(splat_registry: SplatRegistry) -> AdaptationResult:
    """
    Create a new adaptation result object, initialized with the current registry state.
    
    Args:
        splat_registry: The current splat registry
        
    Returns:
        Initialized adaptation result
    """
    result = AdaptationResult()
    
    # Record initial stats
    result.splats_before = len(splat_registry.splats)
    
    # Record splats by level
    for level in splat_registry.hierarchy.levels:
        result.splats_by_level_before[level] = len(splat_registry.get_splats_at_level(level))
    
    return result


def record_adjustment(
    result: AdaptationResult,
    splat: Splat,
    position_before: Optional[np.ndarray] = None,
    amplitude_before: Optional[float] = None,
    metrics_before: Optional[Dict[str, float]] = None
) -> None:
    """
    Record an adjustment to a splat in the adaptation result.
    
    Args:
        result: The adaptation result to update
        splat: The splat being adjusted
        position_before: The splat's position before adjustment (if changed)
        amplitude_before: The splat's amplitude before adjustment (if changed)
        metrics_before: The splat's metrics before adjustment
    """
    change = SplatChange(
        splat_id=splat.id,
        level=splat.level,
        change_type=AdaptationType.ADJUST,
        position_before=position_before,
        position_after=position_before is not None and splat.position.copy(),
        amplitude_before=amplitude_before,
        amplitude_after=amplitude_before is not None and splat.amplitude,
        metrics_before=metrics_before,
    )
    result.add_change(change)


def record_mitosis(
    result: AdaptationResult,
    parent_splat: Splat,
    child_splats: List[Splat],
    metrics_before: Optional[Dict[str, float]] = None
) -> None:
    """
    Record a mitosis (division) operation in the adaptation result.
    
    Args:
        result: The adaptation result to update
        parent_splat: The parent splat that divided
        child_splats: The resulting child splats
        metrics_before: The parent splat's metrics before division
    """
    change = SplatChange(
        splat_id=parent_splat.id,
        level=parent_splat.level,
        change_type=AdaptationType.MITOSIS,
        position_before=parent_splat.position.copy(),
        amplitude_before=parent_splat.amplitude,
        created_splat_ids=[child.id for child in child_splats],
        metrics_before=metrics_before,
    )
    result.add_change(change)


def record_birth(
    result: AdaptationResult,
    new_splat: Splat,
    parent_id: Optional[str] = None
) -> None:
    """
    Record a birth operation in the adaptation result.
    
    Args:
        result: The adaptation result to update
        new_splat: The newly created splat
        parent_id: Optional ID of the parent splat
    """
    change = SplatChange(
        splat_id=new_splat.id,
        level=new_splat.level,
        change_type=AdaptationType.BIRTH,
        position_after=new_splat.position.copy(),
        amplitude_after=new_splat.amplitude,
        parent_id=parent_id,
    )
    result.add_change(change)


def record_death(
    result: AdaptationResult,
    splat: Splat,
    metrics_before: Optional[Dict[str, float]] = None
) -> None:
    """
    Record a death operation in the adaptation result.
    
    Args:
        result: The adaptation result to update
        splat: The splat being removed
        metrics_before: The splat's metrics before removal
    """
    change = SplatChange(
        splat_id=splat.id,
        level=splat.level,
        change_type=AdaptationType.DEATH,
        position_before=splat.position.copy(),
        amplitude_before=splat.amplitude,
        metrics_before=metrics_before,
    )
    result.add_change(change)


def record_merge(
    result: AdaptationResult,
    target_splat: Splat,
    source_splat: Splat,
    position_before: np.ndarray,
    amplitude_before: float,
    metrics_before: Optional[Dict[str, float]] = None
) -> None:
    """
    Record a merge operation in the adaptation result.
    
    Args:
        result: The adaptation result to update
        target_splat: The splat that remains after merging
        source_splat: The splat being merged into the target
        position_before: The target splat's position before merging
        amplitude_before: The target splat's amplitude before merging
        metrics_before: The splats' metrics before merging
    """
    change = SplatChange(
        splat_id=target_splat.id,
        level=target_splat.level,
        change_type=AdaptationType.MERGE,
        position_before=position_before,
        position_after=target_splat.position.copy(),
        amplitude_before=amplitude_before,
        amplitude_after=target_splat.amplitude,
        related_splat_ids=[source_splat.id],
        metrics_before=metrics_before,
    )
    result.add_change(change)
