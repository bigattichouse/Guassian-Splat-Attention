"""
Failure recovery for Hierarchical Splat Attention (HSA).

This module provides functionality for recovering from various types of failures
and pathological configurations in the HSA structure and computations.
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
import numpy as np
import logging

from .splat import Splat
from .registry import SplatRegistry
from .failure_detection_types import FailureType
from .recovery_actions import (
    RecoveryAction, recover_numerical_instability,
    recover_empty_level, recover_orphaned_splats,
    recover_adaptation_stagnation
)
from .recovery_utils import create_random_splats, repair_covariance_matrices

# Configure logging
logger = logging.getLogger(__name__)


class FailureRecovery:
    """Recovery mechanisms for various types of failures in HSA."""
    
    def __init__(
        self,
        registry: SplatRegistry,
        auto_recovery: bool = True,
        max_recovery_attempts: int = 3,
        repair_threshold: float = 0.7
    ):
        """Initialize failure recovery.
        
        Args:
            registry: SplatRegistry to recover
            auto_recovery: Whether to automatically attempt recovery
            max_recovery_attempts: Maximum number of recovery attempts per failure
            repair_threshold: Health score threshold below which to attempt repair
        """
        self.registry = registry
        self.auto_recovery = auto_recovery
        self.max_recovery_attempts = max_recovery_attempts
        self.repair_threshold = repair_threshold
        
        # Initialize failure detector
        from .failure_detection import FailureDetector
        self.detector = FailureDetector()
        
        # Track recovery attempts
        self.recovery_history: Dict[FailureType, List[Dict[str, Any]]] = {
            failure_type: [] for failure_type in FailureType
        }
        self.recovery_count = 0
        
        # Adaptation controller (set by external code if available)
        self.adaptation_controller = None
    
    def detect_and_recover(self) -> Dict[str, Any]:
        """Detect failures and recover if needed.
        
        Returns:
            Dictionary with recovery report
        """
        # First detect issues
        failures = self.detector.detect_pathological_configurations(self.registry)
        
        # Check health
        health = self.detector.categorize_registry_health(self.registry)
        
        # Initialize recovery report
        report = {
            "health_before": health,
            "failures_detected": len(failures),
            "recovery_performed": False,
            "recovery_actions": [],
            "health_after": health
        }
        
        # Return early if no failures or health is good enough
        if not failures or health["health_score"] >= self.repair_threshold:
            return report
        
        # Perform recovery if auto_recovery is enabled
        if self.auto_recovery:
            actions = self.recover(failures)
            
            # Update report
            report["recovery_performed"] = True
            report["recovery_actions"] = actions
            
            # Recheck health
            health_after = self.detector.categorize_registry_health(self.registry)
            report["health_after"] = health_after
        
        return report
    
    def recover(self, failures: List[Tuple[FailureType, str, Any]]) -> List[Dict[str, Any]]:
        """Recover from detected failures.
        
        Args:
            failures: List of (failure_type, message, data) tuples
            
        Returns:
            List of recovery action reports
        """
        self.recovery_count += 1
        actions = []
        
        # Group failures by type for more efficient recovery
        failures_by_type = {}
        for failure_type, message, data in failures:
            if failure_type not in failures_by_type:
                failures_by_type[failure_type] = []
            failures_by_type[failure_type].append((message, data))
        
        # Process each failure type
        for failure_type, failure_data in failures_by_type.items():
            # Skip if too many recovery attempts for this type
            type_history = self.recovery_history[failure_type]
            recent_attempts = sum(1 for h in type_history 
                               if h["recovery_count"] > self.recovery_count - 10)
            
            if recent_attempts >= self.max_recovery_attempts:
                logger.warning(
                    f"Skipping recovery for {failure_type.name}: " +
                    f"Too many recent attempts ({recent_attempts})"
                )
                continue
            
            # Perform type-specific recovery
            if failure_type == FailureType.NUMERICAL_INSTABILITY:
                action = recover_numerical_instability(self.registry, failure_data)
                if action:
                    actions.append(action)
                    
            elif failure_type == FailureType.EMPTY_LEVEL:
                action = recover_empty_level(self.registry, failure_data)
                if action:
                    actions.append(action)
                    
            elif failure_type == FailureType.ORPHANED_SPLAT:
                action = recover_orphaned_splats(self.registry, failure_data)
                if action:
                    actions.append(action)
                    
            elif failure_type == FailureType.ADAPTATION_STAGNATION:
                # Import from recovery_actions to ensure it's properly imported
                from .recovery_actions import recover_adaptation_stagnation
                action = recover_adaptation_stagnation(
                    self.registry, 
                    self.adaptation_controller,
                    failure_data
                )
                if action:
                    actions.append(action)
                    
            elif failure_type == FailureType.ATTENTION_COLLAPSE:
                action = self._recover_attention_collapse(failure_data)
                if action:
                    actions.append(action)
                    
            elif failure_type == FailureType.INFORMATION_BOTTLENECK:
                action = self._recover_information_bottleneck(failure_data)
                if action:
                    actions.append(action)
                    
            elif failure_type == FailureType.MEMORY_OVERFLOW:
                action = self._recover_memory_overflow(failure_data)
                if action:
                    actions.append(action)
            
            # Update recovery history
            self.recovery_history[failure_type].append({
                "recovery_count": self.recovery_count,
                "failure_data": failure_data,
                "action": action["action"] if action else "none"
            })
        
        return actions
    
    def _recover_attention_collapse(
        self, 
        failures: List[Tuple[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Recover from attention collapse by adjusting amplitudes and splats.
        
        Args:
            failures: List of (message, data) tuples
            
        Returns:
            Recovery action report or None if no action taken
        """
        affected_levels = []
        for _, data in failures:
            if isinstance(data, dict) and "level" in data:
                affected_levels.append(data["level"])
        
        if not affected_levels:
            return None
        
        # Perform recovery
        adjustments = 0
        
        for level in affected_levels:
            # Get splats at this level
            level_splats = list(self.registry.get_splats_at_level(level))
            
            if not level_splats:
                continue
                
            # Adjust amplitudes to prevent collapse
            for splat in level_splats:
                # Reset amplitude to default
                old_amplitude = splat.amplitude
                new_amplitude = 1.0
                
                # Update if needed
                if abs(old_amplitude - new_amplitude) > 0.1:
                    splat.update_parameters(amplitude=new_amplitude)
                    adjustments += 1
                    
                # Also consider splitting high-variance splats to improve coverage
                if splat.get_average_activation() < 0.05:
                    try:
                        # Import mitosis module
                        from .mitosis import perform_mitosis
                        result = perform_mitosis(self.registry, splat.id)
                        if result:
                            adjustments += 1
                    except Exception:
                        pass
        
        if adjustments > 0:
            return {
                "action": "reset_amplitudes",
                "levels_affected": affected_levels,
                "adjustments_made": adjustments
            }
        
        return None
    
    def _recover_information_bottleneck(
        self, 
        failures: List[Tuple[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Recover from information bottlenecks by rebalancing levels.
        
        Args:
            failures: List of (message, data) tuples
            
        Returns:
            Recovery action report or None if no action taken
        """
        bottleneck_data = []
        for _, data in failures:
            if isinstance(data, dict) and "lower_level" in data and "higher_level" in data:
                bottleneck_data.append(data)
        
        if not bottleneck_data:
            return None
        
        # Perform recovery
        rebalanced_levels = []
        total_created = 0
        
        for data in bottleneck_data:
            lower_level = data["lower_level"]
            higher_level = data["higher_level"]
            
            # Create more splats at the higher level
            # The ideal ratio is about 3-5 lower splats per higher splat
            lower_count = data["lower_count"]
            higher_count = data["higher_count"]
            
            target_higher_count = max(higher_count, lower_count // 4)
            to_create = target_higher_count - higher_count
            
            if to_create <= 0:
                continue
                
            # Create new splats at higher level
            created = create_random_splats(self.registry, higher_level, to_create)
            
            if created > 0:
                total_created += created
                rebalanced_levels.append(higher_level)
        
        if total_created > 0:
            return {
                "action": "rebalance_hierarchy",
                "levels_rebalanced": rebalanced_levels,
                "splats_created": total_created
            }
        
        return None
    
    def _recover_memory_overflow(
        self, 
        failures: List[Tuple[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Recover from memory overflow by pruning splats.
        
        Args:
            failures: List of (message, data) tuples
            
        Returns:
            Recovery action report or None if no action taken
        """
        # Extract total splat count
        total_splats = 0
        for _, data in failures:
            if isinstance(data, dict) and "total_splats" in data:
                total_splats = data["total_splats"]
                break
        
        if total_splats == 0:
            return None
        
        # Target around 40% reduction
        target_count = int(total_splats * 0.6)
        current_count = total_splats
        to_remove = current_count - target_count
        
        if to_remove <= 0:
            return None
        
        # Import death module
        removed_count = 0
        removed_by_level = {}
        
        try:
            from .death import identify_death_candidates, perform_death
            
            # Find candidates for removal
            candidates = identify_death_candidates(
                self.registry, 
                activation_threshold=0.2,  # Be more aggressive
                min_lifetime=1,            # Include recent splats
                max_candidates=to_remove * 2  # Get more candidates than needed
            )
            
            # Remove splats
            for splat, _ in candidates:
                # Check if we've removed enough
                if removed_count >= to_remove:
                    break
                    
                try:
                    success = perform_death(self.registry, splat.id)
                    if success:
                        removed_count += 1
                        level = splat.level
                        if level not in removed_by_level:
                            removed_by_level[level] = 0
                        removed_by_level[level] += 1
                except Exception as e:
                    logger.error(f"Failed to remove splat {splat.id}: {e}")
            
            if removed_count > 0:
                return {
                    "action": "prune_splats",
                    "splats_removed": removed_count,
                    "by_level": removed_by_level,
                    "target_reduction": to_remove
                }
            
        except ImportError:
            logger.error("Could not import death module for pruning")
            
        # Fallback: simple removal
        if removed_count == 0:
            # Perform simple removal without death module
            all_splats = self.registry.get_all_splats()
            
            for i, splat in enumerate(all_splats):
                if i >= to_remove:
                    break
                    
                try:
                    self.registry.unregister(splat)
                    removed_count += 1
                    level = splat.level
                    if level not in removed_by_level:
                        removed_by_level[level] = 0
                    removed_by_level[level] += 1
                except Exception:
                    pass
            
            if removed_count > 0:
                return {
                    "action": "prune_splats",
                    "splats_removed": removed_count,
                    "by_level": removed_by_level,
                    "target_reduction": to_remove
                }
        
        return None
    
    def repair_integrity(self) -> bool:
        """Repair registry integrity.
        
        This is a wrapper around registry.repair_integrity() that provides
        more detailed error information.
        
        Returns:
            True if repair succeeded, False otherwise
        """
        try:
            # First check integrity
            is_valid = self.registry.verify_integrity()
            
            if is_valid:
                return True
                
            # Attempt repair
            repair_count = self.registry.repair_integrity()
            logger.info(f"Repaired {repair_count} integrity issues")
            
            # Check again
            is_valid_after = self.registry.verify_integrity()
            
            if not is_valid_after:
                logger.warning("Registry integrity still invalid after repair")
                
            return is_valid_after
            
        except Exception as e:
            logger.error(f"Error during integrity repair: {e}")
            return False
    
    def handle_computation_timeout(
        self, 
        partial_results: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Handle timeout during computation.
        
        Args:
            partial_results: Any partial results from the computation
            
        Returns:
            Dictionary with safe result
        """
        # Create a safe default result
        safe_result = {
            "timeout_handled": True,
            "result_type": "fallback",
            "partial_results": partial_results
        }
        
        # Check if we can still use some of the partial results
        if partial_results is not None:
            # Try to use partial attention matrix if available
            if isinstance(partial_results, np.ndarray):
                attention_matrix = partial_results
                
                # Validate matrix
                if len(attention_matrix.shape) == 2:
                    # Check for NaN or Inf
                    if not np.isnan(attention_matrix).any() and not np.isinf(attention_matrix).any():
                        # Normalize rows
                        row_sums = np.sum(attention_matrix, axis=1, keepdims=True)
                        # Avoid division by zero
                        row_sums = np.where(row_sums > 0, row_sums, 1.0)
                        normalized = attention_matrix / row_sums
                        
                        safe_result["result_type"] = "partial_normalized"
                        safe_result["normalized_attention"] = normalized
        
        return safe_result

    def switch_to_fallback_attention(self) -> Dict[str, Any]:
        """Switch to fallback attention mechanism.
        
        Returns:
            Fallback attention configuration
        """
        # Create configuration for fallback attention
        fallback_config = {
            "type": "fallback",
            "mechanism": "dense",  # Use dense attention as fallback
            "use_causal_mask": True,  # Safer default
            "normalize_rows": True,  # Ensure proper normalization
            "skip_splats": True,  # Skip splat computation entirely
            "recovery_active": True
        }
        
        logger.warning("Switched to fallback attention mechanism")
        
        return fallback_config

    def rebalance_hierarchy(self) -> bool:
        """Rebalance splat hierarchy to original configuration.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get original configuration
            hierarchy = self.registry.hierarchy
            target_levels = hierarchy.levels
            target_counts = hierarchy.init_splats_per_level
            
            # Get current counts
            current_counts = {}
            for level in target_levels:
                current_counts[level] = self.registry.count_splats(level)
            
            # Determine actions
            actions = []
            
            for i, level in enumerate(target_levels):
                current = current_counts[level]
                target = target_counts[i]
                
                if current < target:
                    # Need to create splats
                    actions.append({
                        "level": level,
                        "action": "create",
                        "count": target - current
                    })
                elif current > target * 1.5:  # Allow some excess
                    # Need to remove splats
                    actions.append({
                        "level": level,
                        "action": "remove",
                        "count": current - target
                    })
            
            # Execute actions
            for action in actions:
                level = action["level"]
                if action["action"] == "create":
                    create_random_splats(self.registry, level, action["count"])
                elif action["action"] == "remove":
                    # Remove excess splats
                    level_splats = list(self.registry.get_splats_at_level(level))
                    to_remove = min(action["count"], len(level_splats))
                    
                    try:
                        from .death import perform_death
                        for i in range(to_remove):
                            if i < len(level_splats):
                                perform_death(self.registry, level_splats[i].id)
                    except ImportError:
                        # Fallback without death module
                        for i in range(to_remove):
                            if i < len(level_splats):
                                self.registry.unregister(level_splats[i])
            
            # Check if we've executed any actions
            return len(actions) > 0
            
        except Exception as e:
            logger.error(f"Error during hierarchy rebalancing: {e}")
            return False
        
    def reset_to_initial_state(self) -> bool:
        """Reset the registry to its initial state.
        
        This is a more aggressive recovery that resets everything to default.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear the registry
            self.registry.splats.clear()
            for level_set in self.registry.splats_by_level.values():
                level_set.clear()
            
            # Reset counters
            self.registry.registered_count = 0
            self.registry.unregistered_count = 0
            self.registry.recovery_count = 0
            
            # Initialize splats
            try:
                from .birth import create_initial_splats
                create_initial_splats(self.registry)
            except ImportError:
                # Fallback: initialize_splats from registry
                self.registry.initialize_splats()
            
            # Check if initialization succeeded
            total_splats = self.registry.count_splats()
            return total_splats > 0
            
        except Exception as e:
            logger.error(f"Error during reset to initial state: {e}")
            return False

    def recover_from_invalid_state(self) -> bool:
        """Recover from an invalid registry state.
        
        This is a last resort recovery when the registry is in an invalid state.
        
        Returns:
            True if recovery succeeded, False otherwise
        """
        try:
            # First try repairing integrity
            if self.repair_integrity():
                return True
                
            # If repair fails, try rebalancing
            if self.rebalance_hierarchy():
                return True
                
            # If both fail, reset to initial state
            return self.reset_to_initial_state()
            
        except Exception as e:
            logger.error(f"Error during recovery from invalid state: {e}")
            
            # Last resort - full reset
            try:
                return self.reset_to_initial_state()
            except Exception:
                return False

    def _detect_adaptation_issues(
        self, 
        registry: SplatRegistry
    ) -> List[Tuple[FailureType, str, Any]]:
        """Detect issues in adaptation process.
        
        Args:
            registry: SplatRegistry to check
            
        Returns:
            List of (failure_type, message, data) tuples
        """
        issues = []
        
        # Check for adaptation stagnation using failure history
        for failure_type, history in self.recovery_history.items():
            # Skip if not enough history
            if len(history) < 5:
                continue
                
            # Check if the same issue keeps occurring
            recent_history = history[-5:]
            recent_issues = [h["failure_data"] for h in recent_history]
            
            # For certain types, check for repeated identical issues
            if failure_type in [FailureType.EMPTY_LEVEL, FailureType.ORPHANED_SPLAT]:
                # Count unique issues
                unique_issues = set()
                for issue in recent_issues:
                    if isinstance(issue, list):
                        for item in issue:
                            if isinstance(item, tuple) and len(item) >= 2:
                                _, data = item
                                if isinstance(data, dict):
                                    # Convert dict to frozenset of items for hashing
                                    unique_issues.add(frozenset(data.items()))
                
                # If we keep seeing the same issues, adaptation is not fixing them
                if len(unique_issues) == 1 and len(recent_issues) >= 3:
                    issues.append((
                        FailureType.ADAPTATION_STAGNATION,
                        f"Adaptation not fixing recurring {failure_type.name}",
                        {"failure_type": failure_type.name, "occurrences": len(recent_issues)}
                    ))
        
        # Check for memory overflow (too many splats)
        total_splats = len(registry.get_all_splats())
        if total_splats > 1000:  # Arbitrary threshold, adjust as needed
            issues.append((
                FailureType.MEMORY_OVERFLOW,
                f"Excessive number of splats: {total_splats}",
                {"total_splats": total_splats}
            ))
        
        return issues

    def get_health_report(self) -> Dict[str, Any]:
        """Get a comprehensive health report for the registry.
        
        Returns:
            Dictionary with health report
        """
        # Check for issues
        failures = self.detector.detect_pathological_configurations(self.registry)
        
        # Get health score
        health = self.detector.categorize_registry_health(self.registry)
        
        # Get level distribution
        level_counts = {}
        for level in self.registry.hierarchy.levels:
            level_counts[level] = self.registry.count_splats(level)
        
        # Compute level ratios
        level_ratios = {}
        for i in range(1, len(self.registry.hierarchy.levels)):
            higher_level = self.registry.hierarchy.levels[i]
            lower_level = self.registry.hierarchy.levels[i-1]
            
            higher_count = level_counts.get(higher_level, 0)
            lower_count = level_counts.get(lower_level, 0)
            
            if higher_count > 0:
                level_ratios[f"{lower_level}:{higher_level}"] = lower_count / higher_count
        
        # Generate detailed report
        report = {
            "health_score": health["health_score"],
            "health_category": health["category"],
            "total_splats": sum(level_counts.values()),
            "level_distribution": level_counts,
            "level_ratios": level_ratios,
            "failures_by_type": health["issues_by_type"],
            "needs_recovery": health["needs_repair"],
            "recommended_actions": []
        }
        
        # Recommend actions based on issues
        if health["needs_repair"]:
            if "NUMERICAL_INSTABILITY" in health["issues_by_type"]:
                report["recommended_actions"].append("repair_covariance")
                
            if "EMPTY_LEVEL" in health["issues_by_type"]:
                report["recommended_actions"].append("populate_level")
                
            if "ORPHANED_SPLAT" in health["issues_by_type"]:
                report["recommended_actions"].append("repair_relationships")
                
            if "ADAPTATION_STAGNATION" in health["issues_by_type"]:
                report["recommended_actions"].append("restart_adaptation")
                
            if "ATTENTION_COLLAPSE" in health["issues_by_type"]:
                report["recommended_actions"].append("reset_amplitudes")
                
            if "INFORMATION_BOTTLENECK" in health["issues_by_type"]:
                report["recommended_actions"].append("rebalance_hierarchy")
                
            if "MEMORY_OVERFLOW" in health["issues_by_type"]:
                report["recommended_actions"].append("prune_splats")
                
            if not report["recommended_actions"]:
                report["recommended_actions"].append("repair_integrity")
        
        return report
