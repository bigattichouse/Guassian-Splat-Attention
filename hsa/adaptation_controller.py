"""
Adaptation controller for Hierarchical Splat Attention (HSA).

This module provides the main controller for adaptation operations in HSA,
orchestrating the adaptation process including measurement, analysis, and
execution of adaptation operations.
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Union
import numpy as np
import random
import logging

from .splat import Splat
from .registry import SplatRegistry
from .adaptation_types import (
    AdaptationType, AdaptationTrigger, AdaptationPhase, AdaptationTarget,
    AdaptationMetrics, AdaptationConfig, AdaptationResult
)
from .adaptation_metrics_base import (
    AdaptationMetricsComputer, SplatCandidateEvaluator, AdaptationMetricsAggregator
)

# Configure logging
logger = logging.getLogger(__name__)


class AdaptationController:
    """
    Main controller for adaptation operations in HSA.
    
    This class orchestrates the adaptation process, including measuring splat
    metrics, analyzing which adaptations to perform, and executing those adaptations.
    """
    
    def __init__(
        self,
        registry: SplatRegistry,
        metrics_computer: AdaptationMetricsComputer,
        candidate_evaluator: SplatCandidateEvaluator,
        config: Optional[AdaptationConfig] = None
    ):
        """Initialize adaptation controller.
        
        Args:
            registry: SplatRegistry to adapt
            metrics_computer: Computer for adaptation metrics
            candidate_evaluator: Evaluator for adaptation candidates
            config: Configuration for adaptation (if None, uses defaults)
        """
        self.registry = registry
        self.metrics_computer = metrics_computer
        self.candidate_evaluator = candidate_evaluator
        self.config = config or AdaptationConfig()
        
        # Create metrics aggregator
        self.metrics_aggregator = AdaptationMetricsAggregator(metrics_computer)
        
        # Adaptation statistics
        self.step_counter = 0
        self.adaptation_counter = {
            AdaptationType.MITOSIS: 0,
            AdaptationType.BIRTH: 0,
            AdaptationType.DEATH: 0,
            AdaptationType.MERGE: 0,
            AdaptationType.ADJUST: 0
        }
        self.last_adaptation_step = 0
        self.adaptation_history: List[AdaptationResult] = []
        
        # Cache for metrics
        self.metrics_cache: Dict[str, AdaptationMetrics] = {}
    
    def should_adapt(self) -> bool:
        """Check if it's time to perform adaptation.
        
        Returns:
            True if adaptation should be performed, False otherwise
        """
        # Check if enough steps have passed since last adaptation
        steps_since_last = self.step_counter - self.last_adaptation_step
        return steps_since_last >= self.config.adaptation_frequency
    
    def step(self, tokens: Optional[np.ndarray] = None) -> List[AdaptationResult]:
        """Perform one adaptation step.
        
        This should be called regularly during model usage to allow the HSA
        structure to adapt.
        
        Args:
            tokens: Optional token embeddings for context-aware adaptation
            
        Returns:
            List of adaptation results (empty if no adaptations performed)
        """
        self.step_counter += 1
        
        # Check if we should adapt
        if not self.should_adapt():
            return []
        
        # Perform adaptation cycle
        results = self.execute_adaptation_cycle(tokens)
        
        # Update statistics
        self.last_adaptation_step = self.step_counter
        self.adaptation_history.extend(results)
        
        # Limit history size
        if len(self.adaptation_history) > 1000:
            self.adaptation_history = self.adaptation_history[-1000:]
        
        return results
    
    def execute_adaptation_cycle(
        self,
        tokens: Optional[np.ndarray] = None
    ) -> List[AdaptationResult]:
        """Execute a complete adaptation cycle.
        
        Args:
            tokens: Optional token embeddings for context-aware adaptation
            
        Returns:
            List of adaptation results
        """
        results = []
        
        try:
            # Phase 1: Measurement
            logger.info("Adaptation cycle - Phase 1: Measurement")
            self.metrics_cache = self.metrics_aggregator.compute_all_metrics(
                self.registry, tokens
            )
            
            # Phase 2: Analysis
            logger.info("Adaptation cycle - Phase 2: Analysis")
            adaptation_targets = self._analyze_metrics()
            
            # Limit the number of adaptations per cycle
            if len(adaptation_targets) > self.config.max_adaptations_per_cycle:
                # Prioritize targets
                adaptation_targets = self._prioritize_targets(
                    adaptation_targets, 
                    self.config.max_adaptations_per_cycle
                )
            
            logger.info(f"Selected {len(adaptation_targets)} adaptation targets")
            
            # Phase 3: Execution
            logger.info("Adaptation cycle - Phase 3: Execution")
            for target in adaptation_targets:
                result = self._execute_adaptation(target, tokens)
                results.append(result)
                
                # Update statistics
                if result.success:
                    self.adaptation_counter[target.adaptation_type] += 1
            
            # Phase 4: Stabilization
            logger.info("Adaptation cycle - Phase 4: Stabilization")
            # Nothing to do here currently - just allow system to stabilize
            
            # Clear metrics cache after adaptation
            self.metrics_cache = {}
            
            return results
            
        except Exception as e:
            logger.error(f"Error during adaptation cycle: {e}")
            return results
    
    def _analyze_metrics(self) -> List[AdaptationTarget]:
        """Analyze metrics and determine adaptation targets.
        
        Returns:
            List of adaptation targets
        """
        targets = []
        
        # Check for splats with low activation (death candidates)
        death_candidates = self._identify_death_candidates()
        for splat, _ in death_candidates:
            targets.append(AdaptationTarget(
                splat_id=splat.id,
                secondary_splat_id=None,
                adaptation_type=AdaptationType.DEATH,
                trigger=AdaptationTrigger.LOW_ACTIVATION,
                parameters={}
            ))
        
        # Check for splats with high activation/variance (mitosis candidates)
        mitosis_candidates = self._identify_mitosis_candidates()
        for splat, _, _ in mitosis_candidates:
            targets.append(AdaptationTarget(
                splat_id=splat.id,
                secondary_splat_id=None,
                adaptation_type=AdaptationType.MITOSIS,
                trigger=AdaptationTrigger.HIGH_ACTIVATION,
                parameters={}
            ))
        
        # Check for similar splats (merge candidates)
        merge_candidates = self._identify_merge_candidates()
        for splat_a, splat_b, _ in merge_candidates:
            targets.append(AdaptationTarget(
                splat_id=splat_a.id,
                secondary_splat_id=splat_b.id,
                adaptation_type=AdaptationType.MERGE,
                trigger=AdaptationTrigger.HIGH_SIMILARITY,
                parameters={}
            ))
        
        # Check for empty regions (birth candidates)
        birth_regions = self._identify_empty_regions()
        for region in birth_regions:
            targets.append(AdaptationTarget(
                splat_id="",  # No specific splat for birth
                secondary_splat_id=None,
                adaptation_type=AdaptationType.BIRTH,
                trigger=AdaptationTrigger.EMPTY_REGION,
                parameters={"position": region.tolist()}
            ))
        
        # Add some random adjustments for exploration
        adjustment_candidates = self._identify_adjustment_candidates()
        for splat in adjustment_candidates:
            targets.append(AdaptationTarget(
                splat_id=splat.id,
                secondary_splat_id=None,
                adaptation_type=AdaptationType.ADJUST,
                trigger=AdaptationTrigger.SCHEDULED,
                parameters={}
            ))
        
        return targets
    
    def _prioritize_targets(
        self,
        targets: List[AdaptationTarget],
        max_count: int
    ) -> List[AdaptationTarget]:
        """Prioritize adaptation targets to fit within max_count.
        
        Args:
            targets: List of all potential adaptation targets
            max_count: Maximum number of targets to select
            
        Returns:
            Prioritized list of targets
        """
        # Group targets by type
        targets_by_type = {
            adaptation_type: [] for adaptation_type in AdaptationType
        }
        
        for target in targets:
            targets_by_type[target.adaptation_type].append(target)
        
        # Calculate how many of each type to include based on configured probabilities
        type_counts = {}
        remaining = max_count
        
        for adaptation_type in AdaptationType:
            # Get probability for this type
            probability = getattr(
                self.config,
                f"{adaptation_type.name.lower()}_probability"
            )
            
            # Calculate target count
            count = int(max_count * probability)
            
            # Ensure we don't exceed available targets or remaining slots
            available = len(targets_by_type[adaptation_type])
            count = min(count, available, remaining)
            
            type_counts[adaptation_type] = count
            remaining -= count
        
        # Distribute any remaining slots proportionally
        if remaining > 0:
            # Calculate total probability of types with available targets
            total_prob = 0.0
            for adaptation_type in AdaptationType:
                if (len(targets_by_type[adaptation_type]) > type_counts[adaptation_type]):
                    total_prob += getattr(
                        self.config,
                        f"{adaptation_type.name.lower()}_probability"
                    )
            
            # Distribute remaining slots
            for adaptation_type in AdaptationType:
                if remaining <= 0:
                    break
                    
                if len(targets_by_type[adaptation_type]) <= type_counts[adaptation_type]:
                    continue
                    
                # Calculate fair share based on probability
                probability = getattr(
                    self.config,
                    f"{adaptation_type.name.lower()}_probability"
                )
                
                if total_prob > 0:
                    fair_share = int(remaining * (probability / total_prob))
                else:
                    fair_share = 0
                
                # Ensure we don't exceed available targets
                available = len(targets_by_type[adaptation_type]) - type_counts[adaptation_type]
                additional = min(fair_share, available, remaining)
                
                type_counts[adaptation_type] += additional
                remaining -= additional
        
        # If we still have remaining slots, just fill them with whatever is available
        if remaining > 0:
            for adaptation_type in AdaptationType:
                if remaining <= 0:
                    break
                    
                available = len(targets_by_type[adaptation_type]) - type_counts[adaptation_type]
                if available > 0:
                    additional = min(available, remaining)
                    type_counts[adaptation_type] += additional
                    remaining -= additional
        
        # Select the targets
        selected_targets = []
        
        for adaptation_type in AdaptationType:
            count = type_counts[adaptation_type]
            if count <= 0:
                continue
                
            candidates = targets_by_type[adaptation_type]
            
            # Sort candidates (implementation-specific)
            if adaptation_type == AdaptationType.DEATH:
                # Prioritize lowest activation
                candidates.sort(key=lambda t: self._get_splat_activation(t.splat_id))
            elif adaptation_type == AdaptationType.MITOSIS:
                # Prioritize highest activation
                candidates.sort(
                    key=lambda t: self._get_splat_activation(t.splat_id),
                    reverse=True
                )
            
            # Select top candidates
            selected = candidates[:count]
            selected_targets.extend(selected)
        
        return selected_targets
    
    def _get_splat_activation(self, splat_id: str) -> float:
        """Get activation value for a splat.
        
        Args:
            splat_id: ID of the splat
            
        Returns:
            Activation value (0.0 if splat not found)
        """
        try:
            splat = self.registry.get_splat(splat_id)
            return splat.get_average_activation()
        except ValueError:
            return 0.0
    
    def _identify_death_candidates(self) -> List[Tuple[Splat, float]]:
        """Identify splats that are candidates for removal.
        
        Returns:
            List of (splat, activation) tuples sorted by activation (lowest first)
        """
        # Get configured threshold
        threshold = self.config.low_activation_threshold
        min_lifetime = self.config.min_lifetime_before_adaptation
        
        return self.metrics_aggregator.find_splats_for_death(
            self.registry, threshold, min_lifetime
        )
    
    def _identify_mitosis_candidates(self) -> List[Tuple[Splat, float, float]]:
        """Identify splats that are candidates for splitting.
        
        Returns:
            List of (splat, activation, variance) tuples
        """
        # Get configured thresholds
        activation_threshold = self.config.high_activation_threshold
        variance_threshold = self.config.high_variance_threshold
        min_lifetime = self.config.min_lifetime_before_adaptation
        
        return self.metrics_aggregator.find_splats_for_mitosis(
            self.registry, activation_threshold, variance_threshold, min_lifetime
        )
    
    def _identify_merge_candidates(self) -> List[Tuple[Splat, Splat, float]]:
        """Identify pairs of splats that are candidates for merging.
        
        Returns:
            List of (splat_a, splat_b, similarity) tuples
        """
        # Get configured threshold
        threshold = self.config.high_similarity_threshold
        
        return self.metrics_aggregator.find_similar_splats(
            self.registry, threshold, same_level_only=True
        )
    
    def _identify_empty_regions(self) -> List[np.ndarray]:
        """Identify regions in embedding space with no splat coverage.
        
        Returns:
            List of positions (centers of empty regions)
        """
        # This is a placeholder implementation - in production, use more
        # sophisticated methods for identifying empty regions
        
        # For now, just generate some random positions
        # In a real implementation, this would analyze the actual token distribution
        # and find uncovered regions
        
        n_empty_regions = 3  # Arbitrary choice
        dim = self.registry.embedding_dim
        
        empty_regions = []
        for _ in range(n_empty_regions):
            # Generate random position
            position = np.random.normal(0, 1.0, dim)
            
            # Normalize the vector (optional)
            norm = np.linalg.norm(position)
            if norm > 0:
                position = position / norm
            
            empty_regions.append(position)
        
        return empty_regions
    
    def _identify_adjustment_candidates(self) -> List[Splat]:
        """Identify splats that are candidates for parameter adjustment.
        
        Returns:
            List of splats to adjust
        """
        # Simple implementation: select a few random splats
        # In production, use more sophisticated criteria
        
        # Get all splats that have lived long enough
        min_lifetime = self.config.min_lifetime_before_adaptation
        candidates = [
            splat for splat in self.registry.get_all_splats()
            if splat.lifetime >= min_lifetime
        ]
        
        # Randomly select a few
        n_adjust = min(3, len(candidates))
        if n_adjust == 0:
            return []
            
        return random.sample(candidates, n_adjust)
    
    def _execute_adaptation(
        self,
        target: AdaptationTarget,
        tokens: Optional[np.ndarray] = None
    ) -> AdaptationResult:
        """Execute a specific adaptation operation.
        
        Args:
            target: Adaptation target to execute
            tokens: Optional token embeddings for context-aware adaptation
            
        Returns:
            Result of the adaptation operation
        """
        adaptation_type = target.adaptation_type
        
        # Get metrics before adaptation
        metrics_before = None
        
        try:
            if target.splat_id:
                splat = self.registry.get_splat(target.splat_id)
                metrics_before = self.metrics_cache.get(
                    target.splat_id,
                    self.metrics_computer.compute_metrics(splat, self.registry, tokens)
                )
        except ValueError:
            # Splat not found - might have been removed already
            logger.warning(f"Splat {target.splat_id} not found for adaptation")
            return AdaptationResult(
                success=False,
                adaptation_type=adaptation_type,
                original_splat_id=target.splat_id,
                new_splat_ids=[],
                removed_splat_ids=[],
                metrics_before=None,
                metrics_after=None,
                message=f"Splat {target.splat_id} not found"
            )
        
        # Execute the appropriate adaptation type
        if adaptation_type == AdaptationType.MITOSIS:
            return self.perform_mitosis(target, metrics_before, tokens)
        elif adaptation_type == AdaptationType.BIRTH:
            return self.perform_birth(target, tokens)
        elif adaptation_type == AdaptationType.DEATH:
            return self.perform_death(target, metrics_before)
        elif adaptation_type == AdaptationType.MERGE:
            return self.perform_merge(target, metrics_before, tokens)
        elif adaptation_type == AdaptationType.ADJUST:
            return self.perform_adjust(target, metrics_before, tokens)
        else:
            return AdaptationResult(
                success=False,
                adaptation_type=adaptation_type,
                original_splat_id=target.splat_id,
                new_splat_ids=[],
                removed_splat_ids=[],
                metrics_before=metrics_before,
                metrics_after=None,
                message=f"Unknown adaptation type: {adaptation_type}"
            )
    
    def perform_mitosis(
        self,
        target: AdaptationTarget,
        metrics_before: Optional[AdaptationMetrics],
        tokens: Optional[np.ndarray] = None
    ) -> AdaptationResult:
        """Perform mitosis (splitting) operation on a splat.
        
        Args:
            target: Adaptation target
            metrics_before: Metrics before adaptation
            tokens: Optional token embeddings
            
        Returns:
            Result of the operation
        """
        from . import mitosis  # Avoid circular import
        
        try:
            splat = self.registry.get_splat(target.splat_id)
            
            # Use mitosis module to generate candidate splats
            candidates = mitosis.generate_mitosis_candidates(splat)
            
            if not candidates:
                return AdaptationResult(
                    success=False,
                    adaptation_type=AdaptationType.MITOSIS,
                    original_splat_id=target.splat_id,
                    new_splat_ids=[],
                    removed_splat_ids=[],
                    metrics_before=metrics_before,
                    metrics_after=None,
                    message="No valid mitosis candidates generated"
                )
            
            # Evaluate candidates using the provided evaluator
            best_pair = self.candidate_evaluator.evaluate_mitosis_candidates(
                splat, candidates, self.registry, tokens
            )
            
            # Replace the original splat with the new pair
            self.registry.replace_splat(splat, list(best_pair))
            
            # Compute metrics after adaptation
            metrics_after = None
            if metrics_before is not None:
                # Use average of metrics for the two new splats
                metrics_a = self.metrics_computer.compute_metrics(
                    best_pair[0], self.registry, tokens
                )
                metrics_b = self.metrics_computer.compute_metrics(
                    best_pair[1], self.registry, tokens
                )
                
                # This is a simplification - in production, implement proper
                # metrics aggregation for the split splats
                metrics_after = metrics_a  # Just use one for now
            
            return AdaptationResult(
                success=True,
                adaptation_type=AdaptationType.MITOSIS,
                original_splat_id=target.splat_id,
                new_splat_ids=[best_pair[0].id, best_pair[1].id],
                removed_splat_ids=[target.splat_id],
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                message=f"Mitosis: split splat {target.splat_id} into {best_pair[0].id} and {best_pair[1].id}"
            )
            
        except Exception as e:
            logger.error(f"Error during mitosis: {e}")
            return AdaptationResult(
                success=False,
                adaptation_type=AdaptationType.MITOSIS,
                original_splat_id=target.splat_id,
                new_splat_ids=[],
                removed_splat_ids=[],
                metrics_before=metrics_before,
                metrics_after=None,
                message=f"Mitosis failed: {str(e)}"
            )
    
    def perform_birth(
        self,
        target: AdaptationTarget,
        tokens: Optional[np.ndarray] = None
    ) -> AdaptationResult:
        """Perform birth operation (create a new splat).
        
        Args:
            target: Adaptation target
            tokens: Optional token embeddings
            
        Returns:
            Result of the operation
        """
        from . import birth  # Avoid circular import
        
        try:
            # Get target position from parameters (if provided)
            position = None
            if "position" in target.parameters:
                position = np.array(target.parameters["position"])
            
            # Get target level (use lowest level by default)
            level = target.parameters.get("level", self.registry.hierarchy.levels[0])
            
            # Use birth module to generate candidate splats
            candidates = birth.generate_birth_candidates(
                self.registry, level, position, tokens
            )
            
            if not candidates:
                return AdaptationResult(
                    success=False,
                    adaptation_type=AdaptationType.BIRTH,
                    original_splat_id="",
                    new_splat_ids=[],
                    removed_splat_ids=[],
                    metrics_before=None,
                    metrics_after=None,
                    message="No valid birth candidates generated"
                )
            
            # Evaluate candidates using the provided evaluator
            best_splat = self.candidate_evaluator.evaluate_birth_candidates(
                candidates, self.registry, tokens
            )
            
            # Add the new splat to the registry
            self.registry.register(best_splat)
            
            # No need to compute metrics before (this is a new splat)
            # Compute metrics after birth
            metrics_after = self.metrics_computer.compute_metrics(
                best_splat, self.registry, tokens
            )
            
            return AdaptationResult(
                success=True,
                adaptation_type=AdaptationType.BIRTH,
                original_splat_id="",
                new_splat_ids=[best_splat.id],
                removed_splat_ids=[],
                metrics_before=None,
                metrics_after=metrics_after,
                message=f"Birth: created new splat {best_splat.id} at level {level}"
            )
            
        except Exception as e:
            logger.error(f"Error during birth: {e}")
            return AdaptationResult(
                success=False,
                adaptation_type=AdaptationType.BIRTH,
                original_splat_id="",
                new_splat_ids=[],
                removed_splat_ids=[],
                metrics_before=None,
                metrics_after=None,
                message=f"Birth failed: {str(e)}"
            )
    
    def perform_death(
        self,
        target: AdaptationTarget,
        metrics_before: Optional[AdaptationMetrics]
    ) -> AdaptationResult:
        """Perform death operation (remove a splat).
        
        Args:
            target: Adaptation target
            metrics_before: Metrics before adaptation
            
        Returns:
            Result of the operation
        """
        try:
            splat = self.registry.get_splat(target.splat_id)
            
            # Remove the splat from the registry
            self.registry.unregister(splat)
            
            # No metrics after (splat is gone)
            
            return AdaptationResult(
                success=True,
                adaptation_type=AdaptationType.DEATH,
                original_splat_id=target.splat_id,
                new_splat_ids=[],
                removed_splat_ids=[target.splat_id],
                metrics_before=metrics_before,
                metrics_after=None,
                message=f"Death: removed splat {target.splat_id}"
            )
            
        except Exception as e:
            logger.error(f"Error during death: {e}")
            return AdaptationResult(
                success=False,
                adaptation_type=AdaptationType.DEATH,
                original_splat_id=target.splat_id,
                new_splat_ids=[],
                removed_splat_ids=[],
                metrics_before=metrics_before,
                metrics_after=None,
                message=f"Death failed: {str(e)}"
            )
    
    def perform_merge(
        self,
        target: AdaptationTarget,
        metrics_before: Optional[AdaptationMetrics],
        tokens: Optional[np.ndarray] = None
    ) -> AdaptationResult:
        """Perform merge operation (combine two splats).
        
        Args:
            target: Adaptation target
            metrics_before: Metrics before adaptation
            tokens: Optional token embeddings
            
        Returns:
            Result of the operation
        """
        from . import merge  # Avoid circular import
        
        try:
            # Ensure we have both splats
            if not target.secondary_splat_id:
                return AdaptationResult(
                    success=False,
                    adaptation_type=AdaptationType.MERGE,
                    original_splat_id=target.splat_id,
                    new_splat_ids=[],
                    removed_splat_ids=[],
                    metrics_before=metrics_before,
                    metrics_after=None,
                    message="Merge requires a secondary splat ID"
                )
            
            splat_a = self.registry.get_splat(target.splat_id)
            splat_b = self.registry.get_splat(target.secondary_splat_id)
            
            # Use merge module to generate candidate splats
            candidates = merge.generate_merge_candidates(splat_a, splat_b)
            
            if not candidates:
                return AdaptationResult(
                    success=False,
                    adaptation_type=AdaptationType.MERGE,
                    original_splat_id=target.splat_id,
                    new_splat_ids=[],
                    removed_splat_ids=[],
                    metrics_before=metrics_before,
                    metrics_after=None,
                    message="No valid merge candidates generated"
                )
            
            # Evaluate candidates using the provided evaluator
            best_splat = self.candidate_evaluator.evaluate_merge_candidates(
                splat_a, splat_b, candidates, self.registry, tokens
            )
            
            # Replace the original splats with the merged one
            # Remove both originals
            self.registry.unregister(splat_a)
            self.registry.unregister(splat_b)
            
            # Add the merged splat
            self.registry.register(best_splat)
            
            # Compute metrics after merge
            metrics_after = self.metrics_computer.compute_metrics(
                best_splat, self.registry, tokens
            )
            
            return AdaptationResult(
                success=True,
                adaptation_type=AdaptationType.MERGE,
                original_splat_id=target.splat_id,
                new_splat_ids=[best_splat.id],
                removed_splat_ids=[target.splat_id, target.secondary_splat_id],
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                message=f"Merge: combined splats {target.splat_id} and {target.secondary_splat_id} into {best_splat.id}"
            )
            
        except Exception as e:
            logger.error(f"Error during merge: {e}")
            return AdaptationResult(
                success=False,
                adaptation_type=AdaptationType.MERGE,
                original_splat_id=target.splat_id,
                new_splat_ids=[],
                removed_splat_ids=[],
                metrics_before=metrics_before,
                metrics_after=None,
                message=f"Merge failed: {str(e)}"
            )
    
    def perform_adjust(
        self,
        target: AdaptationTarget,
        metrics_before: Optional[AdaptationMetrics],
        tokens: Optional[np.ndarray] = None
    ) -> AdaptationResult:
        """Perform adjust operation (modify splat parameters).
        
        Args:
            target: Adaptation target
            metrics_before: Metrics before adaptation
            tokens: Optional token embeddings
            
        Returns:
            Result of the operation
        """
        try:
            splat = self.registry.get_splat(target.splat_id)
            
            # Generate random adjustments for now
            # In production, use more sophisticated methods
            position_noise = np.random.normal(0, 0.1, splat.dim)
            new_position = splat.position + position_noise
            
            # Create adjusted splat
            adjusted_splat = splat.clone()
            adjusted_splat.update_parameters(position=new_position)
            
            # Evaluate the adjustment
            candidates = [adjusted_splat]
            best_splat = self.candidate_evaluator.evaluate_adjust_candidates(
                splat, candidates, self.registry, tokens
            )
            
            # Update the splat parameters
            splat.update_parameters(position=best_splat.position)
            
            # Compute metrics after adjustment
            metrics_after = self.metrics_computer.compute_metrics(
                splat, self.registry, tokens
            )
            
            return AdaptationResult(
                success=True,
                adaptation_type=AdaptationType.ADJUST,
                original_splat_id=target.splat_id,
                new_splat_ids=[],
                removed_splat_ids=[],
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                message=f"Adjust: updated parameters for splat {target.splat_id}"
            )
            
        except Exception as e:
            logger.error(f"Error during adjustment: {e}")
            return AdaptationResult(
                success=False,
                adaptation_type=AdaptationType.ADJUST,
                original_splat_id=target.splat_id,
                new_splat_ids=[],
                removed_splat_ids=[],
                metrics_before=metrics_before,
                metrics_after=None,
                message=f"Adjustment failed: {str(e)}"
            )
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get statistics about adaptation operations.
        
        Returns:
            Dictionary with adaptation statistics
        """
        return {
            "total_steps": self.step_counter,
            "last_adaptation_step": self.last_adaptation_step,
            "adaptation_counts": {
                adaptation_type.name: count
                for adaptation_type, count in self.adaptation_counter.items()
            },
            "total_adaptations": sum(self.adaptation_counter.values()),
            "recent_history": self.adaptation_history[-10:]  # Last 10 adaptations
        }
    
    def reset_statistics(self) -> None:
        """Reset adaptation statistics."""
        self.adaptation_counter = {
            AdaptationType.MITOSIS: 0,
            AdaptationType.BIRTH: 0,
            AdaptationType.DEATH: 0,
            AdaptationType.MERGE: 0,
            AdaptationType.ADJUST: 0
        }
        self.adaptation_history = []
