"""
Adaptation type definitions for Hierarchical Splat Attention (HSA).

This module defines the enumeration types for adaptation operations in HSA.
"""

from enum import Enum, auto
from typing import Dict, Any, NamedTuple, Optional, List


class AdaptationType(Enum):
    """Types of adaptation operations in HSA."""
    MITOSIS = auto()  # Split one splat into two
    BIRTH = auto()    # Create a new splat
    DEATH = auto()    # Remove a splat
    MERGE = auto()    # Combine two splats
    ADJUST = auto()   # Modify parameters without structural change


class AdaptationTrigger(Enum):
    """Triggers for adaptation operations."""
    LOW_ACTIVATION = auto()      # Splat has low activation
    HIGH_ACTIVATION = auto()     # Splat has high activation
    HIGH_VARIANCE = auto()       # Splat has high internal variance
    HIGH_SIMILARITY = auto()     # Two splats are very similar
    EMPTY_REGION = auto()        # Region with no splat coverage
    INFORMATION_GAIN = auto()    # Expected information gain
    SCHEDULED = auto()           # Regular scheduled adaptation
    MANUAL = auto()              # Manually triggered adaptation


class AdaptationPhase(Enum):
    """Phases of the adaptation cycle."""
    MEASUREMENT = auto()         # Measure splat metrics
    ANALYSIS = auto()            # Analyze metrics and determine actions
    EXECUTION = auto()           # Execute adaptation operations
    STABILIZATION = auto()       # Allow system to stabilize after changes


class AdaptationTarget(NamedTuple):
    """Target for adaptation operation."""
    splat_id: str                        # ID of the primary splat
    secondary_splat_id: Optional[str]    # ID of secondary splat (for MERGE)
    adaptation_type: AdaptationType      # Type of adaptation to perform
    trigger: AdaptationTrigger           # What triggered this adaptation
    parameters: Dict[str, Any]           # Operation-specific parameters


class AdaptationMetrics(NamedTuple):
    """Metrics used for adaptation decisions."""
    activation_mean: float               # Mean activation over recent history
    activation_trend: float              # Trend of activation (positive/negative)
    information_contribution: float      # Information-theoretic contribution
    coverage_uniformity: float           # How uniform is the coverage
    variance: float                      # Internal variance of the splat
    similarity_to_others: Dict[str, float]  # Similarity to other splats


class AdaptationConfig:
    """Configuration for adaptation mechanisms."""
    
    def __init__(
        self,
        # Thresholds
        low_activation_threshold: float = 0.01,
        high_activation_threshold: float = 0.8,
        high_variance_threshold: float = 0.5,
        high_similarity_threshold: float = 0.9,
        
        # Operation probabilities
        mitosis_probability: float = 0.3,
        death_probability: float = 0.2,
        merge_probability: float = 0.2,
        birth_probability: float = 0.2,
        adjust_probability: float = 0.1,
        
        # Adaptation cycle settings
        adaptation_frequency: int = 100,  # How often to run adaptation (in steps)
        max_adaptations_per_cycle: int = 5,  # Maximum number of adaptations per cycle
        min_lifetime_before_adaptation: int = 10,  # Minimum splat lifetime before adaptation
        
        # Level-specific settings
        level_specific_settings: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """Initialize adaptation configuration.
        
        Args:
            low_activation_threshold: Threshold for low activation
            high_activation_threshold: Threshold for high activation
            high_variance_threshold: Threshold for high variance
            high_similarity_threshold: Threshold for high similarity
            mitosis_probability: Probability of mitosis operation
            death_probability: Probability of death operation
            merge_probability: Probability of merge operation
            birth_probability: Probability of birth operation
            adjust_probability: Probability of adjust operation
            adaptation_frequency: How often to run adaptation cycle
            max_adaptations_per_cycle: Maximum adaptations per cycle
            min_lifetime_before_adaptation: Minimum splat lifetime before adaptation
            level_specific_settings: Override settings for specific levels
        """
        # Thresholds
        self.low_activation_threshold = low_activation_threshold
        self.high_activation_threshold = high_activation_threshold
        self.high_variance_threshold = high_variance_threshold
        self.high_similarity_threshold = high_similarity_threshold
        
        # Operation probabilities
        self.mitosis_probability = mitosis_probability
        self.death_probability = death_probability
        self.merge_probability = merge_probability
        self.birth_probability = birth_probability
        self.adjust_probability = adjust_probability
        
        # Validate probabilities sum to approximately 1
        total_prob = (
            mitosis_probability + death_probability + merge_probability + 
            birth_probability + adjust_probability
        )
        if abs(total_prob - 1.0) > 0.01:
            raise ValueError(f"Operation probabilities must sum to 1, got {total_prob}")    
            
        
        # Adaptation cycle settings
        self.adaptation_frequency = adaptation_frequency
        self.max_adaptations_per_cycle = max_adaptations_per_cycle
        self.min_lifetime_before_adaptation = min_lifetime_before_adaptation
        
        # Level-specific settings
        self.level_specific_settings = level_specific_settings or {}
    
    def get_level_config(self, level: str) -> Dict[str, float]:
        """Get configuration values for a specific level.
        
        Args:
            level: Hierarchical level name
            
        Returns:
            Dictionary of configuration values specific to this level
        """
        return self.level_specific_settings.get(level, {})
    
    def get_threshold(self, threshold_name: str, level: Optional[str] = None) -> float:
        """Get a threshold value, potentially level-specific.
        
        Args:
            threshold_name: Name of the threshold to get
            level: Hierarchical level (if None, use default)
            
        Returns:
            Threshold value
            
        Raises:
            ValueError: If threshold name is invalid
        """
        if not hasattr(self, threshold_name):
            raise ValueError(f"Invalid threshold name: {threshold_name}")
        
        default_value = getattr(self, threshold_name)
        
        if level is None:
            return default_value
        
        level_settings = self.level_specific_settings.get(level, {})
        return level_settings.get(threshold_name, default_value)
    
    def get_probability(self, operation_type: AdaptationType, level: Optional[str] = None) -> float:
        """Get operation probability, potentially level-specific.
        
        Args:
            operation_type: Type of adaptation operation
            level: Hierarchical level (if None, use default)
            
        Returns:
            Probability value
        """
        prob_name = f"{operation_type.name.lower()}_probability"
        
        if not hasattr(self, prob_name):
            return 0.0
        
        default_value = getattr(self, prob_name)
        
        if level is None:
            return default_value
        
        level_settings = self.level_specific_settings.get(level, {})
        return level_settings.get(prob_name, default_value)


class AdaptationResult(NamedTuple):
    """Result of an adaptation operation."""
    success: bool                            # Whether the adaptation succeeded
    adaptation_type: AdaptationType          # Type of adaptation performed
    original_splat_id: str                   # ID of the original splat
    new_splat_ids: List[str]                 # IDs of any new splats created
    removed_splat_ids: List[str]             # IDs of any splats removed
    metrics_before: AdaptationMetrics        # Metrics before adaptation
    metrics_after: Optional[AdaptationMetrics]  # Metrics after adaptation
    message: str                             # Description of the operation
