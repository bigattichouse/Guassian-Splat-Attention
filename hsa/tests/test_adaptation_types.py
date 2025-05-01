"""
Tests for the adaptation types and configuration in the HSA implementation.
"""

import pytest
from typing import Dict, List

from hsa.adaptation_types import (
    AdaptationType,
    AdaptationTrigger,
    AdaptationPhase,
    AdaptationTarget,
    AdaptationMetrics,
    AdaptationConfig,
    AdaptationResult
)


class TestAdaptationEnums:
    """Tests for the adaptation enumeration types."""
    
    def test_adaptation_type(self):
        """Test AdaptationType enum."""
        assert AdaptationType.MITOSIS.name == "MITOSIS"
        assert AdaptationType.BIRTH.name == "BIRTH"
        assert AdaptationType.DEATH.name == "DEATH"
        assert AdaptationType.MERGE.name == "MERGE"
        assert AdaptationType.ADJUST.name == "ADJUST"
        
        # Test that each value is unique
        values = [type.value for type in AdaptationType]
        assert len(values) == len(set(values))
    
    def test_adaptation_trigger(self):
        """Test AdaptationTrigger enum."""
        assert AdaptationTrigger.LOW_ACTIVATION.name == "LOW_ACTIVATION"
        assert AdaptationTrigger.HIGH_ACTIVATION.name == "HIGH_ACTIVATION"
        assert AdaptationTrigger.HIGH_VARIANCE.name == "HIGH_VARIANCE"
        assert AdaptationTrigger.HIGH_SIMILARITY.name == "HIGH_SIMILARITY"
        assert AdaptationTrigger.EMPTY_REGION.name == "EMPTY_REGION"
        assert AdaptationTrigger.INFORMATION_GAIN.name == "INFORMATION_GAIN"
        assert AdaptationTrigger.SCHEDULED.name == "SCHEDULED"
        assert AdaptationTrigger.MANUAL.name == "MANUAL"
        
        # Test that each value is unique
        values = [trigger.value for trigger in AdaptationTrigger]
        assert len(values) == len(set(values))
    
    def test_adaptation_phase(self):
        """Test AdaptationPhase enum."""
        assert AdaptationPhase.MEASUREMENT.name == "MEASUREMENT"
        assert AdaptationPhase.ANALYSIS.name == "ANALYSIS"
        assert AdaptationPhase.EXECUTION.name == "EXECUTION"
        assert AdaptationPhase.STABILIZATION.name == "STABILIZATION"
        
        # Test that each value is unique
        values = [phase.value for phase in AdaptationPhase]
        assert len(values) == len(set(values))


class TestAdaptationTarget:
    """Tests for the AdaptationTarget class."""
    
    def test_init(self):
        """Test initialization."""
        target = AdaptationTarget(
            splat_id="test_splat",
            secondary_splat_id="secondary_splat",
            adaptation_type=AdaptationType.MERGE,
            trigger=AdaptationTrigger.HIGH_SIMILARITY,
            parameters={"param1": 1.0, "param2": "value"}
        )
        
        assert target.splat_id == "test_splat"
        assert target.secondary_splat_id == "secondary_splat"
        assert target.adaptation_type == AdaptationType.MERGE
        assert target.trigger == AdaptationTrigger.HIGH_SIMILARITY
        assert target.parameters == {"param1": 1.0, "param2": "value"}
    
    def test_init_minimal(self):
        """Test initialization with minimal parameters."""
        target = AdaptationTarget(
            splat_id="test_splat",
            secondary_splat_id=None,
            adaptation_type=AdaptationType.DEATH,
            trigger=AdaptationTrigger.LOW_ACTIVATION,
            parameters={}
        )
        
        assert target.splat_id == "test_splat"
        assert target.secondary_splat_id is None
        assert target.adaptation_type == AdaptationType.DEATH
        assert target.trigger == AdaptationTrigger.LOW_ACTIVATION
        assert target.parameters == {}


class TestAdaptationMetrics:
    """Tests for the AdaptationMetrics class."""
    
    def test_init(self):
        """Test initialization."""
        metrics = AdaptationMetrics(
            activation_mean=0.5,
            activation_trend=0.1,
            information_contribution=0.3,
            coverage_uniformity=0.7,
            variance=0.2,
            similarity_to_others={"splat1": 0.8, "splat2": 0.3}
        )
        
        assert metrics.activation_mean == 0.5
        assert metrics.activation_trend == 0.1
        assert metrics.information_contribution == 0.3
        assert metrics.coverage_uniformity == 0.7
        assert metrics.variance == 0.2
        assert metrics.similarity_to_others == {"splat1": 0.8, "splat2": 0.3}


class TestAdaptationConfig:
    """Tests for the AdaptationConfig class."""
    
    def test_init_default(self):
        """Test initialization with default values."""
        config = AdaptationConfig()
        
        # Check default thresholds
        assert config.low_activation_threshold == 0.01
        assert config.high_activation_threshold == 0.8
        assert config.high_variance_threshold == 0.5
        assert config.high_similarity_threshold == 0.9
        
        # Check default probabilities
        assert config.mitosis_probability == 0.3
        assert config.death_probability == 0.2
        assert config.merge_probability == 0.2
        assert config.birth_probability == 0.2
        assert config.adjust_probability == 0.1
        
        # Probabilities should sum to 1
        total_prob = (
            config.mitosis_probability + config.death_probability + 
            config.merge_probability + config.birth_probability + 
            config.adjust_probability
        )
        assert pytest.approx(total_prob) == 1.0
        
        # Check other defaults
        assert config.adaptation_frequency == 100
        assert config.max_adaptations_per_cycle == 5
        assert config.min_lifetime_before_adaptation == 10
        assert config.level_specific_settings == {}
    
    def test_init_custom(self):
        """Test initialization with custom values."""
        level_settings = {
            "token": {
                "low_activation_threshold": 0.05,
                "death_probability": 0.3
            },
            "phrase": {
                "high_activation_threshold": 0.7,
                "mitosis_probability": 0.4
            }
        }
        
        config = AdaptationConfig(
            low_activation_threshold=0.02,
            high_activation_threshold=0.9,
            high_variance_threshold=0.6,
            high_similarity_threshold=0.95,
            mitosis_probability=0.25,
            death_probability=0.25,
            merge_probability=0.25,
            birth_probability=0.15,
            adjust_probability=0.1,
            adaptation_frequency=50,
            max_adaptations_per_cycle=10,
            min_lifetime_before_adaptation=5,
            level_specific_settings=level_settings
        )
        
        # Check custom thresholds
        assert config.low_activation_threshold == 0.02
        assert config.high_activation_threshold == 0.9
        assert config.high_variance_threshold == 0.6
        assert config.high_similarity_threshold == 0.95
        
        # Check custom probabilities
        assert config.mitosis_probability == 0.25
        assert config.death_probability == 0.25
        assert config.merge_probability == 0.25
        assert config.birth_probability == 0.15
        assert config.adjust_probability == 0.1
        
        # Probabilities should sum to 1
        total_prob = (
            config.mitosis_probability + config.death_probability + 
            config.merge_probability + config.birth_probability + 
            config.adjust_probability
        )
        assert pytest.approx(total_prob) == 1.0
        
        # Check other custom values
        assert config.adaptation_frequency == 50
        assert config.max_adaptations_per_cycle == 10
        assert config.min_lifetime_before_adaptation == 5
        assert config.level_specific_settings == level_settings
    
    def test_init_invalid_probabilities(self):
        """Test initialization with invalid probabilities."""
        # Total probability too high
        with pytest.raises(ValueError):
            AdaptationConfig(
                mitosis_probability=0.3,
                death_probability=0.3,
                merge_probability=0.3,
                birth_probability=0.3,
                adjust_probability=0.3
            )
        
        # Total probability too low
        with pytest.raises(ValueError):
            AdaptationConfig(
                mitosis_probability=0.1,
                death_probability=0.1,
                merge_probability=0.1,
                birth_probability=0.1,
                adjust_probability=0.1
            )
    
    def test_get_level_config(self):
        """Test getting level-specific configuration."""
        level_settings = {
            "token": {
                "low_activation_threshold": 0.05,
                "death_probability": 0.3
            }
        }
        
        config = AdaptationConfig(level_specific_settings=level_settings)
        
        # Get config for token level
        token_config = config.get_level_config("token")
        assert token_config == level_settings["token"]
        
        # Get config for non-existent level
        phrase_config = config.get_level_config("phrase")
        assert phrase_config == {}
    
    def test_get_threshold(self):
        """Test getting threshold values."""
        level_settings = {
            "token": {
                "low_activation_threshold": 0.05,
                "high_activation_threshold": 0.7
            }
        }
        
        config = AdaptationConfig(
            low_activation_threshold=0.02,
            level_specific_settings=level_settings
        )
        
        # Get default threshold
        threshold = config.get_threshold("low_activation_threshold")
        assert threshold == 0.02
        
        # Get level-specific threshold
        threshold = config.get_threshold("low_activation_threshold", "token")
        assert threshold == 0.05
        
        # Get default when level doesn't override
        threshold = config.get_threshold("high_variance_threshold", "token")
        assert threshold == 0.5  # default value
        
        # Get threshold for non-existent level
        threshold = config.get_threshold("low_activation_threshold", "phrase")
        assert threshold == 0.02  # default value
        
        # Invalid threshold name
        with pytest.raises(ValueError):
            config.get_threshold("invalid_threshold")
    
    def test_get_probability(self):
        """Test getting operation probabilities."""
        level_settings = {
            "token": {
                "death_probability": 0.3,
                "mitosis_probability": 0.2
            }
        }

        # Adjust other probability to ensure sum = 1.0
        config = AdaptationConfig(
            death_probability=0.25,
            mitosis_probability=0.25,  # Reduced from 0.3 to maintain sum = 1.0
            level_specific_settings=level_settings
        )
        
        # Get default probability
        prob = config.get_probability(AdaptationType.DEATH)
        assert prob == 0.25
        
        # Get level-specific probability
        prob = config.get_probability(AdaptationType.DEATH, "token")
        assert prob == 0.3
        
        # Get default when level doesn't override
        prob = config.get_probability(AdaptationType.MERGE, "token")
        assert prob == 0.2  # default value
        
        # Get probability for non-existent level
        prob = config.get_probability(AdaptationType.DEATH, "phrase")
        assert prob == 0.25  # default value


class TestAdaptationResult:
    """Tests for the AdaptationResult class."""
    
    def test_init(self):
        """Test initialization."""
        metrics_before = AdaptationMetrics(
            activation_mean=0.2,
            activation_trend=-0.1,
            information_contribution=0.1,
            coverage_uniformity=0.5,
            variance=0.3,
            similarity_to_others={}
        )
        
        metrics_after = AdaptationMetrics(
            activation_mean=0.3,
            activation_trend=0.1,
            information_contribution=0.2,
            coverage_uniformity=0.6,
            variance=0.2,
            similarity_to_others={}
        )
        
        result = AdaptationResult(
            success=True,
            adaptation_type=AdaptationType.MITOSIS,
            original_splat_id="original_splat",
            new_splat_ids=["new_splat1", "new_splat2"],
            removed_splat_ids=["original_splat"],
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            message="Splat split into two due to high activation"
        )
        
        assert result.success is True
        assert result.adaptation_type == AdaptationType.MITOSIS
        assert result.original_splat_id == "original_splat"
        assert result.new_splat_ids == ["new_splat1", "new_splat2"]
        assert result.removed_splat_ids == ["original_splat"]
        assert result.metrics_before == metrics_before
        assert result.metrics_after == metrics_after
        assert result.message == "Splat split into two due to high activation"
    
    def test_init_failed(self):
        """Test initialization for failed adaptation."""
        metrics = AdaptationMetrics(
            activation_mean=0.2,
            activation_trend=-0.1,
            information_contribution=0.1,
            coverage_uniformity=0.5,
            variance=0.3,
            similarity_to_others={}
        )
        
        result = AdaptationResult(
            success=False,
            adaptation_type=AdaptationType.MITOSIS,
            original_splat_id="original_splat",
            new_splat_ids=[],
            removed_splat_ids=[],
            metrics_before=metrics,
            metrics_after=None,
            message="Failed to split splat due to numerical instability"
        )
        
        assert result.success is False
        assert result.adaptation_type == AdaptationType.MITOSIS
        assert result.original_splat_id == "original_splat"
        assert result.new_splat_ids == []
        assert result.removed_splat_ids == []
        assert result.metrics_before == metrics
        assert result.metrics_after is None
        assert result.message == "Failed to split splat due to numerical instability"
