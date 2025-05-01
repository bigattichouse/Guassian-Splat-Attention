"""
Hierarchical Splat Attention (HSA) package.

HSA is an attention mechanism that replaces standard O(n²) attention with a 
hierarchical, Gaussian-based approach for more efficient sequence processing.
"""

__version__ = "0.1.0"

from .splat import Splat, RingBuffer
from .hierarchy import Hierarchy
from .registry import SplatRegistry
from .attention_interface import AttentionComputer, AttentionConfig, AttentionResult
from .dense_attention import DenseAttentionComputer
from .adaptation_types import (
    AdaptationType, 
    AdaptationTrigger, 
    AdaptationPhase,
    AdaptationTarget,
    AdaptationMetrics,
    AdaptationConfig,
    AdaptationResult
)
from .adaptation_metrics_base import (
    AdaptationMetricsComputer,
    SplatCandidateEvaluator,
    AdaptationMetricsAggregator
)

__all__ = [
    "Splat",
    "RingBuffer",
    "Hierarchy",
    "SplatRegistry",
    "AttentionComputer",
    "AttentionConfig",
    "AttentionResult",
    "DenseAttentionComputer",
    "AdaptationType",
    "AdaptationTrigger",
    "AdaptationPhase",
    "AdaptationTarget",
    "AdaptationMetrics",
    "AdaptationConfig",
    "AdaptationResult",
    "AdaptationMetricsComputer",
    "SplatCandidateEvaluator",
    "AdaptationMetricsAggregator",
]
