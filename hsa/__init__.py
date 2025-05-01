"""
Hierarchical Splat Attention (HSA) - A more efficient attention mechanism.

HSA reimagines transformer attention by replacing the standard O(n²) attention
mechanism with a hierarchical, Gaussian-based approach using intermediaries called "splats".
"""

# Import core data structures
from .splat import Splat, RingBuffer
from .hierarchy import Hierarchy
from .registry import SplatRegistry

# Import attention components
from .attention_interface import AttentionComputer, AttentionConfig, AttentionResult
from .dense_attention import DenseAttentionComputer

# Import adaptation components
from .adaptation_types import (
    AdaptationType, AdaptationTrigger, AdaptationPhase,
    AdaptationTarget, AdaptationMetrics, AdaptationConfig, AdaptationResult
)
from .adaptation_metrics_base import (
    AdaptationMetricsComputer, SplatCandidateEvaluator,
    AdaptationMetricsAggregator
)

# Import utilities
from .attention_cache import AttentionCache, cache_attention
from .attention_utils import apply_causal_mask, normalize_rows, apply_topk_mask

# Package version
__version__ = "0.1.0"

