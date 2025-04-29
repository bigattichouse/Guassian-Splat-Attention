# HSA Package
from .data_structures import Splat, Hierarchy, SplatRegistry
from .initialization import initialize_splats, reinitialize_splat
from .attention import AttentionComputer, create_attention_computer, SplatAttentionMetrics
from .adaptation import (
    AdaptationType, check_adaptation_triggers, perform_adaptations, 
    AdaptationMonitor, default_monitor, AdaptationResult
)
from .model_integration import HSAAttention, HSATransformerLayer, replace_attention_with_hsa
from .training import train_hsa, evaluate_hsa, TrainingConfig
from .visualization import HSAVisualizer

__version__ = "0.1.0"
