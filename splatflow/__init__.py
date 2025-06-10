"""
SplatFlow Package Initialization
Comprehensive SplatFlow implementation with modular architecture.

Usage:
    from splatflow import SplatFlowTrainingOrchestrator, create_production_splatflow_model
    
    # Quick training
    config = create_default_config()
    config.update({'epochs': 10, 'batch_size': 2})
    trainer = SplatFlowTrainingOrchestrator(config)
    results = trainer.train()
    
    # Or create model directly
    model = create_production_splatflow_model(vocab_size=50257)
"""

# Core systems
from .splatflow_core_systems import (
    setup_environment,
    cleanup_memory,
    get_gpu_memory_info,
    DeviceManager,
    ComprehensiveRealDatasetLoader,
    EnhancedRealDataset,
    safe_tensor_to_scalar
)

# Trajectory systems
from .splatflow_trajectory_systems import (
    TrajectoryGuidanceSystem,
    TrajectoryCache,
    TrajectoryAwarePositionalEmbedding,
    EnhancedInterLayerTrajectoryFlow
)

# Attention components
from .splatflow_attention_components import (
    FixedProductionTrajectoryFlowSplat,
    FixedProductionSplatFlowAttention,
    get_quick_model_stats
)

# Model architecture
from .splatflow_model_architecture import (
    ProgressiveLayerTrainer,
    FixedProductionSplatFlowTransformerLayer,
    FixedUltimateProductionSplatFlowGPT,
    create_production_splatflow_model,
    setup_progressive_training,
    initialize_model_for_training
)

# Training orchestrator
from .splatflow_training_orchestrator import (
    SplatFlowTrainingOrchestrator,
    create_default_config,
    quick_start_example
)

# Analyzer (optional)
try:
    from .splatflow_analyzer import SplatFlowAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False
    SplatFlowAnalyzer = None

# Package metadata
__version__ = "1.0.0"
__author__ = "SplatFlow Development Team"
__description__ = "Production-ready SplatFlow attention mechanism with O(n*k) complexity"

# Key exports for easy importing
__all__ = [
    # High-level interfaces
    'SplatFlowTrainingOrchestrator',
    'create_production_splatflow_model', 
    'create_default_config',
    'quick_start_example',
    
    # Core model components
    'FixedUltimateProductionSplatFlowGPT',
    'FixedProductionSplatFlowTransformerLayer',
    'FixedProductionSplatFlowAttention',
    
    # Training utilities
    'ProgressiveLayerTrainer',
    'initialize_model_for_training',
    'setup_progressive_training',
    
    # Trajectory systems
    'EnhancedInterLayerTrajectoryFlow',
    'TrajectoryGuidanceSystem',
    'TrajectoryCache',
    
    # Data handling
    'EnhancedRealDataset',
    'ComprehensiveRealDatasetLoader',
    
    # Utilities
    'DeviceManager',
    'setup_environment',
    'cleanup_memory',
    'get_gpu_memory_info',
    'get_quick_model_stats',
    
    # Optional components
    'SplatFlowAnalyzer',  # Will be None if not available
]

def create_inference_model(checkpoint_path: str):
    """
    Create a SplatFlow model for inference from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        
    Returns:
        FixedUltimateProductionSplatFlowGPT: Loaded model ready for inference
    """
    model, checkpoint = FixedUltimateProductionSplatFlowGPT.load_model_checkpoint(checkpoint_path)
    model.eval()
    
    print(f"📂 Model loaded from {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"   Health: {checkpoint.get('model_stats', {}).get('overall_health', 'Unknown')}")
    
    return model

# Package validation function
def validate_installation():
    """
    Validate that the SplatFlow package is properly installed and functional.
    
    Returns:
        bool: True if validation passes
    """
    try:
        import torch
        print("✅ PyTorch available")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available - will use CPU")
        
        # Test basic model creation
        test_model = create_production_splatflow_model(
            vocab_size=1000,
            model_dim=64,
            num_layers=1,
            num_splats=4
        )
        print(f"✅ Model creation test passed")
        
        # Test forward pass
        test_input = torch.randint(0, 1000, (1, 10))
        test_output = test_model(test_input)
        print(f"✅ Forward pass test passed: {test_output.shape}")
        
        # Test trajectory systems
        trajectory_flow = EnhancedInterLayerTrajectoryFlow(1, 64)
        print(f"✅ Trajectory systems test passed")
        
        print("🎉 SplatFlow installation validation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

# Performance benchmarking function
def benchmark_performance(batch_size: int = 2, seq_length: int = 512, model_dim: int = 256):
    """
    Run a performance benchmark to test SplatFlow efficiency.
    
    Args:
        batch_size (int): Batch size for testing
        seq_length (int): Sequence length for testing
        model_dim (int): Model dimension for testing
        
    Returns:
        Dict: Performance metrics
    """
    import time
    
    print(f"🏁 Running SplatFlow performance benchmark...")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_length}")
    print(f"   Model dimension: {model_dim}")
    
    # Setup
    setup_environment()
    device = DeviceManager.get_primary_device()
    
    # Create model
    model = create_production_splatflow_model(
        vocab_size=10000,
        model_dim=model_dim,
        num_layers=3,
        num_splats=16
    ).to(device)
    
    # Create test input
    test_input = torch.randint(0, 10000, (batch_size, seq_length), device=device)
    
    # Warmup
    for _ in range(5):
        _ = model(test_input)
    
    # Benchmark forward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    num_runs = 10
    for _ in range(num_runs):
        output = model(test_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_run = total_time / num_runs
    tokens_per_second = (batch_size * seq_length * num_runs) / total_time
    
    # Memory usage
    memory_info = get_gpu_memory_info() if torch.cuda.is_available() else None
    
    results = {
        'total_time': total_time,
        'avg_time_per_run': avg_time_per_run,
        'tokens_per_second': tokens_per_second,
        'batch_size': batch_size,
        'seq_length': seq_length,
        'model_dim': model_dim,
        'memory_info': memory_info,
        'device': str(device)
    }
    
    print(f"📊 Benchmark Results:")
    print(f"   Avg time per run: {avg_time_per_run:.4f}s")
    print(f"   Tokens/second: {tokens_per_second:.0f}")
    if memory_info:
        print(f"   GPU memory used: {memory_info['percent_used']:.1f}%")
    
    return results

# Help function
def help():
    """Print comprehensive help information for the SplatFlow package."""
    
    help_text = """
🌟 SplatFlow - Production-Ready O(n*k) Attention Mechanism

QUICK START:
    from splatflow import quick_start_example
    trainer = quick_start_example()
    trainer.train()

MAIN COMPONENTS:

1. Training Orchestrator:
    from splatflow import SplatFlowTrainingOrchestrator, create_default_config
    
    config = create_default_config()
    config.update({'epochs': 10, 'batch_size': 2})
    trainer = SplatFlowTrainingOrchestrator(config)
    results = trainer.train()

2. Direct Model Creation:
    from splatflow import create_production_splatflow_model
    
    model = create_production_splatflow_model(
        vocab_size=50257,
        model_dim=512,
        num_layers=6,
        num_splats=20
    )

3. Inference from Checkpoint:
    from splatflow import create_inference_model
    
    model = create_inference_model('path/to/checkpoint.pt')
    text = model.generate_text(tokenizer, "Hello world", max_length=50)

UTILITIES:
    from splatflow import validate_installation, benchmark_performance
    
    validate_installation()  # Check if everything works
    benchmark_performance()  # Test performance

CONFIGURATION:
    Key config parameters:
    - model_dim: Model embedding dimension
    - num_layers: Number of transformer layers  
    - num_splats: Number of splats per attention layer
    - epochs: Training epochs
    - batch_size: Training batch size
    - learning_rate: Learning rate for optimization

For detailed documentation, see the individual module docstrings.
    """
    
    print(help_text)

# Module-level convenience function
def get_version_info():
    """Get detailed version and capability information."""
    
    import torch
    
    info = {
        'splatflow_version': __version__,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'analyzer_available': ANALYZER_AVAILABLE
    }
    
    return info

# Print package info on import
def _print_package_info():
    """Print package information when imported."""
    print(f"🌟 SplatFlow v{__version__} - Production-Ready O(n*k) Attention")
    print(f"   {__description__}")
    print(f"   Enhanced monitoring: {'✅ Available' if ANALYZER_AVAILABLE else '⚠️  Limited'}")
    print(f"   Run splatflow.help() for usage information")
    print(f"   Run splatflow.quick_start_example() for quick demo")

# Uncomment the next line to show info on import
# _print_package_info()
