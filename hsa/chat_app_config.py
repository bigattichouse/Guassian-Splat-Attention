"""
HSA Chat configuration module.

This module defines the configuration options for the HSA Chat application.
"""

import os
from typing import Dict, List, Optional, Any

# Default models to use
DEFAULT_MODEL = "gpt2"  # Use a small model for testing
SUPPORTED_MODELS = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "mistralai/Mistral-7B-v0.1"  # If you have enough RAM
]

# HSA configuration
HSA_CONFIG = {
    # Core settings
    "use_sparse": True,              # Use sparse attention for efficiency
    "adaptation_enabled": True,      # Enable adaptive behavior
    
    # Hierarchy settings
    "hierarchy_levels": [
        "token",
        "word", 
        "phrase", 
        "sentence", 
        "document"
    ],
    
    # Context extension
    "enable_context_extension": True,  # Enable context window extension
    "max_context_length": 4096,        # Target for maximum context length
    
    # Performance settings
    "device": "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu",
    "half_precision": False,           # Use FP16 for computation
    
    # Output settings
    "show_stats": True,                # Show HSA stats after generation
    "logging_level": "INFO",           # Logging level
    
    # Generation settings
    "temperature": 0.7,                # Generation temperature
    "top_p": 0.9,                      # Nucleus sampling parameter
    "max_length": 2048,                # Maximum generation length
    
    # Chat settings
    "system_prompt": "You are a helpful assistant enhanced with Hierarchical Splat Attention.",
    "chat_template": "{system}\n\n{history}\nUser: {input}\nAssistant: ",
}

# Debug settings
DEBUG_CONFIG = {
    "dump_attention_matrices": False,  # Save attention matrices to disk
    "attention_dir": "attention_dumps",  # Directory for attention dumps
    "trace_splat_evolution": True,    # Track splat changes over time
    "evolution_dir": "splat_evolution",  # Directory for evolution data
    "verbose_output": True,           # Show verbose output
}

def get_config() -> Dict[str, Any]:
    """Get the combined configuration."""
    config = {
        **HSA_CONFIG,
        "debug": DEBUG_CONFIG
    }
    
    # Environment overrides
    if os.environ.get("HSA_MODEL"):
        config["model"] = os.environ.get("HSA_MODEL")
    
    if os.environ.get("HSA_MAX_CONTEXT"):
        config["max_context_length"] = int(os.environ.get("HSA_MAX_CONTEXT"))
    
    if os.environ.get("HSA_TEMPERATURE"):
        config["temperature"] = float(os.environ.get("HSA_TEMPERATURE"))
    
    # Check for config file
    config_file = os.environ.get("HSA_CONFIG_FILE", "hsa_config.json")
    if os.path.exists(config_file):
        import json
        with open(config_file, "r") as f:
            file_config = json.load(f)
            # Merge configs
            config.update(file_config)
    
    return config
