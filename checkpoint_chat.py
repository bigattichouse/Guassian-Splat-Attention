#!/usr/bin/env python
"""
HSA Checkpoint Chat

This script allows you to chat with HSA models saved during training.
It loads checkpoints from the training output directory and enables
interactive conversations with specific context length settings.

Usage:
    python checkpoint_chat.py [options]

Options:
    --checkpoint     Path to the specific checkpoint file to load
    --output_dir     Directory where model checkpoints are stored (default: outputs)
    --model_name     Name of the model to load (default: hsa-tinystories)
    --context_length Maximum context length to use (default: 512)
    --max_length     Maximum length for generated responses (default: 100)
    --temperature    Temperature for generation (default: 0.8)
    --debug          Enable debug output
"""

import os
import sys
import argparse
import json
import logging
import time
import threading
import queue
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hsa_checkpoint_chat.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


from hsa.core import HSA, create_hsa
from hsa.data_structures import Splat, Hierarchy, SplatRegistry
from hsa.adaptation import AdaptationType
from hsa.attention import SplatAttentionMetrics
from hsa.visualization import HSAVisualizer
from hsa.model_integration import create_hsa_transformer


# Try to import transformers
try:
    from transformers import AutoTokenizer
except ImportError:
    print("Error: This script requires the transformers library.")
    print("Please install it with: pip install transformers")
    sys.exit(1)

def print_with_delay(text, delay=0.005):
    """Print text with a typing effect for better UX."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chat with an HSA checkpoint model")
    
    # Main options
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the specific checkpoint file to load")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory where model checkpoints are stored")
    parser.add_argument("--model_name", type=str, default="hsa-tinystories",
                        help="Name of the model to load from the output directory")
    
    # Generation parameters
    parser.add_argument("--context_length", type=int, default=512,
                        help="Maximum context length to use (default: 512)")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length for generated responses (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for text generation (default: 0.8)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter (default: 0.9)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling parameter (default: 40)")
    
    # Other settings
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--device", type=str, 
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda if available, otherwise cpu)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable HSA visualizations during chat")
    
    return parser.parse_args()

def find_best_checkpoint(checkpoints_dir, model_name):
    """Find the best checkpoint in the output directory."""
    model_dir = os.path.join(checkpoints_dir, model_name)
    checkpoints_dir = os.path.join(model_dir, "checkpoints")
    
    if not os.path.exists(checkpoints_dir):
        print(f"Error: Checkpoints directory not found: {checkpoints_dir}")
        return None
    
    # Try to load best_model.pt first
    best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
    if os.path.exists(best_model_path) and os.path.exists(f"{best_model_path}.success"):
        return best_model_path
    
    # Otherwise, look for the latest step or epoch checkpoint
    checkpoints = []
    
    # Look for step checkpoints
    for filename in os.listdir(checkpoints_dir):
        if filename.startswith("step_") and filename.endswith(".pt"):
            check_path = os.path.join(checkpoints_dir, filename)
            success_marker = f"{check_path}.success"
            
            # Only consider checkpoints that have a success marker
            if os.path.exists(success_marker):
                try:
                    step = int(filename.split("_")[1].split(".")[0])
                    checkpoints.append(("step", step, check_path))
                except Exception as e:
                    logger.warning(f"Corrupt checkpoint: {filename}: {e}")
    
    # Look for epoch checkpoints
    for filename in os.listdir(checkpoints_dir):
        if filename.startswith("epoch_") and filename.endswith(".pt"):
            check_path = os.path.join(checkpoints_dir, filename)
            success_marker = f"{check_path}.success"
            
            # Only consider checkpoints that have a success marker
            if os.path.exists(success_marker):
                try:
                    epoch = int(filename.split("_")[1].split(".")[0])
                    checkpoints.append(("epoch", epoch, check_path))
                except Exception as e:
                    logger.warning(f"Corrupt checkpoint: {filename}: {e}")
    
    # Look for final model checkpoint
    final_model_path = os.path.join(checkpoints_dir, "final_model.pt")
    if os.path.exists(final_model_path) and os.path.exists(f"{final_model_path}.success"):
        checkpoints.append(("final", 0, final_model_path))
    
    # Sort by type and step/epoch (prioritize final, then epoch, then step checkpoints)
    type_priority = {"final": 0, "epoch": 1, "step": 2}
    checkpoints.sort(key=lambda x: (type_priority.get(x[0], 3), -x[1]))
    
    if checkpoints:
        return checkpoints[0][2]  # Return the path to the best checkpoint
    
    return None

def load_checkpoint(checkpoint_path, device):
    """Load a model checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None

def restore_hsa_from_checkpoint(hsa_system, hsa_data):
    """Restore HSA state from checkpoint data."""
    try:
        logger.debug("Restoring HSA from checkpoint...")
        logger.debug(f"HSA data keys: {hsa_data.keys()}")
        
        # Verify hierarchy matches
        if hsa_data["hierarchy"]["levels"] != hsa_system.hierarchy.levels:
            print("Warning: Saved HSA hierarchy doesn't match current configuration")
            logger.warning("Saved HSA hierarchy doesn't match current configuration")
            return False
        
        # Create a new splat registry with the saved hierarchy
        from hsa.data_structures import Hierarchy, SplatRegistry, Splat
        
        # Restore hierarchy
        hierarchy = Hierarchy(
            levels=hsa_data["hierarchy"]["levels"],
            init_splats_per_level=hsa_data["hierarchy"]["init_splats_per_level"],
            level_weights=hsa_data["hierarchy"]["level_weights"]
        )
        
        # Create registry
        registry = SplatRegistry(hierarchy)
        
        # Restore splats
        for splat_info in hsa_data["splats"]:
            # Create position array
            position = np.array(splat_info["position"])
            
            # Create simplified covariance (either from diagonal or as identity)
            if "cov_diag" in splat_info:
                covariance = np.diag(splat_info["cov_diag"])
            else:
                # Fallback to identity covariance
                covariance = np.eye(position.shape[0])
            
            # Create the splat
            splat = Splat(
                position=position,
                covariance=covariance,
                amplitude=splat_info["amplitude"],
                level=splat_info["level"],
                splat_id=splat_info["id"]
            )
            
            # Register the splat
            registry.register(splat)
        
        # Update HSA system
        hsa_system.splat_registry = registry
        hsa_system.is_initialized = True
        
        print(f"Restored HSA state with {len(registry.splats)} splats")
        return True
    except Exception as e:
        logger.error(f"Error in restore_hsa_from_checkpoint: {e}")
        traceback.print_exc()
        return False

def initialize_model_and_hsa(checkpoint_path, args):
    """Initialize the model and HSA components from a checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = load_checkpoint(checkpoint_path, args.device)
    
    if checkpoint is None:
        print("Failed to load checkpoint. Exiting.")
        sys.exit(1)
    
    # Get the model's configuration
    model_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    config_path = os.path.join(model_dir, "config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                model_config = json.load(f)
            print(f"Loaded model configuration from {config_path}")
        except Exception as e:
            print(f"Error loading model configuration: {e}")
            model_config = None
    else:
        print(f"Model configuration not found at {config_path}")
        model_config = None
    
    # Get HSA configuration - either from checkpoint or create default
    if "hsa_data" in checkpoint and checkpoint["hsa_data"]:
        hsa_config = {
            "hierarchy": checkpoint["hsa_data"]["hierarchy"],
            "adaptation": {
                "mitosis_threshold": 0.1,
                "death_threshold": 0.01,
                "adaptation_frequency": 5,
                "enable_adaptation": True
            },
            "attention": {
                "sparse_topk": 64
            }
        }
    else:
        # Fallback to default configuration
        hsa_config = {
            "hierarchy": {
                "levels": ["Token", "Phrase", "Section"],
                "init_splats_per_level": [64, 32, 16],
                "level_weights": [0.5, 0.3, 0.2]
            },
            "adaptation": {
                "mitosis_threshold": 0.1,
                "death_threshold": 0.01,
                "adaptation_frequency": 5,
                "enable_adaptation": True
            },
            "attention": {
                "sparse_topk": 64
            }
        }
    
    # Create HSA instance with the configuration
    print("Creating HSA instance...")
    hsa = create_hsa(hsa_config)
    
    # Try to load tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    if os.path.exists(tokenizer_path):
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"Loaded tokenizer from {tokenizer_path}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            print("Falling back to default gpt2 tokenizer")
    else:
        print(f"Tokenizer not found at {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print("Using default gpt2 tokenizer")
    
    # Make sure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model architecture that matches the saved checkpoint
    if model_config:
        # Get model parameters from config
        hidden_size = model_config.get("hidden_size", 128)
        num_layers = model_config.get("num_layers", 4)
        num_heads = model_config.get("num_heads", 4)
        vocab_size = model_config.get("vocab_size", len(tokenizer))
        seq_length = model_config.get("seq_length", args.context_length)
        
        # Override context length if specified
        context_length = args.context_length
    else:
        # Use default parameters
        hidden_size = 128
        num_layers = 4
        num_heads = 4
        vocab_size = len(tokenizer)
        context_length = args.context_length
    
    print(f"Creating model with parameters: hidden_size={hidden_size}, num_layers={num_layers}, "
          f"num_heads={num_heads}, vocab_size={vocab_size}, context_length={context_length}")
    
    # Create the model
    base_model = create_hsa_transformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        hsa_config=hsa_config,
        max_position_embeddings=context_length,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Create language model with head
    class HSALanguageModel(nn.Module):
        def __init__(self, transformer, vocab_size):
            super().__init__()
            self.transformer = transformer
            self.lm_head = nn.Linear(transformer.word_embeddings.embedding_dim, vocab_size, bias=False)
            
        def forward(self, input_ids, attention_mask=None):
            # Get transformer outputs
            hidden_states = self.transformer(input_ids, attention_mask)
            
            # Apply language modeling head
            logits = self.lm_head(hidden_states)
            
            return logits
    
    # Create full language model
    model = HSALanguageModel(base_model, vocab_size)
    model.to(args.device)
    
    # Load model state from checkpoint
    print("Loading model state from checkpoint...")
    try:
        model.load_state_dict(checkpoint["model"])
        print("Model state loaded successfully")
    except Exception as e:
        print(f"Error loading model state: {e}")
        print("The model architecture may not match the checkpoint structure.")
        sys.exit(1)
    
    # Restore HSA state if available
    if "hsa_data" in checkpoint and checkpoint["hsa_data"]:
        print("Restoring HSA state from checkpoint...")
        restore_success = restore_hsa_from_checkpoint(hsa, checkpoint["hsa_data"])
        if not restore_success:
            print("Warning: Failed to restore HSA state from checkpoint")
            print("Initializing HSA with random embeddings...")
            
            # Initialize with word embeddings
            with torch.no_grad():
                sample_embeddings = model.transformer.word_embeddings.weight.detach().cpu().numpy()
                sample_size = min(1000, sample_embeddings.shape[0])
                sample_tokens = sample_embeddings[:sample_size]
                hsa.initialize(sample_tokens)
                print(f"Initialized HSA with {sample_size} token embeddings")
    else:
        print("No HSA state found in checkpoint")
        print("Initializing HSA with word embeddings...")
        
        # Initialize with word embeddings
        with torch.no_grad():
            sample_embeddings = model.transformer.word_embeddings.weight.detach().cpu().numpy()
            sample_size = min(1000, sample_embeddings.shape[0])
            sample_tokens = sample_embeddings[:sample_size]
            hsa.initialize(sample_tokens)
            print(f"Initialized HSA with {sample_size} token embeddings")
    
    # Initialize visualization if enabled
    if args.visualize:
        visualizer = HSAVisualizer(output_dir="hsa_visualizations")
    else:
        visualizer = None
    
    # Create adaptation metrics tracker
    metrics_tracker = SplatAttentionMetrics()
    
    return model, tokenizer, hsa, metrics_tracker, visualizer

def generate_response(model, tokenizer, prompt, args):
    """Generate a response using the model."""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
    
    # Create attention mask (1 for all input tokens)
    attention_mask = torch.ones_like(inputs.input_ids)
    
    # Generate response
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=attention_mask,
                max_length=inputs.input_ids.shape[1] + args.max_length,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                # Add repetition penalty to avoid loops
                repetition_penalty=1.2,
                # Ensure we're using the model's attention mechanism
                use_cache=True
            )
        
        # Extract only the generated response (without the input)
        input_length = inputs.input_ids.shape[1]
        response_tokens = outputs[0][input_length:]
        
        # Decode the response
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        return response, response_tokens
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return f"Error generating response: {e}", None

def run_adaptation(hsa, embeddings, visualizer=None, show_debug=False):
    """Run adaptation with improved error handling and timeout protection."""
    print("\n\033[1mStarting HSA adaptation...\033[0m")
    try:
        # Flatten embeddings if needed
        if len(embeddings.shape) > 2:
            embeddings = embeddings.reshape(-1, embeddings.shape[-1])
        
        # Subsample to reduce computation time
        if len(embeddings) > 256:
            subsample_indices = np.random.choice(
                len(embeddings), 
                size=256, 
                replace=False
            )
            embeddings = embeddings[subsample_indices]
        
        print(f"Running adaptation with {len(embeddings)} token embeddings")
        
        # Use a thread with timeout
        def adaptation_thread(queue_obj, hsa_obj, embs):
            try:
                result = hsa_obj.adapt(embs, cpu_efficient=True)
                queue_obj.put(result)
            except Exception as e:
                queue_obj.put(e)
        
        result_queue = queue.Queue()
        thread = threading.Thread(
            target=adaptation_thread,
            args=(result_queue, hsa, embeddings)
        )
        thread.daemon = True
        thread.start()
        
        # Wait for thread with timeout (45 seconds)
        thread.join(45)
        
        if thread.is_alive():
            print("Adaptation timed out after 45 seconds")
            return {"timeout": True}
        
        if not result_queue.empty():
            result = result_queue.get()
            if isinstance(result, Exception):
                print(f"Adaptation error: {result}")
                return {"error": str(result)}
            
            # Success
            print(f"Adaptation successful: {result}")
            
            # Create visualization if requested
            if visualizer and show_debug:
                try:
                    viz_path = visualizer.visualize_splat_distribution(
                        splat_registry=hsa.splat_registry,
                        tokens=embeddings,
                        title="HSA Splat Distribution after Adaptation",
                        show=False,
                        save=True
                    )
                    print(f"Created visualization at {viz_path}")
                except Exception as e:
                    print(f"Visualization error: {e}")
            
            return result
        else:
            print("Adaptation thread finished but queue is empty")
            return {"error": "Adaptation thread finished but queue is empty"}
    except Exception as e:
        print(f"Error in adaptation: {e}")
        return {"error": str(e)}

def chat_loop(model, tokenizer, hsa, metrics_tracker, visualizer, args):
    """Run the interactive chat loop."""
    print("\n" + "="*50)
    print_with_delay("Welcome to the HSA Checkpoint Chat!")
    print("="*50)
    print_with_delay("Chat with a model trained with Hierarchical Splat Attention.")
    print_with_delay(f"Context length: {args.context_length}, Max response length: {args.max_length}")
    print_with_delay("Type 'exit' to quit, 'help' for more commands.")
    print("="*50 + "\n")
    
    # Display initial stats
    print("Initial HSA Stats:")
    stats = hsa.get_stats()
    for level, count in stats['splat_counts'].items():
        if level != 'total':
            print(f"  - {level} level: {count} splats")
    print(f"  - Total: {stats['splat_counts'].get('total', 0)} splats")
    print()
    
    # Enable debug mode
    show_debug = args.debug
    
    # Initialize chat history
    chat_history = []
    
    # Command handlers
    def handle_stats():
        """Handle the 'stats' command."""
        print("\n\033[1mHSA Statistics:\033[0m")
        stats = hsa.get_stats()
        for level, count in stats['splat_counts'].items():
            if level != 'total':
                print(f"  - {level} level: {count} splats")
        print(f"  - Total: {stats['splat_counts'].get('total', 0)} splats")
        
        print("\nLevel contributions to attention:")
        for level, contrib in stats['level_contributions'].items():
            print(f"  - {level}: {contrib:.2f}")
            
        print(f"\nAttention sparsity: {stats['attention_sparsity']:.2f}")
        print(f"Adaptations - Mitosis: {stats['adaptations']['mitosis']}, Death: {stats['adaptations']['death']}")
    
    def handle_visualize():
        """Handle the 'visualize' command."""
        if visualizer is None:
            print("Visualization is not enabled. Restart with --visualize flag.")
            return
        
        print("\n\033[1mCreating visualizations...\033[0m")
        try:
            # Get embeddings for visualization
            with torch.no_grad():
                # Use the model's word embeddings for visualization
                embeddings = model.transformer.word_embeddings.weight.detach().cpu().numpy()
                # Sample a subset for visualization
                sample_size = min(100, embeddings.shape[0])
                indices = np.random.choice(embeddings.shape[0], size=sample_size, replace=False)
                sample_embeddings = embeddings[indices]
                
                # Create visualizations
                viz_paths = []
                
                # Visualize splat distribution
                dist_path = visualizer.visualize_splat_distribution(
                    splat_registry=hsa.splat_registry,
                    tokens=sample_embeddings,
                    title="HSA Splat Distribution",
                    show=False,
                    save=True
                )
                viz_paths.append(dist_path)
                
                # Visualize hierarchy
                hier_path = visualizer.visualize_hierarchy(
                    splat_registry=hsa.splat_registry,
                    title="HSA Hierarchy",
                    show=False,
                    save=True
                )
                viz_paths.append(hier_path)
                
                # Compute and visualize attention
                try:
                    attention_matrix = hsa.compute_attention(sample_embeddings)
                    
                    # Visualize attention matrix
                    attn_path = visualizer.visualize_attention_matrix(
                        attention_matrix=attention_matrix,
                        title="HSA Attention Matrix",
                        show=False,
                        save=True
                    )
                    viz_paths.append(attn_path)
                    
                    # Visualize attention sparsity
                    sparsity_path = visualizer.visualize_attention_sparsity(
                        attention_matrix=attention_matrix,
                        title="HSA Attention Sparsity",
                        show=False,
                        save=True
                    )
                    viz_paths.append(sparsity_path)
                    
                except Exception as e:
                    print(f"Error computing attention for visualization: {e}")
                
                # Create dashboard
                dashboard_path = visualizer.create_dashboard(
                    splat_registry=hsa.splat_registry,
                    attention_matrix=attention_matrix if 'attention_matrix' in locals() else None,
                    tokens=sample_embeddings,
                    title="HSA Dashboard",
                    show=False,
                    save=True
                )
                viz_paths.append(dashboard_path)
                
                print(f"Created {len(viz_paths)} visualizations in the 'hsa_visualizations' directory:")
                for path in viz_paths:
                    if path:
                        print(f"  - {path}")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
            logger.error(f"Visualization error: {e}")
    
    def handle_adapt():
        """Handle the 'adapt' command."""
        print("\n\033[1mForcing HSA adaptation...\033[0m")
        
        try:
            # Get embeddings from the model's vocabulary
            with torch.no_grad():
                vocab_size = min(1000, model.transformer.word_embeddings.weight.shape[0])
                random_ids = torch.randint(0, vocab_size, (100,))
                embeddings = model.transformer.word_embeddings(random_ids).detach().cpu().numpy()
            
            # Run adaptation with timeout protection
            result = run_adaptation(
                hsa, 
                embeddings, 
                visualizer=visualizer if args.visualize else None,
                show_debug=show_debug
            )
            
            # Report results
            if isinstance(result, dict) and "error" in result:
                print(f"Adaptation failed: {result['error']}")
            elif isinstance(result, dict) and "timeout" in result:
                print("Adaptation timed out")
            else:
                print(f"Adaptation completed!")
                print(f"  - Mitosis events: {result.get('mitosis', 0)}")
                print(f"  - Death events: {result.get('death', 0)}")
        except Exception as e:
            print(f"Error during adaptation: {e}")
    
    def handle_save():
        """Handle the 'save' command."""
        print("\n\033[1mSaving current HSA state...\033[0m")
        
        try:
            # Create a serializable version of the splat registry
            splat_data = []
            for splat_id, splat in hsa.splat_registry.splats.items():
                splat_info = {
                    "id": splat.id,
                    "level": splat.level,
                    "position": splat.position.tolist(),
                    "amplitude": float(splat.amplitude),
                    # Simplified covariance representation
                    "cov_diag": np.diag(splat.covariance).tolist()
                }
                splat_data.append(splat_info)
            
            # Prepare HSA data for saving
            hsa_data = {
                "splats": splat_data,
                "hierarchy": {
                    "levels": hsa.hierarchy.levels,
                    "level_weights": hsa.hierarchy.level_weights,
                    "init_splats_per_level": hsa.hierarchy.init_splats_per_level
                }
            }
            
            # Save to a JSON file
            save_path = "hsa_state.json"
            with open(save_path, "w") as f:
                json.dump(hsa_data, f, indent=2)
            
            print(f"HSA state saved to {save_path}")
        except Exception as e:
            print(f"Error saving HSA state: {e}")
    
    def handle_help():
        """Handle the 'help' command."""
        print("\n\033[1mCommands:\033[0m")
        print("  - Type your message to chat with the model")
        print("  - 'stats': Show current HSA statistics")
        print("  - 'visualize': Create and save HSA visualizations")
        print("  - 'adapt': Force adaptation of HSA splats")
        print("  - 'save': Save current HSA state to a file")
        print("  - 'debug': Toggle debug information")
        print("  - 'clear': Clear the chat history")
        print("  - 'length <number>': Change maximum context length")
        print("  - 'temp <number>': Change temperature (0.1-2.0)")
        print("  - 'help': Show this help message")
        print("  - 'exit' or 'quit': End the chat")
    
    # Start chat loop
    while True:
        # Get user input
        user_input = input("\n\033[1mYou:\033[0m ")
        user_input = user_input.strip()
        
        # Handle special commands
        if user_input.lower() in ['exit', 'quit']:
            print_with_delay("\nThank you for using HSA Checkpoint Chat. Goodbye!")
            break
            
        elif user_input.lower() == 'stats':
            handle_stats()
            continue
            
        elif user_input.lower() == 'visualize':
            handle_visualize()
            continue
            
        elif user_input.lower() == 'adapt':
            handle_adapt()
            continue
            
        elif user_input.lower() == 'save':
            handle_save()
            continue
            
        elif user_input.lower() == 'debug':
            show_debug = not show_debug
            print(f"\n\033[1mDebug mode {'enabled' if show_debug else 'disabled'}\033[0m")
            continue
            
        elif user_input.lower() == 'clear':
            chat_history = []
            print("\n\033[1mChat history cleared\033[0m")
            continue
            
        elif user_input.lower().startswith('length '):
            try:
                new_length = int(user_input.split()[1])
                if new_length < 10 or new_length > 2048:
                    print("Context length must be between 10 and 2048 tokens")
                else:
                    args.context_length = new_length
                    print(f"\n\033[1mContext length set to {new_length} tokens\033[0m")
            except (ValueError, IndexError):
                print("Invalid format. Use 'length <number>'")
            continue
            
        elif user_input.lower().startswith('temp '):
            try:
                new_temp = float(user_input.split()[1])
                if new_temp < 0.1 or new_temp > 2.0:
                    print("Temperature must be between 0.1 and 2.0")
                else:
                    args.temperature = new_temp
                    print(f"\n\033[1mTemperature set to {new_temp}\033[0m")
            except (ValueError, IndexError):
                print("Invalid format. Use 'temp <number>'")
            continue
            
        elif user_input.lower() == 'help':
            handle_help()
            continue
            
        if not user_input:
            continue
        
        # Add user input to history
        chat_history.append(("user", user_input))
        
        # Build prompt from history
        context_builder = []
        
        # Calculate available token budget
        max_context_tokens = args.context_length - args.max_length - 20  # Reserve space for response and overhead
        
        # Add chat history from the most recent to older, until we hit the limit
        token_count = 0
        history_to_include = []
        
        # Always include the current user input
        current_input_tokens = len(tokenizer.encode(user_input))
        token_count += current_input_tokens
        
        # Add previous exchanges, newest first, then reverse
        for role, text in reversed(chat_history[:-1]):  # Skip the current input
            # Estimate tokens for this message
            message_tokens = len(tokenizer.encode(text)) + 10  # +10 for role prefix
            
            # If adding this would exceed our budget, stop
            if token_count + message_tokens > max_context_tokens:
                break
                
            # Otherwise, add it and update count
            history_to_include.append((role, text))
            token_count += message_tokens
        
        # Reverse to get chronological order
        history_to_include.reverse()
        
        # Format the prompt
        for role, text in history_to_include:
            prefix = "User: " if role == "user" else "Assistant: "
            context_builder.append(f"{prefix}{text}")
        
        # Add the current user message
        context_builder.append(f"User: {user_input}")
        context_builder.append("Assistant:")
        
        # Create the full prompt
        full_prompt = "\n\n".join(context_builder)
        
        # Show debug information if enabled
        if show_debug:
            # Estimate tokens in the prompt
            prompt_tokens = len(tokenizer.encode(full_prompt))
            print(f"\n\033[90m(Prompt length: {prompt_tokens} tokens, Context limit: {args.context_length})\033[0m")
        
        # Generate response
        print("\n\033[1mAssistant:\033[0m ", end='', flush=True)
        
        start_time = time.time()
        response, response_tokens = generate_response(model, tokenizer, full_prompt, args)
        generation_time = time.time() - start_time
        
        # Print the response with typing effect
        print_with_delay(response, delay=0.01)
        
        # Update chat history with model's response
        chat_history.append(("assistant", response))
        
        # Show debug information if enabled
        if show_debug:
            token_count = len(response_tokens) if response_tokens is not None else "unknown"
            print(f"\n\033[90m(Generation time: {generation_time:.2f}s, Response tokens: {token_count})\033[0m")
        
        # Run adaptation periodically (every 3 exchanges)
        if len(chat_history) % 3 == 0:
            if show_debug:
                print(f"\n\033[90m(Running periodic adaptation...)\033[0m")
            
            try:
                # Get embeddings from the current exchange
                with torch.no_grad():
                    # Get embeddings from the model using the most recent exchange
                    recent_text = f"{user_input}\n\n{response}"
                    inputs = tokenizer(recent_text, return_tensors="pt").to(args.device)
                    embeddings = model.transformer.word_embeddings(inputs.input_ids).detach().cpu().numpy()[0]
                
                # Run adaptation with improved error handling
                adaptation_result = run_adaptation(
                    hsa, 
                    embeddings, 
                    visualizer=visualizer if args.visualize else None,
                    show_debug=False  # Don't show debug info for automatic adaptations
                )
                
                if show_debug:
                    if isinstance(adaptation_result, dict) and "error" in adaptation_result:
                        print(f"\033[90m(Adaptation error: {adaptation_result['error']})\033[0m")
                    elif isinstance(adaptation_result, dict) and "timeout" in adaptation_result:
                        print("\033[90m(Adaptation timed out)\033[0m")
                    else:
                        print(f"\033[90m(Adaptation complete - Mitosis: {adaptation_result.get('mitosis', 0)}, "
                              f"Death: {adaptation_result.get('death', 0)})\033[0m")
            except Exception as e:
                if show_debug:
                    print(f"\033[90m(Error during periodic adaptation: {e})\033[0m")

def main():
    """Main function to run the checkpoint chat."""
    args = parse_arguments()
    
    # Set the logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        print(f"No checkpoint specified, looking for best checkpoint in {args.output_dir}/{args.model_name}...")
        checkpoint_path = find_best_checkpoint(args.output_dir, args.model_name)
    
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"Error: Could not find a valid checkpoint")
        return
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    # Initialize model and HSA components
    model, tokenizer, hsa, metrics_tracker, visualizer = initialize_model_and_hsa(checkpoint_path, args)
    
    # Run the chat loop
    chat_loop(model, tokenizer, hsa, metrics_tracker, visualizer, args)

if __name__ == "__main__":
    main()
