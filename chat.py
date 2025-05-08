#!/usr/bin/env python3
"""
HSA Chat CLI - Command-line interface for chatting with models using Hierarchical Splat Attention.

This application loads a model from Hugging Face, replaces its attention mechanism with HSA,
and provides a simple chat interface with attention statistics.
"""

import os
import sys
import argparse
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

# Import HSA components
from hsa.splat import Splat
from hsa.hierarchy import Hierarchy
from hsa.registry import SplatRegistry
from hsa.dense_attention import DenseAttentionComputer
from hsa.sparse_attention import SparseAttentionComputer
from hsa.attention_interface import AttentionConfig
from hsa.adaptation_controller import AdaptationController
from hsa.adaptation_metrics_base import AdaptationMetricsComputer, SplatCandidateEvaluator
from hsa.attention_info_metrics import InfoTheoreticMetricsComputer, InfoTheoreticCandidateEvaluator

# Import the HSA model adapter
from hsa.model_adapter import create_adapter_for_model, ContextExtender
from hsa.chat_app_config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ChatCLI:
    """Command-line interface for chatting with HSA-enhanced models."""
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 2048,
        temperature: float = 0.7,
        use_sparse: bool = True,
        adaptation_enabled: bool = True,
        show_stats: bool = True,
        extend_context: bool = False,
        max_context_length: Optional[int] = None
    ):
        """Initialize chat interface.
        
        Args:
            model_name: Name of the model to load from Hugging Face
            max_length: Maximum sequence length for generation
            temperature: Temperature for generation
            use_sparse: Whether to use sparse attention
            adaptation_enabled: Whether to enable HSA adaptation
            show_stats: Whether to show HSA stats after each generation
            extend_context: Whether to extend context window
            max_context_length: Maximum context length if extending
        """
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.use_sparse = use_sparse
        self.adaptation_enabled = adaptation_enabled
        self.show_stats = show_stats
        self.extend_context = extend_context
        self.max_context_length = max_context_length
        
        # Load model and tokenizer
        logger.info(f"Loading model and tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Enable padding in the tokenizer if not already enabled
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create HSA adapter
        logger.info("Creating HSA adapter for model")
        hierarchy_levels = ["token", "phrase", "sentence", "document"]
        
        self.model_patcher, self.registry = create_adapter_for_model(
            model=self.model,
            use_sparse=use_sparse,
            adaptation_enabled=adaptation_enabled,
            hierarchy_levels=hierarchy_levels,
            max_context_length=max_context_length
        )
        
        # Patch the model with HSA
        logger.info("Patching model with HSA attention")
        self.model_patcher.patch_model()
        
        # Set up context extension if enabled
        if extend_context and max_context_length:
            logger.info(f"Setting up context extension to {max_context_length}")
            self.context_extender = ContextExtender(
                model=self.model,
                tokenizer=self.tokenizer,
                patcher=self.model_patcher,
                registry=self.registry
            )
            self.context_extender.extend_context(max_context_length)
        else:
            self.context_extender = None
        
        # Get adaptation controller if available
        self.adaptation_controller = getattr(self.model_patcher, "adaptation_controller", None)
        
        # Chat history
        self.chat_history = []
    
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the model with HSA attention.
        
        Args:
            prompt: The input prompt
            
        Returns:
            The generated response
        """
        # Add the prompt to the history
        self.chat_history.append({"role": "user", "content": prompt})
        
        # Create full conversation input
        full_prompt = self._format_conversation()
        
        # Tokenize input
        inputs = self.tokenizer(full_prompt, return_tensors="pt", padding=True)
        
        # Generate response
        with torch.no_grad():
            # Generate with HSA
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode and extract the response
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output_text[len(full_prompt):].strip()
        
        # Add response to history
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Print stats if enabled
        if self.show_stats:
            self._display_stats()
        
        return response
    
    def _format_conversation(self) -> str:
        """Format the conversation history into a prompt string."""
        # This is a simple implementation - customize based on your model's expected format
        formatted = ""
        for message in self.chat_history:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"
        
        # Add the assistant prefix for the next response
        formatted += "Assistant: "
        
        return formatted
    
    def _display_stats(self):
        """Display HSA statistics."""
        # Get stats from model patcher
        stats = self.model_patcher.get_stats()
        
        # Get registry stats
        registry_stats = {
            "total_splats": self.registry.count_splats(),
            "splats_by_level": {
                level: self.registry.count_splats(level)
                for level in self.registry.hierarchy.levels
            }
        }
        
        # Display stats
        print("\n--- HSA Statistics ---")
        print(f"Total splats: {registry_stats['total_splats']}")
        print("Splats by level:")
        for level, count in registry_stats['splats_by_level'].items():
            print(f"  {level}: {count}")
        
        # Display attention stats
        if "attention_stats" in stats:
            att_stats = stats["attention_stats"]
            print(f"Attention calls: {att_stats.get('calls', 0)}")
            print(f"Tokens processed: {att_stats.get('tokens_processed', 0)}")
        
        # Display adaptation stats if available
        if self.adaptation_controller:
            adapt_stats = self.adaptation_controller.get_adaptation_statistics()
            print("Adaptation stats:")
            print(f"  Total adaptations: {adapt_stats.get('total_adaptations', 0)}")
            for op_type, count in adapt_stats.get('adaptation_counts', {}).items():
                print(f"  {op_type}: {count}")
        
        # Display context extension stats if available
        if self.context_extender:
            ext_stats = self.context_extender.get_stats()
            print("Context extension:")
            print(f"  Original length: {ext_stats.get('original_length', 0)}")
            print(f"  Current length: {ext_stats.get('current_length', 0)}")
        
        print("----------------------\n")
    
    def _display_detailed_stats(self):
        """Display detailed HSA statistics."""
        # Get stats from model patcher
        stats = self.model_patcher.get_stats()
        
        print("\n=== Detailed HSA Statistics ===")
        
        # Registry information
        print("\nRegistry Status:")
        print(f"  Total registered splats: {self.registry.registered_count}")
        print(f"  Total unregistered splats: {self.registry.unregistered_count}")
        print(f"  Recovery count: {self.registry.recovery_count}")
        
        # Hierarchy information
        print("\nHierarchy Configuration:")
        for i, level in enumerate(self.registry.hierarchy.levels):
            weight = self.registry.hierarchy.level_weights[i]
            init_count = self.registry.hierarchy.init_splats_per_level[i]
            current_count = self.registry.count_splats(level)
            print(f"  {level}: weight={weight:.2f}, init={init_count}, current={current_count}")
        
        # Sample splats
        print("\nSample Splats:")
        for level in self.registry.hierarchy.levels:
            splats = list(self.registry.get_splats_at_level(level))
            if not splats:
                continue
                
            # Get a random splat to display
            import random
            sample_splat = random.choice(splats)
            
            print(f"\n  Level: {level}, Sample Splat ID: {sample_splat.id}")
            print(f"    Amplitude: {sample_splat.amplitude:.4f}")
            print(f"    Lifetime: {sample_splat.lifetime}")
            print(f"    Average Activation: {sample_splat.get_average_activation():.4f}")
            print(f"    Children: {len(sample_splat.children)}")
        
        # Model patching information
        print("\nModel Patching:")
        print(f"  Model type: {stats.get('model_type', 'unknown')}")
        print(f"  Patched modules: {stats.get('total_patched', 0)}")
        
        # Attention statistics
        if "attention_stats" in stats:
            att_stats = stats["attention_stats"]
            print("\nAttention Performance:")
            for key, value in att_stats.items():
                print(f"  {key}: {value}")
        
        # Adaptation statistics
        if self.adaptation_controller:
            adapt_stats = self.adaptation_controller.get_adaptation_statistics()
            print("\nAdaptation Statistics:")
            print(f"  Total Adaptations: {adapt_stats.get('total_adaptations', 0)}")
            for op_type, count in adapt_stats.get('adaptation_counts', {}).items():
                print(f"  {op_type}: {count}")
        
        # Context extension stats if available
        if self.context_extender:
            ext_stats = self.context_extender.get_stats()
            print("\nContext Extension:")
            for key, value in ext_stats.items():
                print(f"  {key}: {value}")
        
        print("\n==============================\n")
    
    def _save_state(self):
        """Save the current HSA state."""
        from hsa.serialization_core import HSASerializer
        
        try:
            # Create serializer
            serializer = HSASerializer()
            
            # Save registry
            filename = f"hsa_state_{int(time.time())}.bin"
            serializer.save_to_file(self.registry, filename)
            
            print(f"HSA state saved to {filename}")
        except Exception as e:
            print(f"Error saving state: {e}")
    
    def _reset_hsa(self):
        """Reset HSA to initial state."""
        # Re-initialize registry
        self.registry.initialize_splats()
        
        # Reset adaptation controller if present
        if self.adaptation_controller:
            self.adaptation_controller.reset_statistics()
        
        print("HSA state reset to initial configuration")
    
    def run(self):
        """Run the chat interface."""
        print(f"Chatting with {self.model_name} using Hierarchical Splat Attention")
        print("Type 'exit' to quit, 'stats' to show detailed statistics, 'save' to save state, 'reset' to reset HSA")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "stats":
                self._display_detailed_stats()
                continue
            elif user_input.lower() == "save":
                self._save_state()
                print("Current state saved.")
                continue
            elif user_input.lower() == "reset":
                self._reset_hsa()
                print("HSA state reset.")
                continue
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            
            try:
                response = self.generate_response(user_input)
                print(response)
            except Exception as e:
                print(f"\nError generating response: {e}")
                import traceback
                print(traceback.format_exc())
    
    def cleanup(self):
        """Clean up resources before exiting."""
        logger.info("Cleaning up...")
        
        # Restore original model if needed
        if hasattr(self.model_patcher, "restore_model"):
            logger.info("Restoring original model attention")
            self.model_patcher.restore_model()
        
        # Restore original context if extended
        if self.context_extender:
            logger.info("Restoring original context length")
            self.context_extender.restore_original_context()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Chat with an HSA-enhanced model")
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt2",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=2048,
        help="Maximum sequence length for generation"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for generation"
    )
    parser.add_argument(
        "--no-sparse", 
        action="store_true",
        help="Disable sparse attention"
    )
    parser.add_argument(
        "--no-adaptation", 
        action="store_true",
        help="Disable HSA adaptation"
    )
    parser.add_argument(
        "--no-stats", 
        action="store_true",
        help="Hide HSA stats after generation"
    )
    parser.add_argument(
        "--extend-context", 
        action="store_true",
        help="Enable context window extension"
    )
    parser.add_argument(
        "--max-context", 
        type=int, 
        default=4096,
        help="Maximum context length when extending context"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config()
    
    # Override config with command line arguments
    if args.config:
        try:
            with open(args.config, "r") as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
    
    # Command line args take precedence over config file
    if args.model:
        config["model"] = args.model
    if args.max_length:
        config["max_length"] = args.max_length
    if args.temperature:
        config["temperature"] = args.temperature
    if args.no_sparse:
        config["use_sparse"] = False
    if args.no_adaptation:
        config["adaptation_enabled"] = False
    if args.no_stats:
        config["show_stats"] = False
    if args.extend_context:
        config["enable_context_extension"] = True
    if args.max_context:
        config["max_context_length"] = args.max_context
    
    try:
        # Initialize chat interface
        chat = ChatCLI(
            model_name=config.get("model", "gpt2"),
            max_length=config.get("max_length", 2048),
            temperature=config.get("temperature", 0.7),
            use_sparse=config.get("use_sparse", True),
            adaptation_enabled=config.get("adaptation_enabled", True),
            show_stats=config.get("show_stats", True),
            extend_context=config.get("enable_context_extension", False),
            max_context_length=config.get("max_context_length", 4096)
        )
        
        try:
            # Run chat loop
            chat.run()
        finally:
            # Clean up resources
            chat.cleanup()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
