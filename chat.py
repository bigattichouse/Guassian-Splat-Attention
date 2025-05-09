#!/usr/bin/env python3
"""
HSA Inference CLI - Command-line interface for inference with locally trained HSA models.

This application loads a locally saved HSA model, along with its registry state,
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
from hsa.attention_interface import AttentionConfig
from hsa.adaptation_controller import AdaptationController
from hsa.model_adapter import create_adapter_for_model, ContextExtender
from hsa.serialization_core import HSASerializer
from hsa.compact_serialization import load_registry_compact

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class HSAInference:
    """Interface for inference with HSA-enhanced models."""
    
    def __init__(
        self,
        model_path: str,
        registry_path: Optional[str] = None,
        max_length: int = 2048,
        temperature: float = 0.7,
        use_sparse: bool = True,
        adaptation_enabled: bool = True,
        show_stats: bool = True,
        device: str = "cpu"
    ):
        """Initialize inference interface.
        
        Args:
            model_path: Path to the local model directory
            registry_path: Path to the saved HSA registry file (if None, look in model_path)
            max_length: Maximum sequence length for generation
            temperature: Temperature for generation
            use_sparse: Whether to use sparse attention
            adaptation_enabled: Whether to enable HSA adaptation
            show_stats: Whether to show HSA stats after each generation
            device: Device to run the model on ("cpu", "cuda", etc.)
        """
        self.model_path = model_path
        self.registry_path = registry_path
        self.max_length = max_length
        self.temperature = temperature
        self.use_sparse = use_sparse
        self.adaptation_enabled = adaptation_enabled
        self.show_stats = show_stats
        self.device = device
        
        # Load model and tokenizer
        logger.info(f"Loading model and tokenizer from: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            
            # Move model to the specified device
            self.model.to(device)
            
            logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError(f"Failed to load model from {model_path}: {e}")
        
        # Enable padding in the tokenizer if not already enabled
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Try to find registry file if not specified
        if not self.registry_path:
            # Check for common registry filenames in the model directory
            for filename in ["final_registry.bin", "final_registry_compact.bin", 
                            "registry.bin", "registry_compact.bin"]:
                potential_path = os.path.join(model_path, filename)
                if os.path.exists(potential_path):
                    self.registry_path = potential_path
                    logger.info(f"Found registry file: {potential_path}")
                    break
        
        # Initialize or load registry
        if self.registry_path and os.path.exists(self.registry_path):
            logger.info(f"Loading HSA registry from: {self.registry_path}")
            
            try:
                # Try loading with compact serialization first
                if self.registry_path.endswith(("_compact.bin", ".json")):
                    self.registry = load_registry_compact(self.registry_path)
                else:
                    # Fall back to standard serialization
                    serializer = HSASerializer()
                    self.registry = serializer.load_from_file(self.registry_path)
                
                logger.info(f"Registry loaded successfully, containing {self.registry.count_splats()} splats")
                
                # Create adapter with loaded registry
                self.model_patcher, _ = create_adapter_for_model(
                    model=self.model,
                    use_sparse=use_sparse,
                    adaptation_enabled=adaptation_enabled
                )
                
                # Replace with loaded registry
                self.model_patcher.registry = self.registry
                
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                logger.warning("Initializing new registry instead")
                self.model_patcher, self.registry = create_adapter_for_model(
                    model=self.model,
                    use_sparse=use_sparse,
                    adaptation_enabled=adaptation_enabled
                )
        else:
            logger.info("No registry file found, initializing new registry")
            self.model_patcher, self.registry = create_adapter_for_model(
                model=self.model,
                use_sparse=use_sparse,
                adaptation_enabled=adaptation_enabled
            )
        
        # Patch the model with HSA
        logger.info("Patching model with HSA attention")
        self.model_patcher.patch_model()
        
        # Get adaptation controller if available
        self.adaptation_controller = getattr(self.model_patcher, "adaptation_controller", None)
        
        # Chat history
        self.chat_history = []
        
    def _truncate_history(self, max_tokens=512):
        """Truncate history to avoid exceeding model's context window."""
        if not self.chat_history:
            return
            
        # Calculate current token count
        full_prompt = self._format_conversation()
        tokens = self.tokenizer.encode(full_prompt)
        
        # If we're under the limit, no need to truncate
        if len(tokens) <= max_tokens:
            return
            
        # Remove oldest messages until we're under the limit
        while len(tokens) > max_tokens and len(self.chat_history) > 1:
            self.chat_history.pop(0)  # Remove oldest message
            full_prompt = self._format_conversation()
            tokens = self.tokenizer.encode(full_prompt)
            
        logger.info(f"Truncated history to {len(tokens)} tokens (limit: {max_tokens})")
        
    def generate_response(self, prompt: str) -> str:
        """Generate a response using the model with HSA attention."""
        # Add the prompt to the history
        self.chat_history.append({"role": "user", "content": prompt})
    
        # Truncate history if needed
        self._truncate_history()
    
        # Create full conversation input
        full_prompt = self._format_conversation()
        
        # Tokenize input
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device) if "attention_mask" in inputs else None
        
        input_length = len(input_ids[0])
        
        # Calculate appropriate max_new_tokens value
        max_context = getattr(self.model.config, "max_position_embeddings", 1024)
        max_new_tokens = max(1, min(self.max_length, max_context - input_length))
        
        # Log generation parameters
        logger.info(f"Input length: {input_length}, Max new tokens: {max_new_tokens}")
        
        # Generate response
        with torch.no_grad():
            start_time = time.time()
            
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                attention_mask=attention_mask
            )
            
            generation_time = time.time() - start_time
        
        # Decode and extract the response
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = output_text[len(full_prompt):].strip()
        
        # Add response to history
        self.chat_history.append({"role": "assistant", "content": response})
        
        # Print stats if enabled
        if self.show_stats:
            self._display_stats(generation_time)
        
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
    
    def _display_stats(self, generation_time: float):
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
        print(f"Generation time: {generation_time:.2f} seconds")
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
        
        print("\n==============================\n")
    
    def _save_state(self, output_path=None):
        """Save the current HSA registry state."""
        try:
            # Import necessary components
            from hsa.compact_serialization import save_registry_compact
            
            # Create default output path if not provided
            if not output_path:
                output_dir = "hsa_states"
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"hsa_state_{int(time.time())}.bin")
            
            # Save registry in compact binary format
            size_mb = save_registry_compact(
                registry=self.registry,
                filepath=output_path,
                format="binary",
                include_history=True,
                compression_level=9
            )
            
            print(f"HSA registry state saved to {output_path} ({size_mb:.2f} MB)")
            return output_path
            
        except Exception as e:
            print(f"Error saving state: {e}")
            import traceback
            print(traceback.format_exc())
            return None
    
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
        print(f"HSA Inference CLI: Loaded model from {self.model_path}")
        if self.registry_path:
            print(f"Using HSA registry from {self.registry_path}")
        print(f"Model has {self.registry.count_splats()} active splats")
        print("\nCommands:")
        print("  'exit' to quit")
        print("  'stats' to show detailed statistics")
        print("  'save' to save current HSA state")
        print("  'reset' to reset HSA to initial state")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "stats":
                self._display_detailed_stats()
                continue
            elif user_input.lower() == "save":
                self._save_state()
                continue
            elif user_input.lower() == "reset":
                self._reset_hsa()
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run inference with a locally trained HSA model")
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Path to local model directory"
    )
    parser.add_argument(
        "--registry", 
        type=str, 
        default=None,
        help="Path to HSA registry file"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=1024,
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
        "--device", 
        type=str, 
        default="cpu",
        help="Device to run the model on (cpu, cuda, etc.)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize interface
        inference = HSAInference(
            model_path=args.model,
            registry_path=args.registry,
            max_length=args.max_length,
            temperature=args.temperature,
            use_sparse=not args.no_sparse,
            adaptation_enabled=not args.no_adaptation,
            show_stats=not args.no_stats,
            device=args.device
        )
        
        try:
            # Run chat loop
            inference.run()
        finally:
            # Clean up resources
            inference.cleanup()
            
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
