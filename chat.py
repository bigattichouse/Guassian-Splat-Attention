#!/usr/bin/env python3
"""
HSA-enhanced Chat Demo

This script demonstrates the Hierarchical Splat Attention (HSA) mechanism
by applying it to a pre-trained language model (GPT-2) for chat interactions.

Usage:
    python chat.py [--max_length MAX_LENGTH] [--log_level LOG_LEVEL]

Options:
    --max_length   Maximum length for generated responses (default: 100)
    --log_level    Logging level (default: INFO)
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hsa_chat.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the current directory to the Python path to ensure modules can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: This script requires the transformers library.")
    print("Please install it with: pip install transformers")
    sys.exit(1)

# Import HSA modules with proper error handling
try:
    # Import from local modules
    from hsa.core import create_hsa, HSA
    from hsa.model_integration import replace_attention_with_hsa
    from hsa.adaptation.core import AdaptationType
    from hsa.attention.metrics import SplatAttentionMetrics
    
    # Verify imports worked
    logger.info("Successfully imported HSA modules")
except ImportError as e:
    # Show detailed error message
    print(f"Error importing HSA modules: {e}")
    print("\nTroubleshooting steps:")
    print("1. Make sure the 'hsa' directory is in the same directory as this script")
    print("2. Check that all required __init__.py files exist in the package directories")
    print("3. Try running with Python -m: python -m chat")
    print("4. Make sure all dependencies are installed (numpy, torch, etc.)")
    sys.exit(1)

def print_with_delay(text, delay=0.003):
    """Print text with a typing effect for better UX."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chat with an HSA-enhanced GPT-2 model")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length for generated responses (default: 100)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    return parser.parse_args()

def initialize_model_and_hsa():
    """Initialize the GPT-2 model and HSA components."""
    model_name = "gpt2"
    print(f"Loading {model_name} model...")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Make sure the tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        print("Make sure you have internet access and the model name is correct.")
        sys.exit(1)
    
    print("Creating HSA instance...")
    # Create HSA with custom configuration
    hsa_config = {
        "hierarchy": {
            "levels": ["Token", "Phrase", "Document"],
            "init_splats_per_level": [20, 10, 5],
            "level_weights": [0.5, 0.3, 0.2]
        },
        "adaptation": {
            "mitosis_threshold": 0.005,
            "death_threshold": 0.005,
            "consecutive_batches": 1,
            "adaptation_frequency": 1,
            "enable_adaptation": True
        },
        "attention": {
            "sparse_topk": 64
        },
        "model": {
            "type": "gpt",
            "replace_in_place": True
        }
    }
    hsa = create_hsa(hsa_config)
    
    print("Converting model to use HSA attention...")
    try:
        # Convert the model to use HSA attention
        model = hsa.convert_model(model, model_type="gpt")
        logger.info("Successfully converted model to use HSA attention")
    except Exception as e:
        logger.error(f"Error converting model: {e}")
        print(f"Error converting model: {e}")
        print("Falling back to original model without HSA.")
    
    print("Initializing HSA splats...")
    try:
        # Use model's word embeddings as initialization tokens
        sample_embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
        sample_size = min(1000, sample_embeddings.shape[0])
        sample_tokens = sample_embeddings[:sample_size]
        hsa.initialize(sample_tokens)
        logger.info(f"Successfully initialized HSA with {sample_size} token embeddings")
    except Exception as e:
        logger.error(f"Error initializing HSA splats: {e}")
        # Create fallback embeddings
        dim = model.config.hidden_size
        sample_tokens = np.random.randn(100, dim)
        hsa.initialize(sample_tokens)
        logger.info(f"Initialized HSA with fallback random embeddings (dim={dim})")
    
    # Create adaptation metrics tracker
    metrics_tracker = SplatAttentionMetrics()
    
    return model, tokenizer, hsa, metrics_tracker

def chat_loop(model, tokenizer, hsa, metrics_tracker, max_length):
    """Run the interactive chat loop."""
    print("\n" + "="*50)
    print_with_delay("Welcome to the HSA-enhanced GPT-2 Chat Demo!")
    print("="*50)
    print_with_delay("This demo shows how Hierarchical Splat Attention works with GPT-2.")
    print_with_delay("Type your messages and the model will respond.")
    print_with_delay("Type 'exit' or 'quit' to end the chat.")
    print_with_delay("Type 'stats' to see the current HSA statistics.")
    print_with_delay("Type 'adapt' to force adaptation of the HSA splats.")
    print_with_delay("Type 'debug' to toggle debug information during responses.")
    print_with_delay("Type 'help' to see these instructions again.")
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
    show_debug = False
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Start chat loop
    while True:
        # Get user input
        user_input = input("\n\033[1mYou:\033[0m ")
        user_input = user_input.strip()
        
        # Handle special commands
        if user_input.lower() in ['exit', 'quit']:
            print_with_delay("\nThank you for trying the HSA-enhanced chat. Goodbye!")
            break
            
        if user_input.lower() == 'stats':
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
            print(f"Adaptations - Mitosis: {stats['adaptations']['mitosis']}, "
                 f"Birth: {stats['adaptations']['birth']}, "
                 f"Death: {stats['adaptations']['death']}, "
                 f"Merge: {stats['adaptations']['merge']}")
            continue
            
        if user_input.lower() == 'help':
            print("\n\033[1mCommands:\033[0m")
            print("  - Type your message to chat with the model")
            print("  - 'stats': Show current HSA statistics")
            print("  - 'adapt': Force adaptation of HSA splats")
            print("  - 'debug': Toggle debug information")
            print("  - 'exit' or 'quit': End the chat")
            print("  - 'help': Show this help message")
            continue
            
        if user_input.lower() == 'adapt':
            print("\n\033[1mForcing HSA adaptation...\033[0m")
            try:
                # Get embeddings from the model's vocabulary
                with torch.no_grad():
                    vocab_size = min(1000, model.get_input_embeddings().weight.shape[0])
                    random_ids = torch.randint(0, vocab_size, (50,)).to(device)
                    embeddings = model.get_input_embeddings()(random_ids).detach().cpu().numpy()
                
                # Adapt splats using these embeddings
                adaptation_results = hsa.adapt(embeddings, cpu_efficient=True)
                
                # Report results
                print(f"Adaptation complete!")
                print(f"  - Birth events: {adaptation_results.get('birth', 0)}")
                print(f"  - Mitosis events: {adaptation_results.get('mitosis', 0)}")
                print(f"  - Death events: {adaptation_results.get('death', 0)}")
                print(f"  - Merge events: {adaptation_results.get('merge', 0)}")
            except Exception as e:
                print(f"Error during adaptation: {e}")
                logger.error(f"Error during forced adaptation: {e}")
            continue
            
        if user_input.lower() == 'debug':
            show_debug = not show_debug
            print(f"\n\033[1mDebug mode {'enabled' if show_debug else 'disabled'}\033[0m")
            continue
            
        if not user_input:
            continue
        
        # Process user input
        try:
            # Format prompt for better results - simple conversation format for GPT-2
            prompt = f"Human: {user_input}\nAI:"
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # Generate response
            print("\n\033[1mModel:\033[0m ", end='', flush=True)
            
            # Show debug information about token count if enabled
            if show_debug:
                print(f"\n\033[90m(Input tokens: {inputs.input_ids.shape[1]})\033[0m")
            
            # Generate response with the model
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                )
            generation_time = time.time() - start_time
            
            # Get response without the input prompt
            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the AI's response by removing the prompt
            ai_response = response_text.split("AI:")[1].split("Human:")[0].strip() if "AI:" in response_text else response_text
            
            # Print the response with a typing effect
            print_with_delay(ai_response, delay=0.01)
            
            # Show debug information if enabled
            if show_debug:
                print(f"\n\033[90m(Generation time: {generation_time:.2f}s, "
                     f"Output tokens: {outputs.shape[1] - inputs.input_ids.shape[1]})\033[0m")
            
            # Always adapt HSA splats after each interaction
            print(f"\n\033[90m(Adapting HSA splats...)\033[0m" if show_debug else "", end="")
            try:
                adapt_start = time.time()
                
                # Get embeddings from the model for the input and output
                with torch.no_grad():
                    # Extract embeddings from the full sequence (including generated output)
                    full_embeddings = model.get_input_embeddings()(outputs[0]).detach().cpu().numpy()
                
                # Adapt splats using these embeddings
                adaptation_results = hsa.adapt(full_embeddings, cpu_efficient=True)
                
                # Record the adaptation time
                adapt_time = time.time() - adapt_start
                
                # Show adaptation results if debug mode is enabled
                if show_debug:
                    print(f"\033[90m(Adaptation complete in {adapt_time:.2f}s)\033[0m")
                    print(f"\033[90m(Events: {adaptation_results.get('birth', 0)} birth, "
                         f"{adaptation_results.get('mitosis', 0)} mitosis, "
                         f"{adaptation_results.get('death', 0)} deaths, "
                         f"{adaptation_results.get('merge', 0)} merges)\033[0m")
                
                # Log adaptation results
                logger.info(f"Adaptation results - "
                           f"Birth: {adaptation_results.get('birth', 0)}, "
                           f"Mitosis: {adaptation_results.get('mitosis', 0)}, "
                           f"Death: {adaptation_results.get('death', 0)}, "
                           f"Merge: {adaptation_results.get('merge', 0)}, "
                           f"Total time: {adapt_time:.2f}s")
            except Exception as e:
                error_msg = f"Error during adaptation: {e}"
                if show_debug:
                    print(f"\033[90m({error_msg})\033[0m")
                logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(f"\n{error_msg}")
            logger.error(error_msg)
            print("Try asking something else or check if the model is properly loaded.")

def main():
    """Main function to run the chat demo."""
    args = parse_arguments()
    
    # Set the logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Initialize components
    model, tokenizer, hsa, metrics_tracker = initialize_model_and_hsa()
    
    # Run the chat loop
    chat_loop(model, tokenizer, hsa, metrics_tracker, args.max_length)

if __name__ == "__main__":
    main()
