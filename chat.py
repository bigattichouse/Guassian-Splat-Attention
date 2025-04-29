#!/usr/bin/env python3
"""
HSA-enhanced Chat Demo

This script demonstrates the Hierarchical Splat Attention (HSA) mechanism
by applying it to a pre-trained language model for chat interactions.

Usage:
    python chat_with_hsa.py [--model MODEL_NAME] [--max_length MAX_LENGTH]

Options:
    --model        The transformer model to use (default: gpt2)
    --max_length   Maximum length for generated responses (default: 100)
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import logging
import time

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

# Add the parent directory to the path to find the HSA modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: This script requires the transformers library.")
    print("Please install it with: pip install transformers")
    sys.exit(1)

try:
    from hsa import create_hsa
    from hsa.adaptation import AdaptationType
    from hsa.attention import SplatAttentionMetrics
except ImportError:
    print("Error: Could not import HSA modules.")
    print("Make sure you're running this script from the directory containing the HSA package.")
    sys.exit(1)

def print_with_delay(text, delay=0.003):
    """Print text with a typing effect for better UX."""
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Chat with an HSA-enhanced language model")
    parser.add_argument("--model", type=str, default="gpt2", 
                        help="The transformer model to use (default: gpt2)")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length for generated responses (default: 100)")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    return parser.parse_args()

def initialize_model_and_hsa(model_name):
    """Initialize the model and HSA components."""
    print(f"Loading {model_name} model...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
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
            "init_splats_per_level": [10, 8, 2],
            "level_weights": [0.5, 0.3, 0.2]
        },
        "adaptation": {
            "mitosis_threshold": 0.1,
            "death_threshold": 0.01,
            "consecutive_batches": 3,
            "adaptation_frequency": 5,
            "enable_adaptation": True
        },
        "attention": {
            "sparse_topk": 64
        }
    }
    hsa = create_hsa(hsa_config)
    
    print("Converting model to use HSA attention...")
    
    # Check if we should actually modify the model or just simulate integration
    use_simulation = False  # Setting to True to avoid modifying the model directly
    
    if use_simulation:
        print("Using simulation mode (not actually modifying model attention)...")
        logger.info("Using HSA in simulation mode without modifying the model")
        # Skip actual model conversion
    else:
        try:
            # Try to determine the model type from its name
            if "gpt" in model_name.lower():
                model_type = "gpt"
            elif "bert" in model_name.lower():
                model_type = "bert"
            else:
                model_type = "default"
                
            model = hsa.convert_model(model, model_type=model_type)
            logger.info(f"Successfully converted model to use HSA with model_type={model_type}")
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
        dim = 768 if hasattr(model.config, 'hidden_size') else 64
        sample_tokens = np.random.randn(100, dim)
        hsa.initialize(sample_tokens)
        logger.info(f"Initialized HSA with fallback random embeddings (dim={dim})")
    
    # Create adaptation metrics tracker
    metrics_tracker = SplatAttentionMetrics()
    
    return model, tokenizer, hsa, metrics_tracker

def chat_loop(model, tokenizer, hsa, metrics_tracker, max_length):
    """Run the interactive chat loop."""
    print("\n" + "="*50)
    print_with_delay("Welcome to the HSA-enhanced Chat Demo!")
    print("="*50)
    print_with_delay("This demo shows how Hierarchical Splat Attention works with a language model.")
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
    
    # Create a warm-up system prompt with diverse examples
    warm_up_prompt = """You are a helpful, creative, and thoughtful assistant. You provide detailed, nuanced, and interesting responses that engage with the user's questions thoughtfully.

Example conversations:

User: Tell me about butterflies.
Assistant: Butterflies are fascinating insects that belong to the order Lepidoptera. They're known for their colorful wing patterns and remarkable life cycle that includes complete metamorphosis - from egg to larva (caterpillar) to pupa (chrysalis) and finally to adult butterfly. Each species has unique wing patterns, behaviors, and ecological roles. Many serve as important pollinators and indicators of ecosystem health. Some, like the Monarch butterfly, are known for their incredible migrations spanning thousands of miles. What aspect of butterflies interests you most?

User: I'm feeling sad today.
Assistant: I'm sorry to hear you're feeling sad today. Many things can trigger sadness - disappointments, stress, changes in life, or sometimes it just appears without an obvious cause. Would it help to talk about what's contributing to your feelings? Sometimes expressing what's on your mind can provide some relief. In the meantime, consider some gentle self-care like taking a walk, connecting with a friend, or engaging in an activity that usually brings you joy. Remember that emotions naturally ebb and flow, and sadness, while uncomfortable, is a normal part of the human experience.

User: Write a short story about a space explorer.
Assistant: THE FORGOTTEN MOON

Captain Elara Wei stepped onto the surface of Titan X-7, her boots sinking slightly into the moon's powdery regolith. Nineteen years of interstellar travel, and she was the first human to reach this system.

"Base, I've made landfall," she reported, her voice traveling across light-years. The delay meant no response would come for hours.

As she collected samples, something glinted beneath the dust. Brushing it aside, she revealed a smooth metallic object—artificial, unmistakably.

Elara's breath caught. "Base, you won't believe this. There's... evidence of previous visitation."

She turned the object over and froze. Etched on its surface was a name and date: "E. Wei, 2189."

But this was 2178. And she'd never been here before.

Had she?

User: What's the capital of France?
Assistant: The capital of France is Paris. It's located in the north-central part of the country on the Seine River and is one of the world's major cultural, artistic, and historical centers. Paris is famous for landmarks like the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum, which houses thousands of works of art including the Mona Lisa.

User: How do computers store data?
Assistant: Computers store data through a binary system, using bits (binary digits) that can be either 0 or 1. These bits are physically represented by the presence or absence of electrical charge in storage media.

At the hardware level, different storage technologies implement this binary system in various ways:
- Hard disk drives (HDDs) use magnetic polarity on rotating disks
- Solid-state drives (SSDs) use floating-gate transistors that maintain electrical charges
- RAM uses capacitors that need constant refreshing
- Optical media like CDs and DVDs use physical pits and lands that reflect light differently

These binary digits are organized into larger units (bytes, kilobytes, megabytes, etc.) and structured through file systems that help the computer locate and manage the stored information. The binary nature of computer storage is the foundation that enables everything from simple text documents to complex applications and multimedia.

This conversation will help guide our interactive chat. Please type your message, and I'll respond thoughtfully.
"""

    # Initialize chat history with the warm-up prompt
    chat_history = [("system", warm_up_prompt)]
    
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
            print(f"Adaptations - Mitosis: {stats['adaptations']['mitosis']}, Death: {stats['adaptations']['death']}")
            
            # Show adaptation stats
            print(f"\nOverall adaptation history:")
            print(f"  - Total adaptations: {stats['adaptations']['mitosis'] + stats['adaptations']['death']}")
            print(f"  - Mitosis events: {stats['adaptations']['mitosis']}")
            print(f"  - Death events: {stats['adaptations']['death']}")
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
                    random_ids = torch.randint(0, vocab_size, (50,))
                    embeddings = model.get_input_embeddings()(random_ids).detach().cpu().numpy()
                
                # Adapt splats using these embeddings - updated to use cpu_optimization parameter
                adaptation_results = hsa.adapt(embeddings, cpu_efficient=False)
                
                # Report results
                print(f"Adaptation complete!")
                print(f"  - Mitosis events: {adaptation_results.get('mitosis', 0)}")
                print(f"  - Death events: {adaptation_results.get('death', 0)}")
            except Exception as e:
                print(f"Error during adaptation: {e}")
                logger.error(f"Error during forced adaptation: {e}")
            continue
            
        if user_input.lower() == 'debug':
            show_debug = not show_debug
            print(f"\n\033[1mDebug mode {'enabled' if show_debug else 'disabled'}\033[0m")
            continue
            
        if user_input.lower() == 'fallback':
            print("\n\033[1mFallback mode - testing generation without HSA modifications...\033[0m")
            try:
                # Use fallback generation for testing
                with torch.no_grad():
                    # Create a basic prompt
                    fallback_prompt = "Hello, how are you today?"
                    fallback_inputs = tokenizer(fallback_prompt, return_tensors="pt")
                    
                    # Try to access the original attention mechanism
                    # This is a backup test to see if we can generate without the HSA modifications
                    for module in model.modules():
                        if hasattr(module, 'self_attention') and hasattr(module.self_attention, '_attention_impl'):
                            # Temporarily disable the HSA attention
                            original_impl = module.self_attention._attention_impl
                            module.self_attention._attention_impl = None
                    
                    outputs = model.generate(
                        fallback_inputs.input_ids,
                        max_length=fallback_inputs.input_ids.shape[1] + 20,
                        num_return_sequences=1
                    )
                    
                    # Restore original implementation
                    for module in model.modules():
                        if hasattr(module, 'self_attention') and hasattr(module.self_attention, '_attention_impl'):
                            module.self_attention._attention_impl = original_impl
                    
                    fallback_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    print(f"Fallback response: {fallback_response}")
                    
            except Exception as e:
                print(f"Fallback test also failed: {e}")
            continue
            
        if not user_input:
            continue
        
        # Add to chat history
        chat_history.append(("user", user_input))
        
        # Process user input
        try:
            # Format complete history for context
            full_prompt = ""
            
            # Start with the system message
            if chat_history and chat_history[0][0] == "system":
                system_message = chat_history[0][1]
            else:
                system_message = ""
            
            # Format the prompt for better results
            full_prompt = system_message + "\n\n"
            
            # Only include the last 5 exchanges to keep context manageable
            recent_history = [msg for msg in chat_history if msg[0] != "system"][-5:]
            
            for role, message in recent_history:
                prefix = "User: " if role == "user" else "Assistant: "
                full_prompt += prefix + message + "\n\n"
            
            # Add the current turn
            full_prompt += "User: " + user_input + "\n\nAssistant:"
            
            # Tokenize input
            inputs = tokenizer(full_prompt, return_tensors="pt")
            
            # Generate response
            print("\n\033[1mModel:\033[0m ", end='', flush=True)
            
            # Show debug information about token count if enabled
            if show_debug:
                print(f"\n\033[90m(Input tokens: {inputs.input_ids.shape[1]})\033[0m")
            
            # Create an attention mask (1 for all input tokens)
            attention_mask = torch.ones_like(inputs.input_ids)
            
            try:
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=attention_mask,
                        max_length=inputs.input_ids.shape[1] + max_length,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        # Increase temperature for more diversity in responses
                        do_sample=True,
                        temperature=0.9,
                        top_p=0.92,
                        # Add repetition penalty to reduce looping
                        repetition_penalty=1.2,
                        # Ensure we're using the model's default attention mechanism
                        use_cache=True
                    )
                generation_time = time.time() - start_time
                
                # Get response without the input text
                input_length = inputs.input_ids.shape[1]
                response_ids = outputs[0][input_length:]
                
                # Decode and print response with a typing effect
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
                print_with_delay(response, delay=0.01)
                
                # Update chat history with model's response
                chat_history.append(("assistant", response))
                
                # Show debug information if enabled
                if show_debug:
                    print(f"\n\033[90m(Generation time: {generation_time:.2f}s, Output tokens: {len(response_ids)})\033[0m")
            except Exception as e:
                error_msg = f"Error during generation: {e}"
                print(f"\n{error_msg}")
                logger.error(error_msg)
                print("Falling back to standard generation without HSA...")
                
                # Try fallback generation without special parameters
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs.input_ids,
                            max_length=inputs.input_ids.shape[1] + max_length,
                            num_return_sequences=1,
                            pad_token_id=tokenizer.eos_token_id,
                            # Simpler parameters for fallback
                            do_sample=False
                        )
                    
                    # Get response without the input text
                    input_length = inputs.input_ids.shape[1]
                    response_ids = outputs[0][input_length:]
                    
                    # Decode and print response with a typing effect
                    response = tokenizer.decode(response_ids, skip_special_tokens=True)
                    print_with_delay(response, delay=0.01)
                    
                    # Update chat history with model's response
                    chat_history.append(("assistant", response))
                except Exception as e:
                    logger.error(f"Fallback generation also failed: {e}")
                    print(f"Fallback generation also failed: {e}")
                    response = "I'm having trouble generating a response right now. Please try again."
            
            # Adapt HSA splats based on this interaction
            if len(inputs.input_ids[0]) > 5:  # Only adapt on longer inputs
                should_adapt = True
                adapt_message = "Adapting HSA splats"
                if len(chat_history) % 3 != 0 and not show_debug:
                    should_adapt = False
                    adapt_message = "Skipping adaptation (only adapting every 3 interactions)"
                
                if show_debug or should_adapt:
                    print(f"\n\033[90m({adapt_message}...)\033[0m")
                
                if should_adapt:
                    try:
                        print(f"\n\033[90m(Starting HSA adaptation at {time.strftime('%H:%M:%S')}...)\033[0m")
                        adapt_start = time.time()
                        print(f"\033[90m(Step 1/4: Extracting embeddings...)\033[0m")
                        
                        # Get embeddings from the model
                        embed_start = time.time()
                        with torch.no_grad():
                            embeddings = model.get_input_embeddings()(inputs.input_ids).detach().cpu().numpy()[0]
                        embed_time = time.time() - embed_start
                        print(f"\033[90m(Step 1/4 complete: Extracted {len(embeddings)} embeddings in {embed_time:.2f}s)\033[0m")
                        
                        # Step 2: Compute attention matrix for analysis (optional)
                        print(f"\033[90mVRAM usage: {torch.cuda.memory_allocated() / 1e6:.2f} MB\033[0m")
                        print(f"\033[90m(Step 2/4: Computing attention matrix...)\033[0m")
                        print(f"\033[90mComputing {len(embeddings)} embeddings (shape: {embeddings.shape})\033[0m")

                        attention_start = time.time()
                        # Use the updated compute_attention method
                        attention_matrix = hsa.compute_attention(embeddings)
                        attention_time = time.time() - attention_start
                        print(f"\033[90m(Step 2/4 complete: Computed attention matrix in {attention_time:.2f}s)\033[0m")
                        
                        # Step 3: Checking for adaptation triggers
                        print(f"\033[90m(Step 3/4: Checking adaptation triggers...)\033[0m")
                        trigger_start = time.time()
                        
                        # Count splats before adaptation
                        splats_before = len(hsa.splat_registry.splats)
                        level_counts_before = {}
                        for level in hsa.hierarchy.levels:
                            level_counts_before[level] = len(hsa.splat_registry.get_splats_at_level(level))
                            
                        # Adapt splats - updated to use cpu_optimization parameter
                        print(f"\033[90m(Step 4/4: Adapting splats...)\033[0m")
                        adaptation_results = hsa.adapt(embeddings, cpu_efficient=False)
                        
                        # Count splats after adaptation
                        splats_after = len(hsa.splat_registry.splats)
                        level_counts_after = {}
                        for level in hsa.hierarchy.levels:
                            level_counts_after[level] = len(hsa.splat_registry.get_splats_at_level(level))
                        
                        # Calculate changes
                        total_change = splats_after - splats_before
                        level_changes = {level: level_counts_after[level] - level_counts_before[level] 
                                         for level in hsa.hierarchy.levels}
                        
                        trigger_time = time.time() - trigger_start
                        total_time = time.time() - adapt_start
                        
                        # Print comprehensive report
                        print(f"\033[90m(Adaptation complete at {time.strftime('%H:%M:%S')} - total time: {total_time:.2f}s)\033[0m")
                        print(f"\033[90m(Splats: {splats_before} → {splats_after}, change: {total_change:+d})\033[0m")
                        
                        # Print level changes
                        level_report = " | ".join([f"{level}: {count} → {level_counts_after[level]} ({change:+d})" 
                                                  for level, count, change in 
                                                  zip(hsa.hierarchy.levels, 
                                                      [level_counts_before[l] for l in hsa.hierarchy.levels],
                                                      [level_changes[l] for l in hsa.hierarchy.levels])])
                        print(f"\033[90m(Levels: {level_report})\033[0m")
                        
                        # Show breakdown of adaptation events
                        print(f"\033[90m(Events: {adaptation_results.get('mitosis', 0)} mitosis, {adaptation_results.get('death', 0)} deaths)\033[0m")
                        
                        adapt_time = time.time() - adapt_start
                        
                        if show_debug:
                            print(f"\033[90m(Adaptation performance: embeddings {embed_time:.2f}s, "
                                  f"attention {attention_time:.2f}s, adaptation {trigger_time:.2f}s, "
                                  f"total {adapt_time:.2f}s)\033[0m")
                        
                        # Log adaptation results
                        logger.info(f"Adaptation results - "
                                    f"Mitosis: {adaptation_results.get('mitosis', 0)}, "
                                    f"Death: {adaptation_results.get('death', 0)}, "
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
    model, tokenizer, hsa, metrics_tracker = initialize_model_and_hsa(args.model)
    
    # Run the chat loop
    chat_loop(model, tokenizer, hsa, metrics_tracker, args.max_length)

if __name__ == "__main__":
    main()
