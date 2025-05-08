#!/usr/bin/env python
"""
HSA Evaluation Script.

This script evaluates an HSA-enhanced language model on perplexity,
generation quality, and runtime performance.
"""

import os
import argparse
import logging
import time
import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)
from datasets import load_dataset

# Import HSA components
from hsa.registry import SplatRegistry
from hsa.model_adapter import create_adapter_for_model, ModelPatcher
from hsa.serialization_core import HSASerializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hsa_evaluation.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate HSA-enhanced model")
    
    # Model and dataset arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model directory")
    parser.add_argument("--registry_path", type=str, required=True,
                        help="Path to the saved HSA registry")
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories",
                        help="Dataset name on Huggingface")
    parser.add_argument("--dataset_config", type=str, default="instruction",
                        help="Dataset configuration")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Output directory for evaluation results")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Evaluation batch size")
    parser.add_argument("--max_seq_length", type=int, default=256,
                        help="Maximum sequence length")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--use_base_model", action="store_true",
                        help="Also evaluate the base model without HSA")
    parser.add_argument("--generation_length", type=int, default=50,
                        help="Length of text to generate for generation evaluation")
    
    # HSA-specific arguments
    parser.add_argument("--use_sparse", action="store_true", default=True,
                        help="Use sparse attention computation")
    
    return parser.parse_args()


def load_model_and_registry(model_path, registry_path, use_sparse=True):
    """Load model and HSA registry."""
    logger.info(f"Loading model from {model_path}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Load registry
    logger.info(f"Loading HSA registry from {registry_path}")
    serializer = HSASerializer()
    registry = serializer.load_from_file(registry_path)
    
    # Create and apply HSA adapter
    logger.info("Creating and applying HSA adapter")
    patcher, _ = create_adapter_for_model(
        model=model,
        use_sparse=use_sparse,
        adaptation_enabled=False,  # No adaptation during evaluation
    )
    
    # Replace the registry with the loaded one
    patcher.registry = registry
    
    # Patch model
    patcher.patch_model()
    
    # Store patcher in model for convenience
    model.patcher = patcher
    
    return model, tokenizer, registry


def evaluate_perplexity(model, tokenizer, dataset, batch_size=4, max_length=256, num_samples=100):
    """Evaluate model perplexity on dataset."""
    logger.info("Evaluating perplexity")
    
    # Use a small subset of the dataset for faster evaluation
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    # Tokenize dataset if needed
    if "input_ids" not in dataset.column_names:
        def tokenize_function(examples):
            inputs = examples["text"] if "text" in examples else examples["instruction"]
            return tokenizer(
                inputs,
                max_length=max_length,
                padding="max_length",
                truncation=True
            )
        
        dataset = dataset.map(tokenize_function, batched=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Track total loss and tokens
    total_loss = 0.0
    total_tokens = 0
    
    # Process dataset in batches
    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating perplexity"):
        batch = dataset[i:i+batch_size]
        
        # Convert to PyTorch tensors
        input_ids = torch.tensor(batch["input_ids"]).to(model.device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(model.device)
        
        # Create labels (shift input_ids right)
        labels = input_ids.clone()
        
        # Forward pass with no grad
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        # Get loss
        loss = outputs.loss.item()
        
        # Count tokens (excluding padding)
        num_tokens = attention_mask.sum().item()
        
        # Accumulate
        total_loss += loss * num_tokens
        total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "total_tokens": total_tokens
    }


def evaluate_generation(model, tokenizer, dataset, num_samples=5, max_length=256, gen_length=50):
    """Evaluate text generation quality and speed."""
    logger.info("Evaluating text generation")
    
    # Use a small subset of the dataset for prompts
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    # Track results
    results = []
    total_time = 0.0
    total_tokens = 0
    
    # Generate text for each prompt
    for i, example in enumerate(tqdm(dataset, desc="Generating text")):
        # Get prompt
        prompt = example["text"] if "text" in example else example["instruction"]
        
        # Trim prompt if too long
        if len(prompt) > 200:
            prompt = prompt[:200]
        
        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        
        # Generate text
        start_time = time.time()
        
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=input_ids.shape[1] + gen_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Measure time
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Decode output
            generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Calculate number of tokens generated
            tokens_generated = output_ids.shape[1] - input_ids.shape[1]
            
            # Calculate tokens per second
            tokens_per_second = tokens_generated / generation_time if generation_time > 0 else 0
            
            # Store result
            results.append({
                "prompt": prompt,
                "generated_text": generated_text,
                "tokens_generated": tokens_generated,
                "generation_time": generation_time,
                "tokens_per_second": tokens_per_second
            })
            
            # Update totals
            total_time += generation_time
            total_tokens += tokens_generated
            
        except Exception as e:
            logger.error(f"Error generating text for sample {i}: {e}")
            results.append({
                "prompt": prompt,
                "error": str(e)
            })
    
    # Calculate average generation speed
    avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    return {
        "generation_results": results,
        "avg_tokens_per_second": avg_tokens_per_second,
        "total_tokens_generated": total_tokens,
        "total_generation_time": total_time
    }


def evaluate_attention_pattern(model, tokenizer, dataset, num_samples=1):
    """Analyze the attention pattern distribution in the HSA model."""
    logger.info("Analyzing attention patterns")
    
    # Use a very small subset for this detailed analysis
    if len(dataset) > num_samples:
        dataset = dataset.select(range(num_samples))
    
    results = {}
    
    # Check if the model has a patcher with registry access
    if not hasattr(model, "patcher") or not hasattr(model.patcher, "registry"):
        logger.warning("Model does not have an HSA patcher or registry - skipping attention analysis")
        return {"error": "No HSA registry available"}
    
    registry = model.patcher.registry
    
    # Collect level statistics
    level_stats = {}
    for level in registry.hierarchy.levels:
        splats = list(registry.get_splats_at_level(level))
        
        # Skip empty levels
        if not splats:
            level_stats[level] = {
                "count": 0,
                "avg_activation": 0,
                "max_activation": 0,
                "min_activation": 0
            }
            continue
        
        # Calculate activation statistics
        activations = [splat.get_average_activation() for splat in splats]
        level_stats[level] = {
            "count": len(splats),
            "avg_activation": np.mean(activations),
            "max_activation": np.max(activations),
            "min_activation": np.min(activations),
            "active_splats": sum(1 for act in activations if act > 0.01)
        }
    
    results["level_stats"] = level_stats
    results["total_splats"] = registry.count_splats()
    
    # Process one sample to analyze attention patterns
    if num_samples > 0 and len(dataset) > 0:
        try:
            # Get first example
            example = dataset[0]
            prompt = example["text"] if "text" in example else example["instruction"]
            
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
            
            # Get attention pattern
            with torch.no_grad():
                # Forward pass with output_attentions=True to get attention matrices
                outputs = model(input_ids, output_attentions=True)
            
            # Extract attention matrices
            if outputs.attentions:
                # Average across layers and heads
                attentions = outputs.attentions
                
                # Process first attention matrix as an example
                first_attention = attentions[0][0].cpu().numpy()  # [layer 0, batch 0]
                
                # Calculate statistics
                attn_stats = {
                    "shape": str(first_attention.shape),
                    "mean": float(np.mean(first_attention)),
                    "max": float(np.max(first_attention)),
                    "min": float(np.min(first_attention)),
                    "sparsity": float(np.mean(first_attention < 0.01)),
                }
                
                results["attention_stats"] = attn_stats
        except Exception as e:
            logger.error(f"Error analyzing attention patterns: {e}")
            results["attention_error"] = str(e)
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and registry
    model, tokenizer, registry = load_model_and_registry(
        args.model_path, args.registry_path, args.use_sparse
    )
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    eval_dataset = dataset["validation"] if "validation" in dataset else dataset["test"]
    
    # Evaluate perplexity
    perplexity_results = evaluate_perplexity(
        model, tokenizer, eval_dataset, 
        batch_size=args.batch_size,
        max_length=args.max_seq_length,
        num_samples=args.num_samples
    )
    
    # Evaluate generation
    generation_results = evaluate_generation(
        model, tokenizer, eval_dataset,
        num_samples=min(args.num_samples, 5),  # Use fewer samples for generation
        max_length=args.max_seq_length,
        gen_length=args.generation_length
    )
    
    # Analyze attention patterns
    attention_results = evaluate_attention_pattern(
        model, tokenizer, eval_dataset,
        num_samples=1  # Just use one sample for attention analysis
    )
    
    # Collect all results
    results = {
        "perplexity": perplexity_results,
        "generation": generation_results,
        "attention": attention_results,
        "registry_summary": registry.get_summary(),
    }
    
    # Save results
    import json
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        # Convert numpy values to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # Create serializable results
        serializable_results = json.loads(
            json.dumps(results, default=convert_to_serializable)
        )
        
        json.dump(serializable_results, f, indent=2)
    
    # Also save generation examples separately
    gen_examples_path = os.path.join(args.output_dir, "generation_examples.txt")
    with open(gen_examples_path, "w") as f:
        for i, result in enumerate(generation_results["generation_results"]):
            if "error" in result:
                f.write(f"Example {i+1} - ERROR: {result['error']}\n\n")
                continue
                
            f.write(f"Example {i+1}\n")
            f.write(f"Prompt: {result['prompt']}\n\n")
            f.write(f"Generated Text:\n{result['generated_text']}\n\n")
            f.write(f"Generation Time: {result['generation_time']:.2f}s\n")
            f.write(f"Tokens Generated: {result['tokens_generated']}\n")
            f.write(f"Tokens/second: {result['tokens_per_second']:.2f}\n")
            f.write("-" * 80 + "\n\n")
    
    # Print summary
    logger.info("Evaluation Complete!")
    logger.info(f"Perplexity: {perplexity_results['perplexity']:.2f}")
    logger.info(f"Average Generation Speed: {generation_results['avg_tokens_per_second']:.2f} tokens/second")
    logger.info(f"Total Splats: {attention_results['total_splats']}")
    logger.info(f"Results saved to {args.output_dir}")
    
    # Also evaluate base model if requested
    if args.use_base_model:
        logger.info("Evaluating base model without HSA")
        
        # Restore original model
        model.patcher.restore_model()
        
        # Evaluate perplexity
        base_perplexity_results = evaluate_perplexity(
            model, tokenizer, eval_dataset, 
            batch_size=args.batch_size,
            max_length=args.max_seq_length,
            num_samples=args.num_samples
        )
        
        # Evaluate generation
        base_generation_results = evaluate_generation(
            model, tokenizer, eval_dataset,
            num_samples=min(args.num_samples, 5),
            max_length=args.max_seq_length,
            gen_length=args.generation_length
        )
        
        # Collect base model results
        base_results = {
            "perplexity": base_perplexity_results,
            "generation": base_generation_results,
        }
        
        # Save base model results
        base_results_path = os.path.join(args.output_dir, "base_model_results.json")
        with open(base_results_path, "w") as f:
            serializable_base_results = json.loads(
                json.dumps(base_results, default=convert_to_serializable)
            )
            json.dump(serializable_base_results, f, indent=2)
        
        # Print comparison
        logger.info("Base Model vs HSA Model Comparison:")
        logger.info(f"Base Model Perplexity: {base_perplexity_results['perplexity']:.2f}")
        logger.info(f"HSA Model Perplexity: {perplexity_results['perplexity']:.2f}")
        logger.info(f"Base Model Generation Speed: {base_generation_results['avg_tokens_per_second']:.2f} tokens/second")
        logger.info(f"HSA Model Generation Speed: {generation_results['avg_tokens_per_second']:.2f} tokens/second")


if __name__ == "__main__":
    main()
