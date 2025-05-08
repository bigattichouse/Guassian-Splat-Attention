#!/usr/bin/env python
"""
HSA Training Script for TinyStoriesInstruct dataset.

This script trains a language model enhanced with Hierarchical Splat Attention
using the TinyStoriesInstruct dataset from Huggingface.
"""

import os
import argparse
import logging
import torch
import numpy as np
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    default_data_collator,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset

# Import HSA components
from hsa.registry import SplatRegistry
from hsa.hierarchy import Hierarchy
from hsa.model_adapter import create_adapter_for_model, ModelPatcher
from hsa.adaptation_controller import AdaptationController
from hsa.adaptation_metrics_base import AdaptationMetricsComputer
from hsa.attention_info_metrics import InfoTheoreticMetricsComputer, InfoTheoreticCandidateEvaluator
from hsa.serialization_core import HSASerializer
from hsa.compact_serialization import save_registry_compact

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hsa_training.log")
    ]
)
logger = logging.getLogger(__name__)

# Define training hyperparameters
DEFAULT_CONFIG = {
    "base_model": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",  # Smaller model for faster training
    "dataset_name": "roneneldan/TinyStories",
    "dataset_config": "instruction",
    "output_dir": "output/hsa_trained_model",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 5e-5,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "max_seq_length": 256,
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "save_steps": 1000,
    "save_total_limit": 3,
    "logging_steps": 100,
    "fp16": torch.cuda.is_available(),
    "seed": 42,
}

# HSA configuration
HSA_CONFIG = {
    "use_sparse": True,
    "adaptation_enabled": True,
    "hierarchy_levels": ["token", "word", "phrase", "sentence", "document"],
    "adaptation_frequency": 50,  # Adapt HSA every N steps
    "save_registry_frequency": 500,  # Save HSA registry every N steps
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a model with HSA")
    
    # Model and dataset arguments
    parser.add_argument("--base_model", type=str, default=DEFAULT_CONFIG["base_model"],
                        help="Base model to train with HSA")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_CONFIG["dataset_name"],
                        help="Dataset name on Huggingface or 'local' for local files")
    parser.add_argument("--dataset_config", type=str, default=DEFAULT_CONFIG["dataset_config"],
                        help="Dataset configuration")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Directory with text files for training (when dataset_name is 'local')")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                        help="Output directory for model checkpoints")
    parser.add_argument("--continue_from_checkpoint", type=str, default=None,
                        help="Continue training from a checkpoint directory")
    parser.add_argument("--registry_path", type=str, default=None,
                        help="Load an existing HSA registry instead of initializing a new one")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_train_epochs"],
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["per_device_train_batch_size"],
                        help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"],
                        help="Initial learning rate")
    parser.add_argument("--max_seq_length", type=int, default=DEFAULT_CONFIG["max_seq_length"],
                        help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, 
                        default=DEFAULT_CONFIG["gradient_accumulation_steps"],
                        help="Number of gradient accumulation steps")
    parser.add_argument("--train_val_split", type=float, default=0.9,
                        help="Train/validation split ratio for local files")
    parser.add_argument("--save_every", type=int, default=DEFAULT_CONFIG["save_steps"],
                        help="Save model checkpoint every N steps")
    
    # HSA-specific arguments
    parser.add_argument("--use_sparse", action="store_true", default=HSA_CONFIG["use_sparse"],
                        help="Use sparse attention computation")
    parser.add_argument("--adaptation_enabled", action="store_true", 
                        default=HSA_CONFIG["adaptation_enabled"],
                        help="Enable HSA adaptation during training")
    parser.add_argument("--adaptation_frequency", type=int, 
                        default=HSA_CONFIG["adaptation_frequency"],
                        help="Frequency of HSA adaptation (in training steps)")
    parser.add_argument("--hierarchy_levels", type=str, 
                        default=",".join(HSA_CONFIG["hierarchy_levels"]),
                        help="Comma-separated list of hierarchy levels")
    parser.add_argument("--add_dataset", type=str, default=None,
                        help="Additional dataset to include in training (comma-separated for multiple)")
    parser.add_argument("--add_data_dir", type=str, default=None,
                        help="Additional directory with text files to include in training")
    
    return parser.parse_args()


def load_from_text_files(data_dir, tokenizer, max_seq_length, train_val_split=0.9):
    """Load dataset from text files in a directory.
    
    Args:
        data_dir: Directory containing text files
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        train_val_split: Train/validation split ratio
        
    Returns:
        Dictionary with train and validation datasets
    """
    logger.info(f"Loading text files from {data_dir}")
    
    # List text files
    text_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".txt", ".md", ".json")):
                text_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(text_files)} text files")
    
    # Read text from files
    texts = []
    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Handle JSON files by extracting relevant text fields
                if file_path.endswith(".json"):
                    try:
                        json_data = json.loads(content)
                        # Extract text fields from JSON (adjust as needed)
                        if isinstance(json_data, list):
                            for item in json_data:
                                if isinstance(item, dict):
                                    text = ""
                                    for field in ["text", "content", "instruction", "input", "output"]:
                                        if field in item and isinstance(item[field], str):
                                            text += item[field] + "\n"
                                    if text:
                                        texts.append(text)
                        elif isinstance(json_data, dict):
                            text = ""
                            for field in ["text", "content", "instruction", "input", "output"]:
                                if field in json_data and isinstance(json_data[field], str):
                                    text += json_data[field] + "\n"
                            if text:
                                texts.append(text)
                    except json.JSONDecodeError:
                        # Treat as regular text if not valid JSON
                        texts.append(content)
                else:
                    # Regular text file
                    texts.append(content)
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
    
    # Split into chunks
    chunked_texts = []
    for text in texts:
        # Simple chunking by splitting the text into reasonable segments
        # You may need to adjust this based on your data
        chunks = []
        tokens = tokenizer.encode(text)
        
        # Skip if too short
        if len(tokens) < 10:
            continue
            
        # Create chunks of appropriate size
        for i in range(0, len(tokens), max_seq_length // 2):
            chunk_tokens = tokens[i:i + max_seq_length]
            if len(chunk_tokens) > max_seq_length // 4:  # Only keep chunks of reasonable size
                chunk_text = tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text)
        
        chunked_texts.extend(chunks)
    
    logger.info(f"Created {len(chunked_texts)} text chunks")
    
    # Create dataset
    from datasets import Dataset
    dataset = Dataset.from_dict({"text": chunked_texts})
    
    # Split into train and validation
    from datasets import DatasetDict
    train_val_dataset = dataset.train_test_split(
        test_size=1.0 - train_val_split,
        seed=42
    )
    
    # Rename the test split to validation
    dataset_dict = DatasetDict({
        "train": train_val_dataset["train"],
        "validation": train_val_dataset["test"]
    })
    
    # Tokenize dataset
    return tokenize_dataset(dataset_dict, tokenizer, max_seq_length)


def tokenize_dataset(dataset, tokenizer, max_seq_length):
    """Tokenize a dataset."""
    # Define tokenization function
    def tokenize_function(examples):
        # Handle different column names
        if "text" in examples:
            inputs = examples["text"]
        elif "instruction" in examples:
            inputs = examples["instruction"]
        elif "input" in examples:
            inputs = examples["input"]
        else:
            # Default to first column if text not found
            column_name = list(examples.keys())[0]
            inputs = examples[column_name]
            
        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    # Tokenize datasets
    tokenized_dataset = {}
    for split in dataset.keys():
        tokenized_dataset[split] = dataset[split].map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset[split].column_names 
                            if col not in ["input_ids", "attention_mask", "labels"]],
            desc=f"Tokenizing {split} split"
        )
    
    logger.info(f"Dataset tokenized. Train size: {len(tokenized_dataset['train'])}")
    return tokenized_dataset


def load_tokenized_dataset(dataset_name, dataset_config, tokenizer, max_seq_length, data_dir=None):
    """Load and tokenize the dataset.
    
    Args:
        dataset_name: Huggingface dataset name or "local"
        dataset_config: Dataset configuration or None
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        data_dir: Directory containing text files (only used if dataset_name is "local")
        
    Returns:
        Tokenized dataset dictionary
    """
    # Check if using local files
    if dataset_name.lower() == "local" and data_dir:
        return load_from_text_files(data_dir, tokenizer, max_seq_length)
    
    # Load from Huggingface
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, dataset_config)
    
    return tokenize_dataset(dataset, tokenizer, max_seq_length)


class HSAAdaptationCallback:
    """Custom callback for HSA adaptation during training."""
    
    def __init__(self, registry, adaptation_controller, adaptation_frequency, save_frequency, output_dir):
        """Initialize adaptation callback.
        
        Args:
            registry: SplatRegistry to adapt
            adaptation_controller: AdaptationController instance
            adaptation_frequency: How often to adapt (in steps)
            save_frequency: How often to save registry state (in steps)
            output_dir: Directory to save registry checkpoints
        """
        self.registry = registry
        self.adaptation_controller = adaptation_controller
        self.adaptation_frequency = adaptation_frequency
        self.save_frequency = save_frequency
        self.output_dir = output_dir
        self.steps = 0
        self.last_token_embeddings = None
        self.serializer = HSASerializer()
        
        # Create output directory for registry checkpoints
        os.makedirs(os.path.join(output_dir, "registry_checkpoints"), exist_ok=True)
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each training step."""
        self.steps += 1
        
        # Extract token embeddings from the last batch if model is provided
        if model is not None and hasattr(model, "get_input_embeddings"):
            try:
                # This is a simplified approach - in practice, you'd want to extract
                # actual token embeddings from the current batch
                embed_layer = model.get_input_embeddings()
                sample_tokens = embed_layer.weight[:100].detach().cpu().numpy()
                self.last_token_embeddings = sample_tokens
            except Exception as e:
                logger.warning(f"Failed to extract token embeddings: {e}")
        
        # Perform adaptation if needed
        if self.adaptation_controller and self.steps % self.adaptation_frequency == 0:
            logger.info(f"Performing HSA adaptation at step {self.steps}")
            try:
                results = self.adaptation_controller.step(self.last_token_embeddings)
                logger.info(f"Adaptation results: {len(results)} operations performed")
            except Exception as e:
                logger.error(f"Error during HSA adaptation: {e}")
        
        # Save registry checkpoint if needed
        if self.steps % self.save_frequency == 0:
            self._save_registry_checkpoint()
    
    def _save_registry_checkpoint(self):
        """Save current registry state."""
        try:
            checkpoint_dir = os.path.join(self.output_dir, "registry_checkpoints")
            checkpoint_path = os.path.join(checkpoint_dir, f"registry_step_{self.steps}.bin")
            
            # Save using compact serialization for smaller files
            save_registry_compact(
                self.registry,
                checkpoint_path,
                format="binary",
                include_history=True
            )
            
            logger.info(f"Saved registry checkpoint to {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error saving registry checkpoint: {e}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        # Save final registry state
        try:
            final_path = os.path.join(self.output_dir, "final_registry.bin")
            self.serializer.save_to_file(self.registry, final_path)
            logger.info(f"Saved final registry state to {final_path}")
            
            # Also save in compact format
            compact_path = os.path.join(self.output_dir, "final_registry_compact.bin")
            save_registry_compact(self.registry, compact_path)
            logger.info(f"Saved compact registry to {compact_path}")
        except Exception as e:
            logger.error(f"Error saving final registry: {e}")


class HSATrainer(Trainer):
    """Custom trainer with HSA adaptation support."""
    
    def __init__(self, hsa_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.hsa_callback = hsa_callback
    
    def training_step(self, model, inputs):
        """Override training step to include HSA adaptation."""
        # Regular training step
        loss = super().training_step(model, inputs)
        
        # Call HSA adaptation callback
        if self.hsa_callback:
            self.hsa_callback.on_step_end(
                self.args, self.state, self.control, model=model
            )
        
        return loss
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """Override to include additional HSA metrics in logs."""
        result = super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
        
        # Add HSA metrics to logs if available
        if hasattr(model, "patcher") and hasattr(model.patcher, "get_stats"):
            hsa_stats = model.patcher.get_stats()
            # Add to logs
            if isinstance(result, dict):
                for key, value in hsa_stats.get("attention_stats", {}).items():
                    if isinstance(value, (int, float)):
                        result[f"hsa_{key}"] = value
        
        return result


def load_and_merge_datasets(args, tokenizer):
    """Load and merge multiple datasets."""
    datasets = {}
    
    # Load main dataset
    main_dataset = load_tokenized_dataset(
        args.dataset_name,
        args.dataset_config,
        tokenizer,
        args.max_seq_length,
        args.data_dir
    )
    
    for split in main_dataset:
        datasets[split] = main_dataset[split]
    
    # Load additional datasets if specified
    if args.add_dataset:
        additional_datasets = args.add_dataset.split(",")
        for dataset_name in additional_datasets:
            try:
                logger.info(f"Loading additional dataset: {dataset_name}")
                # Assume no special config for additional datasets
                add_dataset = load_tokenized_dataset(
                    dataset_name.strip(),
                    None,
                    tokenizer,
                    args.max_seq_length
                )
                
                # Merge train splits
                if "train" in add_dataset and "train" in datasets:
                    logger.info(f"Merging train split from {dataset_name}")
                    datasets["train"] = datasets["train"].concatenate(add_dataset["train"])
                
                # Merge validation splits if available
                if "validation" in add_dataset and "validation" in datasets:
                    logger.info(f"Merging validation split from {dataset_name}")
                    datasets["validation"] = datasets["validation"].concatenate(add_dataset["validation"])
            except Exception as e:
                logger.error(f"Error loading additional dataset {dataset_name}: {e}")
    
    # Load additional text files if specified
    if args.add_data_dir:
        try:
            logger.info(f"Loading additional data from directory: {args.add_data_dir}")
            add_text_dataset = load_from_text_files(
                args.add_data_dir,
                tokenizer,
                args.max_seq_length,
                args.train_val_split
            )
            
            # Merge train splits
            if "train" in add_text_dataset and "train" in datasets:
                logger.info(f"Merging train split from {args.add_data_dir}")
                datasets["train"] = datasets["train"].concatenate(add_text_dataset["train"])
            
            # Merge validation splits
            if "validation" in add_text_dataset and "validation" in datasets:
                logger.info(f"Merging validation split from {args.add_data_dir}")
                datasets["validation"] = datasets["validation"].concatenate(add_text_dataset["validation"])
        except Exception as e:
            logger.error(f"Error loading additional data from {args.add_data_dir}: {e}")
    
    logger.info(f"Final dataset sizes - Train: {len(datasets['train'])}, "
                f"Validation: {len(datasets.get('validation', []))}")
    
    return datasets


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(DEFAULT_CONFIG["seed"])
    np.random.seed(DEFAULT_CONFIG["seed"])
    
    # Parse hierarchy levels
    hierarchy_levels = args.hierarchy_levels.split(",")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.base_model}")
    
    if args.continue_from_checkpoint:
        logger.info(f"Continuing training from checkpoint: {args.continue_from_checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(args.continue_from_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(args.continue_from_checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_pretrained(args.base_model)
    
    # Ensure tokenizer has padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create HSA adapter or load existing registry
    if args.registry_path:
        logger.info(f"Loading HSA registry from {args.registry_path}")
        
        # Load existing registry
        serializer = HSASerializer()
        registry = serializer.load_from_file(args.registry_path)
        
        # Create adapter with loaded registry
        logger.info("Creating HSA adapter with loaded registry")
        patcher, _ = create_adapter_for_model(
            model=model,
            use_sparse=args.use_sparse,
            adaptation_enabled=args.adaptation_enabled,
            hierarchy_levels=hierarchy_levels,
        )
        
        # Replace the registry with the loaded one
        patcher.registry = registry
    else:
        # Create new HSA adapter
        logger.info("Creating HSA adapter with new registry")
        patcher, registry = create_adapter_for_model(
            model=model,
            use_sparse=args.use_sparse,
            adaptation_enabled=args.adaptation_enabled,
            hierarchy_levels=hierarchy_levels,
        )
    
    # Store patcher in model for later access
    model.patcher = patcher
    
    # Patch model with HSA
    logger.info("Patching model with HSA")
    patcher.patch_model()
    
    # Create adaptation controller if enabled
    adaptation_controller = None
    if args.adaptation_enabled:
        logger.info("Initializing HSA adaptation controller")
        metrics_computer = InfoTheoreticMetricsComputer()
        candidate_evaluator = InfoTheoreticCandidateEvaluator(metrics_computer)
        
        adaptation_controller = AdaptationController(
            registry=registry,
            metrics_computer=metrics_computer,
            candidate_evaluator=candidate_evaluator
        )
        
        # Store in patcher for later access
        patcher.adaptation_controller = adaptation_controller
    
    # Load and merge datasets
    tokenized_datasets = load_and_merge_datasets(args, tokenizer)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create HSA adaptation callback
    hsa_callback = HSAAdaptationCallback(
        registry=registry,
        adaptation_controller=adaptation_controller,
        adaptation_frequency=args.adaptation_frequency,
        save_frequency=HSA_CONFIG["save_registry_frequency"],
        output_dir=args.output_dir
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=DEFAULT_CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=DEFAULT_CONFIG["weight_decay"],
        warmup_steps=DEFAULT_CONFIG["warmup_steps"],
        evaluation_strategy=DEFAULT_CONFIG["evaluation_strategy"],
        eval_steps=DEFAULT_CONFIG["eval_steps"],
        save_steps=args.save_every,
        save_total_limit=DEFAULT_CONFIG["save_total_limit"],
        logging_steps=DEFAULT_CONFIG["logging_steps"],
        fp16=DEFAULT_CONFIG["fp16"],
        report_to="tensorboard",
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = HSATrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("validation"),
        tokenizer=tokenizer,
        hsa_callback=hsa_callback
    )
    
    # Train the model
    logger.info("Starting training")
    train_result = trainer.train(
        resume_from_checkpoint=args.continue_from_checkpoint
    )
    
    # Save final model and tokenizer
    logger.info("Saving final model")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Call callback's on_train_end
    hsa_callback.on_train_end(trainer.args, trainer.state, trainer.control)
    
    # Log training results
    logger.info(f"Training finished. Results: {train_result}")
    
    # Save training metrics
    with open(os.path.join(args.output_dir, "training_results.txt"), "w") as f:
        f.write(f"Training results: {train_result}\n")
        
        # Log HSA stats
        if hasattr(model, "patcher") and hasattr(model.patcher, "get_stats"):
            f.write(f"HSA stats: {model.patcher.get_stats()}\n")
    
    logger.info(f"Model and results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
