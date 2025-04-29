#!/usr/bin/env python
"""
Training script for Hierarchical Splat Attention (HSA) models.

This script trains a transformer model with HSA from scratch using the TinyStories dataset.
The model is trained for language modeling, and the script handles:
- Dataset loading and preprocessing
- Model creation with HSA
- Training loop with adaptation
- Evaluation
- Checkpointing and saving
- Automatic recovery from crashes
- Multiple HSA initialization strategies

Usage:
    python train.py [options]

Options:
    --model_name: Name of the model to save (default: hsa-tinystories)
    --data_dir: Directory to store the dataset (default: data)
    --output_dir: Directory to save outputs (default: outputs)
    --epochs: Number of epochs to train (default: 3)
    --batch_size: Batch size for training (default: 32)
    --learning_rate: Learning rate (default: 5e-5)
    --seq_length: Sequence length for training (default: 128)
    --vocab_size: Vocabulary size (default: 10000)
    --hidden_size: Hidden size of the model (default: 128)
    --num_layers: Number of layers in the model (default: 4)
    --num_heads: Number of attention heads (default: 4)
    --resume: Resume training from latest checkpoint
    --auto-resume: Automatically resume if a checkpoint exists
    --checkpoint: Specific checkpoint file to resume from
    --hsa_init_method: Method to initialize HSA splats (tokenizer, batch, random)
    --hsa_token_aware: Use token-aware initialization (using tokenizer structure)
"""

import os
import json
import time
import argparse
import signal
import traceback
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import threading
import queue
import time

# Import HSA modules
import hsa
from hsa.data_structures import Hierarchy, Splat, SplatRegistry
from hsa.core import HSA, create_hsa
from hsa.training import TrainingConfig
from hsa.model_integration import create_hsa_transformer
from hsa.visualization import HSAVisualizer
from hsa.initialization import initialize_splats, initialize_from_tokenizer

# Prevent TK threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import os
os.environ['TK_SCREEN'] = 'off'  # Turn off Tk screen updating

# Set up argument parser with ALL arguments defined upfront
parser = argparse.ArgumentParser(description="Train an HSA model on TinyStories")
parser.add_argument("--model_name", type=str, default="hsa-tinystories", help="Name of the model to save")
parser.add_argument("--data_dir", type=str, default="data", help="Directory to store the dataset")
parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for training")
parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of the model")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the model")
parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
parser.add_argument("--save_interval", type=int, default=1000, help="Save checkpoint every N steps")
parser.add_argument("--eval_interval", type=int, default=500, help="Evaluate every N steps")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                    help="Device to train on")
parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
parser.add_argument("--auto-resume", action="store_true", help="Automatically resume if a checkpoint exists")
parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint file to resume from")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
# New arguments for HSA initialization
parser.add_argument("--hsa_init_method", type=str, default="batch",
                    choices=["tokenizer", "batch", "random"],
                    help="Method to initialize HSA splats")
parser.add_argument("--hsa_token_aware", action="store_true",
                    help="Use token-aware initialization (using tokenizer structure)")
parser.add_argument("--hsa_init_seed", type=int, default=42,
                    help="Random seed for HSA initialization")

# Now parse the arguments
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(os.path.join(args.output_dir, args.model_name), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, args.model_name, "checkpoints"), exist_ok=True)
os.makedirs(os.path.join(args.output_dir, args.model_name, "visualizations"), exist_ok=True)

# Configure a debug log
debug_log_path = os.path.join(args.output_dir, args.model_name, "debug_log.txt")

def log_debug(message):
    """Log debug information."""
    if args.debug:
        print(f"DEBUG: {message}")
        with open(debug_log_path, "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")

# Save configuration
with open(os.path.join(args.output_dir, args.model_name, "config.json"), "w") as f:
    json.dump(vars(args), f, indent=4)

# Configure HSA
hsa_config = {
    "hierarchy": {
        "levels": ["Token", "Phrase", "Section"],
        "init_splats_per_level": [200, 100, 100],  # Increased initial counts significantly
        "level_weights": [0.5, 0.3, 0.2]
    },
    "attention": {
        "sparse_topk": 32
    },
    "adaptation": {
        "mitosis_threshold": 0.05,       # Increased to encourage more splits
        "death_threshold": 0.0001,       # Significantly reduced as recommended
        "adaptation_frequency": 25,      # More frequent adaptations (every 25 steps)
        "enable_adaptation": True,
        "min_level_percentage": 0.7,     # Ensure at least 70% of initial splats remain
        "max_death_percentage": 0.01,    # Limit deaths to 2% per adaptation
        "birth_level_threshold": 0.8     # Trigger birth when below 80% of initial count
    },
    "training": {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate
    },
    "initialization": {
        "method": args.hsa_init_method,
        "token_aware": args.hsa_token_aware,
        "random_seed": args.hsa_init_seed
    }
}

print(f"Training configuration: {json.dumps(hsa_config, indent=2)}")
print(f"Running on device: {args.device}")
print(f"HSA initialization method: {args.hsa_init_method}" + 
      (", with token-aware mode" if args.hsa_token_aware else ""))

# Create HSA instance
hsa_system = create_hsa(hsa_config)
log_debug(f"Created HSA instance: {type(hsa_system)}")
log_debug(f"HSA attributes: {dir(hsa_system)}")

# Create tokenizer (using standard pretrained tokenizer for simplicity)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

class TextDataset(Dataset):
    """Dataset for language modeling with TinyStories."""
    
    def __init__(self, dataset, tokenizer, seq_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get text
        text = self.dataset[idx]["text"]
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True, truncation=True, 
                                       max_length=self.seq_length + 1)
        
        # Create inputs and targets for language modeling
        if len(tokens) < 2:
            tokens = tokens + [self.tokenizer.eos_token_id] * (2 - len(tokens))
        
        inputs = tokens[:-1]
        targets = tokens[1:]
        
        # Pad to seq_length
        if len(inputs) < self.seq_length:
            pad_length = self.seq_length - len(inputs)
            inputs = inputs + [self.tokenizer.pad_token_id] * pad_length
            targets = targets + [self.tokenizer.pad_token_id] * pad_length
        
        # Convert to tensors
        input_ids = torch.tensor(inputs, dtype=torch.long)
        target_ids = torch.tensor(targets, dtype=torch.long)
        
        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids
        }

# Create checkpoint recovery log
recovery_log_path = os.path.join(args.output_dir, args.model_name, "recovery_log.txt")

def log_recovery(message):
    """Log recovery information."""
    with open(recovery_log_path, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")

def inspect_hsa_state(hsa_system):
    """Debug function to inspect HSA state in detail."""
    try:
        log_debug(f"HSA system type: {type(hsa_system)}")
        log_debug(f"HSA system attributes: {dir(hsa_system)}")
        
        if hasattr(hsa_system, 'is_initialized'):
            log_debug(f"HSA initialized: {hsa_system.is_initialized}")
        else:
            log_debug("HSA missing is_initialized attribute")
        
        if hasattr(hsa_system, 'hierarchy'):
            log_debug(f"HSA hierarchy: {hsa_system.hierarchy}")
        else:
            log_debug("HSA missing hierarchy attribute")
        
        if hasattr(hsa_system, 'splat_registry'):
            log_debug(f"HSA splat_registry type: {type(hsa_system.splat_registry)}")
            
            if hasattr(hsa_system.splat_registry, 'splats'):
                log_debug(f"Number of splats: {len(hsa_system.splat_registry.splats)}")
                
                # Log a few splat details for debugging
                splat_sample = list(hsa_system.splat_registry.splats.items())[:2]
                for splat_id, splat in splat_sample:
                    log_debug(f"Sample splat {splat_id}: {splat}")
            else:
                log_debug("splat_registry has no splats attribute")
        else:
            log_debug("HSA missing splat_registry attribute")
    except Exception as e:
        log_debug(f"Error inspecting HSA state: {e}")

def restore_hsa_from_checkpoint(hsa_system, hsa_data):
    """Helper function to restore HSA state from checkpoint data."""
    try:
        log_debug("Restoring HSA from checkpoint...")
        log_debug(f"HSA data keys: {hsa_data.keys()}")
        
        # Verify hierarchy matches
        if hsa_data["hierarchy"]["levels"] != hsa_system.hierarchy.levels:
            print("Warning: Saved HSA hierarchy doesn't match current configuration")
            log_recovery("Warning: Saved HSA hierarchy doesn't match current configuration")
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
            
            # Create simplified covariance
            covariance = np.diag(splat_info["cov_diag"])
            
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
        
        # Inspect the resulting state
        log_debug(f"Restored HSA state with {len(registry.splats)} splats")
        inspect_hsa_state(hsa_system)
        
        print(f"Restored HSA state with {len(registry.splats)} splats")
        return True
    except Exception as e:
        log_debug(f"Error in restore_hsa_from_checkpoint: {e}")
        traceback.print_exc()
        return False

# Function to initialize HSA based on initialization method
def initialize_hsa(hsa_system, model, tokenizer, tokens_batch=None):
    """
    Initialize HSA based on the selected initialization method.
    
    Args:
        hsa_system: The HSA system to initialize
        model: The language model
        tokenizer: The tokenizer
        tokens_batch: Optional batch of tokens for batch-based initialization
    
    Returns:
        Whether initialization was successful
    """
    log_debug(f"Initializing HSA with method: {args.hsa_init_method}")
    
    try:
        # Common initialization parameters
        init_params = {
            "random_seed": args.hsa_init_seed
        }
        
        if args.hsa_init_method == "tokenizer":
            # Get the embedding matrix from the model
            embedding_weights = model.transformer.word_embeddings.weight.detach().cpu().numpy()
            log_debug(f"Embedding matrix shape: {embedding_weights.shape}")
            
            if args.hsa_token_aware:
                # Create a simplified implementation of tokenizer-aware initialization
                print("Using token-aware initialization directly")
                
                # Get a sample of special tokens and common tokens for better initialization
                special_tokens = tokenizer.all_special_tokens
                special_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in special_tokens]
                
                # Include the most frequent tokens (usually lower IDs in many tokenizers)
                common_token_ids = list(range(min(500, len(tokenizer))))
                
                # Get embeddings for these tokens
                special_embeddings = embedding_weights[special_token_ids]
                common_embeddings = embedding_weights[common_token_ids]
                
                # Combine them
                combined_embeddings = np.vstack([special_embeddings, common_embeddings])
                
                # Remove duplicates
                combined_embeddings = np.unique(combined_embeddings, axis=0)
                
                # Initialize HSA with this mixed set of embeddings
                log_debug(f"Initializing with token-aware embeddings: {combined_embeddings.shape}")
                hsa_system.initialize(combined_embeddings)
            else:
                # Use standard initialization with the full embedding matrix
                print("Using standard initialization with tokenizer embedding matrix")
                hsa_system.initialize(embedding_weights)
                
        elif args.hsa_init_method == "batch":
            # Initialize using a batch of token embeddings
            if tokens_batch is not None:
                print("Using batch-based initialization")
                # Get embeddings from the model
                with torch.no_grad():
                    input_ids = tokens_batch["input_ids"].to(args.device)  # Use all examples
                    embeddings = model.transformer.word_embeddings(input_ids)
                    embeddings = embeddings.detach().cpu().numpy()
                    
                    # Flatten embeddings
                    flattened_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
                    log_debug(f"Initializing with {flattened_embeddings.shape} shaped embeddings")
                    
                    # Initialize HSA
                    hsa_system.initialize(flattened_embeddings)
            else:
                # No batch provided, use random initialization instead
                print("No batch provided for batch initialization, falling back to random")
                log_debug("Falling back to random initialization")
                
                # Get embedding dimension from model
                embed_dim = model.transformer.word_embeddings.embedding_dim
                random_embeddings = np.random.randn(200, embed_dim)  # Generate more random embeddings
                hsa_system.initialize(random_embeddings)
                
        elif args.hsa_init_method == "random":
            # Random initialization
            print("Using random initialization")
            # Get embedding dimension from model
            embed_dim = model.transformer.word_embeddings.embedding_dim
            
            # Generate random embeddings (more than we need to ensure good coverage)
            np.random.seed(args.hsa_init_seed)
            random_embeddings = np.random.randn(300, embed_dim)  # Increased to 300
            
            # Initialize HSA with random embeddings
            hsa_system.initialize(random_embeddings)
        
        log_debug(f"HSA initialized with {len(hsa_system.splat_registry.splats)} splats")
        inspect_hsa_state(hsa_system)
        
        print(f"HSA initialized with {len(hsa_system.splat_registry.splats)} splats")
        return True
        
    except Exception as e:
        log_debug(f"Error initializing HSA: {e}")
        log_debug(traceback.format_exc())
        print(f"Error initializing HSA: {e}")
        return False
# Function to find the latest checkpoint
def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint in the given directory."""
    checkpoints = []
    
    # Look for step checkpoints that have been successfully saved
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("step_") and filename.endswith(".pt"):
            check_path = os.path.join(checkpoint_dir, filename)
            success_marker = f"{check_path}.success"
            
            # Only consider checkpoints that have a success marker
            if os.path.exists(success_marker):
                try:
                    step = int(filename.split("_")[1].split(".")[0])
                    # Also verify the checkpoint can be loaded
                    checkpoint_data = torch.load(check_path, map_location="cpu")
                    if "model" in checkpoint_data and "step" in checkpoint_data:
                        checkpoints.append(("step", step, check_path))
                except Exception as e:
                    print(f"Warning: Checkpoint {filename} appears corrupted: {e}")
                    log_recovery(f"Warning: Checkpoint {filename} appears corrupted: {e}")
    
    # Look for epoch checkpoints
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("epoch_") and filename.endswith(".pt"):
            check_path = os.path.join(checkpoint_dir, filename)
            success_marker = f"{check_path}.success"
            
            # Only consider checkpoints that have a success marker
            if os.path.exists(success_marker):
                try:
                    epoch = int(filename.split("_")[1].split(".")[0])
                    # Also verify the checkpoint can be loaded
                    checkpoint_data = torch.load(check_path, map_location="cpu")
                    if "model" in checkpoint_data and "epoch" in checkpoint_data:
                        checkpoints.append(("epoch", epoch, check_path))
                except Exception as e:
                    print(f"Warning: Checkpoint {filename} appears corrupted: {e}")
                    log_recovery(f"Warning: Checkpoint {filename} appears corrupted: {e}")
    
    # Look for best model checkpoint
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    best_success_marker = f"{best_model_path}.success"
    if os.path.exists(best_model_path) and os.path.exists(best_success_marker):
        try:
            # Verify the checkpoint can be loaded
            checkpoint_data = torch.load(best_model_path, map_location="cpu")
            if "model" in checkpoint_data:
                checkpoints.append(("best", 0, best_model_path))
        except Exception as e:
            print(f"Warning: Best model checkpoint appears corrupted: {e}")
            log_recovery(f"Warning: Best model checkpoint appears corrupted: {e}")
    
    # Sort by type and step/epoch (prioritize step checkpoints as they're more frequent)
    checkpoints.sort(key=lambda x: (0 if x[0] == "step" else 1, -x[1]))
    
    if checkpoints:
        return checkpoints[0][2]  # Return the path to the latest checkpoint
    return None

# Function to save checkpoint
def save_checkpoint(model, optimizer, scheduler, step, epoch, loss, path, hsa_system=None):
    """Save a training checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "loss": loss
    }
    
    # Save HSA state if provided
    if hsa_system is not None:
        # Debug info - inspect before saving
        log_debug(f"Saving HSA state to checkpoint {path}")
        inspect_hsa_state(hsa_system)
        
        if hasattr(hsa_system, 'is_initialized') and hsa_system.is_initialized:
            if hasattr(hsa_system, 'splat_registry') and hsa_system.splat_registry is not None:
                if hasattr(hsa_system.splat_registry, 'splats'):
                    # Create a serializable version of the splat registry
                    splat_data = []
                    for splat_id, splat in hsa_system.splat_registry.splats.items():
                        splat_info = {
                            "id": splat.id,
                            "level": splat.level,
                            "position": splat.position.tolist(),
                            "amplitude": float(splat.amplitude),
                            # Simplified covariance representation to save space
                            "cov_diag": np.diag(splat.covariance).tolist()
                        }
                        splat_data.append(splat_info)
                    
                    # Add HSA information to checkpoint
                    checkpoint["hsa_data"] = {
                        "splats": splat_data,
                        "hierarchy": {
                            "levels": hsa_system.hierarchy.levels,
                            "level_weights": hsa_system.hierarchy.level_weights,
                            "init_splats_per_level": hsa_system.hierarchy.init_splats_per_level
                        }
                    }
                    log_debug(f"Added HSA data with {len(splat_data)} splats to checkpoint")
                else:
                    log_debug("WARNING: splat_registry has no splats attribute, HSA state not saved")
            else:
                log_debug("WARNING: HSA has no splat_registry attribute, HSA state not saved")
        else:
            log_debug("WARNING: HSA not initialized, HSA state not saved")
    
    # Save the checkpoint
    try:
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
        # Create a small marker file to indicate successful saving
        with open(f"{path}.success", "w") as f:
            f.write("1")
    except Exception as e:
        print(f"Error saving checkpoint to {path}: {e}")
        log_recovery(f"Error saving checkpoint to {path}: {e}")

# Function for evaluating the model
def evaluate(model, data_loader, device, max_batches=None):
    """Evaluate the model on the provided data loader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if max_batches is not None and i >= max_batches:
                break
                
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Reshape logits and targets for loss calculation
            logits = logits.view(-1, logits.size(-1))
            targets = target_ids.view(-1)
            
            # Calculate loss (ignoring padding)
            loss = F.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
            
            total_loss += loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    return total_loss / total_samples

# Function to generate sample text
def generate_sample(model, tokenizer, prompt, max_length=100, temperature=0.8):
    """Generate text from the model."""
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(args.device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model output
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, 1)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop if end of sequence
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the generated sequence
    output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return output

# Load TinyStories dataset
print("Loading TinyStoriesInstruct dataset...")
dataset = load_dataset("roneneldan/TinyStoriesInstruct", split="train")

# Split into train and validation
train_size = int(0.9 * len(dataset))
train_dataset = dataset.select(range(train_size))
val_dataset = dataset.select(range(train_size, len(dataset)))

# Create datasets and data loaders
train_dataset = TextDataset(train_dataset, tokenizer, args.seq_length)
val_dataset = TextDataset(val_dataset, tokenizer, args.seq_length)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

print(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)} examples")

# Create HSA language model
print("Creating model with HSA attention...")
model = create_hsa_transformer(
    vocab_size=len(tokenizer),  # Use tokenizer's vocab size
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_layers,
    num_attention_heads=args.num_heads,
    intermediate_size=args.hidden_size * 4,
    hsa_config=hsa_config,
    max_position_embeddings=args.seq_length,
    pad_token_id=tokenizer.pad_token_id
)

# Add language modeling head
class HSALanguageModel(nn.Module):
    """Language model with HSA attention."""
    
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
model = HSALanguageModel(model, len(tokenizer))
model.to(args.device)

# Print model summary
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# Training setup
optimizer = AdamW(model.parameters(), lr=args.learning_rate)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=100, 
    num_training_steps=len(train_loader) * args.epochs
)

# Initialize HSA visualizer
visualizer = HSAVisualizer(output_dir=os.path.join(args.output_dir, args.model_name, "visualizations"))

# Set up signal handlers for graceful shutdown
def signal_handler(sig, frame):
    print("\nReceived interrupt signal. Saving checkpoint before exiting...")
    # Save emergency checkpoint
    emergency_path = os.path.join(args.output_dir, args.model_name, "checkpoints", "emergency_checkpoint.pt")
    try:
        # Inspect HSA state before saving
        log_debug("Inspecting HSA state before emergency save:")
        inspect_hsa_state(hsa_system)
        
        save_checkpoint(
            model, optimizer, scheduler, global_step, epoch,
            loss.item() if 'loss' in locals() else float('inf'), 
            emergency_path,
            hsa_system
        )
        print(f"Emergency checkpoint saved to {emergency_path}")
        log_recovery(f"Emergency checkpoint saved to {emergency_path}")
    except Exception as e:
        print(f"Error saving emergency checkpoint: {e}")
        log_recovery(f"Error saving emergency checkpoint: {e}")
        traceback.print_exc()
    
    print("Exiting...")
    sys.exit(0)

# Register signal handlers for common interrupt signals
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
if hasattr(signal, 'SIGTERM'):  # SIGTERM might not be available on all platforms
    signal.signal(signal.SIGTERM, signal_handler)  # termination request

# Training loop
print("Starting training...")
train_losses = []
val_losses = []
best_val_loss = float("inf")
global_step = 0
start_time = time.time()

# Auto-resume logic
checkpoint_dir = os.path.join(args.output_dir, args.model_name, "checkpoints")
should_resume = False

# First check if a specific checkpoint is requested
if args.checkpoint is not None and os.path.exists(args.checkpoint):
    latest_checkpoint = args.checkpoint
    should_resume = True
    print(f"Will resume from specified checkpoint: {latest_checkpoint}")
elif args.resume:
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    should_resume = latest_checkpoint is not None
    if should_resume:
        print(f"Will resume from latest checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found. Starting training from scratch.")
elif args.auto_resume:
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    should_resume = latest_checkpoint is not None
    if should_resume:
        print(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found for auto-resume. Starting training from scratch.")

# Try/except around the entire training process
try:
    # Resume from checkpoint if requested and available
    if should_resume:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        log_recovery(f"Resuming training from checkpoint: {latest_checkpoint}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(latest_checkpoint, map_location=args.device)
            
            # Load model state
            model.load_state_dict(checkpoint["model"])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint["optimizer"])
            
            # Load scheduler state
            scheduler.load_state_dict(checkpoint["scheduler"])
            
            # Restore training state
            global_step = checkpoint["step"]
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint.get("loss", float("inf"))
            
            # Restore HSA state if available
            if "hsa_data" in checkpoint and checkpoint["hsa_data"]:
                log_debug(f"Found HSA data in checkpoint: {list(checkpoint['hsa_data'].keys())}")
                restore_success = restore_hsa_from_checkpoint(hsa_system, checkpoint["hsa_data"])
                if not restore_success:
                    print("Warning: Failed to restore HSA state from checkpoint")
            else:
                log_debug("No HSA data found in checkpoint")
            
            print(f"Restored training state - Step: {global_step}, Epoch: {start_epoch}, Loss: {best_val_loss:.4f}")
            log_recovery(f"Restored training state - Step: {global_step}, Epoch: {start_epoch}, Loss: {best_val_loss:.4f}")
            
            # Force first evaluation to verify model works correctly
            print("Running evaluation to verify model restored correctly...")
            try:
                test_loss = evaluate(model, val_loader, args.device, max_batches=5)
                print(f"Model verification - Test Loss: {test_loss:.4f}")
                log_recovery(f"Model verification - Test Loss: {test_loss:.4f}")
            except Exception as e:
                print(f"Warning: Model verification failed: {e}")
                print("Starting from scratch instead")
                log_recovery(f"Warning: Model verification failed: {e}, starting from scratch")
                model = create_hsa_transformer(
                    vocab_size=len(tokenizer), 
                    hidden_size=args.hidden_size,
                    num_hidden_layers=args.num_layers,
                    num_attention_heads=args.num_heads,
                    intermediate_size=args.hidden_size * 4,
                    hsa_config=hsa_config,
                    max_position_embeddings=args.seq_length,
                    pad_token_id=tokenizer.pad_token_id
                )
                model = HSALanguageModel(model, len(tokenizer))
                model.to(args.device)
                optimizer = AdamW(model.parameters(), lr=args.learning_rate)
                scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=100, 
                    num_training_steps=len(train_loader) * args.epochs
                )
                global_step = 0
                start_epoch = 0
                best_val_loss = float("inf")
        
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            log_recovery(f"Error loading checkpoint: {e}")
            print("Starting training from scratch.")
            log_recovery("Starting training from scratch.")
            start_epoch = 0
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        
        # Initialize splats with a batch of tokens if this is the first epoch and not resuming
        # or if resuming but there are no initialized splats
        need_init = (
            (epoch == 0 and global_step == 0) or 
            (should_resume and not hasattr(hsa_system, 'is_initialized')) or
            (hasattr(hsa_system, 'is_initialized') and not hsa_system.is_initialized)
        )
        
        if need_init:
            log_debug("HSA needs initialization")
            # Get a batch for batch-based initialization if needed
            init_batch = next(iter(train_loader))
            
            # Initialize HSA with selected method
            init_success = initialize_hsa(hsa_system, model, tokenizer, init_batch)
            
            if init_success:
                # Try to create a visualization if initialization succeeded
                try:
                    # Get data for visualization
                    if args.hsa_init_method == "tokenizer":
                        vis_data = model.transformer.word_embeddings.weight.detach().cpu().numpy()
                    else:
                        # Just use a batch of token embeddings for visualization
                        vis_input_ids = init_batch["input_ids"][:4].to(args.device)
                        vis_embeddings = model.transformer.word_embeddings(vis_input_ids).detach().cpu().numpy()
                        vis_data = vis_embeddings.reshape(-1, vis_embeddings.shape[-1])
                    
                    # Create visualization
                    viz_path = visualizer.visualize_splat_distribution(
                        splat_registry=hsa_system.splat_registry,
                        tokens=vis_data,
                        title=f"Initial HSA Splat Distribution ({args.hsa_init_method})",
                        show=False,
                        save=True
                    )
                    print(f"Created initial splat distribution visualization: {viz_path}")
                except Exception as viz_error:
                    print(f"Warning: Could not create initial visualization: {viz_error}")
                    log_debug(f"Visualization error: {viz_error}")
            else:
                print("HSA initialization failed! Training will continue but HSA may not function correctly.")
                log_recovery("HSA initialization failed! Training will continue but HSA may not function correctly.")
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            target_ids = batch["target_ids"].to(args.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Reshape logits and targets for loss calculation
            logits = logits.view(-1, logits.size(-1))
            targets = target_ids.view(-1)
            
            # Calculate loss (ignoring padding)
            loss = F.cross_entropy(logits, targets, ignore_index=tokenizer.pad_token_id)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            
            # Update statistics
            epoch_loss += loss.item() * input_ids.size(0)
            epoch_samples += input_ids.size(0)
            train_losses.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": epoch_loss / epoch_samples,
                "lr": scheduler.get_last_lr()[0]
            })
            
            # Check for HSA adaptation
            if global_step % hsa_config["adaptation"]["adaptation_frequency"] == 0 and global_step > 0:
                print(f"\nPerforming HSA adaptation at step {global_step}...")
                log_debug(f"Starting adaptation at step {global_step}")
                
                # Verify HSA state before adaptation
                inspect_hsa_state(hsa_system)
                
                # Only perform adaptation if it doesn't take too long
                adaptation_timeout = 60  # Maximum seconds to spend on adaptation
                adaptation_start_time = time.time()
                
                # Safety checkpoint ONLY if HSA is in a valid state to avoid saving corrupt state
                if (hasattr(hsa_system, 'is_initialized') and hsa_system.is_initialized and
                    hasattr(hsa_system, 'splat_registry') and hasattr(hsa_system.splat_registry, 'splats')):
                    
                    safety_checkpoint_path = os.path.join(
                        args.output_dir, 
                        args.model_name, 
                        "checkpoints", 
                        f"pre_adaptation_step_{global_step}.pt"
                    )
                    try:
                        save_checkpoint(
                            model, optimizer, scheduler, global_step, epoch,
                            loss.item(), safety_checkpoint_path, hsa_system
                        )
                        print(f"Safety checkpoint saved before HSA adaptation at {safety_checkpoint_path}")
                        log_recovery(f"Safety checkpoint saved before HSA adaptation at step {global_step}")
                    except Exception as e:
                        print(f"Warning: Failed to save safety checkpoint before HSA adaptation: {e}")
                        log_recovery(f"Warning: Failed to save safety checkpoint before HSA adaptation: {e}")
                        # Continue anyway, we'll be careful with adaptation
                
                # Get a batch of embeddings for adaptation - use smaller subset for CPU
                try:
                    with torch.no_grad():
                        # Use a smaller number of tokens for adaptation to speed up
                        max_samples = min(64, input_ids.size(0))  # Limit sample count
                        adaptation_input_ids = input_ids[:max_samples]
                        
                        embeddings = model.transformer.word_embeddings(adaptation_input_ids)
                        embeddings = embeddings.detach().cpu().numpy()
                        
                        # Adapt HSA with these token embeddings - limit flattened size
                        flattened_embeddings = embeddings.reshape(-1, embeddings.shape[-1])
                        
                        # Further subsample if too many tokens
                        if len(flattened_embeddings) > 256:
                            subsample_indices = np.random.choice(
                                len(flattened_embeddings), 
                                size=256, 
                                replace=False
                            )
                            flattened_embeddings = flattened_embeddings[subsample_indices]
                        
                        log_debug(f"Running adaptation with {len(flattened_embeddings)} token embeddings, shape: {flattened_embeddings.shape}")
                        print(f"Running adaptation with {len(flattened_embeddings)} token embeddings")
                        
                        # Run adaptation with timeout
                        def adaptation_thread(q, hsa, embeddings):
                            try:
                                # Debug information
                                log_debug(f"Adaptation thread starting with {len(embeddings)} embeddings")
                                log_debug(f"Current splat count: {len(hsa.splat_registry.splats) if hasattr(hsa, 'splat_registry') and hasattr(hsa.splat_registry, 'splats') else 'unknown'}")
                                
                                # Explicitly force birth logic using custom parameters
                                # Send these parameters to ensure births happen
                                custom_args = {
                                    "birth_level_threshold": 0.7,  # Birth when below 70% of initial
                                    "min_distance_threshold": 1.5, # Lower distance threshold for births 
                                    "max_regions": 5,              # Allow more regions for birth
                                    "cpu_efficient": False          # Optimize for CPU
                                }
                                
                                # Try to call adapt with custom args
                                try:
                                    result = hsa.adapt(embeddings, cpu_efficient=False, **custom_args)
                                except TypeError:
                                    # If it doesn't accept custom args, use standard call
                                    result = hsa.adapt(embeddings, cpu_efficient=False)
                                    
                                # More debug info
                                log_debug(f"After adaptation, splat count: {len(hsa.splat_registry.splats) if hasattr(hsa, 'splat_registry') and hasattr(hsa.splat_registry, 'splats') else 'unknown'}")
                                log_debug(f"Adaptation result: {result}")
                                
                                q.put(result)
                            except Exception as e:
                                log_debug(f"Exception in adaptation thread: {e}")
                                log_debug(traceback.format_exc())
                                q.put(e)
                        
                        result_queue = queue.Queue()
                        thread = threading.Thread(
                            target=adaptation_thread,
                            args=(result_queue, hsa_system, flattened_embeddings)
                        )
                        thread.daemon = True
                        thread.start()
                        
                        # Wait for thread with timeout
                        thread.join(adaptation_timeout)
                        
                        adaptations = None
                        
                        if thread.is_alive():
                            print(f"Adaptation timed out after {adaptation_timeout} seconds!")
                            log_recovery(f"Adaptation timed out after {adaptation_timeout} seconds!")
                            log_debug(f"Adaptation thread timed out after {adaptation_timeout} seconds!")
                            # Kill the thread (in Python, we can't actually kill threads,
                            # but we can abandon it and continue)
                            
                            # Skip the rest of adaptation - we'll use the safety checkpoint if needed
                            raise TimeoutError(f"Adaptation took too long (>{adaptation_timeout}s)")
                        
                        if not result_queue.empty():
                            result = result_queue.get()
                            if isinstance(result, Exception):
                                log_debug(f"Got exception from adaptation thread: {result}")
                                raise result
                            adaptations = result
                            log_debug(f"Got adaptation result from thread: {adaptations}")
                        else:
                            log_debug("Adaptation thread finished but queue is empty!")
                        
                        # Log adaptation information
                        if adaptations:
                            print(f"Adaptations performed: {adaptations}")
                            adaptation_time = time.time() - adaptation_start_time
                            print(f"Adaptation completed in {adaptation_time:.2f} seconds")
                            
                            # Create visualization occasionally
                            if global_step % (hsa_config["adaptation"]["adaptation_frequency"] * 5) == 0:
                                try:
                                    # Visualize splat distribution if we have time
                                    if time.time() - adaptation_start_time < adaptation_timeout * 0.8:
                                        viz_path = visualizer.visualize_splat_distribution(
                                            splat_registry=hsa_system.splat_registry,
                                            title=f"HSA Hierarchy at Step {global_step}",
                                            show=False,
                                            save=True
                                        )
                                        print(f"Created splat distribution visualization")
                                except Exception as viz_error:
                                    print(f"Warning: Could not create visualization: {viz_error}")
                                    log_debug(f"Visualization error: {viz_error}")
                        else:
                            print("No adaptations were performed")
                            log_debug("No adaptations were performed - result was empty or None")
                except TimeoutError as timeout_error:
                    print(f"Adaptation timed out: {timeout_error}")
                    log_recovery(f"Adaptation timed out: {timeout_error}")
                    log_debug(f"Adaptation timed out: {timeout_error}")
                    print("Training will continue - checking HSA state...")
                    
                    # Verify HSA state instead of automatically restoring
                    try:
                        inspect_hsa_state(hsa_system)
                        if not (hasattr(hsa_system, 'is_initialized') and 
                                hsa_system.is_initialized and 
                                hasattr(hsa_system, 'splat_registry') and 
                                hasattr(hsa_system.splat_registry, 'splats')):
                            print("HSA state is invalid after timeout - attempting to restore from safety checkpoint")
                            # Try to restore from the safety checkpoint
                            safety_checkpoint_path = os.path.join(
                                args.output_dir, 
                                args.model_name, 
                                "checkpoints", 
                                f"pre_adaptation_step_{global_step}.pt"
                            )
                            if os.path.exists(safety_checkpoint_path) and os.path.exists(f"{safety_checkpoint_path}.success"):
                                print(f"Restoring from safety checkpoint: {safety_checkpoint_path}")
                                checkpoint = torch.load(safety_checkpoint_path, map_location=args.device)
                                
                                # Only restore HSA state, not model, optimizer, etc.
                                if "hsa_data" in checkpoint and checkpoint["hsa_data"]:
                                    restore_hsa_from_checkpoint(hsa_system, checkpoint["hsa_data"])
                                    print(f"Successfully restored HSA state from safety checkpoint")
                                    log_recovery(f"Successfully restored HSA state from safety checkpoint")
                    except Exception as inspect_error:
                        print(f"Error inspecting HSA state: {inspect_error}")
                        log_debug(f"Error inspecting HSA state: {inspect_error}")
                except Exception as adaptation_error:
                    print(f"Error during HSA adaptation at step {global_step}: {adaptation_error}")
                    log_recovery(f"Error during HSA adaptation at step {global_step}: {adaptation_error}")
                    log_debug(f"Error during HSA adaptation: {adaptation_error}")
                    log_debug(traceback.format_exc())
                    print("Training will continue - checking HSA state...")
                    
                    # Verify HSA state instead of automatically restoring
                    try:
                        inspect_hsa_state(hsa_system)
                        if not (hasattr(hsa_system, 'is_initialized') and 
                                hsa_system.is_initialized and 
                                hasattr(hsa_system, 'splat_registry') and 
                                hasattr(hsa_system.splat_registry, 'splats')):
                            print("HSA state is invalid after error - attempting to restore from safety checkpoint")
                            # Try to restore from the safety checkpoint
                            safety_checkpoint_path = os.path.join(
                                args.output_dir, 
                                args.model_name, 
                                "checkpoints", 
                                f"pre_adaptation_step_{global_step}.pt"
                            )
                            if os.path.exists(safety_checkpoint_path) and os.path.exists(f"{safety_checkpoint_path}.success"):
                                print(f"Restoring from safety checkpoint: {safety_checkpoint_path}")
                                checkpoint = torch.load(safety_checkpoint_path, map_location=args.device)
                                
                                # Only restore HSA state, not model, optimizer, etc.
                                if "hsa_data" in checkpoint and checkpoint["hsa_data"]:
                                    restore_hsa_from_checkpoint(hsa_system, checkpoint["hsa_data"])
                                    print(f"Successfully restored HSA state from safety checkpoint")
                                    log_recovery(f"Successfully restored HSA state from safety checkpoint")
                    except Exception as inspect_error:
                        print(f"Error inspecting HSA state: {inspect_error}")
                        log_debug(f"Error inspecting HSA state: {inspect_error}")
            
            # Evaluate periodically
            if global_step % args.eval_interval == 0:
                # Run evaluation
                model.eval()
                val_loss = evaluate(model, val_loader, args.device, max_batches=50)  # Use subset for faster evaluation
                val_losses.append(val_loss)
                
                # Generate a sample
                sample_text = generate_sample(
                    model, 
                    tokenizer, 
                    "Once upon a time",
                    max_length=50
                )
                
                print(f"\nStep {global_step} - Val Loss: {val_loss:.4f}")
                print(f"Sample: {sample_text}\n")
                
                # Check for best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, scheduler, global_step, epoch,
                        val_loss, os.path.join(args.output_dir, args.model_name, "checkpoints", "best_model.pt"),
                        hsa_system
                    )
                
                # Switch back to training mode
                model.train()
            
            # Save checkpoint periodically
            if global_step % args.save_interval == 0 and global_step > 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step, epoch,
                    loss.item(), os.path.join(args.output_dir, args.model_name, "checkpoints", f"step_{global_step}.pt"),
                    hsa_system
                )
            
            global_step += 1
        
        # End of epoch
        epoch_avg_loss = epoch_loss / epoch_samples
        print(f"\nEpoch {epoch+1}/{args.epochs} complete - Avg Loss: {epoch_avg_loss:.4f}")
        
        # Evaluate on full validation set
        val_loss = evaluate(model, val_loader, args.device)
        print(f"Full validation loss: {val_loss:.4f}")
        
        # Save epoch checkpoint
        save_checkpoint(
            model, optimizer, scheduler, global_step, epoch,
            val_loss, os.path.join(args.output_dir, args.model_name, "checkpoints", f"epoch_{epoch+1}.pt"),
            hsa_system
        )
        
        # Plot training progress
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title("Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        
        plt.subplot(1, 2, 2)
        plt.plot(val_losses)
        plt.title("Validation Loss")
        plt.xlabel("Evaluations")
        plt.ylabel("Loss")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, args.model_name, f"training_progress_epoch_{epoch+1}.png"))

    # Training complete
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, global_step, args.epochs,
        val_loss, os.path.join(args.output_dir, args.model_name, "checkpoints", "final_model.pt"),
        hsa_system
    )

    # Save tokenizer
    tokenizer.save_pretrained(os.path.join(args.output_dir, args.model_name, "tokenizer"))

    # Generate a final sample
    final_sample = generate_sample(
        model, 
        tokenizer, 
        "Once upon a time there was a little", 
        max_length=200,
        temperature=0.8
    )
    print(f"\nFinal sample:\n{final_sample}")

    # Save HSA configuration and stats
    hsa_stats = hsa_system.get_stats()
    with open(os.path.join(args.output_dir, args.model_name, "hsa_stats.json"), "w") as f:
        json.dump({
            "config": hsa_config,
            "stats": hsa_stats
        }, f, indent=4)

    # Create final visualizations
    print("Creating final visualizations...")
    try:
        # Get a batch of embeddings for visualization
        with torch.no_grad():
            visualize_batch = next(iter(val_loader))
            vis_input_ids = visualize_batch["input_ids"][:4].to(args.device)
            vis_embeddings = model.transformer.word_embeddings(vis_input_ids)
            vis_embeddings = vis_embeddings.detach().cpu().numpy()
            vis_flattened = vis_embeddings.reshape(-1, vis_embeddings.shape[-1])
            
            # Compute attention for visualization
            attention_matrix = hsa_system.compute_attention(vis_flattened[:100])
            
            # Create dashboard
            dashboard_path = visualizer.create_dashboard(
                splat_registry=hsa_system.splat_registry,
                attention_matrix=attention_matrix,
                tokens=vis_flattened[:100],
                title="Final HSA Dashboard",
                show=False,
                save=True
            )
            
            # Visualize splat distribution
            viz_path = visualizer.visualize_splat_distribution(
                splat_registry=hsa_system.splat_registry,
                tokens=vis_flattened,
                title="Final HSA Splat Distribution",
                show=False,
                save=True
            )
            
            # Visualize hierarchy
            hierarchy_path = visualizer.visualize_hierarchy(
                splat_registry=hsa_system.splat_registry,
                title="Final HSA Hierarchy",
                show=False,
                save=True
            )
            
            # Visualize attention sparsity
            sparsity_path = visualizer.visualize_attention_sparsity(
                attention_matrix=attention_matrix,
                title="Final HSA Attention Sparsity",
                show=False,
                save=True
            )
            
            print("Final visualizations created")
    except Exception as e:
        print(f"Warning: Could not create final visualizations: {e}")
        log_debug(f"Error creating final visualizations: {e}")
        log_debug(traceback.format_exc())

    print(f"\nTraining completed successfully! Model and outputs saved to {os.path.join(args.output_dir, args.model_name)}")

except Exception as e:
    # Handle unexpected errors during training
    print(f"\nERROR: An unexpected error occurred during training:")
    traceback.print_exc()
    log_debug("FATAL ERROR during training:")
    log_debug(traceback.format_exc())
    
    # Try to save emergency checkpoint
    emergency_path = os.path.join(args.output_dir, args.model_name, "checkpoints", "crash_recovery.pt")
    try:
        # Inspect HSA state before saving
        log_debug("Inspecting HSA state before emergency save:")
        inspect_hsa_state(hsa_system)
        
        save_checkpoint(
            model, optimizer, scheduler, global_step, 
            epoch if 'epoch' in locals() else 0,
            loss.item() if 'loss' in locals() else float('inf'), 
            emergency_path,
            hsa_system
        )
        print(f"Crash recovery checkpoint saved to {emergency_path}")
        log_recovery(f"CRASH: {str(e)}\nCrash recovery checkpoint saved to {emergency_path}")
    except Exception as save_error:
        print(f"Failed to save crash recovery checkpoint: {save_error}")
        log_recovery(f"CRASH: {str(e)}\nFailed to save crash recovery checkpoint: {save_error}")
    
    # Exit with error code
    sys.exit(1)

if __name__ == "__main__":
    # This allows the script to be imported without running
    pass
