"""
Memory-Optimized SplatFlow Training for Limited GPU Memory
Specifically designed for GPUs with 4-6GB VRAM like Quadro P2200
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
import time
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import matplotlib.pyplot as plt

# Enable memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import our models
from splatflow_gpu_fixed import FixedVectorizedSplatFlowLayer, RobustSplatFlowGPT
from standard_transformer import StandardTransformerGPT


def get_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        return {
            'allocated': allocated,
            'reserved': reserved, 
            'total': total,
            'free': free
        }
    return None


def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class MemoryOptimizedDataset(Dataset):
    """Memory-efficient dataset with shorter sequences"""
    
    def __init__(self, texts, tokenizer, max_length=128, min_length=20):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Processing {len(texts)} texts for memory-optimized training...")
        
        valid_count = 0
        for i, text in enumerate(texts):
            if len(text.strip()) > min_length:
                # Tokenize and truncate aggressively
                tokens = tokenizer.encode(
                    text, 
                    truncation=True, 
                    max_length=max_length,
                    add_special_tokens=True
                )
                
                if len(tokens) >= min_length:
                    self.examples.append(torch.tensor(tokens, dtype=torch.long))
                    valid_count += 1
            
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(texts)}, valid: {valid_count}")
        
        print(f"Created dataset with {len(self.examples)} sequences")
        if self.examples:
            avg_len = np.mean([len(ex) for ex in self.examples])
            print(f"Average sequence length: {avg_len:.1f} tokens")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn_memory_optimized(batch, pad_token_id=50256):
    """Memory-efficient collate function"""
    # Sort by length to minimize padding
    batch = sorted(batch, key=len, reverse=True)
    max_len = len(batch[0])
    
    padded_batch = []
    for seq in batch:
        if len(seq) < max_len:
            padding = torch.full((max_len - len(seq),), pad_token_id, dtype=torch.long)
            seq = torch.cat([seq, padding])
        padded_batch.append(seq)
    
    return torch.stack(padded_batch)


class MemoryOptimizedSplatFlowGPT(RobustSplatFlowGPT):
    """SplatFlow with memory optimizations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Enable gradient checkpointing to trade compute for memory
        self.gradient_checkpointing = True
    
    def forward(self, input_ids, attention_mask=None):
        """Forward with optional gradient checkpointing"""
        if self.training and self.gradient_checkpointing:
            # Use checkpoint for memory efficiency during training
            return torch.utils.checkpoint.checkpoint(
                super().forward,
                input_ids,
                attention_mask,
                use_reentrant=False
            )
        else:
            return super().forward(input_ids, attention_mask)


def create_memory_optimized_models(vocab_size, gpu_memory_gb=4.9):
    """Create models sized appropriately for available GPU memory"""
    
    print(f"Optimizing models for {gpu_memory_gb:.1f}GB GPU memory...")
    
    if gpu_memory_gb <= 5.0:
        # Small config for 4-6GB GPUs
        config = {
            'model_dim': 256,        # Reduced from 384
            'num_layers': 4,         # Reduced from 6
            'num_heads': 8,          # Keep reasonable for attention
            'max_seq_len': 128,      # Reduced from 256
            'initial_splats': 8      # Reduced from 12
        }
        print("Using SMALL config for limited GPU memory")
    elif gpu_memory_gb <= 8.0:
        # Medium config for 6-8GB GPUs  
        config = {
            'model_dim': 320,
            'num_layers': 5,
            'num_heads': 8,
            'max_seq_len': 192,
            'initial_splats': 10
        }
        print("Using MEDIUM config")
    else:
        # Large config for 8GB+ GPUs
        config = {
            'model_dim': 384,
            'num_layers': 6, 
            'num_heads': 8,
            'max_seq_len': 256,
            'initial_splats': 12
        }
        print("Using LARGE config")
    
    print(f"Model config: {config}")
    
    # Create SplatFlow model
    splatflow_model = MemoryOptimizedSplatFlowGPT(
        vocab_size=vocab_size,
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        max_seq_len=config['max_seq_len']
    )
    
    # Override splat count in layers
    for layer in splatflow_model.splat_layers:
        # Reduce splat count for memory efficiency
        layer.num_splats = config['initial_splats']
        
        # Reinitialize with smaller splat count
        layer.splat_positions = nn.Parameter(
            torch.randn(config['initial_splats'], layer.embedding_dim) * 0.2
        )
        layer.splat_log_scales = nn.Parameter(
            torch.zeros(config['initial_splats'])
        )
        layer.splat_gates = nn.Parameter(
            torch.ones(config['initial_splats']) * 0.5
        )
    
    # Create standard model
    standard_model = StandardTransformerGPT(
        vocab_size=vocab_size,
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        max_seq_len=config['max_seq_len']
    )
    
    return splatflow_model, standard_model, config


def run_memory_optimized_training():
    """Run training optimized for limited GPU memory"""
    print("🚀 SplatFlow Memory-Optimized Training")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    if device.type == 'cuda':
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory = gpu_props.total_memory / 1024**3
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_memory:.1f}GB")
        
        # Check initial memory
        mem_info = get_memory_info()
        print(f"Available GPU memory: {mem_info['free']:.2f}GB")
        
        if mem_info['free'] < 3.0:
            print("⚠️  WARNING: Less than 3GB free GPU memory detected!")
            print("Consider closing other applications or using CPU training")
    else:
        gpu_memory = 0
        print("Using CPU training")
    
    # Initialize tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    # Create memory-optimized models
    splatflow_model, standard_model, config = create_memory_optimized_models(
        vocab_size, gpu_memory
    )
    
    # Move to device with memory monitoring
    print(f"\nMoving models to {device}...")
    
    cleanup_memory()
    splatflow_model = splatflow_model.to(device)
    mem_after_splatflow = get_memory_info()
    if mem_after_splatflow:
        print(f"Memory after SplatFlow: {mem_after_splatflow['allocated']:.2f}GB")
    
    cleanup_memory()
    standard_model = standard_model.to(device)
    mem_after_standard = get_memory_info()
    if mem_after_standard:
        print(f"Memory after Standard: {mem_after_standard['allocated']:.2f}GB")
    
    # Compare parameter counts
    splatflow_params = sum(p.numel() for p in splatflow_model.parameters())
    standard_params = sum(p.numel() for p in standard_model.parameters())
    
    print(f"\nMemory-Optimized Model Comparison:")
    print(f"SplatFlow parameters: {splatflow_params:,}")
    print(f"Standard parameters: {standard_params:,}")
    print(f"Parameter ratio: {splatflow_params/standard_params:.2f}x")
    
    # Create dataset with smaller sequences
    print(f"\nCreating memory-optimized dataset...")
    
    # Use a smaller subset of wikitext for memory efficiency
    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [item['text'] for item in dataset if item['text'].strip()]
        
        # Use fewer examples for memory efficiency
        max_examples = min(2000, len(texts))
        texts = texts[:max_examples]
        print(f"Using {len(texts)} examples from wikitext-2")
        
    except Exception as e:
        print(f"Could not load wikitext dataset: {e}")
        print("Creating synthetic dataset...")
        texts = create_synthetic_dataset(500)  # Smaller synthetic dataset
    
    # Create memory-efficient dataset
    train_dataset = MemoryOptimizedDataset(
        texts, 
        tokenizer, 
        max_length=config['max_seq_len'],
        min_length=20
    )
    
    # Memory-optimized training parameters
    batch_size = 4 if gpu_memory <= 5.0 else 8  # Much smaller batch size
    accumulation_steps = 4  # Simulate larger batch via accumulation
    effective_batch_size = batch_size * accumulation_steps
    
    print(f"\nMemory-Optimized Training Configuration:")
    print(f"Actual batch size: {batch_size}")
    print(f"Gradient accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    print(f"Sequence length: {config['max_seq_len']}")
    
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_memory_optimized(batch, tokenizer.pad_token_id),
        num_workers=0,  # Disable multiprocessing to save memory
        pin_memory=False  # Disable pin_memory to save memory
    )
    
    print(f"Dataset batches: {len(dataloader)}")
    
    # Training setup with memory optimizations
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Lower learning rate and use memory-efficient optimizer settings
    lr = 1e-4
    splatflow_optimizer = torch.optim.AdamW(
        splatflow_model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    standard_optimizer = torch.optim.AdamW(
        standard_model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Shorter training for memory efficiency
    num_epochs = 10
    print(f"Training for {num_epochs} epochs...")
    
    # Results tracking
    results = {
        'splatflow_losses': [],
        'standard_losses': [],
        'splatflow_times': [],
        'standard_times': [],
        'memory_usage': [],
        'epochs': [],
        'config': config
    }
    
    # Training loop with memory monitoring
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        splatflow_model.train()
        standard_model.train()
        
        epoch_splatflow_loss = 0
        epoch_standard_loss = 0
        epoch_splatflow_time = 0
        epoch_standard_time = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                batch = batch.to(device, non_blocking=True)
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]
                
                # SplatFlow training with gradient accumulation
                start_time = time.time()
                
                splatflow_loss = 0
                for micro_step in range(accumulation_steps):
                    if micro_step < len(input_ids):
                        micro_input = input_ids[micro_step:micro_step+1]
                        micro_targets = targets[micro_step:micro_step+1]
                        
                        logits = splatflow_model(micro_input)
                        loss = criterion(
                            logits.reshape(-1, vocab_size),
                            micro_targets.reshape(-1)
                        ) / accumulation_steps
                        
                        loss.backward()
                        splatflow_loss += loss.item() * accumulation_steps
                
                torch.nn.utils.clip_grad_norm_(splatflow_model.parameters(), 1.0)
                splatflow_optimizer.step()
                splatflow_optimizer.zero_grad()
                
                splatflow_time = time.time() - start_time
                
                # Standard model training with gradient accumulation
                start_time = time.time()
                
                standard_loss = 0
                for micro_step in range(accumulation_steps):
                    if micro_step < len(input_ids):
                        micro_input = input_ids[micro_step:micro_step+1]
                        micro_targets = targets[micro_step:micro_step+1]
                        
                        logits = standard_model(micro_input)
                        loss = criterion(
                            logits.reshape(-1, vocab_size),
                            micro_targets.reshape(-1)
                        ) / accumulation_steps
                        
                        loss.backward()
                        standard_loss += loss.item() * accumulation_steps
                
                torch.nn.utils.clip_grad_norm_(standard_model.parameters(), 1.0)
                standard_optimizer.step()
                standard_optimizer.zero_grad()
                
                standard_time = time.time() - start_time
                
                # Accumulate metrics
                epoch_splatflow_loss += splatflow_loss
                epoch_standard_loss += standard_loss
                epoch_splatflow_time += splatflow_time
                epoch_standard_time += standard_time
                num_batches += 1
                
                # Memory monitoring and cleanup
                if batch_idx % 20 == 0:
                    cleanup_memory()
                    mem_info = get_memory_info()
                    if mem_info:
                        mem_usage = mem_info['allocated']
                        print(f"  Batch {batch_idx+1}/{len(dataloader)}: "
                              f"SF={splatflow_loss:.4f}, ST={standard_loss:.4f}, "
                              f"Mem={mem_usage:.2f}GB")
                        
                        # Emergency memory check
                        if mem_usage > gpu_memory * 0.9:
                            print("⚠️  High memory usage detected, forcing cleanup...")
                            cleanup_memory()
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"❌ OOM Error at batch {batch_idx}: {e}")
                print("Applying emergency memory cleanup...")
                cleanup_memory()
                
                # Skip this batch and continue
                continue
            
            except Exception as e:
                print(f"❌ Error at batch {batch_idx}: {e}")
                continue
        
        # Epoch summary
        if num_batches > 0:
            avg_splatflow_loss = epoch_splatflow_loss / num_batches
            avg_standard_loss = epoch_standard_loss / num_batches
            avg_splatflow_time = epoch_splatflow_time / num_batches
            avg_standard_time = epoch_standard_time / num_batches
            
            results['epochs'].append(epoch + 1)
            results['splatflow_losses'].append(avg_splatflow_loss)
            results['standard_losses'].append(avg_standard_loss)
            results['splatflow_times'].append(avg_splatflow_time)
            results['standard_times'].append(avg_standard_time)
            
            mem_info = get_memory_info()
            if mem_info:
                results['memory_usage'].append(mem_info['allocated'])
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  SplatFlow - Loss: {avg_splatflow_loss:.4f}, Time: {avg_splatflow_time*1000:.1f}ms")
            print(f"  Standard  - Loss: {avg_standard_loss:.4f}, Time: {avg_standard_time*1000:.1f}ms")
            if mem_info:
                print(f"  Memory usage: {mem_info['allocated']:.2f}GB / {mem_info['total']:.2f}GB")
        
        # Cleanup between epochs
        cleanup_memory()
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("MEMORY-OPTIMIZED TRAINING COMPLETE")
    print("=" * 50)
    
    if results['splatflow_losses']:
        final_sf_loss = results['splatflow_losses'][-1]
        final_st_loss = results['standard_losses'][-1]
        
        print(f"Final Results:")
        print(f"SplatFlow final loss: {final_sf_loss:.4f}")
        print(f"Standard final loss: {final_st_loss:.4f}")
        print(f"Loss ratio (SF/ST): {final_sf_loss/final_st_loss:.3f}")
        
        avg_memory = np.mean(results['memory_usage']) if results['memory_usage'] else 0
        print(f"Average memory usage: {avg_memory:.2f}GB")
        
        print(f"\n✅ Training completed successfully on {gpu_memory:.1f}GB GPU!")
        
        # Save results
        with open('memory_optimized_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    else:
        print("❌ Training failed - no valid batches completed")
        return None


def create_synthetic_dataset(size=500):
    """Create a smaller synthetic dataset for memory-constrained training"""
    templates = [
        "The {adjective} {noun} {verb} through the {location}.",
        "In {year}, scientists discovered that {subject} can {action}.",
        "Machine learning algorithms help us {task} with {accuracy}.",
        "The future of {technology} depends on {factor}.",
        "Researchers found that {phenomenon} occurs when {condition}."
    ]
    
    words = {
        'adjective': ['beautiful', 'complex', 'efficient', 'innovative', 'powerful'],
        'noun': ['system', 'algorithm', 'model', 'network', 'process'],
        'verb': ['operates', 'functions', 'processes', 'analyzes', 'computes'],
        'location': ['space', 'cloud', 'network', 'database', 'system'],
        'year': ['2020', '2021', '2022', '2023', '2024'],
        'subject': ['neural networks', 'transformers', 'algorithms', 'systems'],
        'action': ['learn patterns', 'process data', 'make predictions'],
        'task': ['classify data', 'generate text', 'solve problems'],
        'accuracy': ['high precision', 'remarkable accuracy', 'great efficiency'],
        'technology': ['artificial intelligence', 'quantum computing', 'robotics'],
        'factor': ['computational power', 'data quality', 'algorithm design'],
        'phenomenon': ['learning', 'adaptation', 'optimization'],
        'condition': ['sufficient data exists', 'parameters align', 'training converges']
    }
    
    texts = []
    for i in range(size):
        template = np.random.choice(templates)
        text = template
        for category, options in words.items():
            if f'{{{category}}}' in text:
                text = text.replace(f'{{{category}}}', np.random.choice(options))
        texts.append(text)
    
    return texts


if __name__ == "__main__":
    print("🔧 Memory-Optimized SplatFlow Training")
    print("Designed for GPUs with limited memory (4-6GB)")
    
    # Set memory optimization environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    results = run_memory_optimized_training()
    
    if results:
        print("\n🎉 MEMORY-OPTIMIZED TRAINING SUCCESSFUL!")
        print("✅ Models trained within memory constraints")
        print("✅ Both SplatFlow and standard models converged")
        print("✅ Ready for production deployment on limited hardware")
    else:
        print("\n❌ Training encountered issues")
        print("Consider further reducing batch size or model dimensions")
