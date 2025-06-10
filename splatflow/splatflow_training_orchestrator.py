"""
SplatFlow Training Orchestrator Module
Main training loop and orchestration for the SplatFlow system.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import os
from typing import Dict, Optional, List, Any
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from .splatflow_core_systems import (
    setup_environment, cleanup_memory, get_gpu_memory_info,
    DeviceManager, EnhancedRealDataset
)
from .splatflow_model_architecture import (
    create_production_splatflow_model,
    initialize_model_for_training
)

logger = logging.getLogger(__name__)


def create_default_config():
    """Create default configuration for SplatFlow training"""
    return {
        # Model architecture
        'model_dim': 512,
        'num_layers': 6,
        'num_splats': 20,
        'max_splats': 64,
        'max_seq_len': 2048,
        'dropout': 0.1,
        
        # Training parameters
        'epochs': 50,
        'batch_size': 2,
        'seq_length': 1024,
        'target_sequences': 10000,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        
        # Progressive training
        'use_progressive_training': True,
        'warmup_epochs': 3,
        
        # Evaluation
        'eval_interval': 5,
        'eval_max_length': 50,
        'eval_temperature': 0.8,
        'eval_top_k': 50,
        
        # Logging and saving
        'log_interval': 10,
        'save_interval': 10,
        'checkpoint_dir': 'splatflow_checkpoints',
        
        # Dataset
        'dataset_name': 'enhanced_real',
        'steps_per_epoch': None  # Will be calculated from dataset size
    }


class SplatFlowTrainingOrchestrator:
    """Main training orchestrator for SplatFlow models"""
    
    def __init__(self, config: Dict):
        self.config = {**create_default_config(), **config}
        self.device = DeviceManager.get_primary_device()
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.optimizer = None
        self.scheduler = None
        self.progressive_trainer = None
        
        # Training state
        self.current_epoch = 0
        self.total_steps = 0
        self.best_loss = float('inf')
        self.training_stats = []
        
        # Create checkpoint directory
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        logger.info(f"🎯 SplatFlow Training Orchestrator initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Checkpoint dir: {self.config['checkpoint_dir']}")
    
    def setup_tokenizer(self):
        """Setup the tokenizer"""
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"✅ Tokenizer loaded: vocab_size={self.tokenizer.vocab_size}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup tokenizer: {e}")
            return False
    
    def setup_model(self):
        """Setup the SplatFlow model"""
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer must be setup before model")
            
            # Create model
            self.model = create_production_splatflow_model(
                vocab_size=self.tokenizer.vocab_size,
                model_dim=self.config['model_dim'],
                num_layers=self.config['num_layers'],
                num_splats=self.config['num_splats'],
                max_splats=self.config['max_splats'],
                max_seq_len=self.config['max_seq_len'],
                dropout=self.config['dropout']
            )
            
            self.model = self.model.to(self.device)
            
            # Get model statistics
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            logger.info(f"✅ Model created and moved to {self.device}")
            logger.info(f"   Total parameters: {total_params:,}")
            logger.info(f"   Trainable parameters: {trainable_params:,}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup model: {e}")
            return False
    
    def setup_dataset(self):
        """Setup the dataset and dataloader"""
        try:
            if not self.tokenizer:
                raise ValueError("Tokenizer must be setup before dataset")
            
            # Create dataset
            self.dataset = EnhancedRealDataset(
                tokenizer=self.tokenizer,
                seq_length=self.config['seq_length'],
                target_sequences=self.config['target_sequences']
            )
            
            # Create dataloader
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                drop_last=True,
                pin_memory=True if torch.cuda.is_available() else False
            )
            
            # Calculate steps per epoch if not specified
            if self.config['steps_per_epoch'] is None:
                self.config['steps_per_epoch'] = len(self.dataloader)
            
            logger.info(f"✅ Dataset and dataloader created")
            logger.info(f"   Dataset size: {len(self.dataset):,} sequences")
            logger.info(f"   Batches per epoch: {len(self.dataloader):,}")
            logger.info(f"   Steps per epoch: {self.config['steps_per_epoch']}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup dataset: {e}")
            return False
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        try:
            if not self.model:
                raise ValueError("Model must be setup before optimizer")
            
            # Create optimizer
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            
            # Create scheduler
            total_steps = self.config['epochs'] * self.config['steps_per_epoch']
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config['learning_rate'] * 0.1
            )
            
            logger.info(f"✅ Optimizer and scheduler created")
            logger.info(f"   Learning rate: {self.config['learning_rate']}")
            logger.info(f"   Weight decay: {self.config['weight_decay']}")
            logger.info(f"   Total training steps: {total_steps}")
            
            return True
        except Exception as e:
            logger.error(f"❌ Failed to setup optimizer: {e}")
            return False
    
    def initialize_training(self):
        """Initialize all components for training"""
        logger.info("🚀 Initializing SplatFlow training components...")
        
        # Setup components in order
        if not self.setup_tokenizer():
            return False
        
        if not self.setup_model():
            return False
        
        if not self.setup_dataset():
            return False
        
        if not self.setup_optimizer():
            return False
        
        # Initialize model with sample batch
        try:
            sample_batch = next(iter(self.dataloader))
            self.progressive_trainer = initialize_model_for_training(
                self.model, 
                sample_batch, 
                setup_progressive=self.config['use_progressive_training']
            )
            
            if self.progressive_trainer:
                logger.info("✅ Progressive layer training enabled")
            
        except Exception as e:
            logger.warning(f"⚠️ Model initialization failed: {e}")
        
        logger.info("✅ All training components initialized successfully!")
        return True
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Update progressive training
        if self.progressive_trainer:
            self.progressive_trainer.update_active_layers(epoch)
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= self.config['steps_per_epoch']:
                break
            
            try:
                # Move batch to device
                batch = batch.to(self.device)
                
                # Create input and target
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(input_ids)
                
                # Compute loss
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=self.tokenizer.pad_token_id
                )
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                
                # Update statistics
                epoch_loss += loss.item()
                num_batches += 1
                self.total_steps += 1
                
                # Apply model maintenance
                if batch_idx % 3 == 0:  # Every 3 batches
                    self.model.apply_progressive_repositioning(batch, epoch)
                
                if batch_idx % 5 == 0:  # Every 5 batches
                    self.model.apply_emergency_rescue(batch, epoch)
                
                # Logging
                if (batch_idx + 1) % self.config['log_interval'] == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    logger.info(f"Epoch {epoch}, Batch {batch_idx + 1}/{self.config['steps_per_epoch']}: "
                               f"loss={loss.item():.4f}, lr={current_lr:.2e}")
                
            except Exception as e:
                logger.warning(f"⚠️ Training step failed at epoch {epoch}, batch {batch_idx}: {e}")
                continue
        
        # Calculate epoch statistics
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / max(num_batches, 1)
        
        # Update best loss
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
        
        # Store epoch stats
        epoch_stats = {
            'epoch': epoch,
            'loss': avg_loss,
            'lr': self.scheduler.get_last_lr()[0],
            'time': epoch_time,
            'steps': num_batches
        }
        self.training_stats.append(epoch_stats)
        
        logger.info(f"📊 Epoch {epoch} completed: loss={avg_loss:.4f}, time={epoch_time:.1f}s")
        
        return epoch_stats
    
    def evaluate_model(self, epoch):
        """Evaluate the model"""
        try:
            logger.info(f"🔍 Evaluating model at epoch {epoch}...")
            
            # Generate sample text
            sample_prompts = ["Once upon a time", "The quick brown fox", "In a world where"]
            
            for prompt in sample_prompts:
                try:
                    generated = self.model.generate_text(
                        self.tokenizer,
                        prompt,
                        max_length=self.config['eval_max_length'],
                        temperature=self.config['eval_temperature'],
                        top_k=self.config['eval_top_k']
                    )
                    logger.info(f"   '{prompt}' -> '{generated}'")
                except Exception as e:
                    logger.warning(f"Generation failed for prompt '{prompt}': {e}")
            
            # Get model statistics
            model_stats = self.model.get_comprehensive_model_stats(epoch)
            logger.info(f"   Model health: {model_stats['overall_health']}")
            logger.info(f"   Healthy splats: {model_stats['total_healthy_splats']}/{model_stats['total_splats']}")
            
            return model_stats
            
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return {}
    
    def save_checkpoint(self, epoch, model_stats):
        """Save model checkpoint"""
        try:
            checkpoint_path = os.path.join(
                self.config['checkpoint_dir'],
                f"splatflow_epoch_{epoch}.pt"
            )
            
            self.model.save_model_checkpoint(
                checkpoint_path,
                epoch,
                self.optimizer.state_dict()
            )
            
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def train(self):
        """Main training loop"""
        logger.info("🚀 Starting SplatFlow training...")
        
        # Initialize training
        if not self.initialize_training():
            raise RuntimeError("Failed to initialize training components")
        
        # Training loop
        try:
            for epoch in range(self.config['epochs']):
                logger.info(f"\n🎯 Starting epoch {epoch + 1}/{self.config['epochs']}")
                
                # Train epoch
                epoch_stats = self.train_epoch(self.dataloader, epoch)
                
                # Evaluation
                if (epoch + 1) % self.config['eval_interval'] == 0:
                    model_stats = self.evaluate_model(epoch)
                else:
                    model_stats = {}
                
                # Save checkpoint
                if (epoch + 1) % self.config['save_interval'] == 0:
                    self.save_checkpoint(epoch, model_stats)
                
                # Memory cleanup
                if epoch % 5 == 0:
                    cleanup_memory()
                
                self.current_epoch = epoch
        
        except KeyboardInterrupt:
            logger.info("⚠️ Training interrupted by user")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        
        # Final evaluation and save
        logger.info("🏁 Training completed, running final evaluation...")
        final_model_stats = self.evaluate_model(self.current_epoch)
        self.save_checkpoint(self.current_epoch, final_model_stats)
        
        # Create training summary
        training_summary = {
            'config': self.config,
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.total_steps,
            'best_loss': self.best_loss,
            'final_model_stats': final_model_stats,
            'training_stats': self.training_stats,
            'device': str(self.device)
        }
        
        logger.info(f"✅ Training summary:")
        logger.info(f"   Total epochs: {training_summary['total_epochs']}")
        logger.info(f"   Total steps: {training_summary['total_steps']}")
        logger.info(f"   Best loss: {training_summary['best_loss']:.4f}")
        logger.info(f"   Final health: {final_model_stats.get('overall_health', 'Unknown')}")
        
        return training_summary


def quick_start_example():
    """Quick start example with minimal configuration"""
    quick_config = {
        'model_dim': 128,
        'num_layers': 2,
        'num_splats': 8,
        'epochs': 5,
        'batch_size': 1,
        'seq_length': 512,
        'target_sequences': 100,
        'steps_per_epoch': 10,
        'eval_interval': 2,
        'save_interval': 5,
        'checkpoint_dir': 'splatflow_quickstart'
    }
    
    trainer = SplatFlowTrainingOrchestrator(quick_config)
    
    logger.info("🚀 SplatFlow Quick Start Example Ready!")
    logger.info("   Run: trainer.train() to start training")
    
    return trainer
