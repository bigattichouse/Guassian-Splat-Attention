"""
SplatFlow Training Orchestrator Module
Main training loop, evaluation, and execution logic for the SplatFlow training system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import json
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer

from .splatflow_core_systems import (
    setup_environment, cleanup_memory, get_gpu_memory_info,
    EnhancedRealDataset
)
from .splatflow_model_architecture import (
    FixedUltimateProductionSplatFlowGPT,
    create_production_splatflow_model,
    initialize_model_for_training
)
from .splatflow_attention_components import get_quick_model_stats

logger = logging.getLogger(__name__)


class SplatFlowTrainingOrchestrator:
    """Main orchestrator for SplatFlow training with comprehensive features"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.progressive_trainer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        logger.info(f"🎭 SplatFlow Training Orchestrator initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Vocab size: {self.tokenizer.vocab_size}")
    
    def create_model(self) -> FixedUltimateProductionSplatFlowGPT:
        """Create and initialize the SplatFlow model"""
        
        logger.info(f"🏗️ Creating SplatFlow model...")
        
        self.model = create_production_splatflow_model(
            vocab_size=self.tokenizer.vocab_size,
            model_dim=self.config.get('model_dim', 512),
            num_layers=self.config.get('num_layers', 6),
            num_splats=self.config.get('num_splats', 20),
            max_splats=self.config.get('max_splats', 64),
            max_seq_len=self.config.get('max_seq_len', 2048),
            dropout=self.config.get('dropout', 0.1)
        )
        
        self.model = self.model.to(self.device)
        
        # Report model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"   ✅ Model created with {total_params:,} parameters")
        
        return self.model
    
    def create_dataset_and_dataloader(self) -> DataLoader:
        """Create dataset and dataloader"""
        
        logger.info(f"📚 Creating enhanced real dataset...")
        
        dataset = EnhancedRealDataset(
            tokenizer=self.tokenizer,
            seq_length=self.config.get('seq_length', 1024),
            target_sequences=self.config.get('target_sequences', 10000)
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 2),
            shuffle=True,
            num_workers=0,  # Keep as 0 for stability
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"   ✅ Dataset created with {len(dataset)} sequences")
        logger.info(f"   ✅ DataLoader created with batch size {self.config.get('batch_size', 2)}")
        
        return dataloader
    
    def setup_optimizer_and_scheduler(self, dataloader_len: int):
        """Setup optimizer and learning rate scheduler with proper error handling"""
        
        logger.info(f"🎯 Setting up optimizer and scheduler...")
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 3e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        # Calculate total steps with safety checks
        epochs = max(1, self.config.get('epochs', 50))
        steps_per_epoch = max(1, self.config.get('steps_per_epoch', min(100, dataloader_len)))
        total_steps = epochs * steps_per_epoch
        
        # Ensure we have at least 1 step
        total_steps = max(1, total_steps)
        
        logger.info(f"   Epochs: {epochs}")
        logger.info(f"   Steps per epoch: {steps_per_epoch}")
        logger.info(f"   Total steps: {total_steps}")
        
        # Create scheduler with safety checks
        try:
            # Use a simple linear warmup and cosine decay if total_steps is very small
            if total_steps <= 10:
                logger.info(f"   Using simple StepLR for small training run")
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=max(1, total_steps // 2),
                    gamma=0.5
                )
            else:
                warmup_steps = max(1, int(total_steps * 0.1))  # 10% warmup
                logger.info(f"   Warmup steps: {warmup_steps}")
                
                self.scheduler = optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.config.get('learning_rate', 3e-4),
                    total_steps=total_steps,
                    pct_start=min(0.3, warmup_steps / total_steps),  # Cap warmup percentage
                    anneal_strategy='cos'
                )
                
        except Exception as e:
            logger.warning(f"   ⚠️ Failed to create OneCycleLR scheduler: {e}")
            logger.info(f"   🔄 Falling back to simple StepLR")
            
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, total_steps // 3),
                gamma=0.8
            )
        
        logger.info(f"   ✅ Optimizer and scheduler setup complete")
    
    def initialize_model_for_training(self, sample_batch: torch.Tensor):
        """Initialize model with proper splat positioning"""
        
        logger.info(f"🎯 Initializing model for training...")
        
        self.progressive_trainer = initialize_model_for_training(
            self.model, 
            sample_batch,
            setup_progressive=self.config.get('use_progressive_training', True)
        )
        
        if self.progressive_trainer:
            logger.info(f"   ✅ Progressive layer training enabled")
        
        logger.info(f"   ✅ Model initialization complete")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch with enhanced error handling"""
        
        self.model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_start_time = time.time()
        
        # Update progressive training
        if self.progressive_trainer:
            try:
                self.progressive_trainer.update_active_layers(epoch)
            except Exception as e:
                logger.warning(f"Progressive training update failed: {e}")
        
        max_steps = self.config.get('steps_per_epoch', len(dataloader))
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                batch = batch.to(self.device)
                
                # Apply progressive repositioning
                if epoch > 0 and hasattr(self.model, 'apply_progressive_repositioning'):
                    try:
                        self.model.apply_progressive_repositioning(batch, epoch)
                    except Exception as e:
                        logger.warning(f"Progressive repositioning failed: {e}")
                
                # Apply emergency rescue if needed
                if epoch > 5 and hasattr(self.model, 'apply_emergency_rescue'):
                    try:
                        self.model.apply_emergency_rescue(batch, epoch)
                    except Exception as e:
                        logger.warning(f"Emergency rescue failed: {e}")
                
                # Forward pass
                input_ids = batch[:, :-1]
                targets = batch[:, 1:]
                
                logits = self.model(input_ids)
                
                # Compute loss with safety checks
                if logits.numel() == 0 or targets.numel() == 0:
                    logger.warning(f"Empty tensors in batch {batch_idx}, skipping")
                    continue
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=self.tokenizer.pad_token_id
                )
                
                # Check for invalid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Invalid loss in batch {batch_idx}: {loss.item()}, skipping")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Update scheduler
                if self.scheduler:
                    try:
                        self.scheduler.step()
                    except Exception as e:
                        logger.warning(f"Scheduler step failed: {e}")
                
                # Update statistics
                epoch_loss += loss.item()
                epoch_steps += 1
                self.global_step += 1
                
                # Log progress
                if batch_idx % self.config.get('log_interval', 10) == 0:
                    try:
                        quick_stats = get_quick_model_stats(self.model)
                        current_lr = self.optimizer.param_groups[0]['lr']
                        logger.info(f"   Batch {batch_idx}: loss={loss.item():.4f}, "
                                  f"lr={current_lr:.6f}, "
                                  f"healthy_splats={quick_stats['healthy_splats']}/{quick_stats['total_splats']}")
                    except Exception as e:
                        logger.warning(f"Failed to get quick stats: {e}")
                        logger.info(f"   Batch {batch_idx}: loss={loss.item():.4f}")
                
                # Memory cleanup
                if batch_idx % 20 == 0:
                    cleanup_memory()
                
                # Break if we've done enough steps for this epoch
                if epoch_steps >= max_steps:
                    break
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                cleanup_memory()
                continue
        
        # Calculate epoch statistics with safety checks
        if epoch_steps > 0:
            avg_loss = epoch_loss / epoch_steps
        else:
            logger.warning("No successful training steps in epoch")
            avg_loss = float('inf')
        
        epoch_time = time.time() - epoch_start_time
        
        # Get current learning rate safely
        try:
            current_lr = self.optimizer.param_groups[0]['lr']
        except Exception:
            current_lr = self.config.get('learning_rate', 3e-4)
        
        return {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'epoch_time': epoch_time,
            'steps': epoch_steps,
            'lr': current_lr
        }
    
    def evaluate_model(self, test_prompts: List[str]) -> Dict:
        """Evaluate model with text generation"""
        
        self.model.eval()
        evaluation_results = {}
        
        logger.info(f"🔍 Evaluating model with {len(test_prompts)} prompts...")
        
        with torch.no_grad():
            for i, prompt in enumerate(test_prompts):
                try:
                    generated_text = self.model.generate_text(
                        self.tokenizer,
                        prompt,
                        max_length=self.config.get('eval_max_length', 50),
                        temperature=self.config.get('eval_temperature', 0.8),
                        top_k=self.config.get('eval_top_k', 50)
                    )
                    
                    evaluation_results[f'prompt_{i}'] = {
                        'prompt': prompt,
                        'generated': generated_text,
                        'length': len(generated_text.split())
                    }
                    
                    logger.info(f"   Prompt {i}: '{prompt[:30]}...' -> '{generated_text[:50]}...'")
                    
                except Exception as e:
                    logger.error(f"Error generating for prompt {i}: {e}")
                    evaluation_results[f'prompt_{i}'] = {
                        'prompt': prompt,
                        'generated': f"[Error: {str(e)}]",
                        'length': 0
                    }
        
        # Get comprehensive model statistics
        try:
            model_stats = self.model.get_comprehensive_model_stats(self.current_epoch)
            evaluation_results['model_stats'] = model_stats
        except Exception as e:
            logger.warning(f"Failed to get model stats: {e}")
            evaluation_results['model_stats'] = {'overall_health': 'Unknown'}
        
        return evaluation_results
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint"""
        
        try:
            checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save regular checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'splatflow_epoch_{epoch}.pt')
            
            optimizer_state = None
            try:
                optimizer_state = self.optimizer.state_dict()
            except Exception as e:
                logger.warning(f"Failed to get optimizer state: {e}")
            
            self.model.save_model_checkpoint(
                checkpoint_path,
                epoch,
                optimizer_state=optimizer_state
            )
            
            # Save best model
            if is_best:
                best_path = os.path.join(checkpoint_dir, 'splatflow_best.pt')
                self.model.save_model_checkpoint(
                    best_path,
                    epoch,
                    optimizer_state=optimizer_state
                )
                logger.info(f"💎 New best model saved at epoch {epoch}")
            
            # Save training history
            history_path = os.path.join(checkpoint_dir, 'training_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def train(self) -> Dict:
        """Main training loop with comprehensive error handling"""
        
        logger.info(f"🚀 Starting SplatFlow training...")
        logger.info(f"   Config: {self.config}")
        
        try:
            # Create model
            self.create_model()
            
            # Create dataset and dataloader
            dataloader = self.create_dataset_and_dataloader()
            
            # Setup optimizer and scheduler
            self.setup_optimizer_and_scheduler(len(dataloader))
            
            # Initialize model with first batch
            try:
                first_batch = next(iter(dataloader))
                self.initialize_model_for_training(first_batch)
            except Exception as e:
                logger.error(f"Failed to initialize model with first batch: {e}")
                # Create a dummy batch for initialization
                dummy_batch = torch.randint(0, 1000, (1, self.config.get('seq_length', 128))).to(self.device)
                self.initialize_model_for_training(dummy_batch)
            
            # Training loop
            for epoch in range(self.config.get('epochs', 50)):
                self.current_epoch = epoch
                
                logger.info(f"\n🎓 Epoch {epoch + 1}/{self.config.get('epochs', 50)}")
                
                # Train epoch
                epoch_results = self.train_epoch(dataloader, epoch)
                
                # Update training history
                self.training_history.append(epoch_results)
                
                # Check if this is the best model
                current_loss = epoch_results.get('avg_loss', float('inf'))
                is_best = current_loss < self.best_loss
                if is_best and current_loss != float('inf'):
                    self.best_loss = current_loss
                
                # Log epoch results
                logger.info(f"   📊 Epoch {epoch + 1} complete:")
                logger.info(f"      Loss: {epoch_results['avg_loss']:.4f}")
                logger.info(f"      Time: {epoch_results['epoch_time']:.1f}s")
                logger.info(f"      Steps: {epoch_results['steps']}")
                logger.info(f"      LR: {epoch_results['lr']:.6f}")
                
                # Evaluation
                if (epoch + 1) % self.config.get('eval_interval', 5) == 0:
                    test_prompts = [
                        "Once upon a time",
                        "The scientist discovered",
                        "In the future, technology will",
                        "The cat sat on"
                    ]
                    
                    try:
                        eval_results = self.evaluate_model(test_prompts)
                        
                        # Log model health
                        model_stats = eval_results.get('model_stats', {})
                        logger.info(f"   🏥 Model Health: {model_stats.get('overall_health', 'Unknown')}")
                        logger.info(f"      Healthy splats: {model_stats.get('total_healthy_splats', 0)}/{model_stats.get('total_splats', 0)}")
                        logger.info(f"      Health percentage: {model_stats.get('health_percentage', 0):.1f}%")
                    except Exception as e:
                        logger.warning(f"Evaluation failed: {e}")
                
                # Save checkpoint
                if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                    self.save_checkpoint(epoch + 1, is_best)
                
                # Memory cleanup
                cleanup_memory()
                
                # GPU memory info
                if torch.cuda.is_available() and (epoch + 1) % 5 == 0:
                    try:
                        mem_info = get_gpu_memory_info()
                        if mem_info:
                            logger.info(f"   💾 GPU Memory: {mem_info['percent_used']:.1f}% used "
                                      f"({mem_info['allocated']:.1f}GB/{mem_info['total']:.1f}GB)")
                    except Exception:
                        pass  # Don't fail on memory info
            
            # Final evaluation
            logger.info(f"\n🎯 Final evaluation...")
            final_prompts = [
                "The story begins when",
                "Scientists have proven that",
                "In a world where",
                "The adventure started",
                "Technology has changed"
            ]
            
            try:
                final_eval = self.evaluate_model(final_prompts)
            except Exception as e:
                logger.warning(f"Final evaluation failed: {e}")
                final_eval = {'model_stats': {'overall_health': 'Unknown'}}
            
            # Save final model
            self.save_checkpoint(self.config.get('epochs', 50), is_best=True)
            
            # Training summary
            training_summary = {
                'total_epochs': self.config.get('epochs', 50),
                'best_loss': self.best_loss,
                'total_steps': self.global_step,
                'final_model_stats': final_eval.get('model_stats', {}),
                'config': self.config,
                'training_history': self.training_history
            }
            
            logger.info(f"✅ Training complete!")
            logger.info(f"   Best loss: {self.best_loss:.4f}")
            logger.info(f"   Total steps: {self.global_step}")
            logger.info(f"   Final health: {final_eval.get('model_stats', {}).get('overall_health', 'Unknown')}")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"❌ Training failed with error: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Return partial results
            return {
                'total_epochs': self.current_epoch,
                'best_loss': self.best_loss,
                'total_steps': self.global_step,
                'error': str(e),
                'config': self.config,
                'training_history': self.training_history
            }


def create_default_config() -> Dict:
    """Create default training configuration"""
    return {
        # Model architecture
        'model_dim': 192,
        'num_layers': 3,
        'num_splats': 20,
        'max_splats': 64,
        'dropout': 0.1,
        
        # Training parameters
        'epochs': 50,
        'batch_size': 2,
        'seq_length': 1024,
        'target_sequences': 10000,
        'steps_per_epoch': 100,
        
        # Optimization
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'max_seq_len': 2048,
        
        # Progressive training
        'use_progressive_training': True,
        
        # Evaluation
        'eval_interval': 5,
        'eval_max_length': 50,
        'eval_temperature': 0.8,
        'eval_top_k': 50,
        
        # Logging and saving
        'log_interval': 10,
        'save_interval': 10,
        'checkpoint_dir': 'splatflow_checkpoints'
    }


def main():
    """Main execution function"""
    
    # Setup environment
    setup_environment()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SplatFlow Model')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    parser.add_argument('--model_dim', type=int, default=192, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--num_splats', type=int, default=20, help='Number of splats per layer')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='splatflow_checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info(f"📁 Loaded config from {args.config}")
    else:
        config = create_default_config()
        logger.info(f"🔧 Using default configuration")
    
    # Override config with command line arguments
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            config[key] = value
    
    # Log final configuration
    logger.info(f"🎛️ Final training configuration:")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")
    
    # Create and run trainer
    try:
        orchestrator = SplatFlowTrainingOrchestrator(config)
        training_summary = orchestrator.train()
        
        # Save training summary
        summary_path = os.path.join(config['checkpoint_dir'], 'training_summary.json')
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        logger.info(f"📋 Training summary saved to {summary_path}")
        logger.info(f"🎉 SplatFlow training completed successfully!")
        
        return training_summary
        
    except KeyboardInterrupt:
        logger.info(f"⏹️ Training interrupted by user")
        return None
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
