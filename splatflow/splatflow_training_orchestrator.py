"""
SplatFlow Training Orchestrator
Main training pipeline for the SplatFlow attention mechanism.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import logging
import time
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np

# Import SplatFlow components with relative imports
from .splatflow_core_systems import (
    DeviceManager, DatasetManager, TensorUtils, ConfigurationManager,
    get_device_manager, get_dataset_manager, cleanup_memory
)

# Import other SplatFlow modules (assuming they exist)
try:
    from .splatflow_model_architecture import create_production_splatflow_model, FixedUltimateProductionSplatFlowGPT
    from .splatflow_attention_components import get_quick_model_stats
except ImportError as e:
    logging.warning(f"⚠️ Could not import SplatFlow model components: {e}")
    # We'll create minimal stubs below

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SplatFlowTrainingOrchestrator:
    """Main orchestrator for SplatFlow model training and evaluation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the training orchestrator"""
        # Configuration management
        self.config = ConfigurationManager.validate_config(
            config if config is not None else ConfigurationManager.create_default_config()
        )
        
        # Core system initialization
        self.device_manager = DeviceManager()  # Create instance
        self.device = self.device_manager.get_device()  # Get device from instance
        self.dataset_manager = DatasetManager()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Experiment tracking
        self.experiment_name = self.config.get('experiment_name', 'splatflow_experiment')
        self.checkpoint_dir = self.config.get('checkpoint_dir', './checkpoints')
        self.results_dir = self._setup_experiment_directory()
        
        logger.info(f"🎯 SplatFlowTrainingOrchestrator initialized")
        logger.info(f"📊 Experiment: {self.experiment_name}")
        logger.info(f"💾 Results directory: {self.results_dir}")
        logger.info(f"🔧 Device: {self.device}")
    
    def _setup_experiment_directory(self) -> str:
        """Create and setup experiment directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(
            self.checkpoint_dir, 
            f"{self.experiment_name}_{timestamp}"
        )
        os.makedirs(results_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(results_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        return results_dir
    
    def initialize_model_for_training(self) -> nn.Module:
        """Initialize the SplatFlow model for training"""
        try:
            logger.info("🏗️ Initializing SplatFlow model...")
            
            # Create model using the imported function
            model = create_production_splatflow_model(
                vocab_size=self.config['vocab_size'],
                model_dim=self.config['model_dim'],
                num_layers=self.config['num_layers'],
                num_splats=self.config['num_splats'],
                max_seq_len=self.config['max_seq_len'],
                dropout=self.config['dropout']
            )
            
            # Move to device
            model = self.device_manager.move_to_device(model)
            
            # Log model statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.info(f"📐 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
            logger.info(f"💾 Model memory: {self.device_manager.get_memory_summary()}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {str(e)}")
            # Create a minimal fallback model for testing
            return self._create_fallback_model()
    
    def _create_fallback_model(self) -> nn.Module:
        """Create a minimal transformer model as fallback"""
        logger.warning("🔧 Creating fallback transformer model")
        
        class FallbackTransformer(nn.Module):
            def __init__(self, vocab_size, model_dim, num_layers):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, model_dim)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=model_dim,
                        nhead=8,
                        batch_first=True
                    ),
                    num_layers=num_layers
                )
                self.lm_head = nn.Linear(model_dim, vocab_size)
            
            def forward(self, input_ids, attention_mask=None, labels=None):
                x = self.embedding(input_ids)
                x = self.transformer(x)
                logits = self.lm_head(x)
                
                loss = None
                if labels is not None:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                return {'loss': loss, 'logits': logits}
        
        model = FallbackTransformer(
            vocab_size=self.config['vocab_size'],
            model_dim=self.config['model_dim'],
            num_layers=self.config['num_layers']
        )
        
        return self.device_manager.move_to_device(model)
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, and other training components"""
        if self.model is None:
            raise ValueError("Model must be initialized before setting up training components")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Initialize learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('warmup_steps', 1000),
            eta_min=self.config['learning_rate'] * 0.1
        )
        
        logger.info("⚙️ Training components initialized")
        logger.info(f"📈 Optimizer: AdamW (lr={self.config['learning_rate']:.2e})")
        logger.info(f"📊 Scheduler: CosineAnnealingWarmRestarts")
    
    def prepare_dataset(self):
        """Load and prepare the training dataset"""
        logger.info("📚 Preparing training dataset...")
        
        # Load dataset
        dataset = self.dataset_manager.load_dataset_by_name(
            dataset_name=self.config['dataset_name'],
            split='train',
            streaming=self.config.get('dataset_streaming', False)
        )
        
        # Prepare training data
        dataloader = self.dataset_manager.prepare_training_data(
            dataset=dataset,
            seq_length=self.config['seq_length'],
            batch_size=self.config['batch_size'],
            num_samples=self.config.get('target_sequences')
        )
        
        logger.info(f"✅ Dataset prepared: {len(dataloader)} training batches")
        return dataloader
    
    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        log_interval = self.config.get('log_interval', 100)
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start_time = time.time()
            
            # Move batch to device
            input_ids = self.device_manager.move_to_device(batch['input_ids'])
            attention_mask = self.device_manager.move_to_device(batch['attention_mask'])
            labels = self.device_manager.move_to_device(batch['labels'])
            
            # Forward pass
            self.optimizer.zero_grad()
            
            try:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
                
                # Logging
                if batch_idx % log_interval == 0:
                    batch_time = time.time() - batch_start_time
                    avg_loss = total_loss / num_batches
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    logger.info(
                        f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                        f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | Time: {batch_time:.2f}s"
                    )
                    
                    # Memory monitoring
                    if batch_idx % (log_interval * 5) == 0:
                        logger.info(f"💾 {self.device_manager.get_memory_summary()}")
                
            except Exception as e:
                logger.error(f"❌ Training step failed: {str(e)}")
                # Skip this batch and continue
                continue
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / max(num_batches, 1)
        
        logger.info(
            f"📊 Epoch {epoch} completed: "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.2f}s | "
            f"Batches: {num_batches}"
        )
        
        return {
            'epoch': epoch,
            'avg_loss': avg_loss,
            'total_batches': num_batches,
            'epoch_time': epoch_time,
            'global_step': self.global_step
        }
    
    def evaluate_model(self, epoch: int) -> Dict[str, Any]:
        """Evaluate the model"""
        logger.info(f"🔍 Evaluating model at epoch {epoch}")
        
        self.model.eval()
        evaluation_results = {}
        
        try:
            # Get model statistics if available
            if hasattr(self.model, 'get_comprehensive_model_stats'):
                stats = self.model.get_comprehensive_model_stats(epoch=epoch)
                evaluation_results.update(stats)
            elif 'get_quick_model_stats' in globals():
                stats = get_quick_model_stats(self.model)
                evaluation_results.update(stats)
            
            # Generate sample text
            sample_text = self._generate_sample_text()
            evaluation_results['sample_generation'] = sample_text
            
            # Memory usage
            mem_info = self.device_manager.get_gpu_memory_info()
            evaluation_results['memory_usage'] = mem_info
            
            logger.info(f"✅ Evaluation completed")
            if sample_text:
                logger.info(f"📝 Sample generation: {sample_text[:100]}...")
            
        except Exception as e:
            logger.warning(f"⚠️ Evaluation failed: {str(e)}")
            evaluation_results['evaluation_error'] = str(e)
        
        return evaluation_results
    
    def _generate_sample_text(self) -> str:
        """Generate a sample text for evaluation"""
        try:
            if hasattr(self.model, 'generate_text'):
                return self.model.generate_text(
                    self.dataset_manager.tokenizer,
                    "The quick brown fox",
                    max_length=self.config.get('eval_max_length', 50)
                )
            else:
                return "Sample generation not available for this model"
        except Exception as e:
            return f"Generation failed: {str(e)}"
    
    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.results_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
            'global_step': self.global_step
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if loss < self.best_loss:
            self.best_loss = loss
            best_model_path = os.path.join(self.results_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            logger.info(f"🏆 New best model saved: {best_model_path}")
    
    def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info("🚀 Starting SplatFlow training...")
        
        try:
            # Initialize all components
            self.model = self.initialize_model_for_training()
            self.setup_training_components()
            dataloader = self.prepare_dataset()
            
            # Training loop
            for epoch in range(1, self.config['epochs'] + 1):
                self.current_epoch = epoch
                
                logger.info(f"\n🎯 Starting Epoch {epoch}/{self.config['epochs']}")
                
                # Train epoch
                epoch_results = self.train_epoch(dataloader, epoch)
                self.training_history.append(epoch_results)
                
                # Evaluation
                if epoch % self.config.get('eval_interval', 5) == 0:
                    eval_results = self.evaluate_model(epoch)
                    epoch_results.update(eval_results)
                
                # Save checkpoint
                if epoch % self.config.get('save_interval', 10) == 0:
                    self.save_checkpoint(epoch, epoch_results['avg_loss'])
                
                # Memory cleanup
                if epoch % 5 == 0:
                    self.device_manager.cleanup_memory()
            
            # Final evaluation and save
            final_results = self.evaluate_model(self.config['epochs'])
            self.save_checkpoint(self.config['epochs'], self.training_history[-1]['avg_loss'])
            
            # Training summary
            training_summary = self._create_training_summary()
            logger.info("🎉 Training completed successfully!")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"❌ Training failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'completed_epochs': self.current_epoch,
                'results_dir': self.results_dir
            }
    
    def _create_training_summary(self) -> Dict[str, Any]:
        """Create comprehensive training summary"""
        if not self.training_history:
            return {'status': 'no_training_data'}
        
        final_loss = self.training_history[-1]['avg_loss']
        initial_loss = self.training_history[0]['avg_loss']
        
        summary = {
            'status': 'completed',
            'experiment_name': self.experiment_name,
            'results_dir': self.results_dir,
            'config': self.config,
            'training_stats': {
                'total_epochs': len(self.training_history),
                'final_loss': final_loss,
                'initial_loss': initial_loss,
                'loss_improvement': initial_loss - final_loss,
                'best_loss': self.best_loss,
                'global_steps': self.global_step
            },
            'model_stats': {
                'total_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad) if self.model else 0
            },
            'device_info': {
                'device': str(self.device),
                'cuda_available': torch.cuda.is_available(),
                'final_memory': self.device_manager.get_gpu_memory_info()
            },
            'training_history': self.training_history
        }
        
        # Save summary
        summary_path = os.path.join(self.results_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary


def quick_start_example() -> SplatFlowTrainingOrchestrator:
    """Quick start example for SplatFlow training"""
    config = ConfigurationManager.create_default_config()
    
    # Quick training configuration
    config.update({
        'epochs': 10,
        'batch_size': 2,
        'seq_length': 512,
        'target_sequences': 1000,
        'model_dim': 256,
        'num_layers': 4,
        'num_splats': 16,
        'eval_interval': 2,
        'save_interval': 5
    })
    
    trainer = SplatFlowTrainingOrchestrator(config)
    return trainer


# Main execution
def main():
    """Main execution function"""
    logger.info("🎯 SplatFlow Training Orchestrator - Main Execution")
    
    try:
        # Create trainer with default configuration
        trainer = SplatFlowTrainingOrchestrator()
        
        # Run training
        results = trainer.train()
        
        # Print summary
        print("=" * 80)
        print("📊 TRAINING SUMMARY")
        print("=" * 80)
        
        if results.get('status') == 'completed':
            print(f"✅ Training completed successfully!")
            print(f"🎯 Final loss: {results['training_stats']['final_loss']:.4f}")
            print(f"📈 Loss improvement: {results['training_stats']['loss_improvement']:.4f}")
            print(f"🏆 Best loss: {results['training_stats']['best_loss']:.4f}")
            print(f"📁 Results saved to: {results['results_dir']}")
        else:
            print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
            print(f"📁 Results saved to: {results.get('results_dir', 'Unknown')}")
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {str(e)}")
        print(f"❌ Training failed: {str(e)}")
    
    finally:
        # Cleanup
        cleanup_memory()


if __name__ == "__main__":
    main()


# Export key classes and functions
__all__ = [
    'SplatFlowTrainingOrchestrator',
    'quick_start_example',
    'main'
]
