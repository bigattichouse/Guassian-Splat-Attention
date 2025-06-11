#!/usr/bin/env python3
"""
FIXED SplatFlow Training System - Memory Crisis Solution
Addresses CUDA OOM and splat birth storms with conservative configurations.
"""

import os
import sys
import time
import torch
import logging
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import SplatFlow components
try:
    from splatflow import (
        SplatFlowTrainingOrchestrator,
        create_default_config,
        setup_environment,
        cleanup_memory,
        get_gpu_memory_info,
        create_production_splatflow_model,
        get_quick_model_stats
    )
    SPLATFLOW_AVAILABLE = True
    print("✅ SplatFlow imports successful")
except ImportError as e:
    print(f"❌ Failed to import SplatFlow: {e}")
    sys.exit(1)


def get_fixed_hardware_configs():
    """FIXED hardware configurations to prevent memory crisis"""
    return {
        "ultra_conservative": {
            "name": "Ultra-Conservative 4GB - Emergency Fix",
            "description": "Emergency fix for memory crisis - 1K context",
            "memory_limit_gb": 3.5,
            "config": {
                "model_dim": 256,        # REDUCED from 384
                "num_layers": 3,         # REDUCED from 4
                "num_splats": 48,         # REDUCED from 12
                "max_splats": 256,        # REDUCED from 24
                "batch_size": 3,         # REDUCED from 2
                "gradient_accumulation_steps": 8,  # Compensate for smaller batch
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "target_sequences": 256,  # REDUCED for faster testing
                "steps_per_epoch": 50,   # REDUCED
                "seq_length": 512,      # REDUCED from 2048
                
                # FIXED: Disable aggressive birth system temporarily
                "disable_adaptive_birth": False,
                "max_births_per_epoch": 8,
                "birth_cooldown": 5,
                "coverage_threshold": 0.03, 
            }
        },
        "conservative": {
            "name": "Conservative 4-6GB - 1K Context",
            "description": "Safe for 4-6GB GPUs with controlled birthing",
            "memory_limit_gb": 4.0,
            "config": {
                "model_dim": 320,        # Moderate reduction
                "num_layers": 3,         
                "num_splats": 8,         # Conservative
                "max_splats": 12,        
                "batch_size": 1,         
                "gradient_accumulation_steps": 4,
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "target_sequences": 1000,
                "steps_per_epoch": 50,
                "seq_length": 1024,
                
                # FIXED: Conservative birth settings
                "max_births_per_epoch": 1,
                "birth_cooldown": 10,
                "coverage_threshold": 0.03,  # Much more lenient
                "min_cluster_size": 15,       # Require bigger clusters
            }
        },
        "safe_2k": {
            "name": "Safe 6GB+ - 2K Context with Fixed Births",
            "description": "2K context with properly controlled birth system",
            "memory_limit_gb": 6.0,
            "config": {
                "model_dim": 384,        
                "num_layers": 4,         
                "num_splats": 10,        # Reduced from 12
                "max_splats": 16,        # Reduced from 24
                "batch_size": 2,         
                "gradient_accumulation_steps": 2,
                "mixed_precision": True,
                "gradient_checkpointing": False,
                "target_sequences": 2000,
                "steps_per_epoch": 100,
                "seq_length": 2048,
                
                # FIXED: Controlled birth system
                "max_births_per_epoch": 1,
                "birth_cooldown": 5,
                "coverage_threshold": 0.05,  # More lenient
                "min_cluster_size": 10,       # Bigger clusters required
            }
        }
    }


def get_conservative_dataset_configs():
    """Conservative dataset configurations for memory-limited training"""
    return {
        "minimal": {
            "description": "Minimal testing dataset",
            "datasets": [
                {"name": "roneneldan/TinyStories", "samples": 100, "tier": 1},
            ],
            "target_sequences": 50
        },
        "smaller": {
            "description": "Small quality dataset for testing",
            "datasets": [
                {"name": "roneneldan/TinyStories", "samples": 300, "tier": 1},
                {"name": "squad", "samples": 200, "tier": 1},
            ],
            "target_sequences": 250
        },
        "conservative": {
            "description": "Conservative quality training",
            "datasets": [

                {"name": "roneneldan/TinyStories", "samples": 5000, "tier": 1},
                {"name": "roneneldan/TinyStories-Instruct", "samples": 5000, "tier": 1},
                {"name": "squad", "samples": 3000, "tier": 1},
                {"name": "ag_news", "samples": 2000, "tier": 1},
                {"name": "imdb", "samples": 2000, "tier": 1},
                
                # Tier 2: Large Scale Quality
                {"name": "cnn_dailymail", "samples": 3000, "tier": 2},
                {"name": "openwebtext", "samples": 4000, "tier": 2},
                {"name": "wikitext", "samples": 2000, "tier": 2},
                
                # Tier 3: Diverse Content
                {"name": "bookcorpus", "samples": 3000, "tier": 3},
                {"name": "amazon_polarity", "samples": 2000, "tier": 3},
                {"name": "yelp_review_full", "samples": 2000, "tier": 3},
                
                # Tier 4: Specialized
                {"name": "multi_news", "samples": 1000, "tier": 4},
                {"name": "xsum", "samples": 1500, "tier": 4},
                {"name": "reddit_tifu", "samples": 1000, "tier": 4},
            ],
            "target_sequences": 500
        }
    }


class MemorySafeSplatFlowTrainer:
    """Memory-safe SplatFlow trainer with birth control"""
    
    def __init__(self, hardware_tier: str = "small", 
                 dataset_config: str = "small", experiment_name: str = None):
        
        self.hardware_tier = hardware_tier
        self.dataset_config = dataset_config
        self.experiment_name = experiment_name or f"splatflow_fixed_{hardware_tier}_{int(time.time())}"
        
        # Setup directory structure
        self.experiment_dir = f"experiments/{self.experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/logs", exist_ok=True)
        
        # Get hardware configuration
        hardware_configs = get_fixed_hardware_configs()
        if hardware_tier not in hardware_configs:
            raise ValueError(f"Unknown hardware tier: {hardware_tier}. Available: {list(hardware_configs.keys())}")
        
        self.hardware_config = hardware_configs[hardware_tier]
        logger.info(f"🖥️  Hardware Tier: {self.hardware_config['name']}")
        logger.info(f"    Description: {self.hardware_config['description']}")
        logger.info(f"    Memory limit: {self.hardware_config['memory_limit_gb']:.1f}GB")
        
        # Build training configuration
        self.config = self._build_memory_safe_config()
        
        # Save configuration
        self._save_experiment_config()
        
        # Initialize trainer
        self.trainer = None
        
    def _build_memory_safe_config(self) -> Dict:
        """Build memory-safe training configuration"""
        
        # Start with SplatFlow defaults
        config = create_default_config()
        
        # Apply hardware tier settings
        config.update(self.hardware_config['config'])
        
        # Memory safety settings
        config.update({
            'epochs': 500,  # Reduced for testing
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'warmup_epochs': 3,  # Reduced
            'eval_interval': 5,
            'save_interval': 10,
            'log_interval': 5,  # More frequent logging
            'checkpoint_dir': f"{self.experiment_dir}/checkpoints",
            
            # Memory optimization
            'use_progressive_training': True,
            
            # Dataset configuration
            'dataset_config': self.dataset_config,
            
            # Monitoring
            'enable_detailed_monitoring': False,  # Disable to save memory
        })
        
        return config
    
    def check_memory_before_training(self):
        """Check available memory before starting training"""
        memory_info = get_gpu_memory_info()
        if memory_info:
            used_pct = memory_info['percent_used']
            available_gb = memory_info['free']
            
            logger.info(f"🔧 Pre-training memory check:")
            logger.info(f"   GPU memory used: {used_pct:.1f}%")
            logger.info(f"   Available memory: {available_gb:.2f}GB")
            
            if used_pct > 20:
                logger.warning(f"⚠️  High memory usage before training: {used_pct:.1f}%")
                logger.info("   Cleaning up memory...")
                cleanup_memory()
                
                # Check again
                memory_info = get_gpu_memory_info()
                if memory_info:
                    logger.info(f"   After cleanup: {memory_info['percent_used']:.1f}% used")
            
            # Warn if available memory is low
            if available_gb < 2.0:
                logger.warning(f"⚠️  Low available memory: {available_gb:.2f}GB")
                logger.warning("   Consider using ultra_conservative configuration")
        else:
            logger.info("🔧 Using CPU - no memory constraints")
    
    def _save_experiment_config(self):
        """Save experiment configuration"""
        config_path = f"{self.experiment_dir}/config.json"
        
        experiment_info = {
            'experiment_name': self.experiment_name,
            'hardware_tier': self.hardware_tier,
            'dataset_config': self.dataset_config,
            'hardware_config': self.hardware_config,
            'training_config': self.config,
            'created_at': datetime.now().isoformat(),
            'context_length': self.config['seq_length'],
            'splatflow_version': "1.0.0-fixed",
            'fixes_applied': [
                "Reduced model dimensions",
                "Conservative splat birth control", 
                "Memory monitoring",
                "Gradient accumulation for small batches"
            ]
        }
        
        with open(config_path, 'w') as f:
            json.dump(experiment_info, f, indent=2)
        
        logger.info(f"💾 Experiment config saved to {config_path}")
    
    def train(self) -> Dict:
        """Run memory-safe SplatFlow training"""
        logger.info(f"🚀 Starting FIXED SplatFlow training...")
        logger.info(f"   Experiment: {self.experiment_name}")
        logger.info(f"   Context length: {self.config['seq_length']:,} tokens")
        logger.info(f"   Hardware tier: {self.hardware_tier}")
        logger.info(f"   Dataset config: {self.dataset_config}")
        logger.info(f"   Model dimension: {self.config['model_dim']}")
        logger.info(f"   Batch size: {self.config['batch_size']} (effective: {self.config['batch_size'] * self.config.get('gradient_accumulation_steps', 1)})")
        
        # Setup environment
        setup_environment()
        
        # Memory check
        self.check_memory_before_training()
        
        # Create trainer
        self.trainer = SplatFlowTrainingOrchestrator(self.config)
        
        try:
            # Run training
            training_summary = self.trainer.train()
            
            # Save final results
            self._save_training_results(training_summary)
            
            logger.info(f"🎉 FIXED training completed successfully!")
            logger.info(f"   Final loss: {training_summary.get('best_loss', 'Unknown')}")
            logger.info(f"   Total epochs: {training_summary.get('total_epochs', 'Unknown')}")
            logger.info(f"   Total steps: {training_summary.get('total_steps', 'Unknown')}")
            
            return training_summary
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            
            # Check if it's a memory error
            if "CUDA out of memory" in str(e):
                logger.error("🚨 MEMORY ERROR DETECTED!")
                logger.error("   Suggestions:")
                logger.error("   1. Use 'ultra_conservative' hardware tier")
                logger.error("   2. Use 'minimal' dataset config")
                logger.error("   3. Reduce batch_size to 1")
                logger.error("   4. Enable gradient_checkpointing")
            
            raise
    
    def _save_training_results(self, training_summary: Dict):
        """Save training results"""
        results_path = f"{self.experiment_dir}/results.json"
        
        results = {
            'experiment_name': self.experiment_name,
            'hardware_tier': self.hardware_tier,
            'dataset_config': self.dataset_config,
            'training_summary': training_summary,
            'completed_at': datetime.now().isoformat(),
            'context_length': self.config['seq_length'],
            'fixes_applied': [
                "Memory-safe configurations",
                "Conservative splat birth control",
                "Gradient accumulation for small batches",
                "Progressive layer training"
            ]
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 Training results saved to {results_path}")
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        logger.info(f"🔍 Evaluating model...")
        
        if not self.trainer or not self.trainer.model:
            raise ValueError("No trained model available for evaluation")
        
        model = self.trainer.model
        
        # Conservative test prompts
        test_prompts = [
            "Once upon a time",
            "The scientist discovered",
            "In the future",
        ]
        
        eval_results = {}
        
        logger.info("📝 Testing generation...")
        for i, prompt in enumerate(test_prompts):
            try:
                generated = model.generate_text(
                    self.trainer.tokenizer,
                    prompt,
                    max_length=50,  # Conservative length
                    temperature=0.8,
                    top_k=50
                )
                eval_results[f'generation_{i}'] = {
                    'prompt': prompt,
                    'generated': generated
                }
                logger.info(f"   '{prompt}' -> '{generated}'")
            except Exception as e:
                logger.warning(f"   Generation {i+1} failed: {e}")
        
        # Model health check
        try:
            stats = get_quick_model_stats(model)
            eval_results['model_health'] = stats
            logger.info(f"🏥 Model health: {stats['health_pct']:.1f}% healthy splats")
            logger.info(f"   Total splats: {stats['total_splats']}")
            logger.info(f"   Birth events: {stats['total_births']}")
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
        
        # Save evaluation results
        eval_path = f"{self.experiment_dir}/evaluation.json"
        with open(eval_path, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info(f"💾 Evaluation results saved to {eval_path}")
        
        return eval_results


def demonstrate_memory_analysis():
    """Demonstrate memory usage for different configurations"""
    
    logger.info("📊 Memory Analysis for Different Configurations")
    logger.info("=" * 60)
    
    configs = get_fixed_hardware_configs()
    
    for name, config in configs.items():
        hw_config = config['config']
        
        # Rough memory estimation
        model_dim = hw_config['model_dim']
        num_layers = hw_config['num_layers']
        seq_len = hw_config['seq_length']
        batch_size = hw_config['batch_size']
        num_splats = hw_config['num_splats']
        
        # Parameter count estimation
        vocab_size = 50257  # GPT-2 vocab
        embed_params = vocab_size * model_dim
        layer_params = (model_dim * model_dim * 4) * num_layers  # Rough estimate
        total_params = embed_params + layer_params
        
        # Memory estimation (very rough)
        param_memory = (total_params * 4) / (1024**3)  # 4 bytes per param
        activation_memory = (batch_size * seq_len * model_dim * 4) / (1024**3)
        attention_memory = (batch_size * seq_len * num_splats * 4) / (1024**3)
        
        total_memory = param_memory + activation_memory + attention_memory
        
        logger.info(f"\n{name.upper()}:")
        logger.info(f"  Model: {model_dim}d, {num_layers} layers, {seq_len} context")
        logger.info(f"  Parameters: ~{total_params/1000000:.1f}M")
        logger.info(f"  Estimated memory: ~{total_memory:.1f}GB")
        logger.info(f"  Splats per layer: {num_splats}")
        logger.info(f"  Batch size: {batch_size}")


def main():
    """Main function with fixed configurations"""
    parser = argparse.ArgumentParser(description="FIXED SplatFlow Training - Memory Crisis Solution")
    
    parser.add_argument("--hardware", "-hw", 
                       choices=list(get_fixed_hardware_configs().keys()), 
                       default="ultra_conservative",
                       help="Hardware tier configuration")
    
    parser.add_argument("--dataset", "-d",
                       choices=list(get_conservative_dataset_configs().keys()),
                       default="conservative", 
                       help="Dataset configuration")
    
    parser.add_argument("--experiment", "-e",
                       type=str,
                       help="Experiment name")
    
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of training epochs")
    
    parser.add_argument("--memory-analysis", action="store_true",
                       help="Show memory analysis for configurations")
    
    args = parser.parse_args()
    
    print("🛠️  SplatFlow FIXED Training - Memory Crisis Solution")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hardware tier: {args.hardware}")
    print(f"Dataset config: {args.dataset}")
    
    # Show memory analysis
    if args.memory_analysis:
        demonstrate_memory_analysis()
        return
    
    # Show configuration info
    hw_configs = get_fixed_hardware_configs()
    ds_configs = get_conservative_dataset_configs()
    
    print(f"\n🖥️  Hardware Configuration: {hw_configs[args.hardware]['name']}")
    print(f"    {hw_configs[args.hardware]['description']}")
    print(f"    Memory limit: {hw_configs[args.hardware]['memory_limit_gb']:.1f}GB")
    
    print(f"\n📚 Dataset Configuration: {args.dataset}")
    print(f"    {ds_configs[args.dataset]['description']}")
    print(f"    Target sequences: {ds_configs[args.dataset]['target_sequences']}")
    
    try:
        # Create trainer
        trainer = MemorySafeSplatFlowTrainer(
            hardware_tier=args.hardware,
            dataset_config=args.dataset,
            experiment_name=args.experiment
        )
        
        print(f"\n📁 Experiment directory: {trainer.experiment_dir}")
        
        # Override epochs if specified
        if args.epochs:
            trainer.config['epochs'] = args.epochs
            logger.info(f"🔧 Override epochs: {args.epochs}")
        
        print(f"\n🚀 Starting FIXED training...")
        
        # Run training
        training_summary = trainer.train()
        
        # Run evaluation
        print(f"\n🔍 Running evaluation...")
        eval_results = trainer.evaluate_model()
        
        print("\n" + "=" * 80)
        print("🎉 FIXED SPLATFLOW TRAINING COMPLETED SUCCESSFULLY!")
        print("Key improvements:")
        print("   ✅ Memory crisis resolved")
        print("   ✅ Conservative splat birth control") 
        print("   ✅ Gradient accumulation for small batches")
        print("   ✅ Progressive layer training")
        print(f"   📁 Results saved in: {trainer.experiment_dir}")
        print(f"   📊 Final loss: {training_summary.get('best_loss', 'Unknown'):.4f}")
        print(f"   🏥 Model health: {eval_results.get('model_health', {}).get('health_pct', 0):.1f}% healthy splats")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        
        # Provide helpful error guidance
        if "CUDA out of memory" in str(e):
            print("\n🚨 MEMORY ERROR GUIDANCE:")
            print("   Try these fixes:")
            print("   1. python script.py --hardware ultra_conservative --dataset minimal")
            print("   2. Reduce batch size further in config")
            print("   3. Use CPU training (slower but no memory limits)")
            print("   4. Close other GPU processes")
        
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
