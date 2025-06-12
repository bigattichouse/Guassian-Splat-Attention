#!/usr/bin/env python3
"""
FIXED: Enhanced SplatFlow Trainer 2K - Phase 2 Integration
Complete testing and training system with Phase 2 O(n*k) optimizations.
Includes stability enhancements, constellation templates, splat specialization, and selective processing.

CRITICAL FIXES:
- Resolves dropout parameter conflicts in configuration creation
- Proper parameter validation and cleaning
- Enhanced error handling and recovery
"""

import torch
import torch.nn as nn
import numpy as np
import json
import argparse
import time
import os
import sys
import logging
from datetime import datetime
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from splatflow import (
    SplatFlowTrainingOrchestrator,
    create_default_config,
    create_onk_enhanced_config,
    get_predefined_configurations,
    create_phase2_config,
    get_phase2_status,
    run_phase2_tests,
    PHASE_2_FEATURES,
    setup_environment,
    cleanup_memory,
    get_gpu_memory_info,
    validate_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase2TrainingMonitor:
    """Enhanced training monitor with Phase 2 feature tracking"""
    
    def __init__(self):
        self.loss_history = deque(maxlen=1000)
        self.epoch_losses = []
        self.best_loss = float('inf')
        self.valid_step_count = 0
        self.failed_step_count = 0
        self.consecutive_failures = 0
        self.recovery_attempts = 0
        
        # Phase 2 specific tracking
        self.specialization_transitions = defaultdict(int)
        self.constellation_adaptations = 0
        self.selective_processing_efficiency = []
        self.onk_computation_savings = []
        self.cache_hit_rates = []
        
        # Feature availability tracking
        self.phase2_status = get_phase2_status()
        
    def record_step(self, loss: float, step_info: Optional[Dict] = None):
        """Record training step with Phase 2 information"""
        if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
            self.failed_step_count += 1
            self.consecutive_failures += 1
            return False
        else:
            self.loss_history.append(loss)
            if loss < self.best_loss:
                self.best_loss = loss
            self.valid_step_count += 1
            self.consecutive_failures = 0
            
            # Record Phase 2 information
            if step_info:
                self._record_phase2_info(step_info)
            
            return True
    
    def _record_phase2_info(self, step_info: Dict):
        """Record Phase 2 specific information"""
        try:
            # Track specialization transitions
            if 'specialization_changes' in step_info:
                for role_change in step_info['specialization_changes']:
                    self.specialization_transitions[role_change] += 1
            
            # Track constellation adaptations
            if 'constellation_adaptations' in step_info:
                self.constellation_adaptations += step_info['constellation_adaptations']
            
            # Track selective processing efficiency
            if 'selective_processing_ratio' in step_info:
                self.selective_processing_efficiency.append(step_info['selective_processing_ratio'])
            
            # Track computation savings
            if 'onk_computation_savings' in step_info:
                self.onk_computation_savings.append(step_info['onk_computation_savings'])
            
            # Track cache performance
            if 'cache_hit_rate' in step_info:
                self.cache_hit_rates.append(step_info['cache_hit_rate'])
                
        except Exception as e:
            logger.warning(f"Failed to record Phase 2 info: {e}")
    
    def get_epoch_summary(self, epoch: int) -> Dict:
        """Get comprehensive epoch summary including Phase 2 metrics"""
        
        base_summary = {
            'epoch': epoch,
            'avg_loss': np.mean(list(self.loss_history)[-50:]) if self.loss_history else float('inf'),
            'best_loss': self.best_loss,
            'valid_steps': self.valid_step_count,
            'failed_steps': self.failed_step_count,
            'success_rate': self.valid_step_count / max(self.valid_step_count + self.failed_step_count, 1),
            'consecutive_failures': self.consecutive_failures,
            'recovery_attempts': self.recovery_attempts
        }
        
        # Add Phase 2 metrics if available
        if self.phase2_status['available_count'] > 0:
            phase2_summary = {
                'phase2_features_active': self.phase2_status['available_count'],
                'specialization_transitions': dict(self.specialization_transitions),
                'constellation_adaptations': self.constellation_adaptations,
                'avg_selective_processing_efficiency': (
                    np.mean(self.selective_processing_efficiency) 
                    if self.selective_processing_efficiency else 0.0
                ),
                'avg_computation_savings': (
                    np.mean(self.onk_computation_savings) 
                    if self.onk_computation_savings else 0.0
                ),
                'avg_cache_hit_rate': (
                    np.mean(self.cache_hit_rates) 
                    if self.cache_hit_rates else 0.0
                )
            }
            base_summary.update(phase2_summary)
        
        return base_summary


class EnhancedPhase2Config:
    """FIXED: Enhanced configuration system with proper parameter handling"""
    
    @staticmethod
    def get_enhanced_configs():
        """Get enhanced configurations with Phase 2 integration and fixed parameter handling"""
        
        # Get base predefined configurations
        base_configs = get_predefined_configurations()
        
        # Enhanced configurations combining stability + Phase 2
        enhanced_configs = {
            # Stability-first configurations (Phase 2 features optional)
            "stability_1k": {
                "name": "🛡️ Stability-First 1K - Ultra Safe",
                "description": "Maximum stability for troubleshooting, basic Phase 2",
                "memory_limit_gb": 3.0,
                "config": {
                    **create_default_config(),
                    "model_dim": 256,
                    "num_layers": 2,
                    "num_splats": 6,
                    "max_splats": 24,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "target_sequences": 500,
                    "steps_per_epoch": 40,
                    "seq_length": 1024,
                    "max_seq_len": 1024,
                    "epochs": 30,
                    
                    # Minimal Phase 2 features for testing
                    "enable_hierarchical_caching": True,
                    "enable_splat_aware_feedforward": False,  # Start conservative
                    "enable_selective_processing": False,
                    "enable_splat_specialization": False,
                    "enable_constellation_templates": False,
                    
                    # FIXED: Clean feedforward kwargs
                    "feedforward_kwargs": {},  # No conflicting parameters
                    
                    # Enhanced stability controls
                    "max_births_per_epoch": 1,
                    "birth_cooldown": 10,
                    "emergency_recovery_enabled": True,
                    "memory_monitoring_enabled": True,
                    "cleanup_frequency": 5,
                }
            },
            
            "stability_2k_phase2": {
                "name": "🛡️🌟 Stability 2K + Phase 2 Basic",
                "description": "Stable 2K with basic Phase 2 features enabled",
                "memory_limit_gb": 4.5,
                "config": {
                    **create_default_config(),
                    "model_dim": 320,
                    "num_layers": 3,
                    "num_splats": 8,
                    "max_splats": 32,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 6,
                    "target_sequences": 800,
                    "steps_per_epoch": 60,
                    "seq_length": 2048,
                    "max_seq_len": 2048,
                    "epochs": 50,
                    
                    # Basic Phase 2 features
                    "enable_hierarchical_caching": True,
                    "enable_splat_aware_feedforward": True,
                    "enable_selective_processing": True,
                    "enable_splat_specialization": False,  # Advanced feature
                    "enable_constellation_templates": False,  # Advanced feature
                    
                    # Phase 2 configurations
                    "feedforward_type": "splat_aware",
                    "selection_threshold": 0.5,
                    "trajectory_cache_levels": ["local", "chunk"],
                    
                    # FIXED: Clean feedforward kwargs
                    "feedforward_kwargs": {},
                    
                    # Stability controls
                    "max_births_per_epoch": 1,
                    "birth_cooldown": 8,
                    "emergency_recovery_enabled": True,
                    "memory_monitoring_enabled": True,
                }
            },
            
            "research_4k_full_phase2": {
                "name": "🔬🌟 Research 4K - Complete Phase 2",
                "description": "Full Phase 2 implementation for research and testing",
                "memory_limit_gb": 8.0,
                "config": {
                    **create_onk_enhanced_config(),
                    "model_dim": 512,
                    "num_layers": 3,
                    "num_splats": 48,
                    "max_splats": 256,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "target_sequences": 3000,
                    "steps_per_epoch": 80,
                    "seq_length": 2048,
                    "max_seq_len": 2048,
                    "epochs": 75,
                    
                    # Full Phase 2 feature set
                    "enable_hierarchical_caching": True,
                    "enable_splat_aware_feedforward": True,
                    "enable_selective_processing": True,
                    "enable_splat_specialization": True,
                    "enable_constellation_templates": True,
                    "enable_hierarchical_norm": True,
                    "enable_sparse_output": True,
                    
                    # Advanced Phase 2 configurations
                    "feedforward_type": "splat_aware",
                    "onk_optimization_level": "standard",
                    "trajectory_cache_levels": ["local", "chunk", "sequence"],
                    
                    # FIXED: Clean feedforward kwargs
                    "feedforward_kwargs": {},
                    
                    # Enhanced stability with Phase 2
                    "max_births_per_epoch": 2,
                    "birth_cooldown": 6,
                    "emergency_recovery_enabled": True,
                    "memory_monitoring_enabled": True,
                }
            },
            
            "production_8k_optimized": {
                "name": "🏭🚀 Production 8K - Optimized Phase 2",
                "description": "Production-ready 8K with optimized Phase 2 features",
                "memory_limit_gb": 12.0,
                "config": {
                    **create_onk_enhanced_config(),
                    "model_dim": 768,
                    "num_layers": 8,
                    "num_splats": 24,
                    "max_splats": 64,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 8,
                    "target_sequences": 5000,
                    "steps_per_epoch": 120,
                    "seq_length": 8192,
                    "max_seq_len": 8192,
                    "epochs": 100,
                    
                    # Production-optimized Phase 2 features
                    "enable_hierarchical_caching": True,
                    "enable_splat_aware_feedforward": True,
                    "enable_selective_processing": True,
                    "enable_splat_specialization": True,
                    "enable_constellation_templates": True,
                    "enable_hierarchical_norm": False,  # Conservative for production
                    "enable_sparse_output": True,
                    
                    # Production Phase 2 configurations
                    "feedforward_type": "splat_aware",
                    "onk_optimization_level": "standard",  # Not aggressive for stability
                    "trajectory_cache_levels": ["local", "chunk"],  # Conservative caching
                    
                    # FIXED: Clean feedforward kwargs
                    "feedforward_kwargs": {},
                    
                    # Production stability
                    "max_births_per_epoch": 1,
                    "birth_cooldown": 10,
                    "emergency_recovery_enabled": True,
                    "memory_monitoring_enabled": True,
                    "cleanup_frequency": 3,  # Frequent cleanup for long sequences
                }
            },
            
            # Include original base configurations for compatibility
            **base_configs
        }
        
        return enhanced_configs


class Phase2SplatFlowTrainer:
    """FIXED: Enhanced trainer with Phase 2 integration and proper configuration validation"""
    
    def __init__(self, config_name: str = "stability_2k_phase2", 
                 dataset_config: str = "conservative", 
                 experiment_name: str = None):
        
        self.config_name = config_name
        self.dataset_config = dataset_config
        self.experiment_name = experiment_name or f"phase2_splatflow_{config_name}_{int(time.time())}"
        
        # Setup directory structure
        self.experiment_dir = f"experiments/{self.experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/phase2_analysis", exist_ok=True)
        
        # Get enhanced configuration
        enhanced_configs = EnhancedPhase2Config.get_enhanced_configs()
        if config_name not in enhanced_configs:
            raise ValueError(f"Unknown configuration: {config_name}. "
                           f"Available: {list(enhanced_configs.keys())}")
        
        self.hardware_config = enhanced_configs[config_name]
        logger.info(f"🌟 Phase 2 Configuration: {self.hardware_config['name']}")
        logger.info(f"    Description: {self.hardware_config['description']}")
        
        # CRITICAL FIX: Build and validate configuration
        self.config = self._build_phase2_config()
        
        # Initialize Phase 2 monitoring
        self.training_monitor = Phase2TrainingMonitor()
        
        # Phase 2 statistics
        self.phase2_analytics = {
            'feature_utilization': defaultdict(int),
            'optimization_gains': [],
            'stability_metrics': [],
            'convergence_analysis': []
        }
        
        # Save configuration
        self._save_experiment_config()
        
        logger.info(f"🌟 Phase 2 Enhanced SplatFlow Trainer initialized")
        logger.info(f"    Phase 2 features enabled: {self._count_enabled_features()}")
        logger.info(f"    Available Phase 2 features: {sum(PHASE_2_FEATURES.values())}/{len(PHASE_2_FEATURES)}")
    
    def _count_enabled_features(self) -> int:
        """Count enabled Phase 2 features in configuration"""
        return sum(1 for key in ['enable_splat_aware_feedforward', 'enable_selective_processing', 
                                'enable_splat_specialization', 'enable_constellation_templates',
                                'enable_hierarchical_caching'] 
                  if self.config.get(key, False))
    
    def _build_phase2_config(self) -> Dict:
        """FIXED: Build Phase 2-enhanced configuration with proper validation"""
        
        # Start with hardware tier settings
        config = self.hardware_config['config'].copy()
        
        # CRITICAL FIX: Validate and clean the configuration
        config = validate_config(config)
        
        # Enhanced dataset settings based on Phase 2 capabilities
        base_target_sequences = config.get('target_sequences', 1000)
        base_steps_per_epoch = config.get('steps_per_epoch', 50)
        
        dataset_configs = {
            "minimal": {
                "target_sequences": base_target_sequences,
                "steps_per_epoch": base_steps_per_epoch
            },
            "conservative": {
                "target_sequences": int(base_target_sequences * 1.2),
                "steps_per_epoch": int(base_steps_per_epoch * 1.1)
            },
            "extensive": {
                "target_sequences": int(base_target_sequences * 1.5),
                "steps_per_epoch": int(base_steps_per_epoch * 1.3)
            }
        }
        
        config.update(dataset_configs[self.dataset_config])
        
        # Enhanced training settings for Phase 2
        phase2_training_enhancements = {
            'learning_rate': 1.5e-4,  # Slightly reduced for Phase 2 stability
            'weight_decay': 0.01,
            'warmup_epochs': 3,
            'eval_interval': 5,
            'save_interval': 10,
            'log_interval': 5,
            'checkpoint_dir': f"{self.experiment_dir}/checkpoints",
            
            # Phase 2 specific settings
            'content_type': 'general',  # For constellation templates
            'enhanced_gradient_management': True,
            'adaptive_batch_sizing': config.get('enable_splat_aware_feedforward', False),
            'dynamic_sequence_padding': config.get('enable_splat_aware_feedforward', False),
            
            # Memory and stability
            'enable_memory_monitoring': True,
            'cleanup_frequency': config.get('cleanup_frequency', 5),
            'cuda_safety_enabled': True,
            'stability_mode_enabled': True,
            
            # Dataset configuration
            'dataset_config': {
                'type': self.dataset_config,
                'phase2_optimized': config.get('enable_splat_aware_feedforward', False)
            }
        }
        
        config.update(phase2_training_enhancements)
        
        # Final validation to ensure all parameters are clean
        config = validate_config(config)
        
        logger.info(f"🌟 Phase 2 Configuration Built:")
        logger.info(f"   Model: {config['model_dim']}d, {config['num_layers']} layers, {config['num_splats']} splats")
        logger.info(f"   Sequence: {config['seq_length']} length, {config['batch_size']} batch size")
        logger.info(f"   Training: {config['epochs']} epochs, {config['steps_per_epoch']} steps/epoch")
        logger.info(f"   Target sequences: {config['target_sequences']:,}")
        
        if config.get('enable_splat_aware_feedforward', False):
            logger.info(f"   🚀 O(n*k) feedforward: {config.get('enable_splat_aware_feedforward', False)}")
            logger.info(f"   🎯 Selective processing: {config.get('enable_selective_processing', False)}")
            logger.info(f"   🎭 Splat specialization: {config.get('enable_splat_specialization', False)}")
            logger.info(f"   🌌 Constellation templates: {config.get('enable_constellation_templates', False)}")
            
            if config.get('onk_optimization_level'):
                logger.info(f"   ⚡ O(n*k) optimization level: {config['onk_optimization_level']}")
        
        return config
    
    def _save_experiment_config(self):
        """Save experiment configuration and Phase 2 status"""
        
        config_path = os.path.join(self.experiment_dir, "experiment_config.json")
        
        experiment_config = {
            'experiment_name': self.experiment_name,
            'config_name': self.config_name,
            'dataset_config': self.dataset_config,
            'hardware_config': self.hardware_config,
            'training_config': self.config,
            'phase2_status': self.training_monitor.phase2_status,
            'created_at': datetime.now().isoformat(),
            'pytorch_version': torch.__version__,
            'phase2_features_available': PHASE_2_FEATURES
        }
        
        with open(config_path, 'w') as f:
            json.dump(experiment_config, f, indent=2, default=str)
        
        logger.info(f"💾 Experiment configuration saved to {config_path}")
    
    def run_phase2_diagnostics(self):
        """Run comprehensive Phase 2 diagnostics"""
        
        logger.info("🔍 Running Phase 2 diagnostics...")
        
        # Check Phase 2 availability
        phase2_status = get_phase2_status()
        logger.info(f"📊 Phase 2 Status: {phase2_status['available_count']}/{phase2_status['total_features']} features available")
        
        for feature, available in phase2_status['features_available'].items():
            status_icon = "✅" if available else "❌"
            logger.info(f"   {status_icon} {feature}: {'Available' if available else 'Not Available'}")
        
        # Run Phase 2 tests if features are available
        if phase2_status['available_count'] > 0:
            logger.info("🧪 Running Phase 2 component tests...")
            test_results = run_phase2_tests()
            
            logger.info(f"🧪 Phase 2 Tests: {test_results['passed']}/{test_results['total']} passed")
            for test_name, result in test_results['results'].items():
                if result is not None:
                    status_icon = "✅" if result else "❌"
                    logger.info(f"   {status_icon} {test_name}: {'PASS' if result else 'FAIL'}")
        
        return phase2_status
    
    def train(self) -> Dict:
        """Execute training with Phase 2 monitoring and analysis"""
        
        logger.info("🚀 Starting Phase 2 Enhanced SplatFlow Training")
        logger.info("=" * 80)
        
        # Run diagnostics
        phase2_status = self.run_phase2_diagnostics()
        
        # Setup environment
        setup_environment()
        
        # Display memory info
        memory_info = get_gpu_memory_info()
        logger.info(f"💾 GPU Memory: {memory_info.get('allocated_gb', 0):.2f}GB allocated, "
                   f"{memory_info.get('available_gb', 0):.2f}GB available")
        
        training_start_time = time.time()
        
        try:
            # Create trainer with validated Phase 2 configuration
            trainer = SplatFlowTrainingOrchestrator(self.config)
            
            # Store reference for monitoring
            self.trainer = trainer
            
            # Execute training
            logger.info(f"🎯 Starting training with {self.config['epochs']} epochs")
            training_results = trainer.train()
            
            # Enhanced results with Phase 2 analysis
            enhanced_results = self._analyze_training_results(training_results, training_start_time)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            
            error_results = {
                'success': False,
                'error': str(e),
                'config_name': self.config_name,
                'experiment_name': self.experiment_name,
                'failure_time': datetime.now().isoformat()
            }
            
            # Save error information
            error_path = os.path.join(self.experiment_dir, "training_error.json")
            with open(error_path, 'w') as f:
                json.dump(error_results, f, indent=2)
            
            return error_results
        
        finally:
            # Cleanup
            cleanup_memory()
    
    def _analyze_training_results(self, training_results: Dict, training_start_time: float) -> Dict:
        """Analyze training results with Phase 2 insights"""
        
        total_time = time.time() - training_start_time
        
        # Base results
        enhanced_results = {
            **training_results,
            'experiment_name': self.experiment_name,
            'config_name': self.config_name,
            'total_training_time_hours': total_time / 3600,
            'phase2_features_used': self._count_enabled_features(),
        }
        
        # Save comprehensive results
        results_path = os.path.join(self.experiment_dir, "comprehensive_results.json")
        with open(results_path, 'w') as f:
            json.dump(enhanced_results, f, indent=2, default=str)
        
        logger.info(f"💾 Comprehensive results saved to {results_path}")
        
        return enhanced_results


def main():
    """FIXED: Main function with enhanced Phase 2 configurations"""
    
    parser = argparse.ArgumentParser(description="Phase 2 Enhanced SplatFlow Training")
    
    # Get available configurations
    available_configs = list(EnhancedPhase2Config.get_enhanced_configs().keys())
    
    parser.add_argument("--config", "-c",
                       choices=available_configs,
                       default="stability_2k_phase2",
                       help="Configuration to use")
    
    parser.add_argument("--dataset", "-d",
                       choices=["minimal", "conservative", "extensive"],
                       default="conservative",
                       help="Dataset configuration")
    
    parser.add_argument("--experiment", "-e",
                       type=str,
                       help="Experiment name")
    
    parser.add_argument("--epochs", type=int, default=None,
                       help="Override number of epochs")
    
    parser.add_argument("--run-phase2-tests", action="store_true",
                       help="Run Phase 2 component tests before training")
    
    parser.add_argument("--phase2-diagnostics", action="store_true",
                       help="Run comprehensive Phase 2 diagnostics")
    
    args = parser.parse_args()
    
    print("🌟 PHASE 2 ENHANCED SPLATFLOW TRAINING")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Configuration: {args.config}")
    print(f"Dataset config: {args.dataset}")
    
    # Show Phase 2 status
    phase2_status = get_phase2_status()
    print(f"\n🌟 Phase 2 Status: {phase2_status['available_count']}/{phase2_status['total_features']} features available")
    print(f"Completion: {phase2_status['completion_percentage']:.1f}%")
    
    # Run Phase 2 tests if requested
    if args.run_phase2_tests:
        print("\n🧪 Running Phase 2 component tests...")
        test_results = run_phase2_tests()
        print(f"Tests passed: {test_results['passed']}/{test_results['total']}")
        if test_results['passed'] != test_results['total']:
            print("⚠️ Some Phase 2 tests failed - proceeding with available features")
    
    # Show configuration details
    configs = EnhancedPhase2Config.get_enhanced_configs()
    selected_config = configs[args.config]
    print(f"\n📊 Configuration: {selected_config['name']}")
    print(f"Description: {selected_config['description']}")
    print(f"Memory limit: {selected_config.get('memory_limit_gb', 'N/A')}GB")
    
    try:
        # Create and run trainer
        trainer = Phase2SplatFlowTrainer(
            config_name=args.config,
            dataset_config=args.dataset,
            experiment_name=args.experiment
        )
        
        # Run diagnostics if requested
        if args.phase2_diagnostics:
            print("\n🔍 Running comprehensive Phase 2 diagnostics...")
            trainer.run_phase2_diagnostics()
        
        # Override epochs if specified
        if args.epochs:
            trainer.config['epochs'] = args.epochs
            print(f"📝 Overriding epochs to {args.epochs}")
        
        print(f"\n🚀 Starting training...")
        print(f"Experiment: {trainer.experiment_name}")
        print(f"Phase 2 enabled features: {trainer._count_enabled_features()}")
        
        # Execute training
        results = trainer.train()
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 TRAINING SUMMARY")
        print("=" * 80)
        
        if results.get('success', False):
            print(f"✅ Training completed successfully!")
            print(f"📈 Final loss: {results.get('final_loss', 'N/A'):.4f}")
            print(f"📊 Final perplexity: {results.get('final_perplexity', 'N/A'):.2f}")
            print(f"⏱️ Total time: {results.get('total_training_time_hours', 0):.2f} hours")
            
            if 'phase2_analysis' in results:
                phase2_analysis = results['phase2_analysis']
                print(f"\n🌟 Phase 2 Analysis:")
                print(f"   Features utilized: {phase2_analysis.get('features_utilized', 0)}")
                print(f"   Optimization gain: {phase2_analysis.get('avg_optimization_gain', 0):.3f}")
                print(f"   Stability score: {phase2_analysis.get('stability_score', 0):.3f}")
        else:
            print(f"❌ Training failed: {results.get('error', 'Unknown error')}")
        
        print(f"\n📁 Results saved to: {trainer.experiment_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise
    finally:
        cleanup_memory()


if __name__ == "__main__":
    main()
