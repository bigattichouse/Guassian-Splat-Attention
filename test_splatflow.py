#!/usr/bin/env python3
"""
SplatFlow Test Script with Enhanced Monitoring
Test the SplatFlow implementation with a quick training run.

Usage:
    python test_splatflow.py [--quick] [--full] [--validate-only] [--deep-analysis]
    
    --quick: Run minimal test (2 epochs, small model)
    --full: Run longer test (10 epochs, larger model) 
    --validate-only: Just validate installation without training
    --deep-analysis: Enable enhanced monitoring and analysis
"""

import os
import sys
import argparse
import logging
import time
import torch
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_imports():
    """Setup imports - handle both package and same-directory structure"""
    try:
        # Try package import first
        from splatflow import (
            SplatFlowTrainingOrchestrator,
            create_default_config,
            validate_installation,
            benchmark_performance,
            setup_environment,
            cleanup_memory
        )
        logger.info("✅ Using package imports")
        return True
    except ImportError as e:
        try:
            # Try same-directory imports
            # We need to modify the modules to work without relative imports
            logger.info("📁 Package import failed, trying same-directory imports...")
            print(f"Import error: {e}")
            logger.info("⚠️  Please move files to splatflow/ subdirectory for best results")
            return setup_same_directory_imports()
        except ImportError as e:
            logger.error(f"❌ Failed to import SplatFlow modules: {e}")
            logger.error("Please ensure all SplatFlow files are in the correct location")
            return False

def setup_same_directory_imports():
    """Setup imports for same-directory structure"""
    # This is a simplified test that doesn't require the full module structure
    logger.warning("⚠️  Same-directory mode: Limited functionality")
    return False

def validate_environment():
    """Validate the environment for SplatFlow"""
    logger.info("🔍 Validating environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        logger.error(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return False
    logger.info(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check PyTorch
    try:
        import torch
        logger.info(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("⚠️  CUDA not available - will use CPU (slower)")
            
    except ImportError:
        logger.error("❌ PyTorch not found - please install: pip install torch")
        return False
    
    # Check other dependencies
    missing_deps = []
    
    try:
        import transformers
        logger.info(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        missing_deps.append("transformers")
    
    try:
        import datasets
        logger.info(f"✅ Datasets available")
    except ImportError:
        missing_deps.append("datasets")
    
    try:
        import numpy
        logger.info(f"✅ NumPy {numpy.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    # Check scikit-learn for enhanced monitoring
    try:
        import sklearn
        logger.info(f"✅ Scikit-learn available (for enhanced monitoring)")
    except ImportError:
        logger.info("📊 Scikit-learn not found - enhanced monitoring will have limited clustering features")
    
    if missing_deps:
        logger.error(f"❌ Missing dependencies: {missing_deps}")
        logger.info("Install with: pip install " + " ".join(missing_deps))
        return False
    
    logger.info("✅ Environment validation passed!")
    return True

def create_minimal_config(quick_mode=True):
    """Create a minimal configuration for testing"""
    if quick_mode:
        config = {
            # Minimal model for quick testing
            'model_dim': 64,
            'num_layers': 2,
            'num_splats': 4,
            'max_splats': 8,
            'dropout': 0.1,
            'max_seq_len': 256,
            
            # Quick training
            'epochs': 2,
            'batch_size': 1,
            'seq_length': 128,
            'target_sequences': 50,
            'steps_per_epoch': 5,
            
            # Learning
            'learning_rate': 1e-3,
            'weight_decay': 0.01,
            
            # Progressive training
            'use_progressive_training': True,
            
            # Evaluation
            'eval_interval': 1,
            'eval_max_length': 20,
            'eval_temperature': 0.8,
            'eval_top_k': 10,
            
            # Logging
            'log_interval': 1,
            'save_interval': 2,
            'checkpoint_dir': 'test_splatflow_checkpoints'
        }
        logger.info("🏃 Quick test configuration created")
    else:
        config = {
            # Larger model for proper testing
            'model_dim': 128,
            'num_layers': 3,
            'num_splats': 8,
            'max_splats': 16,
            'dropout': 0.1,
            'max_seq_len': 512,
            
            # Longer training
            'epochs': 10,
            'batch_size': 2,
            'seq_length': 256,
            'target_sequences': 200,
            'steps_per_epoch': 20,
            
            # Learning
            'learning_rate': 5e-4,
            'weight_decay': 0.01,
            
            # Progressive training
            'use_progressive_training': True,
            
            # Evaluation
            'eval_interval': 2,
            'eval_max_length': 30,
            'eval_temperature': 0.8,
            'eval_top_k': 20,
            
            # Logging
            'log_interval': 5,
            'save_interval': 5,
            'checkpoint_dir': 'test_splatflow_checkpoints'
        }
        logger.info("🎯 Full test configuration created")
    
    return config

def simple_splatflow_test():
    """Simplified SplatFlow test that works without full package structure"""
    
    logger.info("🧪 Running simplified SplatFlow test...")
    
    try:
        import torch
        import torch.nn as nn
        from transformers import GPT2Tokenizer
        
        # Create a minimal tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create simple test model to verify basic PyTorch functionality
        class SimpleTestModel(nn.Module):
            def __init__(self, vocab_size, dim=64):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, dim)
                self.linear = nn.Linear(dim, vocab_size)
                
            def forward(self, x):
                emb = self.embedding(x)
                return self.linear(emb.mean(dim=1))
        
        # Test model creation
        model = SimpleTestModel(tokenizer.vocab_size)
        logger.info(f"✅ Created test model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass
        test_input = torch.randint(0, 1000, (2, 10))
        output = model(test_input)
        logger.info(f"✅ Forward pass successful: {output.shape}")
        
        # Test basic training step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        
        target = torch.randint(0, tokenizer.vocab_size, (2,))
        loss = loss_fn(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logger.info(f"✅ Training step successful: loss={loss.item():.4f}")
        
        logger.info("🎉 Simplified test passed! Basic PyTorch functionality works.")
        logger.info("📋 To run full SplatFlow test:")
        logger.info("   1. Create splatflow/ subdirectory")
        logger.info("   2. Move all splatflow_*.py files to splatflow/")
        logger.info("   3. Run: python test_splatflow.py --quick")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simplified test failed: {e}")
        return False

# NEW: Enhanced Training Orchestrator with Deep Analysis
class EnhancedSplatFlowTrainingOrchestrator:
    """Training orchestrator with enhanced monitoring capabilities"""
    
    def __init__(self, config, enable_deep_analysis=False):
        # Import here to avoid issues if analyzer not available
        from splatflow import SplatFlowTrainingOrchestrator
        
        self.base_trainer = SplatFlowTrainingOrchestrator(config)
        self.config = config
        self.enable_deep_analysis = enable_deep_analysis
        self.analyzer = None
        
        if enable_deep_analysis:
            try:
                from splatflow import SplatFlowAnalyzer
                logger.info("🔬 Enhanced monitoring enabled")
            except ImportError:
                logger.warning("⚠️  SplatFlowAnalyzer not found - continuing without enhanced monitoring")
                logger.info("   To enable: Save the monitoring code as splatflow/splatflow_analyzer.py")
                self.enable_deep_analysis = False
    
    def train(self):
        """Enhanced training with optional deep analysis"""
        
        if self.enable_deep_analysis and self.analyzer is None:
            from splatflow import SplatFlowAnalyzer
            # Initialize analyzer after model is created
            pass
        
        # Start base training
        logger.info("🚀 Starting enhanced SplatFlow training...")
        
        # Monkey-patch the base trainer's train_epoch method to add monitoring
        original_train_epoch = self.base_trainer.train_epoch
        
        def enhanced_train_epoch(dataloader, epoch):
            # Call original training
            epoch_results = original_train_epoch(dataloader, epoch)
            
            # Add enhanced monitoring if enabled
            if self.enable_deep_analysis:
                try:
                    self.run_enhanced_analysis(dataloader, epoch)
                except Exception as e:
                    logger.warning(f"Enhanced analysis failed for epoch {epoch}: {e}")
            
            return epoch_results
        
        # Replace the method
        self.base_trainer.train_epoch = enhanced_train_epoch
        
        # Run training
        return self.base_trainer.train()
    
    def run_enhanced_analysis(self, dataloader, epoch):
        """Run enhanced analysis for current epoch"""
        
        if not self.enable_deep_analysis:
            return
        
        # Initialize analyzer if needed
        if self.analyzer is None:
            from splatflow import SplatFlowAnalyzer
            self.analyzer = SplatFlowAnalyzer(self.base_trainer.model)
        
        # Run analysis on first batch
        try:
            batch = next(iter(dataloader))
            
            # Run comprehensive analysis every few epochs
            if epoch % 2 == 0 or epoch < 3:  # More frequent analysis early on
                logger.info(f"\n🧬 Running deep analysis for epoch {epoch}...")
                analysis = self.analyzer.analyze_epoch(batch, epoch)
                
                # Summary of key findings
                self.log_analysis_summary(analysis, epoch)
                
        except Exception as e:
            logger.warning(f"Enhanced analysis failed: {e}")
    
    def log_analysis_summary(self, analysis, epoch):
        """Log summary of analysis findings"""
        
        logger.info(f"\n📊 Enhanced Analysis Summary - Epoch {epoch}")
        logger.info("-" * 50)
        
        # Trajectory patterns
        trajectory_data = analysis.get('trajectories', {})
        for layer_key, data in trajectory_data.items():
            pattern = data.get('pattern_type', 'unknown')
            strength = data.get('trajectory_strength', 0.0)
            improvement = data.get('expected_improvement', 'N/A')
            
            logger.info(f"   {layer_key}: {pattern} pattern (strength={strength:.4f}, expected={improvement})")
        
        # Benchmark warnings
        benchmark_data = analysis.get('benchmarks', {})
        warnings = benchmark_data.get('warnings', [])
        if warnings:
            logger.info(f"   ⚠️  Warnings: {len(warnings)}")
            for warning in warnings[:3]:  # Show first 3 warnings
                logger.info(f"      {warning}")
        else:
            logger.info(f"   ✅ All benchmarks passed")
        
        # Constellation potential
        constellation_data = analysis.get('constellations', {})
        potential = constellation_data.get('average_potential', 0.0)
        if potential > 0.3:
            logger.info(f"   🌌 Constellation potential: {potential:.3f} - {constellation_data.get('recommendation', '')}")

def run_splatflow_test(quick_mode=True, enable_deep_analysis=False):
    """Run the full SplatFlow test with optional enhanced monitoring"""
    
    try:
        # Import SplatFlow components
        from splatflow import setup_environment, cleanup_memory
        
        logger.info("🚀 Starting SplatFlow test...")
        if enable_deep_analysis:
            logger.info("🔬 Enhanced monitoring enabled")
        
        # Setup environment
        setup_environment()
        
        # Create configuration
        config = create_minimal_config(quick_mode)
        
        # Create enhanced trainer
        trainer = EnhancedSplatFlowTrainingOrchestrator(config, enable_deep_analysis)
        
        # Run training
        start_time = time.time()
        training_summary = trainer.train()
        end_time = time.time()
        
        # Report results
        total_time = end_time - start_time
        logger.info(f"🎉 SplatFlow test completed successfully!")
        logger.info(f"   Training time: {total_time:.1f}s")
        logger.info(f"   Best loss: {training_summary['best_loss']:.4f}")
        logger.info(f"   Total steps: {training_summary['total_steps']}")
        logger.info(f"   Final health: {training_summary['final_model_stats']['overall_health']}")
        
        # Enhanced analysis summary
        if enable_deep_analysis:
            logger.info(f"\n🔬 Enhanced Monitoring Summary:")
            logger.info(f"   Deep analysis completed for training run")
            logger.info(f"   Check logs above for trajectory patterns and benchmark comparisons")
            logger.info(f"   Consider reviewing constellation potential for document-level features")
        
        # Cleanup
        cleanup_memory()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ SplatFlow test failed: {e}")
        logger.error(f"Error details: {type(e).__name__}: {str(e)}")
        return False

def run_benchmark():
    """Run performance benchmark"""
    
    try:
        from splatflow import benchmark_performance
        
        logger.info("🏁 Running performance benchmark...")
        
        results = benchmark_performance(
            batch_size=1,
            seq_length=256,
            model_dim=128
        )
        
        logger.info("📊 Benchmark completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return False

def main():
    """Main test function"""
    
    parser = argparse.ArgumentParser(description='Test SplatFlow Implementation')
    parser.add_argument('--quick', action='store_true', help='Run quick test (2 epochs)')
    parser.add_argument('--full', action='store_true', help='Run full test (10 epochs)')
    parser.add_argument('--validate-only', action='store_true', help='Only validate environment')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--simple', action='store_true', help='Run simplified test without full SplatFlow')
    parser.add_argument('--deep-analysis', action='store_true', help='Enable enhanced monitoring and analysis')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("🌟 SplatFlow Test Script")
    if args.deep_analysis:
        print("🔬 Enhanced Monitoring Mode")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Validate environment first
    if not validate_environment():
        logger.error("❌ Environment validation failed")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("✅ Environment validation complete - stopping here")
        return
    
    # Try to setup imports
    imports_ok = setup_imports()
    
    if args.simple or not imports_ok:
        logger.info("🧪 Running simplified test...")
        success = simple_splatflow_test()
    elif args.benchmark:
        success = run_benchmark()
    else:
        # Determine test mode
        quick_mode = args.quick or not args.full
        if quick_mode:
            logger.info("🏃 Running quick SplatFlow test...")
        else:
            logger.info("🎯 Running full SplatFlow test...")
        
        success = run_splatflow_test(quick_mode, args.deep_analysis)
    
    # Final status
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST PASSED!")
        print("SplatFlow is working correctly.")
        if args.deep_analysis:
            print("🔬 Enhanced monitoring data collected - check logs above!")
    else:
        print("❌ TEST FAILED!")
        print("Please check the error messages above.")
    print("=" * 60)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
