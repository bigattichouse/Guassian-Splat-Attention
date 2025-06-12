"""
SplatFlow Core Systems Module
Device management, dataset loading, and core utilities for the SplatFlow attention mechanism.
"""

import torch
import torch.nn as nn
import logging
import gc
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeviceManager:
    """Comprehensive device management for SplatFlow training"""
    
    def __init__(self):
        self.device = self._initialize_device()
        self.is_cuda = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.is_cuda else 0
        
        logger.info(f"🔧 DeviceManager initialized: {self.device}")
        if self.is_cuda:
            logger.info(f"🚀 CUDA devices available: {self.device_count}")
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
    
    def _initialize_device(self) -> torch.device:
        """Initialize the primary computation device"""
        if torch.cuda.is_available():
            # Use first available GPU
            device = torch.device('cuda:0')
            torch.cuda.set_device(device)
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    
    def get_device(self) -> torch.device:
        """Get the primary computation device"""
        return self.device
    
    def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get detailed GPU memory information"""
        if not self.is_cuda:
            return {
                'allocated': 0.0,
                'available': 0.0,
                'total': 0.0,
                'percent_used': 0.0,
                'cached': 0.0
            }
        
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        available = total - allocated
        
        return {
            'allocated': allocated,
            'available': available,
            'total': total,
            'percent_used': (allocated / total) * 100,
            'cached': cached
        }
    
    def cleanup_memory(self):
        """Clean up GPU memory and Python garbage collection"""
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logger.info("🧹 Memory cleanup completed")
    
    def move_to_device(self, tensor_or_model: Union[torch.Tensor, nn.Module]) -> Union[torch.Tensor, nn.Module]:
        """Move tensor or model to the primary device"""
        return tensor_or_model.to(self.device)
    
    def get_memory_summary(self) -> str:
        """Get a formatted memory usage summary"""
        mem_info = self.get_gpu_memory_info()
        return f"GPU Memory: {mem_info['allocated']:.2f}GB allocated, {mem_info['available']:.2f}GB available"


class DatasetManager:
    """Advanced dataset loading and management for SplatFlow training"""
    
    def __init__(self, tokenizer_name: str = 'gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Available datasets for comprehensive training
        self.available_datasets = {
            'wikitext': 'openwebtext',
            'openwebtext': 'openwebtext',
            'bookcorpus': 'bookcorpus',
            'c4': 'c4',
            'pile': 'EleutherAI/pile',
            'common_crawl': 'common_crawl',
            'github_code': 'codeparrot/github-code',
            'arxiv': 'arxiv_dataset',
            'news': 'cnn_dailymail',
            'dialogue': 'microsoft/DialoGPT-medium',
            'qa': 'squad',
            'stories': 'roneneldan/TinyStories',
            'scientific': 'scientific_papers',
            'legal': 'pile-of-law',
            'medical': 'pubmed_qa'
        }
        
        logger.info(f"📚 DatasetManager initialized with tokenizer: {tokenizer_name}")
    
    def load_dataset_by_name(self, dataset_name: str, split: str = 'train', 
                           streaming: bool = False, num_proc: int = 4) -> Dataset:
        """Load a dataset by name with comprehensive error handling"""
        try:
            if dataset_name not in self.available_datasets:
                raise ValueError(f"Dataset '{dataset_name}' not available. Choose from: {list(self.available_datasets.keys())}")
            
            dataset_path = self.available_datasets[dataset_name]
            logger.info(f"📖 Loading dataset: {dataset_name} ({dataset_path})")
            
            # Special handling for different dataset formats
            if dataset_name == 'wikitext':
                dataset = load_dataset(dataset_path, split=split, streaming=streaming)
            elif dataset_name == 'c4':
                dataset = load_dataset(dataset_path, 'en', split=split, streaming=streaming)
            elif dataset_name == 'cnn_dailymail':
                dataset = load_dataset('cnn_dailymail', '3.0.0', split=split, streaming=streaming)
            else:
                dataset = load_dataset(dataset_path, split=split, streaming=streaming)
            
            logger.info(f"✅ Dataset loaded successfully: {len(dataset) if hasattr(dataset, '__len__') else 'streaming'} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset '{dataset_name}': {str(e)}")
            # Fallback to a simple synthetic dataset
            return self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self, size: int = 1000) -> Dataset:
        """Create a synthetic text dataset for testing"""
        logger.info("🔧 Creating synthetic dataset as fallback")
        
        synthetic_texts = []
        templates = [
            "The quick brown fox jumps over the lazy dog.",
            "In a hole in the ground there lived a hobbit.",
            "It was the best of times, it was the worst of times.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold.",
            "The answer to life, the universe, and everything is 42.",
            "In the beginning was the Word, and the Word was with God.",
            "Space: the final frontier. These are the voyages of the starship Enterprise."
        ]
        
        for i in range(size):
            base_text = templates[i % len(templates)]
            # Add some variation
            synthetic_texts.append(f"Sample {i}: {base_text} This is additional content for sequence variation.")
        
        return Dataset.from_dict({'text': synthetic_texts})
    
    def prepare_training_data(self, dataset: Dataset, seq_length: int = 1024, 
                            batch_size: int = 2, num_samples: Optional[int] = None) -> torch.utils.data.DataLoader:
        """Prepare dataset for training with tokenization and batching"""
        
        # Limit dataset size if specified
        if num_samples and len(dataset) > num_samples:
            dataset = dataset.select(range(num_samples))
        
        logger.info(f"🔧 Preparing training data: {len(dataset)} samples, seq_length={seq_length}")
        
        def tokenize_function(examples):
            # Handle different text column names
            text_key = 'text'
            if 'text' not in examples:
                # Try common alternatives
                possible_keys = ['article', 'content', 'document', 'passage', 'story']
                for key in possible_keys:
                    if key in examples:
                        text_key = key
                        break
                else:
                    # If no text found, use first string column
                    for key, value in examples.items():
                        if isinstance(value[0], str):
                            text_key = key
                            break
            
            # Tokenize the texts
            tokenized = self.tokenizer(
                examples[text_key],
                truncation=True,
                padding=True,
                max_length=seq_length,
                return_tensors='pt'
            )
            
            # Prepare labels for language modeling (shifted input_ids)
            tokenized['labels'] = tokenized['input_ids'].clone()
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Set format for PyTorch
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Create DataLoader
        dataloader = torch.utils.data.DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()
        )
        
        logger.info(f"✅ Training data prepared: {len(dataloader)} batches")
        return dataloader


class TensorUtils:
    """Utility functions for tensor operations in SplatFlow"""
    
    @staticmethod
    def safe_tensor_operation(operation, *args, fallback_value=None, **kwargs):
        """Safely execute tensor operations with error handling"""
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            logger.warning(f"⚠️ Tensor operation failed: {str(e)}")
            if fallback_value is not None:
                return fallback_value
            raise
    
    @staticmethod
    def get_tensor_stats(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, float]:
        """Get comprehensive statistics for a tensor"""
        with torch.no_grad():
            return {
                f'{name}_mean': tensor.mean().item(),
                f'{name}_std': tensor.std().item(),
                f'{name}_min': tensor.min().item(),
                f'{name}_max': tensor.max().item(),
                f'{name}_norm': tensor.norm().item(),
                f'{name}_shape': str(tuple(tensor.shape))
            }
    
    @staticmethod
    def check_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, bool]:
        """Check if tensor contains problematic values"""
        return {
            f'{name}_has_nan': torch.isnan(tensor).any().item(),
            f'{name}_has_inf': torch.isinf(tensor).any().item(),
            f'{name}_all_finite': torch.isfinite(tensor).all().item(),
            f'{name}_has_gradients': tensor.requires_grad and tensor.grad is not None
        }
    
    @staticmethod
    def normalize_tensor(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        """Safely normalize tensor along specified dimension"""
        norm = tensor.norm(dim=dim, keepdim=True)
        return tensor / (norm + eps)


class ConfigurationManager:
    """Configuration management for SplatFlow experiments"""
    
    @staticmethod
    def create_default_config() -> Dict[str, Any]:
        """Create default configuration for SplatFlow training"""
        return {
            # Model Architecture
            'model_dim': 512,
            'num_layers': 6,
            'num_splats': 20,
            'max_splats': 64,
            'max_seq_len': 2048,
            'vocab_size': 50257,  # GPT-2 vocab size
            'dropout': 0.1,
            
            # Training Parameters
            'epochs': 50,
            'batch_size': 2,
            'seq_length': 1024,
            'target_sequences': 10000,
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'warmup_steps': 1000,
            'use_progressive_training': True,
            
            # Evaluation Settings
            'eval_interval': 5,
            'eval_max_length': 50,
            'eval_temperature': 0.8,
            'eval_top_k': 50,
            'save_interval': 10,
            
            # Dataset Configuration
            'dataset_name': 'wikitext',
            'dataset_streaming': False,
            'num_workers': 4,
            
            # SplatFlow Specific
            'trajectory_strength': 0.2,
            'splat_radius': 2.0,
            'adaptive_positioning': True,
            'emergency_rescue_threshold': 0.3,
            'health_check_interval': 5,
            
            # Logging and Monitoring
            'log_level': 'INFO',
            'log_interval': 100,
            'save_checkpoint': True,
            'checkpoint_dir': './checkpoints',
            'experiment_name': 'splatflow_experiment'
        }
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix configuration parameters"""
        validated_config = config.copy()
        
        # Ensure minimum values
        validated_config['model_dim'] = max(64, validated_config.get('model_dim', 512))
        validated_config['num_layers'] = max(1, validated_config.get('num_layers', 6))
        validated_config['num_splats'] = max(4, validated_config.get('num_splats', 20))
        validated_config['batch_size'] = max(1, validated_config.get('batch_size', 2))
        validated_config['seq_length'] = max(64, validated_config.get('seq_length', 1024))
        
        # Ensure learning rate is reasonable
        lr = validated_config.get('learning_rate', 3e-4)
        if lr <= 0 or lr > 1.0:
            validated_config['learning_rate'] = 3e-4
            logger.warning(f"⚠️ Invalid learning rate {lr}, reset to 3e-4")
        
        # Ensure trajectory strength is in valid range
        traj_strength = validated_config.get('trajectory_strength', 0.2)
        validated_config['trajectory_strength'] = max(0.0, min(1.0, traj_strength))
        
        return validated_config


# Utility functions for external access
def get_device_manager() -> DeviceManager:
    """Get a global DeviceManager instance"""
    if not hasattr(get_device_manager, '_instance'):
        get_device_manager._instance = DeviceManager()
    return get_device_manager._instance

def get_dataset_manager(tokenizer_name: str = 'gpt2') -> DatasetManager:
    """Get a DatasetManager instance"""
    return DatasetManager(tokenizer_name)

def cleanup_memory():
    """Global memory cleanup function"""
    device_manager = get_device_manager()
    device_manager.cleanup_memory()

def get_gpu_memory_info() -> Dict[str, float]:
    """Global GPU memory info function"""
    device_manager = get_device_manager()
    return device_manager.get_gpu_memory_info()

# Export key functions and classes
__all__ = [
    'DeviceManager',
    'DatasetManager', 
    'TensorUtils',
    'ConfigurationManager',
    'get_device_manager',
    'get_dataset_manager',
    'cleanup_memory',
    'get_gpu_memory_info'
]
