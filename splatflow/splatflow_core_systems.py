"""
SplatFlow Core Systems Module
Device management, dataset loading, and utility functions for the SplatFlow training system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import os
import gc
import random
from typing import Tuple, Optional, Dict, List, Any
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import logging
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

def setup_environment():
    """Setup optimal training environment"""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_gpu_memory_info():
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
            'free': free,
            'percent_used': (allocated / total) * 100
        }
    return None

def safe_tensor_to_scalar(tensor: torch.Tensor, default: float = 0.0) -> float:
    """Safely convert tensor to scalar with proper error handling"""
    try:
        if tensor.numel() == 1:
            return tensor.item()
        elif tensor.numel() > 1:
            return tensor.mean().item()
        else:
            return default
    except Exception:
        return default


class DeviceManager:
    """Centralized device management to prevent tensor mismatch errors"""
    
    @staticmethod
    def get_primary_device():
        """Get the primary device for the model"""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @staticmethod
    def ensure_tensor_device(tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        """Ensure a tensor is on the target device"""
        if tensor.device != target_device:
            return tensor.to(target_device)
        return tensor
    
    @staticmethod
    def safe_cat(tensors: List[torch.Tensor], dim: int = 0, target_device: torch.device = None) -> torch.Tensor:
        """Safely concatenate tensors ensuring device consistency"""
        if not tensors:
            raise ValueError("Cannot concatenate empty tensor list")
        
        if target_device is None:
            target_device = tensors[0].device
        
        aligned_tensors = [DeviceManager.ensure_tensor_device(t, target_device) for t in tensors]
        return torch.cat(aligned_tensors, dim=dim)


class ComprehensiveRealDatasetLoader:
    """Load multiple high-quality real datasets for robust SplatFlow training"""
    
    def __init__(self, tokenizer, seq_length: int = 1024, target_sequences: int = 10000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.target_sequences = target_sequences
        self.all_texts = []
        
        self.min_tokens_full = max(256, seq_length // 4)
        self.min_tokens_padded = max(128, seq_length // 8)
        
        logger.info(f"📚 Loading COMPREHENSIVE REAL DATASETS for SplatFlow training...")
        logger.info(f"   Target: {target_sequences} sequences of {seq_length} tokens")
        
    def load_priority_real_datasets(self):
        """Load datasets in priority order - highest quality first"""
        
        self._load_tier_1_datasets()
        self._load_tier_2_datasets() 
        self._load_tier_3_datasets()
        self._load_tier_4_datasets()
        
        logger.info(f"   📊 Total texts collected: {len(self.all_texts)}")
        
        if len(self.all_texts) < 1000:
            logger.info(f"   🔄 Adding enhanced fallback content...")
            self._add_enhanced_fallback_content()
        
        return self.all_texts
    
    def _load_tier_1_datasets(self):
        """TIER 1: Highest Quality"""
        logger.info(f"🥇 TIER 1: Loading highest quality datasets...")
        
        # TinyStories - Highest quality narratives
        self._safe_load_dataset(
            dataset_name="roneneldan/TinyStories",
            split="train",
            text_field="text",
            target_count=3000,
            streaming=True,
            description="High-quality children's stories"
        )
        
        # SQuAD Context - Wikipedia paragraphs
        self._safe_load_dataset(
            dataset_name="squad",
            split="train", 
            text_field="context",
            target_count=2000,
            streaming=False,
            description="High-quality Wikipedia paragraphs"
        )
        
        # AG News - Professional articles
        self._safe_load_dataset(
            dataset_name="ag_news",
            split="train",
            text_field="text", 
            target_count=1500,
            streaming=False,
            description="News articles"
        )
    
    def _load_tier_2_datasets(self):
        """TIER 2: High Quality, Large Scale"""
        logger.info(f"🥈 TIER 2: Loading large-scale quality datasets...")
        
        # CNN/DailyMail
        self._safe_load_dataset(
            dataset_name="cnn_dailymail",
            config="3.0.0",
            split="train",
            text_field="article",
            target_count=2000,
            streaming=True,
            description="Professional news articles"
        )
        
        # OpenWebText
        self._safe_load_dataset(
            dataset_name="openwebtext",
            split="train",
            text_field="text",
            target_count=2500,
            streaming=True,
            description="Curated web content"
        )
    
    def _load_tier_3_datasets(self):
        """TIER 3: Diverse Content Sources"""
        logger.info(f"🥉 TIER 3: Loading diverse content datasets...")
        
        # BookCorpus
        self._safe_load_dataset(
            dataset_name="bookcorpus",
            split="train",
            text_field="text",
            target_count=2000,
            streaming=True,
            description="Literature excerpts"
        )
    
    def _load_tier_4_datasets(self):
        """TIER 4: Specialized Content"""
        logger.info(f"🏅 TIER 4: Loading specialized datasets...")
        
        # Multi-News
        self._safe_load_dataset(
            dataset_name="multi_news",
            split="train",
            text_field="document",
            target_count=800,
            streaming=False,
            description="Multi-document news"
        )
    
    def _safe_load_dataset(self, dataset_name: str, split: str, text_field: str,
                          target_count: int, streaming: bool = True,
                          config: Optional[str] = None, description: str = "",
                          special_processing: Optional[str] = None):
        """Safely load a dataset with comprehensive error handling"""
        
        logger.info(f"   📖 Loading {dataset_name}...")
        logger.info(f"      Description: {description}")
        
        texts_loaded = 0
        
        try:
            if config:
                dataset = load_dataset(dataset_name, config, split=split, streaming=streaming)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=streaming)
            
            count = 0
            for item in dataset:
                if count >= target_count:
                    break
                
                try:
                    text = item.get(text_field, "")
                    
                    if isinstance(text, list) and len(text) > 0:
                        text = text[0] if len(text) == 1 else " | ".join([t for t in text if t and len(t.strip()) > 20])
                    
                    if isinstance(text, str) and len(text.strip()) > 100:
                        cleaned_text = self._enhanced_clean_text(text)
                        if len(cleaned_text) > 80:
                            self.all_texts.append(cleaned_text)
                            texts_loaded += 1
                            count += 1
                
                except Exception:
                    continue
            
            logger.info(f"      ✅ Loaded {texts_loaded} texts from {dataset_name}")
            
        except Exception as e:
            logger.warning(f"      ❌ Failed to load {dataset_name}: {e}")
    
    def _enhanced_clean_text(self, text: str) -> str:
        """Enhanced text cleaning for better quality"""
        import re
        
        text = ' '.join(text.split())
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        text = text.replace('==', '').replace('||', '')
        text = text.replace('{{', '').replace('}}', '')
        text = text.replace('[edit]', '').replace('[citation needed]', '')
        text = text.replace('&nbsp;', ' ').replace('&amp;', '&')
        text = text.replace('\n\n\n', '\n\n')
        
        text = re.sub(r'\s+([.!?])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        if len(text) > 3000:
            cutoff = text[:3000].rfind('.')
            if cutoff > 1500:
                text = text[:cutoff + 1]
            else:
                text = text[:3000]
        
        return text.strip()
    
    def _add_enhanced_fallback_content(self):
        """Add enhanced fallback content if dataset loading is insufficient"""
        
        logger.info(f"   🔄 Adding enhanced fallback content...")
        
        base_stories = [
            """The scientist carefully observed the chemical reaction as it progressed through distinct phases. 
            First, the solution changed from clear to pale yellow, indicating initial compound formation. 
            Next, small crystals began to precipitate, creating a cloudy appearance. 
            Finally, the mixture settled into distinct layers with crystalline product visible at the bottom. 
            This sequence demonstrated precise control needed in chemical synthesis.""",
            
            """Marcus learned programming by starting with simple concepts and gradually building complexity. 
            He began with basic syntax and variable declarations, understanding fundamental building blocks. 
            Then he moved on to control structures like loops and conditionals. 
            Eventually, he mastered object-oriented principles and could design elegant architectures. 
            The journey required consistent practice and continuous learning."""
        ]
        
        expanded_content = []
        
        for story_idx, base_story in enumerate(base_stories):
            for variation in range(100):
                intro = f"Chapter {variation + 1}: "
                connection = " This sequence demonstrates systematic progression and careful observation. "
                conclusion = f"This represents example {variation + 1} in our comprehensive study."
                
                full_story = intro + base_story + connection + conclusion
                expanded_content.append(full_story)
        
        self.all_texts.extend(expanded_content)
        logger.info(f"      ✅ Added {len(expanded_content)} enhanced fallback texts")


class EnhancedRealDataset(Dataset):
    """Enhanced dataset using comprehensive real text sources"""
    
    def __init__(self, tokenizer, seq_length: int = 1024, target_sequences: int = 10000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.examples = []
        
        self.min_tokens_full = max(256, seq_length // 4)
        self.min_tokens_padded = max(128, seq_length // 8)
        
        logger.info(f"🏭 Creating ENHANCED production dataset with real sources...")
        
        loader = ComprehensiveRealDatasetLoader(tokenizer, seq_length, target_sequences)
        all_texts = loader.load_priority_real_datasets()
        
        logger.info(f"   📊 Processing {len(all_texts)} source texts into sequences...")
        
        self.create_robust_sequences_from_texts(all_texts, target_sequences)
        self.validate_sequences()
        
        logger.info(f"   ✅ Final enhanced dataset: {len(self.examples)} sequences ready")
    
    def create_robust_sequences_from_texts(self, texts: List[str], target_sequences: int):
        """Create sequences with aggressive and robust approach"""
        sequences_created = 0
        tokenization_failures = 0
        
        logger.info(f"   🔧 Creating sequences with enhanced robust approach...")
        
        for text_idx, text in enumerate(texts):
            if sequences_created >= target_sequences:
                break
            
            try:
                tokens = self.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    max_length=self.seq_length * 4,
                    truncation=True
                )
                
                if len(tokens) >= self.min_tokens_full:
                    stride = max(64, self.seq_length // 8)
                    
                    for start_idx in range(0, len(tokens) - self.min_tokens_full + 1, stride):
                        if sequences_created >= target_sequences:
                            break
                        
                        end_idx = min(start_idx + self.seq_length, len(tokens))
                        sequence = tokens[start_idx:end_idx]
                        
                        if len(sequence) < self.seq_length:
                            padding = [self.tokenizer.eos_token_id] * (self.seq_length - len(sequence))
                            sequence.extend(padding)
                        
                        self.examples.append(torch.tensor(sequence, dtype=torch.long))
                        sequences_created += 1
                
                elif len(tokens) >= self.min_tokens_padded:
                    padding = [self.tokenizer.eos_token_id] * (self.seq_length - len(tokens))
                    padded_sequence = tokens + padding
                    
                    self.examples.append(torch.tensor(padded_sequence, dtype=torch.long))
                    sequences_created += 1
                
                if sequences_created > 0 and sequences_created % 1000 == 0:
                    logger.info(f"      Created {sequences_created}/{target_sequences} sequences...")
                    
            except Exception as e:
                tokenization_failures += 1
                if tokenization_failures <= 10:
                    logger.warning(f"      ⚠️  Tokenization error on text {text_idx}: {e}")
                continue
        
        logger.info(f"   ✅ Sequence creation complete: {sequences_created} sequences")
    
    def validate_sequences(self):
        """Enhanced validation with better error handling"""
        logger.info(f"   🔍 Validating {len(self.examples)} sequences...")
        
        valid_sequences = []
        issues_fixed = 0
        
        for i, sequence in enumerate(self.examples):
            try:
                if len(sequence) != self.seq_length:
                    if len(sequence) < self.seq_length:
                        padding = [self.tokenizer.eos_token_id] * (self.seq_length - len(sequence))
                        sequence = torch.cat([sequence, torch.tensor(padding, dtype=torch.long)])
                    else:
                        sequence = sequence[:self.seq_length]
                    issues_fixed += 1
                
                if torch.any(sequence < 0) or torch.any(sequence >= self.tokenizer.vocab_size):
                    sequence = torch.clamp(sequence, 0, self.tokenizer.vocab_size - 1)
                    issues_fixed += 1
                
                sequence = sequence.to(torch.long)
                valid_sequences.append(sequence)
                
            except Exception as e:
                logger.warning(f"      ⚠️  Validation error on sequence {i}: {e}")
                continue
        
        self.examples = valid_sequences
        
        if issues_fixed > 0:
            logger.info(f"   🔧 Fixed {issues_fixed} sequence issues")
        
        logger.info(f"   ✅ Validation complete: {len(self.examples)} valid sequences")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
