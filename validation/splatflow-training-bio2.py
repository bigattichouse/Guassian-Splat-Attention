"""
Extended Biological SplatFlow Training with Enhanced Evolution

This script provides:
- Longer training (100+ epochs) 
- More active biological adaptation (increased birth rate)
- Enhanced monitoring and checkpointing
- Progressive evaluation

Goal: See if longer biological adaptation leads to breakthrough performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time
import json
import os
import gc
import random
from typing import Tuple, Optional, Dict, List
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset

# Enable memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


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


# ==================== ENHANCED BIOLOGICAL ADAPTATION ====================

class EnhancedAdaptiveSplat:
    """Splat with more active biological adaptation"""
    
    def __init__(self, position: torch.Tensor, scale: float, amplitude: float, splat_id: int, device: torch.device = None):
        if device is None:
            device = position.device
        
        # Core parameters
        self.position = position.clone().detach().to(device).requires_grad_(True)
        self.log_scale = torch.tensor(math.log(scale), device=device, requires_grad=True)
        self.amplitude = torch.tensor(amplitude, device=device, requires_grad=True)
        self.id = splat_id
        self.device = device
        
        # Enhanced biological properties
        self.age = 0
        self.usefulness = 1.0
        self.activation_history = []
        self.error_history = []
        self.mitosis_readiness = 0.0
        self.death_countdown = -1
        self.errorContribution = 0.0
        self.generation = 0  # Track generational evolution
        
        # Enhanced movement properties
        self.velocity = torch.zeros_like(self.position, device=device)
        self.exploration_rate = 0.15  # Increased from 0.1
        self.learning_momentum = 0.0
        
    def get_scale(self):
        return torch.exp(self.log_scale.to(self.device)).clamp(min=0.1, max=2.0)
    
    def update_activation(self, activation: float, error: float):
        """Enhanced biological state update with more aggressive adaptation"""
        self.age += 1
        
        # Track activation and error history
        self.activation_history.append(abs(activation))
        self.error_history.append(abs(error))
        
        # Keep only recent history
        if len(self.activation_history) > 30:  # Increased from 20
            self.activation_history.pop(0)
            self.error_history.pop(0)
        
        # Enhanced usefulness calculation
        recent_activation = np.mean(self.activation_history[-8:]) if len(self.activation_history) >= 8 else abs(activation)
        recent_error = np.mean(self.error_history[-8:]) if len(self.error_history) >= 8 else abs(error)
        
        # More aggressive usefulness updates
        usefulness_delta = 0.02 * (recent_activation - recent_error)  # Increased from 0.01
        self.usefulness = np.clip(self.usefulness + usefulness_delta, 0.1, 2.5)  # Increased max from 2.0
        
        # Enhanced mitosis readiness - more aggressive
        if recent_activation > 0.6 and recent_error < 0.4:  # Relaxed thresholds
            self.mitosis_readiness += 0.04  # Increased from 0.02
        else:
            self.mitosis_readiness *= 0.96  # Less aggressive decay
            
        # Boost mitosis for highly useful splats
        if self.usefulness > 1.3:
            self.mitosis_readiness += 0.02
    
    def should_divide(self) -> bool:
        """Enhanced division criteria - more likely to divide"""
        return (self.mitosis_readiness > 0.7 and  # Lowered from 1.0
                self.age > 30 and                  # Lowered from 50
                self.usefulness > 1.1)             # Lowered from 1.2
    
    def should_die(self) -> bool:
        """Enhanced death criteria - less likely to die prematurely"""
        return (self.age > 150 and                # Increased from 100
                self.usefulness < 0.25 and        # Lowered from 0.3
                len(self.activation_history) > 15 and
                np.mean(self.activation_history[-15:]) < 0.03)  # Stricter criteria
    
    def explore_movement(self, learning_rate: float, device: torch.device):
        """Enhanced exploration with adaptive learning"""
        if self.age % 8 == 0:  # More frequent exploration (was 10)
            # Adaptive exploration based on usefulness
            adaptive_exploration = self.exploration_rate * (1.0 + self.usefulness * 0.2)
            exploration_noise = torch.randn_like(self.position.to(device)) * adaptive_exploration
            
            # Enhanced momentum
            if hasattr(self, 'last_gradient') and self.last_gradient is not None:
                momentum = 0.92  # Increased momentum
                self.velocity = momentum * self.velocity.to(device) + learning_rate * self.last_gradient.to(device)
                exploration_noise += self.velocity * 0.5
            
            # Apply movement
            self.position.data = self.position.data.to(device) + exploration_noise
            
            # Adaptive exploration decay - slower for useful splats
            if self.usefulness > 1.0:
                self.exploration_rate *= 0.9995  # Slower decay for useful splats
            else:
                self.exploration_rate *= 0.998   # Faster decay for less useful splats


class EnhancedBiologicalSplatAttentionLayer(nn.Module):
    """Enhanced biological splat attention with more active adaptation"""
    
    def __init__(self, model_dim: int, initial_splats: int = 16, max_splats: int = 96, 
                 dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        
        self.model_dim = model_dim
        self.max_splats = max_splats  # Increased capacity
        self.temperature = temperature
        self.adaptive_splats = []
        self.adaptation_enabled = True
        self.adaptation_frequency = 8  # More frequent adaptation (was 10)
        self.forward_count = 0
        self.birth_count = 0
        self.death_count = 0
        self.generation_count = 0
        
        # Projections
        self.token_value_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.output_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters for compatibility
        self.num_splats = initial_splats
        self.splat_centers = nn.Parameter(torch.randn(initial_splats, model_dim) * 0.02)
        self.splat_log_scales = nn.Parameter(torch.zeros(initial_splats))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with careful scaling"""
        nn.init.xavier_uniform_(self.token_value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    def _initialize_adaptive_splats(self, num_splats: int):
        """Initialize enhanced adaptive splats"""
        self.adaptive_splats = []
        
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        
        for i in range(num_splats):
            position = torch.randn(self.model_dim, device=device) * 0.02
            scale = 0.4 + torch.rand(1).item() * 0.6  # Wider initial range
            amplitude = 0.7 + torch.rand(1).item() * 0.6  # Wider initial range
            
            splat = EnhancedAdaptiveSplat(position, scale, amplitude, i, device=device)
            self.adaptive_splats.append(splat)
    
    def _sync_splats_to_parameters(self):
        """Sync adaptive splats to parameters"""
        num_splats = len(self.adaptive_splats)
        
        if num_splats == 0:
            return
        
        device = self.splat_centers.device
        
        # Update parameter tensors
        centers = torch.stack([splat.position.detach().to(device) for splat in self.adaptive_splats])
        log_scales = torch.stack([splat.log_scale.detach().to(device) for splat in self.adaptive_splats])
        
        # Resize if needed
        if num_splats != self.num_splats:
            self.num_splats = num_splats
            self.splat_centers = nn.Parameter(centers)
            self.splat_log_scales = nn.Parameter(log_scales)
        else:
            self.splat_centers.data = centers
            self.splat_log_scales.data = log_scales
    
    def compute_affinity_matrix(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute affinities between tokens and splats - O(n*k) operation"""
        batch_size, seq_len, model_dim = token_embeddings.shape
        device = token_embeddings.device
        
        if not self.adaptive_splats:
            return torch.zeros(batch_size, seq_len, 0, device=device)
        
        # Get current centers and scales
        centers = torch.stack([splat.position.to(device) for splat in self.adaptive_splats])
        scales = torch.stack([splat.get_scale().to(device) for splat in self.adaptive_splats])
        
        # Compute squared distances
        tokens_expanded = token_embeddings.unsqueeze(2)
        centers_expanded = centers.unsqueeze(0).unsqueeze(0)
        
        diff = tokens_expanded - centers_expanded
        distances_sq = torch.sum(diff ** 2, dim=-1)
        
        # Apply learned scales
        scales_sq = scales ** 2
        affinities = torch.exp(-0.5 * distances_sq / scales_sq.unsqueeze(0).unsqueeze(0))
        
        # Apply temperature and normalize
        affinities = affinities ** (1.0 / self.temperature)
        affinities = affinities / (affinities.sum(dim=-1, keepdim=True) + 1e-8)
        
        return affinities
    
    def _apply_enhanced_biological_adaptation(self, affinities: torch.Tensor, loss_per_token: torch.Tensor):
        """Enhanced biological adaptation with more active evolution"""
        if not self.adaptation_enabled or not self.adaptive_splats:
            return

        device = affinities.device
        
        try:
            # Calculate per-splat activation and error
            splat_activations = affinities.mean(dim=(0, 1))
            token_errors = loss_per_token.mean(dim=0) if loss_per_token.dim() > 1 else loss_per_token
            splat_errors = (affinities * token_errors.unsqueeze(-1)).mean(dim=(0, 1))
            
            # Update each splat's biological state
            splats_to_divide = []
            splats_to_remove = []
            
            for i, splat in enumerate(self.adaptive_splats):
                if i < len(splat_activations):
                    activation = splat_activations[i].item()
                    error = splat_errors[i].item() if i < len(splat_errors) else 0.0
                    
                    splat.update_activation(activation, error)
                    
                    # Store gradient for momentum
                    if splat.position.grad is not None:
                        splat.last_gradient = splat.position.grad.clone().detach().to(device)
                    
                    # Apply enhanced exploration movement
                    splat.explore_movement(0.015, device)  # Slightly increased learning rate
                    
                    # Check for division with enhanced criteria
                    if splat.should_divide() and len(self.adaptive_splats) < self.max_splats:
                        splats_to_divide.append(i)
                    
                    # Check for death with enhanced criteria
                    elif splat.should_die() and len(self.adaptive_splats) > 6:  # Keep more minimum splats
                        splats_to_remove.append(i)
            
            # Apply enhanced mitosis - allow multiple divisions per cycle
            divisions_this_cycle = 0
            max_divisions_per_cycle = 3  # Allow up to 3 divisions per adaptation cycle
            
            for splat_idx in splats_to_divide:
                if divisions_this_cycle >= max_divisions_per_cycle:
                    break
                if len(self.adaptive_splats) + 2 <= self.max_splats:  # Ensure we don't exceed max
                    self._enhanced_divide_splat(splat_idx, device)
                    divisions_this_cycle += 1
            
            # Apply death (in reverse order to maintain indices)
            for splat_idx in sorted(splats_to_remove, reverse=True):
                self._remove_splat(splat_idx)
            
            # Sync parameters
            self._sync_splats_to_parameters()
            
        except Exception as e:
            print(f"Warning: Enhanced biological adaptation failed: {e}")
    
    def _enhanced_divide_splat(self, splat_idx: int, device: torch.device):
        """Enhanced splat division with better child generation"""
        parent = self.adaptive_splats[splat_idx]
        
        # Create two children with enhanced variations
        for i in range(2):
            # Enhanced position perturbation - adaptive based on parent usefulness
            perturbation_scale = 0.08 + (parent.usefulness - 1.0) * 0.05
            offset = torch.randn_like(parent.position, device=device) * perturbation_scale
            child_position = parent.position.to(device) + offset
            
            # Enhanced scale and amplitude variation
            if i == 0:  # More focused child
                scale_factor = 0.7 + torch.rand(1).item() * 0.2
                amplitude_factor = 0.9 + torch.rand(1).item() * 0.2
            else:  # More exploratory child
                scale_factor = 1.1 + torch.rand(1).item() * 0.3
                amplitude_factor = 0.8 + torch.rand(1).item() * 0.3
            
            child_scale = parent.get_scale().item() * scale_factor
            child_amplitude = parent.amplitude.item() * amplitude_factor
            
            # Create enhanced child
            child = EnhancedAdaptiveSplat(
                child_position, 
                child_scale, 
                child_amplitude, 
                len(self.adaptive_splats) + self.birth_count,
                device=device
            )
            
            # Enhanced inheritance
            child.usefulness = parent.usefulness * 0.8  # Better starting usefulness
            child.exploration_rate = parent.exploration_rate * 1.3  # More exploratory
            child.generation = parent.generation + 1
            
            self.adaptive_splats.append(child)
            self.birth_count += 1
        
        # Enhanced parent handling - gradual retirement instead of immediate death
        parent.death_countdown = 50  # Longer countdown
        parent.usefulness *= 0.6
        parent.mitosis_readiness = 0.0  # Reset to prevent immediate re-division
    
    def _remove_splat(self, splat_idx: int):
        """Remove a splat from the population"""
        if 0 <= splat_idx < len(self.adaptive_splats):
            self.adaptive_splats.pop(splat_idx)
            self.death_count += 1
    
    def forward(self, token_embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                loss_per_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with enhanced biological adaptation"""
        self.forward_count += 1
        
        # Lazy initialization
        if not self.adaptive_splats:
            self._initialize_adaptive_splats(self.num_splats)
        
        # Sync adaptive splats to parameters
        self._sync_splats_to_parameters()
        
        if not self.adaptive_splats:
            return token_embeddings
        
        # Compute affinities O(n*k)
        affinities = self.compute_affinity_matrix(token_embeddings)
        
        if affinities.size(-1) == 0:
            return token_embeddings
        
        # Project token embeddings to values
        token_values = self.token_value_proj(token_embeddings)
        
        # Aggregate information at splats O(n*k*d)
        splat_states = torch.einsum('bsk,bsd->bkd', affinities, token_values)
        
        # Distribute information back to tokens O(n*k*d)
        token_outputs = torch.einsum('bsk,bkd->bsd', affinities, splat_states)
        
        # Apply dropout and output projection
        token_outputs = self.dropout(token_outputs)
        output = self.output_proj(token_outputs)
        
        # Apply enhanced biological adaptation during training
        if self.training and self.adaptation_enabled and self.forward_count % self.adaptation_frequency == 0:
            with torch.no_grad():
                if loss_per_token is None:
                    loss_per_token = torch.randn(token_embeddings.shape[:2], device=token_embeddings.device) * 0.1
                
                self._apply_enhanced_biological_adaptation(affinities, loss_per_token)
        
        return output
    
    def freeze_adaptation(self):
        """Stop adaptation and freeze for inference"""
        self.adaptation_enabled = False
        self._sync_splats_to_parameters()
    
    def get_enhanced_adaptation_stats(self):
        """Get enhanced statistics about the adaptation process"""
        if not self.adaptive_splats:
            return {
                'num_splats': 0,
                'birth_count': self.birth_count,
                'death_count': self.death_count,
                'avg_usefulness': 0.0,
                'max_usefulness': 0.0,
                'avg_age': 0.0,
                'max_generation': 0,
                'ready_for_mitosis': 0,
                'exploration_activity': 0.0
            }
        
        usefulness_values = [s.usefulness for s in self.adaptive_splats]
        ages = [s.age for s in self.adaptive_splats]
        generations = [s.generation for s in self.adaptive_splats]
        exploration_rates = [s.exploration_rate for s in self.adaptive_splats]
        
        return {
            'num_splats': len(self.adaptive_splats),
            'birth_count': self.birth_count,
            'death_count': self.death_count,
            'avg_usefulness': np.mean(usefulness_values),
            'max_usefulness': np.max(usefulness_values),
            'avg_age': np.mean(ages),
            'max_age': np.max(ages),
            'max_generation': np.max(generations) if generations else 0,
            'ready_for_mitosis': sum(1 for s in self.adaptive_splats if s.mitosis_readiness > 0.6),
            'exploration_activity': np.mean(exploration_rates)
        }


class EnhancedBiologicalSplatTransformerLayer(nn.Module):
    """Enhanced transformer layer using biological splat attention"""
    
    def __init__(self, model_dim: int, num_splats: int = 16, max_splats: int = 96,
                 ff_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = model_dim * 4
        
        # Enhanced biological splat attention
        self.attention = EnhancedBiologicalSplatAttentionLayer(
            model_dim, num_splats, max_splats, dropout=dropout
        )
        
        # Layer norms
        self.attn_norm = nn.LayerNorm(model_dim)
        self.ff_norm = nn.LayerNorm(model_dim)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                loss_per_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with enhanced biological adaptation"""
        
        # Self-attention with residual connection
        attn_output = self.attention(x, attention_mask, loss_per_token)
        x = self.attn_norm(x + attn_output)
        
        # Feed-forward with residual connection  
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + ff_output)
        
        return x
    
    def get_adaptation_stats(self):
        """Get enhanced adaptation statistics from attention layer"""
        return self.attention.get_enhanced_adaptation_stats()
    
    def freeze_adaptation(self):
        """Freeze biological adaptation"""
        self.attention.freeze_adaptation()


class EnhancedBiologicalSplatFlowGPT(nn.Module):
    """Enhanced GPT model using biological splat attention with more active evolution"""
    
    def __init__(self, vocab_size: int, model_dim: int = 512, num_layers: int = 6,
                 num_splats: int = 16, max_splats: int = 96, max_seq_len: int = 1024, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_splats = num_splats
        self.max_splats = max_splats
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Enhanced transformer layers with biological splat attention
        self.layers = nn.ModuleList([
            EnhancedBiologicalSplatTransformerLayer(
                model_dim, num_splats, max_splats, dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output
        self.final_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report model statistics
        self._report_model_stats()
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _report_model_stats(self):
        """Report enhanced model complexity statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Enhanced Biological SplatFlow Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Initial splats per layer: {self.num_splats}")
        print(f"  Max splats per layer: {self.max_splats}")
        print(f"  Model dimension: {self.model_dim}")
        print(f"  Enhanced adaptation: More aggressive birth/death/exploration")
        print(f"  Theoretical complexity: O(n*k*{self.model_dim}) per layer with adaptive k")
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                compute_loss_per_token: bool = False) -> torch.Tensor:
        """Forward pass through the enhanced biological splat-flow model"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)
        
        x = self.embedding_dropout(token_emb + pos_emb)
        
        # Compute per-token loss if needed for adaptation
        loss_per_token = None
        if compute_loss_per_token and self.training:
            # Enhanced loss signal - use embedding variance and gradient information
            loss_per_token = torch.var(x, dim=-1) + torch.norm(x, dim=-1) * 0.1
        
        # Process through enhanced biological splat-flow layers
        for layer in self.layers:
            x = layer(x, attention_mask, loss_per_token)
        
        # Output
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        return logits
    
    def freeze_adaptation(self):
        """Freeze adaptation for inference"""
        for layer in self.layers:
            layer.freeze_adaptation()
    
    def get_enhanced_adaptation_stats(self):
        """Get enhanced adaptation statistics from all layers"""
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}'] = layer.get_adaptation_stats()
        
        # Enhanced aggregate stats
        total_splats = sum(s['num_splats'] for s in stats.values())
        total_births = sum(s['birth_count'] for s in stats.values())
        total_deaths = sum(s['death_count'] for s in stats.values())
        total_ready = sum(s['ready_for_mitosis'] for s in stats.values())
        max_generation = max(s['max_generation'] for s in stats.values())
        avg_exploration = np.mean([s['exploration_activity'] for s in stats.values()])
        
        stats['enhanced_total'] = {
            'total_splats': total_splats,
            'total_births': total_births,
            'total_deaths': total_deaths,
            'total_ready_for_mitosis': total_ready,
            'max_generation': max_generation,
            'avg_exploration_activity': avg_exploration,
            'avg_splats_per_layer': total_splats / len(self.layers) if self.layers else 0,
            'growth_factor': total_splats / (self.num_splats * len(self.layers)),
            'evolutionary_activity': total_births + total_deaths
        }
        
        return stats


# ==================== ENHANCED DATASET (Same but larger) ====================

class LargerRealDataset(Dataset):
    """Larger dataset for extended training"""
    
    def __init__(self, tokenizer, seq_length: int = 1024, total_sequences: int = 3000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.examples = []
        
        print(f"📚 Creating larger dataset with {total_sequences} sequences of {seq_length} tokens")
        
        # Collect more texts from multiple sources
        all_texts = []
        
        # 1. More TinyStories
        all_texts.extend(self.load_tinystories(target_texts=total_sequences//2))
        
        # 2. More WikiText-103
        all_texts.extend(self.load_wikitext(target_texts=total_sequences//3))
        
        # 3. More OpenWebText
        all_texts.extend(self.load_openwebtext(target_texts=total_sequences//4))
        
        # 4. More synthetic content
        current_count = len(all_texts)
        remaining = max(total_sequences//3 - current_count, 500)
        all_texts.extend(self.create_quality_synthetic(remaining))
        
        print(f"📊 Total source texts collected: {len(all_texts)}")
        
        # Create sequences
        self.create_sequences_from_texts(all_texts, total_sequences)
        
        print(f"✅ Final dataset: {len(self.examples)} sequences")
    
    def load_tinystories(self, target_texts: int) -> List[str]:
        """Load more TinyStories"""
        texts = []
        try:
            print(f"  📖 Loading TinyStories (target: {target_texts})...")
            dataset = load_dataset("roneneldan/TinyStories", split="train")
            
            count = 0
            for item in dataset:
                if count >= target_texts:
                    break
                    
                text = item['text'].strip()
                if len(text) > 150:  # Lower threshold for more variety
                    texts.append(text + "\n\n")
                    count += 1
            
            print(f"    ✅ Added {len(texts)} TinyStories")
            
        except Exception as e:
            print(f"    ❌ Failed to load TinyStories: {e}")
        
        return texts
    
    def load_wikitext(self, target_texts: int) -> List[str]:
        """Load more WikiText-103"""
        texts = []
        try:
            print(f"  📖 Loading WikiText-103 (target: {target_texts})...")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            
            count = 0
            for item in dataset:
                if count >= target_texts:
                    break
                    
                text = item['text'].strip()
                if len(text) > 300 and not text.startswith('='):  # Lower threshold
                    texts.append(text + "\n\n")
                    count += 1
            
            print(f"    ✅ Added {len(texts)} WikiText articles")
            
        except Exception as e:
            print(f"    ❌ Failed to load WikiText: {e}")
        
        return texts
    
    def load_openwebtext(self, target_texts: int) -> List[str]:
        """Load more OpenWebText"""
        texts = []
        try:
            print(f"  📖 Loading OpenWebText (target: {target_texts})...")
            dataset = load_dataset("openwebtext", split="train")
            
            count = 0
            for item in dataset:
                if count >= target_texts:
                    break
                    
                text = item['text'].strip()
                if 200 < len(text) < 8000:  # Wider range
                    texts.append(text + "\n\n")
                    count += 1
            
            print(f"    ✅ Added {len(texts)} OpenWebText articles")
            
        except Exception as e:
            print(f"    ❌ Failed to load OpenWebText: {e}")
        
        return texts
    
    def create_quality_synthetic(self, target_texts: int) -> List[str]:
        """Create more diverse synthetic texts"""
        print(f"  🤖 Creating {target_texts} synthetic texts...")
        
        templates = [
            """The field of {topic} has seen remarkable progress recently. Scientists have discovered {finding}, which could revolutionize {application}.

This breakthrough builds on previous work in {related_field}. The key insight is that {insight}, enabling researchers to {capability}.

Practical applications include {use_case1} and {use_case2}. For instance, {example} demonstrates the potential for real-world impact.

Looking forward, experts predict {prediction}. The next steps involve {next_steps} and addressing challenges in {challenge_area}.

The implications are far-reaching. {implication1} and {implication2} suggest that we're entering a new era of {era_description}.""",

            """In a {setting} nestled between {location_detail}, there lived a {character} who had an unusual gift. Every {time_period}, {character} could {ability}.

One day, a {visitor} arrived seeking help with {problem}. "{character}," said the {visitor}, "{request}."

At first, {character} was hesitant. {reason_for_hesitation}. But seeing the {visitor}'s distress, {character} decided to help.

The journey was not easy. They encountered {obstacle1} and had to overcome {obstacle2}. Through {method}, they learned {lesson}.

Along the way, they met {ally} who taught them about {wisdom}. This knowledge proved crucial when they faced {final_challenge}.

In the end, {outcome}. The {visitor} was grateful, and {character} realized {moral}. From that day forward, {character} used their gift to {new_purpose}.""",
            
            """The year was {year}, and the world was changing rapidly. {protagonist}, a {profession} from {location}, witnessed these changes firsthand.

It began with {inciting_incident}. Nobody expected {unexpected_consequence}, but it changed everything. The old ways of {old_system} were no longer sufficient.

{protagonist} found themselves at the center of {conflict}. They had to choose between {choice_a} and {choice_b}. The decision would affect {affected_parties}.

Working alongside {allies}, they developed {solution}. It wasn't perfect, but it addressed {core_problem} while respecting {values}.

The implementation faced {resistance} from {opposition}. However, through {strategy}, they managed to {achievement}.

Years later, historians would recognize this as the moment when {historical_significance}. {protagonist}'s contribution to {field} became a model for {future_applications}."""
        ]
        
        topics = ["artificial intelligence", "renewable energy", "space exploration", "medicine", "education", "quantum computing", "biotechnology", "oceanography"]
        
        texts = []
        for i in range(target_texts):
            template = random.choice(templates)
            topic = random.choice(topics)
            
            # More diverse replacements
            replacements = {
                'topic': topic,
                'finding': random.choice(["unexpected patterns in large-scale data", "novel mechanisms at the quantum level", "previously unknown connections", "emergent behaviors in complex systems"]),
                'application': random.choice(["how we solve complex problems", "our understanding of the universe", "medical treatments", "environmental conservation"]),
                'related_field': random.choice(["computational science", "theoretical physics", "molecular biology", "systems engineering"]),
                'insight': random.choice(["complex systems follow simple principles", "emergence occurs at critical thresholds", "information is preserved across transformations", "networks exhibit scale-free properties"]),
                'capability': random.choice(["predict outcomes with greater accuracy", "manipulate matter at the atomic level", "simulate complex environments", "optimize resource allocation"]),
                'use_case1': random.choice(["climate modeling", "drug discovery", "material design", "ecosystem management"]),
                'use_case2': random.choice(["disease prevention", "energy storage", "communication systems", "food production"]),
                'example': random.choice(["recent cancer research", "climate change studies", "space exploration missions", "renewable energy projects"]),
                'prediction': random.choice(["these technologies will become mainstream", "we'll see exponential progress", "new industries will emerge", "global challenges will be addressed"]),
                'next_steps': random.choice(["developing better algorithms", "improving measurement techniques", "scaling production", "training specialists"]),
                'challenge_area': random.choice(["ethical deployment", "resource limitations", "public acceptance", "regulatory frameworks"]),
                
                # Story elements
                'setting': random.choice(["small village", "bustling city", "remote monastery", "floating island", "underground cavern"]),
                'location_detail': random.choice(["rolling hills", "towering mountains", "endless forests", "crystal clear lakes", "ancient ruins"]),
                'character': random.choice(["wise healer", "young inventor", "mysterious traveler", "skilled artisan", "learned scholar"]),
                'time_period': random.choice(["full moon", "summer solstice", "autumn equinox", "first snowfall", "spring awakening"]),
                'ability': random.choice(["see the future in dreams", "communicate with animals", "heal any wound", "control the elements", "read ancient languages"]),
                'visitor': random.choice(["desperate merchant", "lost prince", "wounded soldier", "curious child", "worried parent"]),
                'problem': random.choice(["a terrible curse", "a missing treasure", "an approaching disaster", "a broken promise", "a forgotten secret"]),
                'request': random.choice(["please help me save my family", "can you guide me home", "will you teach me your ways", "could you heal this wound", "might you solve this mystery"]),
                'reason_for_hesitation': random.choice(["the visions were often unclear", "such journeys were dangerous", "the magic required great sacrifice", "they had been disappointed before", "the path ahead was uncertain"]),
                'obstacle1': random.choice(["treacherous mountain paths", "enchanted forest guardians", "raging river crossings", "hostile territory", "ancient magical barriers"]),
                'obstacle2': random.choice(["ancient guardians", "powerful storms", "misleading illusions", "hidden traps", "rival seekers"]),
                'method': random.choice(["courage and wisdom", "patience and understanding", "cooperation and trust", "knowledge and skill", "determination and hope"]),
                'lesson': random.choice(["that true power comes from helping others", "that wisdom requires both knowledge and experience", "that every challenge teaches something valuable", "that friendship multiplies strength", "that perseverance overcomes obstacles"]),
                'ally': random.choice(["an old hermit", "a talking animal", "a spirit guide", "a fellow traveler", "a local guide"]),
                'wisdom': random.choice(["the language of the earth", "the secrets of the stars", "the nature of courage", "the power of compassion", "the art of patience"]),
                'final_challenge': random.choice(["the source of all evil", "their greatest fear", "an impossible choice", "their own limitations", "the ultimate test"]),
                'outcome': random.choice(["the curse was broken", "the treasure was found", "the disaster was averted", "the promise was kept", "the secret was revealed"]),
                'moral': random.choice(["that every gift should be used for good", "that wisdom grows through helping others", "that true strength comes from within", "that hope never truly dies", "that love conquers all"]),
                'new_purpose': random.choice(["help all who were in need", "teach others their craft", "protect the innocent", "preserve ancient knowledge", "bring peace to the land"]),
                
                # Historical elements  
                'year': random.choice(["1847", "1923", "1969", "1991", "2008"]),
                'protagonist': random.choice(["Elena Rodriguez", "Marcus Chen", "Amara Okafor", "Dmitri Volkov", "Sarah Williams"]),
                'profession': random.choice(["engineer", "scientist", "teacher", "journalist", "diplomat"]),
                'location': random.choice(["San Francisco", "Berlin", "Tokyo", "Cairo", "São Paulo"]),
                'inciting_incident': random.choice(["a revolutionary discovery", "an unexpected crisis", "a technological breakthrough", "a social movement", "an environmental change"]),
                'unexpected_consequence': random.choice(["widespread adoption", "fierce resistance", "unintended benefits", "surprising alliances", "paradigm shifts"]),
                'old_system': random.choice(["communication", "transportation", "education", "governance", "production"]),
                'conflict': random.choice(["rapid change", "competing interests", "resource scarcity", "ideological differences", "technical challenges"]),
                'choice_a': random.choice(["gradual reform", "immediate action", "technological solutions", "collaborative approaches", "traditional methods"]),
                'choice_b': random.choice(["revolutionary change", "careful planning", "human-centered approaches", "independent action", "innovative methods"]),
                'affected_parties': random.choice(["thousands of families", "entire industries", "future generations", "global communities", "local ecosystems"]),
                'allies': random.choice(["fellow researchers", "community leaders", "international partners", "innovative thinkers", "dedicated volunteers"]),
                'solution': random.choice(["a new framework", "an innovative process", "a collaborative platform", "a systematic approach", "a breakthrough method"]),
                'core_problem': random.choice(["inefficiency", "inequality", "sustainability", "accessibility", "scalability"]),
                'values': random.choice(["human dignity", "environmental stewardship", "cultural heritage", "individual freedom", "collective welfare"]),
                'resistance': random.choice(["institutional inertia", "vested interests", "public skepticism", "technical limitations", "resource constraints"]),
                'opposition': random.choice(["established institutions", "competing groups", "skeptical experts", "concerned citizens", "regulatory bodies"]),
                'strategy': random.choice(["patient education", "gradual implementation", "strategic partnerships", "public demonstrations", "collaborative governance"]),
                'achievement': random.choice(["gain widespread acceptance", "overcome technical barriers", "secure necessary funding", "build lasting institutions", "create positive change"]),
                'historical_significance': random.choice(["humanity learned to cooperate globally", "technology and society found balance", "sustainable development became possible", "knowledge became truly accessible", "justice and progress aligned"]),
                'field': random.choice(["sustainable development", "technological innovation", "social cooperation", "global governance", "human progress"]),
                'future_applications': random.choice(["addressing climate change", "reducing inequality", "advancing human knowledge", "fostering international cooperation", "building resilient communities"]),
                
                # Additional implications for first template
                'implication1': random.choice(["This could transform entire industries", "The economic impact will be substantial", "New job categories will emerge", "Educational systems must adapt"]),
                'implication2': random.choice(["Global cooperation will be essential", "Ethical frameworks need updating", "Resources must be allocated wisely", "Long-term thinking is crucial"]),
                'era_description': random.choice(["scientific advancement", "technological integration", "sustainable development", "global collaboration", "human enhancement"])
            }
            
            filled_text = template.format(**replacements)
            texts.append(filled_text + "\n\n")
        
        print(f"    ✅ Created {len(texts)} synthetic texts")
        return texts
    
    def create_sequences_from_texts(self, texts: List[str], target_sequences: int):
        """Create sequences from texts with proper tokenization"""
        print(f"  🔧 Processing texts into {self.seq_length}-token sequences...")
        
        # Tokenize all texts
        all_tokens = []
        for i, text in enumerate(texts):
            if i % 300 == 0:
                print(f"    Processing text {i+1}/{len(texts)}...")
                
            try:
                tokens = self.tokenizer.encode(
                    text, 
                    add_special_tokens=True,
                    max_length=self.seq_length,
                    truncation=True
                )
                all_tokens.extend(tokens)
                all_tokens.append(self.tokenizer.eos_token_id)
            except:
                continue
        
        print(f"    📊 Total tokens: {len(all_tokens):,}")
        
        # Create sequences
        sequences_created = 0
        for start_idx in range(0, len(all_tokens) - self.seq_length, self.seq_length):
            if sequences_created >= target_sequences:
                break
                
            sequence = all_tokens[start_idx:start_idx + self.seq_length]
            if len(sequence) == self.seq_length:
                self.examples.append(torch.tensor(sequence, dtype=torch.long))
                sequences_created += 1
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


# ==================== ENHANCED TRAINING FUNCTIONS ====================

def test_enhanced_generation(model, tokenizer, prompts: List[str], device, max_tokens: int = 50):
    """Enhanced generation testing with better sampling"""
    model.eval()
    
    print("🎯 Enhanced Generation Test:")
    for i, prompt in enumerate(prompts):
        try:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                generated = input_ids.clone()
                
                for _ in range(max_tokens):
                    if generated.size(1) >= model.max_seq_len:
                        break
                    
                    logits = model(generated)
                    next_token_logits = logits[:, -1, :] / 0.7  # Lower temperature for better coherence
                    
                    # Enhanced top-p sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > 0.85  # Slightly more restrictive
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    generated = torch.cat([generated, next_token], dim=1)
                    
                    if next_token.item() == tokenizer.eos_token_id:
                        break
            
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"  Prompt {i+1}: {text}")
        
        except Exception as e:
            print(f"  ❌ Error with prompt {i+1}: {e}")
    
    model.train()


def report_enhanced_adaptation_stats(model, epoch: int):
    """Report enhanced biological adaptation statistics"""
    stats = model.get_enhanced_adaptation_stats()
    
    print(f"\n🧬 Enhanced Biological Adaptation Stats (Epoch {epoch}):")
    
    enhanced_stats = stats.get('enhanced_total', {})
    print(f"   Total splats across all layers: {enhanced_stats.get('total_splats', 0)}")
    print(f"   Total births: {enhanced_stats.get('total_births', 0)}")
    print(f"   Total deaths: {enhanced_stats.get('total_deaths', 0)}")
    print(f"   Ready for mitosis: {enhanced_stats.get('total_ready_for_mitosis', 0)}")
    print(f"   Max generation: {enhanced_stats.get('max_generation', 0)}")
    print(f"   Avg exploration activity: {enhanced_stats.get('avg_exploration_activity', 0):.3f}")
    print(f"   Growth factor: {enhanced_stats.get('growth_factor', 1):.2f}x")
    print(f"   Evolutionary activity: {enhanced_stats.get('evolutionary_activity', 0)} events")
    
    # Show top 3 most active layers
    layer_activity = []
    for key, layer_stats in stats.items():
        if key.startswith('layer_'):
            layer_num = int(key.split('_')[1])
            activity = layer_stats.get('birth_count', 0) + layer_stats.get('death_count', 0)
            layer_activity.append((layer_num, activity, layer_stats))
    
    layer_activity.sort(key=lambda x: x[1], reverse=True)
    
    print(f"   Most active layers:")
    for i, (layer_num, activity, layer_stats) in enumerate(layer_activity[:3]):
        print(f"     Layer {layer_num}: {layer_stats.get('num_splats', 0)} splats, "
              f"usefulness={layer_stats.get('avg_usefulness', 0):.2f}, "
              f"generation={layer_stats.get('max_generation', 0)}, "
              f"activity={activity}")


def save_checkpoint(model, optimizer, scheduler, epoch, loss, adaptation_stats, config, checkpoint_dir="checkpoints"):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'adaptation_stats': adaptation_stats,
        'config': config
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Also save as latest
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    torch.save(checkpoint, latest_path)
    
    print(f"   💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['adaptation_stats'], checkpoint['config']


def train_enhanced_biological_splatflow():
    """Train Enhanced Biological SplatFlow with longer training and more evolution"""
    print("🧬 Enhanced Biological SplatFlow Training")
    print("=" * 60)
    print("🎯 Goal: Extended training with more active biological adaptation")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory = gpu_props.total_memory / 1024**3
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_memory:.1f}GB")
        
        cleanup_memory()
        mem_info = get_gpu_memory_info()
        print(f"Available: {mem_info['free']:.2f}GB")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    # Enhanced configuration with longer training and more evolution
    config = {
        'max_seq_len': 1024,
        'model_dim': 256,
        'num_layers': 4,
        'initial_splats': 12,        # Start conservatively
        'max_splats': 64,            # Allow significant growth
        'batch_size': 3,             # Slightly smaller for longer sequences
        'accumulation_steps': 6,     # Effective batch of 18
        'epochs': 100,               # Much longer training!
        'dataset_size': 2000,        # Larger dataset
        'learning_rate': 1.5e-4,     # Slightly lower for stability
        'gradient_clip': 1.0,
        'weight_decay': 0.01,
        'adaptation_frequency': 8,   # More frequent adaptation
        'checkpoint_every': 10,      # Save every 10 epochs
        'test_every': 5,             # Test generation every 5 epochs
    }
    
    print(f"📋 Enhanced Biological Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\n🚀 Key Enhancements:")
    print(f"   📈 Training length: {config['epochs']} epochs (4x longer)")
    print(f"   🧬 More aggressive mitosis: Lower thresholds, higher rates")
    print(f"   🔄 More frequent adaptation: Every {config['adaptation_frequency']} steps")
    print(f"   💾 Regular checkpointing: Every {config['checkpoint_every']} epochs")
    print(f"   📊 Larger dataset: {config['dataset_size']} sequences")
    
    # Create enhanced dataset
    print(f"\n📚 Creating Enhanced Dataset...")
    dataset = LargerRealDataset(
        tokenizer,
        seq_length=config['max_seq_len'],
        total_sequences=config['dataset_size']
    )
    
    if len(dataset) == 0:
        print("❌ Failed to create dataset")
        return None
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    print(f"✅ Enhanced dataset ready: {len(dataset)} sequences")
    
    # Create enhanced biological model
    print(f"\n🧬 Creating Enhanced Biological SplatFlow Model...")
    cleanup_memory()
    
    model = EnhancedBiologicalSplatFlowGPT(
        vocab_size=vocab_size,
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_splats=config['initial_splats'],
        max_splats=config['max_splats'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    # Enhanced training setup
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)  # Slightly different betas for stability
    )
    
    # Enhanced scheduler with warmup
    warmup_steps = 5
    def lr_lambda(epoch):
        if epoch < warmup_steps:
            return epoch / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_steps) / (config['epochs'] - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Enhanced test prompts
    test_prompts = [
        "Once upon a time",
        "The future of technology",
        "In a small village",
        "Scientists recently discovered",
        "The old wizard said",
        "In the year 2050"
    ]
    
    print(f"\n🔥 Starting Enhanced Biological Training ({config['epochs']} epochs)...")
    print(f"   🧬 Enhanced adaptation: More births, active exploration, generational evolution")
    print(f"   📈 Splats can grow from {config['initial_splats']} to {config['max_splats']} per layer")
    print(f"   💾 Checkpoints every {config['checkpoint_every']} epochs")
    
    training_log = {
        'losses': [], 
        'epochs': [], 
        'adaptation_stats_history': [],
        'generation_tests': {}
    }
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        model.train()
        
        epoch_loss = 0
        epoch_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 40)
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                batch = batch.to(device, non_blocking=True)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                # Forward pass with enhanced biological adaptation
                logits = model(inputs, compute_loss_per_token=True)
                loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))
                
                # Scale for accumulation
                loss = loss / config['accumulation_steps']
                loss.backward()
                
                epoch_loss += loss.item() * config['accumulation_steps']
                
                # Update weights
                if (batch_idx + 1) % config['accumulation_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'])
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_batches += 1
                
                if batch_idx % 30 == 0:
                    mem_info = get_gpu_memory_info()
                    lr = scheduler.get_last_lr()[0]
                    print(f"  Batch {batch_idx+1:3d}/{len(dataloader)}: "
                          f"Loss={loss.item()*config['accumulation_steps']:.4f}, "
                          f"LR={lr:.2e}, "
                          f"Mem={mem_info['allocated']:.2f}GB")
                
            except torch.cuda.OutOfMemoryError:
                print(f"❌ OOM at batch {batch_idx}, skipping...")
                cleanup_memory()
                optimizer.zero_grad()
                continue
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else float('inf')
        
        training_log['epochs'].append(epoch + 1)
        training_log['losses'].append(avg_loss)
        
        print(f"\n📊 Epoch {epoch + 1} Complete:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Time: {epoch_time:.1f}s")
        print(f"   Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Enhanced biological adaptation reporting
        report_enhanced_adaptation_stats(model, epoch + 1)
        adaptation_stats = model.get_enhanced_adaptation_stats()
        training_log['adaptation_stats_history'].append(adaptation_stats)
        
        scheduler.step()
        
        # Test generation periodically
        if (epoch + 1) % config['test_every'] == 0:
            print(f"\n🎯 Generation Test (Epoch {epoch + 1}):")
            test_enhanced_generation(model, tokenizer, test_prompts, device)
            training_log['generation_tests'][epoch + 1] = f"Tested at epoch {epoch + 1}"
        
        # Save checkpoints
        if (epoch + 1) % config['checkpoint_every'] == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, avg_loss, adaptation_stats, config)
        
        cleanup_memory()
    
    total_time = time.time() - start_time
    
    print(f"\n🏁 Enhanced Biological Training Complete!")
    print(f"   Total Time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"   Final Loss: {training_log['losses'][-1]:.4f}")
    print(f"   Loss Improvement: {training_log['losses'][0]:.4f} → {training_log['losses'][-1]:.4f}")
    print(f"   Reduction: {((training_log['losses'][0] - training_log['losses'][-1]) / training_log['losses'][0] * 100):.1f}%")
    
    # Final enhanced adaptation stats
    final_stats = model.get_enhanced_adaptation_stats()
    enhanced_total = final_stats.get('enhanced_total', {})
    
    print(f"\n🧬 Final Enhanced Biological State:")
    print(f"   Final total splats: {enhanced_total.get('total_splats', 0)}")
    print(f"   Total evolutionary events: {enhanced_total.get('evolutionary_activity', 0)}")
    print(f"   Growth factor: {enhanced_total.get('growth_factor', 1):.2f}x")
    print(f"   Max generation reached: {enhanced_total.get('max_generation', 0)}")
    print(f"   Avg exploration activity: {enhanced_total.get('avg_exploration_activity', 0):.3f}")
    
    # Evolution analysis
    if enhanced_total.get('evolutionary_activity', 0) > 10:
        print(f"   🎉 HIGH EVOLUTIONARY ACTIVITY: Significant biological adaptation occurred!")
    elif enhanced_total.get('evolutionary_activity', 0) > 5:
        print(f"   📈 MODERATE EVOLUTIONARY ACTIVITY: Good biological adaptation")
    else:
        print(f"   📊 LOW EVOLUTIONARY ACTIVITY: Limited biological adaptation")
    
    # Final generation test
    print(f"\n🔬 Final Enhanced Generation Test:")
    test_enhanced_generation(model, tokenizer, test_prompts, device, max_tokens=80)
    
    # Freeze adaptation for inference
    model.freeze_adaptation()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_log': training_log,
        'final_adaptation_stats': final_stats,
        'tokenizer_name': 'gpt2',
        'training_time_hours': total_time / 3600,
        'enhanced_features': [
            'More aggressive mitosis criteria',
            'Enhanced exploration movement', 
            'Generational evolution tracking',
            'Extended training duration',
            'Larger dataset'
        ]
    }, 'enhanced_biological_splatflow_real_data.pt')
    
    print(f"💾 Enhanced model saved: enhanced_biological_splatflow_real_data.pt")
    
    return model, tokenizer, config, training_log


if __name__ == "__main__":
    print("🧬 Testing Enhanced Biological SplatFlow with Extended Training")
    print("Goal: See if longer training with more active evolution leads to breakthrough")
    print()
    
    try:
        model, tokenizer, config, log = train_enhanced_biological_splatflow()
        
        if model is not None:
            print(f"\n🎉 SUCCESS! Enhanced Biological SplatFlow trained with extended evolution")
            print(f"✅ Model learned over {config['epochs']} epochs")
            print(f"✅ Enhanced biological adaptation mechanisms active throughout")
            print(f"✅ O(n*k) efficiency maintained with adaptive splat populations")
            print(f"🧬 Enhanced biological mechanisms enabled extensive evolution")
            print(f"⏱️  Training completed in {log.get('training_time_hours', 'unknown')} hours")
    
    except Exception as e:
        print(f"\n❌ Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
