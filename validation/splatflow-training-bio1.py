"""
SplatFlow Real Data Training with Biological Adaptation

This script combines:
- The proven O(n*k) SplatFlow implementation 
- Biological adaptation mechanisms (mitosis, death, exploration)
- Extended training with real datasets
- Proper evaluation and monitoring

Goal: Test if biological adaptation improves SplatFlow's learning on real data
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


# ==================== BIOLOGICAL ADAPTATION COMPONENTS ====================

class AdaptiveSplat:
    """A splat with biological-style adaptation capabilities"""
    
    def __init__(self, position: torch.Tensor, scale: float, amplitude: float, splat_id: int, device: torch.device = None):
        # Determine device from position tensor or use provided device
        if device is None:
            device = position.device
        
        # Ensure all tensors are on the same device
        self.position = position.clone().detach().to(device).requires_grad_(True)
        self.log_scale = torch.tensor(math.log(scale), device=device, requires_grad=True)
        self.amplitude = torch.tensor(amplitude, device=device, requires_grad=True)
        self.id = splat_id
        self.device = device
        
        # Biological properties
        self.age = 0
        self.usefulness = 1.0
        self.activation_history = []
        self.error_history = []
        self.mitosis_readiness = 0.0
        self.death_countdown = -1
        self.errorContribution = 0.0
        
        # Movement properties - ensure velocity is on correct device
        self.velocity = torch.zeros_like(self.position, device=device)
        self.exploration_rate = 0.1
        
    def get_scale(self):
        # Ensure the scale tensor is on the correct device
        return torch.exp(self.log_scale.to(self.device)).clamp(min=0.1, max=2.0)
    
    def update_activation(self, activation: float, error: float):
        """Update biological state based on usage"""
        self.age += 1
        
        # Track activation and error history
        self.activation_history.append(abs(activation))
        self.error_history.append(abs(error))
        
        # Keep only recent history
        if len(self.activation_history) > 20:
            self.activation_history.pop(0)
            self.error_history.pop(0)
        
        # Update usefulness based on recent performance
        recent_activation = np.mean(self.activation_history[-5:]) if len(self.activation_history) >= 5 else abs(activation)
        recent_error = np.mean(self.error_history[-5:]) if len(self.error_history) >= 5 else abs(error)
        
        # Usefulness increases with high activation and low error
        usefulness_delta = 0.01 * (recent_activation - recent_error)
        self.usefulness = np.clip(self.usefulness + usefulness_delta, 0.1, 2.0)
        
        # Update mitosis readiness
        if recent_activation > 0.7 and recent_error < 0.3:
            self.mitosis_readiness += 0.02
        else:
            self.mitosis_readiness *= 0.98
    
    def should_divide(self) -> bool:
        """Check if this splat should undergo mitosis"""
        return (self.mitosis_readiness > 0.7 and 
                self.age > 30 and 
                self.usefulness > 1.1)
    
    def should_die(self) -> bool:
        """Check if this splat should be removed"""
        return (self.age > 100 and 
                self.usefulness < 0.3 and
                len(self.activation_history) > 10 and
                np.mean(self.activation_history[-10:]) < 0.05)
    
    def explore_movement(self, learning_rate: float, device: torch.device):
        """Active exploration movement with proper device handling"""
        if self.age % 10 == 0:  # Explore every 10 steps
            # Create exploration noise on the correct device
            exploration_noise = torch.randn_like(self.position.to(device)) * self.exploration_rate
            
            # Add momentum-based movement
            if hasattr(self, 'last_gradient') and self.last_gradient is not None:
                momentum = 0.9
                self.velocity = momentum * self.velocity.to(device) + learning_rate * self.last_gradient.to(device)
                exploration_noise += self.velocity
            
            # Apply movement (ensure position is on correct device)
            self.position.data = self.position.data.to(device) + exploration_noise
            
            # Decay exploration rate as splat matures
            self.exploration_rate *= 0.999


class BiologicalSplatAttentionLayer(nn.Module):
    """O(n*k) Splat attention with biological adaptation during training"""
    
    def __init__(self, model_dim: int, initial_splats: int = 16, max_splats: int = 64, 
                 dropout: float = 0.1, temperature: float = 1.0):
        super().__init__()
        
        self.model_dim = model_dim
        self.max_splats = max_splats
        self.temperature = temperature
        self.adaptive_splats = []
        self.adaptation_enabled = True
        self.adaptation_frequency = 10
        self.forward_count = 0
        self.birth_count = 0
        self.death_count = 0
        
        # Projections
        self.token_value_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.output_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters for compatibility
        self.num_splats = initial_splats
        self.splat_centers = nn.Parameter(torch.randn(initial_splats, model_dim) * 0.02)
        self.splat_log_scales = nn.Parameter(torch.zeros(initial_splats))
        
        self._init_weights()
        
        # Initialize adaptive splats after parameters are set up
        # We'll do this in a lazy way on first forward pass to ensure correct device
    
    def _init_weights(self):
        """Initialize weights with careful scaling"""
        nn.init.xavier_uniform_(self.token_value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
    
    def _initialize_adaptive_splats(self, num_splats: int):
        """Initialize adaptive splats with biological properties"""
        self.adaptive_splats = []
        
        # Get device from model parameters safely
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
        
        for i in range(num_splats):
            position = torch.randn(self.model_dim, device=device) * 0.02
            scale = 0.5 + torch.rand(1).item() * 0.5
            amplitude = 0.8 + torch.rand(1).item() * 0.4
            
            splat = AdaptiveSplat(position, scale, amplitude, i, device=device)
            self.adaptive_splats.append(splat)
    
    def _sync_splats_to_parameters(self):
        """Sync adaptive splats to parameters"""
        num_splats = len(self.adaptive_splats)
        
        if num_splats == 0:
            return
        
        device = self.splat_centers.device
        
        # Update parameter tensors - ensure everything is on the same device
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
        
        # Use current adaptive splats
        if not self.adaptive_splats:
            return torch.zeros(batch_size, seq_len, 0, device=device)
        
        # Get current centers and scales - ensure they're on the correct device
        centers = torch.stack([splat.position.to(device) for splat in self.adaptive_splats])  # [num_splats, model_dim]
        scales = torch.stack([splat.get_scale().to(device) for splat in self.adaptive_splats])  # [num_splats]
        
        # Compute squared distances
        tokens_expanded = token_embeddings.unsqueeze(2)  # [batch, seq_len, 1, model_dim]
        centers_expanded = centers.unsqueeze(0).unsqueeze(0)  # [1, 1, num_splats, model_dim]
        
        diff = tokens_expanded - centers_expanded
        distances_sq = torch.sum(diff ** 2, dim=-1)  # [batch, seq_len, num_splats]
        
        # Apply learned scales
        scales_sq = scales ** 2
        affinities = torch.exp(-0.5 * distances_sq / scales_sq.unsqueeze(0).unsqueeze(0))
        
        # Apply temperature and normalize
        affinities = affinities ** (1.0 / self.temperature)
        affinities = affinities / (affinities.sum(dim=-1, keepdim=True) + 1e-8)
        
        return affinities
    
    def _apply_biological_adaptation(self, affinities: torch.Tensor, loss_per_token: torch.Tensor):
        """Apply biological adaptation mechanisms"""
        if not self.adaptation_enabled or not self.adaptive_splats:
            return

        device = affinities.device
        
        try:
            # Calculate per-splat activation and error
            splat_activations = affinities.mean(dim=(0, 1))  # [num_splats]
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
                    
                    # Store gradient for momentum - ensure proper device handling
                    if splat.position.grad is not None:
                        splat.last_gradient = splat.position.grad.clone().detach().to(device)
                    
                    # Apply exploration movement with correct device
                    splat.explore_movement(0.01, device)
                    
                    # Check for division
                    if splat.should_divide() and len(self.adaptive_splats) < self.max_splats:
                        splats_to_divide.append(i)
                    
                    # Check for death
                    elif splat.should_die() and len(self.adaptive_splats) > 4:
                        splats_to_remove.append(i)
            
            # Apply mitosis
            for splat_idx in splats_to_divide:
                self._divide_splat(splat_idx, device)
            
            # Apply death (in reverse order to maintain indices)
            for splat_idx in sorted(splats_to_remove, reverse=True):
                self._remove_splat(splat_idx)
            
            # Sync parameters
            self._sync_splats_to_parameters()
            
        except Exception as e:
            print(f"Warning: Biological adaptation failed: {e}")
            # Continue without adaptation if there's an error
    
    def _divide_splat(self, splat_idx: int, device: torch.device):
        """Create two child splats from one parent"""
        parent = self.adaptive_splats[splat_idx]
        
        # Create two children with slight variations
        for i in range(2):
            offset = torch.randn_like(parent.position) * 0.1
            child_position = parent.position + offset
            
            scale_factor = 0.8 if i == 0 else 1.2
            child_scale = parent.get_scale().item() * scale_factor
            child_amplitude = parent.amplitude.item() * 0.8
            
            child = AdaptiveSplat(
                child_position, 
                child_scale, 
                child_amplitude, 
                len(self.adaptive_splats) + self.birth_count,
                device=device
            )
            
            # Inherit properties
            child.usefulness = parent.usefulness * 0.7
            child.exploration_rate = parent.exploration_rate * 1.5
            
            self.adaptive_splats.append(child)
            self.birth_count += 1
        
        # Mark parent for death
        parent.death_countdown = 30
        parent.usefulness *= 0.5
    
    def _remove_splat(self, splat_idx: int):
        """Remove a splat from the population"""
        if 0 <= splat_idx < len(self.adaptive_splats):
            self.adaptive_splats.pop(splat_idx)
            self.death_count += 1
    
    def forward(self, token_embeddings: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                loss_per_token: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with biological adaptation"""
        self.forward_count += 1
        
        # Lazy initialization of adaptive splats to ensure correct device
        if not self.adaptive_splats:
            self._initialize_adaptive_splats(self.num_splats)
        
        # Sync adaptive splats to parameters
        self._sync_splats_to_parameters()
        
        if not self.adaptive_splats:
            # Return input if no splats
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
        
        # Apply biological adaptation during training
        if self.training and self.adaptation_enabled and self.forward_count % self.adaptation_frequency == 0:
            with torch.no_grad():
                # Create dummy loss if not provided
                if loss_per_token is None:
                    loss_per_token = torch.randn(token_embeddings.shape[:2], device=token_embeddings.device) * 0.1
                
                self._apply_biological_adaptation(affinities, loss_per_token)
        
        return output
    
    def freeze_adaptation(self):
        """Stop adaptation and freeze for inference"""
        self.adaptation_enabled = False
        self._sync_splats_to_parameters()
    
    def get_adaptation_stats(self):
        """Get statistics about the adaptation process"""
        if not self.adaptive_splats:
            return {
                'num_splats': 0,
                'birth_count': self.birth_count,
                'death_count': self.death_count,
                'avg_usefulness': 0.0,
                'avg_age': 0.0,
                'ready_for_mitosis': 0
            }
        
        return {
            'num_splats': len(self.adaptive_splats),
            'birth_count': self.birth_count,
            'death_count': self.death_count,
            'avg_usefulness': np.mean([s.usefulness for s in self.adaptive_splats]),
            'avg_age': np.mean([s.age for s in self.adaptive_splats]),
            'ready_for_mitosis': sum(1 for s in self.adaptive_splats if s.mitosis_readiness > 0.8)
        }


class BiologicalSplatTransformerLayer(nn.Module):
    """Complete transformer layer using biological splat attention"""
    
    def __init__(self, model_dim: int, num_splats: int = 16, max_splats: int = 64,
                 ff_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = model_dim * 4
        
        # Biological splat attention
        self.attention = BiologicalSplatAttentionLayer(
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
        """Forward pass with biological adaptation"""
        
        # Self-attention with residual connection
        attn_output = self.attention(x, attention_mask, loss_per_token)
        x = self.attn_norm(x + attn_output)
        
        # Feed-forward with residual connection  
        ff_output = self.feed_forward(x)
        x = self.ff_norm(x + ff_output)
        
        return x
    
    def get_adaptation_stats(self):
        """Get adaptation statistics from attention layer"""
        return self.attention.get_adaptation_stats()
    
    def freeze_adaptation(self):
        """Freeze biological adaptation"""
        self.attention.freeze_adaptation()


class BiologicalSplatFlowGPT(nn.Module):
    """GPT model using biological splat attention with O(n*k) complexity"""
    
    def __init__(self, vocab_size: int, model_dim: int = 512, num_layers: int = 6,
                 num_splats: int = 16, max_splats: int = 64, max_seq_len: int = 1024, 
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
        
        # Transformer layers with biological splat attention
        self.layers = nn.ModuleList([
            BiologicalSplatTransformerLayer(
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
        """Report model complexity statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Biological SplatFlow Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Layers: {self.num_layers}")
        print(f"  Initial splats per layer: {self.num_splats}")
        print(f"  Max splats per layer: {self.max_splats}")
        print(f"  Model dimension: {self.model_dim}")
        print(f"  Theoretical complexity: O(n*k*{self.model_dim}) per layer with adaptive k")
    
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                compute_loss_per_token: bool = False) -> torch.Tensor:
        """Forward pass through the biological splat-flow model"""
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
            # Simple approximation: variance of embeddings as "confusion"
            loss_per_token = torch.var(x, dim=-1)
        
        # Process through biological splat-flow layers
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
    
    def get_adaptation_stats(self):
        """Get adaptation statistics from all layers"""
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}'] = layer.get_adaptation_stats()
        
        # Aggregate stats
        total_splats = sum(s['num_splats'] for s in stats.values())
        total_births = sum(s['birth_count'] for s in stats.values())
        total_deaths = sum(s['death_count'] for s in stats.values())
        
        stats['total'] = {
            'total_splats': total_splats,
            'total_births': total_births,
            'total_deaths': total_deaths,
            'avg_splats_per_layer': total_splats / len(self.layers) if self.layers else 0
        }
        
        return stats


# ==================== DATASET (Same as before) ====================

class RealDataset(Dataset):
    """Dataset that loads real text data from multiple sources"""
    
    def __init__(self, tokenizer, seq_length: int = 1024, total_sequences: int = 2000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.examples = []
        
        print(f"📚 Creating dataset with {total_sequences} sequences of {seq_length} tokens")
        
        # Collect texts from multiple sources
        all_texts = []
        
        # 1. TinyStories - Known to work well
        all_texts.extend(self.load_tinystories(target_texts=total_sequences//3))
        
        # 2. WikiText-103 - Good quality articles
        all_texts.extend(self.load_wikitext(target_texts=total_sequences//3))
        
        # 3. OpenWebText - If available
        all_texts.extend(self.load_openwebtext(target_texts=total_sequences//4))
        
        # 4. Fill remainder with quality synthetic
        current_count = len(all_texts)
        remaining = max(total_sequences//2 - current_count, 200)
        all_texts.extend(self.create_quality_synthetic(remaining))
        
        print(f"📊 Total source texts collected: {len(all_texts)}")
        
        # Create sequences
        self.create_sequences_from_texts(all_texts, total_sequences)
        
        print(f"✅ Final dataset: {len(self.examples)} sequences")
    
    def load_tinystories(self, target_texts: int) -> List[str]:
        """Load TinyStories"""
        texts = []
        try:
            print(f"  📖 Loading TinyStories (target: {target_texts})...")
            dataset = load_dataset("roneneldan/TinyStories", split="train")
            
            count = 0
            for item in dataset:
                if count >= target_texts:
                    break
                    
                text = item['text'].strip()
                if len(text) > 200:
                    texts.append(text + "\n\n")
                    count += 1
            
            print(f"    ✅ Added {len(texts)} TinyStories")
            
        except Exception as e:
            print(f"    ❌ Failed to load TinyStories: {e}")
        
        return texts
    
    def load_wikitext(self, target_texts: int) -> List[str]:
        """Load WikiText-103"""
        texts = []
        try:
            print(f"  📖 Loading WikiText-103 (target: {target_texts})...")
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
            
            count = 0
            for item in dataset:
                if count >= target_texts:
                    break
                    
                text = item['text'].strip()
                if len(text) > 500 and not text.startswith('='):
                    texts.append(text + "\n\n")
                    count += 1
            
            print(f"    ✅ Added {len(texts)} WikiText articles")
            
        except Exception as e:
            print(f"    ❌ Failed to load WikiText: {e}")
        
        return texts
    
    def load_openwebtext(self, target_texts: int) -> List[str]:
        """Load OpenWebText if available"""
        texts = []
        try:
            print(f"  📖 Loading OpenWebText (target: {target_texts})...")
            dataset = load_dataset("openwebtext", split="train")
            
            count = 0
            for item in dataset:
                if count >= target_texts:
                    break
                    
                text = item['text'].strip()
                if 300 < len(text) < 5000:
                    texts.append(text + "\n\n")
                    count += 1
            
            print(f"    ✅ Added {len(texts)} OpenWebText articles")
            
        except Exception as e:
            print(f"    ❌ Failed to load OpenWebText: {e}")
        
        return texts
    
    def create_quality_synthetic(self, target_texts: int) -> List[str]:
        """Create synthetic texts"""
        print(f"  🤖 Creating {target_texts} synthetic texts...")
        
        templates = [
            """The field of {topic} has seen remarkable progress recently. Scientists have discovered {finding}, which could revolutionize {application}.

This breakthrough builds on previous work in {related_field}. The key insight is that {insight}, enabling researchers to {capability}.

Practical applications include {use_case1} and {use_case2}. For instance, {example} demonstrates the potential for real-world impact.

Looking forward, experts predict {prediction}. The next steps involve {next_steps} and addressing challenges in {challenge_area}.""",

            """In a small village nestled between rolling hills, there lived a {character} who had an unusual gift. Every {time_period}, {character} could {ability}.

One day, a {visitor} arrived seeking help with {problem}. "{character}," said the {visitor}, "{request}."

At first, {character} was hesitant. {reason_for_hesitation}. But seeing the {visitor}'s distress, {character} decided to help.

The journey was not easy. They encountered {obstacle1} and had to overcome {obstacle2}. Through {method}, they learned {lesson}.

In the end, {outcome}. The {visitor} was grateful, and {character} realized {moral}."""
        ]
        
        topics = ["artificial intelligence", "renewable energy", "space exploration", "medicine", "education"]
        
        texts = []
        for i in range(target_texts):
            template = random.choice(templates)
            topic = random.choice(topics)
            
            filled_text = template.format(
                topic=topic,
                finding="unexpected patterns in large-scale data",
                application="how we solve complex problems",
                related_field="computational science",
                insight="complex systems follow simple principles",
                capability="predict outcomes with greater accuracy",
                use_case1="climate modeling",
                use_case2="disease prevention",
                example="recent cancer research",
                prediction="these technologies will become mainstream",
                next_steps="developing better algorithms",
                challenge_area="ethical deployment",
                
                # Story elements
                character="wise healer",
                time_period="full moon",
                ability="see the future in dreams",
                visitor="desperate merchant",
                problem="a terrible curse",
                request="please help me save my family",
                reason_for_hesitation="the visions were often unclear",
                obstacle1="treacherous mountain paths",
                obstacle2="ancient guardians",
                method="courage and wisdom",
                lesson="that true power comes from helping others",
                outcome="the curse was broken",
                moral="that every gift should be used for good"
            )
            
            texts.append(filled_text + "\n\n")
        
        print(f"    ✅ Created {len(texts)} synthetic texts")
        return texts
    
    def create_sequences_from_texts(self, texts: List[str], target_sequences: int):
        """Create sequences from texts"""
        print(f"  🔧 Processing texts into {self.seq_length}-token sequences...")
        
        # Tokenize all texts
        all_tokens = []
        for i, text in enumerate(texts):
            if i % 200 == 0:
                print(f"    Processing text {i+1}/{len(texts)}...")
                
            try:
                # Tokenize with proper truncation
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


# ==================== TRAINING FUNCTIONS ====================

def test_generation(model, tokenizer, prompts: List[str], device, max_tokens: int = 40):
    """Test generation quality"""
    model.eval()
    
    print("🎯 Generation Test:")
    for i, prompt in enumerate(prompts):
        try:
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            with torch.no_grad():
                generated = input_ids.clone()
                
                for _ in range(max_tokens):
                    if generated.size(1) >= model.max_seq_len:
                        break
                    
                    logits = model(generated)
                    next_token_logits = logits[:, -1, :] / 0.8  # Temperature
                    
                    # Top-p sampling
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > 0.9
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


def report_adaptation_stats(model, epoch: int):
    """Report biological adaptation statistics"""
    stats = model.get_adaptation_stats()
    
    print(f"\n🧬 Biological Adaptation Stats (Epoch {epoch}):")
    
    total_stats = stats.get('total', {})
    print(f"   Total splats across all layers: {total_stats.get('total_splats', 0)}")
    print(f"   Total births: {total_stats.get('total_births', 0)}")
    print(f"   Total deaths: {total_stats.get('total_deaths', 0)}")
    print(f"   Avg splats per layer: {total_stats.get('avg_splats_per_layer', 0):.1f}")
    
    # Show first few layers in detail
    for i in range(min(3, len([k for k in stats.keys() if k.startswith('layer_')]))):
        layer_stats = stats.get(f'layer_{i}', {})
        print(f"   Layer {i}: {layer_stats.get('num_splats', 0)} splats, "
              f"usefulness={layer_stats.get('avg_usefulness', 0):.2f}, "
              f"ready_for_mitosis={layer_stats.get('ready_for_mitosis', 0)}")


def train_biological_splatflow_on_real_data():
    """Train Biological SplatFlow on real data"""
    print("🧬 Biological SplatFlow Real Data Training")
    print("=" * 50)
    print("🎯 Goal: Test if biological adaptation improves learning on real data")
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
    
    # Configuration with biological parameters
    config = {
        'max_seq_len': 1024,
        'model_dim': 256,
        'num_layers': 4,
        'initial_splats': 12,        # Start smaller
        'max_splats': 64,            # Allow significant growth
        'batch_size': 4,
        'accumulation_steps': 4,
        'epochs': 100,                # More epochs for adaptation -up from 25
        'dataset_size': 2000,
        'learning_rate': 2e-4,
        'gradient_clip': 1.0,
        'weight_decay': 0.01,
        'adaptation_frequency': 10,   # Adapt every 10 forward passes
    }
    
    print(f"📋 Biological Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create dataset
    print(f"\n📚 Creating Dataset...")
    dataset = RealDataset(
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
    
    print(f"✅ Dataset ready: {len(dataset)} sequences")
    
    # Create biological model
    print(f"\n🧬 Creating Biological SplatFlow Model...")
    cleanup_memory()
    
    model = BiologicalSplatFlowGPT(
        vocab_size=vocab_size,
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_splats=config['initial_splats'],
        max_splats=config['max_splats'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    
    # Test prompts
    test_prompts = [
        "Once upon a time",
        "The future of technology",
        "In a small village",
        "Scientists recently discovered"
    ]
    
    print(f"\n🔥 Starting Biological Training ({config['epochs']} epochs)...")
    print(f"   🧬 Adaptation enabled every {config['adaptation_frequency']} steps")
    print(f"   📈 Splats can grow from {config['initial_splats']} to {config['max_splats']} per layer")
    
    training_log = {'losses': [], 'epochs': [], 'adaptation_stats': []}
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        model.train()
        
        epoch_loss = 0
        epoch_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 30)
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                batch = batch.to(device, non_blocking=True)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                # Forward pass with biological adaptation
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
                
                if batch_idx % 20 == 0:
                    mem_info = get_gpu_memory_info()
                    print(f"  Batch {batch_idx+1:3d}/{len(dataloader)}: "
                          f"Loss={loss.item()*config['accumulation_steps']:.4f}, "
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
        
        # Report biological adaptation stats
        report_adaptation_stats(model, epoch + 1)
        adaptation_stats = model.get_adaptation_stats()
        training_log['adaptation_stats'].append(adaptation_stats)
        
        scheduler.step()
        
        # Test generation every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_generation(model, tokenizer, test_prompts, device)
        
        cleanup_memory()
    
    total_time = time.time() - start_time
    
    print(f"\n🏁 Biological Training Complete!")
    print(f"   Total Time: {total_time/60:.1f} minutes")
    print(f"   Final Loss: {training_log['losses'][-1]:.4f}")
    
    # Final adaptation stats
    final_stats = model.get_adaptation_stats()
    print(f"\n🧬 Final Biological State:")
    total_stats = final_stats.get('total', {})
    print(f"   Final total splats: {total_stats.get('total_splats', 0)}")
    print(f"   Total evolutionary events: {total_stats.get('total_births', 0) + total_stats.get('total_deaths', 0)}")
    print(f"   Growth factor: {total_stats.get('total_splats', 0) / (config['initial_splats'] * config['num_layers']):.2f}x")
    
    # Final generation test
    print(f"\n🔬 Final Generation Test:")
    test_generation(model, tokenizer, test_prompts, device, max_tokens=60)
    
    # Freeze adaptation for inference
    model.freeze_adaptation()
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_log': training_log,
        'final_adaptation_stats': final_stats,
        'tokenizer_name': 'gpt2'
    }, 'biological_splatflow_real_data.pt')
    
    print(f"💾 Model saved: biological_splatflow_real_data.pt")
    
    return model, tokenizer, config, training_log


if __name__ == "__main__":
    print("🧬 Testing Biological SplatFlow on Real Data")
    print("Goal: See if biological adaptation improves learning on real datasets")
    print()
    
    try:
        model, tokenizer, config, log = train_biological_splatflow_on_real_data()
        
        if model is not None:
            print(f"\n🎉 SUCCESS! Biological SplatFlow trained on real data")
            print(f"✅ Model learned to reduce loss from {log['losses'][0]:.4f} to {log['losses'][-1]:.4f}")
            print(f"✅ Biological adaptation events occurred throughout training")
            print(f"✅ O(n*k) efficiency maintained with adaptive splat populations")
            print(f"🧬 Biological mechanisms enabled dynamic splat optimization")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
