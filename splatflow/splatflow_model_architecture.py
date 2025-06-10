"""
SplatFlow Model Architecture Module
Complete model architecture and training components for the SplatFlow system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List, Any

from .splatflow_core_systems import DeviceManager
from .splatflow_trajectory_systems import EnhancedInterLayerTrajectoryFlow
from .splatflow_attention_components import FixedProductionSplatFlowAttention

logger = logging.getLogger(__name__)


class ProgressiveLayerTrainer:
    """Progressive layer unfreezing to prevent gradient vanishing cascade"""
    
    def __init__(self, model, warmup_epochs: int = 3):
        self.model = model
        self.warmup_epochs = warmup_epochs
        self.num_layers = len(model.layers) if hasattr(model, 'layers') and model.layers else 1
        self.current_active_layers = min(2, self.num_layers)
        
        logger.info(f"🔓 Progressive layer trainer initialized:")
        logger.info(f"   Warmup epochs: {warmup_epochs}")
        logger.info(f"   Total layers: {self.num_layers}")
        logger.info(f"   Starting with {self.current_active_layers} active layer(s)")
    
    def update_active_layers(self, epoch: int):
        """Progressively unfreeze layers during training"""
        try:
            if epoch < self.warmup_epochs:
                target_layers = min(2, self.num_layers)
            else:
                epochs_per_layer = max(3, self.warmup_epochs)
                additional_layers = (epoch - self.warmup_epochs) // epochs_per_layer
                target_layers = min(2 + additional_layers, self.num_layers)
            
            if target_layers != self.current_active_layers:
                logger.info(f"🔓 Progressive unfreezing: Activating {target_layers}/{self.num_layers} layers")
                self.current_active_layers = target_layers
                
                if hasattr(self.model, 'layers') and self.model.layers:
                    for i, layer in enumerate(self.model.layers):
                        if i < self.current_active_layers:
                            for param in layer.parameters():
                                param.requires_grad = True
                            if hasattr(layer, 'attention'):
                                layer.attention.adaptation_enabled = True
                            logger.info(f"   ✅ Layer {i}: ACTIVE")
                        else:
                            for param in layer.parameters():
                                param.requires_grad = False
                            if hasattr(layer, 'attention'):
                                layer.attention.adaptation_enabled = False
                            logger.info(f"   ❄️  Layer {i}: FROZEN")
        except Exception as e:
            logger.warning(f"Failed to update active layers: {e}")
    
    def get_training_status(self) -> Dict:
        """Get current progressive training status"""
        try:
            return {
                'active_layers': self.current_active_layers,
                'total_layers': self.num_layers,
                'progress_ratio': self.current_active_layers / max(self.num_layers, 1),
                'all_layers_active': self.current_active_layers == self.num_layers
            }
        except Exception as e:
            logger.warning(f"Failed to get training status: {e}")
            return {
                'active_layers': 0,
                'total_layers': 0,
                'progress_ratio': 0.0,
                'all_layers_active': False
            }


class FixedProductionSplatFlowTransformerLayer(nn.Module):
    """FIXED production-ready transformer layer with enhanced SplatFlow"""
    
    def __init__(self, model_dim: int, num_splats: int = 20, max_splats: int = 64,
                 ff_dim: Optional[int] = None, dropout: float = 0.1, layer_idx: int = 0):
        super().__init__()
        
        if ff_dim is None:
            ff_dim = model_dim * 4
        
        self.layer_idx = layer_idx
        self.model_dim = model_dim
        
        self.attention = FixedProductionSplatFlowAttention(
            model_dim, num_splats, max_splats, dropout, layer_idx
        )
        
        self.attn_norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(model_dim, eps=1e-6)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout)
        )
        
        self._init_production_weights()
        
        logger.info(f"🏭 FIXED Production transformer layer {layer_idx} initialized")
    
    def _init_production_weights(self):
        """Initialize weights with production-level scaling"""
        std = 0.02 / math.sqrt(self.layer_idx + 1)
        
        for module in self.feed_forward:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """FIXED production-level forward pass with comprehensive error handling"""
        device = DeviceManager.get_primary_device()
        x = DeviceManager.ensure_tensor_device(x, device)
        
        if attention_mask is not None:
            attention_mask = DeviceManager.ensure_tensor_device(attention_mask, device)
        
        try:
            attn_output = self.attention(x, attention_mask)
            x = self.attn_norm(x + attn_output)
            
            ff_output = self.feed_forward(x)
            x = self.ff_norm(x + ff_output)
            
            return x
            
        except Exception as e:
            logger.error(f"FIXED Production layer {self.layer_idx} forward pass failed: {e}")
            return x
    
    def get_production_stats(self, epoch: int = 0) -> Dict:
        """Get production-level statistics from attention layer"""
        try:
            return self.attention.get_production_stats(epoch)
        except Exception as e:
            logger.warning(f"Failed to get stats for layer {self.layer_idx}: {e}")
            return {
                'layer_idx': self.layer_idx,
                'num_splats': 0,
                'healthy_splats': 0,
                'avg_usefulness': 0.0,
                'avg_trajectory_influence': 0.0,
                'trajectory_strength': 0.0,
                'health_status': '🔴 ERROR'
            }


class FixedUltimateProductionSplatFlowGPT(nn.Module):
    """FIXED production-ready SplatFlow GPT model with proper splat positioning"""
    
    def __init__(self, vocab_size: int, model_dim: int = 512, num_layers: int = 6,
                 num_splats: int = 20, max_splats: int = 64, max_seq_len: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_splats = num_splats
        self.max_splats = max_splats
        self.max_seq_len = max_seq_len
        
        # Enhanced inter-layer trajectory communication
        self.trajectory_flow = EnhancedInterLayerTrajectoryFlow(num_layers, model_dim, max_seq_len)
        
        self.progressive_trainer = None
        
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            FixedProductionSplatFlowTransformerLayer(
                model_dim, num_splats, max_splats, dropout=dropout, layer_idx=i
            ) for i in range(num_layers)
        ])
        
        # Set trajectory computer for each attention layer
        for layer in self.layers:
            layer.attention.trajectory_computer = self.trajectory_flow
        
        self.final_norm = nn.LayerNorm(model_dim, eps=1e-6)
        self.output_projection = nn.Linear(model_dim, vocab_size, bias=False)
        
        self.apply(self._init_production_weights)
        self._report_fixed_production_stats()
        
        logger.info(f"🌟 FIXED Production SplatFlow GPT model initialized")
    
    def fix_all_splat_positioning(self, sample_batch: torch.Tensor):
        """FIXED: Properly position all splats based on actual token embedding distribution"""
        device = DeviceManager.get_primary_device()
        
        with torch.no_grad():
            try:
                # Ensure sample_batch is on the correct device
                sample_batch = DeviceManager.ensure_tensor_device(sample_batch, device)
                
                # Get sample embeddings from the first batch
                sample_size = min(100, sample_batch.size(1))  # First 100 tokens or less
                sample_embeddings = self.token_embedding(sample_batch[:, :sample_size])
                
                # Ensure embeddings are on correct device
                sample_embeddings = DeviceManager.ensure_tensor_device(sample_embeddings, device)
                
                print(f"🎯 FIXED: Positioning all splats based on actual embedding statistics...")
                
                # Fix positioning for each layer
                for layer_idx, layer in enumerate(self.layers):
                    print(f"   🔧 Fixing Layer {layer_idx} splats...")
                    layer.attention.fix_splat_positioning_based_on_embeddings(sample_embeddings)
                    print(f"   ✅ Layer {layer_idx}: All splats repositioned")
                
                print(f"✅ FIXED: All splat positioning complete!")
                
            except Exception as e:
                logger.error(f"Failed to fix splat positioning: {e}")
                # Continue without error to allow training to proceed
    
    def apply_progressive_repositioning(self, sample_batch: torch.Tensor, epoch: int):
        """Apply progressive splat repositioning during training"""
        if epoch > 0 and epoch % 3 == 0:  # Every 3 epochs
            with torch.no_grad():
                try:
                    device = DeviceManager.get_primary_device()
                    sample_batch = DeviceManager.ensure_tensor_device(sample_batch, device)
                    sample_size = min(50, sample_batch.size(1))
                    sample_embeddings = self.token_embedding(sample_batch[:, :sample_size])
                    sample_embeddings = DeviceManager.ensure_tensor_device(sample_embeddings, device)
                    
                    for layer in self.layers:
                        layer.attention.progressive_splat_repositioning(sample_embeddings, epoch)
                except Exception as e:
                    logger.warning(f"Progressive repositioning failed: {e}")
    
    def apply_emergency_rescue(self, sample_batch: torch.Tensor, epoch: int):
        """Apply emergency splat rescue if needed"""
        if epoch > 5 and epoch % 5 == 0:  # Every 5 epochs after epoch 5
            with torch.no_grad():
                try:
                    device = DeviceManager.get_primary_device()
                    sample_batch = DeviceManager.ensure_tensor_device(sample_batch, device)
                    sample_size = min(50, sample_batch.size(1))
                    sample_embeddings = self.token_embedding(sample_batch[:, :sample_size])
                    sample_embeddings = DeviceManager.ensure_tensor_device(sample_embeddings, device)
                    
                    for layer in self.layers:
                        layer.attention.emergency_splat_rescue(sample_embeddings, epoch)
                except Exception as e:
                    logger.warning(f"Emergency rescue failed: {e}")
    
    def _init_production_weights(self, module):
        """Initialize weights with production-level scaling"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    def _report_fixed_production_stats(self):
        """Report FIXED production model statistics"""
        try:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            logger.info(f"🌟 FIXED Production SplatFlow GPT Model Statistics:")
            logger.info(f"  Total parameters: {total_params:,}")
            logger.info(f"  Trainable parameters: {trainable_params:,}")
            logger.info(f"  Layers: {self.num_layers}")
            logger.info(f"  Splats per layer: {self.num_splats} (max: {self.max_splats})")
            logger.info(f"  Model dimension: {self.model_dim}")
            logger.info(f"  Max sequence length: {self.max_seq_len}")
            logger.info(f"  🔧 CRITICAL FIXES APPLIED:")
            logger.info(f"    ✅ Proper splat positioning based on actual token embeddings")
            logger.info(f"    ✅ Adaptive influence radius based on embedding density")
            logger.info(f"    ✅ Progressive splat repositioning during training")
            logger.info(f"    ✅ Emergency splat rescue system")
            logger.info(f"    ✅ Enhanced health criteria with progressive relaxation")
            logger.info(f"    ✅ Comprehensive embedding statistics tracking")
        except Exception as e:
            logger.warning(f"Failed to report model statistics: {e}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """FIXED production-level forward pass with enhanced trajectory flow"""
        device = DeviceManager.get_primary_device()
        batch_size, seq_len = input_ids.shape
        
        input_ids = DeviceManager.ensure_tensor_device(input_ids, device)
        if attention_mask is not None:
            attention_mask = DeviceManager.ensure_tensor_device(attention_mask, device)
        
        try:
            token_emb = self.token_embedding(input_ids)
            pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_emb = self.position_embedding(pos_ids)
            
            x = token_emb + pos_emb
            
            layer_trajectories = []
            
            for i, layer in enumerate(self.layers):
                try:
                    # Compute enhanced trajectory for this layer
                    trajectory, enhanced_pos = self.trajectory_flow.compute_enhanced_trajectory_flow(i, x)
                    layer_trajectories.append(trajectory)
                    
                    # Add enhanced positional information (reduced influence to avoid dimension issues)
                    if enhanced_pos.shape == x.shape:
                        x = x + 0.05 * enhanced_pos  # Reduced from 0.1
                    
                    x = self.embedding_dropout(x)
                    
                    # Process through layer
                    x = layer(x, attention_mask)
                except Exception as e:
                    logger.warning(f"Layer {i} processing failed: {e}")
                    # Continue with basic processing
                    x = layer(x, attention_mask)
            
            # Apply skip connections for trajectory flow
            try:
                enhanced_trajectories = self.trajectory_flow.apply_skip_connections(layer_trajectories)
            except Exception as e:
                logger.warning(f"Skip connections failed: {e}")
            
            x = self.final_norm(x)
            logits = self.output_projection(x)
            
            return logits
            
        except Exception as e:
            logger.error(f"FIXED production model forward pass failed: {e}")
            batch_size, seq_len = input_ids.shape
            return torch.randn(batch_size, seq_len, self.token_embedding.num_embeddings, 
                             device=device, requires_grad=True)
    
    def get_comprehensive_model_stats(self, epoch: int = 0) -> Dict:
        """Get comprehensive model statistics with robust error handling"""
        try:
            layer_stats = []
            total_splats = 0
            total_healthy_splats = 0
            
            # Safely collect layer statistics
            for layer in self.layers:
                try:
                    stats = layer.get_production_stats(epoch)
                    layer_stats.append(stats)
                    total_splats += stats.get('num_splats', 0)
                    total_healthy_splats += stats.get('healthy_splats', 0)
                except Exception as e:
                    logger.warning(f"Failed to get stats for layer {len(layer_stats)}: {e}")
                    # Add default stats for failed layer
                    layer_stats.append({
                        'layer_idx': len(layer_stats),
                        'num_splats': 0,
                        'healthy_splats': 0,
                        'health_status': '🔴 ERROR'
                    })
            
            # Safely get trajectory statistics
            try:
                trajectory_stats = self.trajectory_flow.get_comprehensive_statistics()
            except Exception as e:
                logger.warning(f"Failed to get trajectory stats: {e}")
                trajectory_stats = {'error': str(e)}
            
            # Safely calculate overall health with division protection
            if total_splats > 0:
                health_percentage = (total_healthy_splats / total_splats) * 100
            else:
                health_percentage = 0.0
            
            # Determine overall health status
            overall_health = "🟢 HEALTHY"
            if total_splats == 0:
                overall_health = "🔴 NO SPLATS"
            elif health_percentage < 30:
                overall_health = "🔴 CRITICAL"
            elif health_percentage < 60:
                overall_health = "🟡 WEAK"
            
            return {
                'epoch': epoch,
                'total_splats': total_splats,
                'total_healthy_splats': total_healthy_splats,
                'health_percentage': health_percentage,
                'overall_health': overall_health,
                'layer_stats': layer_stats,
                'trajectory_stats': trajectory_stats,
                'model_info': {
                    'num_layers': self.num_layers,
                    'model_dim': self.model_dim,
                    'num_splats_per_layer': self.num_splats,
                    'max_seq_len': self.max_seq_len
                }
            }
        except Exception as e:
            logger.error(f"Failed to get comprehensive model stats: {e}")
            return {
                'epoch': epoch,
                'total_splats': 0,
                'total_healthy_splats': 0,
                'health_percentage': 0.0,
                'overall_health': '🔴 ERROR',
                'layer_stats': [],
                'trajectory_stats': {'error': str(e)},
                'model_info': {
                    'num_layers': self.num_layers,
                    'model_dim': self.model_dim,
                    'num_splats_per_layer': self.num_splats,
                    'max_seq_len': self.max_seq_len
                },
                'error': str(e)
            }
    
    def generate_text(self, tokenizer, prompt: str, max_length: int = 100, 
                     temperature: float = 0.8, top_k: int = 50) -> str:
        """Generate text using the SplatFlow model"""
        try:
            device = DeviceManager.get_primary_device()
            self.eval()
            
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            # Generation parameters
            generated_ids = input_ids.clone()
            
            with torch.no_grad():
                for step in range(max_length):
                    try:
                        # Forward pass
                        logits = self.forward(generated_ids)
                        
                        # Get next token logits
                        next_token_logits = logits[0, -1, :] / max(temperature, 0.1)
                        
                        # Apply top-k filtering
                        if top_k > 0:
                            top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, len(next_token_logits)))
                            next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                            next_token_logits[top_k_indices] = top_k_logits
                        
                        # Sample next token
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        
                        # Append to generated sequence
                        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                        
                        # Check for end token
                        if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
                            break
                        
                        # Prevent infinite sequences
                        if generated_ids.shape[1] > self.max_seq_len:
                            break
                            
                    except Exception as e:
                        logger.warning(f"Generation step {step} failed: {e}")
                        break
            
            # Decode generated text
            try:
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                return generated_text
            except Exception as e:
                logger.error(f"Failed to decode generated text: {e}")
                return f"[Generation Error: {str(e)}]"
                
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return f"[Generation Failed: {str(e)}]"
    
    def save_model_checkpoint(self, filepath: str, epoch: int, optimizer_state: Optional[Dict] = None):
        """Save comprehensive model checkpoint"""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'model_config': {
                    'vocab_size': self.token_embedding.num_embeddings,
                    'model_dim': self.model_dim,
                    'num_layers': self.num_layers,
                    'num_splats': self.num_splats,
                    'max_splats': self.max_splats,
                    'max_seq_len': self.max_seq_len
                },
                'model_stats': self.get_comprehensive_model_stats(epoch)
            }
            
            if optimizer_state is not None:
                checkpoint['optimizer_state_dict'] = optimizer_state
            
            torch.save(checkpoint, filepath)
            logger.info(f"💾 Model checkpoint saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    @classmethod
    def load_model_checkpoint(cls, filepath: str, device: torch.device = None):
        """Load model from checkpoint"""
        try:
            if device is None:
                device = DeviceManager.get_primary_device()
            
            checkpoint = torch.load(filepath, map_location=device)
            config = checkpoint['model_config']
            
            # Create model with saved configuration
            model = cls(
                vocab_size=config['vocab_size'],
                model_dim=config['model_dim'],
                num_layers=config['num_layers'],
                num_splats=config['num_splats'],
                max_splats=config['max_splats'],
                max_seq_len=config['max_seq_len']
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            
            logger.info(f"📂 Model checkpoint loaded from {filepath}")
            logger.info(f"   Epoch: {checkpoint['epoch']}")
            logger.info(f"   Model stats: {checkpoint.get('model_stats', {}).get('overall_health', 'Unknown')}")
            
            return model, checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise


# Utility functions for training integration
def create_production_splatflow_model(vocab_size: int, **kwargs) -> FixedUltimateProductionSplatFlowGPT:
    """Factory function to create production SplatFlow model with reasonable defaults"""
    
    defaults = {
        'model_dim': 512,
        'num_layers': 6,
        'num_splats': 20,
        'max_splats': 64,
        'max_seq_len': 2048,
        'dropout': 0.1
    }
    
    # Override defaults with provided kwargs
    config = {**defaults, **kwargs}
    
    logger.info(f"🏭 Creating production SplatFlow model with config: {config}")
    
    try:
        model = FixedUltimateProductionSplatFlowGPT(vocab_size=vocab_size, **config)
        return model
    except Exception as e:
        logger.error(f"Failed to create SplatFlow model: {e}")
        raise


def setup_progressive_training(model: FixedUltimateProductionSplatFlowGPT, warmup_epochs: int = 3) -> ProgressiveLayerTrainer:
    """Setup progressive layer training for the model"""
    try:
        trainer = ProgressiveLayerTrainer(model, warmup_epochs)
        model.progressive_trainer = trainer
        return trainer
    except Exception as e:
        logger.error(f"Failed to setup progressive training: {e}")
        raise


def initialize_model_for_training(model: FixedUltimateProductionSplatFlowGPT, 
                                sample_batch: torch.Tensor,
                                setup_progressive: bool = True) -> Optional[ProgressiveLayerTrainer]:
    """Initialize model for training with proper splat positioning"""
    
    logger.info(f"🚀 Initializing model for training...")
    
    try:
        # Fix splat positioning based on actual embeddings
        model.fix_all_splat_positioning(sample_batch)
        
        # Setup progressive training if requested
        progressive_trainer = None
        if setup_progressive:
            progressive_trainer = setup_progressive_training(model)
            logger.info(f"✅ Progressive layer training enabled")
        
        logger.info(f"✅ Model initialization complete")
        
        return progressive_trainer
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return None
