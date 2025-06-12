"""
SplatFlow Model Architecture - Complete Implementation
Complete transformer layers with SplatFlow, progressive layer training,
model checkpointing and generation.

This module provides:
- FixedUltimateProductionSplatFlowGPT: Complete model implementation  
- SplatFlowTransformerLayer: Enhanced transformer layers
- Progressive training support
- Model checkpointing and generation capabilities
- Integration with trajectory systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Optional, Tuple, Dict, List, Any, Union
from pathlib import Path
import json
import warnings

# Import from our attention components
from .splatflow_attention_components import (
    FixedProductionSplatFlowAttention,
    SplatFlowBlock,
    global_health_monitor
)

logger = logging.getLogger(__name__)


class FixedUltimateProductionSplatFlowGPT(nn.Module):
    """
    Ultimate production-ready SplatFlow GPT model.
    
    This is the complete model implementation that was missing from the original code.
    Features O(n*k) attention complexity, adaptive splat positioning, trajectory flow,
    and comprehensive health monitoring.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        model_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_splats: int = 20,
        max_seq_len: int = 4096,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_enhanced_trajectory: bool = True,
        splat_attention_ratio: float = 0.4,
        tie_word_embeddings: bool = True,
        prenorm: bool = True
    ):
        super().__init__()
        
        # Store configuration
        self.config = {
            'vocab_size': vocab_size,
            'model_dim': model_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'num_splats': num_splats,
            'max_seq_len': max_seq_len,
            'dropout': dropout,
            'layer_norm_eps': layer_norm_eps,
            'use_enhanced_trajectory': use_enhanced_trajectory,
            'splat_attention_ratio': splat_attention_ratio,
            'tie_word_embeddings': tie_word_embeddings,
            'prenorm': prenorm
        }
        
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.tie_word_embeddings = tie_word_embeddings
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        
        # Enhanced trajectory flow (if available)
        self.trajectory_flow = None
        if use_enhanced_trajectory:
            try:
                from .splatflow_trajectory_systems import EnhancedInterLayerTrajectoryFlow
                self.trajectory_flow = EnhancedInterLayerTrajectoryFlow(
                    model_dim=model_dim,
                    num_layers=num_layers,
                    max_seq_len=max_seq_len
                )
                
                # Verify the trajectory flow has the required methods
                if not hasattr(self.trajectory_flow, 'initialize_trajectories'):
                    logger.warning("⚠️ Trajectory flow missing initialize_trajectories method, disabling")
                    self.trajectory_flow = None
                elif not hasattr(self.trajectory_flow, 'apply_inter_layer_flow'):
                    logger.warning("⚠️ Trajectory flow missing apply_inter_layer_flow method, disabling")
                    self.trajectory_flow = None
                else:
                    logger.info("✅ Enhanced trajectory flow enabled with verified interface")
                    
            except ImportError as e:
                logger.warning(f"⚠️ Enhanced trajectory flow not available: {e}")
                use_enhanced_trajectory = False
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize trajectory flow: {e}")
                self.trajectory_flow = None
                use_enhanced_trajectory = False
        
        # SplatFlow transformer layers
        self.layers = nn.ModuleList([
            SplatFlowTransformerLayer(
                model_dim=model_dim,
                num_heads=num_heads,
                num_splats=num_splats,
                layer_idx=i,
                dropout=dropout,
                layer_norm_eps=layer_norm_eps,
                splat_attention_ratio=splat_attention_ratio,
                prenorm=prenorm
            )
            for i in range(num_layers)
        ])
        
        # Output components
        self.final_norm = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        
        # Tie embeddings (standard practice)
        if tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        
        # Health and monitoring
        self.register_buffer('training_steps', torch.tensor(0))
        self.register_buffer('generation_count', torch.tensor(0))
        self.register_buffer('last_loss', torch.tensor(float('inf')))
        
        # Progressive training support
        self.register_buffer('active_layers', torch.tensor(num_layers))
        self.progressive_training = False
        
        # Performance tracking
        self.performance_history = []
        
        self._init_parameters()
        self._register_health_monitoring()
        
        logger.info(f"🚀 FixedUltimateProductionSplatFlowGPT initialized!")
        logger.info(f"📊 Parameters: {self.get_parameter_count():,}")
        logger.info(f"🧠 Config: {model_dim}d, {num_layers} layers, {num_splats} splats")
        logger.info(f"🎯 Max sequence length: {max_seq_len}")
    
    def _init_parameters(self):
        """Initialize all parameters with proper scaling."""
        # Token embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        
        # Apply residual scaling to output layers  
        for layer in self.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'out_proj'):
                with torch.no_grad():
                    layer.self_attn.out_proj.weight *= 0.02 / math.sqrt(2 * self.num_layers)
            
            # Scale down FFN output for stability
            if hasattr(layer, 'ffn') and len(layer.ffn) > 3:
                with torch.no_grad():
                    layer.ffn[-2].weight *= 0.02 / math.sqrt(2 * self.num_layers)
        
        # Initialize LM head if not tied
        if not self.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, std=0.02)
    
    def _register_health_monitoring(self):
        """Register components with the global health monitor."""
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'self_attn'):
                global_health_monitor.register_component(
                    layer.self_attn, f"layer_{i}_attention"
                )
    
    def get_parameter_count(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameter_count(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def enable_progressive_training(self, start_layers: int = 2):
        """Enable progressive layer training starting with fewer layers."""
        self.progressive_training = True
        self.active_layers = torch.tensor(max(1, min(start_layers, self.num_layers)))
        logger.info(f"🎯 Progressive training enabled, starting with {self.active_layers} layers")
    
    def grow_progressive_layers(self, growth_rate: int = 1):
        """Gradually increase the number of active layers."""
        if self.progressive_training and self.active_layers < self.num_layers:
            old_layers = int(self.active_layers)
            self.active_layers = torch.tensor(min(self.num_layers, old_layers + growth_rate))
            logger.info(f"📈 Progressive training: {old_layers} → {int(self.active_layers)} layers")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        use_cache: bool = False
    ) -> Union[Dict[str, Any], Tuple]:
        """
        Forward pass of the complete SplatFlow GPT model.
        
        Args:
            input_ids: [batch, seq_len] token indices
            attention_mask: Optional attention mask
            position_ids: Optional position indices
            past_key_values: Optional cached key-values for generation
            labels: Optional labels for loss computation
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return a dictionary
            use_cache: Whether to cache key-values for generation
            
        Returns:
            Dictionary with logits, loss, and optional additional outputs
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Increment training steps
        if self.training:
            self.training_steps += 1
        else:
            self.generation_count += 1
        
        # Create position ids if not provided
        if position_ids is None:
            if past_key_values is not None and len(past_key_values) > 0:
                past_length = past_key_values[0][0].shape[-2]
            else:
                past_length = 0
            position_ids = torch.arange(
                past_length, seq_len + past_length, 
                dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + pos_embeds
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Enhanced trajectory initialization
        if self.trajectory_flow is not None:
            try:
                hidden_states = self.trajectory_flow.initialize_trajectories(hidden_states)
            except Exception as e:
                logger.warning(f"⚠️ Trajectory initialization failed: {e}, continuing without trajectory flow")
                self.trajectory_flow = None
        
        # Create causal attention mask if not provided
        if attention_mask is None:
            # Create causal mask
            full_seq_len = seq_len + (past_key_values[0][0].shape[-2] if past_key_values else 0)
            attention_mask = torch.tril(torch.ones(full_seq_len, full_seq_len, device=device))
            attention_mask = attention_mask.view(1, 1, full_seq_len, full_seq_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            
            # If we have past key values, only use the last part of the mask
            if past_key_values is not None:
                attention_mask = attention_mask[:, :, -seq_len:, :]
        
        # Process through transformer layers
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        present_key_values = [] if use_cache else None
        
        # Determine how many layers to use (for progressive training)
        num_active_layers = int(self.active_layers) if self.progressive_training else self.num_layers
        
        for layer_idx in range(num_active_layers):
            layer = self.layers[layer_idx]
            
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Apply trajectory flow between layers
            if self.trajectory_flow is not None:
                try:
                    hidden_states = self.trajectory_flow.apply_inter_layer_flow(
                        hidden_states, layer_idx
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Trajectory flow failed at layer {layer_idx}: {e}")
                    # Disable trajectory flow for remaining layers
                    self.trajectory_flow = None
            
            # Layer forward pass
            past_key_value = past_key_values[layer_idx] if past_key_values else None
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions.append(layer_outputs[1])
            
            if use_cache:
                present_key_values.append(layer_outputs[2])
        
        # Final processing
        hidden_states = self.final_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Update loss tracking
            with torch.no_grad():
                self.last_loss = loss.detach()
        
        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            return output
        
        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': present_key_values,
            'hidden_states': all_hidden_states,
            'attentions': all_attentions
        }
    
    @torch.no_grad()
    def generate_text(
        self,
        tokenizer,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> str:
        """
        Generate text using the SplatFlow model with advanced sampling.
        
        Args:
            tokenizer: Tokenizer for encoding/decoding
            prompt: Input prompt string
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            do_sample: Whether to use sampling vs greedy
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated text string
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.eos_token_id
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        original_length = input_ids.shape[1]
        
        # Track tokens for repetition penalty
        generated_tokens = input_ids.clone()
        
        for step in range(max_length):
            # Forward pass
            outputs = self(input_ids, use_cache=True)
            logits = outputs['logits']
            
            # Get next token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_tokens[0].tolist()):
                    next_token_logits[token_id] /= repetition_penalty
            
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            generated_tokens = torch.cat([generated_tokens, next_token.unsqueeze(0)], dim=-1)
            
            # Check for end of sequence
            if next_token.item() == eos_token_id:
                break
            
            # Check for maximum context length
            if input_ids.shape[1] >= self.max_seq_len:
                logger.warning(f"⚠️ Reached maximum context length ({self.max_seq_len})")
                break
        
        # Decode and return
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
    
    def save_model(self, save_path: Union[str, Path], save_config: bool = True):
        """Save model checkpoint."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        model_path = save_path / "pytorch_model.bin"
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        if save_config:
            config_path = save_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        
        # Save health report
        health_path = save_path / "health_report.json"
        with open(health_path, 'w') as f:
            health = self.get_model_health_report()
            json.dump(health, f, indent=2, default=str)
        
        logger.info(f"💾 Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: Union[str, Path], **kwargs):
        """Load model from checkpoint."""
        load_path = Path(load_path)
        
        # Load configuration
        config_path = load_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Override with any provided kwargs
        config.update(kwargs)
        
        # Create model
        model = cls(**config)
        
        # Load state dict
        model_path = load_path / "pytorch_model.bin"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            logger.info(f"📂 Model loaded from {load_path}")
        else:
            logger.warning(f"⚠️ Model weights not found at {load_path}")
        
        return model
    
    def get_model_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report for the entire model."""
        health_report = {
            'model_info': {
                'parameters': self.get_parameter_count(),
                'trainable_parameters': self.get_trainable_parameter_count(),
                'training_steps': int(self.training_steps),
                'generation_count': int(self.generation_count),
                'last_loss': float(self.last_loss),
                'config': self.config,
                'progressive_training': {
                    'enabled': self.progressive_training,
                    'active_layers': int(self.active_layers),
                    'total_layers': self.num_layers
                }
            },
            'system_health': global_health_monitor.get_system_health(),
            'layer_health': []
        }
        
        # Collect health from each layer
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'get_layer_health'):
                layer_health = layer.get_layer_health()
                layer_health['layer_idx'] = i
                health_report['layer_health'].append(layer_health)
        
        return health_report
    
    def optimize_for_inference(self):
        """Optimize model for inference (disable dropout, set eval mode, etc.)."""
        self.eval()
        
        # Disable dropout in all components
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        
        # Optimize splat positioning (could freeze positions for inference)
        for layer in self.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'splat_positioning'):
                positioning = layer.self_attn.splat_positioning
                positioning.adaptation_rate = 0.0  # Disable adaptation
        
        logger.info("🚀 Model optimized for inference")
    
    def benchmark_performance(
        self,
        batch_size: int = 2,
        seq_len: int = 512,
        num_iterations: int = 10
    ) -> Dict[str, float]:
        """Benchmark model performance."""
        self.eval()
        device = next(self.parameters()).device
        
        # Create dummy input
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self(input_ids)
        
        # Benchmark
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        
        if device.type == 'cuda':
            start_time.record()
        
        import time
        cpu_start = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                outputs = self(input_ids)
                logits = outputs['logits']
        
        if device.type == 'cuda':
            end_time.record()
            torch.cuda.synchronize()
            gpu_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            gpu_time = None
        
        cpu_time = time.time() - cpu_start
        
        # Calculate metrics
        total_tokens = batch_size * seq_len * num_iterations
        
        results = {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'num_iterations': num_iterations,
            'total_tokens': total_tokens,
            'cpu_time': cpu_time,
            'tokens_per_second_cpu': total_tokens / cpu_time,
        }
        
        if gpu_time is not None:
            results['gpu_time'] = gpu_time
            results['tokens_per_second_gpu'] = total_tokens / gpu_time
        
        return results


class SplatFlowTransformerLayer(nn.Module):
    """
    Enhanced transformer layer with SplatFlow attention.
    Includes layer-specific health monitoring and configuration.
    """
    
    def __init__(
        self,
        model_dim: int,
        num_heads: int = 8,
        num_splats: int = 20,
        layer_idx: int = 0,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        splat_attention_ratio: float = 0.4,
        prenorm: bool = True
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.model_dim = model_dim
        self.prenorm = prenorm
        ffn_dim = ffn_dim or 4 * model_dim
        
        # SplatFlow attention
        self.self_attn = FixedProductionSplatFlowAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            num_splats=num_splats,
            dropout=dropout,
            splat_attention_ratio=splat_attention_ratio
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.attn_norm = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(model_dim, eps=layer_norm_eps)
        
        # Health monitoring
        self.register_buffer('layer_forward_count', torch.tensor(0))
        self.register_buffer('layer_loss_contribution', torch.tensor(0.0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Forward pass with residual connections and layer normalization."""
        
        self.layer_forward_count += 1
        
        if self.prenorm:
            # Pre-normalization (recommended for stability)
            # Self-attention
            normed_hidden_states = self.attn_norm(hidden_states)
            attn_output, attn_weights, present_key_value = self.self_attn(
                hidden_states=normed_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            hidden_states = hidden_states + attn_output
            
            # Feed-forward
            normed_hidden_states = self.ffn_norm(hidden_states)
            ffn_output = self.ffn(normed_hidden_states)
            hidden_states = hidden_states + ffn_output
            
        else:
            # Post-normalization (original transformer)
            # Self-attention
            attn_output, attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            hidden_states = self.attn_norm(hidden_states + attn_output)
            
            # Feed-forward
            ffn_output = self.ffn(hidden_states)
            hidden_states = self.ffn_norm(hidden_states + ffn_output)
        
        return hidden_states, attn_weights, present_key_value
    
    def get_layer_health(self) -> Dict[str, Any]:
        """Get health statistics for this layer."""
        attn_health = self.self_attn.get_attention_health()
        
        return {
            'layer_idx': self.layer_idx,
            'layer_forward_count': int(self.layer_forward_count),
            'layer_loss_contribution': float(self.layer_loss_contribution),
            'attention_health': attn_health
        }


# Factory functions for easy model creation
def create_production_splatflow_model(
    vocab_size: int = 50257,
    model_dim: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    num_splats: int = 20,
    max_seq_len: int = 4096,
    dropout: float = 0.1,
    **kwargs
) -> FixedUltimateProductionSplatFlowGPT:
    """
    Factory function to create a production SplatFlow model.
    
    Args:
        vocab_size: Vocabulary size
        model_dim: Model dimension  
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_splats: Number of splats per attention layer
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        **kwargs: Additional configuration options
        
    Returns:
        FixedUltimateProductionSplatFlowGPT model instance
    """
    logger.info(f"🏗️ Creating SplatFlow model: {model_dim}d, {num_layers} layers, {num_splats} splats")
    
    model = FixedUltimateProductionSplatFlowGPT(
        vocab_size=vocab_size,
        model_dim=model_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_splats=num_splats,
        max_seq_len=max_seq_len,
        dropout=dropout,
        **kwargs
    )
    
    logger.info(f"✅ SplatFlow model created: {model.get_parameter_count():,} parameters")
    return model


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for SplatFlow model."""
    return {
        'vocab_size': 50257,
        'model_dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'num_splats': 20,
        'max_seq_len': 4096,
        'dropout': 0.1,
        'layer_norm_eps': 1e-5,
        'splat_attention_ratio': 0.4,
        'tie_word_embeddings': True,
        'prenorm': True,
        'use_enhanced_trajectory': True
    }


def initialize_model_for_training(config: Dict[str, Any]) -> FixedUltimateProductionSplatFlowGPT:
    """Initialize model with training-specific settings."""
    model = create_production_splatflow_model(**config)
    
    # Enable progressive training if requested
    if config.get('progressive_training', False):
        start_layers = config.get('progressive_start_layers', 2)
        model.enable_progressive_training(start_layers)
    
    return model


if __name__ == "__main__":
    # Test the complete model
    print("🧪 Testing Complete SplatFlow Model Architecture...")
    
    # Create test model
    config = create_default_config()
    config.update({
        'model_dim': 256,  # Smaller for testing
        'num_layers': 4,
        'num_splats': 16
    })
    
    model = create_production_splatflow_model(**config)
    print(f"✅ Model created: {model.get_parameter_count():,} parameters")
    
    # Test forward pass
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    labels = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    print(f"📊 Testing forward pass: {input_ids.shape}")
    
    # Forward pass
    outputs = model(input_ids, labels=labels)
    print(f"✅ Forward pass successful!")
    print(f"📈 Loss: {outputs['loss']:.4f}")
    print(f"🎯 Logits shape: {outputs['logits'].shape}")
    
    # Test generation (with dummy tokenizer interface)
    class DummyTokenizer:
        def __init__(self):
            self.eos_token_id = 50256
            self.pad_token_id = 50256
        
        def encode(self, text, return_tensors=None):
            # Simple character-level encoding for demo
            tokens = [ord(c) % 1000 for c in text[:10]]
            if return_tensors == 'pt':
                return torch.tensor([tokens])
            return tokens
        
        def decode(self, tokens, skip_special_tokens=False):
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()
            return ''.join([chr(t % 128 + 32) for t in tokens])
    
    print("🎭 Testing text generation...")
    tokenizer = DummyTokenizer()
    generated = model.generate_text(
        tokenizer, "Hello", max_length=20, temperature=0.8
    )
    print(f"📝 Generated: '{generated}'")
    
    # Test health monitoring
    print("🏥 Testing health monitoring...")
    health = model.get_model_health_report()
    print(f"🏥 System health: {health['system_health']['summary']}")
    
    # Test progressive training
    print("📈 Testing progressive training...")
    model.enable_progressive_training(start_layers=2)
    outputs = model(input_ids, labels=labels)
    print(f"✅ Progressive training works! Active layers: {model.active_layers}")
    
    # Test model saving/loading
    print("💾 Testing model save/load...")
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        model.save_model(temp_dir)
        loaded_model = FixedUltimateProductionSplatFlowGPT.load_model(temp_dir)
        print("✅ Model save/load successful!")
    
    # Performance benchmark
    print("⚡ Running performance benchmark...")
    benchmark = model.benchmark_performance(batch_size=1, seq_len=128, num_iterations=5)
    print(f"📊 Performance: {benchmark['tokens_per_second_cpu']:.1f} tokens/sec")
    
    print("\n🎉 All SplatFlow model architecture tests passed!")
