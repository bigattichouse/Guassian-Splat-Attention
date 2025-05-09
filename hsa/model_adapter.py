"""
HSA Model Adapter - Module to integrate HSA with HuggingFace models.

This module provides utilities to replace the attention mechanism in
transformer models with Hierarchical Splat Attention.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import inspect

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel

# Import HSA components
from hsa.splat import Splat
from hsa.hierarchy import Hierarchy
from hsa.registry import SplatRegistry
from hsa.dense_attention import DenseAttentionComputer
from hsa.sparse_attention import SparseAttentionComputer
from hsa.attention_interface import AttentionConfig, AttentionComputer

logger = logging.getLogger(__name__)

class HSAAttention(nn.Module):
    """HSA attention module that can be used as a drop-in replacement for transformer attention."""
    
    def __init__(
        self,
        registry: SplatRegistry,
        attention_computer: AttentionComputer,
        adapter_config: Dict[str, Any],
        original_module: Optional[nn.Module] = None
    ):
        """Initialize HSA attention.
        
        Args:
            registry: SplatRegistry containing the splats
            attention_computer: AttentionComputer for computing attention
            adapter_config: Configuration for the adapter
            original_module: Original attention module (for parameter reference)
        """
        super().__init__()
        self.registry = registry
        self.attention_computer = attention_computer
        self.config = adapter_config
        self.original_module = original_module
        
        # Record dimensions based on original module if available
        if original_module is not None:
            if hasattr(original_module, "embed_dim"):
                self.embed_dim = original_module.embed_dim
            elif hasattr(original_module, "hidden_size"):
                self.embed_dim = original_module.hidden_size
            else:
                self.embed_dim = adapter_config.get("embed_dim", 768)
        else:
            self.embed_dim = adapter_config.get("embed_dim", 768)
        
        # Use original projections if they exist
        if self.config.get("use_projections", True):
            self.q_proj = self._get_or_create_projection("q_proj")
            self.k_proj = self._get_or_create_projection("k_proj")
            self.v_proj = self._get_or_create_projection("v_proj")
            self.out_proj = self._get_or_create_projection("out_proj")
        else:
            self.q_proj = self.k_proj = self.v_proj = self.out_proj = None
        
        # Attention dropout
        self.dropout = nn.Dropout(adapter_config.get("dropout", 0.1))
        
        # For tracking metrics
        self.attention_stats = {
            "calls": 0,
            "tokens_processed": 0
        }
    
    def _get_or_create_projection(self, proj_name: str) -> nn.Linear:
        """Get projection from original module or create a new one."""
        if (self.original_module is not None and
            hasattr(self.original_module, proj_name)):
            return getattr(self.original_module, proj_name)
        else:
            return nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass.
        
        Args:
            query: Query tensor [batch_size, seq_len, embed_dim]
            key: Key tensor [batch_size, seq_len, embed_dim]
            value: Value tensor [batch_size, seq_len, embed_dim]
            attn_mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Handle different input shapes - support both [batch, seq_len, hidden] and [seq_len, batch, hidden]
        input_shape = query.shape
        
        # Standard shape for HSA: [batch_size, seq_len, embed_dim]
        if len(input_shape) == 3 and input_shape[2] == self.embed_dim:
            # [batch, seq_len, hidden] - standard shape, no reshape needed
            batch_size, seq_len, _ = input_shape
        elif len(input_shape) == 3 and input_shape[1] == self.embed_dim:
            # [seq_len, batch, hidden] - need to transpose
            seq_len, batch_size, _ = input_shape
            query = query.transpose(0, 1)
            key = key.transpose(0, 1) if key is not None else None
            value = value.transpose(0, 1) if value is not None else None
        else:
            # Unsupported shape
            logger.warning(f"Unsupported input shape: {input_shape}")
            # Try to handle it anyway
            if len(input_shape) >= 2:
                batch_size = input_shape[0]
                seq_len = input_shape[1]
            else:
                # Really problematic case
                batch_size = 1
                seq_len = input_shape[0] if len(input_shape) > 0 else 1
        
        # Project inputs if using projections
        if self.config.get("use_projections", True) and self.q_proj is not None:
            # Ensure we're working with tensors
            if not isinstance(query, torch.Tensor):
                logger.warning(f"Query is not a tensor, got {type(query)}")
                # Try to convert or fallback
                if hasattr(query, "hidden_states") and isinstance(query.hidden_states, torch.Tensor):
                    query = query.hidden_states
                else:
                    # Can't proceed with non-tensor
                    logger.error(f"Cannot process non-tensor query: {type(query)}")
                    return query, None
            
            # Normal projection
            query = self.q_proj(query)
            key = self.k_proj(key) if key is not None else self.k_proj(query)
            value = self.v_proj(value) if value is not None else self.v_proj(query)
        else:
            # If no projections, ensure we have key and value
            key = query if key is None else key
            value = query if value is None else value
        
        # Ensure we're working with tensors
        if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            logger.error(f"After projection, inputs must be tensors. Got: query={type(query)}, key={type(key)}, value={type(value)}")
            # Try to use original module as fallback
            if self.original_module is not None and hasattr(self.original_module, "forward"):
                logger.warning("Falling back to original module forward")
                try:
                    return self.original_module.forward(query, key, value, attn_mask, **kwargs)
                except Exception as e:
                    logger.error(f"Original module fallback failed: {e}")
            # Last resort fallback
            return query, None
        
        # Convert to numpy for HSA computation
        tokens_np = query.detach().cpu().numpy()
        
        # Update stats
        self.attention_stats["calls"] += 1
        self.attention_stats["tokens_processed"] += batch_size * seq_len
        
        # Process each item in the batch
        outputs = []
        attention_weights = []
        
        for batch_idx in range(batch_size):
            # Get token embeddings for this batch item
            tokens = tokens_np[batch_idx]
            
            # Compute attention using HSA
            attention_matrix = self.attention_computer.compute_attention(tokens, self.registry)
            
            # Apply mask if provided
            if attn_mask is not None:
                mask_np = None
                # Handle different mask shapes
                if attn_mask.dim() == 2:
                    # Global mask for all batches
                    mask_np = attn_mask.detach().cpu().numpy()
                elif attn_mask.dim() == 3 and attn_mask.size(0) == batch_size:
                    # Batch-specific mask
                    mask_np = attn_mask[batch_idx].detach().cpu().numpy()
                
                if mask_np is not None:
                    # Expand mask if needed
                    if mask_np.shape[0] != seq_len or mask_np.shape[1] != seq_len:
                        # Try to reshape/broadcast
                        if mask_np.shape[0] == 1 and mask_np.shape[1] == seq_len:
                            # Broadcast single row mask
                            mask_np = np.tile(mask_np, (seq_len, 1))
                        elif mask_np.shape[0] == seq_len and mask_np.shape[1] == 1:
                            # Broadcast single column mask
                            mask_np = np.tile(mask_np, (1, seq_len))
                    
                    # Apply mask (element-wise multiplication)
                    if mask_np.shape == attention_matrix.shape:
                        attention_matrix = attention_matrix * mask_np
            
            # Convert back to tensor
            attention_tensor = torch.from_numpy(attention_matrix).to(value.device)
            
            # Apply attention to values
            batch_output = torch.matmul(attention_tensor, value[batch_idx])
            
            outputs.append(batch_output)
            attention_weights.append(attention_tensor)
        
        # Stack outputs
        output = torch.stack(outputs)
        attn_weights = torch.stack(attention_weights) if self.config.get("return_attention_weights", False) else None
        
        # Apply output projection if needed
        if self.config.get("use_projections", True) and self.out_proj is not None:
            output = self.out_proj(output)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Reshape output to match original input shape if needed
        if len(input_shape) == 3 and input_shape[1] == self.embed_dim:
            # Need to transpose back to [seq_len, batch, hidden]
            output = output.transpose(0, 1)
        
        return output, attn_weights


class ModelPatcher:
    """Utility for patching transformer models to use HSA attention."""
    
    def __init__(
        self, 
        model: PreTrainedModel,
        registry: SplatRegistry,
        attention_computer: AttentionComputer,
        config: Dict[str, Any]
    ):
        """Initialize model patcher.
        
        Args:
            model: The transformer model to patch
            registry: SplatRegistry containing splats
            attention_computer: AttentionComputer for HSA
            config: Configuration for patching
        """
        self.model = model
        self.registry = registry
        self.attention_computer = attention_computer
        self.config = config
        
        # Store original modules
        self.original_modules = {}
        
        # Store information about patched modules
        self.patched_modules = {}
        
        # Different model families have different module structures
        self.model_type = getattr(model.config, "model_type", "unknown")
        logger.info(f"Detected model type: {self.model_type}")
        
        # Adaptation controller (set externally)
        self.adaptation_controller = None
        
        # Determine patching strategy
        self._identify_attention_modules()
    
    def _identify_attention_modules(self):
        """Identify attention modules to patch based on model type."""
        # Define patterns for different model types
        if self.model_type == "gpt2":
            self.attention_patterns = [".attn"]
        elif self.model_type == "bert":
            self.attention_patterns = [".attention.self"]
        elif self.model_type == "roberta":
            self.attention_patterns = [".attention.self"]
        elif "gpt_neo" in self.model_type:
            self.attention_patterns = [".attn"]
        elif "llama" in self.model_type:
            self.attention_patterns = [".self_attn"]
        elif "opt" in self.model_type:
            self.attention_patterns = [".self_attn"]
        elif "t5" in self.model_type:
            self.attention_patterns = [".SelfAttention", ".EncDecAttention"]
        else:
            # Generic fallback - look for attention in name
            self.attention_patterns = ["attention", "attn"]
            logger.warning(f"Using generic pattern for unknown model type: {self.model_type}")
    
    def patch_model(self):
        """Patch the model by replacing attention modules with HSA."""
        # Find all modules matching our patterns
        modules_to_patch = []
        
        for name, module in self.model.named_modules():
            if any(pattern in name for pattern in self.attention_patterns):
                # Skip Linear layers that are used in projections
                if isinstance(module, nn.Linear):
                    logger.debug(f"Skipping Linear layer: {name}")
                    continue
                
                # Skip modules that are clearly sub-components
                skip_patterns = ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'dense']
                if any(pattern in name.split('.')[-1] for pattern in skip_patterns):
                    logger.debug(f"Skipping sub-component: {name}")
                    continue
                
                logger.info(f"Found attention module to patch: {name}")
                modules_to_patch.append((name, module))
        
        # Sometimes we find nested modules - filter to get only the outermost ones
        filtered_modules = []
        for name, module in modules_to_patch:
            # Check if this module is a parent of any other module
            is_parent = False
            for other_name, _ in modules_to_patch:
                if other_name.startswith(name + ".") and other_name != name:
                    is_parent = True
                    break
            
            if not is_parent:
                filtered_modules.append((name, module))
        
        # Patch modules
        for name, module in filtered_modules:
            try:
                self._patch_module(name, module)
            except Exception as e:
                logger.warning(f"Failed to patch module {name}: {e}")
        
        logger.info(f"Patched {len(self.patched_modules)} attention modules with HSA")
    
    def _patch_module(self, name: str, module: nn.Module):
        """Patch a specific module with HSA attention.
        
        Args:
            name: Full name of the module
            module: The module to patch
        """
        # Store original module
        self.original_modules[name] = {
            "module": module,
            "forward": module.forward
        }
        
        # Create HSA attention module
        hsa_module = HSAAttention(
            registry=self.registry,
            attention_computer=self.attention_computer,
            adapter_config=self.config,
            original_module=module
        )
        
        # Different model families need different patching approaches
        if self.model_type == "gpt2":
            self._patch_gpt2(name, module, hsa_module)
        elif "gpt_neo" in self.model_type:
            self._patch_gpt_neo(name, module, hsa_module)
        elif self.model_type in ["bert", "roberta"]:
            self._patch_bert(name, module, hsa_module)
        elif "llama" in self.model_type:
            self._patch_llama(name, module, hsa_module)
        elif "opt" in self.model_type:
            self._patch_opt(name, module, hsa_module)
        elif "t5" in self.model_type:
            self._patch_t5(name, module, hsa_module)
        else:
            self._patch_generic(name, module, hsa_module)
        
        # Record patched module
        self.patched_modules[name] = {
            "module": module,
            "hsa_module": hsa_module
        }
    
    def _patch_gpt2(self, name: str, module: nn.Module, hsa_module: HSAAttention):
        """Patch a GPT-2 attention module."""
        # For GPT-2, we only replace the _attn method which computes the attention scores
        def patched_attn(self, query, key, value, attention_mask=None, head_mask=None):
            # Forward to HSA module
            output, weights = hsa_module(query, key, value, attention_mask)
            return output, weights
        
        # Replace the _attn method
        setattr(module, '_attn', patched_attn.__get__(module, type(module)))
        
        # Keep the original forward method
        # This retains all the head splitting/merging logic of the original module
    
    def _patch_gpt_neo(self, name: str, module: nn.Module, hsa_module: HSAAttention):
        """Patch a GPT-Neo attention module."""
        # Store original _attn method
        original_attn = getattr(module, '_attn', None)
        
        # Create custom _attn method that uses HSA
        def patched_attn(self, query, key, value, attention_mask=None, head_mask=None):
            # Forward to HSA module
            output, weights = hsa_module(query, key, value, attention_mask)
            return output, weights
        
        # Replace the _attn method if it exists
        if original_attn is not None:
            setattr(module, '_attn', patched_attn.__get__(module, type(module)))
        else:
            # If _attn doesn't exist, we need to patch the forward method
            original_forward = module.forward
            
            def patched_forward(self, *args, **kwargs):
                # Try to extract inputs
                # For modules that handle input differently, fall back to original
                try:
                    if len(args) >= 3:
                        hidden_states, attention_mask, layer_past = args[:3]
                    else:
                        hidden_states = kwargs.get('hidden_states') or kwargs.get('input')
                        attention_mask = kwargs.get('attention_mask')
                        layer_past = kwargs.get('layer_past')
                        
                    if not isinstance(hidden_states, torch.Tensor):
                        # Fall back to original
                        return original_forward(*args, **kwargs)
                    
                    # Use HSA to compute attention
                    output, weights = hsa_module(hidden_states, hidden_states, hidden_states, attention_mask)
                    
                    # Try to match original return format
                    if isinstance(original_forward(*args, **kwargs), tuple):
                        # Many modules return (output, weights)
                        return (output, weights)
                    else:
                        # Some just return output
                        return output
                except Exception as e:
                    logger.warning(f"Error in patched forward: {e}, falling back to original")
                    # Fall back to original
                    return original_forward(*args, **kwargs)
            
            # Replace the forward method
            module.forward = patched_forward.__get__(module, type(module))
    
    def _patch_bert(self, name: str, module: nn.Module, hsa_module: HSAAttention):
        """Patch a BERT/RoBERTa attention module."""
        # For BERT family, we replace the forward method
        original_forward = module.forward
        
        def patched_forward(
            self, 
            hidden_states, 
            attention_mask=None, 
            head_mask=None, 
            encoder_hidden_states=None, 
            encoder_attention_mask=None, 
            past_key_value=None, 
            output_attentions=False,
            **kwargs
        ):
            # BERTSelfAttention typically splits hidden_states into Q/K/V
            query = key = value = hidden_states
            
            # Forward to HSA module
            output, weights = hsa_module(query, key, value, attention_mask)
            
            outputs = (output, weights) if output_attentions else (output,)
            return outputs
        
        # Replace the forward method
        module.forward = patched_forward.__get__(module, type(module))
    
    def _patch_llama(self, name: str, module: nn.Module, hsa_module: HSAAttention):
        """Patch a LLaMA attention module."""
        # For LLaMA, we replace the forward method
        original_forward = module.forward
        
        def patched_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            **kwargs
        ):
            # LLaMA typically has separate heads, we'll use the combined hidden states
            query = key = value = hidden_states
            
            # Forward to HSA module
            output, attention_weights = hsa_module(query, key, value, attention_mask)
            
            outputs = (output,)
            if output_attentions:
                outputs += (attention_weights,)
            if use_cache:
                outputs += (None,)  # No cache when using HSA
                
            return outputs
        
        # Replace the forward method
        module.forward = patched_forward.__get__(module, type(module))
    
    def _patch_opt(self, name: str, module: nn.Module, hsa_module: HSAAttention):
        """Patch an OPT attention module."""
        # For OPT, similar to others, we replace the forward method
        original_forward = module.forward
        
        def patched_forward(
            self,
            hidden_states,
            key_value_states=None,
            past_key_value=None,
            attention_mask=None,
            layer_head_mask=None,
            output_attentions=False,
            **kwargs
        ):
            # Use hidden_states for Q/K/V
            query = hidden_states
            key = hidden_states if key_value_states is None else key_value_states
            value = hidden_states if key_value_states is None else key_value_states
            
            # Forward to HSA module
            output, attention_weights = hsa_module(query, key, value, attention_mask)
            
            outputs = (output,)
            if output_attentions:
                outputs += (attention_weights,)
            if past_key_value is not None:
                outputs += (past_key_value,)
                
            return outputs
        
        # Replace the forward method
        module.forward = patched_forward.__get__(module, type(module))
    
    def _patch_t5(self, name: str, module: nn.Module, hsa_module: HSAAttention):
        """Patch a T5 attention module."""
        # For T5, similar to others, we replace the forward method
        original_forward = module.forward
        
        def patched_forward(
            self,
            hidden_states,
            mask=None,
            key_value_states=None,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            query_length=None,
            use_cache=False,
            output_attentions=False,
            **kwargs
        ):
            # Use hidden_states for Q/K/V
            query = hidden_states
            key = hidden_states if key_value_states is None else key_value_states
            value = hidden_states if key_value_states is None else key_value_states
            
            # Forward to HSA module
            output, attention_weights = hsa_module(query, key, value, mask)
            
            outputs = (output,)
            if output_attentions:
                outputs += (attention_weights,)
            if use_cache:
                outputs += (None,)  # No cache when using HSA
                
            return outputs
        
        # Replace the forward method
        module.forward = patched_forward.__get__(module, type(module))
    
    def _patch_generic(self, name: str, module: nn.Module, hsa_module: HSAAttention):
        """Generic patching for unknown model types."""
        original_forward = module.forward
        
        # Get original method signature
        sig = inspect.signature(original_forward)
        
        def patched_forward(*args, **kwargs):
            try:
                # Try to identify query, key, value from args/kwargs
                query = None
                key = None
                value = None
                attention_mask = None
                
                # Extract from args if available (common pattern)
                if len(args) >= 1:
                    query = args[0]
                    if len(args) >= 3:
                        key, value = args[1:3]
                    if len(args) >= 4:
                        attention_mask = args[3]
                
                # Extract from common kwarg names
                if query is None:
                    query = kwargs.get('query', kwargs.get('hidden_states', kwargs.get('input', None)))
                    
                if key is None:
                    key = kwargs.get('key', kwargs.get('key_states', query))
                    
                if value is None:
                    value = kwargs.get('value', kwargs.get('value_states', key))
                    
                if attention_mask is None:
                    attention_mask = kwargs.get('attention_mask', kwargs.get('attn_mask', None))
                
                # If we couldn't extract the necessary tensors or query is not a tensor,
                # fall back to original
                if query is None or not isinstance(query, torch.Tensor):
                    logger.warning(f"Falling back to original attention for {name} - couldn't extract inputs")
                    return original_forward(*args, **kwargs)
                
                # Forward to HSA module
                output, attention_weights = hsa_module(query, key, value, attention_mask)
                
                # Try to match original return signature
                # Check if original returns tuple
                try:
                    dummy_args = []
                    for i, param in enumerate(sig.parameters.values()):
                        if i < len(args):
                            dummy_args.append(args[i])
                        elif param.name in kwargs:
                            continue
                        elif param.default != inspect.Parameter.empty:
                            continue
                        else:
                            # Try to provide a dummy value of the expected type
                            if 'tensor' in param.name.lower() or 'state' in param.name.lower():
                                dummy_args.append(query)  # Reuse query as dummy
                            else:
                                dummy_args.append(None)
                    
                    # This is a safe way to check return type without executing
                    returns_tuple = 'tuple' in str(sig.return_annotation)
                    if returns_tuple:
                        return (output, attention_weights)
                    else:
                        return output
                except Exception:
                    # If checking fails, default to returning just the output
                    return output
                
            except Exception as e:
                logger.warning(f"Error in HSA processing: {e}. Falling back to original.")
                # Fall back to original in case of error
                return original_forward(*args, **kwargs)
        
        # Replace the forward method
        module.forward = patched_forward.__get__(module, type(module))
    
    def restore_model(self):
        """Restore the model to its original state."""
        for name, original in self.original_modules.items():
            try:
                # Find the module
                parts = name.split('.')
                parent_path, target = '.'.join(parts[:-1]), parts[-1]
                
                if parent_path:
                    parent = self.model.get_submodule(parent_path)
                    # Restore original forward method
                    module = getattr(parent, target)
                    module.forward = original["forward"]
                    
                    # For GPT-2, also restore _attn if it was patched
                    if self.model_type == "gpt2" and hasattr(module, "_attn"):
                        if hasattr(original["module"], "_attn"):
                            module._attn = original["module"]._attn
                else:
                    # Top-level module
                    module = getattr(self.model, target)
                    module.forward = original["forward"]
                
                logger.info(f"Restored original module: {name}")
            except Exception as e:
                logger.warning(f"Failed to restore module {name}: {e}")
        
        # Clear patched modules
        self.patched_modules.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about patched modules and HSA usage."""
        stats = {
            "total_patched": len(self.patched_modules),
            "attention_stats": {},
            "model_type": self.model_type
        }
        
        # Collect attention stats from all modules
        for name, patched in self.patched_modules.items():
            if hasattr(patched["hsa_module"], "attention_stats"):
                module_stats = patched["hsa_module"].attention_stats
                for key, value in module_stats.items():
                    if key not in stats["attention_stats"]:
                        stats["attention_stats"][key] = 0
                    stats["attention_stats"][key] += value
        
        return stats


def create_adapter_for_model(
    model: PreTrainedModel,
    use_sparse: bool = True,
    adaptation_enabled: bool = True,
    hierarchy_levels: Optional[List[str]] = None,
    max_context_length: Optional[int] = None
) -> Tuple[ModelPatcher, SplatRegistry]:
    """Create an HSA adapter for a model.
    
    Args:
        model: The transformer model to adapt
        use_sparse: Whether to use sparse attention computation
        adaptation_enabled: Whether to enable HSA adaptation
        hierarchy_levels: Optional custom hierarchy levels
        max_context_length: Optional maximum context length to support
        
    Returns:
        Tuple of (model_patcher, registry)
    """
    # Determine embedding dimension
    embedding_dim = 768  # Default fallback
    
    if hasattr(model.config, "hidden_size"):
        embedding_dim = model.config.hidden_size
    elif hasattr(model.config, "d_model"):
        embedding_dim = model.config.d_model
    elif hasattr(model.config, "n_embd"):
        embedding_dim = model.config.n_embd
    
    logger.info(f"Using embedding dimension: {embedding_dim}")
    
    # Create hierarchy
    if hierarchy_levels is None:
        hierarchy_levels = ["token", "word", "phrase", "sentence", "document"]
    
    # Adjust init splats per level based on embedding dimension
    base_splats = [int(embedding_dim * 0.2), int(embedding_dim * 0.1), 
                   int(embedding_dim * 0.05), int(embedding_dim * 0.02), 5]
    
    # Make sure we have enough values
    while len(base_splats) < len(hierarchy_levels):
        base_splats.append(5)  # Default for additional levels
    
    # Trim if we have too many
    init_splats_per_level = base_splats[:len(hierarchy_levels)]
    
    # Create hierarchy
    hierarchy = Hierarchy(
        levels=hierarchy_levels,
        init_splats_per_level=init_splats_per_level
    )
    
    # Create registry
    registry = SplatRegistry(hierarchy, embedding_dim)
    
    # Create attention computer
    attention_config = AttentionConfig(
        normalize_rows=True,
        causal=True,  # Enable causal attention for language models
        use_sparse=use_sparse
    )
    
    if use_sparse:
        attention_computer = SparseAttentionComputer(config=attention_config)
    else:
        attention_computer = DenseAttentionComputer(config=attention_config)
    
    # Create adapter config
    adapter_config = {
        "embed_dim": embedding_dim,
        "use_projections": False,  # Use model's original projections
        "return_attention_weights": True,
        "dropout": getattr(model.config, "attention_dropout", 0.1),
        "max_context_length": max_context_length
    }
    
    # Create patcher
    patcher = ModelPatcher(
        model=model,
        registry=registry,
        attention_computer=attention_computer,
        config=adapter_config
    )
    
    # Initialize registry with splats
    registry.initialize_splats()
    
    # Create adaptation controller if enabled
    if adaptation_enabled:
        try:
            from hsa.adaptation_metrics_base import AdaptationMetricsComputer, SplatCandidateEvaluator
            from hsa.attention_info_metrics import InfoTheoreticMetricsComputer, InfoTheoreticCandidateEvaluator
            from hsa.adaptation_controller import AdaptationController
            
            metrics_computer = InfoTheoreticMetricsComputer()
            candidate_evaluator = InfoTheoreticCandidateEvaluator(metrics_computer)
            
            adaptation_controller = AdaptationController(
                registry=registry,
                metrics_computer=metrics_computer,
                candidate_evaluator=candidate_evaluator
            )
            
            # Store adaptation controller in patcher for later access
            patcher.adaptation_controller = adaptation_controller
        except ImportError:
            logger.warning("Could not import adaptation modules, adaptation disabled")
    
    return patcher, registry


class ContextExtender:
    """Context window extension utility for HSA-enhanced models."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer,
        patcher: ModelPatcher,
        registry: SplatRegistry
    ):
        """Initialize context extender.
        
        Args:
            model: The transformer model
            tokenizer: The tokenizer
            patcher: ModelPatcher instance
            registry: SplatRegistry instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.patcher = patcher
        self.registry = registry
        
        # Get model's default context length
        self.original_context_length = 1024  # Default fallback
        
        if hasattr(model.config, "max_position_embeddings"):
            self.original_context_length = model.config.max_position_embeddings
        elif hasattr(model.config, "n_positions"):
            self.original_context_length = model.config.n_positions
        elif hasattr(model.config, "n_ctx"):
            self.original_context_length = model.config.n_ctx
        
        # Store the original position embeddings if any
        self.original_embeddings = None
        self._store_original_embeddings()
        
        # Context extension statistics
        self.extension_stats = {
            "original_length": self.original_context_length,
            "current_length": self.original_context_length,
            "longest_sequence": 0
        }
    
    def _store_original_embeddings(self):
        """Store original position embeddings for later restoration."""
        # Different models store position embeddings differently
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "wpe"):
            # GPT-2 style
            self.original_embeddings = {
                "name": "transformer.wpe",
                "embeddings": self.model.transformer.wpe.weight.clone()
            }
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "embed_positions"):
            # GPT-Neo style
            self.original_embeddings = {
                "name": "gpt_neox.embed_positions",
                "embeddings": self.model.gpt_neox.embed_positions.weight.clone()
            }
        elif hasattr(self.model, "model") and hasattr(self.model.model, "embed_positions"):
            # OPT style
            self.original_embeddings = {
                "name": "model.embed_positions",
                "embeddings": self.model.model.embed_positions.weight.clone()
            }
        elif hasattr(self.model, "model") and hasattr(self.model.model, "decoder") and hasattr(self.model.model.decoder, "embed_positions"):
            # Newer OPT/Llama style
            self.original_embeddings = {
                "name": "model.decoder.embed_positions",
                "embeddings": self.model.model.decoder.embed_positions.weight.clone()
            }
        else:
            logger.warning("Could not find position embeddings to extend")
    
    def extend_context(self, target_length: int) -> bool:
        """Extend the model's context window.
        
        Args:
            target_length: Desired context length
            
        Returns:
            True if successful, False otherwise
        """
        # Skip if already extended enough
        if self.extension_stats["current_length"] >= target_length:
            return True
        
        # Skip if we don't have original embeddings
        if self.original_embeddings is None:
            logger.warning("Cannot extend context - no original embeddings found")
            return False
        
        try:
            # Different models need different extensions
            if "transformer.wpe" == self.original_embeddings["name"]:
                # GPT-2 style
                return self._extend_gpt2(target_length)
            elif "gpt_neox.embed_positions" == self.original_embeddings["name"]:
                # GPT-Neo style
                return self._extend_gpt_neo(target_length)
            elif "model.embed_positions" == self.original_embeddings["name"]:
                # OPT style
                return self._extend_opt(target_length)
            elif "model.decoder.embed_positions" == self.original_embeddings["name"]:
                # Newer OPT/Llama style
                return self._extend_llama(target_length)
            else:
                logger.warning(f"Unsupported embedding type: {self.original_embeddings['name']}")
                return False
                
        except Exception as e:
            logger.error(f"Error extending context: {e}")
            return False
    
    def _extend_gpt2(self, target_length: int) -> bool:
        """Extend context for GPT-2 style models."""
        original = self.original_embeddings["embeddings"]
        original_len = original.shape[0]
        embed_dim = original.shape[1]
        
        # Create extended embeddings
        extended = torch.zeros((target_length, embed_dim), device=original.device, dtype=original.dtype)
        
        # Copy original embeddings
        extended[:original_len] = original
        
        # Extrapolate remaining positions
        if target_length > original_len:
            # Simple linear extrapolation
            for i in range(original_len, target_length):
                # Use the closest original embedding as base
                base_idx = min(i % original_len, original_len - 1)
                extended[i] = original[base_idx]
        
        # Update embeddings
        self.model.transformer.wpe.weight = nn.Parameter(extended)
        
        # Update model config
        self.model.config.max_position_embeddings = target_length
        self.model.config.n_positions = target_length
        self.model.config.n_ctx = target_length
        
        # Update stats
        self.extension_stats["current_length"] = target_length
        
        logger.info(f"Extended context to {target_length} (from {original_len})")
        return True
    
    def _extend_gpt_neo(self, target_length: int) -> bool:
        """Extend context for GPT-Neo style models."""
        original = self.original_embeddings["embeddings"]
        original_len = original.shape[0]
        embed_dim = original.shape[1]
        
        # Create extended embeddings
        extended = torch.zeros((target_length, embed_dim), device=original.device, dtype=original.dtype)
        
        # Copy original embeddings
        extended[:original_len] = original
        
        # Extrapolate remaining positions
        if target_length > original_len:
            # Use rotary pattern for extrapolation (mimics sinusoidal pattern)
            for i in range(original_len, target_length):
                # Cycle through original embeddings with a slight scaling
                base_idx = i % original_len
                scale_factor = 1.0 + 0.001 * (i // original_len)
                extended[i] = original[base_idx] * scale_factor
        
        # Update embeddings
        self.model.gpt_neox.embed_positions.weight = nn.Parameter(extended)
        
        # Update model config
        self.model.config.max_position_embeddings = target_length
        if hasattr(self.model.config, "n_positions"):
            self.model.config.n_positions = target_length
        if hasattr(self.model.config, "n_ctx"):
            self.model.config.n_ctx = target_length
        
        # Update stats
        self.extension_stats["current_length"] = target_length
        
        logger.info(f"Extended context to {target_length} (from {original_len})")
        return True
    
    def _extend_opt(self, target_length: int) -> bool:
        """Extend context for OPT style models."""
        original = self.original_embeddings["embeddings"]
        original_len = original.shape[0]
        embed_dim = original.shape[1]
        
        # Create extended embeddings
        extended = torch.zeros((target_length, embed_dim), device=original.device, dtype=original.dtype)
        
        # Copy original embeddings
        extended[:original_len] = original
        
        # Extrapolate remaining positions
        if target_length > original_len:
            for i in range(original_len, target_length):
                # Simple extrapolation - modified to avoid exact copies
                base_idx = min(i % original_len, original_len - 1)
                position_factor = float(i) / float(original_len)
                extended[i] = original[base_idx] * (1.0 + 0.01 * position_factor)
        
        # Update embeddings
        self.model.model.embed_positions.weight = nn.Parameter(extended)
        
        # Update model config
        self.model.config.max_position_embeddings = target_length
        
        # Update stats
        self.extension_stats["current_length"] = target_length
        
        logger.info(f"Extended context to {target_length} (from {original_len})")
        return True
    
    def _extend_llama(self, target_length: int) -> bool:
        """Extend context for newer OPT/Llama style models."""
        original = self.original_embeddings["embeddings"]
        original_len = original.shape[0]
        embed_dim = original.shape[1]
        
        # Create extended embeddings
        extended = torch.zeros((target_length, embed_dim), device=original.device, dtype=original.dtype)
        
        # Copy original embeddings
        extended[:original_len] = original
        
        # Extrapolate remaining positions using a more sophisticated approach
        if target_length > original_len:
            # Linear combination of existing embeddings
            for i in range(original_len, target_length):
                phase = (i - original_len) / (target_length - original_len)
                idx1 = int(phase * original_len)
                idx2 = min(idx1 + 1, original_len - 1)
                ratio = phase * original_len - idx1
                
                extended[i] = original[idx1] * (1 - ratio) + original[idx2] * ratio
        
        # Update embeddings
        self.model.model.decoder.embed_positions.weight = nn.Parameter(extended)
        
        # Update model config
        self.model.config.max_position_embeddings = target_length
        
        # Update stats
        self.extension_stats["current_length"] = target_length
        
        logger.info(f"Extended context to {target_length} (from {original_len})")
        return True
    
    def restore_original_context(self) -> bool:
        """Restore original context length and embeddings."""
        if self.original_embeddings is None:
            return False
        
        try:
            # Restore embeddings based on model type
            if "transformer.wpe" == self.original_embeddings["name"]:
                self.model.transformer.wpe.weight = nn.Parameter(self.original_embeddings["embeddings"])
            elif "gpt_neox.embed_positions" == self.original_embeddings["name"]:
                self.model.gpt_neox.embed_positions.weight = nn.Parameter(self.original_embeddings["embeddings"])
            elif "model.embed_positions" == self.original_embeddings["name"]:
                self.model.model.embed_positions.weight = nn.Parameter(self.original_embeddings["embeddings"])
            elif "model.decoder.embed_positions" == self.original_embeddings["name"]:
                self.model.model.decoder.embed_positions.weight = nn.Parameter(self.original_embeddings["embeddings"])
            
            # Restore model config
            self.model.config.max_position_embeddings = self.original_context_length
            if hasattr(self.model.config, "n_positions"):
                self.model.config.n_positions = self.original_context_length
            if hasattr(self.model.config, "n_ctx"):
                self.model.config.n_ctx = self.original_context_length
            
            # Update stats
            self.extension_stats["current_length"] = self.original_context_length
            
            logger.info(f"Restored original context length: {self.original_context_length}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring original context: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context extension statistics."""
        return self.extension_stats
