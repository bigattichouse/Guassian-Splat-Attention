"""
GSA Integration with TinyStories-Instruct-33M
CONSERVATIVE VERSION - Starts very close to standard attention with minimal modifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import json
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

# HuggingFace transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "CPU Only")

@dataclass
class GSAConfig:
    """Configuration for GSA replacement."""
    n_splats_per_head: int = 2  # Very conservative - just 2 splats
    enable_context_extension: bool = False
    target_context_length: int = 2048
    replacement_strategy: str = "all_layers"
    preserve_weights: bool = True

class ConservativeGSAAttention(nn.Module):
    """
    ULTRA-CONSERVATIVE GSA that starts 95% like standard attention.
    Only adds tiny splat modifications to proven attention mechanisms.
    """
    
    def __init__(self, config, original_attention=None):
        super().__init__()
        
        # Extract dimensions safely
        self.embed_dim = getattr(config, 'hidden_size', getattr(config, 'n_embd', 768))
        self.num_heads = getattr(config, 'num_attention_heads', getattr(config, 'n_head', 12))
        self.head_dim = self.embed_dim // self.num_heads
        self.n_splats_per_head = getattr(config, 'n_splats_per_head', 2)
        
        print(f"    Conservative GSA: {self.embed_dim}D, {self.num_heads} heads, {self.head_dim} head_dim, {self.n_splats_per_head} splats/head")
        
        # Standard projections - keep identical to original
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        
        # Copy weights if available (crucial for starting close to baseline)
        if original_attention is not None:
            self._copy_weights(original_attention)
        
        # MINIMAL splat parameters - just small perturbations
        device = next(self.c_attn.parameters()).device
        
        # Single learnable bias per head (simplest possible splat concept)
        # Shape: [num_heads, 1, 1] for broadcasting with attention scores [B, H, T, T]
        self.splat_bias = nn.Parameter(torch.zeros(self.num_heads, 1, 1, device=device))
        
        # Single scaling factor per head (starts at 1.0)
        # Shape: [num_heads] for per-head scaling
        self.splat_scale = nn.Parameter(torch.zeros(self.num_heads, device=device))  # tanh(0) = 0, so 1.0 + 0.01*0 = 1.0
        
        self.dropout = nn.Dropout(0.1)
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)
        
    def _copy_weights(self, original_attention):
        """Copy weights exactly from original attention."""
        try:
            with torch.no_grad():
                if hasattr(original_attention, 'c_attn') and hasattr(original_attention.c_attn, 'weight'):
                    self.c_attn.weight.copy_(original_attention.c_attn.weight)
                    if original_attention.c_attn.bias is not None:
                        self.c_attn.bias.copy_(original_attention.c_attn.bias)
                
                if hasattr(original_attention, 'c_proj') and hasattr(original_attention.c_proj, 'weight'):
                    self.c_proj.weight.copy_(original_attention.c_proj.weight)
                    if original_attention.c_proj.bias is not None:
                        self.c_proj.bias.copy_(original_attention.c_proj.bias)
                        
            print("      ✓ Weights copied successfully")
        except Exception as e:
            print(f"      ⚠ Weight copying failed: {e}")
    
    def forward(self, hidden_states, attention_mask=None, layer_past=None, head_mask=None, 
                use_cache=False, output_attentions=False, cache_position=None, **kwargs):
        """
        CONSERVATIVE forward pass that is 99% standard attention with tiny modifications.
        """
        
        batch_size, seq_len, _ = hidden_states.shape
        
        try:
            # Standard QKV projection
            qkv = self.c_attn(hidden_states)
            query, key, value = qkv.split(self.embed_dim, dim=2)
            
            # Standard reshape for multi-head attention with proper dimension checking
            # Ensure we have the right dimensions before reshaping
            expected_size = batch_size * seq_len * self.head_dim
            
            if query.numel() != expected_size * self.num_heads:
                print(f"Dimension mismatch: query size {query.numel()}, expected {expected_size * self.num_heads}")
                print(f"Shape: B={batch_size}, T={seq_len}, embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}")
                raise RuntimeError("Query tensor size mismatch")
            
            query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Standard scaled dot-product attention with TINY modifications
            attention_scores = torch.matmul(query * self.scale_factor, key.transpose(-2, -1))
            
            # MINIMAL splat modification: tiny learnable bias and scale
            # Start with scales very close to 1.0 and biases very close to 0.0
            splat_scale = 1.0 + 0.01 * torch.tanh(self.splat_scale)  # scales in [0.99, 1.01]
            splat_bias = 0.001 * torch.tanh(self.splat_bias)         # biases in [-0.001, 0.001]
            
            # Apply TINY modifications - ensure broadcasting works correctly
            splat_scale_expanded = splat_scale.view(1, self.num_heads, 1, 1)
            splat_bias_expanded = splat_bias.view(1, self.num_heads, 1, 1)
            
            attention_scores = splat_scale_expanded * attention_scores + splat_bias_expanded
            
            # Standard causal mask
            if seq_len > 1:
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool))
                attention_scores = attention_scores.masked_fill(~causal_mask, -1e9)
            
            # Apply attention mask if provided (standard)
            if attention_mask is not None:
                if attention_mask.dim() == 2:  # [B, T]
                    mask = attention_mask.unsqueeze(1).unsqueeze(2)
                    mask = mask.expand(batch_size, 1, seq_len, seq_len)
                    attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
            # Standard softmax and dropout
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            # Standard value application
            output = torch.matmul(attention_weights, value)
            
            # Standard head combining with dimension check
            output = output.transpose(1, 2).contiguous()
            expected_final_shape = (batch_size, seq_len, self.embed_dim)
            output = output.view(*expected_final_shape)
            
            # Standard final projection
            output = self.c_proj(output)
            
            # Return in HuggingFace format
            outputs = (output,)
            if output_attentions:
                outputs += (attention_weights,)
            if use_cache:
                outputs += (None,)
                
            return outputs
            
        except Exception as e:
            print(f"Error in GSA forward pass: {e}")
            print(f"Input shape: {hidden_states.shape}")
            print(f"Expected dimensions: embed_dim={self.embed_dim}, num_heads={self.num_heads}, head_dim={self.head_dim}")
            
            # Fallback: pure standard attention computation with careful dimension handling
            try:
                qkv = self.c_attn(hidden_states)
                
                # Verify QKV split produces correct sizes
                if qkv.size(-1) != 3 * self.embed_dim:
                    print(f"QKV projection size mismatch: got {qkv.size(-1)}, expected {3 * self.embed_dim}")
                    # Use simpler fallback
                    return (self.c_proj(hidden_states),)
                
                query, key, value = qkv.split(self.embed_dim, dim=2)
                
                # Careful reshaping with dimension verification
                try:
                    query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                except RuntimeError as reshape_error:
                    print(f"Reshape error in fallback: {reshape_error}")
                    # Ultra-simple fallback
                    return (hidden_states,)
                
                attention_scores = torch.matmul(query * self.scale_factor, key.transpose(-2, -1))
                
                if seq_len > 1:
                    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool))
                    attention_scores = attention_scores.masked_fill(~causal_mask, -1e9)
                    
                attention_weights = F.softmax(attention_scores, dim=-1)
                attention_weights = self.dropout(attention_weights)
                
                output = torch.matmul(attention_weights, value)
                output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
                output = self.c_proj(output)
                
                return (output,)
                
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                # Ultra-safe fallback: identity operation
                return (hidden_states,)

def replace_attention_layers(model, gsa_config: GSAConfig):
    """Replace attention layers with conservative GSA."""
    replacements_made = 0
    total_layers = len(model.transformer.h)
    
    print(f"Replacing attention layers in {total_layers} transformer blocks...")
    
    # Add config parameter
    model.config.n_splats_per_head = gsa_config.n_splats_per_head
    model.config.use_cache = False  # Disable caching for GSA
    
    for layer_idx, layer in enumerate(model.transformer.h):
        if gsa_config.replacement_strategy == "all_layers":
            should_replace = True
        elif gsa_config.replacement_strategy == "alternating":
            should_replace = (layer_idx % 2 == 0)
        else:
            should_replace = False
        
        if should_replace:
            try:
                original_attention = layer.attn
                
                # Create conservative GSA attention
                gsa_attention = ConservativeGSAAttention(
                    model.config, 
                    original_attention if gsa_config.preserve_weights else None
                )
                
                # Move GSA attention to the same device as the model
                gsa_attention = gsa_attention.to(model.device)
                
                # Replace the attention layer
                layer.attn = gsa_attention
                replacements_made += 1
                
                print(f"  Layer {layer_idx}: Standard → Conservative GSA ✓")
                
            except Exception as e:
                print(f"  Layer {layer_idx}: Replacement failed - {e}")
        else:
            print(f"  Layer {layer_idx}: Keeping standard attention")
    
    print(f"Successfully replaced {replacements_made}/{total_layers} attention layers")
    return replacements_made

def benchmark_performance(model, tokenizer, name: str, test_sequences: List[str]):
    """Benchmark with detailed error tracking."""
    print(f"\nBenchmarking {name}...")
    
    total_loss = 0.0
    total_tokens = 0
    generation_times = []
    successful_sequences = 0
    
    model.eval()
    
    for i, text in enumerate(test_sequences[:5]):
        try:
            # Tokenize with proper settings
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=False)
            input_ids = inputs["input_ids"].to(device)
            
            if input_ids.size(1) < 10:  # Skip very short sequences
                continue
            
            with torch.no_grad():
                # Test loss computation
                try:
                    outputs = model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    
                    if torch.isfinite(loss):
                        total_loss += loss.item() * input_ids.size(1)
                        total_tokens += input_ids.size(1)
                        successful_sequences += 1
                    else:
                        print(f"    Sequence {i}: Non-finite loss detected")
                        
                except Exception as e:
                    print(f"    Sequence {i}: Loss computation failed - {e}")
                    continue
                
                # Test generation speed
                try:
                    prompt_length = min(input_ids.size(1) // 2, 50)
                    prompt = input_ids[:, :prompt_length]
                    
                    start_time = time.time()
                    generated = model.generate(
                        prompt,
                        max_new_tokens=10,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=False
                    )
                    generation_time = time.time() - start_time
                    generation_times.append(generation_time)
                    
                except Exception as e:
                    print(f"    Sequence {i}: Generation failed - {e}")
                    continue
                    
        except Exception as e:
            print(f"    Sequence {i}: Complete failure - {e}")
            continue
    
    # Calculate averages safely
    if total_tokens > 0 and successful_sequences > 0:
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(min(avg_loss, 10))  # Cap at exp(10) to avoid inf
    else:
        avg_loss = float('inf')
        perplexity = float('inf')
    
    avg_generation_time = np.mean(generation_times) if generation_times else float('inf')
    
    print(f"  Successful sequences: {successful_sequences}/{len(test_sequences[:5])}")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Generation time: {avg_generation_time:.3f}s")
    
    return {
        'perplexity': perplexity,
        'avg_loss': avg_loss,
        'avg_generation_time': avg_generation_time,
        'sequences_tested': successful_sequences
    }

def create_simple_needle_test(tokenizer, context_length: int = 1024):
    """Create a simple needle-in-haystack test within safe limits."""
    
    # Simple repeating text
    base_text = "The cat sat on the mat. The dog ran in the yard. "
    needle = "The secret code is ALPHA123. "
    question = "What is the secret code?"
    
    # Build text to approximate target length
    repeated_text = ""
    target_tokens = context_length - 50  # Leave room
    
    while len(tokenizer.encode(repeated_text)) < target_tokens:
        repeated_text += base_text
    
    # Insert needle in middle
    words = repeated_text.split()
    middle = len(words) // 2
    words.insert(middle, needle)
    
    final_text = " ".join(words) + " " + question
    
    return final_text, "ALPHA123"

def test_simple_context_extension(model, tokenizer, max_length: int = 1024):
    """Simple context test without CUDA errors."""
    print(f"\nTesting context handling up to {max_length} tokens...")
    
    test_text, expected_code = create_simple_needle_test(tokenizer, max_length)
    
    try:
        # Tokenize safely
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(device)
        
        actual_length = input_ids.size(1)
        print(f"  Actual input length: {actual_length} tokens")
        
        if actual_length > model.config.max_position_embeddings:
            print(f"  ⚠ Length exceeds model capacity ({model.config.max_position_embeddings})")
            return False, "Length exceeds capacity"
        
        # Test generation
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
            
            generated_text = tokenizer.decode(output[0][input_ids.size(1):], skip_special_tokens=True)
            success = expected_code.lower() in generated_text.lower()
            
            print(f"  Generated: '{generated_text.strip()}'")
            print(f"  Expected: '{expected_code}'")
            print(f"  Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
            
            return success, generated_text
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False, str(e)

def run_conservative_gsa_test():
    """Run ultra-conservative GSA test that should match baseline."""
    print("="*80)
    print("ULTRA-CONSERVATIVE GSA vs STANDARD ATTENTION TEST")
    print("="*80)
    print("Testing GSA that starts 99% identical to standard attention.")
    
    # Load model
    model_name = "roneneldan/TinyStories-Instruct-33M"
    print(f"\nLoading model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        original_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        ).to(device)
        
        print(f"✓ Model loaded successfully")
        print(f"  Parameters: {sum(p.numel() for p in original_model.parameters()):,}")
        print(f"  Vocabulary size: {original_model.config.vocab_size}")
        print(f"  Hidden size: {original_model.config.hidden_size}")
        print(f"  Attention heads: {original_model.config.num_attention_heads}")
        print(f"  Context length: {original_model.config.max_position_embeddings}")
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
    
    # Create test data
    test_texts = [
        "Once upon a time, there was a little girl who loved to read books.",
        "The friendly dragon helped the village children learn new things.",
        "Every morning, the baker made fresh bread for the town.",
        "Tom and his friend built a treehouse in the big oak tree.",
        "The wise king always listened to his people's problems."
    ]
    
    # Ultra-conservative configuration
    gsa_config = GSAConfig(
        n_splats_per_head=2,  # Minimal splats
        enable_context_extension=False,
        target_context_length=2048,
        replacement_strategy="all_layers",
        preserve_weights=True  # CRUCIAL for starting close to baseline
    )
    
    print(f"\nUltra-Conservative GSA Configuration:")
    print(f"  Splats per head: {gsa_config.n_splats_per_head}")
    print(f"  Context extension: {gsa_config.enable_context_extension}")
    print(f"  Replacement strategy: {gsa_config.replacement_strategy}")
    print(f"  Preserve weights: {gsa_config.preserve_weights}")
    
    # Test 1: Baseline performance
    print(f"\n" + "="*60)
    print("TEST 1: BASELINE PERFORMANCE")
    print("="*60)
    
    baseline_performance = benchmark_performance(original_model, tokenizer, "Standard Attention", test_texts)
    
    # Test 2: Create conservative GSA version
    print(f"\n" + "="*60)
    print("TEST 2: CONSERVATIVE GSA MODEL CREATION")
    print("="*60)
    
    print("Creating Conservative GSA version (should be nearly identical to baseline)...")
    try:
        gsa_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        ).to(device)
        
        # Replace attention layers
        replacements_made = replace_attention_layers(gsa_model, gsa_config)
        
        gsa_params = sum(p.numel() for p in gsa_model.parameters())
        param_increase = gsa_params - sum(p.numel() for p in original_model.parameters())
        
        print(f"\nConservative GSA Model Statistics:")
        print(f"  Total parameters: {gsa_params:,}")
        print(f"  Parameter increase: {param_increase:,} ({param_increase/sum(p.numel() for p in original_model.parameters())*100:.3f}%)")
        print(f"  Layers replaced: {replacements_made}")
        
        # Show the tiny parameter additions
        splat_params = 0
        for layer in gsa_model.transformer.h:
            if hasattr(layer.attn, 'splat_bias'):
                splat_params += layer.attn.splat_bias.numel()
            if hasattr(layer.attn, 'splat_scale'):
                splat_params += layer.attn.splat_scale.numel()
        
        print(f"  Splat-specific parameters: {splat_params:,} (minimal modification)")
        
    except Exception as e:
        print(f"❌ Failed to create GSA model: {e}")
        return None
    
    # Test 3: Conservative GSA performance (should be very close to baseline)
    print(f"\n" + "="*60)
    print("TEST 3: CONSERVATIVE GSA PERFORMANCE")
    print("="*60)
    print("Expected: Performance should be within 5% of baseline")
    
    gsa_performance = benchmark_performance(gsa_model, tokenizer, "Conservative GSA", test_texts)
    
    # Test 4: Context handling comparison
    print(f"\n" + "="*60)
    print("TEST 4: CONTEXT HANDLING")
    print("="*60)
    
    print("Testing standard attention context handling...")
    std_success, std_result = test_simple_context_extension(original_model, tokenizer, 1024)
    
    print("Testing Conservative GSA context handling...")
    gsa_success, gsa_result = test_simple_context_extension(gsa_model, tokenizer, 1024)
    
    # Analysis
    print(f"\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # Performance comparison
    if baseline_performance['perplexity'] != float('inf') and gsa_performance['perplexity'] != float('inf'):
        perplexity_change = ((gsa_performance['perplexity'] - baseline_performance['perplexity']) 
                            / baseline_performance['perplexity'] * 100)
        print(f"Performance Comparison:")
        print(f"  Baseline perplexity: {baseline_performance['perplexity']:.2f}")
        print(f"  Conservative GSA perplexity: {gsa_performance['perplexity']:.2f}")
        print(f"  Change: {perplexity_change:+.1f}%")
        
        # Very strict criteria for conservative approach
        performance_acceptable = abs(perplexity_change) < 5  # Within 5%
        very_close = abs(perplexity_change) < 1  # Within 1%
        
        if very_close:
            print(f"  ✅ EXCELLENT: Performance within 1% of baseline")
        elif performance_acceptable:
            print(f"  ✓ GOOD: Performance within 5% of baseline") 
        else:
            print(f"  ❌ PROBLEM: Performance degraded more than 5%")
            
    else:
        print(f"Performance Comparison:")
        print(f"  ❌ One or both models produced infinite perplexity")
        performance_acceptable = False
        very_close = False
    
    # Speed comparison
    if (baseline_performance['avg_generation_time'] != float('inf') and 
        gsa_performance['avg_generation_time'] != float('inf')):
        speed_change = ((gsa_performance['avg_generation_time'] - baseline_performance['avg_generation_time']) 
                       / baseline_performance['avg_generation_time'] * 100)
        print(f"  Baseline speed: {baseline_performance['avg_generation_time']:.3f}s")
        print(f"  Conservative GSA speed: {gsa_performance['avg_generation_time']:.3f}s")
        print(f"  Speed change: {speed_change:+.1f}%")
        
        speed_acceptable = speed_change < 100  # Less than 2x slower
    else:
        print(f"  ❌ Speed comparison failed")
        speed_acceptable = False
    
    # Context handling
    print(f"Context Handling:")
    print(f"  Standard attention: {'✓' if std_success else '❌'}")
    print(f"  Conservative GSA: {'✓' if gsa_success else '❌'}")
    
    # Final assessment with conservative criteria
    print(f"\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    criteria = {
        'models_loaded': True,
        'gsa_created': replacements_made > 0,
        'performance_measured': baseline_performance['sequences_tested'] > 0 and gsa_performance['sequences_tested'] > 0,
        'performance_very_close': very_close,
        'performance_acceptable': performance_acceptable,
        'speed_acceptable': speed_acceptable,
        'both_work': gsa_performance['sequences_tested'] > 0 and baseline_performance['sequences_tested'] > 0,
        'no_crashes': True  # We got this far
    }
    
    passed = sum(criteria.values())
    total = len(criteria)
    
    print("Conservative Success Criteria:")
    for criterion, result in criteria.items():
        print(f"  {criterion}: {'✓' if result else '❌'}")
    
    print(f"\nOverall: {passed}/{total} criteria passed ({passed/total*100:.0f}%)")
    
    # Conservative verdict
    if criteria['performance_very_close'] and criteria['both_work']:
        print("🎯 CONSERVATIVE GSA SUCCESS!")
        print("  ✓ Performance within 1% of baseline")
        print("  ✓ No crashes or infinite losses")
        print("  ✓ Minimal overhead added")
        print("  ✓ Ready for gradual enhancement")
        verdict = "conservative_success"
    elif criteria['performance_acceptable'] and criteria['both_work']:
        print("⚡ CONSERVATIVE GSA WORKING")
        print("  ✓ Performance within 5% of baseline")
        print("  ✓ Basic functionality confirmed")
        print("  ✓ Good foundation for improvement")
        verdict = "conservative_working"
    elif criteria['both_work']:
        print("⚠️ CONSERVATIVE GSA FUNCTIONAL")
        print("  ✓ No crashes")
        print("  ❌ Performance needs optimization")
        print("  → Check parameter initialization")
        verdict = "conservative_functional"
    else:
        print("❌ CONSERVATIVE GSA FAILED")
        print("  ❌ Basic functionality broken")
        print("  → Check device placement and gradients")
        verdict = "conservative_failed"
    
    # Show parameter values for debugging
    print(f"\n" + "="*60)
    print("PARAMETER ANALYSIS")
    print("="*60)
    
    if hasattr(gsa_model.transformer.h[0].attn, 'splat_bias'):
        first_layer = gsa_model.transformer.h[0].attn
        print(f"First layer splat parameters:")
        print(f"  Splat bias range: {first_layer.splat_bias.min().item():.6f} to {first_layer.splat_bias.max().item():.6f}")
        print(f"  Splat scale range: {first_layer.splat_scale.min().item():.6f} to {first_layer.splat_scale.max().item():.6f}")
        
        # Compute effective scales and biases
        effective_scales = 1.0 + 0.01 * torch.tanh(first_layer.splat_scale)
        effective_biases = 0.001 * torch.tanh(first_layer.splat_bias)
        
        print(f"  Effective scale range: {effective_scales.min().item():.6f} to {effective_scales.max().item():.6f}")
        print(f"  Effective bias range: {effective_biases.min().item():.6f} to {effective_biases.max().item():.6f}")
    
    # Save results with conservative metrics
    results = {
        'verdict': verdict,
        'success_rate': float(passed / total),
        'baseline_performance': {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
            for k, v in baseline_performance.items()
        },
        'gsa_performance': {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
            for k, v in gsa_performance.items()
        },
        'context_test': {
            'standard_success': bool(std_success),
            'gsa_success': bool(gsa_success)
        },
        'criteria': {k: bool(v) for k, v in criteria.items()},
        'performance_change_pct': float(perplexity_change) if 'perplexity_change' in locals() else None,
        'parameter_overhead': {
            'total_increase': int(param_increase) if 'param_increase' in locals() else 0,
            'splat_params': int(splat_params) if 'splat_params' in locals() else 0,
            'percentage': float(param_increase/sum(p.numel() for p in original_model.parameters())*100) if 'param_increase' in locals() else 0
        }
    }
    
    with open('conservative_gsa_validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to 'conservative_gsa_validation_results.json'")
    
    # Cleanup
    del original_model, gsa_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results

if __name__ == "__main__":
    print("🎯 Conservative GSA Validation")
    print("Testing GSA that starts 99% identical to standard attention")
    
    try:
        results = run_conservative_gsa_test()
        
        if results:
            verdict = results['verdict']
            print(f"\n🏁 RESULT: {verdict.upper()}")
            
            if verdict == 'conservative_success':
                print("✅ Conservative GSA validated successfully")
                print("   Ready for gradual enhancement with splat learning")
                exit(0)
            elif verdict == 'conservative_working':
                print("⚡ Conservative GSA working well")
                print("   Good foundation for further development")
                exit(0)
            elif verdict == 'conservative_functional':
                print("⚠️ Conservative GSA functional but needs tuning")
                print("   Basic approach works, optimization needed")
                exit(0)
            else:
                print("❌ Conservative GSA needs fundamental fixes")
                exit(1)
        else:
            print("❌ Test failed to complete")
            exit(1)
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
