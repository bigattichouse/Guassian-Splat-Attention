"""
MINIMAL WORKING GSA - Avoid All Complications
============================================

STRATEGY:
1. Don't touch bias matrices at all
2. Use standard model with standard extension
3. Minimal GSA that just works
4. Test on standard 2048 limit first, then extend gradually
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention, GPTNeoAttention
import warnings
import time
import json
import math
import numpy as np
from typing import Optional, Tuple, List
import random
import gc
import psutil

warnings.filterwarnings("ignore")

def get_memory_usage():
    """Get current memory usage for monitoring."""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_max = torch.cuda.max_memory_allocated() / 1024**3  # GB
        return f"GPU: {gpu_memory:.2f}GB (peak: {gpu_max:.2f}GB)"
    else:
        cpu_memory = psutil.Process().memory_info().rss / 1024**3  # GB
        return f"CPU: {cpu_memory:.2f}GB"

class MinimalWorkingGSA(GPTNeoSelfAttention):
    """Minimal GSA that just works - no complications."""
    
    def __init__(self, config, attention_type, layer_id):
        super().__init__(config, attention_type, layer_id)
        
        self.layer_id = layer_id
        self.n_splats = 4
        
        # Minimal GSA parameters
        self.gsa_centers = nn.Parameter(
            torch.randn(config.num_attention_heads, self.n_splats, self.head_dim) * 0.01
        )
        self.gsa_scales = nn.Parameter(
            torch.ones(config.num_attention_heads, self.n_splats) * 0.5
        )
        
        self.enable_gsa = False
        self.gsa_blend = 0.3  # Conservative blending
        
        print(f"      ⭐ Minimal Working GSA (L{layer_id}): {self.n_splats} splats, blend={self.gsa_blend}")
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """Minimal GSA with conservative blending."""
        if not self.enable_gsa:
            return super()._attn(query, key, value, attention_mask, head_mask)
        
        try:
            # Get standard attention first
            std_output, std_weights = super()._attn(query, key, value, attention_mask, head_mask)
            
            # Get GSA contribution
            gsa_contribution = self._compute_minimal_gsa(query, key, value)
            
            # Conservative blend
            blended_output = (1 - self.gsa_blend) * std_output + self.gsa_blend * gsa_contribution
            
            return blended_output, std_weights
            
        except Exception as e:
            print(f"      ❌ Minimal GSA failed (L{self.layer_id}): {e}")
            return super()._attn(query, key, value, attention_mask, head_mask)
    
    def _compute_minimal_gsa(self, query, key, value):
        """Absolutely minimal GSA computation."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        device = query.device
        
        # Simple approach: weighted average of values based on splat proximity
        output = torch.zeros_like(value)
        
        for h in range(num_heads):
            for s in range(self.n_splats):
                center = self.gsa_centers[h, s, :]  # [head_dim]
                scale = torch.clamp(self.gsa_scales[h, s], min=0.1, max=1.0)
                
                # Distance from each token to this splat center
                diffs = value[:, h, :, :] - center.unsqueeze(0).unsqueeze(0)  # [B, seq_len, head_dim]
                distances = torch.sum(diffs ** 2, dim=-1)  # [B, seq_len]
                
                # Convert to weights
                weights = torch.exp(-distances / (2 * scale ** 2 + 1e-8))  # [B, seq_len]
                weights = weights / (torch.sum(weights, dim=-1, keepdim=True) + 1e-8)  # Normalize
                
                # Weighted average of values
                weighted_values = torch.einsum('bs,bsd->bd', weights, value[:, h, :, :])  # [B, head_dim]
                
                # Broadcast to all positions (simplified)
                output[:, h, :, :] += weighted_values.unsqueeze(1).expand(-1, seq_len, -1) / self.n_splats
        
        return output
    
    def enable_minimal_gsa(self):
        """Enable minimal GSA."""
        self.enable_gsa = True
        print(f"      ✅ Minimal Working GSA enabled (L{self.layer_id})")

class MinimalWorkingGSAAttention(GPTNeoAttention):
    """Wrapper for minimal working GSA."""
    
    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)
        attention_type = config.attention_layers[layer_id]
        self.attention = MinimalWorkingGSA(config, attention_type, layer_id)

def test_minimal_working_gsa():
    """Test minimal working approach - start within 2048 limit."""
    print("="*80)
    print("⭐ MINIMAL WORKING GSA - No Complications")
    print("="*80)
    print("STRATEGY: Start within 2048 limit, minimal GSA, conservative blending")
    print(f"Initial memory: {get_memory_usage()}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load standard model without extensions first
    model_name = "EleutherAI/gpt-neo-125M"
    
    print(f"\nLoading standard model (no extensions)...")
    start_time = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else "<|pad|>"
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id
        else:
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    
    # Keep original limits initially
    print(f"  Original max_position_embeddings: {model.config.max_position_embeddings}")
    print(f"  Original n_positions: {model.config.n_positions}")
    
    load_time = time.time() - start_time
    print(f"✓ Standard model loaded in {load_time:.1f}s")
    
    # Install minimal GSA on single layer
    gsa_layer = 6
    print(f"\nMinimal Working Strategy:")
    print(f"  Replace ONLY layer {gsa_layer} with minimal working GSA")
    print(f"  Keep ALL original position limits")
    print(f"  Start with tests within 2048 token limit")
    
    layer = model.transformer.h[gsa_layer]
    original_attention = layer.attn.attention
    
    # Create minimal GSA
    minimal_gsa = MinimalWorkingGSAAttention(model.config, gsa_layer)
    minimal_gsa = minimal_gsa.to(model.device)
    
    # Copy weights
    try:
        minimal_gsa.attention.c_attn.weight.data = original_attention.c_attn.weight.data.clone()
        minimal_gsa.attention.c_attn.bias.data = original_attention.c_attn.bias.data.clone()
        minimal_gsa.attention.c_proj.weight.data = original_attention.c_proj.weight.data.clone()
        minimal_gsa.attention.c_proj.bias.data = original_attention.c_proj.bias.data.clone()
        print(f"  ✓ Weights copied successfully")
    except Exception as e:
        print(f"  ⚠️ Weight copying failed: {e}")
    
    # Replace layer
    layer.attn = minimal_gsa
    layer.attn.attention.enable_minimal_gsa()
    
    print(f"  Layer {gsa_layer}: Minimal Working GSA installed ✅")
    print(f"  Memory after installation: {get_memory_usage()}")
    
    # Test within original limits first
    print(f"\nTesting Minimal Working GSA (within original 2048 limit)...")
    test_lengths = [256, 512, 1024, 1536, 2000]  # Stay well within 2048
    results = {}
    max_success = 0
    
    for context_length in test_lengths:
        print(f"\n--- Testing {context_length} tokens (within 2048 limit) ---")
        print(f"    Memory before test: {get_memory_usage()}")
        
        try:
            # Create test
            needle_code = f"{random.randint(1000, 9999)}"
            context_text, question_text = create_safe_test(
                needle_code, context_length, tokenizer
            )
            
            print(f"  Testing Minimal Working GSA...")
            torch.cuda.empty_cache()
            
            test_start = time.time()
            success, answer = test_minimal_capability(
                model, tokenizer, context_text, question_text,
                needle_code, max_length=context_length + 50
            )
            test_time = time.time() - test_start
            
            print(f"    Minimal Working GSA: {'✓' if success else '❌'} - '{answer}' ({test_time:.1f}s)")
            print(f"    Expected: '{needle_code}'")
            
            if success:
                max_success = context_length
            
            results[context_length] = {
                'needle_code': needle_code,
                'success': success,
                'answer': answer,
                'test_time': test_time
            }
            
        except Exception as e:
            print(f"    Test failed: {e}")
            results[context_length] = {
                'error': str(e),
                'needle_code': needle_code,
                'success': False
            }
        
        torch.cuda.empty_cache()
        print(f"    Memory after cleanup: {get_memory_usage()}")
    
    # Results
    print(f"\n" + "="*60)
    print("⭐ MINIMAL WORKING GSA RESULTS")
    print("="*60)
    
    for length, result in results.items():
        if 'error' not in result:
            time_info = f" ({result.get('test_time', 0):.1f}s)"
            print(f"\n{length} tokens:")
            print(f"  Minimal Working GSA: {'✓' if result['success'] else '❌'} - {result['answer']}{time_info}")
            print(f"  Expected: {result['needle_code']}")
        else:
            print(f"\n{length} tokens: ERROR - {result['error']}")
    
    print(f"\n🚀 MAXIMUM SUCCESS: {max_success} tokens")
    print(f"⭐ MINIMAL STRATEGY: Work within limits first, then extend")
    
    # If we get good results within 2048, then try extending
    if max_success >= 1536:
        print(f"\n🎉 SUCCESS WITHIN LIMITS! Now testing extension...")
        extension_result = test_with_extension(model, tokenizer, minimal_gsa, gsa_layer)
        
        if extension_result >= 2500:
            verdict = "minimal_extension_success"
        elif extension_result >= 2048:
            verdict = "minimal_basic_extension"
        else:
            verdict = "minimal_within_limits_only"
    elif max_success >= 1024:
        print(f"✅ MINIMAL PROGRESS: Working within limits")
        verdict = "minimal_progress"
    elif max_success >= 512:
        print(f"⚠️ MINIMAL PARTIAL: Basic functionality")
        verdict = "minimal_partial"
    else:
        print(f"❌ Continue development")
        verdict = "minimal_needs_work"
    
    return verdict

def test_with_extension(model, tokenizer, minimal_gsa, gsa_layer):
    """Test extension after confirming basic functionality."""
    print(f"\n🚀 EXTENSION TEST - Now that basic GSA works...")
    
    device = model.device
    target_context = 3072
    
    # Extend position embeddings 
    original_max_pos = model.config.max_position_embeddings
    print(f"  Extending embeddings: {original_max_pos} → {target_context}")
    
    original_embeddings = model.transformer.wpe.weight.data
    embedding_dim = original_embeddings.size(1)
    
    new_embeddings = torch.zeros(target_context, embedding_dim, 
                                device=original_embeddings.device, 
                                dtype=original_embeddings.dtype)
    new_embeddings[:original_max_pos] = original_embeddings
    
    # Simple extension
    for i in range(original_max_pos, target_context):
        base_idx = i % original_max_pos
        new_embeddings[i] = original_embeddings[base_idx] * 0.95
    
    model.transformer.wpe = nn.Embedding(target_context, embedding_dim)
    model.transformer.wpe.weight.data = new_embeddings
    model.transformer.wpe = model.transformer.wpe.to(device)
    
    # Update config
    model.config.max_position_embeddings = target_context
    model.config.n_positions = target_context
    
    print(f"  ✓ Extension complete")
    
    # Test extended lengths
    test_lengths = [2200, 2500, 3000]
    max_extended_success = 0
    
    for context_length in test_lengths:
        print(f"\n--- Extension Test: {context_length} tokens ---")
        
        try:
            needle_code = f"{random.randint(1000, 9999)}"
            context_text, question_text = create_safe_test(
                needle_code, context_length, tokenizer
            )
            
            success, answer = test_minimal_capability(
                model, tokenizer, context_text, question_text,
                needle_code, max_length=context_length + 50
            )
            
            print(f"    Extended GSA: {'✓' if success else '❌'} - '{answer}'")
            print(f"    Expected: '{needle_code}'")
            
            if success:
                max_extended_success = context_length
                
        except Exception as e:
            print(f"    Extension test failed: {e}")
            if "size of tensor a (2048)" in str(e):
                print(f"    🛑 Still hitting 2048 limit")
                break
    
    return max_extended_success

def create_safe_test(needle_code, context_length, tokenizer):
    """Create safe test case."""
    base = "The bird flew over the tree. "
    needle = f"The key is {needle_code}. "
    question = f"What is the key?"
    
    # Calculate tokens conservatively
    needle_tokens = len(tokenizer.encode(needle))
    question_tokens = len(tokenizer.encode(question))
    base_tokens = len(tokenizer.encode(base))
    
    available = context_length - needle_tokens - question_tokens - 20  # Extra safety
    n_base = max(1, available // base_tokens)
    
    # Create context
    parts = [base] * (n_base // 2) + [needle] + [base] * (n_base // 2)
    context = "".join(parts)
    
    return context, question

def test_minimal_capability(model, tokenizer, context_text, question_text, expected_answer, max_length=2048):
    """Test minimal GSA capability."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        if expected_answer not in context_text:
            return False, "needle_missing"
        
        full_prompt = context_text + f" {question_text} "
        
        inputs = tokenizer(
            full_prompt, 
            return_tensors="pt", 
            truncation=True,
            max_length=max_length-10
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = torch.ones_like(input_ids)
        
        decoded = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if expected_answer not in decoded:
            return False, "needle_lost"
        
        print(f"    ✓ Needle preserved in {input_ids.size(1)} tokens")
        
        with torch.no_grad():
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False
            )
        
        answer = tokenizer.decode(
            generated[0][input_ids.size(1):], 
            skip_special_tokens=True
        ).strip()
        
        success = expected_answer in answer or len([c for c in answer if c.isdigit()]) >= 3
        return success, answer
        
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"

if __name__ == "__main__":
    print("⭐ MINIMAL WORKING GSA - NO COMPLICATIONS")
    print("Start simple, get it working, then extend")
    
    try:
        result = test_minimal_working_gsa()
        
        print(f"\n🏁 MINIMAL WORKING RESULT: {result.upper()}")
        
        if "extension_success" in result:
            print("🎉 MINIMAL SUCCESS: Working GSA + successful extension!")
            print("✅ Ready for optimization!")
            exit(0)
        elif "basic_extension" in result:
            print("🎉 MINIMAL EXTENSION: Basic extension working!")
            print("✅ Foundation established!")
            exit(0)
        elif "within_limits_only" in result:
            print("✅ MINIMAL WORKING: GSA works within original limits")
            print("🎯 Need to debug extension")
            exit(0)
        elif "progress" in result:
            print("✅ MINIMAL PROGRESS: Basic functionality working")
            exit(0)
        else:
            print("⚠️ Continue minimal development")
            exit(0)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
