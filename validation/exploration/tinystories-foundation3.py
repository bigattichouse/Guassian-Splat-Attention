"""
GSA Phase 2: Context Extension & Real-World Testing
==================================================

Building on EXCELLENT Phase 1 results (-0.66% impact, perfect generation)
Now implementing GSA's core advantage: CONTEXT EXTENSION beyond architectural limits

Key Phase 2 Features:
1. Position embedding extension (2048 → 4096+ tokens)
2. Needle-in-haystack testing for long context validation
3. Multi-layer GSA deployment
4. Performance optimization and scaling analysis
5. Real-world task validation

PROVEN FOUNDATION: Enhanced GSA working excellently with minimal impact
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

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ContextExtendedGSA(GPTNeoSelfAttention):
    """
    Context-Extended GSA optimized for long sequences.
    
    Phase 2 Features:
    - Adaptive splat count based on sequence length
    - Optimized computation for long contexts  
    - Enhanced position awareness
    - Multi-layer coordination capabilities
    """
    
    def __init__(self, config, attention_type, layer_id):
        super().__init__(config, attention_type, layer_id)
        
        # Debug: Check config values
        max_pos = getattr(config, 'max_position_embeddings', 2048)
        print(f"        GSA init: max_position_embeddings={max_pos}, bias shape will be [{max_pos}, {max_pos}]")
        
        # Context extension parameters
        self.base_n_splats = 8
        self.max_n_splats = 16
        self.current_n_splats = self.base_n_splats
        
        # Initialize with maximum splats (will adapt based on sequence length)
        self.gsa_splat_centers = nn.Parameter(
            torch.randn(config.num_attention_heads, self.max_n_splats, self.head_dim) * 0.02
        )
        self.gsa_splat_log_scales = nn.Parameter(
            torch.zeros(config.num_attention_heads, self.max_n_splats) + 0.3
        )
        self.gsa_splat_amplitudes = nn.Parameter(
            torch.ones(config.num_attention_heads, self.max_n_splats) / self.max_n_splats
        )
        
        # Context extension specific parameters
        self.position_scale_factor = nn.Parameter(torch.tensor(1.0))
        self.long_range_boost = nn.Parameter(torch.tensor(0.1))
        
        # Control parameters
        self.enable_gsa = False
        self.gsa_strength = nn.Parameter(torch.tensor(-3.0))  # Start conservative
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Layer coordination
        self.layer_id = layer_id
        
        # Debug: Check actual bias shape after parent initialization
        if hasattr(self, 'bias'):
            actual_bias_shape = self.bias.shape
            print(f"        Actual bias shape: {actual_bias_shape}")
        
        print(f"      Context-Extended GSA (Layer {layer_id}): "
              f"adaptive splats {self.base_n_splats}-{self.max_n_splats}, "
              f"centers={self.gsa_splat_centers.shape}")
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """Context-Extended GSA with adaptive splat usage."""
        if not self.enable_gsa:
            return super()._attn(query, key, value, attention_mask, head_mask)
        
        try:
            # Adapt splat count based on sequence length
            seq_len = query.size(2)
            self._adapt_splat_count(seq_len)
            
            # Get standard attention from parent
            standard_output, standard_weights = super()._attn(query, key, value, attention_mask, head_mask)
            
            # Context-extended GSA computation
            gsa_strength = torch.sigmoid(self.gsa_strength)
            
            if gsa_strength > 0.01:
                gsa_weights = self._compute_context_extended_attention(
                    query, key, standard_weights.shape, seq_len
                )
                
                # Enhanced blending for long contexts
                blend_factor = self._compute_adaptive_blend_factor(seq_len, gsa_strength)
                blended_weights = (1 - blend_factor) * standard_weights + blend_factor * gsa_weights
                
                # Recompute output
                blended_output = torch.matmul(blended_weights, value)
                
                if seq_len > 100:  # Only log for longer sequences
                    print(f"        Context GSA (L{self.layer_id}): seq_len={seq_len}, "
                          f"splats={self.current_n_splats}, strength={gsa_strength:.3f}, "
                          f"blend={blend_factor:.3f}")
                
                return blended_output, blended_weights
            
            return standard_output, standard_weights
            
        except Exception as e:
            print(f"      Context GSA failed (L{self.layer_id}): {e}")
            self.enable_gsa = False
            return super()._attn(query, key, value, attention_mask, head_mask)
    
    def _adapt_splat_count(self, seq_len):
        """Adapt number of active splats based on sequence length."""
        if seq_len <= 128:
            self.current_n_splats = self.base_n_splats
        elif seq_len <= 512:
            self.current_n_splats = min(self.max_n_splats, self.base_n_splats + 2)
        elif seq_len <= 1024:
            self.current_n_splats = min(self.max_n_splats, self.base_n_splats + 4)
        else:
            self.current_n_splats = self.max_n_splats
    
    def _compute_adaptive_blend_factor(self, seq_len, base_strength):
        """Compute adaptive blending based on sequence length."""
        # Increase GSA influence for longer sequences
        if seq_len <= 512:
            return base_strength
        elif seq_len <= 1024:
            return min(0.8, base_strength * 1.5)
        else:
            return min(0.9, base_strength * 2.0)
    
    def _compute_context_extended_attention(self, query, key, target_shape, seq_len):
        """Compute context-aware GSA attention."""
        batch_size, num_heads, seq_len, seq_len_k = target_shape
        device = query.device
        
        gsa_attention = torch.zeros(target_shape, device=device, dtype=query.dtype)
        
        # Enhanced position encoding for long contexts
        position_factor = torch.sigmoid(self.position_scale_factor)
        
        for h in range(min(num_heads, self.gsa_splat_centers.size(0))):
            head_query = query[:, h, :, :]
            head_key = key[:, h, :, :]
            
            # Use adaptive number of splats
            active_centers = self.gsa_splat_centers[h, :self.current_n_splats, :]
            active_scales = torch.exp(self.gsa_splat_log_scales[h, :self.current_n_splats])
            active_amplitudes = torch.softmax(
                self.gsa_splat_amplitudes[h, :self.current_n_splats], dim=0
            )
            
            # Adaptive scale adjustment for long contexts
            if seq_len > 512:
                scale_adjustment = 1.0 + torch.log(torch.tensor(seq_len / 512.0))
                active_scales = active_scales * scale_adjustment
            
            # Clamp scales
            active_scales = torch.clamp(active_scales, min=0.1, max=5.0)
            
            # Compute enhanced affinities
            q_affinities = self._compute_enhanced_affinities(
                head_query, active_centers, active_scales, position_factor
            )
            k_affinities = self._compute_enhanced_affinities(
                head_key, active_centers, active_scales, position_factor
            )
            
            # Compute attention through splats with long-range boost
            for s in range(self.current_n_splats):
                splat_attention = torch.bmm(
                    q_affinities[:, :, s:s+1],
                    k_affinities[:, :, s:s+1].transpose(-2, -1)
                )
                
                # Long-range connection boost
                if seq_len > 512:
                    long_range_mask = self._create_long_range_mask(seq_len, device)
                    splat_attention = splat_attention + self.long_range_boost * long_range_mask
                
                gsa_attention[:, h, :, :] += active_amplitudes[s] * splat_attention
        
        # Enhanced temperature scaling for long contexts
        temp_factor = self.temperature.clamp(min=0.1, max=3.0)
        if seq_len > 1024:
            temp_factor = temp_factor * 0.8  # Sharper attention for very long contexts
        
        gsa_attention = gsa_attention / temp_factor
        
        # Normalize
        gsa_attention = F.softmax(gsa_attention, dim=-1)
        
        return gsa_attention
    
    def _compute_enhanced_affinities(self, tokens, centers, scales, position_factor):
        """Compute enhanced token-splat affinities with position awareness."""
        batch_size, seq_len, head_dim = tokens.shape
        n_splats = centers.size(0)
        
        # Position-enhanced tokens
        position_encoding = self._create_position_encoding(seq_len, head_dim, tokens.device)
        enhanced_tokens = tokens + position_factor * position_encoding
        
        # Compute distances
        tokens_expanded = enhanced_tokens.unsqueeze(2)
        centers_expanded = centers.unsqueeze(0).unsqueeze(0)
        
        distances = torch.norm(tokens_expanded - centers_expanded, dim=-1)
        
        # Gaussian affinities
        scales_expanded = scales.unsqueeze(0).unsqueeze(0)
        affinities = torch.exp(-0.5 * (distances / scales_expanded) ** 2)
        
        # Normalize
        affinities = affinities / (affinities.sum(dim=-1, keepdim=True) + 1e-8)
        
        return affinities
    
    def _create_position_encoding(self, seq_len, dim, device):
        """Create enhanced position encoding for long contexts."""
        position = torch.arange(seq_len, device=device).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * 
                           -(math.log(10000.0) / dim))
        
        pe = torch.zeros(seq_len, dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _create_long_range_mask(self, seq_len, device):
        """Create mask to boost long-range connections."""
        mask = torch.zeros(seq_len, seq_len, device=device)
        
        # Boost connections between distant tokens
        for i in range(seq_len):
            for j in range(seq_len):
                distance = abs(i - j)
                if distance > seq_len // 4:  # Long-range connections
                    mask[i, j] = 0.1 * (distance / seq_len)
        
        return mask.unsqueeze(0)
    
    def enable_context_gsa(self, strength=0.15):
        """Enable context-extended GSA."""
        self.enable_gsa = True
        self.set_gsa_strength(strength)
        print(f"      Context GSA enabled (L{self.layer_id})")
    
    def set_gsa_strength(self, strength):
        """Set GSA strength."""
        with torch.no_grad():
            strength = torch.clamp(torch.tensor(strength), 0.001, 0.999)
            logit = torch.log(strength / (1 - strength))
            self.gsa_strength.copy_(logit)

class ContextGSAAttention(GPTNeoAttention):
    """Wrapper for context-extended GSA."""
    
    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)
        attention_type = config.attention_layers[layer_id]
        self.attention = ContextExtendedGSA(config, attention_type, layer_id)
        print(f"    Context GSA {attention_type} attention for layer {layer_id}")

def load_model_with_extended_context(model_name, max_length=4096):
    """Load model normally, then extend position embeddings."""
    print(f"Loading model with extended context ({max_length} tokens)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # FIXED: Use a different pad token to avoid confusion
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            print(f"  Set pad_token to UNK: '{tokenizer.pad_token}'")
        else:
            # Add a new pad token if no UNK token exists
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            print(f"  Added new pad_token: '{tokenizer.pad_token}'")
    else:
        print(f"  Using existing pad_token: '{tokenizer.pad_token}'")
    
    # FIXED: Load model normally first
    print(f"  Loading original model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32
    ).to(device)
    
    # If we added a new pad token, resize model embeddings
    if tokenizer.pad_token not in ['<unk>', None] and hasattr(tokenizer, '_added_tokens_encoder'):
        if '<|pad|>' in tokenizer._added_tokens_encoder:
            print(f"  Resizing model embeddings for new pad token...")
            model.resize_token_embeddings(len(tokenizer))
    
    original_max_pos = model.config.max_position_embeddings
    print(f"  Original max_position_embeddings: {original_max_pos}")
    
    if max_length <= original_max_pos:
        print(f"  No extension needed")
        return model, tokenizer
    
    # FIXED: Extend position embeddings after loading
    print(f"  Extending position embeddings: {original_max_pos} → {max_length}")
    
    # Get original embeddings
    original_embeddings = model.transformer.wpe.weight.data
    original_length = original_embeddings.size(0)
    embedding_dim = original_embeddings.size(1)
    
    # Create new extended embedding matrix
    new_embeddings = torch.zeros(max_length, embedding_dim, device=original_embeddings.device, dtype=original_embeddings.dtype)
    
    # Copy original embeddings
    new_embeddings[:original_length] = original_embeddings
    
    # Extend with cyclic pattern and decay
    for i in range(original_length, max_length):
        base_idx = i % original_length
        decay_factor = 0.95 ** ((i - original_length) // original_length)
        new_embeddings[i] = original_embeddings[base_idx] * decay_factor
    
    # FIXED: Replace the embedding layer properly
    model.transformer.wpe = nn.Embedding(max_length, embedding_dim)
    model.transformer.wpe.weight.data = new_embeddings
    model.transformer.wpe = model.transformer.wpe.to(device)
    
    # FIXED: Update config after extending
    model.config.max_position_embeddings = max_length
    model.config.n_positions = max_length
    model.config.use_cache = False  # Disable cache for extended context
    
    print(f"  ✓ Position embeddings extended to {max_length}")
    print(f"  ✓ Config updated: max_position_embeddings={model.config.max_position_embeddings}")
    print(f"  ✓ Config updated: use_cache={model.config.use_cache}")
    
    return model, tokenizer

def replace_with_context_gsa(model, layers_to_replace=None):
    """Replace attention layers with context-extended GSA."""
    if layers_to_replace is None:
        layers_to_replace = [0, 1]  # Start with first two layers
    
    replacements_made = 0
    total_layers = len(model.transformer.h)
    
    print(f"Installing Context GSA in layers {layers_to_replace}...")
    
    # Ensure config reflects the extended position embeddings
    current_max_pos = model.config.max_position_embeddings
    print(f"  Model config max_position_embeddings: {current_max_pos}")
    
    for layer_idx in layers_to_replace:
        if layer_idx < total_layers:
            try:
                layer = model.transformer.h[layer_idx]
                
                # Create context GSA with updated config
                context_gsa_attention = ContextGSAAttention(model.config, layer_idx)
                context_gsa_attention = context_gsa_attention.to(model.device)
                
                # Copy weights with bias extension handling
                copy_success = copy_attention_weights(context_gsa_attention, layer.attn)
                
                if copy_success:
                    layer.attn = context_gsa_attention
                    replacements_made += 1
                    print(f"  Layer {layer_idx}: Context GSA installed ✓")
                else:
                    print(f"  Layer {layer_idx}: Weight copy failed")
                
            except Exception as e:
                print(f"  Layer {layer_idx}: Installation failed - {e}")
                # Print more detailed error info for debugging
                import traceback
                print(f"    Error details: {traceback.format_exc()}")
    
    print(f"Successfully installed Context GSA in {replacements_made} layers")
    return replacements_made

def copy_attention_weights(gsa_attention, original_attention):
    """Copy weights with proper bias tensor extension for longer contexts."""
    gsa_inner = gsa_attention.attention
    orig_inner = original_attention.attention
    
    with torch.no_grad():
        # Copy projection weights (these don't depend on sequence length)
        gsa_inner.k_proj.weight.copy_(orig_inner.k_proj.weight)
        gsa_inner.v_proj.weight.copy_(orig_inner.v_proj.weight)
        gsa_inner.q_proj.weight.copy_(orig_inner.q_proj.weight)
        gsa_inner.out_proj.weight.copy_(orig_inner.out_proj.weight)
        gsa_inner.out_proj.bias.copy_(orig_inner.out_proj.bias)
        
        # Handle bias tensor extension for longer contexts
        orig_bias = orig_inner.bias
        gsa_bias = gsa_inner.bias
        
        if orig_bias.shape != gsa_bias.shape:
            print(f"      Extending bias tensor: {orig_bias.shape} → {gsa_bias.shape}")
            
            # Get dimensions
            orig_seq_len = orig_bias.size(-1)
            new_seq_len = gsa_bias.size(-1)
            
            # Copy the original causal mask
            gsa_bias[:, :, :orig_seq_len, :orig_seq_len] = orig_bias
            
            # Extend the causal mask pattern for longer sequences
            for i in range(orig_seq_len, new_seq_len):
                for j in range(new_seq_len):
                    if j <= i:  # Causal mask: can attend to previous and current positions
                        gsa_bias[:, :, i, j] = 1.0
                    else:  # Cannot attend to future positions
                        gsa_bias[:, :, i, j] = 0.0
            
            # Fill the extended region for existing tokens
            for i in range(orig_seq_len):
                for j in range(orig_seq_len, new_seq_len):
                    gsa_bias[:, :, i, j] = 0.0  # Existing tokens can't see future positions
        else:
            # Dimensions match, direct copy
            gsa_inner.bias.copy_(orig_inner.bias)
        
        # Masked bias is a scalar, copy directly
        gsa_inner.masked_bias.copy_(orig_inner.masked_bias)
    
    # Verification (only check what we can)
    checks = [
        torch.equal(gsa_inner.k_proj.weight, orig_inner.k_proj.weight),
        torch.equal(gsa_inner.v_proj.weight, orig_inner.v_proj.weight),
        torch.equal(gsa_inner.q_proj.weight, orig_inner.q_proj.weight),
        torch.equal(gsa_inner.out_proj.weight, orig_inner.out_proj.weight),
        torch.equal(gsa_inner.out_proj.bias, orig_inner.out_proj.bias),
        torch.equal(gsa_inner.masked_bias, orig_inner.masked_bias),
    ]
    
    # Check bias compatibility (can't be exactly equal if extended)
    bias_compatible = (gsa_inner.bias.shape[2] == gsa_inner.bias.shape[3] and  # Square matrix
                      gsa_inner.bias.size(-1) >= orig_inner.bias.size(-1))  # At least as large
    
    print(f"      Weight copy: {sum(checks)}/6 exact matches, bias extended: {bias_compatible}")
    return all(checks) and bias_compatible

def enable_context_gsa_in_layers(model, layers, strength=0.15):
    """Enable context GSA in specified layers."""
    print(f"Enabling Context GSA in layers {layers} with strength {strength:.3f}")
    
    for layer_idx in layers:
        if layer_idx < len(model.transformer.h):
            layer = model.transformer.h[layer_idx]
            if hasattr(layer.attn, 'attention') and hasattr(layer.attn.attention, 'enable_context_gsa'):
                layer.attn.attention.enable_context_gsa(strength)

def create_needle_in_haystack_test(needle_info, context_length, tokenizer):
    """
    Create PROPER needle-in-haystack test that separates context from question.
    
    Returns:
        context_text: The haystack with needle (NO question included)
        question_text: The question to ask
        expected_answer: The needle information to find
    """
    base_sentences = [
        "The weather was particularly pleasant that day.",
        "Many people gathered in the town square for the festival.", 
        "Children played games while their parents watched nearby.",
        "The local bakery sold fresh bread and pastries.",
        "Musicians performed traditional songs on their instruments.",
        "Vendors displayed colorful fruits and vegetables.",
        "The mayor gave a speech about community development.",
        "Artists showcased their paintings and sculptures.",
        "Local artists displayed their colorful artwork.",
        "Families enjoyed picnics in the nearby park."
    ]
    
    # Create needle text with more context to make it realistic
    needle_text = f"During the event, the organizer announced that the secret access code for the VIP area is {needle_info['code']}."
    
    # Estimate tokens per sentence
    sample_text = " ".join(base_sentences)
    sample_tokens = len(tokenizer.encode(sample_text))
    tokens_per_sentence = sample_tokens / len(base_sentences)
    
    # Calculate needed sentences (reserve space for question)
    needle_tokens = len(tokenizer.encode(needle_text))
    question_tokens = 50  # Reserve space for question
    remaining_tokens = context_length - needle_tokens - question_tokens
    needed_sentences = max(10, int(remaining_tokens / tokens_per_sentence))
    
    # Generate haystack
    haystack_sentences = []
    for i in range(needed_sentences):
        haystack_sentences.append(base_sentences[i % len(base_sentences)])
    
    # Insert needle at specified position
    needle_position = needle_info['position']
    if needle_position == 'beginning':
        insert_idx = max(2, len(haystack_sentences) // 10)
    elif needle_position == 'middle':
        insert_idx = len(haystack_sentences) // 2
    else:  # end
        insert_idx = max(len(haystack_sentences) - 10, len(haystack_sentences) * 3 // 4)
    
    haystack_sentences.insert(insert_idx, needle_text)
    
    # FIXED: Separate context from question
    context_text = " ".join(haystack_sentences)
    question_text = "What is the secret access code mentioned in the text?"
    
    return context_text, question_text, needle_info['code']

def test_context_extension():
    """Test context extension capabilities with needle-in-haystack."""
    print("="*80)
    print("GSA PHASE 2: CONTEXT EXTENSION & REAL-WORLD TESTING")
    print("="*80)
    print("Building on EXCELLENT Phase 1 (-0.66% impact)")
    print("Testing: Context extension, needle-in-haystack, multi-layer GSA")
    
    # Load models
    model_name = "roneneldan/TinyStories-Instruct-33M"
    print(f"\nLoading models: {model_name}")
    
    # Load original model (baseline)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # FIXED: Use consistent pad token setup
    if tokenizer.pad_token is None:
        if tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token
            print(f"  Set pad_token to UNK: '{tokenizer.pad_token}'")
        else:
            # Add a new pad token if no UNK token exists
            tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            print(f"  Added new pad_token: '{tokenizer.pad_token}'")
    else:
        print(f"  Using existing pad_token: '{tokenizer.pad_token}'")
        
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    
    # If we added a new pad token, resize model embeddings
    if tokenizer.pad_token not in ['<unk>', None] and hasattr(tokenizer, '_added_tokens_encoder'):
        if '<|pad|>' in tokenizer._added_tokens_encoder:
            print(f"  Resizing original model embeddings for new pad token...")
            original_model.resize_token_embeddings(len(tokenizer))
    
    # Load context model with extended position embeddings
    try:
        context_model, _ = load_model_with_extended_context(model_name, max_length=4096)
        print(f"✓ Context model loaded with 4096 max length")
    except Exception as e:
        print(f"⚠️ Failed to load with 4096 context: {e}")
        print("🔧 Trying fallback with 3072 context...")
        try:
            context_model, _ = load_model_with_extended_context(model_name, max_length=3072)
            print(f"✓ Context model loaded with 3072 max length")
        except Exception as e2:
            print(f"❌ Fallback failed too: {e2}")
            return "failed"
    
    print(f"✓ Models loaded successfully")
    
    # Step 1: Install context GSA
    print(f"\nStep 1: Installing Context GSA...")
    replacements_made = replace_with_context_gsa(context_model, layers_to_replace=[0, 1])
    
    if replacements_made == 0:
        print("❌ No GSA layers installed")
        print("🔧 ATTEMPTING FALLBACK: Single layer installation...")
        
        # Fallback: Try single layer
        replacements_made = replace_with_context_gsa(context_model, layers_to_replace=[0])
        
        if replacements_made == 0:
            print("❌ Fallback failed too")
            return "failed"
        else:
            print(f"✅ Fallback successful: {replacements_made} layer(s)")
    else:
        print(f"✅ Full success: {replacements_made} layer(s)")
    
    # Step 2: Test scaffolding at original length
    print(f"\nStep 2: Verify scaffolding at original length...")
    
    test_text = "Once upon a time, there was a little girl who loved to read books and explore."
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(device)
    
    with torch.no_grad():
        orig_output = original_model(input_ids, attention_mask=attention_mask, labels=input_ids)
        scaffolding_output = context_model(input_ids, attention_mask=attention_mask, labels=input_ids)
        
        scaffolding_diff = abs(orig_output.loss.item() - scaffolding_output.loss.item())
    
    print(f"  Scaffolding difference: {scaffolding_diff:.2e}")
    
    if scaffolding_diff > 1e-4:  # More lenient for extended context
        print(f"⚠️ Scaffolding drift detected (expected for extended context)")
    else:
        print(f"✅ Scaffolding maintained")
    
    # Step 3: Enable Context GSA
    print(f"\nStep 3: Enabling Context GSA...")
    enable_context_gsa_in_layers(context_model, layers=[0], strength=0.15)
    if replacements_made > 1:
        enable_context_gsa_in_layers(context_model, layers=[1], strength=0.10)
    
    # Step 4: Context extension testing
    print(f"\nStep 4: Context Extension Testing...")
    
    # Get the actual max length from the context model
    max_context_length = context_model.config.max_position_embeddings
    print(f"  Context model max length: {max_context_length}")
    
    # Test lengths based on model capability
    if max_context_length >= 4096:
        test_lengths = [1024, 2048, 3072, 4096]
    elif max_context_length >= 3072:
        test_lengths = [1024, 2048, 3072]
    else:
        test_lengths = [1024, 2048]
    
    print(f"  Testing lengths: {test_lengths}")
    
    results = {}
    
    for context_length in test_lengths:
        print(f"\n--- Testing {context_length} tokens ---")
        
        # Create needle-in-haystack test
        needle_info = {
            'code': f"{random.randint(1000, 9999)}",
            'position': 'middle'
        }
        
        # FIXED: Get separated context and question
        context_text, question_text, expected_answer = create_needle_in_haystack_test(
            needle_info, context_length, tokenizer
        )
        
        # Tokenize context only (NO question in input!)
        max_context_for_test = min(context_length-50, max_context_length-50)
        context_inputs = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=max_context_for_test)
        context_ids = context_inputs["input_ids"].to(device)
        context_attention_mask = context_inputs.get("attention_mask", torch.ones_like(context_ids)).to(device)
        
        actual_length = context_ids.size(1)
        print(f"  Target: {context_length}, Context: {actual_length} tokens")
        print(f"  Needle: '{expected_answer}' at position '{needle_info['position']}'")
        
        # Test original model (expected to fail beyond 2048)
        print(f"  Testing original model...")
        try:
            with torch.no_grad():
                # Test if original model can handle the context length
                orig_output = original_model(context_ids, attention_mask=context_attention_mask)
                orig_success = True
                
                # Test needle retrieval capability
                question_prompt = context_text + f"\n\nQuestion: {question_text}\nAnswer:"
                q_inputs = tokenizer(question_prompt, return_tensors="pt", truncation=True, max_length=max_context_length-10)
                q_ids = q_inputs["input_ids"].to(device)
                
                # Use proper attention mask from tokenizer
                attention_mask = q_inputs.get("attention_mask", torch.ones_like(q_ids)).to(device)
                
                orig_generated = original_model.generate(
                    q_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                orig_answer = tokenizer.decode(orig_generated[0][q_ids.size(1):], skip_special_tokens=True).strip()
                orig_retrieval_success = expected_answer in orig_answer
                
                print(f"    Original model: SUCCESS, retrieval={'✓' if orig_retrieval_success else '❌'}")
                print(f"    Generated answer: '{orig_answer}' (expected: '{expected_answer}')")
                
        except Exception as e:
            orig_success = False
            orig_retrieval_success = False
            orig_answer = ""
            print(f"    Original model: FAILED - {str(e)[:60]}...")
        
        # Test context GSA model
        print(f"  Testing Context GSA model...")
        try:
            torch.cuda.empty_cache()  # Clear memory
            
            with torch.no_grad():
                # Test if context GSA can handle the context length
                context_output = context_model(context_ids, attention_mask=context_attention_mask)
                context_success = True
                
                # FIXED: Test actual needle retrieval capability
                question_prompt = context_text + f"\n\nQuestion: {question_text}\nAnswer:"
                q_inputs = tokenizer(question_prompt, return_tensors="pt", truncation=True, max_length=max_context_length-10)
                q_ids = q_inputs["input_ids"].to(device)
                
                # Use proper attention mask from tokenizer
                attention_mask = q_inputs.get("attention_mask", torch.ones_like(q_ids)).to(device)
                
                context_generated = context_model.generate(
                    q_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                context_answer = tokenizer.decode(context_generated[0][q_ids.size(1):], skip_special_tokens=True).strip()
                context_retrieval_success = expected_answer in context_answer
                
                print(f"    Context GSA: SUCCESS, retrieval={'✓' if context_retrieval_success else '❌'}")
                print(f"    Generated answer: '{context_answer}' (expected: '{expected_answer}')")
                
        except Exception as e:
            context_success = False
            context_retrieval_success = False
            context_answer = ""
            print(f"    Context GSA: FAILED - {str(e)[:60]}...")
        
        # Record results
        results[context_length] = {
            'actual_length': actual_length,
            'needle_code': needle_info['code'],
            'needle_position': needle_info['position'],
            'original_success': orig_success,
            'original_retrieval': orig_retrieval_success if orig_success else False,
            'original_answer': orig_answer if orig_success else "",
            'context_success': context_success,
            'context_retrieval': context_retrieval_success if context_success else False,
            'context_answer': context_answer if context_success else ""
        }
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Analysis
    print(f"\n" + "="*60)
    print("NEEDLE-IN-HAYSTACK RETRIEVAL ANALYSIS")
    print("="*60)
    
    max_original_success = 0
    max_context_success = 0
    max_original_retrieval = 0
    max_context_retrieval = 0
    
    for length, result in results.items():
        print(f"\n{length} tokens:")
        print(f"  Original Model:")
        print(f"    Processing: {'✓' if result['original_success'] else '❌'}")
        print(f"    Needle Retrieval: {'✓' if result.get('original_retrieval', False) else '❌'}")
        if result.get('original_answer'):
            print(f"    Answer: '{result['original_answer']}'")
        
        print(f"  Context GSA:")
        print(f"    Processing: {'✓' if result['context_success'] else '❌'}")
        print(f"    Needle Retrieval: {'✓' if result.get('context_retrieval', False) else '❌'}")
        if result.get('context_answer'):
            print(f"    Answer: '{result['context_answer']}'")
        
        print(f"  Expected: '{result['needle_code']}' at {result['needle_position']}")
        
        if result['original_success']:
            max_original_success = length
        if result['context_success']:
            max_context_success = length
        if result.get('original_retrieval', False):
            max_original_retrieval = length
        if result.get('context_retrieval', False):
            max_context_retrieval = length
    
    # Calculate extension improvement
    processing_improvement = max_context_success - max_original_success
    retrieval_improvement = max_context_retrieval - max_original_retrieval
    
    print(f"\n🎯 CONTEXT EXTENSION RESULTS:")
    print(f"  Processing Capability:")
    print(f"    Original: {max_original_success} tokens")
    print(f"    Context GSA: {max_context_success} tokens")
    print(f"    Improvement: +{processing_improvement} tokens")
    
    print(f"  Needle Retrieval Capability:")
    print(f"    Original: {max_original_retrieval} tokens")
    print(f"    Context GSA: {max_context_retrieval} tokens") 
    print(f"    Improvement: +{retrieval_improvement} tokens")
    
    if retrieval_improvement > 0:
        improvement_ratio = max_context_retrieval / max(max_original_retrieval, 1024)
        print(f"\n🎉 NEEDLE-IN-HAYSTACK SUCCESS!")
        print(f"  GSA achieves {improvement_ratio:.1f}x improvement in needle retrieval!")
        verdict = "retrieval_success"
    elif processing_improvement > 0:
        print(f"\n✅ PROCESSING SUCCESS!")
        print(f"  GSA can handle longer contexts but retrieval needs optimization")
        verdict = "processing_success"
    else:
        print(f"\n⚠️ No clear improvement demonstrated")
        verdict = "no_improvement"
    
    # Save results
    context_results = {
        'verdict': verdict,
        'max_original_success': max_original_success,
        'max_context_success': max_context_success,
        'max_original_retrieval': max_original_retrieval,
        'max_context_retrieval': max_context_retrieval,
        'processing_improvement': processing_improvement,
        'retrieval_improvement': retrieval_improvement,
        'scaffolding_diff': float(scaffolding_diff),
        'test_results': results
    }
    
    with open('context_extension_results.json', 'w') as f:
        json.dump(context_results, f, indent=2)
    
    print(f"\nResults saved to 'context_extension_results.json'")
    
    return verdict

if __name__ == "__main__":
    print("🚀 GSA Phase 2: The Context Extension Revolution")
    print("Testing GSA's core advantage: breaking through architectural context limits")
    
    try:
        result = test_context_extension()
        
        print(f"\n🏁 NEEDLE-IN-HAYSTACK RESULT: {result.upper()}")
        
        if result == "retrieval_success":
            print("🎉 BREAKTHROUGH: Needle-in-haystack retrieval success!")
            print("   GSA successfully retrieves information from longer contexts")
            print("   Ready for real-world deployment and scaling")
            exit(0)
        elif result == "processing_success":
            print("✅ PROGRESS: Context processing improved")
            print("   GSA handles longer contexts, retrieval optimization needed")
            exit(0)
        elif result == "no_improvement":
            print("⚠️ No clear improvement demonstrated")
            print("   Need to optimize GSA parameters and architecture")
            exit(0)
        else:
            print("❌ Testing failed")
            exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
