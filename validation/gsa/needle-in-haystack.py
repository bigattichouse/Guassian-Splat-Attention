#!/usr/bin/env python3
"""
GSA Needle-in-Haystack Test

Simple test app that loads your custom text file and tests GSA's ability
to find information in long contexts.

Usage:
    python needle-in-haystack.py 4096 inputfile.txt
    python needle-in-haystack.py 8192 my_test.txt --model roneneldan/TinyStories-Instruct-33M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import argparse
import time
import sys
import os
from dataclasses import dataclass
from typing import Optional

# GSA Implementation (simplified)
@dataclass
class GSAConfig:
    dim: int = 256
    n_heads: int = 8
    n_splats_per_head: int = 16
    movement_scale: float = 0.08
    temperature_init: float = 1.0
    scale_init: float = 0.5
    
    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

class GSAMechanism(nn.Module):
    def __init__(self, config: GSAConfig):
        super().__init__()
        self.config = config
        
        # Core splat parameters
        self.splat_centers = nn.Parameter(torch.randn(config.n_heads, config.n_splats_per_head, config.head_dim) * 0.2)
        self.splat_deltas = nn.Parameter(torch.zeros(config.n_heads, config.n_splats_per_head, config.head_dim))
        self.splat_log_scales = nn.Parameter(torch.randn(config.n_heads, config.n_splats_per_head) * 0.2 + np.log(config.scale_init))
        self.splat_log_amplitudes = nn.Parameter(torch.randn(config.n_heads, config.n_splats_per_head) * 0.1 - 0.5)
        
        self.movement_scale = nn.Parameter(torch.tensor(config.movement_scale))
        self.temperature = nn.Parameter(torch.tensor(config.temperature_init))
        
        self.qkv_proj = nn.Linear(config.dim, 3 * config.dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        H, S = self.config.n_heads, self.config.n_splats_per_head
        head_dim = self.config.head_dim
        
        qkv = self.qkv_proj(x).reshape(B, T, 3, H, head_dim)
        q, k, v = qkv.unbind(2)
        
        centers = self.splat_centers + self.splat_deltas * torch.sigmoid(self.movement_scale) * 0.2
        scales = torch.exp(self.splat_log_scales).clamp(min=0.01, max=2.0)
        amplitudes = torch.exp(self.splat_log_amplitudes).clamp(min=1e-6, max=10.0)
        
        attention_logits = torch.zeros(B, T, T, H, device=x.device)
        
        for h in range(H):
            q_dists = torch.sum((q[:, :, h].unsqueeze(2) - centers[h]) ** 2, dim=-1)
            k_dists = torch.sum((k[:, :, h].unsqueeze(2) - centers[h]) ** 2, dim=-1)
            
            q_weights = torch.exp(-0.5 * q_dists / (scales[h] ** 2 + 1e-8))
            k_weights = torch.exp(-0.5 * k_dists / (scales[h] ** 2 + 1e-8))
            
            attention_logits[:, :, :, h] = torch.einsum('bis,bjs,s->bij', q_weights, k_weights, amplitudes[h])
        
        attention = F.softmax(attention_logits / self.temperature.clamp(min=0.1, max=10.0), dim=2)
        output = torch.einsum('btjh,bjhd->bthd', attention, v)
        output = output.reshape(B, T, D)
        return self.out_proj(output)

class GSALayer(nn.Module):
    def __init__(self, config: GSAConfig):
        super().__init__()
        self.attention = GSAMechanism(config)
        self.norm = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, hidden_states, attention_mask=None, layer_past=None, 
                head_mask=None, use_cache=False, output_attentions=False, **kwargs):
        normed = self.norm(hidden_states)
        attn_out = self.attention(normed, attention_mask)
        output = hidden_states + self.dropout(attn_out)
        
        outputs = (output,)
        if use_cache:
            outputs = outputs + (None,)
        if output_attentions:
            outputs = outputs + (None,)
        return outputs

class GSAModelWrapper(nn.Module):
    def __init__(self, base_model, max_context_length=16384):
        super().__init__()
        self.model = base_model
        self.original_max_pos = getattr(base_model.config, 'max_position_embeddings', 2048)
        self.max_context_length = max_context_length
        
        # Create GSA config
        self.gsa_config = GSAConfig(
            dim=base_model.config.hidden_size,
            n_heads=base_model.config.num_attention_heads,
            n_splats_per_head=max(8, min(32, max_context_length // 128)),  # Scale splats with context
            movement_scale=0.08,
            temperature_init=1.0
        )
        
        self._extend_positions()
        self._replace_attention()
        
    def _extend_positions(self):
        """Extend position embeddings"""
        if hasattr(self.model.transformer, 'wpe') and self.max_context_length > self.original_max_pos:
            print(f"📏 Extending positions: {self.original_max_pos} → {self.max_context_length}")
            
            old_pos_emb = self.model.transformer.wpe.weight.data
            embed_dim = old_pos_emb.shape[1]
            
            new_pos_emb = nn.Embedding(self.max_context_length, embed_dim)
            new_pos_emb.weight.data[:self.original_max_pos] = old_pos_emb
            
            # Extend with cyclic pattern
            for i in range(self.original_max_pos, self.max_context_length):
                old_idx = i % self.original_max_pos
                new_pos_emb.weight.data[i] = old_pos_emb[old_idx] * 0.95
            
            self.model.transformer.wpe = new_pos_emb
            self.model.config.max_position_embeddings = self.max_context_length
            self.model.config.n_positions = self.max_context_length
    
    def _replace_attention(self):
        """Replace standard attention with GSA"""
        print(f"🔧 Replacing {len(self.model.transformer.h)} attention layers with GSA")
        print(f"   Splats per head: {self.gsa_config.n_splats_per_head}")
        
        for i, layer in enumerate(self.model.transformer.h):
            original_attn = layer.attn
            gsa_layer = GSALayer(self.gsa_config)
            
            # Copy weights if possible
            try:
                if hasattr(original_attn, 'c_attn'):
                    gsa_layer.attention.qkv_proj.weight.data.copy_(original_attn.c_attn.weight.data)
                elif hasattr(original_attn, 'q_proj'):
                    q_w = original_attn.q_proj.weight.data
                    k_w = original_attn.k_proj.weight.data
                    v_w = original_attn.v_proj.weight.data
                    qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
                    gsa_layer.attention.qkv_proj.weight.data.copy_(qkv_w)
                
                if hasattr(original_attn, 'c_proj'):
                    gsa_layer.attention.out_proj.weight.data.copy_(original_attn.c_proj.weight.data)
                elif hasattr(original_attn, 'out_proj'):
                    gsa_layer.attention.out_proj.weight.data.copy_(original_attn.out_proj.weight.data)
                    
            except Exception as e:
                print(f"   Layer {i}: Using random weights ({e})")
            
            layer.attn = gsa_layer
        
        print("✅ GSA replacement complete")
    
    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

class NeedleInHaystackTester:
    def __init__(self, model_name="roneneldan/TinyStories-Instruct-33M", 
                 context_length=4096, device="auto"):
        
        self.context_length = context_length
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Initializing GSA Needle-in-Haystack Tester")
        print(f"   Model: {model_name}")
        print(f"   Context Length: {context_length:,} tokens")
        print(f"   Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load and wrap model
        print("📦 Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        base_model.config.use_cache = False
        
        print("🎯 Creating GSA model...")
        self.model = GSAModelWrapper(base_model, context_length).to(self.device)
        self.model.eval()
        
        # Run diagnostic
        self.diagnose_gsa()
        
        print("✅ Initialization complete!\n")
    
    def diagnose_gsa(self):
        """Quick diagnostic to check if GSA is working properly"""
        print("\n🔍 GSA DIAGNOSTIC")
        print("-" * 30)
        
        try:
            # Check first layer
            first_layer = self.model.model.transformer.h[0].attn
            
            if hasattr(first_layer, 'attention') and hasattr(first_layer.attention, 'splat_centers'):
                splat_centers = first_layer.attention.splat_centers
                print(f"✅ GSA detected: {splat_centers.shape}")
                
                # Check for NaN or extreme values
                if torch.isnan(splat_centers).any():
                    print("❌ NaN values in splat centers!")
                elif torch.isinf(splat_centers).any():
                    print("❌ Infinite values in splat centers!")
                else:
                    center_stats = {
                        'min': splat_centers.min().item(),
                        'max': splat_centers.max().item(),
                        'mean': splat_centers.mean().item(),
                        'std': splat_centers.std().item()
                    }
                    print(f"📊 Splat centers: min={center_stats['min']:.3f}, max={center_stats['max']:.3f}")
                    print(f"   mean={center_stats['mean']:.3f}, std={center_stats['std']:.3f}")
                
                # Test with tiny input
                print("🧪 Testing GSA with small input...")
                test_input = torch.randint(0, 1000, (1, 10)).to(self.device)
                
                with torch.no_grad():
                    try:
                        test_output = self.model(test_input)
                        print("✅ GSA forward pass successful")
                        
                        if torch.isnan(test_output.logits).any():
                            print("❌ NaN in output logits!")
                        else:
                            print("✅ Output logits are finite")
                            
                    except Exception as e:
                        print(f"❌ GSA forward pass failed: {e}")
                        
            else:
                print("❌ GSA not detected - using standard attention")
                
        except Exception as e:
            print(f"❌ Diagnostic failed: {e}")
        
        print("-" * 30)
    
    def test_file(self, file_path):
        """Test a file containing text and questions"""
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"📖 Loading test file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            print("❌ File is empty")
            return
        
        # Check if content has clear separators for questions
        if '\n---\n' in content or '\n###\n' in content:
            self._test_structured_file(content)
        else:
            self._test_single_text(content)
    
    def _test_structured_file(self, content):
        """Test file with structured test cases"""
        if '\n---\n' in content:
            sections = content.split('\n---\n')
        else:
            sections = content.split('\n###\n')
        
        print(f"🧪 Found {len(sections)} test sections")
        
        for i, section in enumerate(sections, 1):
            section = section.strip()
            if not section:
                continue
                
            print(f"\n{'='*60}")
            print(f"🎯 TEST {i}/{len(sections)}")
            print(f"{'='*60}")
            
            preview = section[:200] + "..." if len(section) > 200 else section
            print(f"📝 Section preview: {preview}")
            
            self._run_single_test(section, f"Test {i}")
    
    def _test_single_text(self, content):
        """Test single block of text with interactive questions"""
        print(f"📄 Text length: {len(content):,} characters")
        
        tokens = self.tokenizer.encode(content)
        print(f"🔢 Token count: {len(tokens):,}")
        
        if len(tokens) > self.context_length - 100:
            print(f"⚠️  Text is very long, will be truncated to fit {self.context_length} tokens")
        
        print(f"\n📋 Text preview (first 500 chars):")
        print("-" * 50)
        print(content[:500] + ("..." if len(content) > 500 else ""))
        print("-" * 50)
        
        print(f"\n💬 Interactive mode - enter questions about this text")
        print(f"   Type 'quit' to exit, 'show' to see text again")
        
        while True:
            try:
                question = input("\n🤔 Your question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                elif question.lower() == 'show':
                    print(f"\n📄 Full text:\n{content}\n")
                    continue
                elif not question:
                    continue
                
                test_prompt = f"{content}\n\nQuestion: {question}\nAnswer:"
                self._run_single_test(test_prompt, f"Question: {question}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    
    def _run_single_test(self, prompt, test_name):
        """Run a single test with the given prompt"""
        print(f"\n🧪 Running: {test_name}")
        
        # Tokenize
        start_time = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt", 
                               max_length=self.context_length, 
                               truncation=True).to(self.device)
        
        input_length = inputs['input_ids'].shape[1]
        tokenize_time = time.time() - start_time
        
        print(f"📊 Input: {input_length:,} tokens (tokenized in {tokenize_time:.3f}s)")
        
        if input_length >= self.context_length - 10:
            print(f"⚠️  Input is very close to context limit ({self.context_length})")
        
        # Generate
        print("🤖 Generating response...")
        print(f"🔧 Generation settings: temp=0.3, max_tokens=50, rep_penalty=1.3")
        
        # Show a sample of the input for debugging
        if input_length > 100:
            sample_input = self.tokenizer.decode(inputs['input_ids'][0][-100:], skip_special_tokens=True)
            print(f"📋 Input ending: ...{sample_input}")
        
        try:
            with torch.no_grad():
                gen_start = time.time()
                
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    max_new_tokens=50,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                    early_stopping=True
                )
                
                gen_time = time.time() - gen_start
                
            # Decode response
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the generated part
            prompt_text = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
            if len(full_response) > len(prompt_text):
                generated_text = full_response[len(prompt_text):].strip()
            else:
                generated_text = full_response.strip()
            
            # Debug: Check for repetitive patterns
            if len(generated_text) > 10:
                char_counts = {}
                for char in generated_text:
                    char_counts[char] = char_counts.get(char, 0) + 1
                
                most_common_char = max(char_counts, key=char_counts.get)
                if char_counts[most_common_char] / len(generated_text) > 0.8:
                    print(f"⚠️  REPETITIVE OUTPUT DETECTED: '{most_common_char}' appears {char_counts[most_common_char]} times")
                    print(f"   This suggests the model is struggling with this input length")
                    
                    # Try a simpler generation
                    print("🔄 Retrying with simpler parameters...")
                    try:
                        simple_outputs = self.model.generate(
                            input_ids=inputs['input_ids'],
                            max_new_tokens=20,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            use_cache=False
                        )
                        
                        simple_response = self.tokenizer.decode(simple_outputs[0], skip_special_tokens=True)
                        if len(simple_response) > len(prompt_text):
                            generated_text = simple_response[len(prompt_text):].strip()
                            print(f"🔄 Simplified generation: '{generated_text}'")
                        
                    except Exception as e:
                        print(f"❌ Simplified generation also failed: {e}")
            
            print(f"⚡ Generated in {gen_time:.2f}s")
            print(f"📝 Response ({len(generated_text)} chars):")
            print("-" * 50)
            print(generated_text)
            print("-" * 50)
            
            # Show memory usage if on GPU
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**2
                memory_peak = torch.cuda.max_memory_allocated() / 1024**2
                print(f"💾 GPU Memory: {memory_used:.1f}MB used, {memory_peak:.1f}MB peak")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            if "out of memory" in str(e).lower():
                print("💡 Try reducing context length or using a smaller model")

def main():
    parser = argparse.ArgumentParser(description="GSA Needle-in-Haystack Test")
    parser.add_argument("context_length", type=int, help="Maximum context length in tokens")
    parser.add_argument("input_file", help="Text file to test")
    parser.add_argument("--model", default="roneneldan/TinyStories-Instruct-33M", 
                       help="HuggingFace model name")
    parser.add_argument("--device", default="auto", help="Device (cuda/cpu/auto)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"❌ Input file not found: {args.input_file}")
        return 1
    
    if args.context_length < 512:
        print(f"❌ Context length too small: {args.context_length} (minimum: 512)")
        return 1
    
    try:
        tester = NeedleInHaystackTester(
            model_name=args.model,
            context_length=args.context_length,
            device=args.device
        )
        
        tester.test_file(args.input_file)
        
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
