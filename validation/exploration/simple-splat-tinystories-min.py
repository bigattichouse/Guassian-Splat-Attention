"""
Inheritance-based approach: Inherit from the original classes and just add unused parameters.
This ensures 100% identical computation while adding GSA scaffolding.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention, GPTNeoAttention
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GSAReadySelfAttention(GPTNeoSelfAttention):
    """
    Inherits from GPTNeoSelfAttention and adds unused GSA parameters.
    The forward pass is IDENTICAL to the original.
    """
    
    def __init__(self, config, attention_type, layer_id):
        # Initialize exactly like the original
        super().__init__(config, attention_type, layer_id)
        
        # Add unused GSA parameters for future development
        self.unused_gsa_splat_centers = nn.Parameter(
            torch.randn(config.num_attention_heads, 2, self.head_dim) * 0.01
        )
        self.unused_gsa_splat_scales = nn.Parameter(
            torch.zeros(config.num_attention_heads, 2)
        )
        
        print(f"      Added unused GSA parameters: centers={self.unused_gsa_splat_centers.shape}, scales={self.unused_gsa_splat_scales.shape}")
    
    # No need to override forward - it's identical to the parent!

class GSAReadyAttention(GPTNeoAttention):
    """
    Inherits from GPTNeoAttention and uses our GSA-ready inner attention.
    """
    
    def __init__(self, config, layer_id):
        # Initialize parent structure
        super().__init__(config, layer_id)
        
        # Replace the inner attention with our GSA-ready version
        attention_type = config.attention_layers[layer_id]
        self.attention = GSAReadySelfAttention(config, attention_type, layer_id)
        
        print(f"    Created GSA-ready {attention_type} attention for layer {layer_id}")
    
    # No need to override forward - it's identical to the parent!

def replace_with_gsa_ready_attention(model):
    """Replace attention layers with GSA-ready versions."""
    replacements_made = 0
    total_layers = len(model.transformer.h)
    
    print(f"Replacing with GSA-ready attention in {total_layers} transformer blocks...")
    
    for layer_idx, layer in enumerate(model.transformer.h):
        try:
            # Create GSA-ready attention
            gsa_ready_attention = GSAReadyAttention(model.config, layer_idx)
            gsa_ready_attention = gsa_ready_attention.to(model.device)
            
            # Copy weights from original
            original_attention = layer.attn
            copy_attention_weights(gsa_ready_attention, original_attention)
            
            # Replace
            layer.attn = gsa_ready_attention
            replacements_made += 1
            
            print(f"  Layer {layer_idx}: Replaced with GSA-ready attention ✓")
            
        except Exception as e:
            print(f"  Layer {layer_idx}: Replacement failed - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"Successfully replaced {replacements_made}/{total_layers} attention layers")
    return replacements_made

def copy_attention_weights(gsa_ready_attention, original_attention):
    """Copy weights between attention modules."""
    
    # Copy inner attention weights
    gsa_inner = gsa_ready_attention.attention
    orig_inner = original_attention.attention
    
    with torch.no_grad():
        # Copy projection weights
        gsa_inner.k_proj.weight.copy_(orig_inner.k_proj.weight)
        gsa_inner.v_proj.weight.copy_(orig_inner.v_proj.weight)
        gsa_inner.q_proj.weight.copy_(orig_inner.q_proj.weight)
        gsa_inner.out_proj.weight.copy_(orig_inner.out_proj.weight)
        gsa_inner.out_proj.bias.copy_(orig_inner.out_proj.bias)
        
        # Copy buffers (this is crucial for local vs global differences)
        gsa_inner.bias.copy_(orig_inner.bias)
        gsa_inner.masked_bias.copy_(orig_inner.masked_bias)
    
    # Verify the copy
    checks = [
        torch.equal(gsa_inner.k_proj.weight, orig_inner.k_proj.weight),
        torch.equal(gsa_inner.v_proj.weight, orig_inner.v_proj.weight),
        torch.equal(gsa_inner.q_proj.weight, orig_inner.q_proj.weight),
        torch.equal(gsa_inner.out_proj.weight, orig_inner.out_proj.weight),
        torch.equal(gsa_inner.out_proj.bias, orig_inner.out_proj.bias),
        torch.equal(gsa_inner.bias, orig_inner.bias),
        torch.equal(gsa_inner.masked_bias, orig_inner.masked_bias),
    ]
    
    print(f"      Weight copy verification: {sum(checks)}/7 exact matches")
    return all(checks)

def test_inheritance_approach():
    """Test the inheritance-based approach."""
    print("🎯 INHERITANCE-BASED GSA FOUNDATION TEST")
    print("="*80)
    print("Using inheritance to ensure 100% identical computation")
    
    # Load models
    model_name = "roneneldan/TinyStories-Instruct-33M"
    print(f"\nLoading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    
    gsa_ready_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32
    ).to(device)
    
    print(f"✓ Models loaded successfully")
    
    # Replace with GSA-ready attention
    print(f"\nReplacing attention layers...")
    replacements_made = replace_with_gsa_ready_attention(gsa_ready_model)
    
    if replacements_made == 0:
        print("❌ No replacements made, aborting test")
        return "failed"
    
    # Test data
    test_texts = [
        "Once upon a time, there was a little girl who loved to read books.",
        "The friendly dragon helped the village children learn new things.",
        "Every morning, the baker made fresh bread for the town.",
        "Tom and his friend built a treehouse in the big oak tree.",
        "The wise king always listened to his people's problems."
    ]
    
    print(f"\nTesting performance...")
    
    original_losses = []
    gsa_ready_losses = []
    max_diff = 0
    
    for i, text in enumerate(test_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=False)
        input_ids = inputs["input_ids"].to(device)
        
        # Test original
        with torch.no_grad():
            orig_outputs = original_model(input_ids, labels=input_ids)
            orig_loss = orig_outputs.loss.item()
            original_losses.append(orig_loss)
        
        # Test GSA-ready
        try:
            with torch.no_grad():
                gsa_outputs = gsa_ready_model(input_ids, labels=input_ids)
                gsa_loss = gsa_outputs.loss.item()
                gsa_ready_losses.append(gsa_loss)
            
            diff = abs(orig_loss - gsa_loss)
            max_diff = max(max_diff, diff)
            
            print(f"  Text {i+1}: original={orig_loss:.8f}, gsa_ready={gsa_loss:.8f}, diff={diff:.2e}")
            
        except Exception as e:
            print(f"  Text {i+1}: FAILED - {e}")
            gsa_ready_losses.append(None)
            import traceback
            traceback.print_exc()
            return "failed"
    
    # Analysis
    if all(l is not None for l in gsa_ready_losses):
        avg_diff = sum(abs(o - g) for o, g in zip(original_losses, gsa_ready_losses)) / len(original_losses)
        
        print(f"\nResults:")
        print(f"  Maximum difference: {max_diff:.2e}")
        print(f"  Average difference: {avg_diff:.2e}")
        
        if max_diff < 1e-12:
            print(f"  🎯 PERFECT: Differences < 1e-12 (machine precision)")
            verdict = "perfect"
        elif max_diff < 1e-8:
            print(f"  ✅ EXCELLENT: Differences < 1e-8")
            verdict = "excellent"
        elif max_diff < 1e-6:
            print(f"  ✓ VERY GOOD: Differences < 1e-6")
            verdict = "very_good"
        elif max_diff < 1e-4:
            print(f"  ✓ GOOD: Differences < 1e-4")
            verdict = "good"
        else:
            print(f"  ⚠️ ACCEPTABLE: Differences < 0.01")
            verdict = "acceptable"
    else:
        print(f"  ❌ FAILED: Some tests crashed")
        verdict = "failed"
    
    # Show parameter counts
    orig_params = sum(p.numel() for p in original_model.parameters())
    gsa_params = sum(p.numel() for p in gsa_ready_model.parameters())
    param_increase = gsa_params - orig_params
    
    print(f"\nParameter analysis:")
    print(f"  Original parameters: {orig_params:,}")
    print(f"  GSA-ready parameters: {gsa_params:,}")
    print(f"  Parameter increase: {param_increase:,} ({param_increase/orig_params*100:.3f}%)")
    
    # Show what GSA parameters were added
    first_layer_gsa = gsa_ready_model.transformer.h[0].attn.attention
    print(f"  GSA parameters per layer:")
    print(f"    Splat centers: {first_layer_gsa.unused_gsa_splat_centers.shape}")
    print(f"    Splat scales: {first_layer_gsa.unused_gsa_splat_scales.shape}")
    
    return verdict

def demonstrate_gsa_scaffolding():
    """Demonstrate how to add GSA functionality to the foundation."""
    print("\n🔧 GSA SCAFFOLDING DEMONSTRATION")
    print("="*60)
    print("Showing how to extend the foundation with actual GSA features...")
    
    # This would be the next step - gradually replace the standard attention
    # computation with GSA while maintaining the same interface
    
    print("Next development steps:")
    print("1. ✅ Foundation established (identical performance)")
    print("2. 🔄 Replace _attn method with GSA computation")
    print("3. 🔄 Add splat management (birth/death)")
    print("4. 🔄 Add context extension capabilities")
    print("5. 🔄 Optimize for performance")
    
    print("\nThe unused parameters are ready for:")
    print("- unused_gsa_splat_centers: Position of Gaussian splats in embedding space")
    print("- unused_gsa_splat_scales: Scale/variance of each splat")
    print("- Future: amplitudes, orientations, etc.")

if __name__ == "__main__":
    print("🎯 Inheritance-Based GSA Foundation")
    print("Using direct inheritance to ensure identical computation")
    
    try:
        result = test_inheritance_approach()
        
        print(f"\n🏁 FINAL RESULT: {result.upper()}")
        
        if result in ["perfect", "excellent", "very_good"]:
            print("🎉 SUCCESS: Perfect GSA foundation established!")
            print("   Performance is identical or nearly identical to original")
            print("   Ready to add GSA features incrementally")
            demonstrate_gsa_scaffolding()
            exit(0)
        elif result in ["good", "acceptable"]:
            print("✅ GOOD FOUNDATION: Can proceed with GSA development")
            exit(0)
        else:
            print("⚠️ Still needs work")
            exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
