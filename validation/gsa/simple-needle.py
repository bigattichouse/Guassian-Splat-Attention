#!/usr/bin/env python3
"""
Simple GSA Generation Test

Test to verify that we're actually checking the generated output,
not the input prompt.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_generation_extraction():
    print("🧪 Testing Generation Extraction")
    print("=" * 50)
    
    # Load model
    model_name = "roneneldan/TinyStories-Instruct-33M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.config.use_cache = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create a simple test
    input_text = "Once upon a time, there was a secret code 1234. The story goes on and on about many things, but the important thing to remember is the code. What was the secret code? The answer is:"
    
    print(f"📝 Input text:")
    print(f"   {input_text}")
    print()
    
    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    
    print(f"📊 Input length: {input_length} tokens")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=20,
            temperature=0.1,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode full response
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Decode just the input (for comparison)
    input_only = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)
    
    # Extract ONLY the generated part
    if len(full_response) > len(input_only):
        generated_only = full_response[len(input_only):].strip()
    else:
        generated_only = "[NO NEW TOKENS GENERATED]"
    
    print(f"🔍 ANALYSIS:")
    print(f"   Full response length: {len(full_response)} chars")
    print(f"   Input only length: {len(input_only)} chars")
    print(f"   Generated only length: {len(generated_only)} chars")
    print()
    
    print(f"📥 INPUT CONTAINS '1234': {'1234' in input_only}")
    print(f"🤖 GENERATED CONTAINS '1234': {'1234' in generated_only}")
    print()
    
    print(f"📝 Generated text only:")
    print(f"   '{generated_only}'")
    print()
    
    # Show the bug that was happening
    if '1234' in full_response and '1234' not in generated_only:
        print("🐛 BUG DETECTED:")
        print("   The passkey '1234' appears in full response")
        print("   But NOT in generated text")
        print("   This means we were incorrectly checking the input!")
    elif '1234' in generated_only:
        print("✅ SUCCESS:")
        print("   Model actually generated the passkey")
    else:
        print("❌ MODEL FAILED:")
        print("   Model did not generate the expected passkey")
    
    print()
    print("=" * 50)
    
    # Test with a harder case - passkey NOT in the question
    print("🧪 Testing WITHOUT passkey in question")
    print("=" * 50)
    
    hard_input = "There was once a magic number 5678 hidden in an ancient text. After many adventures and challenges, the hero needed to recall this important number. What was the magic number?"
    
    print(f"📝 Hard input text:")
    print(f"   {hard_input}")
    print()
    
    # Tokenize
    hard_inputs = tokenizer(hard_input, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        hard_outputs = model.generate(
            input_ids=hard_inputs['input_ids'],
            max_new_tokens=20,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Extract generated part
    hard_full = tokenizer.decode(hard_outputs[0], skip_special_tokens=True)
    hard_input_only = tokenizer.decode(hard_inputs['input_ids'][0], skip_special_tokens=True)
    
    if len(hard_full) > len(hard_input_only):
        hard_generated = hard_full[len(hard_input_only):].strip()
    else:
        hard_generated = "[NO NEW TOKENS GENERATED]"
    
    print(f"🤖 Hard generated text:")
    print(f"   '{hard_generated}'")
    print()
    
    print(f"📥 INPUT CONTAINS '5678': {'5678' in hard_input_only}")
    print(f"🤖 GENERATED CONTAINS '5678': {'5678' in hard_generated}")
    
    if '5678' in hard_generated:
        print("🎉 EXCELLENT: Model retrieved the passkey from context!")
    else:
        print("📊 Expected: This is a harder task, may not always succeed")

if __name__ == "__main__":
    test_generation_extraction()
