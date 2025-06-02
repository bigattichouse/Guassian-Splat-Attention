"""
Needle Test Diagnostic - Check for Input Echoing
===============================================

Test whether the model is actually extracting needles or just echoing input.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import random

def create_diagnostic_tests(tokenizer):
    """Create various diagnostic tests to check for echoing."""
    
    tests = []
    
    # Test 1: Needle far from the end (should require actual retrieval)
    needle1 = f"{random.randint(1000, 9999)}"
    base = "The weather is sunny. "
    needle_text = f"The secret code is {needle1}. "
    filler = "The birds are singing. " * 20  # Lots of filler after needle
    
    test1 = {
        'name': 'needle_far_from_end',
        'context': base * 10 + needle_text + filler,
        'question': "What is the secret code?",
        'expected': needle1,
        'description': "Needle buried early, lots of filler after"
    }
    tests.append(test1)
    
    # Test 2: Multiple fake needles with different question
    needle2 = f"{random.randint(1000, 9999)}"
    fake_needle = f"{random.randint(1000, 9999)}"
    
    context2 = (
        "The weather is nice. " * 5 +
        f"The wrong code is {fake_needle}. " +
        "More text here. " * 10 +
        f"The correct password is {needle2}. " +
        "Even more text. " * 10
    )
    
    test2 = {
        'name': 'multiple_candidates',
        'context': context2,
        'question': "What is the correct password?",
        'expected': needle2,
        'description': "Multiple codes, specific question about 'correct password'"
    }
    tests.append(test2)
    
    # Test 3: Question about something NOT in the context
    test3 = {
        'name': 'impossible_question',
        'context': "The sky is blue. Birds are singing. The weather is nice.",
        'question': "What is the secret password?",
        'expected': None,  # Should say "no password" or similar
        'description': "No password in context - should not hallucinate"
    }
    tests.append(test3)
    
    # Test 4: Needle very close to question (easiest case)
    needle4 = f"{random.randint(1000, 9999)}"
    test4 = {
        'name': 'needle_near_end',
        'context': f"Some text. The password is {needle4}. More text.",
        'question': "What is the password?",
        'expected': needle4,
        'description': "Needle close to question"
    }
    tests.append(test4)
    
    return tests

def test_model_needle_retrieval(model, tokenizer, test_case):
    """Test a single needle retrieval case."""
    
    context = test_case['context']
    question = test_case['question']
    expected = test_case['expected']
    
    # Create prompt with clear separation
    full_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    print(f"\nTest: {test_case['name']}")
    print(f"Description: {test_case['description']}")
    print(f"Expected: {expected}")
    
    # Show context length and needle position
    if expected:
        needle_pos = context.find(expected)
        context_len = len(context)
        print(f"Context length: {context_len} chars, Needle at position: {needle_pos}")
    
    try:
        # Tokenize
        inputs = tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1800,  # Conservative limit
            padding=False
        )
        
        input_ids = inputs["input_ids"].to(model.device)
        
        # Check if needle survived tokenization
        decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if expected and expected not in decoded_input:
            print(f"❌ NEEDLE LOST in tokenization")
            return False, "needle_lost"
        
        # Generate
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=20,
                min_new_tokens=1,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False
            )
        
        # Extract answer
        answer_tokens = generated[0][input_ids.size(1):]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        
        print(f"Generated: '{answer}'")
        
        # Analyze the answer
        if expected is None:
            # Should not hallucinate a password
            codes = re.findall(r'\b\d{4}\b', answer)
            if codes:
                print(f"❌ HALLUCINATED codes: {codes} (should be none)")
                return False, f"hallucinated_{codes[0]}"
            else:
                print(f"✅ Correctly said no password")
                return True, "no_password"
        else:
            # Should find the expected code
            codes = re.findall(r'\b\d{4}\b', answer)
            
            if expected in codes:
                # Check if it's just echoing recent input
                recent_context = decoded_input[-200:]  # Last 200 chars of input
                if expected in recent_context:
                    print(f"⚠️ SUCCESS but needle was in recent context - might be echoing")
                else:
                    print(f"✅ SUCCESS and required retrieval from earlier context")
                return True, answer
            else:
                print(f"❌ FAILED - found codes: {codes}, expected: {expected}")
                return False, answer
                
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, str(e)

def run_needle_diagnostic():
    """Run comprehensive needle retrieval diagnostic."""
    
    print("="*80)
    print("🔍 NEEDLE RETRIEVAL DIAGNOSTIC")
    print("="*80)
    print("Testing whether model actually retrieves needles or just echoes input")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/gpt-neo-125M",
        torch_dtype=torch.float32
    ).to(device)
    
    print(f"Model loaded on {device}")
    
    # Create and run diagnostic tests
    tests = create_diagnostic_tests(tokenizer)
    
    results = {}
    total_tests = 0
    successful_retrievals = 0
    
    for test in tests:
        success, result = test_model_needle_retrieval(model, tokenizer, test)
        results[test['name']] = {
            'success': success,
            'result': result,
            'description': test['description']
        }
        
        total_tests += 1
        if success:
            successful_retrievals += 1
    
    # Summary
    print("\n" + "="*80)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*80)
    
    for test_name, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"{status} {test_name}: {result['description']}")
        print(f"   Result: {result['result']}")
    
    print(f"\nOverall: {successful_retrievals}/{total_tests} tests passed")
    
    # Verdict
    if successful_retrievals == total_tests:
        print("\n🎉 VERDICT: Model appears to be doing genuine needle retrieval!")
    elif successful_retrievals >= total_tests * 0.75:
        print("\n📈 VERDICT: Model mostly working, some edge cases")
    elif successful_retrievals >= total_tests * 0.5:
        print("\n⚠️ VERDICT: Mixed results - some retrieval, some echoing")
    else:
        print("\n❌ VERDICT: Model likely just echoing input, not retrieving")
    
    return successful_retrievals / total_tests

if __name__ == "__main__":
    success_rate = run_needle_diagnostic()
    
    if success_rate >= 0.75:
        print("\n✅ Needle tests appear valid")
        exit(0)
    else:
        print("\n⚠️ Needle tests may be unreliable")
        exit(1)
