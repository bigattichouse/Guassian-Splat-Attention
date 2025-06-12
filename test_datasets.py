
#!/usr/bin/env python3
"""
Test real dataset loading for SplatFlow training.
Run this to verify datasets load correctly before training.
"""

from datasets import load_dataset
import torch
from transformers import GPT2Tokenizer

def test_dataset_loading():
    """Test loading various datasets to find what works."""
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Test configurations
    test_configs = [
        ('wikitext', 'wikitext-2-raw-v1', 'WikiText-2'),
        ('wikitext', 'wikitext-103-raw-v1', 'WikiText-103'), 
        ('tiny_shakespeare', None, 'Shakespeare'),
        ('openwebtext', None, 'OpenWebText (subset)'),
    ]
    
    working_datasets = []
    
    for dataset_name, config_name, description in test_configs:
        try:
            print(f"\n🧪 Testing {description}...")
            
            # Load dataset
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split='train')
            else:
                dataset = load_dataset(dataset_name, split='train')
            
            # Take a small subset for testing
            if len(dataset) > 1000:
                dataset = dataset.select(range(1000))
            
            print(f"✅ {description}: {len(dataset)} examples loaded")
            
            # Test tokenization
            def tokenize_sample(example):
                if 'text' in example:
                    text = example['text']
                elif 'content' in example:
                    text = example['content'] 
                else:
                    text = str(example)
                
                tokens = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                return tokens
            
            # Test a sample
            sample = dataset[0]
            tokens = tokenize_sample(sample)
            print(f"✅ {description}: Tokenization successful, shape: {tokens['input_ids'].shape}")
            
            working_datasets.append({
                'name': dataset_name,
                'config': config_name,
                'description': description,
                'size': len(dataset),
                'status': 'working'
            })
            
        except Exception as e:
            print(f"❌ {description}: Failed - {e}")
    
    print("\n" + "="*60)
    print("📊 DATASET TEST RESULTS")
    print("="*60)
    
    if working_datasets:
        print("✅ Working datasets:")
        for ds in working_datasets:
            config_str = f" (config: {ds['config']})" if ds['config'] else ""
            print(f"   • {ds['description']}: {ds['name']}{config_str} - {ds['size']} examples")
        
        print("\n🚀 Recommended configuration:")
        best = working_datasets[0]  # Use first working dataset
        config_line = f", config='{best['config']}'" if best['config'] else ""
        print(f"   dataset_name='{best['name']}'{config_line}")
        
    else:
        print("❌ No datasets working - will need to use synthetic data")
    
    return working_datasets

if __name__ == "__main__":
    working = test_dataset_loading()
