"""
Real Dataset Configuration Fix for SplatFlow
Provides correct dataset names and configurations for loading real data instead of synthetic.
"""

# Correct dataset configurations that actually work
WORKING_DATASETS = {
    # Text datasets that definitely exist
    'wikitext': {
        'name': 'wikitext',
        'config': 'wikitext-103-raw-v1',  # ✅ This exists
        'description': 'Large-scale Wikipedia text dataset',
        'size': 'Large (~500MB)',
        'quality': 'High quality prose'
    },
    
    'wikitext_small': {
        'name': 'wikitext', 
        'config': 'wikitext-2-raw-v1',   # ✅ This also exists
        'description': 'Smaller Wikipedia text dataset',
        'size': 'Medium (~12MB)', 
        'quality': 'High quality prose'
    },
    
    'openwebtext': {
        'name': 'openwebtext',
        'config': None,  # Default config
        'description': 'OpenAI WebText recreation',
        'size': 'Very large (~37GB)',
        'quality': 'Diverse web content'
    },
    
    'tiny_shakespeare': {
        'name': 'tiny_shakespeare', 
        'config': None,
        'description': 'Complete works of Shakespeare',
        'size': 'Small (~1.1MB)',
        'quality': 'Literary text'
    },
    
    'bookcorpus': {
        'name': 'bookcorpus',
        'config': None,
        'description': 'Collection of over 11,000 books', 
        'size': 'Large (~5GB)',
        'quality': 'Literary content'
    },
    
    'c4': {
        'name': 'c4',
        'config': 'en',
        'description': 'Colossal Clean Crawled Corpus',
        'size': 'Massive (~300GB)',
        'quality': 'Clean web text'
    }
}

def get_dataset_config(dataset_choice='conservative'):
    """
    Get the correct dataset configuration based on choice.
    
    Args:
        dataset_choice: 'conservative', 'aggressive', 'balanced', or specific dataset name
        
    Returns:
        Dictionary with correct dataset configuration
    """
    
    if dataset_choice == 'conservative':
        # Conservative: reliable, smaller datasets
        return {
            'dataset_name': 'wikitext',
            'dataset_config': 'wikitext-2-raw-v1',  # ✅ FIXED: correct config name
            'subset_size': 10000,
            'max_length': 2048,
            'description': 'Conservative: Small WikiText for reliable training'
        }
    
    elif dataset_choice == 'balanced':
        # Balanced: medium size, good quality
        return {
            'dataset_name': 'wikitext',
            'dataset_config': 'wikitext-103-raw-v1',  # ✅ Larger but still manageable
            'subset_size': 50000,
            'max_length': 4096,
            'description': 'Balanced: Full WikiText for comprehensive training'
        }
    
    elif dataset_choice == 'aggressive':
        # Aggressive: large scale training
        return {
            'dataset_name': 'openwebtext',
            'dataset_config': None,
            'subset_size': 100000,
            'max_length': 8192,
            'description': 'Aggressive: OpenWebText for large-scale training'
        }
    
    elif dataset_choice == 'shakespeare':
        return {
            'dataset_name': 'tiny_shakespeare',
            'dataset_config': None,
            'subset_size': None,  # Use all data
            'max_length': 1024,
            'description': 'Literary: Complete works of Shakespeare'
        }
    
    else:
        # Try to match against known datasets
        for key, config in WORKING_DATASETS.items():
            if dataset_choice.lower() in key.lower():
                return {
                    'dataset_name': config['name'],
                    'dataset_config': config['config'],
                    'subset_size': 10000,
                    'max_length': 2048,
                    'description': f"Matched: {config['description']}"
                }
        
        # Default fallback
        return get_dataset_config('conservative')

def create_dataset_loading_script():
    """Create a script to test dataset loading."""
    
    script = '''
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
            print(f"\\n🧪 Testing {description}...")
            
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
    
    print("\\n" + "="*60)
    print("📊 DATASET TEST RESULTS")
    print("="*60)
    
    if working_datasets:
        print("✅ Working datasets:")
        for ds in working_datasets:
            config_str = f" (config: {ds['config']})" if ds['config'] else ""
            print(f"   • {ds['description']}: {ds['name']}{config_str} - {ds['size']} examples")
        
        print("\\n🚀 Recommended configuration:")
        best = working_datasets[0]  # Use first working dataset
        config_line = f", config='{best['config']}'" if best['config'] else ""
        print(f"   dataset_name='{best['name']}'{config_line}")
        
    else:
        print("❌ No datasets working - will need to use synthetic data")
    
    return working_datasets

if __name__ == "__main__":
    working = test_dataset_loading()
'''
    
    return script

def get_quick_fix_config():
    """Get a quick configuration fix for immediate use."""
    
    return {
        # CORRECTED dataset configuration
        'dataset_name': 'wikitext',
        'dataset_config': 'wikitext-2-raw-v1',  # ✅ This is the correct name
        
        # Alternative working options
        'fallback_options': [
            {'name': 'tiny_shakespeare', 'config': None},
            {'name': 'wikitext', 'config': 'wikitext-103-raw-v1'},
            {'name': 'openwebtext', 'config': None}
        ],
        
        # Data processing
        'subset_size': 10000,
        'max_length': 8192,
        'min_length': 50,
        
        # Training parameters
        'batch_size': 1,  # Keep low for 8K sequences
        'gradient_accumulation_steps': 4,
        
        # Tokenization
        'tokenizer_name': 'gpt2',
        'add_special_tokens': True,
        'padding': 'max_length',
        'truncation': True
    }

# Quick configuration update
def update_training_config(current_config):
    """Update existing config with correct dataset settings."""
    
    updated_config = current_config.copy()
    
    # Fix the dataset configuration
    updated_config.update({
        'dataset_name': 'wikitext',
        'dataset_config': 'wikitext-2-raw-v1',  # ✅ CORRECTED
        'subset_size': 10000,
        'verify_dataset': True,  # Add verification step
    })
    
    return updated_config

def print_dataset_options():
    """Print available dataset options."""
    
    print("📚 Available Real Datasets:")
    print("="*50)
    
    for key, config in WORKING_DATASETS.items():
        size_info = config['size']
        quality_info = config['quality']
        config_str = f" (config: {config['config']})" if config['config'] else ""
        
        print(f"🔹 {key}:")
        print(f"   Name: {config['name']}{config_str}")
        print(f"   Size: {size_info}")
        print(f"   Quality: {quality_info}")
        print(f"   Description: {config['description']}")
        print()

if __name__ == "__main__":
    print("🔧 SplatFlow Real Dataset Configuration")
    print("="*50)
    
    # Show dataset options
    print_dataset_options()
    
    # Show quick fix
    print("🚀 Quick Fix Configuration:")
    quick_fix = get_quick_fix_config()
    for key, value in quick_fix.items():
        if key != 'fallback_options':
            print(f"   {key}: {value}")
    
    print("\\n🧪 To test dataset loading, run:")
    print("   python test_datasets.py")
    
    # Create test script
    test_script = create_dataset_loading_script()
    with open('test_datasets.py', 'w') as f:
        f.write(test_script)
    
    print("✅ Created test_datasets.py for verification")
