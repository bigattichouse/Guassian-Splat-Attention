"""
TinyStories-Instruct Chat Training with O(n*k) SplatFlow - FIXED VERSION
Train a conversational model and test long-context needle-in-haystack capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import os
import gc
import random
import re
from typing import Dict, List, Tuple, Optional

# Our breakthrough O(n*k) implementation
from splatflow_attention import SplatFlowGPT

# Enable memory optimizations
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def cleanup_memory():
    """Aggressive memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': free
        }
    return None


class TinyStoriesInstructDataset(Dataset):
    """Dataset for TinyStories-Instruct with chat formatting"""
    
    def __init__(self, tokenizer, max_length: int = 1024, max_examples: int = 3000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Loading TinyStories-Instruct dataset...")
        
        try:
            # Load the instruction-tuned dataset
            dataset = load_dataset("roneneldan/TinyStories-Instruct", split="train")
            
            print(f"Processing {min(max_examples, len(dataset))} examples...")
            
            processed = 0
            for i, example in enumerate(dataset):
                if processed >= max_examples:
                    break
                
                # Format as instruction-response pairs
                if 'prompt' in example and 'completion' in example:
                    formatted_text = self.format_instruction(example['prompt'], example['completion'])
                elif 'text' in example:
                    # Fallback for different formats
                    formatted_text = example['text']
                else:
                    continue
                
                # Tokenize
                tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
                
                if len(tokens) >= 32 and len(tokens) <= max_length:
                    self.examples.append(torch.tensor(tokens, dtype=torch.long))
                    processed += 1
                
                if i % 500 == 0:
                    print(f"  Processed {i}/{min(max_examples, len(dataset))}, valid: {processed}")
            
            print(f"Created dataset with {len(self.examples)} instruction examples")
            
        except Exception as e:
            print(f"Could not load TinyStories-Instruct: {e}")
            print("Creating synthetic instruction dataset...")
            self.examples = self.create_synthetic_instructions(tokenizer, max_examples, max_length)
    
    def format_instruction(self, prompt: str, completion: str) -> str:
        """Format instruction-response pairs for training"""
        return f"### Instruction:\n{prompt.strip()}\n\n### Response:\n{completion.strip()}{self.tokenizer.eos_token}"
    
    def create_synthetic_instructions(self, tokenizer, max_examples: int, max_length: int) -> List[torch.Tensor]:
        """Create synthetic instruction-following examples"""
        print("Creating synthetic instruction dataset...")
        
        instruction_templates = [
            ("Tell me a short story about {topic}.", "Once upon a time, there was a {character} who loved {activity}. One day, they discovered {discovery} and learned that {lesson}. From that day forward, they always remembered to {action}."),
            ("What is {topic}?", "{topic} is {definition}. It's important because {reason}. Many people use it for {purpose}."),
            ("How do you {action}?", "To {action}, you first need to {step1}. Then you should {step2}. Finally, make sure to {step3}. This will help you {result}."),
            ("Write a poem about {topic}.", "{topic} is so {adjective},\nLike {comparison} in the {place}.\nIt makes me feel {emotion},\nAnd fills my heart with {feeling}."),
            ("Explain why {topic} is important.", "{topic} is important because {reason1}. Without it, we would {consequence}. It helps us {benefit} and makes our lives {improvement}.")
        ]
        
        topics = ["friendship", "learning", "kindness", "adventure", "creativity", "honesty", "helping others", "reading", "nature", "family"]
        characters = ["little girl", "brave boy", "wise owl", "friendly dog", "curious cat", "kind teacher", "helpful friend"]
        activities = ["reading books", "playing games", "exploring forests", "helping friends", "learning new things", "drawing pictures"]
        discoveries = ["a magical book", "a hidden treasure", "a new friend", "a special talent", "an important lesson"]
        lessons = ["sharing is caring", "practice makes perfect", "kindness matters", "friends help each other", "learning never stops"]
        actions = ["be kind to others", "keep trying", "help those in need", "share with friends", "never give up"]
        
        examples = []
        for i in range(max_examples):
            template_prompt, template_response = random.choice(instruction_templates)
            
            # Fill in template variables
            topic = random.choice(topics)
            character = random.choice(characters)
            activity = random.choice(activities)
            discovery = random.choice(discoveries)
            lesson = random.choice(lessons)
            action = random.choice(actions)
            
            prompt = template_prompt.format(
                topic=topic, character=character, activity=activity,
                discovery=discovery, lesson=lesson, action=action
            )
            
            response = template_response.format(
                topic=topic, character=character, activity=activity,
                discovery=discovery, lesson=lesson, action=action,
                definition=f"a wonderful thing that brings joy",
                reason=f"it helps us grow and learn",
                purpose=f"making life better",
                step1=f"think carefully about what you want to do",
                step2=f"take your time and be patient",
                step3=f"practice and keep trying",
                result=f"succeed and feel proud",
                adjective=f"beautiful and amazing",
                comparison=f"sunshine",
                place=f"sky",
                emotion=f"happy and excited",
                feeling=f"joy and wonder",
                reason1=f"it teaches us valuable lessons",
                consequence=f"miss out on important experiences",
                benefit=f"become better people",
                improvement=f"more meaningful and fun"
            )
            
            formatted_text = self.format_instruction(prompt, response)
            tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
            
            if len(tokens) <= max_length:
                examples.append(torch.tensor(tokens, dtype=torch.long))
        
        print(f"Created {len(examples)} synthetic instruction examples")
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_chat_fn(batch, pad_token_id=50256):
    """Collate function for variable-length chat sequences"""
    if not batch:
        return torch.empty(0, 0, dtype=torch.long)
    
    # Sort by length for efficient padding
    batch = sorted(batch, key=len, reverse=True)
    max_len = len(batch[0])
    
    padded_batch = []
    for seq in batch:
        if len(seq) < max_len:
            padding = torch.full((max_len - len(seq),), pad_token_id, dtype=torch.long)
            seq = torch.cat([seq, padding])
        padded_batch.append(seq)
    
    return torch.stack(padded_batch)


def train_chat_model():
    """Train SplatFlow on TinyStories-Instruct for chat capabilities"""
    print("🤖 Training O(n*k) SplatFlow Chat Model")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_memory = gpu_props.total_memory / 1024**3
        print(f"GPU: {gpu_props.name}")
        print(f"GPU Memory: {gpu_memory:.1f}GB")
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    # Memory-optimized config for chat training
    config = {
        'max_seq_len': 8192,  # Good balance for chat
        'model_dim': 256,    # Larger for better chat quality
        'num_layers': 4,     # Deeper for better understanding
        'num_splats': 24,    # More splats for longer context
        'batch_size': 4,     # Conservative for memory
        'accumulation_steps': 4,
        'epochs': 6
    }
    
    print(f"\nChat Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create dataset
    dataset = TinyStoriesInstructDataset(
        tokenizer, 
        max_length=config['max_seq_len'],
        max_examples=2000  # Enough for good training
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_chat_fn(batch, tokenizer.pad_token_id),
        num_workers=0
    )
    
    print(f"Training data: {len(dataset)} examples, {len(dataloader)} batches")
    
    # Create SplatFlow model
    cleanup_memory()
    
    model = SplatFlowGPT(
        vocab_size=vocab_size,
        model_dim=config['model_dim'],
        num_layers=config['num_layers'],
        num_splats=config['num_splats'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters")
    print(f"Theoretical complexity: O({config['max_seq_len']} * {config['num_splats']} * {config['model_dim']})")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'] * len(dataloader)
    )
    
    # Training loop
    print(f"\nTraining for {config['epochs']} epochs...")
    
    training_losses = []
    
    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        print("-" * 40)
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                batch = batch.to(device)
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                
                # Gradient accumulation
                optimizer.zero_grad()
                
                for acc_step in range(config['accumulation_steps']):
                    if acc_step < inputs.size(0):
                        step_input = inputs[acc_step:acc_step+1]
                        step_target = targets[acc_step:acc_step+1]
                        
                        logits = model(step_input)
                        loss = criterion(
                            logits.reshape(-1, vocab_size),
                            step_target.reshape(-1)
                        ) / config['accumulation_steps']
                        
                        loss.backward()
                        epoch_loss += loss.item() * config['accumulation_steps']
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                num_batches += 1
                
                if batch_idx % 20 == 0:
                    mem_info = get_gpu_memory_info()
                    mem_usage = mem_info['allocated'] if mem_info else 0
                    print(f"  Batch {batch_idx+1:3d}/{len(dataloader)}: "
                          f"Loss={epoch_loss/num_batches:.4f}, "
                          f"LR={scheduler.get_last_lr()[0]:.2e}, "
                          f"Mem={mem_usage:.2f}GB")
            
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at batch {batch_idx}, skipping...")
                cleanup_memory()
                continue
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')
        training_losses.append(avg_loss)
        
        print(f"Epoch {epoch + 1} Complete: Average Loss = {avg_loss:.4f}")
        
        # Cleanup between epochs
        cleanup_memory()
    
    # Save the trained model
    print("\nSaving trained chat model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'tokenizer_name': 'gpt2',
        'training_losses': training_losses
    }, 'splatflow_chat_model.pt')
    
    print("✅ Training complete! Model saved as 'splatflow_chat_model.pt'")
    
    return model, tokenizer, config


def load_chat_model(device):
    """Load a trained chat model"""
    try:
        checkpoint = torch.load('splatflow_chat_model.pt', map_location=device)
        config = checkpoint['config']
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        model = SplatFlowGPT(
            vocab_size=tokenizer.vocab_size,
            model_dim=config['model_dim'],
            num_layers=config['num_layers'],
            num_splats=config['num_splats'],
            max_seq_len=config['max_seq_len']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Loaded trained chat model with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model, tokenizer, config
        
    except FileNotFoundError:
        print("❌ No trained model found. Please train first.")
        return None, None, None


def chat_with_model(model, tokenizer, config, device):
    """Interactive chat with the trained model"""
    print("\n🤖 SplatFlow Chat Interface")
    print("=" * 30)
    print("Type 'quit' to exit, 'needle' to run needle tests")
    print()
    
    model.eval()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        elif user_input.lower() == 'needle':
            run_needle_in_haystack_tests(model, tokenizer, config, device)
            continue
        elif not user_input:
            continue
        
        # Format as instruction
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        
        # Generate response
        response = generate_response(model, tokenizer, prompt, device, max_new_tokens=100)
        
        # Extract just the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        print(f"Bot: {response}")
        print()


def generate_response(model, tokenizer, prompt: str, device, max_new_tokens: int = 100,
                     temperature: float = 0.8, top_p: float = 0.9) -> str:
    """Generate a response from the model"""
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            if generated.size(1) >= model.max_seq_len:
                break
            
            logits = model(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def run_needle_in_haystack_tests(model, tokenizer, config, device):
    """Run needle-in-haystack tests to demonstrate long-context capabilities"""
    print("\n🔍 Needle-in-Haystack Tests - Demonstrating O(n*k) Long Context")
    print("=" * 70)
    
    # Test different context lengths
    test_configs = [
        #{'length': 512, 'name': 'Short Context'}, 
        {'length': 8192, 'name': 'Longer Context (where standard attention fails!)'}
    ]
    
    results = []
    
    for test_config in test_configs:
        context_length = test_config['length']
        test_name = test_config['name']
        
        print(f"\n{test_name} ({context_length} tokens)")
        print("-" * 50)
        
        # Create needle-in-haystack test
        needle_tests = create_needle_tests(context_length, tokenizer)  # FIX: Pass tokenizer
        
        correct = 0
        total = len(needle_tests)
        
        for i, test in enumerate(needle_tests):
            context = test['context']
            question = test['question']
            answer = test['answer']
            
            # Test if model can find the needle
            full_prompt = f"{context}\n\n### Instruction:\n{question}\n\n### Response:\n"
            
            # Truncate if too long
            tokens = tokenizer.encode(full_prompt)
            if len(tokens) > config['max_seq_len'] - 50:  # Leave room for response
                tokens = tokens[:config['max_seq_len'] - 50]
                full_prompt = tokenizer.decode(tokens)
            
            response = generate_response(model, tokenizer, full_prompt, device, max_new_tokens=20)
            
            # Extract response
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            # Check if answer is in response
            is_correct = answer.lower() in response.lower()
            if is_correct:
                correct += 1
            
            print(f"  Test {i+1}: {'✅' if is_correct else '❌'} - Expected: {answer}, Got: {response[:50]}...")
        
        accuracy = correct / total if total > 0 else 0
        results.append({
            'name': test_name,
            'length': context_length,
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        })
        
        print(f"\n{test_name} Results: {correct}/{total} correct ({accuracy:.1%})")
    
    # Summary
    print(f"\n{'='*70}")
    print("🎯 NEEDLE-IN-HAYSTACK SUMMARY")
    print(f"{'='*70}")
    
    for result in results:
        print(f"{result['name']:25} ({result['length']:4d} tokens): {result['accuracy']:6.1%} accuracy")
    
    # Highlight the achievement
    long_context_result = next((r for r in results if r['length'] >= 1024), None)
    if long_context_result and long_context_result['accuracy'] > 0.5:
        print(f"\n🏆 BREAKTHROUGH: O(n*k) SplatFlow successfully handles {long_context_result['length']} token contexts!")
        print(f"🚀 Standard attention would likely fail with OOM at this length on 4.9GB GPU!")


def create_needle_tests(context_length: int, tokenizer) -> List[Dict]:  # FIX: Add tokenizer parameter
    """Create needle-in-haystack tests for given context length"""
    
    # Base story templates
    story_parts = [
        "Once upon a time in a magical forest, there lived many woodland creatures.",
        "The trees were tall and green, and sunlight filtered through the leaves.",
        "Birds sang beautiful songs while squirrels played among the branches.",
        "In this peaceful place, animals lived together in harmony.",
        "Every morning, the forest would wake up to the sound of chirping birds.",
        "The streams flowed gently, providing fresh water for all the animals.",
        "Flowers bloomed in vibrant colors throughout the forest floor.",
        "Butterflies danced from flower to flower, spreading pollen everywhere.",
        "The wise old owl watched over everyone from his perch in the oak tree.",
        "Rabbits hopped playfully through the meadows and clearings.",
        "Deer grazed peacefully in the sunny spots between the trees.",
        "Fox families played games of hide and seek in the underbrush.",
        "The forest was a place where friendship and kindness always prevailed.",
        "Every evening, all the animals would gather to share stories.",
        "They would talk about their adventures and the lessons they learned.",
        "The youngest animals always listened carefully to the elder's wisdom.",
        "This forest was truly a special and magical place to live."
    ]
    
    # Information to hide (needles)
    needles = [
        ("The secret code is BLUE42", "What is the secret code?", "BLUE42"),
        ("Sarah's favorite number is 47", "What is Sarah's favorite number?", "47"),
        ("The magic word is SUNSHINE", "What is the magic word?", "SUNSHINE"),
        ("The treasure is buried under the red oak tree", "Where is the treasure buried?", "under the red oak tree"),
        ("The password is RAINBOW123", "What is the password?", "RAINBOW123")
    ]
    
    tests = []
    
    for needle_info, question, answer in needles:
        # Create context by repeating and varying story parts
        context_parts = []
        current_length = 0
        
        # Add story parts until we reach desired length
        while current_length < context_length - 200:  # Leave room for needle and question
            for part in story_parts:
                context_parts.append(part)
                current_length += len(tokenizer.encode(part))
                if current_length >= context_length - 200:
                    break
        
        # Insert needle at random position
        insert_position = random.randint(len(context_parts) // 4, 3 * len(context_parts) // 4)
        context_parts.insert(insert_position, needle_info)
        
        # Join context
        context = " ".join(context_parts)
        
        # Truncate if too long
        tokens = tokenizer.encode(context)
        if len(tokens) > context_length - 100:
            tokens = tokens[:context_length - 100]
            context = tokenizer.decode(tokens)
        
        tests.append({
            'context': context,
            'question': question,
            'answer': answer
        })
    
    return tests


def main():
    """Main function to train and test the chat model"""
    print("🚀 O(n*k) SplatFlow Chat Training & Testing")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if model already exists
    model, tokenizer, config = load_chat_model(device)
    
    if model is None:
        print("No trained model found. Starting training...")
        model, tokenizer, config = train_chat_model()
    
    if model is not None:
        print("\n" + "=" * 50)
        print("🎯 TESTING PHASE")
        print("=" * 50)
        
        # Run needle tests first
        run_needle_in_haystack_tests(model, tokenizer, config, device)
        
        # Start interactive chat
        chat_with_model(model, tokenizer, config, device)
    
    print("\n👋 Thanks for testing O(n*k) SplatFlow!")


if __name__ == "__main__":
    main()
