"""
Run the full attention pattern tests with your SplatFlow implementation
"""

import sys
import os

# Add current directory to path so we can import splatflow_attention
sys.path.append(os.getcwd())

try:
    from splatflow_attention import TrueSplatAttentionLayer
    print("✅ Successfully imported SplatFlow")
except ImportError as e:
    print(f"❌ Failed to import SplatFlow: {e}")
    print("Make sure splatflow_attention.py is in the current directory")
    sys.exit(1)

# Import the test framework (copy the class definition from previous artifact)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TestResult:
    """Results from a single attention pattern test"""
    task_name: str
    splatflow_accuracy: float
    standard_accuracy: float
    splatflow_convergence_steps: int
    standard_convergence_steps: int
    splatflow_final_loss: float
    standard_final_loss: float

class SimpleStandardAttention(nn.Module):
    """Simple standard attention for fair comparison"""
    
    def __init__(self, model_dim: int, num_heads: int = 4):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Standard scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.reshape(B, T, D)
        
        return self.out(out)

class SimpleTransformer(nn.Module):
    """Memory-efficient transformer for testing attention patterns"""
    
    def __init__(self, vocab_size: int, model_dim: int, attention_layer: nn.Module, 
                 max_seq_len: int = 64, num_layers: int = 2):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Create layers with memory efficiency in mind
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                # Use the provided attention layer for the first layer
                attn_layer = attention_layer
            else:
                # Create similar attention layers for subsequent layers
                if hasattr(attention_layer, 'num_splats'):
                    # SplatFlow layer
                    attn_layer = TrueSplatAttentionLayer(model_dim, attention_layer.num_splats, dropout=0.1)
                else:
                    # Standard attention layer  
                    attn_layer = SimpleStandardAttention(model_dim, num_heads=4)  # Fewer heads to save memory
            
            # Smaller FF dimension to save memory
            ff_dim = min(model_dim * 2, 512)  # Cap FF size
            
            self.layers.append(nn.ModuleDict({
                'attention': attn_layer,
                'attn_norm': nn.LayerNorm(model_dim),
                'ff': nn.Sequential(
                    nn.Linear(model_dim, ff_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(ff_dim, model_dim),
                    nn.Dropout(0.1)
                ),
                'ff_norm': nn.LayerNorm(model_dim)
            }))
        
        self.final_norm = nn.LayerNorm(model_dim)
        self.output = nn.Linear(model_dim, vocab_size)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        
        # Embeddings
        token_emb = self.embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1))
        x = self.embedding_dropout(token_emb + pos_emb)
        
        # Process through all layers
        for layer in self.layers:
            # Self-attention with residual
            attn_out = layer['attention'](x)
            x = layer['attn_norm'](x + attn_out)
            
            # Feed-forward with residual
            ff_out = layer['ff'](x)
            x = layer['ff_norm'](x + ff_out)
        
        # Final output
        x = self.final_norm(x)
        return self.output(x)

class AttentionPatternTester:
    """Test suite for fundamental attention patterns"""
    
    def __init__(self, device: str = 'cuda', vocab_size: int = 50, model_dim: int = 256):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.COPY_TOKEN = 3
        self.QUERY_TOKEN = 4
        
        print(f"🧪 Attention Pattern Tester initialized")
        print(f"   Device: {self.device}")
        print(f"   Vocab size: {vocab_size}")
        print(f"   Model dim: {model_dim}")
    
    def create_simple_copy_task(self, num_samples: int = 1000, seq_len: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ultra-simple copying: [A] [COPY] -> [A] (single token)"""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            # Single token to copy
            to_copy = torch.randint(5, min(20, self.vocab_size), (1,))
            
            # Input: [token] [COPY_TOKEN] [padding]
            input_seq = torch.cat([
                to_copy,
                torch.tensor([self.COPY_TOKEN]),
                torch.zeros(seq_len - 2, dtype=torch.long)
            ])
            
            # Target: predict the copied token right after COPY
            target_seq = torch.zeros(seq_len, dtype=torch.long)
            target_seq[2] = to_copy[0]  # Position 2 (after COPY token)
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return torch.stack(inputs), torch.stack(targets)
    
    def create_next_token_task(self, num_samples: int = 1000, seq_len: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple next token: [5 6 7] -> predict 8"""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            # Create simple sequence (e.g., 5, 6, 7)
            start_token = torch.randint(5, min(15, self.vocab_size - 3), (1,)).item()
            sequence = torch.arange(start_token, start_token + 3)
            next_token = start_token + 3
            
            # Ensure we don't exceed vocab size
            if next_token >= self.vocab_size:
                continue
                
            # Input: [start, start+1, start+2, padding...]
            input_seq = torch.cat([
                sequence,
                torch.zeros(seq_len - 3, dtype=torch.long)
            ])
            
            # Target: predict next_token at position 3
            target_seq = torch.zeros(seq_len, dtype=torch.long)
            target_seq[3] = next_token
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return torch.stack(inputs), torch.stack(targets)
    def create_copying_task(self, num_samples: int = 1000, seq_len: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
        """Copying task: [A B] [COPY] -> [A B] (2-3 tokens)"""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            copy_len = np.random.randint(2, 4)  # 2-3 tokens only
            to_copy = torch.randint(5, min(25, self.vocab_size), (copy_len,))
            
            input_seq = torch.cat([
                to_copy,
                torch.tensor([self.COPY_TOKEN]),
                torch.zeros(seq_len - copy_len - 1, dtype=torch.long)
            ])
            
            target_seq = torch.zeros(seq_len, dtype=torch.long)
            # Place targets right after COPY token
            start_pos = copy_len + 1
            if start_pos + copy_len <= seq_len:
                target_seq[start_pos:start_pos + copy_len] = to_copy
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return torch.stack(inputs), torch.stack(targets)
    
    def create_induction_task(self, num_samples: int = 1000, seq_len: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """Induction heads: [A B] ... [A] -> [B]"""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            A = torch.randint(5, self.vocab_size, (1,))
            B = torch.randint(5, self.vocab_size, (1,))
            while B == A:
                B = torch.randint(5, self.vocab_size, (1,))
            
            middle_len = np.random.randint(3, seq_len - 5)
            middle = torch.randint(5, self.vocab_size, (middle_len,))
            
            input_seq = torch.cat([
                A, B, middle, A,
                torch.zeros(seq_len - middle_len - 3, dtype=torch.long)
            ])
            
            target_seq = torch.zeros(seq_len, dtype=torch.long)
            target_seq[middle_len + 3] = B
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return torch.stack(inputs), torch.stack(targets)
    
    def create_position_task(self, num_samples: int = 1000, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Position-based: Always copy from position 2"""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            seq = torch.randint(5, self.vocab_size, (seq_len,))
            seq[-1] = self.QUERY_TOKEN
            
            target = torch.zeros(seq_len, dtype=torch.long)
            target[-1] = seq[2]
            
            inputs.append(seq)
            targets.append(target)
        
        return torch.stack(inputs), torch.stack(targets)
    
    def train_and_evaluate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, 
                          task_name: str, max_steps: int = 3000) -> Dict[str, float]:
        """Memory-efficient training with proper evaluation"""
        model = model.to(self.device)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Memory-conscious batch size
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 6:
                batch_size = 16
            elif gpu_memory < 12:
                batch_size = 32
            else:
                batch_size = 64
        else:
            batch_size = 8
        
        # Better optimizer settings
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-5)
        
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        
        losses = []
        accuracies = []
        convergence_step = max_steps
        best_accuracy = 0.0
        patience = 0
        max_patience = 8  # Early stopping patience
        
        # Split data into train/test
        num_train = int(0.8 * len(inputs))
        train_inputs, test_inputs = inputs[:num_train], inputs[num_train:]
        train_targets, test_targets = targets[:num_train], targets[num_train:]
        
        print(f"      Training on {num_train} samples, testing on {len(test_inputs)} samples")
        print(f"      Batch size: {batch_size}")
        
        for step in range(max_steps):
            model.train()
            
            # Sample batch
            actual_batch_size = min(batch_size, len(train_inputs))
            indices = torch.randperm(len(train_inputs))[:actual_batch_size]
            batch_inputs = train_inputs[indices]
            batch_targets = train_targets[indices]
            
            # Forward pass
            logits = model(batch_inputs)
            
            # Compute loss only on non-zero targets
            mask = batch_targets != 0
            if mask.sum() == 0:
                continue
                
            loss = criterion(logits[mask], batch_targets[mask])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            
            # Clear cache periodically
            if self.device.type == 'cuda' and step % 100 == 0:
                torch.cuda.empty_cache()
            
            # Evaluate every 250 steps
            if step % 250 == 0:
                model.eval()
                with torch.no_grad():
                    # Test on held-out data with smaller batches
                    test_batch_size = min(32, len(test_inputs))
                    test_logits = []
                    
                    for i in range(0, len(test_inputs), test_batch_size):
                        batch_test = test_inputs[i:i+test_batch_size]
                        batch_logits = model(batch_test)
                        test_logits.append(batch_logits)
                    
                    test_logits = torch.cat(test_logits, dim=0)
                    predictions = test_logits.argmax(dim=-1)
                    
                    test_mask = test_targets != 0
                    if test_mask.sum() > 0:
                        accuracy = (predictions[test_mask] == test_targets[test_mask]).float().mean().item()
                        accuracies.append(accuracy)
                        
                        # Check for improvement
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            patience = 0
                            
                            # Check convergence (85% accuracy for reasonable threshold)
                            if accuracy > 0.85 and convergence_step == max_steps:
                                convergence_step = step
                        else:
                            patience += 1
                        
                        # Early stopping
                        if patience >= max_patience and step > 1000:
                            print(f"      Early stopping at step {step} (accuracy={accuracy:.3f})")
                            break
                    else:
                        accuracy = 0.0
                        accuracies.append(0.0)
                    
                    if step % 1000 == 0:
                        lr = scheduler.get_last_lr()[0]
                        print(f"      {task_name} step {step}: loss={loss.item():.4f}, acc={accuracy:.3f}, lr={lr:.2e}")
        
        final_accuracy = best_accuracy
        final_loss = losses[-1] if losses else float('inf')
        
        # Clean up
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'accuracy': final_accuracy,
            'convergence_steps': convergence_step,
            'final_loss': final_loss,
            'losses': losses,
            'accuracies': accuracies
        }
    
    def run_all_tests(self, num_splats: int = 12) -> List[TestResult]:
        """Run memory-efficient attention pattern tests"""
        print(f"\n🧪 Running Memory-Efficient Attention Pattern Tests")
        print(f"=" * 60)
        
        # Memory-conscious tasks
        tasks = [
            ("Simple Copy", self.create_simple_copy_task, 8, 2000),     # Single token copy
            ("Next Token", self.create_next_token_task, 8, 2000),      # Sequence prediction  
            ("Copying", self.create_copying_task, 12, 2500),           # Multi-token copy
            ("Position", self.create_position_task, 16, 3000),         # Position-based
        ]
        
        # Add induction only if we have enough memory
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= 6:
                tasks.append(("Induction", self.create_induction_task, 20, 3000))
        
        results = []
        
        for task_name, task_creator, seq_len, max_steps in tasks:
            print(f"\n🔬 Testing {task_name} Task")
            print(f"-" * 40)
            
            try:
                # Clear memory before each task
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Create smaller datasets to save memory
                if task_name in ["Simple Copy", "Next Token"]:
                    inputs, targets = task_creator(num_samples=400, seq_len=seq_len)
                else:
                    inputs, targets = task_creator(num_samples=300, seq_len=seq_len)
                    
                print(f"   Data shape: {inputs.shape}")
                print(f"   Task complexity: {(targets != 0).sum().item()} non-zero targets")
                print(f"   Max training steps: {max_steps}")
                
                # Determine number of layers based on memory
                if self.device.type == 'cuda':
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    num_layers = 2 if gpu_memory < 8 else 3
                else:
                    num_layers = 1
                
                # Test SplatFlow attention
                print(f"   🤖 Training SplatFlow model...")
                splatflow_attn = TrueSplatAttentionLayer(self.model_dim, num_splats, dropout=0.1)
                splatflow_model = SimpleTransformer(self.vocab_size, self.model_dim, splatflow_attn, 
                                                  max_seq_len=seq_len + 5, num_layers=num_layers)
                splatflow_results = self.train_and_evaluate(splatflow_model, inputs, targets, 
                                                           f"SplatFlow-{task_name}", max_steps)
                
                # Clear memory before next model
                del splatflow_model, splatflow_attn
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Test Standard attention
                print(f"   🔧 Training Standard model...")
                standard_attn = SimpleStandardAttention(self.model_dim, num_heads=4)
                standard_model = SimpleTransformer(self.vocab_size, self.model_dim, standard_attn, 
                                                 max_seq_len=seq_len + 5, num_layers=num_layers)
                standard_results = self.train_and_evaluate(standard_model, inputs, targets, 
                                                         f"Standard-{task_name}", max_steps)
                
                # Clear memory after task
                del standard_model, standard_attn, inputs, targets
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Compare results
                result = TestResult(
                    task_name=task_name,
                    splatflow_accuracy=splatflow_results['accuracy'],
                    standard_accuracy=standard_results['accuracy'],
                    splatflow_convergence_steps=splatflow_results['convergence_steps'],
                    standard_convergence_steps=standard_results['convergence_steps'],
                    splatflow_final_loss=splatflow_results['final_loss'],
                    standard_final_loss=standard_results['final_loss']
                )
                
                results.append(result)
                
                # Print comparison
                print(f"   📊 Results:")
                print(f"      SplatFlow:  {splatflow_results['accuracy']:.3f} acc, {splatflow_results['convergence_steps']} steps")
                print(f"      Standard:   {standard_results['accuracy']:.3f} acc, {standard_results['convergence_steps']} steps")
                
                diff = splatflow_results['accuracy'] - standard_results['accuracy']
                if diff > 0.03:
                    print(f"      ✅ SplatFlow wins on {task_name} (+{diff:.3f})")
                elif diff < -0.03:
                    print(f"      ❌ Standard wins on {task_name} ({diff:.3f})")
                else:
                    print(f"      🤝 Tie on {task_name} ({diff:+.3f})")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"   ❌ OOM on {task_name} - skipping this task")
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        return results
    
    def print_summary(self, results: List[TestResult]):
        """Print comprehensive test summary"""
        print(f"\n📋 ATTENTION PATTERN TEST SUMMARY")
        print(f"=" * 60)
        
        print(f"{'Task':<15} {'SplatFlow Acc':<12} {'Standard Acc':<12} {'Winner':<10}")
        print(f"-" * 60)
        
        splatflow_wins = 0
        standard_wins = 0
        ties = 0
        
        for result in results:
            if result.splatflow_accuracy > result.standard_accuracy + 0.01:
                winner = "SplatFlow"
                splatflow_wins += 1
            elif result.standard_accuracy > result.splatflow_accuracy + 0.01:
                winner = "Standard"
                standard_wins += 1
            else:
                winner = "Tie"
                ties += 1
            
            print(f"{result.task_name:<15} {result.splatflow_accuracy:<12.3f} {result.standard_accuracy:<12.3f} {winner:<10}")
        
        print(f"\n🏆 FINAL SCORE:")
        print(f"   SplatFlow wins: {splatflow_wins}")
        print(f"   Standard wins: {standard_wins}")
        print(f"   Ties: {ties}")
        
        # Overall assessment
        if splatflow_wins > standard_wins:
            print(f"\n✅ SplatFlow shows superior pattern learning capability!")
        elif standard_wins > splatflow_wins:
            print(f"\n❌ Standard attention outperforms SplatFlow on pattern learning")
            print(f"   This suggests SplatFlow may have expressiveness limitations")
        else:
            print(f"\n🤝 SplatFlow matches standard attention capability")
        
        # Identify specific weaknesses
        weak_tasks = [r for r in results if r.splatflow_accuracy < r.standard_accuracy - 0.05]
        if weak_tasks:
            print(f"\n⚠️  SplatFlow struggles with: {', '.join(r.task_name for r in weak_tasks)}")
            print(f"   These may represent fundamental limitations of the splat approach")
        
        strong_tasks = [r for r in results if r.splatflow_accuracy > r.standard_accuracy + 0.05]
        if strong_tasks:
            print(f"\n🎯 SplatFlow excels at: {', '.join(r.task_name for r in strong_tasks)}")

def main():
    """Run the comprehensive pattern testing suite with memory management"""
    print("🚀 SplatFlow Fundamental Pattern Tests v2.0 (Memory Optimized)")
    print("Testing the building blocks of attention with GPU memory management...")
    
    # Memory-conscious configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        # Check available GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f}GB")
        
        if gpu_memory < 6:
            model_dim = 128
            num_layers = 2
            print("Using small config for limited GPU memory")
        elif gpu_memory < 12:
            model_dim = 192
            num_layers = 2
            print("Using medium config for moderate GPU memory") 
        else:
            model_dim = 256
            num_layers = 3
            print("Using large config for high GPU memory")
    else:
        model_dim = 64
        num_layers = 2
        print("Using minimal config for CPU")
    
    # Clear any existing GPU memory
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    tester = AttentionPatternTester(device=device, model_dim=model_dim)
    
    print(f"🔧 Test Configuration:")
    print(f"   Model dimension: {model_dim}")
    print(f"   Number of layers: {num_layers}")
    print(f"   Device: {device}")
    print(f"   Memory optimized training")
    
    # Run tests with memory management
    try:
        results = tester.run_all_tests(num_splats=12)  # Fewer splats to save memory
        tester.print_summary(results)
        
        print(f"\n🎯 COMPREHENSIVE ANALYSIS:")
        
        # Check if models actually learned the tasks
        learnable_tasks = [r for r in results if max(r.splatflow_accuracy, r.standard_accuracy) > 0.7]
        failed_tasks = [r for r in results if max(r.splatflow_accuracy, r.standard_accuracy) < 0.4]
        
        if len(learnable_tasks) == 0:
            print(f"   ❌ CRITICAL: Both models failed all tasks - need more training or different approach")
        elif len(failed_tasks) > len(results) // 2:
            print(f"   ⚠️  WARNING: Most tasks challenging for current setup")
        else:
            print(f"   ✅ SUCCESS: Models learned {len(learnable_tasks)}/{len(results)} tasks")
            
            # Analyze SplatFlow performance on learned tasks
            splatflow_better = [r for r in learnable_tasks if r.splatflow_accuracy > r.standard_accuracy + 0.03]
            standard_better = [r for r in learnable_tasks if r.standard_accuracy > r.splatflow_accuracy + 0.03]
            
            if len(splatflow_better) > len(standard_better):
                print(f"   🏆 SplatFlow shows superior learning on fundamental patterns")
            elif len(standard_better) > len(splatflow_better):
                print(f"   ⚠️  Standard attention outperforms SplatFlow on learned tasks")
                print(f"       This suggests potential expressiveness limitations")
            else:
                print(f"   🤝 SplatFlow matches standard attention capability")
        
        # Memory usage summary
        if device == 'cuda':
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"   📊 Peak GPU memory used: {memory_used:.2f}GB")
        
        print(f"\n💡 INTERPRETATION:")
        print(f"   These results show SplatFlow's ability to learn fundamental reasoning patterns")
        print(f"   Performance relative to standard attention reveals true capabilities")
        print(f"   Memory efficiency enables testing on resource-constrained hardware")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
