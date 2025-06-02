"""
High Splat Count Capability Test - Testing Information Bottleneck Hypothesis

This experiment tests whether dramatically increasing splat count (64, 128, 256+) 
can overcome the capability limitations observed in complex reasoning tasks.

Key Scientific Question: Is the capability ceiling due to insufficient splats 
or fundamental architectural limitations?
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import gc

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from splatflow_attention import TrueSplatAttentionLayer
    print("✅ Successfully imported SplatFlow")
except ImportError as e:
    print(f"❌ Failed to import SplatFlow: {e}")
    sys.exit(1)

@dataclass
class SplatScalingResult:
    """Results from testing different splat counts"""
    task_name: str
    splat_counts: List[int]
    accuracies: List[float]
    convergence_steps: List[int]
    final_losses: List[float]
    training_times: List[float]

class SimpleStandardAttention(nn.Module):
    """Optimized standard attention baseline"""
    
    def __init__(self, model_dim: int, num_heads: int = 8):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        
        # Better initialization
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.out.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        # Standard scaled dot-product attention with proper scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.reshape(B, T, D)
        
        return self.out(out)

class HighCapacityTransformer(nn.Module):
    """Enhanced transformer for high splat count testing"""
    
    def __init__(self, vocab_size: int, model_dim: int, attention_layer: nn.Module, 
                 max_seq_len: int = 64, num_layers: int = 3):
        super().__init__()
        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        
        # Embeddings with better initialization
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embedding_dropout = nn.Dropout(0.1)
        
        # Create layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                attn_layer = attention_layer
            else:
                # Create similar attention layers
                if hasattr(attention_layer, 'num_splats'):
                    attn_layer = TrueSplatAttentionLayer(
                        model_dim, attention_layer.num_splats, 
                        dropout=0.1, temperature=1.0
                    )
                else:
                    attn_layer = SimpleStandardAttention(model_dim, num_heads=8)
            
            # Larger FF for higher capacity
            ff_dim = model_dim * 4
            
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
        
        # Better weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
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
        
        # Process through layers
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

class HighSplatTester:
    """Test suite focused on high splat counts and complex reasoning"""
    
    def __init__(self, device: str = 'cuda', vocab_size: int = 50, model_dim: int = 192):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.COPY_TOKEN = 3
        self.QUERY_TOKEN = 4
        
        print(f"🔬 High Splat Count Tester initialized")
        print(f"   Device: {self.device}")
        print(f"   Model dim: {model_dim}")
        print(f"   Focus: Testing information bottleneck hypothesis")
    
    def create_complex_induction_task(self, num_samples: int = 500, seq_len: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """More complex induction: [A B C] ... [A B] -> [C]"""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            # Create pattern of length 3
            pattern_len = 3
            pattern = torch.randint(5, self.vocab_size // 2, (pattern_len,))
            
            # Ensure all different
            while len(torch.unique(pattern)) != pattern_len:
                pattern = torch.randint(5, self.vocab_size // 2, (pattern_len,))
            
            A, B, C = pattern[0], pattern[1], pattern[2]
            
            # Random middle content
            middle_len = np.random.randint(5, seq_len - 8)
            middle = torch.randint(5, self.vocab_size // 2, (middle_len,))
            
            # Input: [A B C] [middle] [A B] [padding]
            input_seq = torch.cat([
                torch.tensor([A, B, C]),
                middle,
                torch.tensor([A, B]),
                torch.zeros(seq_len - middle_len - 5, dtype=torch.long)
            ])
            
            # Target: predict C after [A B]
            target_seq = torch.zeros(seq_len, dtype=torch.long)
            if middle_len + 5 < seq_len:
                target_seq[middle_len + 5] = C
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return torch.stack(inputs), torch.stack(targets)
    
    def create_nested_reasoning_task(self, num_samples: int = 500, seq_len: int = 36) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nested pattern: [A [B C] D] [A [B] -> [C D]"""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            A = torch.randint(5, 20, (1,)).item()
            B = torch.randint(5, 20, (1,)).item()
            C = torch.randint(5, 20, (1,)).item()
            D = torch.randint(5, 20, (1,)).item()
            
            # Ensure all different
            values = [A, B, C, D]
            while len(set(values)) != 4:
                A = torch.randint(5, 20, (1,)).item()
                B = torch.randint(5, 20, (1,)).item()
                C = torch.randint(5, 20, (1,)).item()
                D = torch.randint(5, 20, (1,)).item()
                values = [A, B, C, D]
            
            # Random separators
            sep1_len = np.random.randint(2, 5)
            sep2_len = np.random.randint(2, 5)
            sep1 = torch.randint(25, 35, (sep1_len,))
            sep2 = torch.randint(25, 35, (sep2_len,))
            
            # Input: [A] [sep1] [B C] [sep2] [D] ... [A] [sep1] [B] [query]
            total_pattern_len = 1 + sep1_len + 2 + sep2_len + 1  # A sep1 B C sep2 D
            query_start = total_pattern_len + 3  # some gap
            
            if query_start + 1 + sep1_len + 1 + 2 < seq_len:  # space for A sep1 B + 2 outputs
                input_seq = torch.cat([
                    torch.tensor([A]),
                    sep1,
                    torch.tensor([B, C]),
                    sep2,
                    torch.tensor([D]),
                    torch.randint(35, 45, (3,)),  # separator
                    torch.tensor([A]),
                    sep1,
                    torch.tensor([B, self.QUERY_TOKEN]),
                    torch.zeros(seq_len - query_start - 1 - sep1_len - 2, dtype=torch.long)
                ])
                
                # Target: should output [C, D] after query
                target_seq = torch.zeros(seq_len, dtype=torch.long)
                output_pos = query_start + 1 + sep1_len + 2
                if output_pos + 1 < seq_len:
                    target_seq[output_pos] = C
                    target_seq[output_pos + 1] = D
                
                inputs.append(input_seq)
                targets.append(target_seq)
        
        return torch.stack(inputs), torch.stack(targets)
    
    def create_multi_hop_reasoning_task(self, num_samples: int = 500, seq_len: int = 40) -> Tuple[torch.Tensor, torch.Tensor]:
        """Multi-hop: [A->B] [B->C] [C->D] ... [A] -> [D] (3 hops)"""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            # Create chain: A -> B -> C -> D
            chain = torch.randint(5, 25, (4,))
            while len(torch.unique(chain)) != 4:
                chain = torch.randint(5, 25, (4,))
            
            A, B, C, D = chain[0], chain[1], chain[2], chain[3]
            
            # Create pairs with separators
            sep = torch.randint(30, 40, (2,))
            
            # Input: [A B] [sep] [B C] [sep] [C D] [sep] ... [A] [query]
            input_parts = [
                torch.tensor([A, B]),
                sep,
                torch.tensor([B, C]),
                sep,
                torch.tensor([C, D]),
                sep,
                torch.randint(40, 45, (np.random.randint(3, 8),)),  # random middle
                torch.tensor([A, self.QUERY_TOKEN])
            ]
            
            input_seq = torch.cat(input_parts)
            
            if len(input_seq) <= seq_len - 2:
                # Pad to seq_len
                input_seq = torch.cat([
                    input_seq,
                    torch.zeros(seq_len - len(input_seq), dtype=torch.long)
                ])
                
                # Target: output D after the query
                target_seq = torch.zeros(seq_len, dtype=torch.long)
                query_pos = None
                for i, token in enumerate(input_seq):
                    if token == self.QUERY_TOKEN:
                        query_pos = i
                        break
                
                if query_pos is not None and query_pos + 1 < seq_len:
                    target_seq[query_pos + 1] = D
                
                inputs.append(input_seq)
                targets.append(target_seq)
        
        return torch.stack(inputs), torch.stack(targets)
    
    def create_compositional_task(self, num_samples: int = 500, seq_len: int = 44) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compositional reasoning: combine multiple simple rules"""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            # Rule 1: If you see X, remember Y
            # Rule 2: If you see Z, output the remembered Y
            X = torch.randint(5, 15, (1,)).item()
            Y = torch.randint(15, 25, (1,)).item()
            Z = torch.randint(25, 35, (1,)).item()
            
            # Create sequence with noise
            noise_len = np.random.randint(8, 15)
            noise1 = torch.randint(35, 45, (noise_len,))
            noise2 = torch.randint(35, 45, (noise_len,))
            
            # Input: [noise1] [X Y] [noise2] [Z] [query]
            input_seq = torch.cat([
                noise1,
                torch.tensor([X, Y]),
                noise2,
                torch.tensor([Z, self.QUERY_TOKEN]),
                torch.zeros(seq_len - len(noise1) - 2 - len(noise2) - 2, dtype=torch.long)
            ])
            
            if len(input_seq) == seq_len:
                # Target: output Y after seeing Z
                target_seq = torch.zeros(seq_len, dtype=torch.long)
                
                # Find Z position
                z_pos = None
                for i, token in enumerate(input_seq):
                    if token == Z:
                        z_pos = i
                        break
                
                if z_pos is not None and z_pos + 2 < seq_len:
                    target_seq[z_pos + 2] = Y  # After QUERY_TOKEN
                
                inputs.append(input_seq)
                targets.append(target_seq)
        
        return torch.stack(inputs), torch.stack(targets)
    
    def enhanced_train_and_evaluate(self, model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor, 
                                  task_name: str, max_steps: int = 5000) -> Dict[str, float]:
        """Enhanced training with better optimization for high-capacity models"""
        model = model.to(self.device)
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        start_time = time.time()
        
        # Dynamic batch size based on memory and model size
        if hasattr(model.layers[0]['attention'], 'num_splats'):
            num_splats = model.layers[0]['attention'].num_splats
            if num_splats > 100:
                batch_size = 8
            elif num_splats > 50:
                batch_size = 12
            else:
                batch_size = 16
        else:
            batch_size = 16
        
        # Better optimizer for high-capacity models
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=2e-4,  # Slightly lower for stability
            weight_decay=0.01, 
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Warmup + cosine schedule
        warmup_steps = max_steps // 10
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(step / warmup_steps, 1.0) * (0.5 * (1 + np.cos(np.pi * step / max_steps)))
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
        
        losses = []
        accuracies = []
        convergence_step = max_steps
        best_accuracy = 0.0
        patience = 0
        max_patience = 10
        
        # Split data
        num_train = int(0.8 * len(inputs))
        train_inputs, test_inputs = inputs[:num_train], inputs[num_train:]
        train_targets, test_targets = targets[:num_train], targets[num_train:]
        
        print(f"      Training {task_name} with {batch_size} batch size")
        
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
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"      NaN loss detected at step {step}, stopping")
                break
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            
            # Memory cleanup
            if step % 200 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Evaluate every 500 steps (less frequent for longer training)
            if step % 500 == 0:
                model.eval()
                with torch.no_grad():
                    test_batch_size = min(16, len(test_inputs))
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
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            patience = 0
                            
                            # Higher threshold for convergence with complex tasks
                            if accuracy > 0.80 and convergence_step == max_steps:
                                convergence_step = step
                        else:
                            patience += 1
                        
                        # Early stopping
                        if patience >= max_patience and step > 2000:
                            print(f"      Early stopping at step {step} (accuracy={accuracy:.3f})")
                            break
                    else:
                        accuracy = 0.0
                        accuracies.append(0.0)
                    
                    if step % 2000 == 0:
                        lr = scheduler.get_last_lr()[0]
                        print(f"      {task_name} step {step}: loss={loss.item():.4f}, acc={accuracy:.3f}, lr={lr:.2e}")
        
        training_time = time.time() - start_time
        final_accuracy = best_accuracy
        final_loss = losses[-1] if losses else float('inf')
        
        # Cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return {
            'accuracy': final_accuracy,
            'convergence_steps': convergence_step,
            'final_loss': final_loss,
            'training_time': training_time,
            'losses': losses,
            'accuracies': accuracies
        }
    
    def test_splat_scaling(self, splat_counts: List[int] = [16, 32, 64, 128, 256]) -> List[SplatScalingResult]:
        """Test capability scaling with different splat counts"""
        print(f"\n🔬 Testing Splat Count Scaling")
        print(f"=" * 60)
        print(f"Splat counts to test: {splat_counts}")
        print(f"Focus: Complex reasoning tasks where capability ceiling was observed")
        
        # Focus on tasks where we saw limitations
        complex_tasks = [
            ("Complex Induction", self.create_complex_induction_task, 32, 6000),
            ("Nested Reasoning", self.create_nested_reasoning_task, 36, 7000),
            ("Multi-hop", self.create_multi_hop_reasoning_task, 40, 8000),
            ("Compositional", self.create_compositional_task, 44, 8000),
        ]
        
        # Also test one simple task as baseline
        simple_tasks = [
            ("Enhanced Copy", self.create_simple_enhanced_copy_task, 16, 3000),
        ]
        
        # Adjust based on available memory
        if self.device.type == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory < 8:
                splat_counts = [16, 32, 64]  # Limit for smaller GPUs
                print(f"   Limited to {splat_counts} due to {gpu_memory:.1f}GB GPU memory")
        
        all_results = []
        
        for task_name, task_creator, seq_len, max_steps in simple_tasks + complex_tasks:
            print(f"\n🧪 Testing {task_name} across splat counts")
            print(f"-" * 50)
            
            # Create test data once
            try:
                inputs, targets = task_creator(num_samples=400, seq_len=seq_len)
                print(f"   Data shape: {inputs.shape}")
                complexity = (targets != 0).sum().item()
                print(f"   Task complexity: {complexity} non-zero targets")
                
            except Exception as e:
                print(f"   ❌ Failed to create {task_name} data: {e}")
                continue
            
            task_accuracies = []
            task_convergence = []
            task_losses = []
            task_times = []
            
            for num_splats in splat_counts:
                print(f"\n   🔧 Testing {num_splats} splats...")
                
                try:
                    # Clear memory before each test
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        gc.collect()
                    
                    # Create SplatFlow model with specific splat count
                    splat_attn = TrueSplatAttentionLayer(
                        self.model_dim, 
                        num_splats, 
                        dropout=0.1, 
                        temperature=1.0
                    )
                    
                    model = HighCapacityTransformer(
                        self.vocab_size, 
                        self.model_dim, 
                        splat_attn, 
                        max_seq_len=seq_len + 5,
                        num_layers=3
                    )
                    
                    # Train and evaluate
                    results = self.enhanced_train_and_evaluate(
                        model, inputs, targets, 
                        f"{task_name}-{num_splats}splats", 
                        max_steps
                    )
                    
                    task_accuracies.append(results['accuracy'])
                    task_convergence.append(results['convergence_steps'])
                    task_losses.append(results['final_loss'])
                    task_times.append(results['training_time'])
                    
                    print(f"      Result: {results['accuracy']:.3f} accuracy, {results['convergence_steps']} steps, {results['training_time']:.1f}s")
                    
                    # Memory usage report
                    if self.device.type == 'cuda':
                        memory_used = torch.cuda.max_memory_allocated() / 1024**3
                        print(f"      Memory: {memory_used:.2f}GB peak")
                        torch.cuda.reset_peak_memory_stats()
                    
                    # Clean up before next test
                    del model, splat_attn
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"      ❌ OOM with {num_splats} splats")
                        task_accuracies.append(0.0)
                        task_convergence.append(max_steps)
                        task_losses.append(float('inf'))
                        task_times.append(0.0)
                        
                        # Clear memory and continue
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                        gc.collect()
                        break
                    else:
                        raise e
            
            # Store results for this task
            result = SplatScalingResult(
                task_name=task_name,
                splat_counts=splat_counts[:len(task_accuracies)],
                accuracies=task_accuracies,
                convergence_steps=task_convergence,
                final_losses=task_losses,
                training_times=task_times
            )
            
            all_results.append(result)
            
            # Print task summary
            print(f"\n   📊 {task_name} Summary:")
            for i, (splats, acc) in enumerate(zip(result.splat_counts, result.accuracies)):
                print(f"      {splats:3d} splats: {acc:.3f} accuracy")
            
            # Check for improvement with more splats
            if len(task_accuracies) > 1:
                improvement = task_accuracies[-1] - task_accuracies[0]
                print(f"      📈 Improvement from {result.splat_counts[0]} to {result.splat_counts[-1]} splats: {improvement:+.3f}")
                
                if improvement > 0.1:
                    print(f"      ✅ Significant improvement with more splats!")
                elif improvement > 0.03:
                    print(f"      📊 Modest improvement with more splats")
                else:
                    print(f"      ⚠️  Minimal improvement - possible capability ceiling")
        
        return all_results
    
    def create_simple_enhanced_copy_task(self, num_samples: int = 400, seq_len: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced copying task for baseline comparison"""
        inputs = []
        targets = []
        
        for _ in range(num_samples):
            copy_len = np.random.randint(3, 6)  # 3-5 tokens
            to_copy = torch.randint(5, min(30, self.vocab_size), (copy_len,))
            
            input_seq = torch.cat([
                to_copy,
                torch.tensor([self.COPY_TOKEN]),
                torch.zeros(seq_len - copy_len - 1, dtype=torch.long)
            ])
            
            target_seq = torch.zeros(seq_len, dtype=torch.long)
            start_pos = copy_len + 1
            if start_pos + copy_len <= seq_len:
                target_seq[start_pos:start_pos + copy_len] = to_copy
            
            inputs.append(input_seq)
            targets.append(target_seq)
        
        return torch.stack(inputs), torch.stack(targets)
    
    def analyze_scaling_results(self, results: List[SplatScalingResult]):
        """Comprehensive analysis of splat scaling results"""
        print(f"\n📊 SPLAT SCALING ANALYSIS")
        print(f"=" * 60)
        
        print(f"{'Task':<18} {'16→32':<8} {'32→64':<8} {'64→128':<8} {'128→256':<8} {'Ceiling?':<8}")
        print(f"-" * 70)
        
        significant_improvements = 0
        capability_ceilings = 0
        
        for result in results:
            improvements = []
            
            # Calculate improvements between consecutive splat counts
            for i in range(1, len(result.accuracies)):
                prev_acc = result.accuracies[i-1]
                curr_acc = result.accuracies[i]
                improvement = curr_acc - prev_acc
                improvements.append(improvement)
            
            # Format improvements for display
            imp_strs = []
            for i, imp in enumerate(improvements):
                if i < 4:  # Only show first 4 transitions
                    if imp > 0.05:
                        imp_strs.append(f"+{imp:.2f}")
                        if i == len(improvements) - 1:  # Last improvement
                            significant_improvements += 1
                    elif imp > 0.01:
                        imp_strs.append(f"+{imp:.2f}")
                    else:
                        imp_strs.append(f"{imp:+.2f}")
            
            # Pad to 4 columns
            while len(imp_strs) < 4:
                imp_strs.append("--")
            
            # Detect capability ceiling
            if len(improvements) >= 2:
                recent_improvements = improvements[-2:]
                if all(imp < 0.02 for imp in recent_improvements):
                    ceiling = "YES"
                    capability_ceilings += 1
                else:
                    ceiling = "NO"
            else:
                ceiling = "?"
            
            print(f"{result.task_name:<18} {imp_strs[0]:<8} {imp_strs[1]:<8} {imp_strs[2]:<8} {imp_strs[3]:<8} {ceiling:<8}")
        
        print(f"\n🔍 INFORMATION BOTTLENECK ANALYSIS:")
        print(f"   Tasks showing significant improvement with more splats: {significant_improvements}/{len(results)}")
        print(f"   Tasks hitting capability ceiling: {capability_ceilings}/{len(results)}")
        
        if capability_ceilings > significant_improvements:
            print(f"   📊 CONCLUSION: Strong evidence of capability ceiling")
            print(f"   💡 Information bottleneck appears fundamental, not just due to insufficient splats")
        elif significant_improvements > capability_ceilings:
            print(f"   📊 CONCLUSION: Capability scales with splat count") 
            print(f"   💡 Previous limitations may have been due to insufficient capacity")
        else:
            print(f"   📊 CONCLUSION: Mixed evidence - task-dependent scaling")
        
        # Analyze by task type
        complex_tasks = [r for r in results if r.task_name in ["Complex Induction", "Nested Reasoning", "Multi-hop", "Compositional"]]
        simple_tasks = [r for r in results if r.task_name not in ["Complex Induction", "Nested Reasoning", "Multi-hop", "Compositional"]]
        
        if complex_tasks and simple_tasks:
            print(f"\n🧠 COMPLEXITY ANALYSIS:")
            
            # Average final accuracy for each type
            complex_final_accs = [r.accuracies[-1] for r in complex_tasks if r.accuracies]
            simple_final_accs = [r.accuracies[-1] for r in simple_tasks if r.accuracies]
            
            if complex_final_accs and simple_final_accs:
                complex_avg = np.mean(complex_final_accs)
                simple_avg = np.mean(simple_final_accs)
                
                print(f"   Simple tasks final accuracy: {simple_avg:.3f}")
                print(f"   Complex tasks final accuracy: {complex_avg:.3f}")
                print(f"   Complexity gap: {simple_avg - complex_avg:.3f}")
                
                if simple_avg - complex_avg > 0.3:
                    print(f"   ⚠️  Large complexity gap persists even with high splat counts")
                    print(f"   💡 This suggests fundamental architectural limitations")
        
        # Memory and efficiency analysis
        print(f"\n⚡ EFFICIENCY ANALYSIS:")
        for result in results:
            if len(result.splat_counts) >= 2 and len(result.training_times) >= 2:
                time_ratio = result.training_times[-1] / result.training_times[0]
                acc_ratio = result.accuracies[-1] / max(result.accuracies[0], 0.001)
                efficiency = acc_ratio / time_ratio
                
                print(f"   {result.task_name}: {time_ratio:.1f}x time, {acc_ratio:.1f}x accuracy, {efficiency:.2f} efficiency")
        
        # Final scientific assessment
        print(f"\n🔬 SCIENTIFIC CONCLUSIONS:")
        
        # Check if any complex task achieved high performance
        high_performing_complex = [r for r in complex_tasks if r.accuracies and max(r.accuracies) > 0.8]
        
        if high_performing_complex:
            print(f"   ✅ HIGH SPLAT COUNT BREAKTHROUGH: Complex reasoning achievable with sufficient splats")
            print(f"   💡 Previous capability ceiling was due to insufficient capacity")
        else:
            max_complex_acc = max([max(r.accuracies) for r in complex_tasks if r.accuracies] + [0])
            print(f"   ⚠️  CAPABILITY CEILING CONFIRMED: Max complex task accuracy = {max_complex_acc:.3f}")
            print(f"   💡 Even with 256+ splats, complex reasoning remains limited")
            print(f"   🧠 This suggests fundamental information bottleneck in the architecture")
        
        # Check for scaling patterns
        consistently_improving = sum(1 for r in results if len(r.accuracies) >= 3 and r.accuracies[-1] > r.accuracies[0] + 0.05)
        print(f"   📈 Tasks showing consistent improvement: {consistently_improving}/{len(results)}")
        
        if consistently_improving >= len(results) * 0.6:
            print(f"   💫 SCALING SUCCESS: Most tasks benefit from higher splat counts")
        else:
            print(f"   🔒 SCALING LIMITS: Higher splat counts provide diminishing returns")

def main():
    """Run high splat count capability testing"""
    print("🚀 High Splat Count Capability Test")
    print("Testing whether dramatically more splats can overcome capability limitations...")
    print()
    
    # Check GPU memory for splat count planning
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory: {gpu_memory:.1f}GB")
        
        if gpu_memory < 6:
            model_dim = 128
            max_splats = [16, 32, 64]
            print("Using conservative config for limited GPU memory")
        elif gpu_memory < 12:
            model_dim = 192
            max_splats = [16, 32, 64, 128]
            print("Using moderate config")
        else:
            model_dim = 256
            max_splats = [16, 32, 64, 128, 256]
            print("Using high-capacity config for ample GPU memory")
    else:
        model_dim = 128
        max_splats = [16, 32]
        print("Using minimal config for CPU")
    
    print(f"\n🔧 Test Configuration:")
    print(f"   Model dimension: {model_dim}")
    print(f"   Splat counts to test: {max_splats}")
    print(f"   Focus: Complex reasoning tasks with capability ceiling")
    print(f"   Scientific question: Is ceiling due to insufficient splats or fundamental limits?")
    
    # Clear memory
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    tester = HighSplatTester(device=device, model_dim=model_dim)
    
    print(f"\n🧪 Experimental Design:")
    print(f"   📊 Test same complex tasks with varying splat counts")
    print(f"   🔬 Look for: Accuracy improvement with more splats")
    print(f"   💡 Key insight: If more splats don't help, bottleneck is fundamental")
    print(f"   ⚖️  If they do help, previous tests were capacity-limited")
    
    try:
        results = tester.test_splat_scaling(max_splats)
        tester.analyze_scaling_results(results)
        
        print(f"\n🎯 FINAL VERDICT:")
        
        # Calculate overall trend
        total_improvements = 0
        total_comparisons = 0
        
        for result in results:
            if len(result.accuracies) >= 2:
                improvement = result.accuracies[-1] - result.accuracies[0]
                total_improvements += improvement
                total_comparisons += 1
        
        if total_comparisons > 0:
            avg_improvement = total_improvements / total_comparisons
            
            if avg_improvement > 0.2:
                print(f"   🎉 BREAKTHROUGH: Average improvement of {avg_improvement:.3f} with more splats")
                print(f"   ✅ Previous capability ceiling was due to insufficient splat capacity")
                print(f"   💡 SplatFlow can handle complex reasoning with enough splats")
                print(f"   🚀 This validates the O(n*k) efficiency claims with maintained capability")
            elif avg_improvement > 0.05:
                print(f"   📊 MODERATE SUCCESS: Average improvement of {avg_improvement:.3f}")
                print(f"   ⚖️  More splats help but don't eliminate all limitations")
                print(f"   💡 Suggests both capacity and architectural factors at play")
            else:
                print(f"   ⚠️  CAPABILITY CEILING CONFIRMED: Minimal improvement ({avg_improvement:.3f})")
                print(f"   🔒 Even with order-of-magnitude more splats, complex reasoning limited")
                print(f"   💡 This confirms fundamental information bottleneck in the architecture")
                print(f"   🧠 SplatFlow trades reasoning capability for computational efficiency")
        
        # Compare to standard attention performance expectations
        print(f"\n🔬 ARCHITECTURAL IMPLICATIONS:")
        print(f"   📊 These results reveal the true capability/efficiency tradeoff")
        print(f"   ⚡ O(n*k) efficiency comes with reasoning constraints")
        print(f"   🎯 Optimal use case: Tasks where moderate reasoning + high efficiency valuable")
        print(f"   ❌ Poor fit: Applications requiring complex multi-hop reasoning")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
