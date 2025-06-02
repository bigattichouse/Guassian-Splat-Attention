"""
Comprehensive DirectionalGSA Test - Complete Implementation
==========================================================

The definitive test of DirectionalGSA with SplatNN-inspired directional information flow.
Includes all bug fixes, validated needle retrieval, and comprehensive context extension testing.

This represents the culmination of our DirectionalGSA research, combining:
- Proper GPT-Neo weight copying
- Fixed tensor operations and bias extension
- Validated needle-in-haystack evaluation methodology
- Systematic strength scaling and context extension testing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention, GPTNeoAttention
import warnings
import json
import math
import random
import logging
from pathlib import Path
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        return f"GPU: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
    else:
        import psutil
        return f"CPU: {psutil.Process().memory_info().rss / 1024**3:.2f}GB"

@dataclass
class ExperimentConfig:
    """Configuration for the comprehensive DirectionalGSA experiment."""
    model_name: str = "EleutherAI/gpt-neo-125M"
    layer_to_replace: int = 6
    n_splats: int = 8
    
    # Testing parameters
    baseline_lengths: List[int] = None
    strength_range: List[float] = None
    extended_lengths: List[int] = None
    tests_per_length: int = 3
    
    # GSA parameters
    initial_strength: float = 0.001
    max_strength: float = 0.2
    target_context_length: int = 4096
    
    # Output
    results_dir: str = "comprehensive_gsa_results"
    
    def __post_init__(self):
        if self.baseline_lengths is None:
            self.baseline_lengths = [512, 1024, 1536, 2048]
        if self.strength_range is None:
            self.strength_range = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15]
        if self.extended_lengths is None:
            self.extended_lengths = [2560, 3072, 3584, 4000]

class ComprehensiveDirectionalGSA(GPTNeoSelfAttention):
    """
    Comprehensive DirectionalGSA implementation with all fixes and optimizations.
    
    Features:
    - SplatNN-inspired directional information flow
    - Proper tensor operations throughout
    - Adaptive causal masking for context extension
    - Conservative blending for stability
    - Comprehensive parameter management
    """
    
    def __init__(self, config, attention_type, layer_id):
        super().__init__(config, attention_type, layer_id)
        
        self.layer_id = layer_id
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.n_splats = 8  # Moderate number for balance of expressiveness and efficiency
        
        # Core DirectionalGSA parameters
        self.splat_positions = nn.Parameter(
            torch.randn(config.num_attention_heads, self.n_splats, self.head_dim) * 0.02
        )
        self.splat_directions = nn.Parameter(  # Key innovation from SplatNN
            torch.randn(config.num_attention_heads, self.n_splats, self.head_dim) * 0.02
        )
        self.splat_log_scales = nn.Parameter(
            torch.zeros(config.num_attention_heads, self.n_splats) + math.log(0.8)
        )
        self.splat_log_amplitudes = nn.Parameter(
            torch.zeros(config.num_attention_heads, self.n_splats) - math.log(float(self.n_splats))
        )
        
        # Control parameters
        self.directional_strength = nn.Parameter(torch.tensor(0.3))  # Balance spatial vs directional
        self.causal_strength = nn.Parameter(torch.tensor(2.0))       # Long-range connection decay
        self.temperature = nn.Parameter(torch.tensor(1.0))           # Attention sharpness
        self.gsa_strength = nn.Parameter(torch.tensor(-6.0))         # Overall GSA influence
        
        # State management
        self.enable_gsa = False
        self.max_extended_length = 2048  # Track maximum sequence length seen
        
        logger.info(f"ComprehensiveDirectionalGSA initialized for layer {layer_id}")
        logger.info(f"  - Splats: {self.n_splats}")
        logger.info(f"  - Head dimension: {self.head_dim}")
        logger.info(f"  - Parameter count: {sum(p.numel() for p in self.parameters() if 'splat' in str(p))}")
    
    def _extend_bias_safely(self, target_length):
        """Safely extend bias matrix with proper cleanup."""
        current_size = self.bias.size(-1)
        if current_size >= target_length:
            return
            
        logger.info(f"Extending bias for layer {self.layer_id}: {current_size} → {target_length}")
        
        device = self.bias.device
        dtype = self.bias.dtype
        
        # Create new bias tensor
        new_bias = torch.tril(torch.ones((target_length, target_length), device=device, dtype=dtype))
        new_bias = new_bias.view(1, 1, target_length, target_length)
        new_bias = (1.0 - new_bias) * -1e4
        
        # Copy existing values
        with torch.no_grad():
            new_bias[:, :, :current_size, :current_size] = self.bias
        
        # Replace buffer safely
        delattr(self, 'bias')
        self.register_buffer("bias", new_bias)
        
        self.max_extended_length = max(self.max_extended_length, target_length)
        logger.info(f"✓ Bias extended to {target_length}x{target_length}")
    
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        """Main attention computation with DirectionalGSA integration."""
        seq_len = query.size(2)
        
        # Extend bias if needed for longer sequences
        self._extend_bias_safely(seq_len)
        
        if not self.enable_gsa:
            return super()._attn(query, key, value, attention_mask, head_mask)
        
        try:
            gsa_strength = torch.sigmoid(self.gsa_strength)
            
            # Minimum threshold for GSA activation
            if gsa_strength < 0.001:
                return super()._attn(query, key, value, attention_mask, head_mask)
            
            # Always get standard attention as baseline
            standard_output, standard_weights = super()._attn(query, key, value, attention_mask, head_mask)
            
            # Compute DirectionalGSA output
            directional_output = self._compute_directional_gsa(query, key, value)
            
            # Adaptive blending based on sequence length and GSA strength
            blend_factor = self._compute_adaptive_blend_factor(gsa_strength, seq_len)
            
            # Blend outputs
            blended_output = (1 - blend_factor) * standard_output + blend_factor * directional_output
            
            return blended_output, standard_weights
            
        except Exception as e:
            logger.warning(f"DirectionalGSA failed on layer {self.layer_id}: {e}")
            logger.warning(f"Falling back to standard attention")
            self.enable_gsa = False
            return super()._attn(query, key, value, attention_mask, head_mask)
    
    def _compute_adaptive_blend_factor(self, gsa_strength, seq_len):
        """Compute adaptive blend factor based on GSA strength and sequence length."""
        # Base blend factor from GSA strength
        base_blend = min(0.4, gsa_strength)
        
        # Increase blend factor for longer sequences where DirectionalGSA should help more
        if seq_len > 1024:
            length_bonus = min(0.3, (seq_len - 1024) / 2048)  # Up to 30% bonus
            return min(0.6, base_blend + length_bonus)
        else:
            return base_blend
    
    def _compute_directional_gsa(self, query, key, value):
        """
        Core DirectionalGSA computation with SplatNN-inspired directional flow.
        
        This is the key innovation: each splat has both a position in embedding space
        and a direction vector that guides information flow.
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        device = query.device
        
        # Get splat parameters with proper bounds
        positions = F.normalize(self.splat_positions[:num_heads], dim=-1) * math.sqrt(head_dim) * 0.4
        directions = F.normalize(self.splat_directions[:num_heads], dim=-1)
        scales = torch.exp(self.splat_log_scales[:num_heads]).clamp(min=0.2, max=1.5)
        amplitudes = F.softmax(self.splat_log_amplitudes[:num_heads], dim=-1)
        
        # Control parameters
        dir_strength = torch.sigmoid(self.directional_strength)
        causal_strength = torch.clamp(self.causal_strength, min=1.0, max=4.0)
        temp = torch.clamp(self.temperature, min=0.8, max=1.3)
        
        # Compute token-to-splat affinities (spatial + directional)
        query_affinities = self._compute_combined_affinities(
            query, positions, directions, scales, dir_strength
        )
        key_affinities = self._compute_combined_affinities(
            key, positions, directions, scales, dir_strength
        )
        
        # Compute attention through splats (core DirectionalGSA mechanism)
        attention_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len, device=device)
        
        for s in range(self.n_splats):
            # Get affinities for this splat
            q_aff = query_affinities[:, :, :, s]  # [B, H, N]
            k_aff = key_affinities[:, :, :, s]    # [B, H, N]
            amp = amplitudes[:, s]                # [H]
            
            # Information flows through this splat from keys to queries
            splat_attention = torch.einsum('bhi,bhj->bhij', q_aff, k_aff)
            splat_attention = splat_attention * amp.view(1, -1, 1, 1)
            
            attention_weights = attention_weights + splat_attention
        
        # Apply adaptive causal mask for context extension
        causal_mask = self._create_adaptive_causal_mask(seq_len, device, causal_strength)
        attention_weights = attention_weights * causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Normalize and apply temperature
        attention_weights = F.softmax(attention_weights / temp, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output
    
    def _compute_combined_affinities(self, tokens, positions, directions, scales, dir_strength):
        """
        Compute combined spatial and directional affinities.
        
        This implements the core SplatNN concept: tokens have both spatial proximity
        to splats AND directional alignment with the splat's information flow direction.
        """
        batch_size, num_heads, seq_len, head_dim = tokens.shape
        
        # Normalize tokens for stable computation
        tokens_norm = F.normalize(tokens, dim=-1) * math.sqrt(head_dim) * 0.4
        
        # Expand tensors for broadcasting
        tokens_exp = tokens_norm.unsqueeze(3)  # [B, H, N, 1, D]
        positions_exp = positions.unsqueeze(0).unsqueeze(2)  # [1, H, 1, K, D]
        directions_exp = directions.unsqueeze(0).unsqueeze(2)  # [1, H, 1, K, D]
        scales_exp = scales.unsqueeze(0).unsqueeze(2)  # [1, H, 1, K]
        
        # 1. Spatial affinities (how close is token to splat position?)
        spatial_diff = tokens_exp - positions_exp
        spatial_distances = torch.sqrt(torch.sum(spatial_diff ** 2, dim=-1) + 1e-6)
        spatial_affinities = torch.exp(-0.5 * (spatial_distances / scales_exp) ** 2)
        
        # 2. Directional affinities (how well does token align with splat's flow direction?)
        # This is the key SplatNN innovation
        directional_alignment = torch.sum(tokens_exp * directions_exp, dim=-1)
        directional_affinities = torch.sigmoid(directional_alignment)
        
        # 3. Combine spatial and directional affinities
        # dir_strength controls the balance: 0 = pure spatial, 1 = pure directional
        combined_affinities = (
            (1 - dir_strength) * spatial_affinities + 
            dir_strength * directional_affinities
        )
        
        # Normalize across splats for each token
        combined_affinities = combined_affinities / (
            combined_affinities.sum(dim=-1, keepdim=True) + 1e-6
        )
        
        return combined_affinities
    
    def _create_adaptive_causal_mask(self, seq_len, device, causal_strength):
        """
        Create adaptive causal mask that enables context extension.
        
        For longer sequences, allows some information flow from future tokens
        with exponential decay, enabling longer-range dependencies.
        """
        # Basic causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        
        # For longer sequences, add exponentially decaying future connections
        if seq_len > 1024:
            # Create index tensors for vectorized computation
            i_indices = torch.arange(seq_len, device=device).unsqueeze(1)
            j_indices = torch.arange(seq_len, device=device).unsqueeze(0)
            
            # Compute distances for future positions
            future_mask = j_indices > i_indices
            distances = (j_indices - i_indices).float()
            
            # Exponential decay for future connections
            decay_values = torch.exp(-distances / causal_strength)
            
            # Only add significant connections
            significant_connections = decay_values > 0.02
            long_range_connections = future_mask & significant_connections
            
            # Add to mask
            mask = mask + (long_range_connections.float() * decay_values)
        
        return mask
    
    def set_gsa_strength(self, strength: float):
        """Set GSA strength for experiments."""
        with torch.no_grad():
            strength = torch.clamp(torch.tensor(strength), 0.0001, 0.8)
            logit = torch.log(strength / (1 - strength))
            self.gsa_strength.copy_(logit)
        logger.info(f"Layer {self.layer_id}: GSA strength set to {strength:.4f}")
    
    def enable_directional_gsa(self, strength: float = 0.01):
        """Enable DirectionalGSA with specified strength."""
        self.enable_gsa = True
        self.set_gsa_strength(strength)
        logger.info(f"Layer {self.layer_id}: DirectionalGSA enabled")
    
    def get_diagnostics(self) -> Dict:
        """Get comprehensive diagnostics about GSA state."""
        if not self.enable_gsa:
            return {"enabled": False}
        
        with torch.no_grad():
            diagnostics = {
                "enabled": True,
                "gsa_strength": torch.sigmoid(self.gsa_strength).item(),
                "directional_strength": torch.sigmoid(self.directional_strength).item(),
                "causal_strength": self.causal_strength.item(),
                "temperature": self.temperature.item(),
                "n_splats": self.n_splats,
                "max_extended_length": self.max_extended_length,
                
                # Splat statistics
                "avg_scale": torch.exp(self.splat_log_scales).mean().item(),
                "scale_std": torch.exp(self.splat_log_scales).std().item(),
                "amplitudes": F.softmax(self.splat_log_amplitudes, dim=-1).mean(dim=0).tolist(),
                "active_splats": (F.softmax(self.splat_log_amplitudes, dim=-1).mean(dim=0) > 0.05).sum().item(),
                
                # Parameter norms
                "position_norm": self.splat_positions.norm().item(),
                "direction_norm": self.splat_directions.norm().item(),
            }
        
        return diagnostics

class ComprehensiveGSAAttention(GPTNeoAttention):
    """Wrapper for ComprehensiveDirectionalGSA."""
    
    def __init__(self, config, layer_id):
        super().__init__(config, layer_id)
        attention_type = config.attention_layers[layer_id]
        self.attention = ComprehensiveDirectionalGSA(config, attention_type, layer_id)

class ModelManager:
    """Manages model loading, GSA installation, and context extension."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.gsa_layer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load_model(self):
        """Load the base model and tokenizer."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32
        ).to(self.device)
        
        logger.info(f"Model loaded on {self.device}. {get_memory_usage()}")
    
    def install_directional_gsa(self) -> bool:
        """Install DirectionalGSA with proper weight copying."""
        layer_idx = self.config.layer_to_replace
        logger.info(f"Installing ComprehensiveDirectionalGSA on layer {layer_idx}")
        
        try:
            # Create GSA layer
            gsa_attention = ComprehensiveGSAAttention(self.model.config, layer_idx)
            gsa_attention = gsa_attention.to(self.device)
            
            # Copy weights from original layer
            original = self.model.transformer.h[layer_idx].attn.attention
            gsa = gsa_attention.attention
            
            with torch.no_grad():
                # Copy all projection weights (GPT-Neo uses separate projections)
                gsa.q_proj.weight.copy_(original.q_proj.weight)
                gsa.k_proj.weight.copy_(original.k_proj.weight)
                gsa.v_proj.weight.copy_(original.v_proj.weight)
                gsa.out_proj.weight.copy_(original.out_proj.weight)
                
                # Copy biases if they exist
                for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                    orig_proj = getattr(original, proj_name)
                    gsa_proj = getattr(gsa, proj_name)
                    if hasattr(orig_proj, 'bias') and orig_proj.bias is not None:
                        gsa_proj.bias.copy_(orig_proj.bias)
                
                # Copy attention buffers
                gsa.masked_bias.copy_(original.masked_bias)
                gsa.bias.copy_(original.bias)
            
            # Replace the layer
            self.model.transformer.h[layer_idx].attn = gsa_attention
            self.gsa_layer = gsa_attention.attention
            
            logger.info("✓ ComprehensiveDirectionalGSA installed successfully")
            logger.info(f"  - GSA parameters added: {sum(p.numel() for name, p in gsa.named_parameters() if 'splat' in name)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to install DirectionalGSA: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extend_context(self):
        """Extend model context length."""
        target_length = self.config.target_context_length
        original_max = self.model.config.max_position_embeddings
        
        if target_length <= original_max:
            logger.info(f"Context length already sufficient: {original_max}")
            return
        
        logger.info(f"Extending context: {original_max} → {target_length}")
        
        # Extend position embeddings
        original_embeddings = self.model.transformer.wpe.weight.data
        embedding_dim = original_embeddings.size(1)
        
        new_embeddings = torch.zeros(
            target_length, embedding_dim,
            device=original_embeddings.device,
            dtype=original_embeddings.dtype
        )
        
        # Copy original embeddings
        new_embeddings[:original_max] = original_embeddings
        
        # Extend with cyclic pattern and gradual decay
        for i in range(original_max, target_length):
            base_idx = i % original_max
            cycle = i // original_max
            decay = 0.95 ** cycle
            new_embeddings[i] = original_embeddings[base_idx] * decay
        
        # Replace embedding layer
        self.model.transformer.wpe = nn.Embedding(target_length, embedding_dim)
        self.model.transformer.wpe.weight.data = new_embeddings
        self.model.transformer.wpe = self.model.transformer.wpe.to(self.device)
        
        # Update all relevant config parameters
        self.model.config.max_position_embeddings = target_length
        self.model.config.n_positions = target_length
        if hasattr(self.model.config, 'n_ctx'):
            self.model.config.n_ctx = target_length
        self.model.config.use_cache = False
        
        logger.info(f"✓ Context extended to {target_length}")

class NeedleTestSuite:
    """Comprehensive needle-in-haystack testing suite."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
    
    def create_needle_test(self, needle_code: str, context_length: int, needle_position: str = "middle") -> Tuple[str, str]:
        """Create a needle-in-haystack test with specified parameters."""
        base_text = "The sky is blue today. Birds are singing in the trees. The weather is pleasant. "
        needle_text = f"The secret password is {needle_code}. "
        question = "What is the secret password?"
        
        # Calculate token requirements
        base_tokens = len(self.tokenizer.encode(base_text))
        needle_tokens = len(self.tokenizer.encode(needle_text))
        question_tokens = len(self.tokenizer.encode(question))
        
        # Calculate filler needed
        available_tokens = context_length - needle_tokens - question_tokens - 30  # Safety margin
        n_base_texts = max(1, available_tokens // base_tokens)
        
        # Position needle based on parameter
        if needle_position == "beginning":
            first_part = 1
            second_part = n_base_texts - 1
        elif needle_position == "end":
            first_part = n_base_texts - 1
            second_part = 1
        else:  # middle
            first_part = n_base_texts // 2
            second_part = n_base_texts - first_part
        
        # Construct context
        context_parts = (
            [base_text] * first_part +
            [needle_text] +
            [base_text] * second_part
        )
        context = "".join(context_parts)
        
        return context, question
    
    def test_needle_retrieval(self, context: str, question: str, expected_answer: str) -> Tuple[bool, str, Dict]:
        """Test needle retrieval with detailed diagnostics."""
        try:
            # Verify needle is in context
            if expected_answer not in context:
                return False, "needle_missing", {"error": "needle not in context"}
            
            # Create prompt with clear structure
            full_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Tokenize with attention to truncation
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.max_position_embeddings - 20,
                padding=False
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Verify needle survived tokenization
            decoded_input = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            if expected_answer not in decoded_input:
                return False, "needle_truncated", {"error": "needle lost in tokenization"}
            
            # Generate response
            start_time = time.time()
            with torch.no_grad():
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=15,
                    min_new_tokens=1,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=False
                )
            generation_time = time.time() - start_time
            
            # Extract and analyze answer
            answer_tokens = generated[0][input_ids.size(1):]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
            
            # Look for the expected code in the answer
            import re
            found_codes = re.findall(r'\b\d{4}\b', answer)
            success = expected_answer in found_codes
            
            # Calculate diagnostics
            context_tokens = len(self.tokenizer.encode(context))
            total_input_tokens = input_ids.size(1)
            
            diagnostics = {
                "success": success,
                "context_length_chars": len(context),
                "context_length_tokens": context_tokens,
                "total_input_tokens": total_input_tokens,
                "generated_tokens": len(answer_tokens),
                "generation_time": generation_time,
                "found_codes": found_codes,
                "answer": answer
            }
            
            return success, answer, diagnostics
            
        except Exception as e:
            error_msg = str(e)[:100]
            return False, f"Error: {error_msg}", {"error": error_msg}

class ComprehensiveExperiment:
    """Main experiment coordinator."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.results = {
            "config": config.__dict__,
            "start_time": time.time(),
            "baseline_results": {},
            "strength_scaling_results": {},
            "context_extension_results": {},
            "diagnostics": {},
            "summary": {}
        }
        
        # Create results directory
        Path(config.results_dir).mkdir(exist_ok=True)
        
        # Setup file logging
        log_file = Path(config.results_dir) / "experiment.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    def run_baseline_tests(self) -> Dict:
        """Run baseline tests without GSA."""
        logger.info("="*60)
        logger.info("📊 BASELINE TESTING")
        logger.info("="*60)
        
        test_suite = NeedleTestSuite(self.model_manager.model, self.model_manager.tokenizer)
        baseline_results = {}
        
        for length in self.config.baseline_lengths:
            logger.info(f"\nTesting baseline at {length} tokens...")
            
            length_results = []
            success_count = 0
            
            for i in range(self.config.tests_per_length):
                needle_code = f"{random.randint(1000, 9999)}"
                context, question = test_suite.create_needle_test(needle_code, length)
                
                success, answer, diagnostics = test_suite.test_needle_retrieval(
                    context, question, needle_code
                )
                
                length_results.append({
                    "test_id": i,
                    "needle_code": needle_code,
                    "success": success,
                    "answer": answer,
                    "diagnostics": diagnostics
                })
                
                if success:
                    success_count += 1
                
                status = "✓" if success else "✗"
                logger.info(f"  Test {i+1}: {status} (Expected: {needle_code}, Got: {answer})")
            
            success_rate = success_count / len(length_results)
            baseline_results[length] = {
                "success_rate": success_rate,
                "tests": length_results
            }
            
            logger.info(f"Baseline {length} tokens: {success_rate:.1%} success rate")
        
        self.results["baseline_results"] = baseline_results
        
        # Calculate overall baseline performance
        overall_success = np.mean([r["success_rate"] for r in baseline_results.values()])
        logger.info(f"\nOverall baseline success rate: {overall_success:.1%}")
        
        return baseline_results
    
    def run_strength_scaling_tests(self) -> Dict:
        """Test DirectionalGSA across different strength levels."""
        logger.info("="*60)
        logger.info("📈 GSA STRENGTH SCALING TESTS")
        logger.info("="*60)
        
        test_suite = NeedleTestSuite(self.model_manager.model, self.model_manager.tokenizer)
        scaling_results = {}
        
        for strength in self.config.strength_range:
            logger.info(f"\n--- Testing GSA Strength: {strength:.1%} ---")
            
            # Set GSA strength
            self.model_manager.gsa_layer.set_gsa_strength(strength)
            self.model_manager.gsa_layer.enable_gsa = True
            
            strength_results = {}
            
            # Test on subset of lengths for efficiency
            test_lengths = [1024, 2048]
            
            for length in test_lengths:
                logger.info(f"  Testing {length} tokens...")
                
                length_results = []
                success_count = 0
                
                for i in range(self.config.tests_per_length):
                    needle_code = f"{random.randint(1000, 9999)}"
                    context, question = test_suite.create_needle_test(needle_code, length)
                    
                    success, answer, diagnostics = test_suite.test_needle_retrieval(
                        context, question, needle_code
                    )
                    
                    length_results.append({
                        "needle_code": needle_code,
                        "success": success,
                        "answer": answer,
                        "diagnostics": diagnostics
                    })
                    
                    if success:
                        success_count += 1
                    
                    status = "✓" if success else "✗"
                    logger.info(f"    Test {i+1}: {status} (Expected: {needle_code}, Got: {answer})")
                
                success_rate = success_count / len(length_results)
                strength_results[length] = {
                    "success_rate": success_rate,
                    "tests": length_results
                }
                
                logger.info(f"    {length} tokens: {success_rate:.1%} success rate")
            
            # Calculate overall success rate for this strength
            overall_success = np.mean([r["success_rate"] for r in strength_results.values()])
            scaling_results[strength] = {
                "overall_success_rate": overall_success,
                "length_results": strength_results,
                "diagnostics": self.model_manager.gsa_layer.get_diagnostics()
            }
            
            logger.info(f"  Overall success rate: {overall_success:.1%}")
            
            # Stop if performance degrades significantly
            if overall_success < 0.5:
                logger.info(f"  Performance degraded at {strength:.1%} - stopping scaling")
                break
        
        self.results["strength_scaling_results"] = scaling_results
        return scaling_results
    
    def run_context_extension_tests(self, optimal_strength: float) -> Dict:
        """Test context extension capabilities."""
        logger.info("="*60)
        logger.info("🚀 CONTEXT EXTENSION TESTING")
        logger.info("="*60)
        
        # Extend model context
        self.model_manager.extend_context()
        
        # Set optimal GSA strength with bonus for longer contexts
        extended_strength = min(0.3, optimal_strength * 1.5)
        self.model_manager.gsa_layer.set_gsa_strength(extended_strength)
        self.model_manager.gsa_layer.enable_gsa = True
        
        logger.info(f"Using GSA strength: {extended_strength:.1%} for context extension")
        
        test_suite = NeedleTestSuite(self.model_manager.model, self.model_manager.tokenizer)
        extension_results = {}
        max_successful_length = 2048
        
        for length in self.config.extended_lengths:
            logger.info(f"\nTesting {length} tokens...")
            
            length_results = []
            success_count = 0
            
            for i in range(self.config.tests_per_length):
                try:
                    needle_code = f"{random.randint(1000, 9999)}"
                    context, question = test_suite.create_needle_test(needle_code, length)
                    
                    success, answer, diagnostics = test_suite.test_needle_retrieval(
                        context, question, needle_code
                    )
                    
                    length_results.append({
                        "needle_code": needle_code,
                        "success": success,
                        "answer": answer,
                        "diagnostics": diagnostics
                    })
                    
                    if success:
                        success_count += 1
                    
                    status = "✓" if success else "✗"
                    logger.info(f"  Test {i+1}: {status} (Expected: {needle_code}, Got: {answer})")
                    
                except Exception as e:
                    logger.info(f"  Test {i+1}: ✗ (Error: {str(e)[:50]})")
                    length_results.append({
                        "needle_code": f"{random.randint(1000, 9999)}",
                        "success": False,
                        "answer": f"Error: {str(e)[:50]}",
                        "diagnostics": {"error": str(e)}
                    })
            
            success_rate = success_count / len(length_results)
            extension_results[length] = {
                "success_rate": success_rate,
                "tests": length_results
            }
            
            logger.info(f"{length} tokens: {success_rate:.1%} success rate")
            
            # Update maximum successful length
            if success_rate >= 0.6:  # Consider 60%+ as successful
                max_successful_length = length
            elif success_rate < 0.3:  # Stop if performance is very poor
                logger.info(f"Stopping extension tests - performance too low at {length} tokens")
                break
        
        extension_results["max_successful_length"] = max_successful_length
        extension_results["extension_ratio"] = max_successful_length / 2048
        
        self.results["context_extension_results"] = extension_results
        
        logger.info(f"\nMaximum successful context length: {max_successful_length} tokens")
        logger.info(f"Context extension ratio: {extension_results['extension_ratio']:.2f}x")
        
        return extension_results
    
    def find_optimal_strength(self) -> float:
        """Find optimal GSA strength from scaling results."""
        scaling_results = self.results.get("strength_scaling_results", {})
        
        if not scaling_results:
            return self.config.initial_strength
        
        best_strength = self.config.initial_strength
        best_success_rate = 0.0
        
        for strength, results in scaling_results.items():
            success_rate = results["overall_success_rate"]
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_strength = strength
        
        logger.info(f"Optimal GSA strength: {best_strength:.1%} (Success rate: {best_success_rate:.1%})")
        return best_strength
    
    def generate_summary(self) -> Dict:
        """Generate comprehensive experiment summary."""
        logger.info("="*60)
        logger.info("📋 GENERATING SUMMARY")
        logger.info("="*60)
        
        # Calculate performance metrics
        baseline_results = self.results.get("baseline_results", {})
        scaling_results = self.results.get("strength_scaling_results", {})
        extension_results = self.results.get("context_extension_results", {})
        
        baseline_avg = np.mean([r["success_rate"] for r in baseline_results.values()]) if baseline_results else 0
        
        optimal_strength = self.find_optimal_strength()
        optimal_success_rate = 0
        if scaling_results and optimal_strength in scaling_results:
            optimal_success_rate = scaling_results[optimal_strength]["overall_success_rate"]
        
        max_context_length = extension_results.get("max_successful_length", 2048)
        extension_ratio = extension_results.get("extension_ratio", 1.0)
        
        # Determine overall verdict
        if extension_ratio >= 1.5 and optimal_success_rate >= 0.8:
            verdict = "BREAKTHROUGH"
            verdict_detail = "DirectionalGSA achieves significant context extension with maintained performance"
        elif extension_ratio >= 1.2 and optimal_success_rate >= 0.7:
            verdict = "SUCCESS"
            verdict_detail = "DirectionalGSA demonstrates clear context extension capabilities"
        elif optimal_success_rate >= 0.8:
            verdict = "STABLE"
            verdict_detail = "DirectionalGSA maintains baseline performance but limited extension"
        elif optimal_success_rate >= 0.5:
            verdict = "PROMISING"
            verdict_detail = "DirectionalGSA shows potential but needs optimization"
        else:
            verdict = "NEEDS_WORK"
            verdict_detail = "DirectionalGSA requires further development"
        
        # Collect final diagnostics
        final_diagnostics = {}
        if self.model_manager.gsa_layer:
            final_diagnostics = self.model_manager.gsa_layer.get_diagnostics()
        
        summary = {
            "verdict": verdict,
            "verdict_detail": verdict_detail,
            "baseline_performance": baseline_avg,
            "optimal_gsa_strength": optimal_strength,
            "optimal_success_rate": optimal_success_rate,
            "max_context_length": max_context_length,
            "context_extension_ratio": extension_ratio,
            "experiment_duration": time.time() - self.results["start_time"],
            "total_tests_run": self._count_total_tests(),
            "final_diagnostics": final_diagnostics,
            "memory_usage": get_memory_usage()
        }
        
        self.results["summary"] = summary
        
        # Log summary
        logger.info(f"Verdict: {verdict}")
        logger.info(f"Detail: {verdict_detail}")
        logger.info(f"Baseline performance: {baseline_avg:.1%}")
        logger.info(f"Optimal GSA strength: {optimal_strength:.1%}")
        logger.info(f"Best GSA performance: {optimal_success_rate:.1%}")
        logger.info(f"Max context length: {max_context_length} tokens")
        logger.info(f"Extension ratio: {extension_ratio:.2f}x")
        logger.info(f"Total tests: {summary['total_tests_run']}")
        logger.info(f"Duration: {summary['experiment_duration']:.1f} seconds")
        
        return summary
    
    def _count_total_tests(self) -> int:
        """Count total number of tests run."""
        total = 0
        
        # Baseline tests
        for results in self.results.get("baseline_results", {}).values():
            total += len(results.get("tests", []))
        
        # Strength scaling tests
        for strength_result in self.results.get("strength_scaling_results", {}).values():
            for length_result in strength_result.get("length_results", {}).values():
                total += len(length_result.get("tests", []))
        
        # Extension tests
        for results in self.results.get("context_extension_results", {}).values():
            if isinstance(results, dict) and "tests" in results:
                total += len(results["tests"])
        
        return total
    
    def save_results(self):
        """Save comprehensive results to file."""
        results_file = Path(self.config.results_dir) / "comprehensive_results.json"
        
        # Convert numpy types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_file}")
    
    def run_complete_experiment(self) -> Dict:
        """Run the complete comprehensive experiment."""
        logger.info("="*80)
        logger.info("🧭 COMPREHENSIVE DIRECTIONAL GSA EXPERIMENT")
        logger.info("="*80)
        logger.info("Complete evaluation of SplatNN-inspired directional information flow")
        logger.info(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Initial memory: {get_memory_usage()}")
        
        try:
            # Phase 1: Setup
            logger.info("\n🔧 Phase 1: Model Setup")
            self.model_manager.load_model()
            
            if not self.model_manager.install_directional_gsa():
                raise Exception("Failed to install DirectionalGSA")
            
            # Phase 2: Baseline Testing
            logger.info("\n📊 Phase 2: Baseline Testing")
            baseline_results = self.run_baseline_tests()
            
            # Check if baseline is acceptable
            baseline_avg = np.mean([r["success_rate"] for r in baseline_results.values()])
            if baseline_avg < 0.8:
                logger.warning(f"Baseline performance low ({baseline_avg:.1%}) - continuing anyway")
            
            # Phase 3: Strength Scaling
            logger.info("\n📈 Phase 3: GSA Strength Scaling")
            scaling_results = self.run_strength_scaling_tests()
            
            # Phase 4: Context Extension
            optimal_strength = self.find_optimal_strength()
            if optimal_strength > 0 and scaling_results:
                logger.info("\n🚀 Phase 4: Context Extension Testing")
                extension_results = self.run_context_extension_tests(optimal_strength)
            else:
                logger.warning("Skipping context extension - no suitable GSA strength found")
                self.results["context_extension_results"] = {"skipped": True}
            
            # Phase 5: Analysis and Summary
            logger.info("\n📋 Phase 5: Analysis and Summary")
            summary = self.generate_summary()
            
            # Save all results
            self.save_results()
            
            logger.info("\n" + "="*80)
            logger.info("🏁 EXPERIMENT COMPLETE")
            logger.info("="*80)
            logger.info(f"Duration: {summary['experiment_duration']:.1f} seconds")
            logger.info(f"Final memory: {get_memory_usage()}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Save partial results
            self.results["error"] = str(e)
            self.results["traceback"] = traceback.format_exc()
            self.save_results()
            
            return {"verdict": "FAILED", "error": str(e)}

def main():
    """Main function to run the comprehensive DirectionalGSA experiment."""
    
    # Configure experiment
    config = ExperimentConfig(
        model_name="EleutherAI/gpt-neo-125M",
        layer_to_replace=6,
        baseline_lengths=[512, 1024, 1536, 2048],
        strength_range=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
        extended_lengths=[2560, 3072, 3584, 4000],
        tests_per_length=3,
        target_context_length=4096,
        results_dir="comprehensive_gsa_results"
    )
    
    # Run comprehensive experiment
    experiment = ComprehensiveExperiment(config)
    results = experiment.run_complete_experiment()
    
    # Return appropriate exit code
    verdict = results.get("verdict", "FAILED")
    if verdict in ["BREAKTHROUGH", "SUCCESS"]:
        print(f"\n🎉 {verdict}: DirectionalGSA experiment successful!")
        return 0
    elif verdict in ["STABLE", "PROMISING"]:
        print(f"\n📈 {verdict}: DirectionalGSA shows positive results!")
        return 0
    else:
        print(f"\n⚠️ {verdict}: DirectionalGSA needs further work")
        return 1

if __name__ == "__main__":
    exit(main())
