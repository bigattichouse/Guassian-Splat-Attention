"""
SplatFlow Selective Processing Module
Attention-based selective processing for O(n*k) optimization.
Processes tokens with varying complexity based on attention importance and splat groupings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List, Any, Union
from collections import defaultdict, deque
from enum import Enum
import random

logger = logging.getLogger(__name__)


class ProcessingTier(Enum):
    """Processing complexity tiers based on attention importance"""
    MINIMAL = "minimal"       # <10% attention - basic processing
    STANDARD = "standard"     # 10-40% attention - normal processing  
    ENHANCED = "enhanced"     # 40-70% attention - enriched processing
    PREMIUM = "premium"       # >70% attention - full complexity processing


class SelectiveTokenProcessor(nn.Module):
    """
    Process tokens differently based on attention scores and splat groupings.
    Applies varying computational complexity based on token importance.
    """
    
    def __init__(self, model_dim: int, num_splats: int = 20, 
                 processing_config: Optional[Dict] = None):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_splats = num_splats
        
        # Parse processing configuration
        self.config = processing_config or {}
        self.attention_thresholds = self.config.get('attention_thresholds', {
            'minimal': 0.1,
            'standard': 0.4, 
            'enhanced': 0.7,
            'premium': float('inf')
        })
        
        # Processing networks for each tier
        self.processors = self._create_tiered_processors()
        
        # Attention importance analyzer
        self.importance_analyzer = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, model_dim // 4),
            nn.GELU(),
            nn.Linear(model_dim // 4, 4),  # 4 processing tiers
            nn.Softmax(dim=-1)
        )
        
        # Splat-aware token router
        self.splat_router = nn.ModuleDict({
            f'splat_{i}': nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.GELU(),
                nn.Linear(model_dim // 2, 4),  # Route to processing tier
                nn.Softmax(dim=-1)
            ) for i in range(min(num_splats, 8))  # Limit for efficiency
        })
        
        # Dynamic complexity controller
        self.complexity_controller = nn.Sequential(
            nn.Linear(model_dim + 4, model_dim // 2),  # token + tier probabilities
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Processing statistics
        self.processing_stats = {
            'tokens_by_tier': defaultdict(int),
            'total_tokens_processed': 0,
            'splat_routing_efficiency': defaultdict(float),
            'dynamic_adjustments': 0,
            'processing_time_by_tier': defaultdict(float)
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds = nn.ParameterDict({
            tier: nn.Parameter(torch.tensor(threshold))
            for tier, threshold in self.attention_thresholds.items()
            if tier != 'premium'
        })
        
        logger.info(f"🎯 Selective token processor initialized")
        logger.info(f"   Processing tiers: {list(self.attention_thresholds.keys())}")
        logger.info(f"   Attention thresholds: {self.attention_thresholds}")
    
    def _create_tiered_processors(self) -> nn.ModuleDict:
        """Create processing networks for different complexity tiers"""
        
        processors = nn.ModuleDict()
        
        # MINIMAL: Very basic processing
        processors['minimal'] = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.ReLU(),  # Fast activation
            nn.Linear(self.model_dim // 2, self.model_dim)
        )
        
        # STANDARD: Normal transformer-like processing
        processors['standard'] = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.model_dim, self.model_dim)
        )
        
        # ENHANCED: Richer processing with multiple paths
        processors['enhanced'] = EnhancedTokenProcessor(self.model_dim)
        
        # PREMIUM: Full complexity processing
        processors['premium'] = PremiumTokenProcessor(self.model_dim)
        
        return processors
    
    def forward(self, token_embeddings: torch.Tensor,
                attention_weights: Optional[torch.Tensor] = None,
                splat_groups: Optional[Dict] = None,
                return_processing_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Selective processing based on attention importance and splat groupings
        
        Args:
            token_embeddings: [batch_size, seq_len, model_dim]
            attention_weights: [batch_size, seq_len, num_splats] optional attention weights
            splat_groups: Optional splat grouping information
            return_processing_info: Return detailed processing information
            
        Returns:
            Processed embeddings, optionally with processing info
        """
        
        batch_size, seq_len, model_dim = token_embeddings.shape
        device = token_embeddings.device
        
        processing_info = {
            'tier_assignments': {},
            'routing_decisions': {},
            'complexity_adjustments': {},
            'processing_efficiency': {}
        }
        
        try:
            # Determine processing tiers for each token
            tier_assignments = self._assign_processing_tiers(
                token_embeddings, attention_weights, splat_groups
            )
            
            if return_processing_info:
                processing_info['tier_assignments'] = tier_assignments
            
            # Apply selective processing
            processed_tokens = self._apply_tiered_processing(
                token_embeddings, tier_assignments, processing_info
            )
            
            # Apply dynamic complexity adjustments
            if self.config.get('enable_dynamic_adjustment', True):
                processed_tokens = self._apply_dynamic_adjustments(
                    processed_tokens, tier_assignments, processing_info
                )
            
            # Update statistics
            self._update_processing_statistics(tier_assignments, batch_size, seq_len)
            
            if return_processing_info:
                return processed_tokens, processing_info
            else:
                return processed_tokens
                
        except Exception as e:
            logger.warning(f"Selective processing failed: {e}")
            # Fallback to standard processing
            fallback_output = self.processors['standard'](token_embeddings)
            
            if return_processing_info:
                processing_info['error'] = str(e)
                processing_info['fallback_used'] = True
                return fallback_output, processing_info
            else:
                return fallback_output
    
    def _assign_processing_tiers(self, token_embeddings: torch.Tensor,
                                attention_weights: Optional[torch.Tensor],
                                splat_groups: Optional[Dict]) -> torch.Tensor:
        """Assign processing tier to each token based on importance"""
        
        batch_size, seq_len, model_dim = token_embeddings.shape
        device = token_embeddings.device
        
        try:
            # Method 1: Use attention weights if available
            if attention_weights is not None:
                # Calculate total attention per token
                token_attention = attention_weights.sum(dim=-1)  # [batch, seq]
                
                # Normalize attention scores
                max_attention = token_attention.max(dim=-1, keepdim=True)[0]
                normalized_attention = token_attention / (max_attention + 1e-8)
                
                # Assign tiers based on thresholds
                tier_assignments = torch.zeros_like(normalized_attention, dtype=torch.long)
                
                # Get adaptive thresholds
                thresh_minimal = torch.sigmoid(self.adaptive_thresholds['minimal'])
                thresh_standard = torch.sigmoid(self.adaptive_thresholds['standard']) 
                thresh_enhanced = torch.sigmoid(self.adaptive_thresholds['enhanced'])
                
                tier_assignments[normalized_attention >= thresh_enhanced] = 3  # premium
                tier_assignments[(normalized_attention >= thresh_standard) & 
                               (normalized_attention < thresh_enhanced)] = 2  # enhanced
                tier_assignments[(normalized_attention >= thresh_minimal) & 
                               (normalized_attention < thresh_standard)] = 1  # standard
                # Rest remain 0 (minimal)
                
            else:
                # Method 2: Use learned importance analyzer
                importance_probs = self.importance_analyzer(token_embeddings)
                tier_assignments = torch.argmax(importance_probs, dim=-1)
            
            # Method 3: Apply splat-based routing if available
            if splat_groups is not None:
                tier_assignments = self._apply_splat_routing(
                    token_embeddings, tier_assignments, splat_groups
                )
            
            return tier_assignments
            
        except Exception as e:
            logger.warning(f"Tier assignment failed: {e}")
            # Default to standard processing for all tokens
            return torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
    
    def _apply_splat_routing(self, token_embeddings: torch.Tensor,
                           base_assignments: torch.Tensor,
                           splat_groups: Dict) -> torch.Tensor:
        """Apply splat-based routing adjustments"""
        
        try:
            batch_size, seq_len, model_dim = token_embeddings.shape
            adjusted_assignments = base_assignments.clone()
            
            # Apply splat-specific routing for each splat group
            for splat_id, token_indices in splat_groups.items():
                if isinstance(token_indices, (list, tuple)) and len(token_indices) > 0:
                    # Get splat router if available
                    router_key = f'splat_{min(splat_id, 7)}'  # Limit to available routers
                    
                    if router_key in self.splat_router:
                        # Get tokens for this splat
                        splat_tokens = token_embeddings.view(-1, model_dim)[token_indices]
                        
                        if len(splat_tokens) > 0:
                            # Get routing probabilities
                            routing_probs = self.splat_router[router_key](splat_tokens)
                            splat_tier_assignments = torch.argmax(routing_probs, dim=-1)
                            
                            # Apply routing decisions (blend with base assignments)
                            for i, global_idx in enumerate(token_indices):
                                if i < len(splat_tier_assignments):
                                    batch_idx = global_idx // seq_len
                                    token_idx = global_idx % seq_len
                                    
                                    if batch_idx < batch_size and token_idx < seq_len:
                                        # Blend splat routing with base assignment
                                        base_tier = adjusted_assignments[batch_idx, token_idx]
                                        splat_tier = splat_tier_assignments[i]
                                        
                                        # Take higher tier (more processing)
                                        adjusted_assignments[batch_idx, token_idx] = max(base_tier, splat_tier)
            
            return adjusted_assignments
            
        except Exception as e:
            logger.warning(f"Splat routing failed: {e}")
            return base_assignments
    
    def _apply_tiered_processing(self, token_embeddings: torch.Tensor,
                               tier_assignments: torch.Tensor,
                               processing_info: Dict) -> torch.Tensor:
        """Apply processing based on tier assignments"""
        
        batch_size, seq_len, model_dim = token_embeddings.shape
        device = token_embeddings.device
        
        # Initialize output
        processed_tokens = torch.zeros_like(token_embeddings)
        
        # Process each tier separately for efficiency
        tier_names = ['minimal', 'standard', 'enhanced', 'premium']
        
        for tier_idx, tier_name in enumerate(tier_names):
            # Find tokens assigned to this tier
            tier_mask = tier_assignments == tier_idx
            
            if tier_mask.any():
                tier_tokens = token_embeddings[tier_mask]
                
                if len(tier_tokens) > 0:
                    # Process tokens through appropriate processor
                    try:
                        processed_tier_tokens = self.processors[tier_name](tier_tokens)
                        processed_tokens[tier_mask] = processed_tier_tokens
                        
                        # Record processing info
                        processing_info['routing_decisions'][tier_name] = {
                            'token_count': len(tier_tokens),
                            'percentage': (len(tier_tokens) / (batch_size * seq_len)) * 100
                        }
                        
                    except Exception as e:
                        logger.warning(f"Processing failed for tier {tier_name}: {e}")
                        # Fallback to standard processing
                        fallback_tokens = self.processors['standard'](tier_tokens)
                        processed_tokens[tier_mask] = fallback_tokens
        
        return processed_tokens
    
    def _apply_dynamic_adjustments(self, processed_tokens: torch.Tensor,
                                 tier_assignments: torch.Tensor,
                                 processing_info: Dict) -> torch.Tensor:
        """Apply dynamic complexity adjustments based on processing results"""
        
        try:
            batch_size, seq_len, model_dim = processed_tokens.shape
            
            # Create tier probability encoding
            tier_probs = F.one_hot(tier_assignments, num_classes=4).float()
            
            # Combine token features with tier information
            combined_features = torch.cat([
                processed_tokens, 
                tier_probs
            ], dim=-1)
            
            # Calculate complexity adjustment factors
            adjustment_factors = self.complexity_controller(combined_features)
            
            # Apply adjustments
            adjusted_tokens = processed_tokens * (1.0 + 0.2 * adjustment_factors)
            
            # Record statistics
            avg_adjustment = adjustment_factors.mean().item()
            processing_info['complexity_adjustments'] = {
                'avg_adjustment_factor': avg_adjustment,
                'adjustment_range': {
                    'min': adjustment_factors.min().item(),
                    'max': adjustment_factors.max().item()
                }
            }
            
            self.processing_stats['dynamic_adjustments'] += 1
            
            return adjusted_tokens
            
        except Exception as e:
            logger.warning(f"Dynamic adjustment failed: {e}")
            return processed_tokens
    
    def _update_processing_statistics(self, tier_assignments: torch.Tensor,
                                    batch_size: int, seq_len: int):
        """Update processing statistics"""
        
        try:
            # Count tokens by tier
            tier_names = ['minimal', 'standard', 'enhanced', 'premium']
            for tier_idx, tier_name in enumerate(tier_names):
                count = (tier_assignments == tier_idx).sum().item()
                self.processing_stats['tokens_by_tier'][tier_name] += count
            
            self.processing_stats['total_tokens_processed'] += batch_size * seq_len
            
        except Exception as e:
            logger.warning(f"Failed to update processing statistics: {e}")
    
    def get_processing_statistics(self) -> Dict:
        """Get comprehensive processing statistics"""
        
        try:
            total_tokens = self.processing_stats['total_tokens_processed']
            
            if total_tokens > 0:
                tier_distribution = {}
                for tier, count in self.processing_stats['tokens_by_tier'].items():
                    tier_distribution[tier] = {
                        'count': count,
                        'percentage': (count / total_tokens) * 100
                    }
            else:
                tier_distribution = {}
            
            # Calculate efficiency metrics
            processing_efficiency = {}
            if tier_distribution:
                # Higher tier usage indicates more selective processing
                enhanced_premium_ratio = (
                    tier_distribution.get('enhanced', {}).get('percentage', 0) +
                    tier_distribution.get('premium', {}).get('percentage', 0)
                ) / 100
                
                processing_efficiency = {
                    'selectivity_ratio': enhanced_premium_ratio,
                    'avg_processing_tier': sum(
                        i * tier_distribution.get(tier, {}).get('count', 0) 
                        for i, tier in enumerate(['minimal', 'standard', 'enhanced', 'premium'])
                    ) / max(total_tokens, 1),
                    'processing_diversity': len([
                        tier for tier, data in tier_distribution.items() 
                        if data.get('percentage', 0) > 5
                    ])
                }
            
            return {
                'total_tokens_processed': total_tokens,
                'tier_distribution': tier_distribution,
                'processing_efficiency': processing_efficiency,
                'dynamic_adjustments': self.processing_stats['dynamic_adjustments'],
                'adaptive_thresholds': {
                    name: torch.sigmoid(param).item() 
                    for name, param in self.adaptive_thresholds.items()
                },
                'processor_type': 'SelectiveTokenProcessor'
            }
            
        except Exception as e:
            logger.warning(f"Failed to get processing statistics: {e}")
            return {
                'error': str(e),
                'processor_type': 'SelectiveTokenProcessor'
            }


class EnhancedTokenProcessor(nn.Module):
    """Enhanced processing tier with multiple pathways"""
    
    def __init__(self, model_dim: int):
        super().__init__()
        
        self.model_dim = model_dim
        
        # Multi-path processing
        self.path_a = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
        
        self.path_b = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, model_dim)
        )
        
        # Path combiner
        self.path_combiner = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Process through both paths
        out_a = self.path_a(x)
        out_b = self.path_b(x)
        
        # Combine paths
        combined = torch.cat([out_a, out_b], dim=-1)
        output = self.path_combiner(combined)
        
        return output


class PremiumTokenProcessor(nn.Module):
    """Premium processing tier with maximum complexity"""
    
    def __init__(self, model_dim: int):
        super().__init__()
        
        self.model_dim = model_dim
        
        # Multi-layer processing with residual connections
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, model_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(model_dim * 2, model_dim)
            ) for _ in range(3)
        ])
        
        # Attention-like self-enhancement
        self.self_attention = nn.MultiheadAttention(
            model_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply multi-layer processing with residuals
        for layer in self.layers:
            x = x + layer(x)
        
        # Self-attention enhancement
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            attn_out, _ = self.self_attention(x, x, x)
            x = attn_out.squeeze(0)
        else:
            attn_out, _ = self.self_attention(x, x, x)
            x = attn_out
        
        # Final processing
        output = self.final_proj(x)
        
        return output


class SparseOutputProjection(nn.Module):
    """
    O(n*k) sparse output projection that focuses computational resources
    on high-attention tokens while using lightweight processing for others.
    """
    
    def __init__(self, model_dim: int, vocab_size: int, 
                 sparsity_config: Optional[Dict] = None):
        super().__init__()
        
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        
        # Parse sparsity configuration
        self.config = sparsity_config or {}
        self.high_attention_threshold = self.config.get('high_attention_threshold', 0.7)
        self.medium_attention_threshold = self.config.get('medium_attention_threshold', 0.3)
        self.sparse_ratio = self.config.get('sparse_ratio', 0.8)
        
        # Multi-tier output projections
        self.full_projection = nn.Linear(model_dim, vocab_size, bias=False)
        
        self.sparse_projection = nn.Sequential(
            nn.Linear(model_dim, vocab_size // 4),
            nn.GELU(),
            nn.Linear(vocab_size // 4, vocab_size)
        )
        
        self.minimal_projection = nn.Linear(model_dim, vocab_size // 8)
        self.minimal_expander = nn.Linear(vocab_size // 8, vocab_size)
        
        # Attention-based gate
        self.attention_gate = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 3),  # 3 projection tiers
            nn.Softmax(dim=-1)
        )
        
        # Adaptive vocabulary focusing
        self.vocab_focuser = nn.Sequential(
            nn.Linear(model_dim, vocab_size // 10),
            nn.GELU(),
            nn.Linear(vocab_size // 10, vocab_size),
            nn.Sigmoid()
        )
        
        # Statistics
        self.projection_stats = {
            'full_projections': 0,
            'sparse_projections': 0,
            'minimal_projections': 0,
            'total_tokens': 0,
            'computation_savings': 0.0
        }
        
        logger.info(f"🎯 Sparse output projection initialized")
        logger.info(f"   Vocab size: {vocab_size}, Model dim: {model_dim}")
        logger.info(f"   Sparsity ratio: {self.sparse_ratio}")
    
    def forward(self, hidden_states: torch.Tensor,
                attention_weights: Optional[torch.Tensor] = None,
                return_projection_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Sparse output projection with attention-guided processing
        
        Args:
            hidden_states: [batch_size, seq_len, model_dim]
            attention_weights: [batch_size, seq_len, num_splats] optional attention weights
            return_projection_info: Return detailed projection information
            
        Returns:
            Logits [batch_size, seq_len, vocab_size], optionally with projection info
        """
        
        batch_size, seq_len, model_dim = hidden_states.shape
        device = hidden_states.device
        
        projection_info = {
            'projection_tiers': {},
            'computation_savings': 0.0,
            'vocab_focusing': {}
        }
        
        try:
            # Determine projection strategy for each token
            if attention_weights is not None:
                projection_strategy = self._determine_projection_strategy_from_attention(
                    hidden_states, attention_weights
                )
            else:
                projection_strategy = self._determine_projection_strategy_from_content(
                    hidden_states
                )
            
            # Apply sparse projection
            logits = self._apply_sparse_projection(
                hidden_states, projection_strategy, projection_info
            )
            
            # Apply vocabulary focusing if enabled
            if self.config.get('enable_vocab_focusing', True):
                logits = self._apply_vocabulary_focusing(
                    logits, hidden_states, projection_info
                )
            
            # Update statistics
            self._update_projection_statistics(projection_strategy, batch_size, seq_len)
            
            if return_projection_info:
                return logits, projection_info
            else:
                return logits
                
        except Exception as e:
            logger.warning(f"Sparse projection failed: {e}")
            # Fallback to full projection
            fallback_logits = self.full_projection(hidden_states)
            
            if return_projection_info:
                projection_info['error'] = str(e)
                projection_info['fallback_used'] = True
                return fallback_logits, projection_info
            else:
                return fallback_logits
    
    def _determine_projection_strategy_from_attention(self, hidden_states: torch.Tensor,
                                                    attention_weights: torch.Tensor) -> torch.Tensor:
        """Determine projection strategy based on attention weights"""
        
        try:
            # Calculate total attention per token
            token_attention = attention_weights.sum(dim=-1)  # [batch, seq]
            
            # Normalize attention scores
            max_attention = token_attention.max(dim=-1, keepdim=True)[0]
            normalized_attention = token_attention / (max_attention + 1e-8)
            
            # Assign projection strategies
            strategy = torch.zeros_like(normalized_attention, dtype=torch.long)
            
            strategy[normalized_attention >= self.high_attention_threshold] = 2  # full
            strategy[(normalized_attention >= self.medium_attention_threshold) & 
                    (normalized_attention < self.high_attention_threshold)] = 1  # sparse
            # Rest remain 0 (minimal)
            
            return strategy
            
        except Exception as e:
            logger.warning(f"Attention-based strategy determination failed: {e}")
            # Default to sparse for all tokens
            batch_size, seq_len = hidden_states.shape[:2]
            return torch.ones(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
    
    def _determine_projection_strategy_from_content(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Determine projection strategy based on content analysis"""
        
        try:
            # Use attention gate to determine projection strategy
            gate_probs = self.attention_gate(hidden_states)  # [batch, seq, 3]
            strategy = torch.argmax(gate_probs, dim=-1)  # [batch, seq]
            
            return strategy
            
        except Exception as e:
            logger.warning(f"Content-based strategy determination failed: {e}")
            # Default to sparse for all tokens
            batch_size, seq_len = hidden_states.shape[:2]
            return torch.ones(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)
    
    def _apply_sparse_projection(self, hidden_states: torch.Tensor,
                               projection_strategy: torch.Tensor,
                               projection_info: Dict) -> torch.Tensor:
        """Apply sparse projection based on strategy"""
        
        batch_size, seq_len, model_dim = hidden_states.shape
        device = hidden_states.device
        
        # Initialize output logits
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        
        # Process each projection tier
        for strategy_id, proj_name in enumerate(['minimal', 'sparse', 'full']):
            strategy_mask = projection_strategy == strategy_id
            
            if strategy_mask.any():
                strategy_tokens = hidden_states[strategy_mask]
                
                if len(strategy_tokens) > 0:
                    if strategy_id == 2:  # Full projection
                        strategy_logits = self.full_projection(strategy_tokens)
                        self.projection_stats['full_projections'] += len(strategy_tokens)
                        
                    elif strategy_id == 1:  # Sparse projection
                        strategy_logits = self.sparse_projection(strategy_tokens)
                        self.projection_stats['sparse_projections'] += len(strategy_tokens)
                        
                    else:  # Minimal projection
                        minimal_features = self.minimal_projection(strategy_tokens)
                        strategy_logits = self.minimal_expander(minimal_features)
                        self.projection_stats['minimal_projections'] += len(strategy_tokens)
                    
                    logits[strategy_mask] = strategy_logits
                    
                    # Record projection info
                    projection_info['projection_tiers'][proj_name] = {
                        'token_count': len(strategy_tokens),
                        'percentage': (len(strategy_tokens) / (batch_size * seq_len)) * 100
                    }
        
        # Calculate computation savings
        full_ops = batch_size * seq_len * model_dim * self.vocab_size
        actual_ops = (
            self.projection_stats['full_projections'] * model_dim * self.vocab_size +
            self.projection_stats['sparse_projections'] * model_dim * (self.vocab_size // 4 + self.vocab_size) +
            self.projection_stats['minimal_projections'] * model_dim * (self.vocab_size // 8 + self.vocab_size)
        )
        
        savings = max(0.0, 1.0 - (actual_ops / max(full_ops, 1)))
        projection_info['computation_savings'] = savings
        self.projection_stats['computation_savings'] = savings
        
        return logits
    
    def _apply_vocabulary_focusing(self, logits: torch.Tensor,
                                 hidden_states: torch.Tensor,
                                 projection_info: Dict) -> torch.Tensor:
        """Apply vocabulary focusing to concentrate probability mass"""
        
        try:
            # Generate vocabulary focus mask
            vocab_focus = self.vocab_focuser(hidden_states)  # [batch, seq, vocab_size]
            
            # Apply focusing (enhances likely tokens, suppresses unlikely ones)
            focused_logits = logits * (1.0 + vocab_focus)
            
            # Calculate focusing statistics
            focus_strength = vocab_focus.mean().item()
            focus_sparsity = (vocab_focus < 0.1).float().mean().item()
            
            projection_info['vocab_focusing'] = {
                'focus_strength': focus_strength,
                'focus_sparsity': focus_sparsity,
                'enabled': True
            }
            
            return focused_logits
            
        except Exception as e:
            logger.warning(f"Vocabulary focusing failed: {e}")
            projection_info['vocab_focusing'] = {'enabled': False, 'error': str(e)}
            return logits
    
    def _update_projection_statistics(self, projection_strategy: torch.Tensor,
                                    batch_size: int, seq_len: int):
        """Update projection statistics"""
        
        try:
            self.projection_stats['total_tokens'] += batch_size * seq_len
            
        except Exception as e:
            logger.warning(f"Failed to update projection statistics: {e}")
    
    def get_projection_statistics(self) -> Dict:
        """Get comprehensive projection statistics"""
        
        try:
            total_tokens = self.projection_stats['total_tokens']
            
            if total_tokens > 0:
                tier_distribution = {
                    'full': {
                        'count': self.projection_stats['full_projections'],
                        'percentage': (self.projection_stats['full_projections'] / total_tokens) * 100
                    },
                    'sparse': {
                        'count': self.projection_stats['sparse_projections'],
                        'percentage': (self.projection_stats['sparse_projections'] / total_tokens) * 100
                    },
                    'minimal': {
                        'count': self.projection_stats['minimal_projections'],
                        'percentage': (self.projection_stats['minimal_projections'] / total_tokens) * 100
                    }
                }
            else:
                tier_distribution = {}
            
            return {
                'total_tokens': total_tokens,
                'projection_distribution': tier_distribution,
                'computation_savings': self.projection_stats['computation_savings'],
                'vocab_size': self.vocab_size,
                'model_dim': self.model_dim,
                'sparsity_ratio': self.sparse_ratio,
                'projection_type': 'SparseOutputProjection'
            }
            
        except Exception as e:
            logger.warning(f"Failed to get projection statistics: {e}")
            return {
                'error': str(e),
                'projection_type': 'SparseOutputProjection'
            }


class HierarchicalLayerNorm(nn.Module):
    """
    Splat-grouped layer normalization that applies different normalization
    strategies based on token groupings and attention patterns.
    """
    
    def __init__(self, model_dim: int, num_splats: int = 20,
                 norm_config: Optional[Dict] = None):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_splats = num_splats
        
        # Parse normalization configuration
        self.config = norm_config or {}
        self.enable_splat_grouping = self.config.get('enable_splat_grouping', True)
        self.enable_adaptive_epsilon = self.config.get('enable_adaptive_epsilon', True)
        
        # Standard layer normalization as base
        self.base_norm = nn.LayerNorm(model_dim, eps=1e-6)
        
        # Splat-specific normalization parameters
        if self.enable_splat_grouping:
            self.splat_norms = nn.ModuleList([
                nn.LayerNorm(model_dim, eps=1e-6) for _ in range(min(num_splats, 8))
            ])
        
        # Adaptive epsilon controller
        if self.enable_adaptive_epsilon:
            self.epsilon_controller = nn.Sequential(
                nn.Linear(model_dim, model_dim // 4),
                nn.GELU(),
                nn.Linear(model_dim // 4, 1),
                nn.Sigmoid()
            )
        
        # Attention-based norm selector
        self.norm_selector = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 2),  # standard vs adaptive
            nn.Softmax(dim=-1)
        )
        
        # Statistics
        self.norm_stats = {
            'standard_norm_count': 0,
            'splat_norm_count': defaultdict(int),
            'adaptive_epsilon_count': 0,
            'total_normalizations': 0
        }
        
        logger.info(f"📐 Hierarchical layer norm initialized")
        logger.info(f"   Model dim: {model_dim}, Num splats: {num_splats}")
        logger.info(f"   Splat grouping: {self.enable_splat_grouping}")
        logger.info(f"   Adaptive epsilon: {self.enable_adaptive_epsilon}")
    
    def forward(self, x: torch.Tensor,
                attention_weights: Optional[torch.Tensor] = None,
                splat_groups: Optional[Dict] = None,
                return_norm_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Hierarchical layer normalization
        
        Args:
            x: Input tensor [batch_size, seq_len, model_dim]
            attention_weights: Optional attention weights [batch_size, seq_len, num_splats]
            splat_groups: Optional splat grouping information
            return_norm_info: Return detailed normalization information
            
        Returns:
            Normalized tensor, optionally with normalization info
        """
        
        batch_size, seq_len, model_dim = x.shape
        device = x.device
        
        norm_info = {
            'normalization_strategy': {},
            'epsilon_adjustments': {},
            'splat_group_norms': {}
        }
        
        try:
            # Determine normalization strategy
            if self.enable_splat_grouping and splat_groups is not None:
                normalized_x = self._apply_splat_grouped_normalization(
                    x, splat_groups, norm_info
                )
            elif self.enable_adaptive_epsilon:
                normalized_x = self._apply_adaptive_normalization(
                    x, attention_weights, norm_info
                )
            else:
                normalized_x = self._apply_standard_normalization(x, norm_info)
            
            # Update statistics
            self._update_norm_statistics(norm_info, batch_size, seq_len)
            
            if return_norm_info:
                return normalized_x, norm_info
            else:
                return normalized_x
                
        except Exception as e:
            logger.warning(f"Hierarchical normalization failed: {e}")
            # Fallback to standard normalization
            fallback_x = self.base_norm(x)
            
            if return_norm_info:
                norm_info['error'] = str(e)
                norm_info['fallback_used'] = True
                return fallback_x, norm_info
            else:
                return fallback_x
    
    def _apply_splat_grouped_normalization(self, x: torch.Tensor,
                                         splat_groups: Dict,
                                         norm_info: Dict) -> torch.Tensor:
        """Apply normalization grouped by splat assignments"""
        
        try:
            batch_size, seq_len, model_dim = x.shape
            normalized_x = torch.zeros_like(x)
            
            # Apply splat-specific normalization
            tokens_processed = set()
            
            for splat_id, token_indices in splat_groups.items():
                if isinstance(token_indices, (list, tuple)) and len(token_indices) > 0:
                    # Get corresponding norm layer
                    norm_idx = min(splat_id, len(self.splat_norms) - 1)
                    splat_norm = self.splat_norms[norm_idx]
                    
                    # Extract tokens for this splat
                    splat_tokens = []
                    valid_positions = []
                    
                    for global_idx in token_indices:
                        batch_idx = global_idx // seq_len
                        token_idx = global_idx % seq_len
                        
                        if (batch_idx < batch_size and token_idx < seq_len and 
                            global_idx not in tokens_processed):
                            splat_tokens.append(x[batch_idx, token_idx])
                            valid_positions.append((batch_idx, token_idx))
                            tokens_processed.add(global_idx)
                    
                    if splat_tokens:
                        # Normalize splat tokens together
                        splat_tensor = torch.stack(splat_tokens)
                        normalized_splat_tokens = splat_norm(splat_tensor)
                        
                        # Place back in output tensor
                        for i, (batch_idx, token_idx) in enumerate(valid_positions):
                            normalized_x[batch_idx, token_idx] = normalized_splat_tokens[i]
                        
                        # Record statistics
                        self.norm_stats['splat_norm_count'][splat_id] += len(splat_tokens)
                        
                        norm_info['splat_group_norms'][f'splat_{splat_id}'] = {
                            'tokens_normalized': len(splat_tokens),
                            'norm_layer': norm_idx
                        }
            
            # Handle any remaining tokens with standard normalization
            remaining_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
            for global_idx in tokens_processed:
                batch_idx = global_idx // seq_len
                token_idx = global_idx % seq_len
                if batch_idx < batch_size and token_idx < seq_len:
                    remaining_mask[batch_idx, token_idx] = False
            
            if remaining_mask.any():
                remaining_tokens = x[remaining_mask]
                if len(remaining_tokens) > 0:
                    normalized_remaining = self.base_norm(remaining_tokens)
                    normalized_x[remaining_mask] = normalized_remaining
                    
                    self.norm_stats['standard_norm_count'] += len(remaining_tokens)
            
            norm_info['normalization_strategy']['type'] = 'splat_grouped'
            norm_info['normalization_strategy']['groups_processed'] = len(splat_groups)
            
            return normalized_x
            
        except Exception as e:
            logger.warning(f"Splat-grouped normalization failed: {e}")
            return self.base_norm(x)
    
    def _apply_adaptive_normalization(self, x: torch.Tensor,
                                    attention_weights: Optional[torch.Tensor],
                                    norm_info: Dict) -> torch.Tensor:
        """Apply normalization with adaptive epsilon"""
        
        try:
            # Calculate adaptive epsilon values
            epsilon_adjustments = self.epsilon_controller(x)  # [batch, seq, 1]
            
            # Scale epsilon based on attention if available
            if attention_weights is not None:
                attention_magnitude = attention_weights.sum(dim=-1, keepdim=True)
                attention_scale = attention_magnitude / (attention_magnitude.max() + 1e-8)
                epsilon_adjustments = epsilon_adjustments * (1.0 + attention_scale)
            
            # Apply normalization with adaptive epsilon
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            
            # Adaptive epsilon: higher for high-attention tokens
            base_eps = 1e-6
            adaptive_eps = base_eps * (1.0 + epsilon_adjustments.squeeze(-1).unsqueeze(-1))
            
            normalized_x = (x - mean) / torch.sqrt(var + adaptive_eps)
            
            # Apply learnable parameters from base norm
            normalized_x = normalized_x * self.base_norm.weight + self.base_norm.bias
            
            # Record statistics
            avg_epsilon_adj = epsilon_adjustments.mean().item()
            norm_info['epsilon_adjustments'] = {
                'avg_adjustment': avg_epsilon_adj,
                'adjustment_range': {
                    'min': epsilon_adjustments.min().item(),
                    'max': epsilon_adjustments.max().item()
                }
            }
            
            norm_info['normalization_strategy']['type'] = 'adaptive_epsilon'
            self.norm_stats['adaptive_epsilon_count'] += x.numel()
            
            return normalized_x
            
        except Exception as e:
            logger.warning(f"Adaptive normalization failed: {e}")
            return self.base_norm(x)
    
    def _apply_standard_normalization(self, x: torch.Tensor, norm_info: Dict) -> torch.Tensor:
        """Apply standard layer normalization"""
        
        try:
            normalized_x = self.base_norm(x)
            
            norm_info['normalization_strategy']['type'] = 'standard'
            self.norm_stats['standard_norm_count'] += x.numel()
            
            return normalized_x
            
        except Exception as e:
            logger.warning(f"Standard normalization failed: {e}")
            return x
    
    def _update_norm_statistics(self, norm_info: Dict, batch_size: int, seq_len: int):
        """Update normalization statistics"""
        
        try:
            self.norm_stats['total_normalizations'] += batch_size * seq_len
            
        except Exception as e:
            logger.warning(f"Failed to update norm statistics: {e}")
    
    def get_normalization_statistics(self) -> Dict:
        """Get comprehensive normalization statistics"""
        
        try:
            total_norms = self.norm_stats['total_normalizations']
            
            return {
                'total_normalizations': total_norms,
                'standard_norm_count': self.norm_stats['standard_norm_count'],
                'adaptive_epsilon_count': self.norm_stats['adaptive_epsilon_count'],
                'splat_norm_counts': dict(self.norm_stats['splat_norm_count']),
                'normalization_efficiency': {
                    'standard_ratio': self.norm_stats['standard_norm_count'] / max(total_norms, 1),
                    'adaptive_ratio': self.norm_stats['adaptive_epsilon_count'] / max(total_norms, 1),
                    'splat_grouping_ratio': sum(self.norm_stats['splat_norm_count'].values()) / max(total_norms, 1)
                },
                'model_dim': self.model_dim,
                'num_splats': self.num_splats,
                'normalization_type': 'HierarchicalLayerNorm'
            }
            
        except Exception as e:
            logger.warning(f"Failed to get normalization statistics: {e}")
            return {
                'error': str(e),
                'normalization_type': 'HierarchicalLayerNorm'
            }


# Utility functions for integration

def create_selective_processing_components(model_dim: int, vocab_size: int, num_splats: int = 20,
                                         processing_config: Optional[Dict] = None) -> Dict[str, nn.Module]:
    """
    Factory function to create all selective processing components
    
    Args:
        model_dim: Model dimension
        vocab_size: Vocabulary size for output projection
        num_splats: Number of splats
        processing_config: Configuration for processing components
        
    Returns:
        Dictionary of selective processing components
    """
    
    try:
        config = processing_config or {}
        
        components = {
            'token_processor': SelectiveTokenProcessor(
                model_dim, num_splats, 
                processing_config=config.get('token_processor', {})
            ),
            'output_projection': SparseOutputProjection(
                model_dim, vocab_size,
                sparsity_config=config.get('output_projection', {})
            ),
            'layer_norm': HierarchicalLayerNorm(
                model_dim, num_splats,
                norm_config=config.get('layer_norm', {})
            )
        }
        
        logger.info(f"🚀 Created selective processing components")
        logger.info(f"   Components: {list(components.keys())}")
        
        return components
        
    except Exception as e:
        logger.error(f"Failed to create selective processing components: {e}")
        raise


def get_selective_processing_statistics(components: Dict[str, nn.Module]) -> Dict:
    """Get comprehensive statistics from all selective processing components"""
    
    try:
        stats = {}
        
        for name, component in components.items():
            if hasattr(component, 'get_processing_statistics'):
                stats[name] = component.get_processing_statistics()
            elif hasattr(component, 'get_projection_statistics'):
                stats[name] = component.get_projection_statistics()
            elif hasattr(component, 'get_normalization_statistics'):
                stats[name] = component.get_normalization_statistics()
            else:
                stats[name] = {
                    'type': type(component).__name__,
                    'parameters': sum(p.numel() for p in component.parameters()),
                    'status': 'no_statistics_available'
                }
        
        # Calculate overall selective processing efficiency
        if stats:
            efficiency_metrics = []
            for component_stats in stats.values():
                if isinstance(component_stats, dict):
                    # Extract efficiency indicators
                    if 'processing_efficiency' in component_stats:
                        efficiency_metrics.append(component_stats['processing_efficiency'].get('selectivity_ratio', 0))
                    elif 'computation_savings' in component_stats:
                        efficiency_metrics.append(component_stats['computation_savings'])
            
            if efficiency_metrics:
                stats['overall_efficiency'] = {
                    'avg_efficiency': np.mean(efficiency_metrics),
                    'efficiency_variance': np.var(efficiency_metrics),
                    'efficiency_components': len(efficiency_metrics)
                }
        
        return stats
        
    except Exception as e:
        logger.warning(f"Failed to get selective processing statistics: {e}")
        return {'error': str(e)}
