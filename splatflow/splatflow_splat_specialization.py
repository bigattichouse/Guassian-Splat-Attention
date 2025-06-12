"""
SplatFlow Splat Specialization Module
Manages splat role specialization system for O(n*k) optimization.
Enables splats to discover and specialize in different content patterns: syntax, semantics, long-range dependencies, etc.
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


class SplatRole(Enum):
    """Specialized roles that splats can adopt"""
    GENERAL = "general"           # Default unspecialized role
    SYNTAX = "syntax"             # Local syntactic patterns, grammar
    SEMANTICS = "semantics"       # Semantic relationships, meaning
    LONG_RANGE = "long_range"     # Long-distance dependencies
    POSITIONAL = "positional"     # Position-sensitive patterns
    STRUCTURAL = "structural"     # Document structure, formatting
    TEMPORAL = "temporal"         # Sequential, temporal patterns
    ATTENTION_HUB = "attention_hub"  # High-attention coordination
    CONTEXT_BRIDGE = "context_bridge"  # Cross-context connections


class SplatSpecializationManager(nn.Module):
    """
    Manages splat roles and specialization processes.
    Tracks specialization evolution and coordinates role assignments.
    """
    
    def __init__(self, model_dim: int, num_splats: int = 20, 
                 specialization_config: Optional[Dict] = None):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_splats = num_splats
        
        # Parse specialization configuration
        self.config = specialization_config or {}
        self.enable_dynamic_specialization = self.config.get('enable_dynamic_specialization', True)
        self.specialization_strength = self.config.get('specialization_strength', 0.1)
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        self.role_transition_cooldown = self.config.get('role_transition_cooldown', 50)
        
        # Splat role tracking
        self.splat_roles = [SplatRole.GENERAL] * num_splats
        self.role_confidences = torch.zeros(num_splats)
        self.role_histories = [deque(maxlen=100) for _ in range(num_splats)]
        self.role_transition_counters = torch.zeros(num_splats)
        
        # Role assignment networks
        self.role_classifier = SplatRoleClassifier(model_dim, len(SplatRole))
        
        # Specialization-guided processing
        self.specialized_processors = nn.ModuleDict({
            role.value: SpecializationGuidedFF(model_dim, role)
            for role in SplatRole
        })
        
        # Role stability tracking
        self.role_stability_scores = torch.ones(num_splats)
        self.specialization_effectiveness = defaultdict(list)
        
        # Usage pattern analysis
        self.usage_pattern_analyzer = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),  # token + position + context
            nn.GELU(),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, len(SplatRole)),
            nn.Softmax(dim=-1)
        )
        
        # Statistics
        self.specialization_stats = {
            'role_transitions': defaultdict(int),
            'specialization_successes': defaultdict(int),
            'role_effectiveness': defaultdict(float),
            'total_specializations': 0,
            'stable_specializations': 0
        }
        
        logger.info(f"🎭 Splat specialization manager initialized")
        logger.info(f"   Num splats: {num_splats}, Available roles: {len(SplatRole)}")
        logger.info(f"   Dynamic specialization: {self.enable_dynamic_specialization}")
    
    def forward(self, token_embeddings: torch.Tensor,
                splat_states: List[torch.Tensor],
                attention_weights: Optional[torch.Tensor] = None,
                position_info: Optional[torch.Tensor] = None,
                return_specialization_info: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Apply specialization-guided processing to splat states
        
        Args:
            token_embeddings: [batch_size, seq_len, model_dim]
            splat_states: List of splat state tensors
            attention_weights: Optional attention patterns
            position_info: Optional positional information
            return_specialization_info: Return detailed specialization info
            
        Returns:
            Processed splat states, optionally with specialization info
        """
        
        specialization_info = {
            'role_assignments': {},
            'confidence_scores': {},
            'role_transitions': {},
            'processing_paths': {}
        }
        
        try:
            processed_splat_states = []
            
            for splat_idx, splat_state in enumerate(splat_states):
                if splat_idx >= self.num_splats:
                    # Handle case where more splats than expected
                    processed_splat_states.append(splat_state)
                    continue
                
                # Analyze current usage patterns
                usage_patterns = self._analyze_usage_patterns(
                    splat_state, token_embeddings, attention_weights, position_info
                )
                
                # Update role assignment if needed
                if self.enable_dynamic_specialization:
                    self._update_splat_role(splat_idx, usage_patterns, specialization_info)
                
                # Apply specialized processing
                current_role = self.splat_roles[splat_idx]
                processed_state = self._apply_specialized_processing(
                    splat_state, current_role, usage_patterns, splat_idx, specialization_info
                )
                
                processed_splat_states.append(processed_state)
                
                # Record role assignment info
                specialization_info['role_assignments'][f'splat_{splat_idx}'] = {
                    'role': current_role.value,
                    'confidence': self.role_confidences[splat_idx].item(),
                    'stability': self.role_stability_scores[splat_idx].item()
                }
            
            # Update statistics
            self._update_specialization_statistics(specialization_info)
            
            if return_specialization_info:
                return processed_splat_states, specialization_info
            else:
                return processed_splat_states
                
        except Exception as e:
            logger.warning(f"Specialization processing failed: {e}")
            # Fallback to unmodified splat states
            if return_specialization_info:
                specialization_info['error'] = str(e)
                specialization_info['fallback_used'] = True
                return splat_states, specialization_info
            else:
                return splat_states
    
    def _analyze_usage_patterns(self, splat_state: torch.Tensor,
                               token_embeddings: torch.Tensor,
                               attention_weights: Optional[torch.Tensor],
                               position_info: Optional[torch.Tensor]) -> Dict:
        """Analyze how the splat is being used to determine optimal role"""
        
        try:
            batch_size, seq_len, model_dim = token_embeddings.shape
            
            # Create context vector for analysis
            if position_info is not None:
                context = torch.cat([
                    splat_state.mean(dim=0, keepdim=True),  # Splat representation
                    token_embeddings.mean(dim=(0, 1), keepdim=True),  # Token context
                    position_info.mean(dim=(0, 1), keepdim=True) if position_info.dim() > 1 else position_info.unsqueeze(0)
                ], dim=-1)
            else:
                # Create dummy position info
                dummy_pos = torch.zeros(1, model_dim, device=splat_state.device)
                context = torch.cat([
                    splat_state.mean(dim=0, keepdim=True),
                    token_embeddings.mean(dim=(0, 1), keepdim=True),
                    dummy_pos
                ], dim=-1)
            
            # Analyze usage patterns
            usage_probs = self.usage_pattern_analyzer(context)  # [1, num_roles]
            
            # Additional pattern analysis
            patterns = {
                'role_probabilities': usage_probs.squeeze(0),
                'attention_diversity': self._calculate_attention_diversity(attention_weights),
                'position_sensitivity': self._calculate_position_sensitivity(splat_state, position_info),
                'semantic_complexity': self._calculate_semantic_complexity(splat_state, token_embeddings),
                'temporal_patterns': self._calculate_temporal_patterns(splat_state)
            }
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Usage pattern analysis failed: {e}")
            # Return default patterns
            return {
                'role_probabilities': torch.zeros(len(SplatRole)),
                'attention_diversity': 0.0,
                'position_sensitivity': 0.0,
                'semantic_complexity': 0.0,
                'temporal_patterns': 0.0
            }
    
    def _calculate_attention_diversity(self, attention_weights: Optional[torch.Tensor]) -> float:
        """Calculate how diverse the attention patterns are"""
        
        try:
            if attention_weights is None:
                return 0.0
            
            # Calculate entropy of attention distribution
            attention_probs = F.softmax(attention_weights.flatten(), dim=0)
            entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum()
            
            # Normalize entropy
            max_entropy = math.log(attention_weights.numel())
            diversity = (entropy / max_entropy).item()
            
            return diversity
            
        except Exception as e:
            logger.warning(f"Attention diversity calculation failed: {e}")
            return 0.0
    
    def _calculate_position_sensitivity(self, splat_state: torch.Tensor,
                                      position_info: Optional[torch.Tensor]) -> float:
        """Calculate how sensitive the splat is to positional information"""
        
        try:
            if position_info is None:
                return 0.0
            
            # Calculate correlation between splat state and position
            splat_flat = splat_state.flatten()
            pos_flat = position_info.flatten()
            
            if len(splat_flat) != len(pos_flat):
                # Adjust sizes if needed
                min_len = min(len(splat_flat), len(pos_flat))
                splat_flat = splat_flat[:min_len]
                pos_flat = pos_flat[:min_len]
            
            correlation = torch.corrcoef(torch.stack([splat_flat, pos_flat]))[0, 1]
            
            if torch.isnan(correlation):
                return 0.0
            
            return abs(correlation.item())
            
        except Exception as e:
            logger.warning(f"Position sensitivity calculation failed: {e}")
            return 0.0
    
    def _calculate_semantic_complexity(self, splat_state: torch.Tensor,
                                     token_embeddings: torch.Tensor) -> float:
        """Calculate semantic complexity of splat interactions"""
        
        try:
            # Calculate variance in splat-token interactions
            splat_repr = splat_state.mean(dim=0)
            token_repr = token_embeddings.mean(dim=(0, 1))
            
            # Compute interaction complexity
            interaction = torch.dot(splat_repr, token_repr)
            complexity = torch.var(splat_state).item() + torch.var(token_embeddings).item()
            
            return complexity * abs(interaction.item())
            
        except Exception as e:
            logger.warning(f"Semantic complexity calculation failed: {e}")
            return 0.0
    
    def _calculate_temporal_patterns(self, splat_state: torch.Tensor) -> float:
        """Calculate temporal pattern strength in splat state"""
        
        try:
            if splat_state.dim() < 2:
                return 0.0
            
            # Calculate autocorrelation to detect temporal patterns
            state_seq = splat_state.mean(dim=-1)  # Average across features
            
            if len(state_seq) < 2:
                return 0.0
            
            # Simple autocorrelation at lag 1
            shifted = torch.roll(state_seq, 1)
            correlation = torch.corrcoef(torch.stack([state_seq, shifted]))[0, 1]
            
            if torch.isnan(correlation):
                return 0.0
            
            return abs(correlation.item())
            
        except Exception as e:
            logger.warning(f"Temporal pattern calculation failed: {e}")
            return 0.0
    
    def _update_splat_role(self, splat_idx: int, usage_patterns: Dict,
                          specialization_info: Dict):
        """Update splat role based on usage patterns"""
        
        try:
            # Check if role transition is allowed (cooldown)
            if self.role_transition_counters[splat_idx] > 0:
                self.role_transition_counters[splat_idx] -= 1
                return
            
            # Get role probabilities
            role_probs = usage_patterns['role_probabilities']
            best_role_idx = torch.argmax(role_probs)
            best_role_prob = role_probs[best_role_idx].item()
            
            # Convert index to role
            roles_list = list(SplatRole)
            candidate_role = roles_list[best_role_idx]
            current_role = self.splat_roles[splat_idx]
            
            # Decide whether to transition
            should_transition = (
                candidate_role != current_role and
                best_role_prob > self.min_confidence_threshold and
                best_role_prob > self.role_confidences[splat_idx] + 0.1  # Hysteresis
            )
            
            if should_transition:
                # Record transition
                old_role = current_role
                self.splat_roles[splat_idx] = candidate_role
                self.role_confidences[splat_idx] = best_role_prob
                self.role_transition_counters[splat_idx] = self.role_transition_cooldown
                
                # Update statistics
                transition_key = f"{old_role.value}_to_{candidate_role.value}"
                self.specialization_stats['role_transitions'][transition_key] += 1
                self.specialization_stats['total_specializations'] += 1
                
                # Record in history
                self.role_histories[splat_idx].append({
                    'from_role': old_role.value,
                    'to_role': candidate_role.value,
                    'confidence': best_role_prob,
                    'timestamp': len(self.role_histories[splat_idx])
                })
                
                specialization_info['role_transitions'][f'splat_{splat_idx}'] = {
                    'from': old_role.value,
                    'to': candidate_role.value,
                    'confidence': best_role_prob,
                    'reason': 'usage_pattern_analysis'
                }
                
                logger.debug(f"Splat {splat_idx}: {old_role.value} → {candidate_role.value} "
                           f"(confidence: {best_role_prob:.3f})")
            
            else:
                # Update confidence for current role
                if candidate_role == current_role:
                    # Reinforce current role
                    self.role_confidences[splat_idx] = max(
                        self.role_confidences[splat_idx],
                        best_role_prob
                    )
                
                specialization_info['confidence_scores'][f'splat_{splat_idx}'] = {
                    'current_role_confidence': self.role_confidences[splat_idx].item(),
                    'candidate_confidence': best_role_prob,
                    'transition_allowed': self.role_transition_counters[splat_idx] == 0
                }
            
        except Exception as e:
            logger.warning(f"Role update failed for splat {splat_idx}: {e}")
    
    def _apply_specialized_processing(self, splat_state: torch.Tensor,
                                    role: SplatRole, usage_patterns: Dict,
                                    splat_idx: int, specialization_info: Dict) -> torch.Tensor:
        """Apply role-specific processing to splat state"""
        
        try:
            # Get specialized processor for this role
            processor = self.specialized_processors[role.value]
            
            # Apply specialized processing
            processed_state = processor(
                splat_state, 
                usage_patterns=usage_patterns,
                splat_idx=splat_idx
            )
            
            # Record processing path
            specialization_info['processing_paths'][f'splat_{splat_idx}'] = {
                'processor_type': role.value,
                'input_norm': torch.norm(splat_state).item(),
                'output_norm': torch.norm(processed_state).item(),
                'processing_gain': torch.norm(processed_state).item() / (torch.norm(splat_state).item() + 1e-8)
            }
            
            return processed_state
            
        except Exception as e:
            logger.warning(f"Specialized processing failed for splat {splat_idx}, role {role.value}: {e}")
            # Fallback to general processing
            general_processor = self.specialized_processors[SplatRole.GENERAL.value]
            return general_processor(splat_state, usage_patterns=usage_patterns, splat_idx=splat_idx)
    
    def _update_specialization_statistics(self, specialization_info: Dict):
        """Update specialization statistics"""
        
        try:
            # Count stable specializations
            stable_count = 0
            for splat_idx in range(self.num_splats):
                if (self.role_confidences[splat_idx] > self.min_confidence_threshold and
                    self.role_transition_counters[splat_idx] == 0):
                    stable_count += 1
            
            self.specialization_stats['stable_specializations'] = stable_count
            
            # Update role effectiveness tracking
            for role in SplatRole:
                role_splats = [i for i, r in enumerate(self.splat_roles) if r == role]
                if role_splats:
                    avg_confidence = self.role_confidences[role_splats].mean().item()
                    self.specialization_stats['role_effectiveness'][role.value] = avg_confidence
            
        except Exception as e:
            logger.warning(f"Failed to update specialization statistics: {e}")
    
    def get_specialization_summary(self) -> Dict:
        """Get comprehensive specialization summary"""
        
        try:
            # Role distribution
            role_counts = defaultdict(int)
            role_confidences_by_type = defaultdict(list)
            
            for i, role in enumerate(self.splat_roles):
                role_counts[role.value] += 1
                role_confidences_by_type[role.value].append(self.role_confidences[i].item())
            
            # Calculate average confidences
            avg_confidences = {}
            for role, confidences in role_confidences_by_type.items():
                avg_confidences[role] = np.mean(confidences) if confidences else 0.0
            
            # Specialization effectiveness
            specialization_ratio = (
                self.specialization_stats['stable_specializations'] / max(self.num_splats, 1)
            )
            
            return {
                'role_distribution': dict(role_counts),
                'average_role_confidences': avg_confidences,
                'specialization_ratio': specialization_ratio,
                'total_role_transitions': sum(self.specialization_stats['role_transitions'].values()),
                'role_transition_history': dict(self.specialization_stats['role_transitions']),
                'role_effectiveness': dict(self.specialization_stats['role_effectiveness']),
                'stable_specializations': self.specialization_stats['stable_specializations'],
                'num_splats': self.num_splats,
                'dynamic_specialization_enabled': self.enable_dynamic_specialization,
                'specialization_type': 'SplatSpecializationManager'
            }
            
        except Exception as e:
            logger.warning(f"Failed to get specialization summary: {e}")
            return {
                'error': str(e),
                'specialization_type': 'SplatSpecializationManager'
            }
    
    def force_role_assignment(self, splat_idx: int, role: SplatRole, confidence: float = 1.0):
        """Manually assign a role to a specific splat"""
        
        try:
            if 0 <= splat_idx < self.num_splats:
                old_role = self.splat_roles[splat_idx]
                self.splat_roles[splat_idx] = role
                self.role_confidences[splat_idx] = confidence
                self.role_transition_counters[splat_idx] = 0  # Allow immediate transition
                
                logger.info(f"Manually assigned splat {splat_idx}: {old_role.value} → {role.value}")
                return True
            else:
                logger.warning(f"Invalid splat index: {splat_idx}")
                return False
                
        except Exception as e:
            logger.warning(f"Failed to force role assignment: {e}")
            return False


class SplatRoleClassifier(nn.Module):
    """
    Classifies optimal roles for splats based on usage patterns and content analysis.
    Learns to recognize when splats should specialize in specific types of processing.
    """
    
    def __init__(self, model_dim: int, num_roles: int):
        super().__init__()
        
        self.model_dim = model_dim
        self.num_roles = num_roles
        
        # Multi-layer role classification network
        self.feature_extractor = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Role-specific analysis heads
        self.role_analyzers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim // 2, model_dim // 4),
                nn.GELU(),
                nn.Linear(model_dim // 4, 1),
                nn.Sigmoid()
            ) for _ in range(num_roles)
        ])
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(model_dim // 2, model_dim // 4),
            nn.GELU(),
            nn.Linear(model_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Context-aware role selection
        self.context_integrator = nn.Sequential(
            nn.Linear(model_dim + num_roles, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, num_roles),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, splat_features: torch.Tensor,
                context_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify optimal role for splat
        
        Args:
            splat_features: Features describing splat usage patterns
            context_features: Optional context information
            
        Returns:
            role_probabilities: [num_roles] probability distribution over roles
            confidence: [1] confidence in the classification
        """
        
        try:
            # Extract features
            features = self.feature_extractor(splat_features)
            
            # Analyze suitability for each role
            role_scores = []
            for analyzer in self.role_analyzers:
                score = analyzer(features)
                role_scores.append(score)
            
            role_scores = torch.cat(role_scores, dim=-1)  # [num_roles]
            
            # Estimate confidence
            confidence = self.confidence_estimator(features)
            
            # Integrate context if available
            if context_features is not None:
                combined_features = torch.cat([splat_features, role_scores], dim=-1)
                role_probabilities = self.context_integrator(combined_features)
            else:
                role_probabilities = F.softmax(role_scores, dim=-1)
            
            return role_probabilities, confidence
            
        except Exception as e:
            logger.warning(f"Role classification failed: {e}")
            # Return uniform distribution and low confidence
            uniform_probs = torch.ones(self.num_roles) / self.num_roles
            low_confidence = torch.tensor([0.1])
            return uniform_probs, low_confidence


class SpecializationGuidedFF(nn.Module):
    """
    Role-specific feed-forward processor that adapts its architecture
    based on the specialized role of the splat.
    """
    
    def __init__(self, model_dim: int, role: SplatRole):
        super().__init__()
        
        self.model_dim = model_dim
        self.role = role
        
        # Create role-specific architecture
        if role == SplatRole.SYNTAX:
            # Syntax processing: focus on local patterns, faster processing
            self.processor = self._create_syntax_processor()
        elif role == SplatRole.SEMANTICS:
            # Semantic processing: deeper analysis, meaning extraction
            self.processor = self._create_semantic_processor()
        elif role == SplatRole.LONG_RANGE:
            # Long-range processing: attention to distant relationships
            self.processor = self._create_long_range_processor()
        elif role == SplatRole.POSITIONAL:
            # Positional processing: position-aware computations
            self.processor = self._create_positional_processor()
        elif role == SplatRole.STRUCTURAL:
            # Structural processing: document organization patterns
            self.processor = self._create_structural_processor()
        elif role == SplatRole.TEMPORAL:
            # Temporal processing: sequential pattern recognition
            self.processor = self._create_temporal_processor()
        elif role == SplatRole.ATTENTION_HUB:
            # Attention hub: coordinate high-attention regions
            self.processor = self._create_attention_hub_processor()
        elif role == SplatRole.CONTEXT_BRIDGE:
            # Context bridge: connect different contexts
            self.processor = self._create_context_bridge_processor()
        else:
            # General processing: balanced, versatile
            self.processor = self._create_general_processor()
        
        # Role-specific statistics
        self.processing_count = 0
        self.effectiveness_history = deque(maxlen=50)
    
    def _create_syntax_processor(self) -> nn.Module:
        """Create processor optimized for syntactic patterns"""
        return nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.ReLU(),  # Fast activation for syntax
            nn.Linear(self.model_dim // 2, self.model_dim),
            nn.Dropout(0.05)  # Lower dropout for local patterns
        )
    
    def _create_semantic_processor(self) -> nn.Module:
        """Create processor optimized for semantic analysis"""
        return nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim * 2),
            nn.GELU(),  # Better for semantic processing
            nn.Dropout(0.1),
            nn.Linear(self.model_dim * 2, self.model_dim),
            nn.LayerNorm(self.model_dim)
        )
    
    def _create_long_range_processor(self) -> nn.Module:
        """Create processor optimized for long-range dependencies"""
        return nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(self.model_dim * 2, self.model_dim)
        )
    
    def _create_positional_processor(self) -> nn.Module:
        """Create processor optimized for positional patterns"""
        # Add positional encoding enhancement
        class PositionallyAwareProcessor(nn.Module):
            def __init__(self, model_dim):
                super().__init__()
                self.base_processor = nn.Sequential(
                    nn.Linear(model_dim, model_dim),
                    nn.GELU(),
                    nn.Linear(model_dim, model_dim)
                )
                self.positional_enhancement = nn.Linear(model_dim, model_dim)
            
            def forward(self, x, **kwargs):
                base_output = self.base_processor(x)
                pos_enhanced = self.positional_enhancement(x)
                return base_output + 0.2 * pos_enhanced
        
        return PositionallyAwareProcessor(self.model_dim)
    
    def _create_structural_processor(self) -> nn.Module:
        """Create processor optimized for structural patterns"""
        return nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim // 2),
            nn.GELU(),
            nn.Linear(self.model_dim // 2, self.model_dim // 2),
            nn.GELU(),
            nn.Linear(self.model_dim // 2, self.model_dim),
            nn.Dropout(0.1)
        )
    
    def _create_temporal_processor(self) -> nn.Module:
        """Create processor optimized for temporal patterns"""
        # Simple GRU-like processing for temporal awareness
        class TemporalProcessor(nn.Module):
            def __init__(self, model_dim):
                super().__init__()
                self.temporal_gate = nn.Sequential(
                    nn.Linear(model_dim, model_dim),
                    nn.Sigmoid()
                )
                self.update_gate = nn.Sequential(
                    nn.Linear(model_dim, model_dim),
                    nn.Tanh()
                )
                self.final_proj = nn.Linear(model_dim, model_dim)
            
            def forward(self, x, **kwargs):
                gate = self.temporal_gate(x)
                update = self.update_gate(x)
                combined = x * gate + update * (1 - gate)
                return self.final_proj(combined)
        
        return TemporalProcessor(self.model_dim)
    
    def _create_attention_hub_processor(self) -> nn.Module:
        """Create processor optimized for attention coordination"""
        return nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim * 3),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.model_dim * 3, self.model_dim),
            nn.LayerNorm(self.model_dim)
        )
    
    def _create_context_bridge_processor(self) -> nn.Module:
        """Create processor optimized for context bridging"""
        return nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.model_dim * 2, self.model_dim)
        )
    
    def _create_general_processor(self) -> nn.Module:
        """Create general-purpose processor"""
        return nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.model_dim * 2, self.model_dim)
        )
    
    def forward(self, splat_state: torch.Tensor,
                usage_patterns: Optional[Dict] = None,
                splat_idx: Optional[int] = None) -> torch.Tensor:
        """
        Apply role-specific processing to splat state
        
        Args:
            splat_state: Current splat state tensor
            usage_patterns: Optional usage pattern information
            splat_idx: Optional splat index for tracking
            
        Returns:
            Processed splat state
        """
        
        try:
            # Apply role-specific processing
            processed_state = self.processor(splat_state)
            
            # Track effectiveness
            self.processing_count += 1
            
            if usage_patterns is not None:
                # Calculate processing effectiveness
                input_norm = torch.norm(splat_state).item()
                output_norm = torch.norm(processed_state).item()
                effectiveness = output_norm / (input_norm + 1e-8)
                self.effectiveness_history.append(effectiveness)
            
            return processed_state
            
        except Exception as e:
            logger.warning(f"Specialized processing failed for role {self.role.value}: {e}")
            # Fallback to identity
            return splat_state
    
    def get_effectiveness_score(self) -> float:
        """Get average effectiveness score for this processor"""
        if self.effectiveness_history:
            return np.mean(list(self.effectiveness_history))
        else:
            return 1.0


# Utility functions for integration

def create_specialized_splat_system(model_dim: int, num_splats: int,
                                   specialization_config: Optional[Dict] = None) -> SplatSpecializationManager:
    """
    Factory function to create a complete splat specialization system
    
    Args:
        model_dim: Model dimension
        num_splats: Number of splats to manage
        specialization_config: Configuration for specialization system
        
    Returns:
        Configured specialization manager
    """
    
    try:
        manager = SplatSpecializationManager(
            model_dim=model_dim,
            num_splats=num_splats,
            specialization_config=specialization_config
        )
        
        logger.info(f"🎭 Created specialized splat system")
        logger.info(f"   Model dim: {model_dim}, Num splats: {num_splats}")
        logger.info(f"   Available roles: {[role.value for role in SplatRole]}")
        
        return manager
        
    except Exception as e:
        logger.error(f"Failed to create specialized splat system: {e}")
        raise


def get_specialization_statistics(specialization_manager: SplatSpecializationManager) -> Dict:
    """Get comprehensive specialization statistics"""
    
    try:
        return specialization_manager.get_specialization_summary()
        
    except Exception as e:
        logger.warning(f"Failed to get specialization statistics: {e}")
        return {
            'error': str(e),
            'specialization_type': 'SplatSpecializationManager (Error)'
        }


def analyze_role_effectiveness(specialization_manager: SplatSpecializationManager) -> Dict:
    """Analyze effectiveness of different specialized roles"""
    
    try:
        summary = specialization_manager.get_specialization_summary()
        
        # Calculate role effectiveness metrics
        role_analysis = {}
        
        for role_name, confidence in summary.get('average_role_confidences', {}).items():
            role_count = summary.get('role_distribution', {}).get(role_name, 0)
            effectiveness = summary.get('role_effectiveness', {}).get(role_name, 0.0)
            
            role_analysis[role_name] = {
                'count': role_count,
                'avg_confidence': confidence,
                'effectiveness': effectiveness,
                'usage_ratio': role_count / max(summary.get('num_splats', 1), 1)
            }
        
        # Find most and least effective roles
        if role_analysis:
            sorted_by_effectiveness = sorted(
                role_analysis.items(),
                key=lambda x: x[1]['effectiveness'],
                reverse=True
            )
            
            most_effective = sorted_by_effectiveness[0] if sorted_by_effectiveness else None
            least_effective = sorted_by_effectiveness[-1] if sorted_by_effectiveness else None
        else:
            most_effective = least_effective = None
        
        return {
            'role_analysis': role_analysis,
            'most_effective_role': most_effective,
            'least_effective_role': least_effective,
            'total_specialized_ratio': summary.get('specialization_ratio', 0.0),
            'analysis_timestamp': summary.get('total_role_transitions', 0)
        }
        
    except Exception as e:
        logger.warning(f"Failed to analyze role effectiveness: {e}")
        return {
            'error': str(e),
            'analysis_type': 'role_effectiveness_analysis'
        }
