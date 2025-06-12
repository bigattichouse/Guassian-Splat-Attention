"""
FIXED: SplatFlow Constellation Templates Module
Resolves the ConstellationPattern.STRUCTURED attribute error and enhances base template initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List, Any, Union
from collections import defaultdict, OrderedDict
from enum import Enum
import json
import pickle
import threading

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content types for constellation template selection"""
    GENERAL = "general"               # General text, mixed content
    CODE = "code"                     # Source code, programming
    DIALOGUE = "dialogue"             # Conversations, chat
    NARRATIVE = "narrative"           # Stories, creative writing
    TECHNICAL = "technical"           # Documentation, manuals
    ACADEMIC = "academic"             # Papers, research
    NEWS = "news"                     # News articles, journalism
    STRUCTURED = "structured"         # Tables, lists, formatted data
    MATHEMATICAL = "mathematical"     # Math expressions, equations
    MULTILINGUAL = "multilingual"     # Mixed language content


class ConstellationPattern(Enum):
    """FIXED: Base constellation patterns with STRUCTURED pattern added"""
    UNIFORM_GRID = "uniform_grid"         # Evenly spaced grid pattern
    HIERARCHICAL = "hierarchical"         # Tree-like hierarchy
    CLUSTERED = "clustered"               # Grouped clusters
    SPIRAL = "spiral"                     # Spiral pattern
    ATTENTION_FOCUSED = "attention_focused"  # High-attention regions
    SEQUENTIAL = "sequential"             # Sequential processing order
    RADIAL = "radial"                     # Radial from center
    ADAPTIVE_DENSITY = "adaptive_density"  # Variable density regions
    STRUCTURED = "structured"             # FIXED: Added structured grid-like patterns


class ContentTypeClassifier(nn.Module):
    """Classifies content type from token embeddings"""
    
    def __init__(self, model_dim: int):
        super().__init__()
        
        self.model_dim = model_dim
        
        # Content type classification network
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(model_dim // 2, model_dim // 4),
            nn.GELU(),
            nn.Linear(model_dim // 4, len(ContentType)),
            nn.Softmax(dim=-1)
        )
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(model_dim, model_dim // 4),
            nn.GELU(),
            nn.Linear(model_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, token_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify content type from token embeddings
        
        Args:
            token_embeddings: [batch_size, seq_len, model_dim]
            
        Returns:
            (content_type_probs, confidence_score)
        """
        
        # Use mean pooling for sequence representation
        sequence_repr = torch.mean(token_embeddings, dim=1)  # [batch_size, model_dim]
        
        # Classify content type
        content_probs = self.classifier(sequence_repr)
        confidence = self.confidence_estimator(sequence_repr)
        
        return content_probs, confidence


class ConstellationTemplateLibrary(nn.Module):
    """
    FIXED: Library of proven splat arrangements for different content types.
    Stores and retrieves optimal constellation patterns based on content analysis.
    """
    
    def __init__(self, model_dim: int, max_templates: int = 1000,
                 template_config: Optional[Dict] = None):
        super().__init__()
        
        self.model_dim = model_dim
        self.max_templates = max_templates
        
        # Parse template configuration
        self.config = template_config or {}
        self.enable_adaptive_templates = self.config.get('enable_adaptive_templates', True)
        self.template_similarity_threshold = self.config.get('similarity_threshold', 0.85)
        self.quality_threshold = self.config.get('quality_threshold', 0.7)
        
        # Template storage by content type and pattern
        self.templates = defaultdict(lambda: defaultdict(list))
        self.template_metadata = {}
        self.template_performance = defaultdict(list)
        
        # Content type classification
        self.content_classifier = ContentTypeClassifier(model_dim)
        
        # Template quality analyzer
        self.quality_analyzer = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),  # position + performance + context
            nn.GELU(),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Template adaptation networks
        self.position_adapter = nn.Sequential(
            nn.Linear(model_dim + 2, model_dim),  # position + seq_len + target_len
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
            nn.Tanh()  # Keep positions bounded
        )
        
        # Base constellation generators
        self.constellation_generators = self._create_base_generators()
        
        # Statistics
        self.template_stats = {
            'templates_created': 0,
            'templates_used': 0,
            'adaptations_performed': 0,
            'quality_improvements': 0,
            'content_type_detections': defaultdict(int)
        }
        
        # Thread safety for template access
        self._template_lock = threading.RLock()
        
        # Initialize with base templates
        self._initialize_base_templates()
        
        logger.info(f"🌌 Constellation template library initialized")
        logger.info(f"   Model dim: {model_dim}, Max templates: {max_templates}")
        logger.info(f"   Content types: {len(ContentType)}, Patterns: {len(ConstellationPattern)}")
    
    def _create_base_generators(self) -> Dict[str, callable]:
        """Create base constellation pattern generators"""
        
        generators = {}
        
        generators['uniform_grid'] = self._generate_uniform_grid
        generators['hierarchical'] = self._generate_hierarchical
        generators['clustered'] = self._generate_clustered
        generators['spiral'] = self._generate_spiral
        generators['attention_focused'] = self._generate_attention_focused
        generators['sequential'] = self._generate_sequential
        generators['radial'] = self._generate_radial
        generators['adaptive_density'] = self._generate_adaptive_density
        generators['structured'] = self._generate_structured  # FIXED: Add structured generator
        
        return generators
    
    def _initialize_base_templates(self):
        """FIXED: Initialize library with proven base templates"""
        
        try:
            # Create base templates for each content type and pattern combination
            base_configs = [
                # General content templates
                {'content': ContentType.GENERAL, 'pattern': ConstellationPattern.UNIFORM_GRID, 'num_splats': 20, 'seq_len': 1024},
                {'content': ContentType.GENERAL, 'pattern': ConstellationPattern.CLUSTERED, 'num_splats': 24, 'seq_len': 1024},
                
                # Code-specific templates
                {'content': ContentType.CODE, 'pattern': ConstellationPattern.HIERARCHICAL, 'num_splats': 16, 'seq_len': 2048},
                {'content': ContentType.CODE, 'pattern': ConstellationPattern.SEQUENTIAL, 'num_splats': 20, 'seq_len': 1024},
                
                # Dialogue templates
                {'content': ContentType.DIALOGUE, 'pattern': ConstellationPattern.SEQUENTIAL, 'num_splats': 18, 'seq_len': 512},
                {'content': ContentType.DIALOGUE, 'pattern': ConstellationPattern.ATTENTION_FOCUSED, 'num_splats': 22, 'seq_len': 1024},
                
                # Narrative templates
                {'content': ContentType.NARRATIVE, 'pattern': ConstellationPattern.SPIRAL, 'num_splats': 25, 'seq_len': 2048},
                {'content': ContentType.NARRATIVE, 'pattern': ConstellationPattern.ADAPTIVE_DENSITY, 'num_splats': 28, 'seq_len': 1536},
                
                # Technical documentation - FIXED: Use valid patterns
                {'content': ContentType.TECHNICAL, 'pattern': ConstellationPattern.HIERARCHICAL, 'num_splats': 20, 'seq_len': 2048},
                {'content': ContentType.TECHNICAL, 'pattern': ConstellationPattern.STRUCTURED, 'num_splats': 24, 'seq_len': 1024},  # FIXED: Now valid
                
                # Academic papers - FIXED: Use valid patterns
                {'content': ContentType.ACADEMIC, 'pattern': ConstellationPattern.RADIAL, 'num_splats': 32, 'seq_len': 3072},
                {'content': ContentType.ACADEMIC, 'pattern': ConstellationPattern.HIERARCHICAL, 'num_splats': 28, 'seq_len': 2048},
                
                # News and journalism
                {'content': ContentType.NEWS, 'pattern': ConstellationPattern.ATTENTION_FOCUSED, 'num_splats': 22, 'seq_len': 1024},
                {'content': ContentType.NEWS, 'pattern': ConstellationPattern.SEQUENTIAL, 'num_splats': 18, 'seq_len': 512},
                
                # Structured data - FIXED: Use the new STRUCTURED pattern
                {'content': ContentType.STRUCTURED, 'pattern': ConstellationPattern.STRUCTURED, 'num_splats': 16, 'seq_len': 1024},
                {'content': ContentType.STRUCTURED, 'pattern': ConstellationPattern.UNIFORM_GRID, 'num_splats': 20, 'seq_len': 512},
                
                # Mathematical content
                {'content': ContentType.MATHEMATICAL, 'pattern': ConstellationPattern.CLUSTERED, 'num_splats': 18, 'seq_len': 1024},
                {'content': ContentType.MATHEMATICAL, 'pattern': ConstellationPattern.HIERARCHICAL, 'num_splats': 22, 'seq_len': 1536},
                
                # Multilingual content
                {'content': ContentType.MULTILINGUAL, 'pattern': ConstellationPattern.ADAPTIVE_DENSITY, 'num_splats': 24, 'seq_len': 1024},
                {'content': ContentType.MULTILINGUAL, 'pattern': ConstellationPattern.CLUSTERED, 'num_splats': 20, 'seq_len': 2048}
            ]
            
            templates_created = 0
            for config in base_configs:
                try:
                    positions = self._generate_base_constellation(
                        config['content'], config['pattern'], 
                        config['num_splats'], config['seq_len']
                    )
                    
                    if positions is not None:
                        self._store_template(
                            content_type=config['content'],
                            pattern=config['pattern'],
                            positions=positions,
                            sequence_length=config['seq_len'],
                            performance_score=0.8,  # Base template score
                            metadata={'source': 'base_initialization', 'proven': True}
                        )
                        templates_created += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to create template for {config['content'].value}/{config['pattern'].value}: {e}")
                    continue
            
            logger.info(f"🌟 Initialized {templates_created} base constellation templates")
            
        except Exception as e:
            logger.warning(f"Failed to initialize base templates: {e}")
    
    def _generate_structured(self, num_splats: int, sequence_length: int, **kwargs) -> List[torch.Tensor]:
        """FIXED: Generate structured grid-like constellation pattern"""
        
        try:
            positions = []
            
            # Create a structured grid pattern suitable for tabular/structured data
            grid_size = int(math.ceil(math.sqrt(num_splats)))
            
            # Create evenly spaced grid positions
            for i in range(num_splats):
                row = i // grid_size
                col = i % grid_size
                
                # Normalize positions to sequence length
                pos_x = (col / max(grid_size - 1, 1)) * sequence_length
                pos_y = (row / max(grid_size - 1, 1)) * 0.1  # Small vertical offset
                
                # Create position tensor
                position = torch.tensor([pos_x, pos_y], dtype=torch.float32)
                positions.append(position)
            
            return positions
            
        except Exception as e:
            logger.warning(f"Structured pattern generation failed: {e}")
            return self._generate_uniform_grid(num_splats, sequence_length, **kwargs)
    
    def _generate_base_constellation(self, content_type: ContentType, pattern: ConstellationPattern,
                                   num_splats: int, sequence_length: int) -> Optional[List[torch.Tensor]]:
        """Generate base constellation using specified pattern"""
        
        try:
            generator_name = pattern.value
            if generator_name in self.constellation_generators:
                return self.constellation_generators[generator_name](num_splats, sequence_length)
            else:
                logger.warning(f"Unknown pattern generator: {generator_name}")
                return self._generate_uniform_grid(num_splats, sequence_length)
                
        except Exception as e:
            logger.warning(f"Base constellation generation failed: {e}")
            return None
    
    def _generate_uniform_grid(self, num_splats: int, sequence_length: int, **kwargs) -> List[torch.Tensor]:
        """Generate uniform grid pattern"""
        
        positions = []
        spacing = sequence_length / max(num_splats - 1, 1)
        
        for i in range(num_splats):
            pos = torch.tensor([i * spacing], dtype=torch.float32)
            positions.append(pos)
        
        return positions
    
    def _generate_hierarchical(self, num_splats: int, sequence_length: int, **kwargs) -> List[torch.Tensor]:
        """Generate hierarchical tree pattern"""
        
        positions = []
        levels = int(math.ceil(math.log2(num_splats + 1)))
        
        for i in range(num_splats):
            level = int(math.log2(i + 1))
            pos_in_level = i - (2**level - 1)
            level_spacing = sequence_length / (2**level)
            
            pos = torch.tensor([pos_in_level * level_spacing + level_spacing/2], dtype=torch.float32)
            positions.append(pos)
        
        return positions
    
    def _generate_clustered(self, num_splats: int, sequence_length: int, **kwargs) -> List[torch.Tensor]:
        """Generate clustered pattern"""
        
        positions = []
        num_clusters = max(1, num_splats // 4)
        splats_per_cluster = num_splats // num_clusters
        
        for cluster in range(num_clusters):
            cluster_center = (cluster + 0.5) * sequence_length / num_clusters
            
            for i in range(splats_per_cluster):
                if len(positions) >= num_splats:
                    break
                offset = (i - splats_per_cluster/2) * 10
                pos = torch.tensor([cluster_center + offset], dtype=torch.float32)
                positions.append(pos)
        
        # Add remaining splats if needed
        while len(positions) < num_splats:
            pos = torch.tensor([len(positions) * sequence_length / num_splats], dtype=torch.float32)
            positions.append(pos)
        
        return positions
    
    def _generate_spiral(self, num_splats: int, sequence_length: int, **kwargs) -> List[torch.Tensor]:
        """Generate spiral pattern"""
        
        positions = []
        center = sequence_length / 2
        
        for i in range(num_splats):
            angle = i * 2 * math.pi / num_splats
            radius = (i / num_splats) * (sequence_length / 4)
            
            x = center + radius * math.cos(angle)
            y = radius * math.sin(angle) * 0.1  # Small y component
            
            pos = torch.tensor([x, y], dtype=torch.float32)
            positions.append(pos)
        
        return positions
    
    def _generate_attention_focused(self, num_splats: int, sequence_length: int, **kwargs) -> List[torch.Tensor]:
        """Generate attention-focused pattern"""
        
        positions = []
        
        # Place more splats at the beginning and end (high attention areas)
        focus_regions = [0.1, 0.9]  # 10% and 90% of sequence
        
        for i, focus in enumerate(focus_regions):
            region_splats = num_splats // len(focus_regions)
            region_center = focus * sequence_length
            
            for j in range(region_splats):
                if len(positions) >= num_splats:
                    break
                offset = (j - region_splats/2) * 20
                pos = torch.tensor([region_center + offset], dtype=torch.float32)
                positions.append(pos)
        
        # Fill remaining with uniform distribution
        while len(positions) < num_splats:
            pos = torch.tensor([len(positions) * sequence_length / num_splats], dtype=torch.float32)
            positions.append(pos)
        
        return positions
    
    def _generate_sequential(self, num_splats: int, sequence_length: int, **kwargs) -> List[torch.Tensor]:
        """Generate sequential pattern"""
        
        positions = []
        
        for i in range(num_splats):
            # Sequential with slight overlap
            pos = torch.tensor([i * sequence_length * 0.9 / max(num_splats - 1, 1)], dtype=torch.float32)
            positions.append(pos)
        
        return positions
    
    def _generate_radial(self, num_splats: int, sequence_length: int, **kwargs) -> List[torch.Tensor]:
        """Generate radial pattern"""
        
        positions = []
        center = sequence_length / 2
        
        for i in range(num_splats):
            angle = i * 2 * math.pi / num_splats
            radius = sequence_length / 4
            
            x = center + radius * math.cos(angle)
            y = radius * math.sin(angle) * 0.05  # Very small y component
            
            pos = torch.tensor([x, y], dtype=torch.float32)
            positions.append(pos)
        
        return positions
    
    def _generate_adaptive_density(self, num_splats: int, sequence_length: int, **kwargs) -> List[torch.Tensor]:
        """Generate adaptive density pattern"""
        
        positions = []
        
        # Higher density at beginning, middle, and end
        density_peaks = [0.0, 0.5, 1.0]
        total_weight = sum([1.0, 2.0, 1.0])  # Middle gets more weight
        
        splats_allocated = 0
        for i, (peak, weight) in enumerate(zip(density_peaks, [1.0, 2.0, 1.0])):
            region_splats = int(num_splats * weight / total_weight)
            region_center = peak * sequence_length
            
            for j in range(region_splats):
                if splats_allocated >= num_splats:
                    break
                offset = (j - region_splats/2) * 15
                pos = torch.tensor([region_center + offset], dtype=torch.float32)
                positions.append(pos)
                splats_allocated += 1
        
        # Fill any remaining splats
        while len(positions) < num_splats:
            pos = torch.tensor([len(positions) * sequence_length / num_splats], dtype=torch.float32)
            positions.append(pos)
        
        return positions
    
    def _store_template(self, content_type: ContentType, pattern: ConstellationPattern,
                       positions: List[torch.Tensor], sequence_length: int,
                       performance_score: float, metadata: Optional[Dict] = None):
        """Store a constellation template"""
        
        with self._template_lock:
            try:
                template_id = f"{content_type.value}_{pattern.value}_{sequence_length}_{len(positions)}_{len(self.templates)}"
                
                template = {
                    'template_id': template_id,
                    'content_type': content_type,
                    'pattern': pattern,
                    'positions': positions,
                    'sequence_length': sequence_length,
                    'num_splats': len(positions),
                    'performance_score': performance_score,
                    'created_timestamp': torch.tensor(0.0),  # Placeholder
                    'usage_count': 0,
                    'metadata': metadata or {}
                }
                
                # Store template
                self.templates[content_type.value][pattern.value].append(template)
                self.template_metadata[template_id] = template
                
                # Update statistics
                self.template_stats['templates_created'] += 1
                
                # Maintain template limit
                self._maintain_template_limit()
                
            except Exception as e:
                logger.warning(f"Failed to store template: {e}")
    
    def _maintain_template_limit(self):
        """Maintain template library within size limits"""
        
        try:
            total_templates = sum(
                len(patterns.values()) for patterns in self.templates.values()
                for pattern_templates in patterns.values()
            )
            
            if total_templates > self.max_templates:
                # Remove lowest performing templates
                all_templates = []
                for content_templates in self.templates.values():
                    for pattern_templates in content_templates.values():
                        all_templates.extend(pattern_templates)
                
                # Sort by performance score (ascending)
                all_templates.sort(key=lambda t: t['performance_score'])
                
                # Remove lowest performing
                to_remove = total_templates - self.max_templates
                for _ in range(to_remove):
                    if all_templates:
                        removed = all_templates.pop(0)
                        
                        # Remove from storage
                        content_key = removed['content_type'].value
                        pattern_key = removed['pattern'].value
                        if removed in self.templates[content_key][pattern_key]:
                            self.templates[content_key][pattern_key].remove(removed)
                        
                        # Clean up metadata
                        if removed['template_id'] in self.template_metadata:
                            del self.template_metadata[removed['template_id']]
                
        except Exception as e:
            logger.warning(f"Template limit maintenance failed: {e}")


# Factory function to create constellation system
def create_constellation_system(model_dim: int, template_config: Optional[Dict] = None,
                               adaptation_config: Optional[Dict] = None):
    """Create constellation template system"""
    
    try:
        template_library = ConstellationTemplateLibrary(
            model_dim=model_dim,
            template_config=template_config
        )
        
        constellation_loader = AdaptiveConstellationLoader(
            template_library=template_library,
            adaptation_config=adaptation_config
        )
        
        return template_library, constellation_loader
        
    except Exception as e:
        logger.error(f"Failed to create constellation system: {e}")
        return None, None


class AdaptiveConstellationLoader(nn.Module):
    """Adaptive loader for constellation templates with real-time optimization"""
    
    def __init__(self, template_library: ConstellationTemplateLibrary,
                 adaptation_config: Optional[Dict] = None):
        super().__init__()
        
        self.template_library = template_library
        self.config = adaptation_config or {}
        
        # Adaptation settings
        self.enable_real_time_adaptation = self.config.get('enable_real_time_adaptation', True)
        self.quality_threshold = self.config.get('quality_threshold', 0.7)
        self.adaptation_strength = self.config.get('adaptation_strength', 0.1)
        
        # Statistics
        self.adaptation_stats = {
            'templates_loaded': 0,
            'adaptations_performed': 0,
            'content_type_matches': defaultdict(int),
            'pattern_usage': defaultdict(int)
        }
        
        logger.info("🔄 Adaptive constellation loader initialized")
    
    def load_optimal_constellation(self, token_embeddings: torch.Tensor,
                                 num_splats: int, sequence_length: int,
                                 prefer_pattern: Optional[ConstellationPattern] = None,
                                 return_metadata: bool = False) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Dict]]:
        """Load optimal constellation for given context"""
        
        metadata = {
            'content_type_detected': None,
            'pattern_used': None,
            'template_source': None,
            'adaptation_applied': False,
            'confidence_score': 0.0
        }
        
        try:
            # Classify content type
            content_type_probs, confidence = self.template_library.content_classifier(token_embeddings)
            detected_content_type_idx = torch.argmax(content_type_probs)
            detected_content_type = list(ContentType)[detected_content_type_idx]
            
            metadata['content_type_detected'] = detected_content_type.value
            metadata['confidence_score'] = confidence.item()
            
            # Use fallback constellation generation for now
            positions = self._generate_simple_fallback(num_splats, sequence_length)
            metadata['template_source'] = 'fallback'
            
            if return_metadata:
                return positions, metadata
            else:
                return positions
                
        except Exception as e:
            logger.warning(f"Constellation loading failed: {e}")
            
            # Fallback to simple generation
            fallback_positions = self._generate_simple_fallback(num_splats, sequence_length)
            metadata['template_source'] = 'fallback'
            metadata['error'] = str(e)
            
            if return_metadata:
                return fallback_positions, metadata
            else:
                return fallback_positions
    
    def _generate_simple_fallback(self, num_splats: int, sequence_length: int) -> List[torch.Tensor]:
        """Generate simple fallback constellation"""
        
        positions = []
        spacing = sequence_length / max(num_splats - 1, 1) if num_splats > 1 else sequence_length
        
        for i in range(num_splats):
            pos = torch.tensor([i * spacing], dtype=torch.float32)
            positions.append(pos)
        
        return positions


def get_constellation_statistics(template_library: ConstellationTemplateLibrary,
                               constellation_loader: AdaptiveConstellationLoader) -> Dict:
    """Get constellation system statistics"""
    
    try:
        stats = {
            'template_library': {
                'total_templates': len(template_library.template_metadata),
                'content_types': len(template_library.templates),
                'max_templates': template_library.max_templates,
                'statistics': template_library.template_stats
            },
            'adaptive_loader': {
                'adaptation_enabled': constellation_loader.enable_real_time_adaptation,
                'quality_threshold': constellation_loader.quality_threshold,
                'statistics': constellation_loader.adaptation_stats
            }
        }
        
        return stats
        
    except Exception as e:
        logger.warning(f"Failed to get constellation statistics: {e}")
        return {'error': str(e)}
