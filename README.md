# Hierarchical Gaussian K-Splat Attention (HSA)

A novel attention mechanism for transformers that enables efficient processing of extremely long contexts while capturing hierarchical structure in data.

## Origin

I've been fascinated by Gaussian splats for use in creating 3D video from static images. (see https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ ) for some time. I'm not an expert or anything, I just like the idea a lot.

Additionally, I've been toying with the idea of neural networks using Gaussian splats for neurons. My idea is kind of based on Conway's game of life, with neurons being born, dying, splitting, and merging to form the final network.

I've also been playing with local LLMs quite a bit, I started to wonder if Gaussian splats might possibly work as a form of attention... perhaps a smaller representation of the attention matrix. So, I began chatting with Claude.AI about the idea.  It might be possible to change O(n^2) attention to O(n*k) ... saving a ton of space on the attention matrix - and allowing longer context.

Another cool idea that happened out of the blue: Because of the mention of Gaussian splats, the AI added some nice visualization tools, which is fun to watch.

Unfortunately, I have no idea what I'm doing - so I set out to try something new. I dislike the term "vibe coding", but that's essentially what I've done.

## Experiment

I started small and focused on an MVP version of the splatting attention, and as each file grew (see initialization, attention, and adaptation), I would split that file into smaller modules... but I had claude do 99% of the coding. What I focussed on was creating tests, and asking for tests, at every step.  There are still some warnings and errors in the current codebase, but I'm learning about how the limits of the tool work.  My goal is to train an LLM on a crappy machine, make sure it's mostly working, then upgrade to a bigger/better machine when I can. For now, I want to keep my tools constrained to force me to focus on efficiency.

Very little double checking has been done here, but that will also be part of the plan - to trim the codebase down while building more and more tests.

## The AI

Below is a summary create by claude to sum up what we've been working on. Yes, I said we. I also say thank you to the AI.

## Theoretical Basis

Hierarchical Gaussian K-Splat Attention (HSA) is a new approach to attention in transformer models that addresses key limitations of standard attention mechanisms:

1. **Quadratic Complexity**: Traditional attention requires computing relationships between all token pairs, resulting in O(n²) complexity which becomes prohibitive for long contexts.

2. **Flat Structure**: Standard attention doesn't explicitly model hierarchical relationships in data (tokens, phrases, sections, documents).

3. **Static Parameters**: Once trained, attention patterns in traditional models remain fixed, regardless of input.

HSA solves these problems by introducing "splats" - Gaussian attention fields in the embedding space that adaptively focus computation where it matters most. Key innovations include:

- **Multi-level hierarchy**: Attention is organized across multiple levels (Token, Phrase, Section, Document), explicitly modeling structure at different scales.

- **Adaptive fields**: Splats can adjust their position, shape, and amplitude during inference through mitosis (splitting) or death (pruning) operations.

- **Information-theoretic foundations**: Adaptation decisions are guided by information-theoretic measures, ensuring changes preserve or enhance representational capacity.

- **Sparse computation**: By focusing computation on relevant token relationships as defined by splats, HSA achieves sub-quadratic complexity.

## Architecture

HSA is organized as a hierarchical system with the following components:

- **Splats**: Core attention units with learnable position, covariance, amplitude and temperature parameters
- **Hierarchy**: Multi-level organization from fine-grained (Token) to coarse (Document) levels
- **Adaptation**: Mechanisms for dynamically adjusting splat parameters, including mitosis and death
- **Information Theory**: Metrics to guide adaptation decisions based on information preservation and gain

The architecture is designed to be a drop-in replacement for standard attention layers in transformer models, with additional benefits that emerge from its hierarchical, adaptive nature.

## Installation

```bash
pip install hsa-attention
```

For development installation:

```bash
git clone https://github.com/bigattichouse/Heirarchical-Guassian-K-Splat-Attention.git
cd hsa-attention
pip install -e .
```

### Requirements

- Python 3.8+
- PyTorch 2.0+ 
- NumPy
- SciPy
- scikit-learn
- Matplotlib (for visualization)

## Quick Start

### Basic Usage

Replace standard attention with HSA in your transformer model:

```python
import torch
from transformers import AutoModel
from hsa_attention import replace_attention_with_hsa

# Load pre-trained model
model = AutoModel.from_pretrained("bert-base-uncased")

# Configure HSA
hsa_config = {
    "hierarchy": {
        "levels": ["Token", "Phrase", "Section", "Document"],
        "init_splats_per_level": [100, 50, 20, 5],
        "level_weights": [0.4, 0.3, 0.2, 0.1]
    },
    "adaptation": {
        "enable_adaptation": True,
        "adaptation_frequency": 5,
        "mitosis_threshold": 0.1,
        "death_threshold": 0.01
    },
    "sparse_topk": 64
}

# Replace attention with HSA
hsa_model = replace_attention_with_hsa(model, hsa_config)

# Use the model as usual
inputs = torch.randint(0, 1000, (1, 512))
outputs = hsa_model(inputs)
```

### Creating an HSA Transformer from Scratch

```python
from hsa_attention import create_hsa_transformer

# Create a transformer model with HSA attention
model = create_hsa_transformer(
    vocab_size=30000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hsa_config=hsa_config
)
```

### Visualization

HSA provides tools to visualize attention patterns and hierarchical structure:

```python
from hsa_attention.visualization import HSAVisualizer

visualizer = HSAVisualizer()
visualizer.visualize_attention_matrix(attention_matrix, tokens)
visualizer.visualize_splat_distribution(splat_registry, tokens)
visualizer.visualize_hierarchy(splat_registry)
visualizer.create_dashboard(splat_registry, attention_matrix, tokens)
```

## Advanced Features

### Adaptation Monitoring

```python
from hsa_attention.adaptation import AdaptationMonitor

monitor = AdaptationMonitor()
adaptation_stats = monitor.get_adaptation_stats()
```

### Information Metrics

```python
from hsa_attention.information_metrics import InformationMetricsTracker

tracker = InformationMetricsTracker()
tracker.compute_all_metrics(splat_registry, tokens, attention_matrix)
info_metrics = tracker.get_top_info_contributors(10)
```

### Long-Context Optimization

```python
# Configure for extremely long contexts
long_context_config = {
    "levels": ["Token", "Phrase", "Section", "Document", "Global"],
    "init_splats_per_level": [200, 100, 50, 20, 5],
    "level_weights": [0.3, 0.25, 0.2, 0.15, 0.1],
    "sparse_topk": 128,
    "use_spatial": True
}

# Create HSA with long-context optimization
from hsa_attention.factory import create_preset_config
config = create_preset_config("long_context", sequence_length=10000)
```

## Expected Benefits

HSA provides several advantages over standard attention:

1. **Improved Efficiency**: Sub-quadratic scaling for long contexts through adaptive sparsity.

2. **Hierarchical Understanding**: Explicit modeling of multi-scale structure improves capture of long-range dependencies.

3. **Adaptive Behavior**: Dynamic adaptation to input allows specialized attention patterns for different content types.

4. **Interpretability**: Visualization of splats and hierarchical relationships provides insights into model reasoning.

5. **Domain Adaptation**: Continuous refinement of attention structure makes HSA particularly well-suited to domain specialization.

## Citation

If you use this work in your research, please cite:

```
@misc{hsa2024,
  author = {Michael E. Johnson},
  title = {Hierarchical Gaussian K-Splat Attention},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/bigattichouse/Heirarchical-Guassian-K-Splat-Attention}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This project was inspired by:
- Transformer models and attention mechanisms (Vaswani et al., 2017)
- Mixture of Experts architectures (Shazeer et al., 2017)
- Information Bottleneck Theory (Tishby et al., 1999)
- Gaussian Mixture Models for spatial representation
