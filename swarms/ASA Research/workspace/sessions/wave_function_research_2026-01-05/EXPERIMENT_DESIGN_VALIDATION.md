# Wave-Function ASA: Validation Experiment Design

**Created**: 2026-01-05
**Researcher**: ASA Research Agent
**Hardware Target**: M2 Mac Mini 8GB
**Status**: EXPERIMENT DESIGN

---

## Executive Summary

This document defines validation experiments for wave-function semantic representations on severely memory-constrained hardware (8GB unified memory). The experiments prioritize:
1. **Memory efficiency** - All experiments must run within ~5GB working memory
2. **Reproducibility** - Deterministic seeds, fixed configurations
3. **Statistical rigor** - Multiple runs, confidence intervals
4. **Incremental validation** - Build from minimal proofs to comprehensive benchmarks

---

## Part 1: Proof-of-Concept Minimal Tests

### 1.1 Experiment POC-1: Complex Embedding Interference

**Objective**: Validate that complex-valued embeddings produce meaningful interference patterns.

**Hypothesis**: Complex dot products with learnable phases will produce constructive interference for semantically similar words and destructive interference for dissimilar words.

**Minimal Implementation**:
```python
# Target: <50MB memory
import torch

class MinimalComplexEmbedding(torch.nn.Module):
    def __init__(self, vocab_size=1000, dim=64):
        super().__init__()
        self.amplitude = torch.nn.Parameter(torch.randn(vocab_size, dim) * 0.1)
        self.phase = torch.nn.Parameter(torch.zeros(vocab_size, dim))

    def forward(self, ids):
        a = self.amplitude[ids]
        φ = self.phase[ids]
        return a * torch.exp(1j * φ)  # Complex tensor

    def similarity(self, z1, z2):
        # Complex inner product - includes interference
        return torch.real(torch.sum(z1.conj() * z2, dim=-1))
```

**Dataset**: 1000-word vocabulary from GloVe (top 1000 by frequency)

**Training Task**: Contrastive learning on word similarity pairs
- Use SimLex-999 word pairs as ground truth
- Positive pairs: similarity > 0.6
- Negative pairs: similarity < 0.3

**Success Criteria**:
- [ ] Memory usage < 100MB during training
- [ ] Phase values diverge from initialization (phases become meaningful)
- [ ] Learned similarity correlation with SimLex-999: Spearman ρ > 0.3
- [ ] Training converges in < 100 epochs

**Failure Criteria**:
- Phase values remain near zero (phase not being used)
- Real-only baseline achieves equivalent performance
- Memory exceeds 200MB

**Estimated Resources**:
- Memory: ~50-80MB
- Time: ~5 minutes training
- Storage: <1MB model checkpoint

---

### 1.2 Experiment POC-2: Superposition Polysemy Resolution

**Objective**: Demonstrate that superposition representations can encode multiple meanings which context disambiguates.

**Hypothesis**: A word in superposition of meaning states can be "collapsed" to correct meaning via projection with context vector.

**Minimal Implementation**:
```python
class PolysemyModel(torch.nn.Module):
    def __init__(self, n_words=100, n_senses=3, dim=32):
        super().__init__()
        # Each word has multiple sense vectors
        self.sense_vectors = torch.nn.Parameter(
            torch.randn(n_words, n_senses, dim) * 0.1
        )
        # Superposition weights (amplitudes)
        self.sense_weights = torch.nn.Parameter(
            torch.ones(n_words, n_senses) / n_senses
        )
        # Context projection
        self.context_proj = torch.nn.Linear(dim, dim)

    def get_superposition(self, word_id):
        """Get word as weighted superposition of senses"""
        weights = torch.softmax(self.sense_weights[word_id], dim=-1)
        senses = self.sense_vectors[word_id]
        return torch.sum(weights.unsqueeze(-1) * senses, dim=-2)

    def resolve_with_context(self, word_id, context_vec):
        """Project superposition onto context-appropriate subspace"""
        senses = self.sense_vectors[word_id]  # [n_senses, dim]
        ctx = self.context_proj(context_vec)   # [dim]
        # Attention over senses based on context
        scores = torch.matmul(senses, ctx)     # [n_senses]
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(weights.unsqueeze(-1) * senses, dim=-2)
```

**Dataset**: Hand-crafted polysemy test set (50 polysemous words)
- "bank" → {financial, river, verb}
- "bat" → {animal, sports}
- "crane" → {bird, machine}
- Each with 10 disambiguating context sentences

**Training Task**: Word sense disambiguation via context matching

**Success Criteria**:
- [ ] Memory usage < 50MB
- [ ] Sense disambiguation accuracy > 80%
- [ ] Superposition representation encodes multiple senses (entropy > 0.5)
- [ ] Context projection selects correct sense

**Failure Criteria**:
- Model collapses to single sense per word
- Random baseline achieves similar accuracy
- Context vectors don't influence sense selection

**Estimated Resources**:
- Memory: ~20-40MB
- Time: ~2 minutes training
- Storage: <500KB model

---

### 1.3 Experiment POC-3: Tensor Product Composition

**Objective**: Validate that tensor product composition produces compositional representations.

**Hypothesis**: Meaning of "adj noun" can be computed via tensor contraction, and this preserves compositional structure better than simple addition.

**Minimal Implementation**:
```python
class TensorComposition(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        # Adjective as linear map (matrix)
        self.adj_tensors = torch.nn.ParameterDict()
        # Nouns as vectors
        self.noun_vectors = torch.nn.ParameterDict()

    def compose_adj_noun(self, adj_id, noun_id):
        """adj ⊗ noun composition via matrix-vector product"""
        adj_mat = self.adj_tensors[adj_id]  # [dim, dim]
        noun_vec = self.noun_vectors[noun_id]  # [dim]
        return torch.matmul(adj_mat, noun_vec)  # [dim]
```

**Dataset**: Mitchell & Lapata 2010 adjective-noun composition dataset
- 324 adjective-noun pairs with human similarity judgments
- e.g., "large child" vs "small child"

**Comparison Baselines**:
1. **Vector addition**: adj + noun
2. **Element-wise multiplication**: adj * noun
3. **Concatenation**: [adj; noun]
4. **Tensor product**: adj @ noun (our method)

**Success Criteria**:
- [ ] Memory usage < 80MB
- [ ] Tensor composition correlation with human judgments > baselines
- [ ] Spearman ρ > 0.4 (literature benchmark ~0.45)

**Failure Criteria**:
- Addition baseline equals or exceeds tensor method
- Correlation < 0.3

**Estimated Resources**:
- Memory: ~30-60MB
- Time: ~10 minutes training
- Storage: <2MB model

---

## Part 2: Comparison Benchmarks vs Standard Attention

### 2.1 Experiment BENCH-1: Minimal Transformer Comparison

**Objective**: Direct comparison of complex-valued attention vs real-valued attention.

**Setup**:
| Component | Real Baseline | Complex Wave |
|-----------|---------------|--------------|
| Embedding dim | 64 | 64 (complex) |
| Attention heads | 2 | 2 |
| Layers | 1 | 1 |
| Parameters | ~50K | ~100K* |
| Memory footprint | ~200KB | ~400KB |

*Complex has 2x params for same dim due to real+imaginary

**Fair Comparison Strategy**:
1. **Parameter-matched**: Complex dim=45 to match ~50K params
2. **Memory-matched**: Complex dim=32 to match memory footprint
3. **Dimension-matched**: Same dim=64, accept param difference

**Complex Attention Implementation**:
```python
class ComplexAttention(torch.nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
        # Complex projections
        self.Wq_r = torch.nn.Linear(dim, dim, bias=False)
        self.Wq_i = torch.nn.Linear(dim, dim, bias=False)
        self.Wk_r = torch.nn.Linear(dim, dim, bias=False)
        self.Wk_i = torch.nn.Linear(dim, dim, bias=False)
        self.Wv_r = torch.nn.Linear(dim, dim, bias=False)
        self.Wv_i = torch.nn.Linear(dim, dim, bias=False)

    def forward(self, x_r, x_i):
        # Complex Q, K, V projections
        q_r = self.Wq_r(x_r) - self.Wq_i(x_i)
        q_i = self.Wq_r(x_i) + self.Wq_i(x_r)
        # ... similar for K, V

        # Complex attention scores
        # Re(q · k*) = qr·kr + qi·ki (captures interference)
        scores = (q_r @ k_r.T + q_i @ k_i.T) / math.sqrt(self.dim)
        attn = torch.softmax(scores, dim=-1)

        # Complex output
        out_r = attn @ v_r
        out_i = attn @ v_i
        return out_r, out_i
```

**Task**: Sequence classification on small text datasets

**Datasets**:
1. **SST-2 subset**: 1000 train, 200 test (sentiment)
2. **TREC-6 subset**: 500 train, 100 test (question type)

**Metrics**:
- Classification accuracy
- Training loss curve
- Attention pattern analysis (entropy, sparsity)
- Inference time per sample

**Success Criteria**:
- [ ] Complex attention ≥ real attention accuracy (within 2%)
- [ ] Complex attention shows different attention patterns
- [ ] Memory overhead < 3x real baseline
- [ ] Training converges in similar epochs

**Failure Criteria**:
- Complex attention > 5% worse than real
- Training is unstable (loss spikes)
- Memory exceeds 3x baseline

---

### 2.2 Experiment BENCH-2: Compositional Generalization

**Objective**: Test if wave-function representations improve compositional generalization.

**Hypothesis**: Quantum-inspired composition should generalize better to novel combinations.

**Dataset**: COGS (Compositional Generalization Challenge Set)
- Subset: 500 training, 100 test (structural generalization split)
- Focus: Novel verb-argument combinations

**Alternative (smaller)**: SCAN dataset
- Primitive → action sequence mapping
- "jump" → JUMP, "jump twice" → JUMP JUMP
- Test: novel combinations like "jump around left"

**Models**:
1. **Baseline LSTM**: Standard encoder-decoder
2. **Baseline Transformer**: 1-layer, 2-head
3. **Wave-Function**: Complex embeddings + tensor composition

**Success Criteria**:
- [ ] Wave-function model > 10% improvement on novel combinations
- [ ] Maintains in-distribution performance
- [ ] Training stability comparable to baselines

---

### 2.3 Experiment BENCH-3: Semantic Similarity Benchmarks

**Objective**: Evaluate wave-function embeddings on standard similarity benchmarks.

**Benchmarks** (small enough for 8GB):
1. **SimLex-999**: 999 word pairs, semantic similarity
2. **WordSim-353**: 353 word pairs, relatedness
3. **MEN**: 3000 pairs, relatedness

**Models**:
1. **Static embeddings**: GloVe 50d (baseline)
2. **Complex static**: Our complex 50d (25 complex dims)
3. **Learned complex**: Trained with contrastive objective

**Metrics**:
- Spearman correlation with human judgments
- Comparison at matched dimensionality

---

## Part 3: Memory Profiling Methodology

### 3.1 M2 Mac Mini 8GB Constraints

**Available Resources**:
- Total RAM: 8GB unified memory
- Usable for ML: ~5-6GB (OS reserves ~2GB)
- Safe working set: **4.5GB maximum**
- Swap: Available but avoid (10-100x slower)

**Memory Budget Allocation**:
| Component | Budget |
|-----------|--------|
| Model parameters | 500MB max |
| Activations (batch) | 1GB max |
| Optimizer states | 1GB max |
| Data loading | 500MB max |
| System overhead | 1.5GB |
| **Total** | **4.5GB** |

### 3.2 Profiling Tools

**Primary**: PyTorch Memory Profiler
```python
import torch
from torch.profiler import profile, ProfilerActivity

def profile_memory(model, input_data):
    torch.mps.empty_cache()  # M2 uses MPS backend

    with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        output = model(input_data)
        loss = output.sum()
        loss.backward()

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage"))
    return prof
```

**MPS-Specific Monitoring**:
```python
def get_mps_memory():
    """Get current MPS memory usage (M2 GPU memory)"""
    if torch.backends.mps.is_available():
        # Note: MPS doesn't have direct memory query yet
        # Use system monitoring instead
        import subprocess
        result = subprocess.run(
            ['memory_pressure'],
            capture_output=True, text=True
        )
        return result.stdout
    return None
```

**Continuous Monitoring Script**:
```bash
#!/bin/bash
# memory_monitor.sh - Run during experiments
while true; do
    echo "$(date): $(memory_pressure | grep 'System-wide memory')"
    sleep 5
done
```

### 3.3 Memory Profiling Protocol

**Pre-Experiment Checklist**:
1. [ ] Close all non-essential applications
2. [ ] Clear Python cache: `torch.mps.empty_cache()`
3. [ ] Record baseline memory: `memory_pressure`
4. [ ] Set PyTorch memory allocator: `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`

**During Experiment**:
1. Log memory every 10 batches
2. Track peak memory per epoch
3. Monitor for swap activity (indicates memory pressure)

**Memory Logging Template**:
```python
class MemoryLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.measurements = []

    def log(self, epoch, batch, phase):
        import psutil
        mem = psutil.Process().memory_info()
        entry = {
            'timestamp': time.time(),
            'epoch': epoch,
            'batch': batch,
            'phase': phase,  # 'forward', 'backward', 'optimizer'
            'rss_mb': mem.rss / 1e6,
            'vms_mb': mem.vms / 1e6,
        }
        self.measurements.append(entry)

    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.measurements, f)
```

### 3.4 Memory Optimization Techniques

**Gradient Checkpointing**:
```python
# Trade compute for memory
from torch.utils.checkpoint import checkpoint

class MemoryEfficientModel(nn.Module):
    def forward(self, x):
        # Checkpoint every layer
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x
```

**Mixed Precision** (where applicable):
```python
# Note: Complex tensors have limited autocast support
# Use manual casting where needed
with torch.autocast(device_type='mps', dtype=torch.float16):
    # Real-valued operations benefit
    pass
```

**Micro-Batching**:
```python
def accumulate_gradients(model, data_loader, accumulation_steps=4):
    optimizer.zero_grad()
    for i, batch in enumerate(data_loader):
        loss = model(batch) / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

---

## Part 4: Success/Failure Criteria

### 4.1 Tier 1: Must-Pass Criteria (Fundamental Validity)

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| **Memory feasibility** | Peak < 5GB | Memory profiler |
| **Training convergence** | Loss decreases monotonically | Training curve |
| **Phase utilization** | Phase variance > 0.1 | Parameter statistics |
| **Complex vs real parity** | Accuracy within 5% | Benchmark scores |

**If Tier 1 fails**: Fundamental approach issues - reconsider architecture.

### 4.2 Tier 2: Should-Pass Criteria (Expected Benefits)

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| **Compositional improvement** | >5% over baseline | COGS/SCAN accuracy |
| **Similarity correlation** | Spearman ρ > 0.35 | SimLex-999 |
| **Polysemy resolution** | >75% accuracy | WSD task |
| **Attention interpretability** | Different patterns | Visual inspection |

**If Tier 2 fails**: Benefits not realized - consider modifications.

### 4.3 Tier 3: Aspirational Criteria (Research Contribution)

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| **SOTA on small benchmark** | Top-3 on any | Benchmark comparison |
| **Novel attention patterns** | Qualitatively distinct | Analysis |
| **Theoretical insight** | Publishable finding | Human judgment |

**If Tier 3 fails**: Still valid research, limited impact.

### 4.4 Automated Testing Framework

```python
class ExperimentValidator:
    def __init__(self, config):
        self.config = config
        self.results = {}

    def validate_tier1(self):
        checks = {
            'memory_under_5gb': self.results['peak_memory_gb'] < 5.0,
            'training_converged': self.results['final_loss'] < self.results['initial_loss'] * 0.5,
            'phase_utilized': self.results['phase_variance'] > 0.1,
            'accuracy_parity': abs(self.results['complex_acc'] - self.results['real_acc']) < 0.05,
        }
        return all(checks.values()), checks

    def validate_tier2(self):
        checks = {
            'compositional_gain': self.results.get('compositional_improvement', 0) > 0.05,
            'similarity_correlation': self.results.get('simlex_spearman', 0) > 0.35,
            'polysemy_accuracy': self.results.get('wsd_accuracy', 0) > 0.75,
        }
        return all(checks.values()), checks

    def generate_report(self):
        tier1_pass, tier1_details = self.validate_tier1()
        tier2_pass, tier2_details = self.validate_tier2()

        return {
            'tier1': {'passed': tier1_pass, 'details': tier1_details},
            'tier2': {'passed': tier2_pass, 'details': tier2_details},
            'recommendation': 'PROCEED' if tier1_pass else 'RECONSIDER'
        }
```

---

## Part 5: Dataset Requirements

### 5.1 Dataset Size Constraints

**Memory Budget for Data**: 500MB maximum

**Implication**: Cannot use full-scale datasets. Must use:
- Subsets of standard benchmarks
- Synthetically generated data
- Pre-tokenized/cached representations

### 5.2 Required Datasets

#### Primary Datasets

| Dataset | Size | Purpose | Source |
|---------|------|---------|--------|
| **GloVe-1K** | ~5MB | Vocabulary embeddings | Subset of GloVe.6B |
| **SimLex-999** | <1MB | Word similarity eval | simlex999.eval.qmul.ac.uk |
| **WordSim-353** | <1MB | Word relatedness eval | alfonseca.org/wordsim353 |
| **SST-2-mini** | ~10MB | Sentiment (1K samples) | Subset of SST-2 |
| **SCAN-simple** | ~5MB | Compositional gen | github.com/brendenlake/SCAN |

#### Synthetic Datasets (Generate On-Demand)

| Dataset | Purpose | Generation Method |
|---------|---------|-------------------|
| **Polysemy-50** | WSD testing | Manual curation |
| **AdjNoun-100** | Composition | Mitchell & Lapata subset |
| **Interference-Test** | Phase validation | Synthetic word pairs |

### 5.3 Dataset Preparation Scripts

```python
# datasets/prepare_all.py

def prepare_glove_subset(n_words=1000, dim=50):
    """Extract top-N GloVe embeddings"""
    # Download if not exists
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    # ... download and extract ...

    embeddings = {}
    with open(f'glove.6B.{dim}d.txt') as f:
        for i, line in enumerate(f):
            if i >= n_words:
                break
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]])
            embeddings[word] = vec

    np.save('glove_1k.npy', embeddings)
    return embeddings

def prepare_simlex():
    """Download and parse SimLex-999"""
    url = "https://fh295.github.io/SimLex-999.zip"
    # ... download and parse ...

    pairs = []
    with open('SimLex-999.txt') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split('\t')
            pairs.append({
                'word1': parts[0],
                'word2': parts[1],
                'similarity': float(parts[3])  # SimLex999 column
            })

    return pairs

def prepare_sst2_mini(n_train=1000, n_test=200):
    """Create small SST-2 subset"""
    from datasets import load_dataset

    dataset = load_dataset('sst2')
    train_subset = dataset['train'].shuffle(seed=42).select(range(n_train))
    test_subset = dataset['validation'].shuffle(seed=42).select(range(n_test))

    return train_subset, test_subset
```

### 5.4 Data Loading Strategy

**Lazy Loading**: Don't load all data into memory
```python
class LazyDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        # Only load index, not data
        with open(file_path + '.idx') as f:
            self.index = json.load(f)

    def __getitem__(self, idx):
        # Load single item on demand
        offset = self.index[idx]
        with open(self.file_path, 'rb') as f:
            f.seek(offset)
            return pickle.load(f)
```

**Pre-tokenization**: Store tokenized versions
```python
def pretokenize_dataset(dataset, tokenizer, output_path):
    """Pre-tokenize and save to disk"""
    tokenized = []
    for item in dataset:
        tokens = tokenizer(item['text'], max_length=64, truncation=True)
        tokenized.append({
            'input_ids': tokens['input_ids'],
            'label': item['label']
        })

    torch.save(tokenized, output_path)
```

---

## Part 6: Experiment Execution Plan

### 6.1 Phase 1: Proof of Concept (Week 1)

**Day 1-2**: POC-1 (Complex Embedding Interference)
- Implement MinimalComplexEmbedding
- Train on word similarity task
- Validate phase utilization

**Day 3-4**: POC-2 (Superposition Polysemy)
- Implement PolysemyModel
- Create Polysemy-50 dataset
- Validate sense disambiguation

**Day 5**: POC-3 (Tensor Composition)
- Implement TensorComposition
- Evaluate on Mitchell & Lapata data
- Compare to baselines

**Checkpoint**: All POC experiments pass Tier 1 criteria

### 6.2 Phase 2: Benchmarks (Week 2)

**Day 6-8**: BENCH-1 (Transformer Comparison)
- Implement ComplexAttention
- Train on SST-2-mini and TREC-6
- Compare accuracy, memory, patterns

**Day 9-10**: BENCH-2 & BENCH-3 (Compositional & Similarity)
- Run SCAN experiments
- Evaluate on SimLex-999, WordSim-353
- Compile comparison tables

**Checkpoint**: Tier 2 criteria evaluation

### 6.3 Phase 3: Analysis & Reporting (Week 3)

**Day 11-12**: Deep Analysis
- Attention pattern visualization
- Phase evolution analysis
- Failure mode investigation

**Day 13-14**: Documentation
- Write results report
- Create visualizations
- Recommendations for next steps

---

## Part 7: Risk Mitigation

### 7.1 Memory Overflow

**Risk**: Experiment exceeds 8GB, system becomes unresponsive.

**Mitigation**:
1. Start with smallest configs, scale up
2. Implement memory watchdog:
```python
def memory_watchdog(threshold_gb=5.0):
    import psutil
    mem = psutil.virtual_memory()
    if mem.used / 1e9 > threshold_gb:
        raise MemoryError(f"Memory usage {mem.used/1e9:.1f}GB exceeds {threshold_gb}GB")
```
3. Use subprocess isolation for risky experiments

### 7.2 Training Instability

**Risk**: Complex gradients cause NaN/Inf.

**Mitigation**:
1. Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
2. Learning rate warmup
3. Wirtinger derivative implementation verification

### 7.3 Negative Results

**Risk**: Wave-function approach shows no benefits.

**Mitigation**:
1. Document negative results (still valuable)
2. Investigate why (dimensionality? task mismatch?)
3. Identify conditions where it might work
4. Pivot to alternative quantum-inspired approaches (density matrices, tensor networks)

---

## Appendix A: Hardware Verification Script

```python
# verify_hardware.py
import torch
import platform

def verify_environment():
    print("=== Hardware Verification ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    # MPS (Metal Performance Shaders) for M2
    print(f"\nMPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    # Memory test
    import psutil
    mem = psutil.virtual_memory()
    print(f"\nTotal RAM: {mem.total / 1e9:.1f} GB")
    print(f"Available: {mem.available / 1e9:.1f} GB")

    # Complex tensor support
    try:
        z = torch.complex(torch.randn(100), torch.randn(100))
        if torch.backends.mps.is_available():
            z_mps = z.to('mps')
            print(f"\nComplex on MPS: SUPPORTED")
        else:
            print(f"\nComplex on MPS: MPS not available")
    except Exception as e:
        print(f"\nComplex on MPS: NOT SUPPORTED ({e})")

    return True

if __name__ == '__main__':
    verify_environment()
```

---

## Appendix B: Experiment Configuration Template

```yaml
# config/experiment_template.yaml

experiment:
  name: "poc-1-interference"
  seed: 42
  device: "mps"  # or "cpu" for safety

model:
  type: "complex_embedding"
  vocab_size: 1000
  embedding_dim: 64
  use_complex: true

training:
  batch_size: 32
  epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  gradient_clip: 1.0

memory:
  max_gb: 4.5
  checkpoint_gradients: false
  mixed_precision: false  # limited complex support

logging:
  log_interval: 10
  save_checkpoints: true
  memory_profiling: true

validation:
  tier1_memory_threshold: 5.0
  tier1_loss_improvement: 0.5
  tier2_accuracy_threshold: 0.75
```

---

## Document Status

**Created**: 2026-01-05
**Status**: READY FOR IMPLEMENTATION
**Next Steps**:
1. Run hardware verification script
2. Prepare datasets
3. Begin POC-1 experiment

---

**Research Note**: All experiments designed for reproducibility. Random seeds fixed at 42. Results should be averaged over 3 runs with different seeds (42, 123, 456) for statistical validity.
