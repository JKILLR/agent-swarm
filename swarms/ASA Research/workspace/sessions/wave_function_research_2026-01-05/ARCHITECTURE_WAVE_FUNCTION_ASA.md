# Wave-Function ASA: Concrete Architecture for Standard GPUs

**Version**: 1.0
**Date**: 2026-01-05
**Status**: DESIGN SPECIFICATION
**Target**: 8GB GPU RAM (RTX 3060/4060 class)

---

## Executive Summary

This document specifies a concrete, implementable architecture for Wave-Function Atomic Semantic Attention (ASA) that runs on consumer GPUs. The design draws from quantum mechanics mathematics while remaining firmly classical in implementation. Key innovations:

1. **Atoms as Probability Distributions** - Tokens exist in superposition until contextual measurement
2. **Pre-computed Orbital Shells** - Semantic structure learned offline, frozen at inference
3. **Physics-Derived Sparse Attention** - Selection rules determine which atoms can interact
4. **8GB Memory Layout** - Careful budget allocation for consumer hardware

---

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [Atom Representation Architecture](#2-atom-representation-architecture)
3. [Pre-computed Orbital Shells](#3-pre-computed-orbital-shells)
4. [Physics-Based Sparse Attention](#4-physics-based-sparse-attention)
5. [Memory Layout for 8GB](#5-memory-layout-for-8gb)
6. [Complete System Architecture](#6-complete-system-architecture)
7. [Implementation Pseudocode](#7-implementation-pseudocode)
8. [Performance Projections](#8-performance-projections)

---

## 1. Core Concepts

### 1.1 The Atomic Metaphor

```
┌─────────────────────────────────────────────────────────────────┐
│                    WAVE-FUNCTION ASA OVERVIEW                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   TOKEN "bank"                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌─────────────────────────────────────────┐                  │
│   │     SUPERPOSITION STATE                  │                  │
│   │                                          │                  │
│   │   |ψ⟩ = α|financial⟩ + β|river⟩ + γ|tilt⟩│                  │
│   │                                          │                  │
│   │   Probability Amplitudes (Complex):      │                  │
│   │   α = 0.7 + 0.2i  →  P = |α|² = 0.53    │                  │
│   │   β = 0.4 - 0.1i  →  P = |β|² = 0.17    │                  │
│   │   γ = 0.3 + 0.3i  →  P = |γ|² = 0.18    │                  │
│   └─────────────────────────────────────────┘                  │
│        │                                                        │
│        │ CONTEXT: "money in the ___"                           │
│        ▼                                                        │
│   ┌─────────────────────────────────────────┐                  │
│   │     MEASUREMENT (COLLAPSE)               │                  │
│   │                                          │                  │
│   │   Context projects onto |financial⟩     │                  │
│   │   Result: Definite meaning vector        │                  │
│   └─────────────────────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Quantum Numbers for Semantics

We define semantic quantum numbers analogous to atomic physics:

| Quantum # | Symbol | Range | Semantic Meaning |
|-----------|--------|-------|------------------|
| Principal | n | 1-4 | Abstraction level (concrete→abstract) |
| Angular | l | 0 to n-1 | Semantic category breadth |
| Magnetic | m | -l to +l | Specific sense within category |
| Spin | s | ±½ | Grammatical polarity (noun/verb) |

**Total basis states**: Σ(n=1 to 4) Σ(l=0 to n-1) (2l+1) × 2 = 2 × (1 + 3 + 5 + 7) = **60 basis states**

---

## 2. Atom Representation Architecture

### 2.1 Complex Amplitude Vector

Each token is represented as a complex vector in superposition over semantic basis states:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ATOM DATA STRUCTURE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  struct SemanticAtom {                                          │
│      // Complex amplitudes for each orbital (60 basis states)  │
│      complex<float16> amplitudes[60];  // 240 bytes            │
│                                                                 │
│      // Phase information for interference                      │
│      float16 global_phase;              // 2 bytes             │
│                                                                 │
│      // Density matrix diagonal (for mixed states)             │
│      float16 populations[60];           // 120 bytes           │
│                                                                 │
│      // Coherence flags (which superpositions are active)      │
│      uint64 coherence_mask;             // 8 bytes             │
│  }                                                              │
│  // Total: 370 bytes per atom                                   │
│                                                                 │
│  // Memory layout (coalesced access):                          │
│  // [amp_real_0...amp_real_59][amp_imag_0...amp_imag_59][...]  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Superposition Until Measurement

```
┌─────────────────────────────────────────────────────────────────┐
│                 SUPERPOSITION LIFECYCLE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PHASE 1: EMBEDDING LOOKUP                                      │
│  ─────────────────────────                                      │
│  token_id → Pre-computed orbital shell → Initial |ψ⟩           │
│                                                                 │
│  |ψ_initial⟩ = Σ_{nlms} c_{nlms} |nlms⟩                        │
│                                                                 │
│  PHASE 2: LAYER PROPAGATION (SUPERPOSITION MAINTAINED)         │
│  ─────────────────────────────────────────────────────         │
│  For each layer:                                                │
│    1. Apply sparse attention (unitary-like transform)          │
│    2. Amplitudes evolve: |ψ'⟩ = U|ψ⟩                           │
│    3. NO collapse - maintain full superposition                 │
│                                                                 │
│  PHASE 3: FINAL MEASUREMENT                                     │
│  ─────────────────────────                                      │
│  At output layer only:                                          │
│    1. Context vector C determines measurement basis             │
│    2. Project: ρ_out = P_C ρ P_C† / Tr(P_C ρ P_C†)             │
│    3. Extract classical output from collapsed state             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Density Matrix for Mixed States

For ambiguous tokens, we use a sparse density matrix representation:

```python
# Density matrix: ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|
# For pure state: ρ = |ψ⟩⟨ψ|

class DensityMatrix:
    """Sparse density matrix for semantic states"""

    # Store only non-zero elements (typically 5-15 per token)
    # Full 60x60 = 3600 elements → Sparse: ~50 elements

    diagonal: float16[60]           # Always stored (populations)
    off_diag_indices: uint16[MAX_COHERENCE]  # (row, col) packed
    off_diag_values: complex16[MAX_COHERENCE]  # Coherences
    n_coherences: uint8             # Number of active coherences

    # MAX_COHERENCE = 30 typical, 60 max
    # Memory: 60×2 + 30×2 + 30×4 + 1 = 301 bytes
```

---

## 3. Pre-computed Orbital Shells

### 3.1 Offline Learning Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│              PRE-COMPUTATION PIPELINE (OFFLINE)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   CORPUS     │───▶│  SEMANTIC    │───▶│   ORBITAL    │      │
│  │  (Training)  │    │  CLUSTERING  │    │  ASSIGNMENT  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────────────────────────────────────────────┐      │
│  │              FROZEN ORBITAL SHELLS                    │      │
│  │                                                       │      │
│  │  vocab_orbitals[50257] = {                           │      │
│  │      token_0: [(n,l,m,s, amplitude), ...],           │      │
│  │      token_1: [(n,l,m,s, amplitude), ...],           │      │
│  │      ...                                              │      │
│  │  }                                                    │      │
│  │                                                       │      │
│  │  Storage: 50K tokens × 370 bytes = 18.5 MB           │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  KEY INSIGHT: No runtime learning of token representations!    │
│  All semantic structure is pre-computed and frozen.            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Orbital Shell Computation Algorithm

```python
def compute_orbital_shells(corpus, vocab, n_orbitals=60):
    """
    Pre-compute semantic orbital assignments for entire vocabulary.
    Run ONCE offline, freeze forever.
    """

    # Step 1: Extract semantic features from corpus
    # Using existing embeddings (word2vec, BERT) as starting point
    base_embeddings = extract_contextual_embeddings(corpus, vocab)

    # Step 2: Cluster into semantic categories (angular momentum l)
    # l=0: Core meaning (1 state)
    # l=1: Primary variations (3 states)
    # l=2: Secondary distinctions (5 states)
    # l=3: Fine-grained senses (7 states)

    category_tree = build_semantic_hierarchy(base_embeddings)

    # Step 3: Assign quantum numbers
    orbital_assignments = {}

    for token_id, embedding in enumerate(base_embeddings):
        # Determine abstraction level (n)
        n = classify_abstraction_level(embedding)  # 1-4

        # Determine category breadth (l)
        l = classify_category_breadth(embedding, category_tree)  # 0 to n-1

        # Determine specific sense (m)
        m_values = classify_senses(token_id, corpus)  # -l to +l

        # Determine grammatical polarity (s)
        s_values = classify_grammatical_role(token_id, corpus)  # ±½

        # Compute complex amplitudes via projection
        amplitudes = project_to_orbital_basis(
            embedding, n, l, m_values, s_values
        )

        orbital_assignments[token_id] = SemanticAtom(
            amplitudes=amplitudes,
            global_phase=0.0,
            populations=np.abs(amplitudes)**2,
            coherence_mask=compute_coherence_mask(amplitudes)
        )

    return orbital_assignments
```

### 3.3 Basis State Definitions

```
┌─────────────────────────────────────────────────────────────────┐
│                 SEMANTIC BASIS STATES                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  n=1 (CONCRETE ENTITIES): 2 states                             │
│  ─────────────────────────────────                             │
│  |1,0,0,+½⟩ = Concrete noun (e.g., "table", "dog")             │
│  |1,0,0,-½⟩ = Concrete verb (e.g., "run", "eat")               │
│                                                                 │
│  n=2 (RELATIONAL): 8 states                                    │
│  ─────────────────────────                                     │
│  |2,0,0,±½⟩ = Core relations                                   │
│  |2,1,-1,±½⟩ = Spatial relations                               │
│  |2,1,0,±½⟩ = Temporal relations                               │
│  |2,1,+1,±½⟩ = Causal relations                                │
│                                                                 │
│  n=3 (ABSTRACT CONCEPTS): 18 states                            │
│  ─────────────────────────────────                             │
│  |3,0,0,±½⟩ = Core abstractions                                │
│  |3,1,m,±½⟩ = Category abstractions (m=-1,0,+1)                │
│  |3,2,m,±½⟩ = Property abstractions (m=-2,-1,0,+1,+2)          │
│                                                                 │
│  n=4 (META-COGNITIVE): 32 states                               │
│  ────────────────────────────                                  │
│  |4,0,0,±½⟩ = Core meta-concepts                               │
│  |4,1,m,±½⟩ = Epistemic modality                               │
│  |4,2,m,±½⟩ = Deontic modality                                 │
│  |4,3,m,±½⟩ = Discourse functions                              │
│                                                                 │
│  TOTAL: 2 + 8 + 18 + 32 = 60 basis states                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Physics-Based Sparse Attention

### 4.1 Selection Rules

Just as atomic transitions follow selection rules (Δl = ±1, Δm = 0, ±1), semantic attention follows analogous constraints:

```
┌─────────────────────────────────────────────────────────────────┐
│                   ATTENTION SELECTION RULES                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  RULE 1: Abstraction Compatibility (Δn)                        │
│  ──────────────────────────────────────                        │
│  |Δn| ≤ 1: Adjacent abstraction levels can interact            │
│  |Δn| > 1: Attention weight = 0 (forbidden transition)         │
│                                                                 │
│  Example: Concrete noun (n=1) CAN attend to relational (n=2)   │
│           Concrete noun (n=1) CANNOT attend to meta (n=4)      │
│                                                                 │
│  RULE 2: Category Compatibility (Δl)                           │
│  ──────────────────────────────────                            │
│  Δl = 0, ±1: Allowed transitions                               │
│  |Δl| > 1: Forbidden (would skip semantic hierarchy)           │
│                                                                 │
│  RULE 3: Sense Compatibility (Δm)                              │
│  ─────────────────────────────────                             │
│  Δm = 0, ±1: Allowed (gradual sense shift)                     │
│  |Δm| > 1: Heavily penalized (large sense jump)                │
│                                                                 │
│  RULE 4: Grammatical Coherence (Δs)                            │
│  ─────────────────────────────────                             │
│  Same s: Strong interaction (noun-noun, verb-verb)             │
│  Different s: Weaker but allowed (noun-verb binding)           │
│                                                                 │
│  COMBINED SPARSITY:                                             │
│  ─────────────────                                              │
│  ~15-25% of attention pairs have non-zero base weight          │
│  After thresholding: ~5-10% active connections                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Sparse Attention Implementation

```python
def compute_selection_rule_mask(atoms_q, atoms_k, n_heads=8):
    """
    Pre-compute sparse attention mask from physics selection rules.

    Args:
        atoms_q: Query atoms [batch, seq_q, 60] quantum numbers
        atoms_k: Key atoms [batch, seq_k, 60] quantum numbers

    Returns:
        mask: Sparse attention mask [batch, heads, seq_q, seq_k]
    """
    batch, seq_q = atoms_q.shape[:2]
    seq_k = atoms_k.shape[1]

    # Extract dominant quantum numbers for each atom
    # (token's most probable orbital)
    n_q, l_q, m_q, s_q = extract_dominant_quantum_numbers(atoms_q)
    n_k, l_k, m_k, s_k = extract_dominant_quantum_numbers(atoms_k)

    # Compute selection rule violations
    delta_n = torch.abs(n_q.unsqueeze(-1) - n_k.unsqueeze(-2))  # [B, Sq, Sk]
    delta_l = torch.abs(l_q.unsqueeze(-1) - l_k.unsqueeze(-2))
    delta_m = torch.abs(m_q.unsqueeze(-1) - m_k.unsqueeze(-2))
    delta_s = (s_q.unsqueeze(-1) != s_k.unsqueeze(-2)).float()

    # Selection rule weights
    allowed = (
        (delta_n <= 1) &  # Abstraction compatibility
        (delta_l <= 1) &  # Category compatibility
        (delta_m <= 1)    # Sense compatibility
    ).float()

    # Soft penalties for near-violations
    penalty = (
        0.1 * delta_n.clamp(max=2) +
        0.2 * delta_l.clamp(max=2) +
        0.1 * delta_m.clamp(max=2) +
        0.05 * delta_s
    )

    # Final mask: allowed × exp(-penalty)
    mask = allowed * torch.exp(-penalty)

    # Different heads can have different selection rule relaxations
    head_masks = []
    for h in range(n_heads):
        relaxation = 0.1 * h  # Heads progressively more permissive
        head_mask = allowed * torch.exp(-penalty * (1 - relaxation))
        head_masks.append(head_mask)

    return torch.stack(head_masks, dim=1)  # [B, H, Sq, Sk]
```

### 4.3 Sparse Attention Kernel

```
┌─────────────────────────────────────────────────────────────────┐
│              SPARSE ATTENTION COMPUTATION                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Standard Attention: O(n²) memory, O(n²d) compute              │
│  Sparse Attention:   O(kn) memory, O(knd) compute where k<<n   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  SPARSE ATTENTION KERNEL                             │       │
│  │                                                      │       │
│  │  Input:                                              │       │
│  │    Q: [batch, seq, heads, d_k] complex              │       │
│  │    K: [batch, seq, heads, d_k] complex              │       │
│  │    V: [batch, seq, heads, d_v] complex              │       │
│  │    mask: sparse [batch, heads, seq, seq]            │       │
│  │                                                      │       │
│  │  Algorithm:                                          │       │
│  │    1. For each query position q:                    │       │
│  │       a. Retrieve non-zero mask indices for q       │       │
│  │       b. Compute QK* only for allowed positions     │       │
│  │       c. Apply complex softmax:                     │       │
│  │          α = softmax(Re(QK*) / √d)                  │       │
│  │       d. Compute weighted sum: out = Σ α_i V_i      │       │
│  │                                                      │       │
│  │  Output: [batch, seq, heads, d_v] complex           │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  SPARSITY BENEFIT:                                              │
│  With 10% density, 2048 context:                               │
│  Dense: 2048² = 4.2M operations per head                       │
│  Sparse: 2048 × 205 = 420K operations per head (10× savings)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Interference in Attention

```python
def complex_attention(Q, K, V, mask):
    """
    Complex-valued attention with interference effects.

    The key insight: complex dot product Q·K* gives:
    - Real part: Constructive/destructive interference
    - Imaginary part: Phase relationship

    We use the real part for attention weights (interference)
    and the imaginary part modulates the value aggregation.
    """
    d_k = Q.shape[-1]

    # Complex dot product: (a+bi)(c-di) = (ac+bd) + (bc-ad)i
    # Q: [B, Sq, H, D] complex
    # K: [B, Sk, H, D] complex

    # Compute Q @ K^H (Hermitian transpose)
    QK_star = torch.einsum('bqhd,bkhd->bhqk', Q, K.conj())

    # Real part determines attention strength (interference)
    attention_logits = QK_star.real / math.sqrt(d_k)

    # Apply physics-based sparse mask
    attention_logits = attention_logits.masked_fill(mask == 0, float('-inf'))
    attention_logits = attention_logits + torch.log(mask + 1e-8)

    # Softmax over keys
    attention_weights = F.softmax(attention_logits, dim=-1)  # Real

    # Phase modulation from imaginary part
    phase_modulation = torch.tanh(QK_star.imag / math.sqrt(d_k))

    # Complex value aggregation
    # V: [B, Sk, H, D] complex
    # Modulate values by phase before aggregation
    V_modulated = V * torch.exp(1j * phase_modulation.unsqueeze(-1))

    # Weighted sum
    output = torch.einsum('bhqk,bkhd->bqhd', attention_weights, V_modulated)

    return output
```

---

## 5. Memory Layout for 8GB

### 5.1 Memory Budget

```
┌─────────────────────────────────────────────────────────────────┐
│                   8GB GPU MEMORY BUDGET                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  COMPONENT                          SIZE      CUMULATIVE        │
│  ─────────────────────────────────────────────────────────     │
│                                                                 │
│  1. FROZEN ORBITAL SHELLS                                      │
│     50,257 tokens × 370 bytes      18.6 MB    18.6 MB          │
│                                                                 │
│  2. MODEL WEIGHTS (6 layers)                                   │
│     Attention (Q,K,V,O):                                       │
│       4 × 60 × 256 × 2 × 6        737 KB                       │
│     FFN (up/down projection):                                  │
│       2 × 256 × 1024 × 2 × 6      6.3 MB                       │
│     LayerNorm:                                                 │
│       2 × 256 × 2 × 6             6 KB                         │
│     TOTAL WEIGHTS:                 7.0 MB     25.6 MB          │
│                                                                 │
│  3. KV CACHE (context=2048)                                    │
│     2048 × 6 layers × 2(K,V) ×                                 │
│     8 heads × 32 dim × 4 bytes    6.3 MB     31.9 MB          │
│                                                                 │
│  4. ACTIVATION MEMORY (batch=1)                                │
│     Input atoms: 2048 × 370       760 KB                       │
│     Layer activations: 2048 ×                                  │
│       256 × 4 × 6                 12.6 MB                      │
│     Attention intermediates:                                   │
│       8 heads × 2048² × 0.1 ×    6.7 MB                       │
│       4 bytes (sparse 10%)                                     │
│     TOTAL ACTIVATIONS:            20.0 MB    51.9 MB          │
│                                                                 │
│  5. SPARSE MASK STORAGE                                        │
│     CSR format for 2048² × 10%:  1.7 MB     53.6 MB           │
│                                                                 │
│  6. GRADIENT BUFFER (training)                                 │
│     Mirror of weights:            7.0 MB     60.6 MB          │
│                                                                 │
│  7. OPTIMIZER STATE (AdamW)                                    │
│     2 × weights:                  14.0 MB    74.6 MB          │
│                                                                 │
│  8. SAFETY MARGIN                  ~50 MB     ~125 MB         │
│                                                                 │
│  ═════════════════════════════════════════════════════════     │
│  TOTAL TRAINING:                  ~125 MB (1.5% of 8GB)        │
│  TOTAL INFERENCE:                 ~54 MB  (0.7% of 8GB)        │
│                                                                 │
│  REMAINING FOR LARGER MODELS:     ~7.9 GB                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Scaling to Production Size

```
┌─────────────────────────────────────────────────────────────────┐
│                PRODUCTION MODEL SCALING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Configuration Options for 8GB GPU:                            │
│                                                                 │
│  SMALL (Research/Prototype):                                   │
│  ────────────────────────────                                  │
│  - Layers: 6                                                   │
│  - Hidden: 256                                                 │
│  - Heads: 8                                                    │
│  - Context: 2048                                               │
│  - Params: ~2M                                                 │
│  - Memory: ~125 MB training                                    │
│                                                                 │
│  MEDIUM (Production):                                          │
│  ────────────────────                                          │
│  - Layers: 12                                                  │
│  - Hidden: 512                                                 │
│  - Heads: 8                                                    │
│  - Context: 4096                                               │
│  - Params: ~25M                                                │
│  - Memory: ~1.2 GB training                                    │
│                                                                 │
│  LARGE (Max for 8GB):                                          │
│  ─────────────────────                                         │
│  - Layers: 24                                                  │
│  - Hidden: 768                                                 │
│  - Heads: 12                                                   │
│  - Context: 4096                                               │
│  - Params: ~85M                                                │
│  - Memory: ~5.5 GB training                                    │
│  - Note: Requires gradient checkpointing                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Memory-Efficient Data Structures

```python
class MemoryEfficientOrbitalCache:
    """
    Memory layout optimized for GPU access patterns.
    """

    def __init__(self, vocab_size=50257, n_orbitals=60):
        # Store amplitudes in Structure of Arrays (SoA) format
        # for coalesced GPU memory access

        # Real parts: [vocab_size, n_orbitals] float16
        self.amp_real = torch.zeros(vocab_size, n_orbitals, dtype=torch.float16)

        # Imaginary parts: [vocab_size, n_orbitals] float16
        self.amp_imag = torch.zeros(vocab_size, n_orbitals, dtype=torch.float16)

        # Populations (|amplitude|²): [vocab_size, n_orbitals] float16
        # Pre-computed to avoid runtime complex magnitude
        self.populations = torch.zeros(vocab_size, n_orbitals, dtype=torch.float16)

        # Dominant orbital index per token: [vocab_size] uint8
        # For fast selection rule computation
        self.dominant_orbital = torch.zeros(vocab_size, dtype=torch.uint8)

        # Sparsity mask: [vocab_size, n_orbitals] bit-packed
        # Which orbitals are non-negligible
        self.orbital_mask = torch.zeros(vocab_size, (n_orbitals + 63) // 64,
                                        dtype=torch.int64)

    def lookup(self, token_ids):
        """
        Efficient batched lookup.
        Returns complex tensor [batch, n_orbitals]
        """
        real = self.amp_real[token_ids]
        imag = self.amp_imag[token_ids]
        return torch.complex(real, imag)

    def memory_footprint(self):
        """Total memory in bytes"""
        return (
            self.amp_real.numel() * 2 +      # float16
            self.amp_imag.numel() * 2 +      # float16
            self.populations.numel() * 2 +   # float16
            self.dominant_orbital.numel() +  # uint8
            self.orbital_mask.numel() * 8    # int64
        )
        # = 50257 × (60×2 + 60×2 + 60×2 + 1 + 1×8)
        # = 50257 × 369 = 18.5 MB
```

### 5.4 Sparse Attention Storage

```python
class SparseAttentionMask:
    """
    CSR (Compressed Sparse Row) format for attention masks.

    For 10% density on 2048×2048:
    - Dense: 4M × 4 bytes = 16 MB
    - CSR: 400K values + 400K indices + 2K row_ptrs = 3.2 MB
    """

    def __init__(self, max_seq_len=2048, density=0.1):
        max_nnz = int(max_seq_len * max_seq_len * density)

        # CSR components
        self.values = torch.zeros(max_nnz, dtype=torch.float16)
        self.col_indices = torch.zeros(max_nnz, dtype=torch.int16)
        self.row_ptrs = torch.zeros(max_seq_len + 1, dtype=torch.int32)

        # Actual number of non-zeros
        self.nnz = 0

    def from_dense(self, dense_mask, threshold=0.01):
        """Convert dense mask to CSR format"""
        # Find non-zero positions
        rows, cols = torch.where(dense_mask > threshold)
        self.nnz = len(rows)

        # Store values and column indices
        self.values[:self.nnz] = dense_mask[rows, cols].half()
        self.col_indices[:self.nnz] = cols.short()

        # Compute row pointers
        for i in range(dense_mask.shape[0] + 1):
            self.row_ptrs[i] = (rows < i).sum()

    def memory_footprint(self):
        return (
            self.nnz * 2 +           # values (float16)
            self.nnz * 2 +           # col_indices (int16)
            len(self.row_ptrs) * 4   # row_ptrs (int32)
        )
```

---

## 6. Complete System Architecture

### 6.1 Full Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 WAVE-FUNCTION ASA MODEL                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: token_ids [batch, seq_len]                             │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  ORBITAL LOOKUP (Frozen)                                 │   │
│  │  token_ids → pre-computed orbital shells                 │   │
│  │  Output: atoms [batch, seq, 60] complex                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SELECTION RULE MASK GENERATOR                           │   │
│  │  atoms → sparse attention mask per head                  │   │
│  │  Output: mask [batch, heads, seq, seq] sparse            │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  ╔═════════════════════════════════════════════════════════╗   │
│  ║  WAVE-FUNCTION TRANSFORMER BLOCK (×N layers)           ║   │
│  ║                                                         ║   │
│  ║  ┌───────────────────────────────────────────────────┐ ║   │
│  ║  │  Complex Layer Norm                               │ ║   │
│  ║  │  (Separate real/imag normalization)               │ ║   │
│  ║  └───────────────────────────────────────────────────┘ ║   │
│  ║         │                                               ║   │
│  ║         ▼                                               ║   │
│  ║  ┌───────────────────────────────────────────────────┐ ║   │
│  ║  │  Sparse Complex Multi-Head Attention              │ ║   │
│  ║  │  - Q,K,V projections (complex linear)             │ ║   │
│  ║  │  - Interference-based attention weights           │ ║   │
│  ║  │  - Physics selection rule masking                 │ ║   │
│  ║  │  - Phase-modulated value aggregation              │ ║   │
│  ║  └───────────────────────────────────────────────────┘ ║   │
│  ║         │                                               ║   │
│  ║         ▼                                               ║   │
│  ║  ┌───────────────────────────────────────────────────┐ ║   │
│  ║  │  Complex FFN                                      │ ║   │
│  ║  │  - Up projection (complex linear)                 │ ║   │
│  ║  │  - modReLU activation                             │ ║   │
│  ║  │  - Down projection (complex linear)               │ ║   │
│  ║  └───────────────────────────────────────────────────┘ ║   │
│  ║         │                                               ║   │
│  ╚═════════╪═══════════════════════════════════════════════╝   │
│            │                                                    │
│            ▼                                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MEASUREMENT LAYER (Collapse)                           │   │
│  │  - Project superposition to output space                │   │
│  │  - Take magnitude for real-valued logits                │   │
│  │  Output: logits [batch, seq, vocab_size] real           │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                                                       │
│         ▼                                                       │
│  OUTPUT: next token probabilities                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Layer-by-Layer Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA FLOW DIAGRAM                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 0 (Embedding):                                          │
│  ────────────────────                                          │
│  tokens [B, S]                                                 │
│     │                                                           │
│     ├──▶ orbital_lookup ──▶ atoms [B, S, 60] ℂ                 │
│     │                                                           │
│     └──▶ position_phase ──▶ phase [B, S, 60] ℝ                 │
│                │                                                │
│                ▼                                                │
│  atoms × exp(i × phase) ──▶ positioned_atoms [B, S, 60] ℂ      │
│                                                                 │
│  Layer 1-N (Transformer):                                      │
│  ────────────────────────                                      │
│  positioned_atoms [B, S, 60] ℂ                                 │
│     │                                                           │
│     ├──▶ project_to_hidden ──▶ h [B, S, D] ℂ                   │
│     │                                                           │
│     ├──▶ Q_proj ──▶ Q [B, S, H, D/H] ℂ                         │
│     ├──▶ K_proj ──▶ K [B, S, H, D/H] ℂ                         │
│     ├──▶ V_proj ──▶ V [B, S, H, D/H] ℂ                         │
│     │                                                           │
│     ├──▶ selection_mask ──▶ mask [B, H, S, S] sparse           │
│     │                                                           │
│     ├──▶ sparse_complex_attn(Q,K,V,mask) ──▶ attn [B,S,H,D/H]ℂ│
│     │                                                           │
│     └──▶ ffn ──▶ out [B, S, D] ℂ                               │
│                                                                 │
│  Layer N+1 (Output):                                           │
│  ───────────────────                                           │
│  final_hidden [B, S, D] ℂ                                      │
│     │                                                           │
│     ├──▶ project_to_orbitals ──▶ orbital_logits [B, S, 60] ℂ   │
│     │                                                           │
│     ├──▶ |orbital_logits|² ──▶ probabilities [B, S, 60] ℝ      │
│     │                                                           │
│     └──▶ orbital_to_vocab ──▶ vocab_logits [B, S, V] ℝ         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Pseudocode

### 7.1 Core Model Class

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

@dataclass
class WaveFunctionASAConfig:
    vocab_size: int = 50257
    n_orbitals: int = 60
    hidden_size: int = 256
    n_layers: int = 6
    n_heads: int = 8
    max_seq_len: int = 2048
    attention_density: float = 0.1
    dropout: float = 0.1


class WaveFunctionASA(nn.Module):
    """
    Wave-Function Atomic Semantic Attention model.

    Key features:
    - Pre-computed orbital embeddings (frozen)
    - Complex-valued representations throughout
    - Physics-based sparse attention
    - Measurement collapse at output
    """

    def __init__(self, config: WaveFunctionASAConfig,
                 orbital_cache: MemoryEfficientOrbitalCache):
        super().__init__()
        self.config = config

        # Frozen orbital embeddings
        self.orbital_cache = orbital_cache
        self.orbital_cache.requires_grad_(False)

        # Orbital to hidden projection (complex)
        self.orbital_proj = ComplexLinear(config.n_orbitals, config.hidden_size)

        # Position encoding via phase rotation
        self.position_phases = nn.Parameter(
            torch.zeros(config.max_seq_len, config.n_orbitals)
        )

        # Transformer layers
        self.layers = nn.ModuleList([
            WaveFunctionTransformerBlock(config)
            for _ in range(config.n_layers)
        ])

        # Output projection (collapse measurement)
        self.output_proj = ComplexLinear(config.hidden_size, config.n_orbitals)
        self.vocab_decoder = nn.Linear(config.n_orbitals, config.vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through wave-function model.

        Args:
            token_ids: [batch, seq_len] token indices

        Returns:
            logits: [batch, seq_len, vocab_size] next-token logits
        """
        batch, seq_len = token_ids.shape

        # Step 1: Orbital lookup (frozen, no gradients)
        with torch.no_grad():
            atoms = self.orbital_cache.lookup(token_ids)  # [B, S, 60] complex

        # Step 2: Apply position-dependent phase rotation
        phases = self.position_phases[:seq_len]  # [S, 60]
        atoms = atoms * torch.exp(1j * phases)   # [B, S, 60] complex

        # Step 3: Project to hidden dimension
        hidden = self.orbital_proj(atoms)  # [B, S, D] complex

        # Step 4: Compute selection rule mask
        selection_mask = self._compute_selection_mask(atoms)  # [B, H, S, S]

        # Step 5: Transform through layers (maintaining superposition)
        for layer in self.layers:
            hidden = layer(hidden, selection_mask)

        # Step 6: Measurement (collapse to output)
        orbital_amplitudes = self.output_proj(hidden)  # [B, S, 60] complex

        # Take squared magnitude (Born rule)
        orbital_probs = (orbital_amplitudes.abs() ** 2)  # [B, S, 60] real

        # Map to vocabulary
        logits = self.vocab_decoder(orbital_probs)  # [B, S, V] real

        return logits

    def _compute_selection_mask(self, atoms: torch.Tensor) -> torch.Tensor:
        """Compute physics-based sparse attention mask."""
        return compute_selection_rule_mask(
            atoms, atoms,
            n_heads=self.config.n_heads,
            density=self.config.attention_density
        )


class ComplexLinear(nn.Module):
    """Complex-valued linear layer."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Separate real and imaginary weight matrices
        self.W_real = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.W_imag = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complex matrix multiply: (W_r + iW_i)(x_r + ix_i)
        = (W_r x_r - W_i x_i) + i(W_r x_i + W_i x_r)
        """
        x_real, x_imag = x.real, x.imag

        out_real = F.linear(x_real, self.W_real) - F.linear(x_imag, self.W_imag)
        out_imag = F.linear(x_real, self.W_imag) + F.linear(x_imag, self.W_real)

        return torch.complex(out_real, out_imag) + self.bias
```

### 7.2 Transformer Block

```python
class WaveFunctionTransformerBlock(nn.Module):
    """Single transformer block with wave-function modifications."""

    def __init__(self, config: WaveFunctionASAConfig):
        super().__init__()
        self.config = config

        # Pre-norm architecture
        self.norm1 = ComplexLayerNorm(config.hidden_size)
        self.norm2 = ComplexLayerNorm(config.hidden_size)

        # Multi-head attention
        self.attention = SparseComplexAttention(
            hidden_size=config.hidden_size,
            n_heads=config.n_heads,
            dropout=config.dropout
        )

        # FFN
        self.ffn = ComplexFFN(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * 4,
            dropout=config.dropout
        )

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        # Attention with residual
        normed = self.norm1(x)
        attn_out = self.attention(normed, normed, normed, attention_mask)
        x = x + attn_out

        # FFN with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x


class ComplexLayerNorm(nn.Module):
    """Layer normalization for complex tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize real and imaginary parts separately
        # then recombine

        # Compute variance of magnitude
        mag = x.abs()
        mean_mag = mag.mean(dim=-1, keepdim=True)
        var_mag = ((mag - mean_mag) ** 2).mean(dim=-1, keepdim=True)

        # Normalize
        x_normed = x / (torch.sqrt(var_mag + self.eps))

        # Apply learnable affine transform
        return x_normed * self.gamma + self.beta


class SparseComplexAttention(nn.Module):
    """Multi-head attention with complex values and sparse masking."""

    def __init__(self, hidden_size: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.Q_proj = ComplexLinear(hidden_size, hidden_size)
        self.K_proj = ComplexLinear(hidden_size, hidden_size)
        self.V_proj = ComplexLinear(hidden_size, hidden_size)
        self.O_proj = ComplexLinear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = query.shape

        # Project
        Q = self.Q_proj(query).view(batch, seq_len, self.n_heads, self.head_dim)
        K = self.K_proj(key).view(batch, seq_len, self.n_heads, self.head_dim)
        V = self.V_proj(value).view(batch, seq_len, self.n_heads, self.head_dim)

        # Complex attention with interference
        attn_out = complex_attention(Q, K, V, mask)

        # Reshape and project
        attn_out = attn_out.view(batch, seq_len, -1)
        return self.O_proj(attn_out)


class ComplexFFN(nn.Module):
    """Feed-forward network with modReLU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float):
        super().__init__()
        self.up_proj = ComplexLinear(hidden_size, intermediate_size)
        self.down_proj = ComplexLinear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up_proj(x)
        h = mod_relu(h)
        h = self.dropout(h)
        return self.down_proj(h)


def mod_relu(z: torch.Tensor, bias: float = 0.0) -> torch.Tensor:
    """
    modReLU activation for complex numbers.
    modReLU(z) = ReLU(|z| + b) * z/|z| if |z| + b > 0, else 0
    """
    magnitude = z.abs()
    phase = z / (magnitude + 1e-8)
    activated_magnitude = F.relu(magnitude + bias)
    return activated_magnitude * phase
```

### 7.3 Orbital Pre-computation

```python
def precompute_orbital_shells(
    corpus_path: str,
    vocab_path: str,
    output_path: str,
    base_embedding_model: str = "bert-base-uncased"
) -> MemoryEfficientOrbitalCache:
    """
    Offline pre-computation of orbital shells for entire vocabulary.

    This runs ONCE and produces frozen embeddings.
    """
    from transformers import AutoModel, AutoTokenizer

    # Load base model for initial embeddings
    tokenizer = AutoTokenizer.from_pretrained(base_embedding_model)
    model = AutoModel.from_pretrained(base_embedding_model)
    vocab_size = tokenizer.vocab_size

    print(f"Pre-computing orbital shells for {vocab_size} tokens...")

    # Initialize orbital cache
    cache = MemoryEfficientOrbitalCache(vocab_size=vocab_size, n_orbitals=60)

    # Step 1: Extract base embeddings from corpus
    print("Step 1: Extracting base embeddings...")
    base_embeddings = extract_token_embeddings(model, tokenizer, corpus_path)

    # Step 2: Build semantic hierarchy (determines l quantum number)
    print("Step 2: Building semantic hierarchy...")
    hierarchy = build_semantic_hierarchy(base_embeddings)

    # Step 3: Cluster abstraction levels (determines n quantum number)
    print("Step 3: Clustering abstraction levels...")
    abstraction_levels = cluster_abstraction_levels(base_embeddings, hierarchy)

    # Step 4: Assign quantum numbers to each token
    print("Step 4: Assigning quantum numbers...")
    for token_id in tqdm(range(vocab_size)):
        n = abstraction_levels[token_id]  # 1-4
        l = hierarchy.get_category_level(token_id)  # 0 to n-1

        # Get sense distribution (m values)
        sense_dist = get_polysemy_distribution(token_id, corpus_path)

        # Get grammatical role distribution (s values)
        grammar_dist = get_grammatical_distribution(token_id, corpus_path)

        # Compute complex amplitudes
        amplitudes = compute_orbital_amplitudes(
            base_embeddings[token_id],
            n, l, sense_dist, grammar_dist
        )

        # Store in cache
        cache.amp_real[token_id] = amplitudes.real.half()
        cache.amp_imag[token_id] = amplitudes.imag.half()
        cache.populations[token_id] = (amplitudes.abs() ** 2).half()
        cache.dominant_orbital[token_id] = amplitudes.abs().argmax().byte()

    # Save to disk
    print(f"Saving to {output_path}...")
    torch.save({
        'amp_real': cache.amp_real,
        'amp_imag': cache.amp_imag,
        'populations': cache.populations,
        'dominant_orbital': cache.dominant_orbital,
        'orbital_mask': cache.orbital_mask
    }, output_path)

    print(f"Done! Cache size: {cache.memory_footprint() / 1e6:.2f} MB")
    return cache


def compute_orbital_amplitudes(
    base_embedding: torch.Tensor,
    n: int, l: int,
    sense_dist: dict,
    grammar_dist: dict
) -> torch.Tensor:
    """
    Project base embedding onto orbital basis with quantum numbers.

    Returns complex amplitude vector [60].
    """
    amplitudes = torch.zeros(60, dtype=torch.cfloat)

    # Iterate over allowed m values for this l
    for m in range(-l, l + 1):
        # Iterate over spin values
        for s_idx, s in enumerate([0.5, -0.5]):
            # Compute orbital index
            orbital_idx = get_orbital_index(n, l, m, s)

            # Amplitude magnitude from sense and grammar distributions
            sense_weight = sense_dist.get(m, 0.1)
            grammar_weight = grammar_dist.get(s, 0.5)
            magnitude = math.sqrt(sense_weight * grammar_weight)

            # Phase from base embedding projection
            # Use spherical harmonics-like projection
            phase = compute_phase_from_embedding(base_embedding, n, l, m)

            amplitudes[orbital_idx] = magnitude * torch.exp(1j * phase)

    # Normalize to unit sum of probabilities
    amplitudes = amplitudes / amplitudes.abs().sum().sqrt()

    return amplitudes


def get_orbital_index(n: int, l: int, m: int, s: float) -> int:
    """Map quantum numbers to flat orbital index."""
    # n=1: indices 0-1 (2 states)
    # n=2: indices 2-9 (8 states)
    # n=3: indices 10-27 (18 states)
    # n=4: indices 28-59 (32 states)

    base = 0
    for n_prev in range(1, n):
        for l_prev in range(n_prev):
            base += (2 * l_prev + 1) * 2  # 2 for spin

    for l_prev in range(l):
        base += (2 * l_prev + 1) * 2

    m_offset = m + l  # m ranges from -l to +l
    s_offset = 0 if s > 0 else 1

    return base + m_offset * 2 + s_offset
```

---

## 8. Performance Projections

### 8.1 Theoretical Analysis

```
┌─────────────────────────────────────────────────────────────────┐
│              PERFORMANCE COMPARISON                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  OPERATION          │ Standard Transformer │ Wave-Function ASA │
│  ───────────────────┼──────────────────────┼──────────────────  │
│  Embedding lookup   │ O(1) per token       │ O(1) per token    │
│  Attention compute  │ O(n² × d)            │ O(n × k × d)      │
│  Attention memory   │ O(n²)                │ O(n × k)          │
│  FFN forward        │ O(n × d²)            │ O(n × 4d²) [2×]   │
│  Total FLOPs        │ O(n² × d)            │ O(n × k × d)      │
│                                                                 │
│  Where:                                                         │
│  - n = sequence length (2048)                                  │
│  - d = hidden dimension (256)                                  │
│  - k = average attention connections per token (~200 at 10%)   │
│                                                                 │
│  SPEEDUP ANALYSIS (seq_len=2048, d=256):                       │
│  ─────────────────────────────────────────                     │
│  Standard attention: 2048² × 256 = 1.07B FLOPs                 │
│  Sparse attention:   2048 × 200 × 256 = 105M FLOPs             │
│  Attention speedup:  ~10× (limited by density parameter)       │
│                                                                 │
│  Complex arithmetic overhead: ~2× per operation                │
│  Net speedup: ~5× for attention-bound workloads                │
│                                                                 │
│  MEMORY COMPARISON (seq_len=2048):                             │
│  ─────────────────────────────────                             │
│  Standard: 2048² × 4 bytes = 16.8 MB per layer                 │
│  Sparse:   2048 × 200 × 4 bytes = 1.6 MB per layer             │
│  Memory savings: ~10×                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Benchmarking Plan

```python
def benchmark_wave_function_asa():
    """
    Benchmark suite for Wave-Function ASA.
    """
    benchmarks = {
        'throughput': {
            'description': 'Tokens per second on inference',
            'target': '1000+ tok/s on RTX 3060',
            'method': 'time 1000 forward passes, average'
        },
        'memory_efficiency': {
            'description': 'Peak GPU memory usage',
            'target': '<2GB for context=2048',
            'method': 'torch.cuda.max_memory_allocated()'
        },
        'compositional_generalization': {
            'description': 'Performance on SCAN/COGS benchmarks',
            'target': '>90% on simple compositions',
            'method': 'Standard benchmark evaluation'
        },
        'polysemy_resolution': {
            'description': 'WSD accuracy on ambiguous words',
            'target': '>85% on SemCor',
            'method': 'Compare collapsed state to gold sense'
        },
        'interference_effects': {
            'description': 'Verify interference improves similarity',
            'target': 'Constructive interference for synonyms',
            'method': 'Measure attention pattern correlations'
        }
    }
    return benchmarks
```

### 8.3 Expected Limitations

```
┌─────────────────────────────────────────────────────────────────┐
│                   LIMITATIONS & MITIGATIONS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  LIMITATION 1: Complex Arithmetic Overhead                     │
│  ────────────────────────────────────────                      │
│  Issue: Complex ops are ~2-4× slower than real                 │
│  Mitigation: Use float16, fused kernels, structural sparsity  │
│                                                                 │
│  LIMITATION 2: Fixed Orbital Basis                             │
│  ─────────────────────────────────                             │
│  Issue: 60 orbitals may not capture all semantic nuance       │
│  Mitigation: Hierarchical basis, residual connections to       │
│              learned embeddings                                 │
│                                                                 │
│  LIMITATION 3: Selection Rule Rigidity                         │
│  ─────────────────────────────────────                         │
│  Issue: Physics rules may not match linguistic patterns        │
│  Mitigation: Learned relaxation parameters per head,           │
│              multiple heads with different strictness          │
│                                                                 │
│  LIMITATION 4: Pre-computation Requirements                    │
│  ──────────────────────────────────────────                    │
│  Issue: Needs large corpus for good orbital assignment         │
│  Mitigation: Use pre-trained embeddings as initialization,    │
│              fine-tune orbital populations end-to-end          │
│                                                                 │
│  LIMITATION 5: Novel Token Handling                            │
│  ──────────────────────────────────                            │
│  Issue: OOV tokens have no pre-computed orbitals              │
│  Mitigation: Character-level fallback, interpolation from     │
│              nearest vocabulary neighbors                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Mathematical Details

### A.1 Spherical Harmonics Reference

```
Y_l^m(θ,φ) = sqrt((2l+1)/(4π) × (l-m)!/(l+m)!) × P_l^m(cos θ) × e^(imφ)

For l=0: Y_0^0 = 1/sqrt(4π)  (s orbital - spherical)

For l=1: Y_1^{-1} = sqrt(3/8π) sin θ e^{-iφ}  (p orbital - dumbbell)
         Y_1^0 = sqrt(3/4π) cos θ
         Y_1^1 = -sqrt(3/8π) sin θ e^{iφ}

For l=2: Y_2^m forms 5 d-orbitals (cloverleaf shapes)

For l=3: Y_3^m forms 7 f-orbitals (complex multilobed shapes)
```

### A.2 Selection Rule Derivation

```
From atomic physics, electric dipole transitions obey:
  Δl = ±1 (angular momentum conservation)
  Δm = 0, ±1 (magnetic quantum number)
  Δs = 0 (spin unchanged in non-relativistic limit)

Semantic analogy:
  Δn = ±1: Abstraction shift must be gradual
  Δl = 0, ±1: Category shift must be local
  Δm = 0, ±1: Sense shift must be gradual
  Δs = any: Grammatical category can shift (noun↔verb)

This yields ~15-25% base connectivity in attention matrix.
```

### A.3 Interference Formula

```
For two complex embeddings z₁ = a₁e^(iφ₁) and z₂ = a₂e^(iφ₂):

Inner product: ⟨z₁|z₂⟩ = a₁a₂e^(i(φ₂-φ₁))

Constructive interference: When φ₁ ≈ φ₂, Re(⟨z₁|z₂⟩) = a₁a₂ (maximum)
Destructive interference: When φ₁ ≈ φ₂ + π, Re(⟨z₁|z₂⟩) = -a₁a₂ (minimum)

This enables richer attention patterns than real-valued dot products.
```

---

## Appendix B: Implementation Checklist

```
□ Phase 1: Orbital Pre-computation Pipeline
  □ Extract base embeddings from pre-trained model
  □ Build semantic hierarchy
  □ Cluster abstraction levels
  □ Assign quantum numbers
  □ Compute complex amplitudes
  □ Validate orbital cache

□ Phase 2: Core Model Implementation
  □ ComplexLinear layer
  □ ComplexLayerNorm
  □ modReLU activation
  □ SparseComplexAttention
  □ ComplexFFN
  □ WaveFunctionTransformerBlock
  □ Full WaveFunctionASA model

□ Phase 3: Sparse Attention Kernels
  □ Selection rule mask generator
  □ CSR sparse matrix storage
  □ Sparse complex matmul kernel
  □ Fused attention kernel

□ Phase 4: Training Infrastructure
  □ Wirtinger derivative backprop
  □ Complex optimizer (Adam for complex)
  □ Learning rate scheduling
  □ Gradient clipping for complex

□ Phase 5: Benchmarking & Validation
  □ Throughput benchmarks
  □ Memory profiling
  □ Compositional generalization tests
  □ Polysemy resolution evaluation
  □ Interference pattern analysis
```

---

**Document Status**: COMPLETE
**Ready for Implementation**: YES
**Next Steps**: Prototype Phase 1 (Orbital Pre-computation)
