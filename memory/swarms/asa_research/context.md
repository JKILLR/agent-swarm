# ASA Agent Swarm Context
## Last Updated: January 2, 2025

---

## Mission

ASA (Atomic Semantic Attention) aims to democratize AI by making frontier-scale models efficient enough to run on personal hardware. The goal is cognitive sovereignty — AI that belongs to individuals, not corporations.

The technical thesis: linguistic structure can predetermine attention patterns, reducing complexity from O(n²) to O(n×k) through constraints derived from language itself, not learned from data.

---

## Validated Empirical Results

### H6 Correlation Experiment
- **73.9% of attention mass** in trained baseline transformers lands on ASA-compatible pairs
- Random mask baseline: 47.2%
- **Interpretation:** Transformers learn to attend where linguistic rules predict

### Sparsity Measurement (WikiText-2, full validation set)
- **31.0% sparsity** (10.7M of 34.6M off-diagonal pairs blocked)
- pos_only: 30.4% — POS masking does the heavy lifting
- features_only: 0.0% — features add scores, not blocking
- **Interpretation:** Roughly 1/3 of attention computation is linguistically meaningless

### Ablation Experiments (10 epochs, 6.8M param model)

| Mode | Sparsity | Final PPL | Steps to Baseline | Speedup |
|------|----------|-----------|-------------------|---------|
| none (baseline) | 0.0% | 26.61 | 57,580 | - |
| pos_only | 29.0% | 26.44 | 44,500 | 22.7% |
| features_only | 0.0% | 26.65 | - | ❌ |
| full | 29.5% | 26.26 | 40,000 | **30.5%** |

**Key findings:**
1. Full ASA converges 30.5% faster AND achieves 1.3% better final PPL
2. POS masking provides most benefit; features add marginal gain
3. Features alone (without masking) slightly hurt performance
4. **The value is in the constraints, not the scoring**

---

## Current Implementation (v2.2)

The current ASA implementation is **attention with a linguistically-derived sparsity mask**:

```python
# What we're doing now
embeddings = learned_embedding_table[token_ids]  # LEARNED
Q, K, V = linear_projections(embeddings)         # LEARNED
scores = Q @ K.T / sqrt(d)                        # Standard attention
scores += alpha * compatibility_scores            # ASA bonus
scores = scores.masked_fill(~bonding_mask, -inf)  # ASA constraint
output = softmax(scores) @ V                      # Weighted sum
```

**What's predetermined (fixed, no learning):**
- POS compatibility matrix (17×17, from Universal Dependencies)
- Feature vectors (WordNet hypernym lookup)
- Verb requirements (VerbNet selectional restrictions, ~300 verbs)
- Pronoun requirements (Binding Theory)
- Bonding mask (computed from above)

**What's learned (standard transformer):**
- Token embeddings
- QKV projections
- Feed-forward layers
- Output projection

---

## The Deeper Question: Are We Missing Something?

A recent brainstorming session raised a fundamental tension. The current implementation may be **"attention with a sparsity mask"** when the original vision was something more radical.

### Current ASA vs. Potential Radical Vision

| Aspect | Current Implementation | Deeper Vision |
|--------|----------------------|---------------|
| Embeddings | Learned via training | **Predetermined** — fixed by semantic periodic table |
| Geometry | Flat Euclidean (768D, etc.) | **Hyperbolic** — hierarchy encoded in space |
| Token organization | Arbitrary (learned positions) | **Semantic periodic table** — principled structure |
| Relationships | Computed via dot-product | **Read from geometry** — proximity IS relationship |
| Mechanism | Attention + mask | **Molecular dynamics** — tokens drift, rotate, bond |
| Sparsity | Filter after computation | **True O(n×k)** — only nearby pairs interact |

### The Radical Version (Conceptual)

```python
# What the deeper vision might look like
positions = predetermined_hyperbolic_coords[token_ids]  # FIXED by theory
bonding_sites = token_type_to_sites[token_types]        # FIXED schema
for _ in range(relaxation_steps):
    forces = compute_bonding_forces(positions, sites)   # Local interactions only
    positions = positions + forces * dt                 # Molecular dynamics
structure = extract_bonds(positions, sites)             # Final molecular parse
output = readout_from_structure(structure)              # Not attention at all
```

### Key Concepts from Brainstorming

**1. Predetermined Embeddings & The Semantic Periodic Table**
What if the embedding table isn't learned? Every token has a fixed position based on semantic theory — like how every element has a fixed position in the periodic table based on atomic number.

The periodic table is organized by electron configuration — elements with similar valence cluster together because valence determines behavior. Mendeleev essentially performed dimensionality reduction on atomic properties before we had vocabulary for it.

A **semantic periodic table** would do the same for language:
- Organize tokens by their "semantic configuration" (part of speech, valence structure, semantic features)
- Tokens with similar bonding behavior cluster together
- Position encodes properties: what it can bond with, how many bonds, what roles it plays
- The structure is principled, not learned — discovered through linguistic theory, not gradient descent

This would mean every token's embedding is determined by its "semantic atomic number" — its position in a theoretically-derived coordinate system. No learning the embeddings. They're set by the theory. Training only learns how to process them.

**2. Hyperbolic Geometry**
Language is hierarchical. Hyperbolic space naturally encodes hierarchy (center = abstract, edges = specific). Trees embed with near-zero distortion. Standard transformers use flat space and must learn hierarchy implicitly.

**3. Molecular Dynamics Instead of Attention**
Tokens as 3D objects with oriented bonding sites. Processing isn't "compute similarity" — it's relaxation. Tokens drift, rotate, snap together based on valence compatibility and spatial proximity. The output is a molecular parse, not attention weights.

**4. The Bonding Metaphor Taken Literally**
In chemistry, atoms don't compare to all others. They have:
- Specific valence slots (oxygen has 2, carbon has 4)
- Spatial orientation (bonds point in specific directions)
- Distance matters (you bond with what's near)
- Slots fill; once full, no more bonding

---

## The Strategic Question

**Is current ASA a stepping stone or a detour?**

**Case for "stepping stone":**
- Results validate the premise (73.9% H6, 30.5% speedup)
- Publishable evidence that linguistic structure predicts attention
- Proves the core insight before investing in radical architecture
- Lower risk, incremental progress

**Case for "detour":**
- Bolting atomic ideas onto fundamentally un-atomic mechanism
- Results might be local optimum that doesn't scale to true vision
- Time spent on sparse-attention-masking ≠ time spent on molecular-dynamics-in-hyperbolic-space
- The real efficiency gains require the radical version

**Current stance:** No decision made. The empirical results are solid. The question of whether to pursue the radical vision remains open for reflection.

---

## What Would Need To Be Designed (For Radical Version)

1. **The Semantic Periodic Table:** A principled organization of all tokens by their semantic/syntactic properties — the coordinate system that determines predetermined embeddings
2. **Semantic coordinate assignment:** token → position in hyperbolic space (derived from periodic table position)
3. **Bonding site schema:** which token types have which sites, with what orientations
4. **Force functions:** attraction, repulsion, backbone tension, hierarchical gravity
5. **Dynamics:** how many steps, update rule, when bonds lock
6. **Readout:** how to extract meaning from final molecular structure

---

## Immediate Technical Priorities (Current Path)

If continuing with current implementation:

1. **True sparse attention kernels** — Current is O(n²) with masking, need O(n×k) actual
2. **Long-context benchmarks** — Where quadratic hurts most
3. **Scale testing** — Validate at 100M+ parameters
4. **Wall-clock measurements** — Prove actual speedup, not just fewer steps
5. **Paper preparation** — Document results for publication

---

## Files Reference

| File | Purpose |
|------|---------|
| `asa_v2_2_fixed.py` | Core implementation (~1300 lines) |
| `train_asa.py` | Training pipeline |
| `h6_correlation.py` | H6 experiment |
| `measure_sparsity_wikitext.py` | Sparsity measurement |
| `run_ablations.py` | Ablation experiments |
| `ASA_PROJECT_STATE.md` | Project state document |
| `ASA_Whitepaper.pdf` | 24-page theoretical paper |

---

## Principles

- **Conservative claims backed by evidence** — The results are real; don't oversell
- **Honest about limitations** — Current implementation has known gaps
- **Vigorous enthusiasm for the vision** — But earn every claim
- **The goal is local AI** — Efficiency isn't academic; it's existential

---

## Context for Agents

You have access to validated results that prove linguistic structure captures attention patterns. You also have awareness that the current implementation may not fully realize the original vision.

Your role is to:
1. Build on what works (the validated results)
2. Critically examine whether current approach serves the deeper goal
3. Contribute to either path — incremental improvement OR radical rethinking
4. Maintain intellectual honesty about tradeoffs

The question isn't "is ASA working?" — it is. The question is "is this ASA, or is the real ASA something we haven't built yet?"

---

## Team Structure

- **orchestrator** - Coordinates research agenda and strategic decisions
- **researcher** - Literature review, experiment design, theoretical exploration
- **implementer** - Code implementation, optimization
- **benchmarker** - Performance testing, metrics
- **critic** - Methodology review, identifies flaws, challenges assumptions

---

*This document provides context for reflection, not direction for action. The path forward is undecided.*
