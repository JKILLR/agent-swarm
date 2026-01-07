# ASA Wave Pilot Code Audit

**Date**: 2026-01-05
**Status**: TECHNICAL ASSESSMENT
**Purpose**: Document current code state, capabilities, gaps, and requirements for lesion study implementation

---

## Executive Summary

**Critical Finding**: There is NO actual `asa_wave_pilot.py` implementation file. The codebase consists entirely of design documentation and research notes. All "code" referenced exists only as pseudocode within markdown files.

| Aspect | Status |
|--------|--------|
| Python Implementation | **DOES NOT EXIST** |
| Design Documentation | Extensive (~60,000 LOC equivalent in specs) |
| Theoretical Framework | Well-developed |
| Runnable Experiments | **NONE** |

---

## 1. What `asa_wave_pilot.py` Is Supposed to Implement

Based on design documents (`wave_pilot_review_input.md`, `wave_pilot_implementer_review.md`, `ARCHITECTURE_WAVE_FUNCTION_ASA.md`):

### 1.1 Core Data Structures

```python
# Intended but NOT implemented:
wave_matrix: np.ndarray      # Shape: (100 tokens, 21 bases)
compatibility_matrix: np.ndarray  # Shape: (100, 100) token-token attention
vocab: List[str]             # 100-word vocabulary
bases: List[str]             # 21 relational bases
```

### 1.2 21 Relational Bases (Designed)

| Category | Bases | Count |
|----------|-------|-------|
| Syntactic Relations | DET_NOUN, ADJ_NOUN, ADV_VERB, ADV_ADJ, SUBJ_PRED, VERB_OBJ, AUX_VERB, PREP_COMP | 8 |
| Thematic Roles | AGENT_ACTION, PATIENT_ACTION, EXPERIENCER_STATE, INSTRUMENT_ACTION, LOCATION_EVENT, THEME_TRANSFER | 6 |
| Referential | PRONOUN_ANTECEDENT, POSSESSOR_POSSESSED, COORD_ITEMS | 3 |
| Semantic Fields | ANIMACY_FIELD, CONCRETE_FIELD, ABSTRACT_FIELD | 3 |
| Vacuum | GLOBAL_CONTEXT | 1 |

### 1.3 Core Computation (Specified)

```python
# Wave function: ψ_token = Σ α_r · φ_r (over 21 bases)
# Compatibility: ⟨ψ_i | ψ_j⟩ = Σ_r α_i(r) · α_j(r)

def compute_compatibility(wave_matrix):
    """Inner product gives token-token attention strength."""
    return wave_matrix @ wave_matrix.T
```

---

## 2. Functions That Should Exist (But Don't)

### 2.1 Static Wave Model Functions

| Function | Purpose | Status |
|----------|---------|--------|
| `build_vocabulary()` | Create 100-word vocab with linguistic annotations | NOT IMPLEMENTED |
| `define_bases()` | Define 21 relational bases | NOT IMPLEMENTED |
| `assign_wave_amplitudes()` | Map tokens to basis amplitudes | NOT IMPLEMENTED |
| `compute_compatibility_matrix()` | Calculate ⟨ψ_i\|ψ_j⟩ for all pairs | NOT IMPLEMENTED |
| `analyze_sparsity()` | Report natural attention sparsity | NOT IMPLEMENTED |

### 2.2 Dynamic Extension Functions (For DPP)

From `wave_pilot_implementer_review.md`:

| Function | Purpose | Estimated LOC |
|----------|---------|---------------|
| `DynamicWaveFunction.__init__()` | Initialize evolving wave state | 40 |
| `DynamicWaveFunction.evolve()` | Single timestep propagation | 50 |
| `SemanticHeartbeat._heartbeat_loop()` | Continuous background dynamics | 30 |
| `SemanticHeartbeat._tune_criticality()` | Self-organize to edge of chaos | 20 |
| `DPPLesionController.dynamics_lesion()` | Freeze dynamics, keep structure | 30 |
| `DPPLesionController.structure_lesion()` | Corrupt structure, keep dynamics | 30 |
| `FrozenSystem.process_context()` | Static-only processing | 20 |
| `ZombieBaseline.process()` | Lookup table approximation | 40 |

### 2.3 Measurement Infrastructure

| Component | Functions | Status |
|-----------|-----------|--------|
| `CriticalityThermometer` | `measure_avalanche_distribution()`, `criticality_score()` | NOT IMPLEMENTED |
| `ConfusionThermometer` | `measure_confusion_signature()`, entropy tracking | NOT IMPLEMENTED |
| `SemanticCalorimeter` | `measure_basal_rate()`, energy consumption | NOT IMPLEMENTED |
| `BenchmarkSuite` | `analogy_test()`, `similarity_judgment()`, `novel_combination()` | NOT IMPLEMENTED |

---

## 3. What Needs to Be Added for Lesion Study

### 3.1 Minimum Viable Implementation for DPP Phase 1

The Dynamics Primacy Protocol (lesion study) requires:

#### Priority 0 (Must Have)

| Component | LOC Estimate | Dependencies |
|-----------|--------------|--------------|
| Static wave pilot core | 200 | numpy |
| DynamicWaveFunction class | 150 | Static core |
| SemanticHeartbeat class | 80 | DynamicWaveFunction |
| DPPLesionController | 120 | DynamicWaveFunction, Heartbeat |
| FrozenSystem wrapper | 60 | None |
| ZombieBaseline | 100 | DynamicWaveFunction |
| **Subtotal P0** | **710** | |

#### Priority 1 (Required for Analysis)

| Component | LOC Estimate | Dependencies |
|-----------|--------------|--------------|
| CriticalityThermometer | 200 | powerlaw library |
| ConfusionThermometer | 60 | DynamicWaveFunction |
| BenchmarkSuite | 200 | All systems |
| StatisticalAnalyzer | 150 | scipy.stats |
| **Subtotal P1** | **610** | |

#### Total Minimum Implementation: ~1,320 LOC

### 3.2 Phase 1 Experiments (From round3_implementer.md)

| Experiment | Purpose | Falsification Criterion |
|------------|---------|------------------------|
| 1.1 Criticality-Correctness | Does criticality = semantic correctness? | r < 0.3 |
| 1.2 Zombie Baseline | Build I/O-matched static control | match < 0.95 |
| 1.3 Semantic Heartbeat | Does system show resting activity? | BSR = 0 |
| 1.4 Topology-Semantics | Same structure = same errors? | correlation < 0.5 |

### 3.3 Key DPP Test Structure

```
DYNAMICS PRIMACY PROTOCOL
├── CONDITION 1: DYNAMICS LESION
│   ├── Stop heartbeat
│   ├── Freeze wave evolution
│   ├── PRESERVE: wave amplitudes, compatibility matrix
│   └── MEASURE: Can system still do semantic tasks?
│
├── CONDITION 2: STRUCTURE LESION
│   ├── Keep heartbeat running
│   ├── ADD NOISE to wave amplitudes
│   ├── PRESERVE: dynamics engine, evolution rules
│   └── MEASURE: Can dynamics compensate?
│
└── KEY PREDICTION:
    If meaning is VERB: dynamics lesion kills performance
    If meaning is NOUN: structure lesion kills performance
```

---

## 4. Data and Dependencies Required

### 4.1 Python Dependencies

```python
# Core
numpy >= 1.24
scipy >= 1.10
networkx >= 3.0

# Criticality analysis
powerlaw >= 1.5.0

# Machine learning (comparison)
torch >= 2.0

# Visualization
matplotlib >= 3.7
plotly >= 5.0

# Statistics
statsmodels >= 0.14

# Development
pytest >= 7.0
hypothesis >= 6.0  # Property-based testing
```

### 4.2 Data Requirements

| Data | Source | Purpose |
|------|--------|---------|
| 100-word vocabulary | Manual curation | Token set for pilot |
| Linguistic annotations | POS tagger, WordNet | Basis amplitude assignment |
| Similarity test pairs | SimLex-999, WordSim-353 | Benchmark validation |
| Analogy test set | Google Analogies | Semantic task testing |
| Co-occurrence corpus | Wikipedia, CommonCrawl | Usage pattern learning |

### 4.3 Hardware Constraints

From STATE.md:
- **Target**: M2 Mac Mini 8GB RAM
- **Constraint**: All experiments must include memory watchdog
- **No experiment runs without memory profiling**

---

## 5. Gap Analysis

### 5.1 Current State

```
DOCUMENTATION          CODE IMPLEMENTATION
     ████████████████       (empty)
     100% complete          0% complete
```

### 5.2 Critical Path to Lesion Study

```
Week 1: Foundation
├── Day 1-2: Implement static wave pilot core (200 LOC)
│   ├── build_vocabulary() - 100 words with annotations
│   ├── define_bases() - 21 relational bases
│   └── compute_compatibility() - wave function inner product
│
├── Day 3-4: Add dynamics (230 LOC)
│   ├── DynamicWaveFunction - temporal evolution
│   └── SemanticHeartbeat - background activity
│
└── Day 5: Lesion infrastructure (180 LOC)
    ├── DPPLesionController
    ├── FrozenSystem
    └── ZombieBaseline

Week 2: Measurement & Experiments
├── Day 6-7: Measurement infrastructure (410 LOC)
│   ├── CriticalityThermometer
│   ├── ConfusionThermometer
│   └── BenchmarkSuite
│
├── Day 8-9: Phase 1 experiments (400 LOC)
│   ├── Exp 1.1: Criticality-Correctness
│   ├── Exp 1.2: Zombie Baseline
│   ├── Exp 1.3: Semantic Heartbeat
│   └── Exp 1.4: Topology-Semantics
│
└── Day 10: Integration & testing
```

---

## 6. Architecture Decision: Wave Functions vs Graph

Two architectures are specified in documentation:

### Option A: Wave Function Formalism (wave_pilot)
- Token = wave function ψ = Σ α_r · φ_r
- Compatibility = ⟨ψ_i|ψ_j⟩
- Natural sparsity from orthogonal bases
- **Pros**: Interpretable, physics-grounded math
- **Cons**: May not scale to larger vocabularies

### Option B: Graph-Based (round3_implementer)
- Token = node in nx.Graph
- Semantics = node states + edge weights
- Criticality via graph dynamics
- **Pros**: More flexible topology
- **Cons**: Less constrained, harder to interpret

### Recommendation

Implement BOTH in parallel:
1. Wave pilot for interpretability and mathematical rigor
2. Graph-based for scalability experiments
3. Compare DPP results across architectures
4. Architecture-independent validation = stronger evidence

---

## 7. Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| 21 bases insufficient | Medium | High | Allow dynamic basis expansion |
| 100-word vocab too small | High | Medium | Design for vocab scaling |
| Heartbeat threading unstable | Medium | High | Fallback to synchronous mode |
| Power law fitting unreliable | Medium | Medium | Multiple fitting methods |
| 8GB memory exceeded | High | High | Memory watchdog, batch processing |
| Criticality not observable | Medium | High | Alternative metrics (variance, correlation length) |

---

## 8. Immediate Next Steps

1. **Create `asa_wave_pilot.py`** - Start with static core
   - Define vocabulary
   - Define bases
   - Compute wave matrix
   - Compute compatibility matrix

2. **Add dynamics** - `DynamicWaveFunction` class
   - Temporal evolution
   - Temperature parameter
   - Trajectory recording

3. **Implement lesion infrastructure**
   - Freeze dynamics
   - Corrupt structure
   - Build zombie baseline

4. **Run Phase 1 experiments**
   - Criticality-Correctness (Exp 1.1)
   - Zombie Baseline (Exp 1.2)
   - Semantic Heartbeat (Exp 1.3)
   - Topology-Semantics (Exp 1.4)

---

## 9. Success Criteria (Phase 1)

| Metric | Threshold | Implication |
|--------|-----------|-------------|
| Criticality-Correctness r | > 0.5 | Criticality thesis viable |
| Zombie I/O match rate | > 0.95 | Valid control established |
| Semantic heartbeat BSR | > 0.01 | Dynamics thesis viable |
| Topology-semantics correlation | > 0.8 | Structure = semantics viable |

**Go/No-Go**: If 3/4 pass → proceed to Phase 2. If <3 pass → pause and investigate.

---

## 10. Conclusion

The ASA Wave Pilot project has extensive theoretical documentation but **zero runnable code**. Before any lesion study experiments can proceed:

1. Static wave function model must be implemented (~200 LOC)
2. Dynamics must be added (~230 LOC)
3. Lesion infrastructure must be built (~180 LOC)
4. Measurement tools must be created (~410 LOC)
5. Experiments must be coded (~400 LOC)

**Total implementation gap**: ~1,420 LOC minimum

The design is sound. The theory is well-developed. The experiments are well-specified with explicit falsification criteria. What's missing is the actual Python implementation to run them.

---

**Audit Status**: COMPLETE
**Code Status**: NOT YET IMPLEMENTED
**Recommendation**: Begin implementation starting with static wave pilot core

---

*"A beautiful theory is only as good as the experiment that tests it."*
