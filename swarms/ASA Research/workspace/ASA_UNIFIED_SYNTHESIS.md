# ASA Research: Unified Synthesis & Final Decision

**Date**: 2026-01-05
**Status**: DECISION FINAL
**Participants**: COO, Orchestrator, Critic, Researcher, External Claude Sessions

---

## Executive Summary

After comprehensive review integrating:
1. Internal agent consensus (Wave Pilot = substrate, experiments = dynamics)
2. External Claude critique (scope creep warning, falsification gates)
3. External Claude update (lesion study immediately actionable, Noun vs Verb is hardware question)
4. Technical audit (no code exists - 100% documentation, 0% implementation)

**THE CALL**: Execute POC-1 immediately. No more strategy documents until code runs.

---

## The Integration That Matters

### External Claude's Key Insight

> "Noun vs Verb" as a hardware question. If relationships are STORED (learned embeddings), you need memory. If relationships are COMPUTED (predetermined bases), you trade memory for compute. That's not philosophy—it's architecture.

**This reframes everything.** The "immanent semantics" vs "stored semantics" debate becomes empirically testable via resource profiling, not metaphysics.

### How This Changes Our Framework

| Old Frame | New Frame |
|-----------|-----------|
| Philosophical: Is meaning inherent? | Engineering: Where does compute happen? |
| Metaphysical: VERB vs NOUN | Practical: Static lookup vs dynamic computation |
| Abstract: Structure IS semantics | Measurable: Which correlates with correctness? |

---

## What All Agents Agreed On

### Consensus Points (100% Agreement)

1. **Wave Pilot = Substrate, Experiments = Dynamics**
   - The 21-basis wave function is the *stage*
   - The experiments test what *happens on the stage*

2. **POC-1 Must Come First**
   - If phases don't learn anything, nothing else matters
   - 5-minute validation gatekeeps everything

3. **DPP (Lesion Study) is "Clever and Testable"**
   - Freeze dynamics, corrupt structure → which kills performance?
   - Direct falsification of VERB vs NOUN hypothesis

4. **No Timelines, Only Gates**
   - Removed 10-week estimate
   - Each experiment passes or fails before proceeding

### What Got Killed

| Killed | Reason | By Who |
|--------|--------|--------|
| "Wave Function" terminology | Quantum mysticism | All agents |
| Semantic Bell Test | Unfalsifiable without prerequisites | Critic |
| Ontogenesis / TRUE ZERO | Operationally undefined | External Claude |
| 10-week timeline | Unrealistic | External Claude |
| Spherical harmonics / basis recovery | "Numerology" | Critical Analysis |
| Separate philosophy track | Same experiments serve both | Orchestrator |

---

## Critical Finding: No Code Exists

### The Reality Check

| Aspect | Status |
|--------|--------|
| Design Documentation | ~60,000 LOC equivalent |
| Python Implementation | **0 lines** |
| Runnable Experiments | **None** |
| asa_wave_pilot.py | **Does not exist** |

> *"A beautiful theory is only as good as the experiment that tests it."*

### The Gap

```
DOCUMENTATION          IMPLEMENTATION
████████████████       ░░░░░░░░░░░░░░░░
100% complete          0% complete
```

---

## The Execution Plan

### Phase 1: POC-1 (Complex Embedding Interference)

**Goal**: Do phases learn anything at all?

**Implementation**:
```python
# MinimalComplexEmbedding - ~100 LOC
class MinimalComplexEmbedding:
    def __init__(self, vocab_size=1000, dim=50):
        # Complex embedding: a + bi
        self.real = nn.Embedding(vocab_size, dim)
        self.imag = nn.Embedding(vocab_size, dim)

    def forward(self, x):
        return self.real(x) + 1j * self.imag(x)
```

**Success Criteria**:
- Phase variance > 0.1 after training
- Spearman ρ > 0.3 on SimLex-999
- Memory < 100MB

**Failure = Project Pivot**. No Phase 2. Done.

### Phase 2: POC-2 + POC-3 (If POC-1 Passes)

Run in parallel:
- **POC-2**: Superposition polysemy (does context collapse meaning?)
- **POC-3**: Tensor composition (does tensor beat addition for adj-noun?)

### Phase 3: DPP Lesion Study (If POCs Pass)

The external Claude's proposed experiment, refined:

```python
class LesionExperiment:
    def condition_0_baseline(self):
        """Intact system - upper bound."""
        return self._evaluate(wave_matrix, use_attention=True)

    def condition_A_corrupt_structure(self, corruption=0.2):
        """Corrupt amplitudes, preserve attention."""
        corrupted = self._corrupt_amplitudes(corruption)
        return self._evaluate(corrupted, use_attention=True)

    def condition_B_freeze_attention(self):
        """Intact amplitudes, uniform attention."""
        return self._evaluate(wave_matrix, use_attention=False)
```

**Interpretation**:
- Condition B hurts more → Meaning is VERB (dynamics primacy)
- Condition A hurts more → Meaning is NOUN (structure primacy)
- Both hurt equally → Both matter (unified path)

### Phase 4: BENCH-2 (If DPP Shows Signal)

SCAN/COGS compositional benchmarks only after DPP validation.

---

## Decision Tree

```
POC-1 (Complex Interference)
    │
    ├── FAIL → STOP (reassess entire direction)
    │
    └── PASS → POC-2 + POC-3 (parallel)
                    │
                    ├── BOTH FAIL → STOP
                    │
                    ├── ONE PASSES → Continue with what works
                    │
                    └── BOTH PASS → DPP Lesion Study
                                        │
                                        ├── NO SIGNAL → Document null, stop
                                        │
                                        └── SIGNAL → BENCH-2 (Compositional)
                                                        │
                                                        └── Results → PAPER
```

---

## Implementation Requirements

### Minimum Viable POC-1 (~320 LOC)

| Component | LOC | Status |
|-----------|-----|--------|
| Complex embedding class | 50 | Not started |
| Training loop | 80 | Not started |
| Contrastive loss | 40 | Not started |
| Phase variance metrics | 30 | Not started |
| SimLex evaluation | 80 | Not started |
| Memory profiling | 40 | Not started |
| **Total** | **320** | **0% complete** |

### Dependencies

```
numpy >= 1.24
torch >= 2.0
scipy >= 1.10
matplotlib >= 3.7
```

### Hardware Constraint

**M2 Mac Mini 8GB** - All experiments must include memory watchdog

---

## Addressing All Inputs

### Agent Consensus
**Adopted**: Unified path, substrate/dynamics split, P1-P4 priority
**Modified**: Stricter gates, no parallel philosophy track

### External Claude Critique
**Accepted**:
- ASA vs Immanent Semantics are different projects
- 10-week estimate unrealistic
- Some experiments unfalsifiable

**Rejected**:
- "Finish A then B" → Unified experiments test both

### External Claude Update
**Accepted**:
- "Noun vs Verb = hardware question" (key reframe)
- Lesion study immediately actionable
- Falsification gates good

**Modified**:
- POC-1 must come first (can't lesion what doesn't exist)

---

## The Bottom Line

**One experiment at a time.**
**Clear pass/fail criteria.**
**No sunk cost fallacy.**

If POC-1 fails in 5 minutes, we save 10 weeks.
If POC-1 passes, we have evidence that phases encode *something*.
If DPP shows dynamics matter, we have evidence for both engineering AND meaning.
If DPP shows structure matters, we have evidence for static attention (also useful).

**Either way, we learn something real.**

---

## Immediate Action

**Execute POC-1 within 24 hours.**

Steps:
1. Create `swarms/ASA Research/workspace/code/poc1_complex_embedding.py`
2. Download GloVe-1K subset and SimLex-999
3. Implement MinimalComplexEmbedding
4. Train with contrastive objective
5. Measure phase variance
6. Evaluate on SimLex-999
7. Report result in STATE.md

---

## Commitment

**No more strategic documents until POC-1 results exist.**

The best plan is the one that runs.

---

*"Execute POC-1."*

---

**Document Status**: FINAL
**Decision Status**: APPROVED
**Next Action**: POC-1 Implementation
**Deadline**: Results within 24 hours
