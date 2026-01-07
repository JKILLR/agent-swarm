# Lesion Experiment Feasibility Analysis

**Date**: 2026-01-05
**Reviewer**: Internal Critical Review
**Subject**: External Claude's Proposed Phase 1.5 Dynamics Primacy Test

---

## Proposed Experiment

```python
# Phase 1.5: Dynamics Primacy for ASA
# Condition A: Corrupt 20% of amplitudes randomly
# Condition B: Force uniform attention (freeze dynamics)
# Measure: Which degrades subject-verb identification more?
```

---

## 1. Is This Feasible with Existing `asa_wave_pilot.py`?

### CRITICAL FINDING: `asa_wave_pilot.py` DOES NOT EXIST

The file referenced as `asa_wave_pilot.py` is **not implemented**. It exists only as:
- A conceptual reference in review documents
- Pseudo-code specifications in `wave_pilot_implementer_review.md`
- An architectural design in `ARCHITECTURE_WAVE_FUNCTION_ASA.md`

**What Actually Exists:**
| Component | Status | Location |
|-----------|--------|----------|
| Wave function concept | DESIGNED | `ARCHITECTURE_WAVE_FUNCTION_ASA.md` |
| 21 relational bases | SPECIFIED | `wave_pilot_implementer_review.md` |
| 100-word vocabulary | MENTIONED | Review documents |
| Compatibility matrix | CONCEPTUAL | Not implemented |
| Any Python code | **NONE** | No .py files in workspace |

### Verdict: NOT IMMEDIATELY FEASIBLE

The experiment cannot run today. However, the proposed lesion test is **well-designed for the specified architecture** and could be implemented alongside the pilot.

---

## 2. What Would This Prove/Disprove?

### Condition A: Corrupt 20% of Amplitudes Randomly
**Tests**: Does the *static structure* of wave function amplitudes carry semantic information?

| Outcome | Interpretation |
|---------|---------------|
| Degradation > 30% | Amplitudes encode semantics (structure matters) |
| Degradation < 10% | Amplitudes are noise-robust or redundant |
| Mixed pattern | Depends which amplitudes corrupted (informative!) |

### Condition B: Force Uniform Attention (Freeze Dynamics)
**Tests**: Is the *attention computation process* essential to semantics?

| Outcome | Interpretation |
|---------|---------------|
| Degradation > 30% | Dynamics essential (meaning is a VERB) |
| Degradation < 10% | Static lookups sufficient (meaning is a NOUN) |
| Similar to baseline | Attention selection matters, not dynamics |

### The Key Prediction (from STATE.md's unified path)

If the CSA hypothesis holds ("structure IS semantics"):
- **Condition A should hurt more** - corrupting the structure corrupts meaning
- **Condition B should hurt less** - dynamics can compensate with intact structure

If the Dynamics Primacy hypothesis holds ("meaning is a VERB"):
- **Condition B should hurt more** - freezing dynamics kills meaning
- **Condition A should hurt less** - dynamics can recover from structural noise

### Scientific Value

**High** - This is a clean, falsifiable test that directly addresses the core question in STATE.md:
> "Is meaning a VERB (immanent—the doing) or a NOUN (transcendent—the storing)?"

The subject-verb identification task is particularly well-chosen because:
1. It's semantically meaningful (tests understanding, not pattern matching)
2. It has clear ground truth (syntactic labels exist)
3. It requires integration (can't be solved by single-token lookup)
4. It's simple enough to implement with 100-word vocabulary

---

## 3. Implementation Complexity

### Minimum Viable Experiment (MVE)

```
Components needed:
1. Wave function representation (100 tokens × 21 bases)    ~100 LOC
2. Compatibility/attention computation                      ~50 LOC
3. Subject-verb identification task                         ~80 LOC
4. Condition A: Amplitude corruption                        ~20 LOC
5. Condition B: Uniform attention forcing                   ~20 LOC
6. Metrics and comparison                                   ~50 LOC
─────────────────────────────────────────────────────────
TOTAL: ~320 LOC
```

### Comparison to Full DPP Infrastructure

| Component | Full DPP (wave_pilot_implementer_review) | MVE |
|-----------|------------------------------------------|-----|
| DynamicWaveFunction | 150 LOC | 100 LOC (static version) |
| SemanticHeartbeat | 80 LOC | NOT NEEDED |
| DPPLesionController | 120 LOC | 40 LOC (simplified) |
| CriticalityThermometer | 200 LOC | NOT NEEDED |
| BenchmarkSuite | 200 LOC | 80 LOC (single task) |
| **Total** | **~1,800 LOC** | **~320 LOC** |

### Time Estimate

| Phase | Work |
|-------|------|
| 1. Static wave pilot | Implement minimal wave matrix, compatibility. 2-3 hours. |
| 2. Subject-verb task | Create test sentences, labels, evaluation. 1-2 hours. |
| 3. Lesion conditions | Amplitude corruption + uniform forcing. 1 hour. |
| 4. Run & analyze | Execute both conditions, compare. 1-2 hours. |
| **Total** | **5-8 hours focused work** |

This is achievable in **one focused day**, not the 15 days estimated for full DPP infrastructure.

---

## 4. Fatal Flaws Analysis

### Flaw 1: The Pilot Doesn't Exist (MAJOR)
**Severity**: Blocking
**Mitigation**: Implement MVE from scratch using the specifications in `ARCHITECTURE_WAVE_FUNCTION_ASA.md`

### Flaw 2: "Dynamics" in a Static System (MODERATE)
**Issue**: The proposed experiment talks about "dynamics" but the wave pilot architecture is fundamentally static:
- Wave amplitudes are pre-computed and frozen
- No temporal evolution
- No heartbeat

**The external Claude's Condition B ("freeze dynamics") is ambiguous in this context.**

**Clarification needed**: Does "freeze dynamics" mean:
- (a) Force attention weights to uniform distribution? (no selection)
- (b) Bypass attention entirely? (direct embedding lookup)
- (c) Something else?

**Recommended interpretation**: (a) Force uniform attention - this tests whether the *compatibility-based sparse selection* matters, which IS the "dynamics" of the wave function model (the evolving attention pattern, not temporal evolution).

### Flaw 3: Subject-Verb Task May Be Too Easy (MINOR)
**Issue**: With only 100 words, subject-verb identification might be trivially solvable via part-of-speech membership alone.

**Mitigation**: Use ambiguous cases where the same word could be subject or verb:
- "The fish swims" vs "Fish the pond"
- "The run was long" vs "Run quickly"

This forces the system to use context, not just word identity.

### Flaw 4: 20% Corruption May Be Arbitrary (MINOR)
**Issue**: Why 20%? This threshold wasn't justified.

**Recommendation**: Run with multiple corruption levels (5%, 10%, 20%, 30%, 50%) to find the degradation curve. This provides more information than a single threshold.

### Flaw 5: No Baseline Comparison (MODERATE)
**Issue**: The experiment compares Condition A vs Condition B, but not against:
- Uncorrupted system (baseline)
- Random baseline (chance performance)

**Mitigation**: Add:
- Condition 0: Intact system (upper bound)
- Condition C: Random output (lower bound / chance)

---

## 5. Recommended Experimental Design

```python
# Refined Phase 1.5 Design

class LesionExperiment:
    """
    Minimal viable test of structure vs dynamics in wave-function semantics.
    """

    def __init__(self):
        # Create static wave pilot (100 tokens × 21 bases)
        self.wave_matrix = self._init_wave_matrix()
        self.vocab = self._init_vocabulary()

    def condition_0_baseline(self) -> float:
        """Intact system - upper bound."""
        return self._evaluate(self.wave_matrix, use_attention=True)

    def condition_A_corrupt_structure(self, corruption: float = 0.2) -> float:
        """Corrupt amplitudes, preserve attention computation."""
        corrupted = self._corrupt_amplitudes(corruption)
        return self._evaluate(corrupted, use_attention=True)

    def condition_B_freeze_attention(self) -> float:
        """Intact amplitudes, uniform attention (no selection)."""
        return self._evaluate(self.wave_matrix, use_attention=False)

    def condition_C_random(self) -> float:
        """Random baseline - lower bound."""
        return 0.5  # Chance for binary S-V classification

    def run_all(self) -> dict:
        """Run all conditions at multiple corruption levels."""
        results = {
            'baseline': self.condition_0_baseline(),
            'structure_corruption': {
                level: self.condition_A_corrupt_structure(level)
                for level in [0.05, 0.1, 0.2, 0.3, 0.5]
            },
            'frozen_attention': self.condition_B_freeze_attention(),
            'random_chance': self.condition_C_random()
        }
        return self._analyze(results)

    def _analyze(self, results: dict) -> dict:
        """Determine which matters more: structure or dynamics."""
        baseline = results['baseline']
        frozen = results['frozen_attention']

        # Find corruption level that matches frozen attention degradation
        for level, score in results['structure_corruption'].items():
            if abs(score - frozen) < 0.05:
                return {
                    'equivalent_corruption': level,
                    'interpretation': (
                        f"Freezing attention ≈ corrupting {level*100:.0f}% of structure. "
                        f"{'DYNAMICS PRIMACY' if level < 0.15 else 'STRUCTURE PRIMACY'}"
                    )
                }

        return {'interpretation': 'No clear equivalence found'}
```

---

## 6. Recommendation

### Should We Implement This?

**YES** - with modifications.

**Rationale**:
1. The core idea is sound and directly tests the central hypothesis
2. The MVE is achievable in one day (~320 LOC)
3. It provides falsifiable evidence before committing to full infrastructure
4. It aligns with STATE.md's "falsification gate" philosophy

### Implementation Priority

```
1. [IMMEDIATE] Implement minimal static wave pilot (100×21)
2. [IMMEDIATE] Create subject-verb identification task
3. [IMMEDIATE] Implement Conditions 0, A, B, C
4. [IMMEDIATE] Run at multiple corruption levels
5. [THEN] Use results to decide whether to build full DPP infrastructure
```

### Decision Gate

| Result | Action |
|--------|--------|
| Condition B degradation >> Condition A (at 20%) | Dynamics primacy supported. Build full temporal system. |
| Condition A degradation >> Condition B | Structure primacy supported. Focus on static optimization. |
| Both degrade similarly | Both matter. Continue with unified path. |
| Neither degrades much | Task too easy or system too redundant. Need harder task. |

---

## 7. Summary

| Question | Answer |
|----------|--------|
| **Is this feasible?** | NOT YET - no code exists. MVE buildable in 1 day. |
| **What would it prove?** | Structure vs dynamics primacy for semantics. Clean falsification. |
| **Implementation time?** | 5-8 hours for MVE (vs 15 days for full DPP). |
| **Fatal flaws?** | No code exists (blocking). "Dynamics" needs clarification (moderate). |
| **Recommendation** | **IMPLEMENT MVE FIRST** as falsification gate before full infrastructure. |

---

**Status**: ANALYSIS COMPLETE
**Next Step**: Implement `asa_wave_pilot_minimal.py` with lesion conditions
