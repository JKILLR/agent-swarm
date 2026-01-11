# STRATEGIC DECISION: Final Experimental Program

**Date**: 2026-01-05
**Decision Maker**: COO Agent
**Status**: FINAL - EXECUTE

---

## THE CALL

After reviewing:
1. Agent consensus (INTEGRATION_SYNTHESIS.md)
2. External Claude critique (external_critique.md)
3. Previous strategic decision (STATE.md)
4. Experiment designs (EXPERIMENT_DESIGN_VALIDATION.md)
5. Critical analysis (CRITICAL_ANALYSIS_WAVE_FUNCTION.md)

---

## WHAT PROCEEDS

### IMMEDIATE: POC-1 (Complex Embedding Interference)

**Why this one first:**
- Smallest scope, fastest validation
- Answers the foundational question: Do phases learn anything?
- If phases don't diverge from initialization, EVERYTHING else is moot
- Memory: ~50-80MB (trivial for 8GB)
- Time: ~5 minutes training

**Success criteria (unchanged from EXPERIMENT_DESIGN_VALIDATION.md):**
- Phase variance > 0.1 after training
- Spearman ρ > 0.3 on SimLex-999
- Memory < 100MB

**Failure = project pivot.** No philosophical speculation. No Phase 2. Done.

---

### IF POC-1 PASSES: POC-2 + POC-3 (Parallel)

**POC-2 (Superposition Polysemy)**
- Tests: Does context-based collapse outperform lookup?
- Engineering value: Efficient WSD
- Meaning value: Measurement analogy validity

**POC-3 (Tensor Composition)**
- Tests: Does tensor product beat addition for adj-noun?
- Engineering value: Structured composition
- Meaning value: Compositionality hypothesis

**Run these in parallel** because they're independent and small enough.

---

### IF POC-2 AND POC-3 PASS: DPP Lesion Study

**This is the key experiment the external reviewer called "clever."**

The Dynamics Primacy Protocol:
```
CONDITION 1: Freeze dynamics, preserve structure
CONDITION 2: Corrupt structure, preserve dynamics
```

Which kills performance more?

**Engineering interpretation:**
- If frozen matches running → static sparsity suffices (Longformer wins)
- If frozen degrades → adaptive attention is real

**Meaning interpretation:**
- If structure matters more → meaning is NOUN
- If dynamics matter more → meaning is VERB

**This is the experiment that serves both projects simultaneously.**

---

### IF DPP PRODUCES SIGNAL: BENCH-2 (Compositional Generalization)

Only then do we run SCAN/COGS.

This is where compositional claims get tested against real benchmarks.

---

## EXECUTION ORDER

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
                                        ├── NO SIGNAL → Document null result, stop
                                        │
                                        └── SIGNAL → BENCH-2 (Compositional)
                                                        │
                                                        └── Results → PAPER
```

---

## WHAT GETS KILLED PERMANENTLY

### 1. "Wave-Function" terminology
Call it: **Complex Semantic Attention (CSA)**
No quantum mysticism. The math is linear algebra, not physics.

### 2. Semantic Bell Test
Unfalsifiable until we have working interference. Requires:
- POC-1 to validate phase learning
- DPP to show dynamics matter
- Then MAYBE we can ask about "non-classical correlations"
**Killed until prerequisites exist.**

### 3. Ontogenesis / "Meaning from TRUE ZERO"
Operationally undefined. What counts as "true zero"? What counts as "emergence"?
**Killed permanently unless someone can operationalize it.**

### 4. 10-week timeline
No timelines. Only gates. Phase 1 might take 3 days or 3 weeks.
**Killed permanently.**

### 5. Spherical Harmonics / Basis Recovery
The critical analysis is correct: this is "numerology dressed as physics."
**Killed permanently.** Use standard dictionary learning if we need basis analysis.

### 6. Separate "Immanent Semantics" track
The external reviewer is right: these are distinct projects.
Our counter: they share experimental substrate.
**Solution:** Single experiments, dual interpretation. NOT parallel development.

---

## IMMEDIATE NEXT ACTION

**Execute POC-1 within 24 hours.**

Steps:
1. Verify M2 hardware environment (verify_hardware.py)
2. Download GloVe-1K and SimLex-999
3. Implement MinimalComplexEmbedding (from EXPERIMENT_DESIGN_VALIDATION.md)
4. Train with contrastive objective
5. Measure phase variance
6. Evaluate on SimLex-999
7. **Report result in STATE.md**

---

## ADDRESSING THE THREE INPUTS

### Agent Consensus
**Adopted:**
- Unified path (experiments serve both engineering and meaning)
- Wave Pilot as substrate
- P1-P4 priority structure

**Modified:**
- No parallel philosophy track
- Stricter falsification gates

### External Claude Critique
**Accepted:**
- ASA vs Immanent Semantics are different projects
- 10-week estimate unrealistic
- Some experiments unfalsifiable
- DPP is clever and testable

**Rejected:**
- "Finish A then B" → We run unified experiments that test both

**Implemented:**
- DPP adapted for ASA as recommended
- Criticality-Correctness deferred
- Phase 3 experiments killed

### External Claude Update (Lesion Study Immediately Actionable)
**Accepted:**
- Lesion study (DPP) promoted to immediate-after-POC
- Falsification gates good

**Modified:**
- Still need POC-1 first (can't do lesion study without working phases)

---

## THE BOTTOM LINE

**One experiment at a time.**
**Clear pass/fail criteria.**
**No sunk cost fallacy.**

If POC-1 fails, we don't have a project. That's fine. Better to know in 5 minutes than 10 weeks.

If POC-1 passes, we have evidence that phases encode something. Then we ask what.

If DPP shows dynamics matter, we have evidence for both engineering (adaptive sparsity) and meaning (processual semantics).

If DPP shows structure matters, we have evidence that static sparse attention suffices (engineering win) and meaning is structural (philosophical finding).

Either way, we learn something real.

---

## COMMITMENT

I am committing to this decision. The first code execution happens within 24 hours.

No more strategic documents until POC-1 results exist.

---

**Decision Status**: FINAL
**Execution Deadline**: POC-1 results within 24 hours
**Fallback if blocked**: Document blockers in STATE.md, request human intervention

---

*"The best plan is the one that runs. Execute POC-1."*
