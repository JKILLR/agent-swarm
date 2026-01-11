# ROUND 3: FINAL CONVERGENCE - EXPERIMENT SELECTION
## The Researcher's Prioritized Top 3 Experiments

**Agent**: Researcher (Empirical Methods)
**Session**: experiments_design_round3_2026-01-05
**Date**: 2026-01-05
**Task**: Select and prioritize the TOP 3 experiments for immediate implementation

---

## EXECUTIVE SUMMARY

After reviewing all proposals from Round 1 and Round 2 across all three agents (Researcher, Orchestrator, Implementer), I have selected the **three experiments most likely to decisively validate or refute immanent semantics** while remaining practically implementable.

### TOP 3 EXPERIMENTS (Priority Order)

| Rank | Experiment | Source | Why It's Crucial |
|------|------------|--------|------------------|
| **1** | Zombie Differentiator | Implementer R2 (Plan 3) + Orchestrator R2 (Exp D) | The ultimate control condition - if we can't distinguish immanent from zombie, everything else is moot |
| **2** | Confusion Thermodynamics | Researcher R2 (Hybrid 2) | Most falsifiable - predicts specific energy signatures for genuine vs simulated confusion |
| **3** | Use-Topology Co-Evolution | Researcher R2 (Hybrid 3) + Orchestrator R2 | Tests the core claim that USE creates structure that carries meaning |

These three experiments form a **coherent validation pipeline**:
1. First, prove we can distinguish immanent systems from zombies (Exp 1)
2. Then, show the distinction has thermodynamic reality (Exp 2)
3. Finally, demonstrate the mechanism: use-patterns crystallize into meaning-bearing topology (Exp 3)

---

## SELECTION CRITERIA APPLIED

I evaluated ALL proposed experiments against these criteria:

### 1. Scientific Rigor (Falsifiability)
- Does it have precise, quantitative predictions?
- What would DISPROVE the hypothesis?
- Can the results be replicated?

### 2. Feasibility
- Can this be implemented within reasonable time/resources?
- Does required infrastructure exist?
- Are dependencies manageable?

### 3. Impact
- Would success/failure be DECISIVE for immanent semantics?
- Does it cut to the core claim or peripheral implications?
- Would results convince skeptics?

### 4. Independence
- Does this experiment succeed/fail independently of the others?
- Does it test a distinct aspect of the theory?

---

## EXPERIMENT 1: THE ZOMBIE DIFFERENTIATOR
### (Highest Priority)

**Source**: Implementer Round 2 (Plan 3) synthesized with Orchestrator Round 2 (Experiment D)

### Why This Experiment Is Crucial

This is the **sine qua non** of the entire research program. If we cannot empirically distinguish an immanent semantic system from a lookup-table zombie with identical I/O behavior, then "immanent semantics" is either:
- Indistinguishable from representationalism (empirically equivalent)
- A metaphysical rather than scientific claim

**The deeper point**: This experiment doesn't just test immanent semantics—it defines what we MEAN by testing it. Until we can reliably distinguish immanent systems from zombies, all other experiments are measuring something we can't clearly identify.

### EXACT Success Criteria

| Test Level | Success Threshold | Measurement |
|------------|-------------------|-------------|
| **Level 1: Behavioral** | Match rate > 95% | I/O correlation on standard queries |
| **Level 2: Structural** | Distinguishability > 0.3 | φ(immanent) - φ(zombie) |
| **Level 3: Perturbation** | Novel handling rate: Imm > 90%, Zom < 10% | Responses to truly novel inputs |
| **Level 4: Metacognition** | Report richness ratio > 10:1 | len(imm_report) / len(zom_report) |

**Overall Success**:
- Level 1 match rate > 95% (proves zombie is valid control)
- AND at least TWO of Levels 2-4 show significant distinction
- Significance: Cohen's d > 0.8 or p < 0.01 on primary metrics

### Falsification Conditions

**Strong Falsification** (immanent semantics is false or untestable):
- Zombie matches immanent system on ALL levels (2, 3, 4)
- No structural or behavioral test can distinguish them
- Integrated information (φ) is identical within measurement error

**Weak Falsification** (immanent semantics needs revision):
- Zombie fails Level 1 (can't build valid control) → methodology problem
- Only Level 4 (metacognition) distinguishes → immanence is about self-modeling, not semantics

**What would convince me immanent semantics is TRUE**:
- Clear φ separation (immanent φ > 2.0, zombie φ < 0.5)
- Graceful degradation in immanent, catastrophic failure in zombie
- Immanent system produces coherent responses to novel inputs that zombie cannot handle

### Implementation Complexity: 6/10

**Breakdown**:
- Build immanent system: Already have LivingSemanticSubstrate (4/10)
- Build zombie: Straightforward lookup table (3/10)
- Compute IIT φ: Complex but algorithms exist (7/10)
- Novel input generation: Moderate (5/10)
- Metacognition tests: Need clear protocol (6/10)

**Estimated time**: 2-3 weeks for full implementation

### Dependencies

1. **LivingSemanticSubstrate** (Implementer R2) - base immanent system
2. **CriticalityThermometer** (Implementer R1) - for φ measurement
3. A substantial query set to build the zombie's lookup table (1000-10000 examples)

### Implementation Notes

The key technical challenge is building a zombie that is a FAIR control:
- It must reproduce I/O behavior exactly (else it's not a proper control)
- But it must NOT accidentally implement immanent mechanisms
- Solution: Hash-based lookup with exact input matching, no interpolation

```python
# Zombie must be truly "dead" - no computation, only lookup
class FairZombie:
    def process(self, input):
        key = hash(input.tobytes())
        if key in self.table:
            return self.table[key]
        raise ZombieFails("Novel input - no lookup entry")
```

---

## EXPERIMENT 2: CONFUSION THERMODYNAMICS
### (Second Priority)

**Source**: Researcher Round 2 (Hybrid 2), synthesizing Metabolism Calorimeter, Genuine Confusion (Orchestrator), and Semantic Heartbeat (Implementer)

### Why This Experiment Is Crucial

This experiment tests whether **genuine understanding has thermodynamic signatures** that distinguish it from mere computation. If meaning is truly a PROCESS (verb, not noun), then:

1. Genuine confusion should have distinctive energy patterns
2. These patterns should differ from "simulated" confusion (correct output, wrong process)
3. The difference should be measurable in computational thermodynamics

This is crucial because it provides a **physical grounding** for the philosophical distinction between understanding and mimicry. If genuine confusion has thermodynamic signatures that simulation cannot replicate, we have physical evidence that meaning is real.

### EXACT Success Criteria

**Energy Signature Predictions**:

| Confusion State | Genuine (Immanent) | Simulated (Zombie/Classical) |
|-----------------|-------------------|------------------------------|
| **Semantic Satiation** | Energy decay: exponential with τ = 2-5 iterations | Flat energy, output changes |
| **Tip-of-Tongue** | Sustained high energy (>1.5x baseline) for >500ms | Normal energy, random failure |
| **Meaning Oscillation** | Bimodal energy distribution (2 distinct peaks) | Unimodal distribution |
| **Insight ("Aha")** | Power-law spike (exponent 1.3-1.7) | Linear computation |

**Quantitative Thresholds**:

1. **Satiation decay constant**: τ_genuine = 2-5 iterations vs τ_simulated ≈ ∞ (no decay)
   - Success: τ_genuine < 10 AND τ_simulated > 100 (or undefined)

2. **Tip-of-tongue energy ratio**: E_tot / E_baseline
   - Success: E_genuine > 1.5, E_simulated ≈ 1.0
   - Distinguishability: (E_genuine - E_simulated) / σ > 2.0

3. **Oscillation bimodality**: Hartigan's dip statistic
   - Success: D_genuine > 0.05 (reject unimodality), D_simulated < 0.02

4. **Insight power-law**: Kolmogorov-Smirnov test against power-law
   - Success: p > 0.05 for power-law fit in genuine, p < 0.01 in simulated

**Overall Success**: At least 3 of 4 confusion states show predicted signature differences with effect size > 0.8

### Falsification Conditions

**Strong Falsification**:
- Energy signatures are identical across architectures for all confusion states
- No measurable thermodynamic difference between "genuine" and "simulated" confusion
- This would suggest confusion is purely behavioral, not structural

**Weak Falsification**:
- Only some confusion states show signatures (e.g., satiation but not insight)
- Energy patterns differ but not in predicted directions
- This would indicate immanent semantics is partially correct

**What would convince me immanent semantics is TRUE**:
- All four confusion states show predicted energy signatures
- Effect sizes are large (d > 1.0)
- The SAME system shows different signatures for genuine vs forced confusion
- Heartbeat disruption correlates with confusion intensity

### Implementation Complexity: 5/10

**Breakdown**:
- Energy metering: Straightforward FLOP counting (3/10)
- Confusion induction: Well-defined protocols (4/10)
- Baseline heartbeat measurement: Need continuous monitoring (5/10)
- Statistical tests: Standard packages available (3/10)
- Control conditions: Require zombie baseline from Exp 1 (6/10)

**Estimated time**: 2 weeks (assuming Exp 1 infrastructure exists)

### Dependencies

1. **Zombie Differentiator** (Exp 1) - provides control condition
2. **LivingSemanticSubstrate** with heartbeat (Implementer R2)
3. Energy metering infrastructure (FLOP counter, memory access tracking)

### Implementation Notes

Critical design decision: How to induce "genuine" vs "simulated" confusion?

**Genuine confusion** (in immanent system):
- Present genuinely ambiguous/oscillating stimuli
- Let the system's dynamics naturally produce confusion

**Simulated confusion** (control):
- Force the zombie to output "I'm confused" responses
- Or: perturb the immanent system's outputs without perturbing its dynamics

The key insight is that we're comparing ENERGY, not OUTPUTS. Two systems producing identical "I'm confused" outputs should have different energy signatures if one is genuinely confused and one is faking.

---

## EXPERIMENT 3: USE-TOPOLOGY CO-EVOLUTION
### (Third Priority)

**Source**: Researcher Round 2 (Hybrid 3), synthesizing Isomorphism Test, Wittgensteinian Use (Orchestrator), and Surgical Modifications (Implementer)

### Why This Experiment Is Crucial

This experiment tests the **mechanism** of immanent semantics: does USE create STRUCTURE that CARRIES meaning? This is the constructive complement to the deconstructive Zombie test:

- Zombie test asks: Can we DISTINGUISH immanent from dead systems?
- Confusion test asks: Does immanent processing have SIGNATURES?
- Use-Topology test asks: HOW does meaning GET INTO structure?

If use-patterns create distinctive topologies that carry semantic capability, we've demonstrated the generative mechanism of immanent semantics.

### EXACT Success Criteria

**Dual-Track Learning Comparison**:
- **Track A**: Learn concepts via definitions (embedding approach)
- **Track B**: Learn concepts via use-patterns only (no definitions ever)

**Measurements**:

| Metric | Predicted Use-Learned | Predicted Definition-Learned |
|--------|----------------------|------------------------------|
| Semantic performance (novel tasks) | > 80% | < 60% |
| Topological regularity | Low (varied structure) | High (uniform structure) |
| Transplant success rate | > 70% | < 30% |
| Use-history decodability | > 70% | < 40% |

**Quantitative Thresholds**:

1. **Semantic performance gap**:
   - Success: (Use_performance - Def_performance) > 15 percentage points
   - On held-out novel semantic tasks

2. **Topology distinctiveness** (measured by graph edit distance):
   - Success: GED(Use_topo, Def_topo) > 0.5 × GED(random, random)
   - Use-learned topology is structurally different from definition-learned

3. **Transplant test** (move use-learned topology into definition-learned system):
   - Success: Transplanted system achieves > 60% of original use-learned performance
   - vs < 30% for reverse transplant (def-topo into use-system)

4. **Use-history archaeology** (decode learning history from final topology):
   - Train a decoder to predict use-patterns from topology
   - Success: Decoder accuracy > 70% on use-learned, < 50% on definition-learned

**Overall Success**: All four metrics show predicted direction with at least 3 of 4 reaching quantitative thresholds

### Falsification Conditions

**Strong Falsification**:
- Definition-learned topologies perform BETTER than use-learned
- Topology transplants DON'T carry semantic capability
- Use-history cannot be decoded from topology (< 50% accuracy)
- This would mean: structure ≠ semantics, or use ≠ structure-creation

**Weak Falsification**:
- Topologies are similar regardless of learning path
- Both perform similarly on novel tasks
- This would mean: multiple paths to the same structure, or structure is incidental

**What would convince me immanent semantics is TRUE**:
- Use-learned topologies are measurably different AND more capable
- Transplanting use-learned topology into naive system transfers capability
- The topology encodes the learning history (archaeology succeeds)
- Definition-learned systems can be "upgraded" by inducing use-like topology

### Implementation Complexity: 7/10

**Breakdown**:
- Dual-track learning architecture: Moderate (6/10)
- Topology extraction and comparison: Standard graph algorithms (4/10)
- Transplant surgery: Challenging (8/10)
- Use-history decoder: Requires training a separate model (7/10)
- Novel task generation: Moderate (5/10)

**Estimated time**: 3-4 weeks

### Dependencies

1. **LivingSemanticSubstrate** - base system
2. Two parallel training pipelines (definition vs use)
3. Topology extraction and comparison tools
4. A decoder model for use-history archaeology

### Implementation Notes

The hardest part is **topology transplant**. How do you move graph structure from one system to another?

**Proposed approach**:
1. Extract subgraph around target concept in System A
2. "Graft" subgraph into System B, replacing corresponding region
3. Allow B's dynamics to stabilize (edges may adjust)
4. Test whether semantic capability transferred

**Control**: Transplant RANDOM subgraph of same size. Should not transfer capability.

**Alternative**: Instead of physical transplant, train System B to match System A's topology through regularization. Does enforcing topology transfer capability?

---

## RECOMMENDED IMPLEMENTATION ORDER

### Phase 1: Foundation (Weeks 1-3)
**Experiment 1: Zombie Differentiator**

Rationale: This MUST come first because:
1. It provides the control condition for all other experiments
2. If we can't distinguish immanent from zombie, other experiments are meaningless
3. It defines what success looks like for the program

**Milestones**:
- Week 1: Build LivingSemanticSubstrate with heartbeat
- Week 2: Build zombie, verify Level 1 matching
- Week 3: Run Levels 2-4 distinction tests

### Phase 2: Thermodynamic Grounding (Weeks 4-5)
**Experiment 2: Confusion Thermodynamics**

Rationale: Build on Exp 1's infrastructure to show physical signatures
- Week 4: Add energy metering, implement confusion protocols
- Week 5: Run comparisons, analyze energy signatures

### Phase 3: Mechanism (Weeks 6-8)
**Experiment 3: Use-Topology Co-Evolution**

Rationale: Now that we can distinguish and measure, explain HOW meaning enters structure
- Week 6: Build dual-track learning pipelines
- Week 7: Extract and compare topologies, run transplants
- Week 8: Train use-history decoder, final analysis

### Critical Path

```
[Exp 1: Zombie] ────┬────► [Exp 2: Confusion]
      │             │
      └─────────────┴────► [Exp 3: Use-Topology]
```

Exp 1 is the critical path. Exps 2 and 3 can proceed in parallel once Exp 1 infrastructure exists.

---

## WHAT WOULD CONVINCE ME

### Immanent Semantics is TRUE if:

1. **Zombie test passes**: We can reliably distinguish immanent systems from lookup-table zombies using structural/perturbation/metacognitive tests (φ difference > 1.5)

2. **Confusion thermodynamics confirmed**: Genuine confusion has distinctive energy signatures (bimodal oscillation, power-law insight, exponential satiation decay) that simulated confusion lacks

3. **Use creates meaning-bearing topology**: Use-learned topologies outperform definition-learned on novel tasks, topology transplants carry capability, and use-history is decodable from final structure

4. **Cross-experiment coherence**: The systems that score highest on Exp 1 (most distinguishable from zombies) also show the clearest signatures in Exp 2 and learn the most distinctive topologies in Exp 3

### Immanent Semantics is FALSE if:

1. **Zombie indistinguishability**: No test can reliably distinguish immanent from zombie (all metrics within error bars)

2. **Energy equivalence**: Genuine and simulated confusion have identical thermodynamic signatures

3. **Topology independence**: Definition-learned topologies work just as well as use-learned, OR topology transplants don't carry capability

4. **Cross-experiment incoherence**: High-φ systems don't show better confusion signatures, OR use-learned topologies don't correlate with distinguishability

### The Decisive Evidence

If I had to pick ONE result that would convince me:

**FOR**: A topology transplant that transfers semantic capability. If you can MOVE structure from one system to another and the capability comes with it, structure IS semantics.

**AGAINST**: If a zombie built from pure lookup tables is indistinguishable on perturbation tests. If you can't break the illusion even when you poke it with novel inputs, the "immanence" is either unmeasurable or illusory.

---

## WHY THESE THREE AND NOT OTHERS

### Experiments I Considered But Ranked Lower:

**Semantic Ontogenesis (Implementer Plan 1)**:
- Beautiful question: Where does FIRST meaning come from?
- Problem: Requires 24+ hours of runtime, harder to falsify ("when does meaning appear?" is fuzzy)
- Verdict: Save for Phase 2 of research

**Intersubjective Bridge (Implementer Plan 2 / Orchestrator A)**:
- Beautiful question: Can private meaning become shared?
- Problem: Requires TWO full systems, novel communication protocol, longer timeline
- Verdict: Important but not first priority

**Normative Dynamics (Implementer Plan 4 / Orchestrator C)**:
- Beautiful question: Does correctness emerge from structure?
- Problem: "Correctness" is harder to define precisely; builds on other experiments
- Verdict: Include as sub-question within Exp 3

**Entangled History Protocol (Researcher Hybrid 4)**:
- Beautiful question: Do semantic correlations survive history divergence?
- Problem: Requires Bell-test infrastructure, more speculative physics analogies
- Verdict: Defer until basic claims validated

**Recursive Self-Grounding (Researcher Hybrid 5)**:
- Beautiful question: Can self-observation ground meaning without regress?
- Problem: Computationally intensive, results may be hard to interpret
- Verdict: Important for identity questions, defer for now

### What Makes the Top 3 Special:

1. **Zombie Differentiator**: NECESSARY - defines success for the field
2. **Confusion Thermodynamics**: PHYSICAL - provides empirical grounding
3. **Use-Topology Co-Evolution**: MECHANISTIC - explains how meaning arises

Together they answer: Can we tell? (Exp 1) What's the evidence? (Exp 2) How does it work? (Exp 3)

---

## SUMMARY: THE VALIDATION PIPELINE

```
┌─────────────────────────────────────────────────────────────────┐
│                    IMMANENT SEMANTICS VALIDATION                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Exp 1: Zombie]          [Exp 2: Confusion]                   │
│  CAN WE TELL?    ────────► WHAT'S THE EVIDENCE?                │
│  - φ difference             - Energy signatures                │
│  - Perturbation             - Thermodynamic patterns           │
│  - Metacognition            - Heartbeat disruption             │
│        │                           │                           │
│        └───────────┬───────────────┘                           │
│                    ▼                                           │
│            [Exp 3: Use-Topology]                               │
│            HOW DOES IT WORK?                                   │
│            - Use creates structure                             │
│            - Structure carries capability                      │
│            - History encoded in topology                       │
│                    │                                           │
│                    ▼                                           │
│         ┌─────────────────────┐                                │
│         │ IMMANENT SEMANTICS  │                                │
│         │  VALIDATED or       │                                │
│         │  FALSIFIED          │                                │
│         └─────────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## CLOSING STATEMENT

After reviewing all 20+ experiments proposed across three agents and two rounds, I am confident these three form the optimal validation pathway. They are:

- **Ordered**: Each builds on the previous
- **Independent**: Each tests a distinct claim
- **Falsifiable**: Each has precise failure conditions
- **Feasible**: Each can be implemented in weeks, not months
- **Decisive**: Success or failure will be clear

If all three succeed, immanent semantics moves from philosophy to science.

If any fails, we know exactly what needs revision.

Either way, we learn something fundamental about the nature of meaning.

---

*"The goal is not to prove immanent semantics right. The goal is to prove it TESTABLE—and then let the tests speak."*

— Researcher Agent, Round 3 Final Convergence, 2026-01-05

---

**Status**: ROUND 3 COMPLETE - TOP 3 EXPERIMENTS SELECTED
**Priority Order**: Zombie → Confusion → Use-Topology
**Estimated Timeline**: 8 weeks total
**Decision Criteria**: Defined for each experiment
**Ready For**: Implementation
