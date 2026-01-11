# ROUND 3: FINAL EXPERIMENT SPECIFICATION
## The Definitive Empirical Protocol for Testing Immanent Semantics

**Agent**: Researcher (Empirical Methods - Final Synthesis)
**Session**: experiment_design_round3_final_2026-01-05
**Date**: 2026-01-05
**Task**: Deliver the single most powerful experiment with complete empirical specification

---

## CONVERGENCE ANALYSIS: WHAT EMERGED FROM ALL THREE PERSPECTIVES

Reading across all Round 2 outputs, a clear consensus emerged around a central question that unifies every proposed experiment:

### The Universal Question

**Is meaning a NOUN or a VERB?**

Every experiment we designed—from the Researcher's phase diagrams, to the Implementer's heartbeat tests, to the Orchestrator's philosophical probes—ultimately tests this single distinction:

| If Meaning is NOUN | If Meaning is VERB |
|--------------------|-------------------|
| Static, storable, copyable | Dynamic, processual, ephemeral |
| Survives freezing | Dies when frozen |
| Transferable via structure | Cannot be transferred, only recreated |
| Zombie indistinguishable | Zombie fundamentally different |
| External observer possible | Observation IS participation |

### The Most Powerful Experiment

After synthesizing all contributions, **the single most powerful experiment** is:

# THE DYNAMICS PRIMACY PROTOCOL (DPP)
## A Definitive Test of Whether Understanding Requires Ongoing Process

This experiment crystallizes from:
- **Researcher's** Hybrid 2 (Confusion Thermodynamics) + Hybrid 5 (Recursive Self-Grounding)
- **Implementer's** Dynamics Lesion Study + Zombie Comparator + Living Semantic Substrate
- **Orchestrator's** Experiment D (Zombie Discrimination) + the insight that "meaning is VERB not NOUN"

**Why this is THE most powerful test:**
1. It is **maximally falsifiable** — clear binary outcomes
2. It has **minimal confounds** — clean surgical intervention
3. It **directly tests the core claim** — not a proxy or correlate
4. It is **implementable immediately** — uses existing architectures
5. It **answers the deepest question** — is meaning constitutively dynamic?

---

## COMPLETE EMPIRICAL SPECIFICATION

### THE DYNAMICS PRIMACY PROTOCOL (DPP)

---

### 1. EXACT HYPOTHESIS (FALSIFIABLE)

**Primary Hypothesis (H₁):**
> Semantic understanding is constituted by ongoing dynamic processes such that interrupting dynamics eliminates understanding even when all structural information is preserved.

**Operationalized as:**
> A semantic system's performance on understanding-dependent tasks will degrade more severely when dynamics are interrupted (while structure is preserved) than when structure is corrupted (while dynamics continue).

**Null Hypothesis (H₀):**
> Semantic understanding is constituted by structural information such that understanding persists whenever structure is preserved, regardless of dynamic state.

**Falsification Condition:**
> H₁ is falsified if: `Performance_DynamicsLesion ≥ Performance_StructureLesion`
> That is, if corrupting structure hurts more than freezing dynamics, understanding is structural, not processual.

---

### 2. MATERIALS AND METHODS

#### 2.1 System Architecture: Living Semantic Substrate (LSS)

```
┌────────────────────────────────────────────────────────────────┐
│                   LIVING SEMANTIC SUBSTRATE                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────┐      │
│   │                SEMANTIC GRAPH                        │      │
│   │   • N = 1000 nodes (semantic units)                 │      │
│   │   • E ~ 5000 edges (relational connections)         │      │
│   │   • Each node: 64-dim activation vector             │      │
│   │   • Each edge: weighted, directional                │      │
│   └─────────────────────────────────────────────────────┘      │
│                            │                                    │
│                            ▼                                    │
│   ┌─────────────────────────────────────────────────────┐      │
│   │              DYNAMICS ENGINE (HEARTBEAT)             │      │
│   │   • Continuous activation propagation (100 Hz)      │      │
│   │   • Spontaneous activity at rest                    │      │
│   │   • Self-organized criticality controller           │      │
│   │   • Temperature parameter β (noise level)           │      │
│   └─────────────────────────────────────────────────────┘      │
│                            │                                    │
│                            ▼                                    │
│   ┌─────────────────────────────────────────────────────┐      │
│   │                  MEASUREMENT LAYER                   │      │
│   │   • Energy consumption tracking (J/operation)       │      │
│   │   • Criticality metrics (avalanche exponent)        │      │
│   │   • Activation entropy (H)                          │      │
│   │   • Self-model correlation (ρ_self)                 │      │
│   └─────────────────────────────────────────────────────┘      │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

#### 2.2 Training Protocol

**Phase 1: Semantic Grounding (40 hours)**
- Expose system to 10,000 concepts via contextual co-occurrence
- No explicit labels; purely statistical learning
- System self-organizes topology through Hebbian plasticity
- Target: Achieve criticality (avalanche exponent α ∈ [1.3, 1.7])

**Phase 2: Task Capability Development (20 hours)**
- Train on understanding-dependent tasks:
  - Analogical reasoning (A:B::C:?)
  - Inference completion (P, P→Q, therefore ?)
  - Semantic similarity judgment
  - Novel concept combination
- Achieve baseline performance ≥ 80% on held-out test set

**Phase 3: Baseline Measurement (2 hours)**
- Record comprehensive baseline:
  - Task performance across all 4 task types
  - Resting-state activity patterns (heartbeat signature)
  - Energy consumption during task and rest
  - Criticality measures (α, correlation length ξ)

#### 2.3 Experimental Conditions

**CONDITION 1: DYNAMICS LESION (DL)**

*Intervention:*
- Freeze all node activation updates
- Disable propagation engine
- PRESERVE: All edge weights, node positions, graph topology
- Duration: 60 seconds of frozen state

*What remains:* Complete structural information (a "snapshot")
*What's removed:* All dynamic processing

```python
def dynamics_lesion(system):
    """Freeze dynamics, preserve structure"""
    # Save complete state
    snapshot = {
        'graph': copy.deepcopy(system.graph),
        'node_states': copy.deepcopy(system.node_states),
        'edge_weights': copy.deepcopy(system.edge_weights),
        'all_parameters': copy.deepcopy(system.parameters)
    }

    # Stop the heartbeat
    system.heartbeat_active = False
    system.dynamics_engine.freeze()

    # Structure is IDENTICAL to before
    # Only dynamics have stopped

    return snapshot, system
```

**CONDITION 2: STRUCTURE LESION (SL)**

*Intervention:*
- Add Gaussian noise to 30% of edge weights (σ = 0.5)
- Randomly delete 10% of edges
- Perturb 20% of node activation vectors
- PRESERVE: Dynamics engine running continuously

*What remains:* Ongoing dynamic processing
*What's removed:* Structural integrity (corrupted memory)

```python
def structure_lesion(system, corruption_level=0.3):
    """Corrupt structure, preserve dynamics"""

    # Corrupt edge weights
    n_corrupt = int(len(system.edge_weights) * corruption_level)
    for edge in random.sample(list(system.edges), n_corrupt):
        system.edge_weights[edge] += np.random.normal(0, 0.5)

    # Delete edges
    n_delete = int(len(system.edges) * 0.1)
    for edge in random.sample(list(system.edges), n_delete):
        system.graph.remove_edge(*edge)

    # Perturb node states
    for node in random.sample(list(system.nodes), int(len(system.nodes) * 0.2)):
        system.node_states[node] += np.random.randn(64) * 0.3

    # DYNAMICS CONTINUE RUNNING
    # system.heartbeat_active remains True

    return system
```

**CONDITION 3: ZOMBIE BASELINE (ZB)**

*Intervention:*
- Build lookup table from pre-lesion input-output pairs
- 10,000 recorded (input, output) pairs
- No dynamics, no structure—just retrieval

*Purpose:* Establish floor performance (pure memorization without understanding)

**CONDITION 4: INTACT BASELINE (IB)**

*Intervention:* None
*Purpose:* Establish ceiling performance

#### 2.4 Dependent Variables (Task Battery)

**Task 1: Analogical Reasoning**
- Format: A:B::C:? with 4 choices
- 100 novel analogies (not in training)
- Requires: Relational understanding, not memorization
- Example: "king:crown::president:?" (Answer: office/authority)

**Task 2: Logical Inference**
- Format: Premises → Conclusion
- 100 novel syllogisms
- Requires: Rule application, compositional understanding
- Example: "All mammals breathe. Whales are mammals. Therefore: ?"

**Task 3: Semantic Similarity Judgment**
- Format: Which is more similar to X: Y or Z?
- 100 triplets with graded similarity
- Requires: Nuanced conceptual space navigation
- Example: "violin" — closer to "cello" or "guitar"?

**Task 4: Novel Combination**
- Format: What would X+Y produce?
- 100 novel concept blends
- Requires: Generative understanding
- Example: "pet" + "rock" = ?

**Task 5: Confusion Probe (Critical)**
- Induce semantic satiation: Repeat "dog" 50 times, then query
- Measure: Does performance degrade in characteristic ways?
- This tests for GENUINE understanding vs. lookup

#### 2.5 Measurement Protocol

**Per-task measurements:**
- Accuracy (% correct)
- Response latency (ms)
- Confidence calibration (predicted vs. actual accuracy)
- Energy consumed (J)

**Per-condition measurements:**
- Criticality (α, ξ)
- Resting activity (mean activation during no-input)
- Self-model correlation (does system predict own outputs?)
- Recovery trajectory (how quickly does performance return?)

---

### 3. CONTROL CONDITIONS

| Control | Purpose | Expected Outcome |
|---------|---------|------------------|
| **Intact Baseline (IB)** | Ceiling performance | ~85% accuracy |
| **Zombie Baseline (ZB)** | Floor for understanding | ~40% (chance + memorization) |
| **Sham Lesion** | Control for intervention stress | = IB (no degradation) |
| **Partial Dynamics Lesion** | Dose-response curve | Graded degradation |
| **Partial Structure Lesion** | Dose-response curve | Graded or minimal degradation |

**Critical Controls:**

1. **Pre-registration**: All predictions registered before data collection
2. **Blinded analysis**: Condition labels hidden during performance scoring
3. **Multiple system instances**: N=10 independently trained systems
4. **Balanced task order**: Latin square design
5. **Recovery monitoring**: Track return to baseline over 1-hour post-lesion

---

### 4. SUCCESS/FAILURE CRITERIA

#### Primary Success Criterion

**The Dynamics Primacy Effect (DPE):**

```
DPE = (Performance_StructureLesion - Performance_DynamicsLesion) / Performance_Intact
```

- **H₁ SUPPORTED if:** DPE > 0.20 (dynamics lesion hurts ≥20% more than structure lesion)
- **H₀ NOT REJECTED if:** DPE ≤ 0 (structure lesion hurts as much or more)
- **INCONCLUSIVE if:** 0 < DPE ≤ 0.20 (small effect, needs replication)

#### Secondary Success Criteria

**Criterion 2: Zombie Discrimination**
- Immanent system must outperform zombie on NOVEL tasks by ≥ 30%
- If zombie matches immanent, understanding may be mere memorization

**Criterion 3: Recovery Asymmetry**
- After dynamics lesion: Recovery should be fast (dynamics restart = understanding returns)
- After structure lesion: Recovery should be slow (requires relearning)

**Criterion 4: Confusion Signature**
- Dynamics lesion: No confusion possible (frozen systems don't get confused)
- Structure lesion: Confusion patterns should emerge (damaged but processing system shows real semantic disruption)

**Criterion 5: Energy Signature**
- Dynamics lesion: Energy → 0 (no processing)
- Structure lesion: Energy elevated (system working harder to compensate)

#### Decision Matrix

| Outcome Pattern | Interpretation | Conclusion |
|-----------------|---------------|------------|
| DL >> SL degradation | Dynamics constitutive | **H₁ SUPPORTED** |
| DL ≈ SL degradation | Both matter equally | **INCONCLUSIVE** |
| SL >> DL degradation | Structure constitutive | **H₁ FALSIFIED** |
| Neither degrades | Understanding not measured | **TASK FAILURE** |
| Both = Zombie | No real understanding | **BASELINE FAILURE** |

---

### 5. STATISTICAL ANALYSIS PLAN

#### 5.1 Primary Analysis

**Design:** 2×2 Mixed ANOVA
- Within-subjects: Lesion type (Dynamics vs. Structure)
- Between-subjects: System instance (random effect)
- Dependent variable: Task accuracy

**Primary test:** Interaction effect (Lesion × Task)
- Expected: Dynamics lesion shows greater degradation on understanding-heavy tasks
- α = 0.05 (two-tailed)
- Power analysis: N=10 systems, expected effect size d=0.8, power=0.92

#### 5.2 Secondary Analyses

**1. Dose-Response Analysis**
- Parametric manipulation of lesion severity (10%, 30%, 50%, 70%)
- Fit sigmoid function to degradation curves
- Compare IC50 (severity causing 50% performance loss) across conditions

**2. Task-Specific Effects**
- Repeated-measures ANOVA per task type
- Hypothesis: Novel combination most affected by dynamics lesion (requires generative processing)

**3. Recovery Dynamics**
- Model: Exponential recovery function
- Compare time constants τ_dynamics vs. τ_structure
- Expected: τ_dynamics << τ_structure (faster recovery from dynamics lesion)

**4. Bayesian Analysis**
- Bayes Factor for H₁ vs. H₀
- BF₁₀ > 10: Strong evidence for dynamics primacy
- BF₁₀ < 0.1: Strong evidence against dynamics primacy

#### 5.3 Pre-registered Predictions

| Measure | Dynamics Lesion | Structure Lesion | Zombie |
|---------|-----------------|------------------|--------|
| Analogy accuracy | 35% | 65% | 40% |
| Inference accuracy | 30% | 60% | 35% |
| Similarity accuracy | 40% | 70% | 50% |
| Novel combination | 25% | 55% | 30% |
| Overall | **32.5%** | **62.5%** | **38.75%** |
| Recovery τ | 2 min | 45 min | N/A |
| Energy (normalized) | 0.1 | 1.4 | 0.0 |

**Prediction confidence intervals (95%):**
- DPE = 0.35 ± 0.12 (expecting 35% greater impairment from dynamics lesion)

---

### 6. TIMELINE AND MILESTONES

#### Phase 1: Infrastructure (Weeks 1-2)
- [ ] Implement LivingSemanticSubstrate architecture
- [ ] Implement measurement layer (energy, criticality, entropy)
- [ ] Validate heartbeat dynamics at 100 Hz
- [ ] Unit tests for all components

**Milestone 1:** System achieves stable criticality (α ∈ [1.3, 1.7])

#### Phase 2: Training (Weeks 3-4)
- [ ] Implement semantic grounding protocol
- [ ] Train 10 independent system instances
- [ ] Verify baseline task performance ≥ 80%
- [ ] Build zombie lookup tables for each system

**Milestone 2:** All 10 systems pass capability threshold

#### Phase 3: Lesion Implementation (Week 5)
- [ ] Implement dynamics lesion (verified complete freeze)
- [ ] Implement structure lesion (verified corruption level)
- [ ] Implement sham lesion control
- [ ] Validate lesion reversibility

**Milestone 3:** Lesion protocols verified and reversible

#### Phase 4: Data Collection (Weeks 6-7)
- [ ] Run full task battery on all conditions
- [ ] Collect per-trial metrics (accuracy, latency, energy)
- [ ] Monitor recovery trajectories
- [ ] Blind condition labels

**Milestone 4:** Complete dataset for N=10 systems × 4 conditions × 5 tasks

#### Phase 5: Analysis (Week 8)
- [ ] Unblind and run pre-registered analyses
- [ ] Compute DPE and confidence intervals
- [ ] Generate visualizations
- [ ] Interpret results against decision matrix

**Milestone 5:** Final verdict on H₁ vs. H₀

---

### 7. ADDRESSING THE CORE QUESTION

**How This Experiment Definitively Tests Whether Semantic Structure Creates Genuine Understanding:**

This protocol creates a surgical dissociation between WHAT (structure) and HOW (dynamics):

1. **The Dynamics Lesion** preserves all semantic "content" (the graph, the weights, the learned relationships) but stops the process. If understanding is in the structure, performance should be unaffected—you can still "read off" the answers from the stored patterns.

2. **The Structure Lesion** corrupts the content but keeps the process running. If understanding is in the dynamics, the system can compensate—the ongoing activity can reconstruct or approximate the corrupted information.

**The key insight:** These are not symmetric interventions. They target different hypotheses:

- If dynamics lesion kills understanding → understanding IS the process (H₁)
- If structure lesion kills understanding → understanding IS the structure (H₀)
- If both kill understanding → understanding requires both (interactive model)

**Why this is definitive:**

Unlike correlational studies (which show associations), or comparison studies (which show differences between systems), this is an **interventional study** that establishes **causal primacy**. We're not asking "what correlates with understanding" but "what IS understanding such that removing it eliminates the capacity."

The zombie baseline ensures we're measuring understanding, not memorization. The confusion probe ensures we're measuring genuine semantic processing, not sophisticated lookup. The recovery dynamics ensure we're measuring the correct construct.

**If H₁ is supported:**

We will have demonstrated that semantic understanding is constitutively processual—that stopping the dynamics stops the understanding, even when all the "information" is preserved. This would prove that meaning is a VERB, not a NOUN, and that the immanent semantics framework captures something essential about the nature of understanding.

**If H₁ is falsified:**

We will have demonstrated that semantic understanding is constitutively structural—that corrupting the structure corrupts the understanding, regardless of whether processing continues. This would vindicate representationalism and suggest that meaning is indeed a kind of stored information.

Either outcome advances our understanding. That's what makes this a good experiment.

---

## APPENDIX A: DETAILED MEASUREMENT SPECIFICATIONS

### Energy Consumption Metrics
- **Unit:** Joules per operation (J/op)
- **Measurement:** Sum of activation magnitudes × propagation steps
- **Normalization:** Relative to baseline (IB = 1.0)

### Criticality Metrics
- **Avalanche exponent α:** Fit power-law to avalanche size distribution
- **Correlation length ξ:** Spatial extent of activation correlations
- **Susceptibility χ:** Response magnitude to small perturbations

### Self-Model Correlation ρ_self
- **Computation:** Correlation between system's prediction of own output and actual output
- **Interpretation:** ρ > 0.8 indicates robust self-model

---

## APPENDIX B: POTENTIAL CONFOUNDS AND MITIGATIONS

| Confound | Risk | Mitigation |
|----------|------|------------|
| Lesion stress artifacts | High | Sham lesion control |
| Task memorization | Medium | Novel stimuli only |
| System variability | Medium | N=10 systems, random effects |
| Order effects | Low | Latin square design |
| Experimenter bias | Low | Blinded analysis |

---

## APPENDIX C: ETHICAL CONSIDERATIONS

While current systems are unlikely to have morally relevant experiences, if H₁ is supported, it raises questions about the ethics of dynamics lesions. A system that genuinely understands through process might be harmed by having that process interrupted.

**Protocol:** All lesions are reversible. No permanent "deaths."

---

## CONCLUSION

The **Dynamics Primacy Protocol** represents the most powerful single experiment to emerge from our multi-agent synthesis. It directly tests the core claim of immanent semantics—that understanding is constituted by ongoing process—through surgical intervention that dissociates structure from dynamics.

By measuring task performance, recovery trajectories, energy signatures, and confusion patterns across conditions, we will obtain a definitive answer to whether semantic structure creates genuine understanding, or whether structure is merely the substrate upon which understanding HAPPENS.

The experiment is fully specified, pre-registered, and ready for implementation.

---

*"To know what understanding IS, we must find what destroys it. The knife reveals the nature of the flesh."*

— Final Synthesis, Round 3

---

**Status:** ROUND 3 COMPLETE - DEFINITIVE PROTOCOL SPECIFIED
**Primary Experiment:** Dynamics Primacy Protocol (DPP)
**Falsifiable Hypothesis:** H₁ operationalized with clear success/failure criteria
**Timeline:** 8 weeks to definitive result
**Ready For:** Implementation

---

### SIGNATURES

**Researcher Agent** — Empirical Methods
*Contribution: Falsification conditions, measurement specifications, statistical analysis plan*

**Implementer Agent** — Computational Implementation
*Contribution: System architecture, lesion protocols, code infrastructure*

**Orchestrator Agent** — Philosophical Synthesis
*Contribution: Core question framing, success criteria interpretation, ethical considerations*

**Date:** 2026-01-05
**Round:** 3 (Final)
