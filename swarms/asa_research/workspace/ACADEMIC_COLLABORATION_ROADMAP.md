---
created: 2025-01-02 00:00
updated: 2026-01-02
---

# ASA Academic Collaboration Roadmap
## Strategic Research Plan Following Outreach to Pustejovsky and Palmer

**Date:** January 2, 2025
**Author:** System Architect
**Status:** Active — Awaiting Academic Response

---

## Executive Summary

Two foundational theoretical documents have been sent to leading academics:
- **James Pustejovsky** (Brandeis) — Creator of Generative Lexicon, Qualia Structure expert
- **Martha Palmer** (University of Colorado) — Creator of VerbNet, PropBank co-developer

This roadmap defines what validation work should proceed BEFORE hearing back, what materials to prepare FOR collaboration discussions, and which research directions deserve deepest investigation.

**Key Insight:** The 5-axis constraint framework directly maps to these experts' life work:
- Pustejovsky = Axis 3 (Qualia/Internal Decomposition)
- Palmer = Axis 2 (Valence Structure/Thematic Roles)

This is not coincidence — it reflects the genuine convergence of independent research traditions that the theoretical documents claim.

---

## Part 1: Immediate Validation Work (Before Academic Response)

### 1.1 Empirical Tests to Strengthen Theoretical Claims

The theoretical documents make strong claims about convergence across frameworks. These need empirical grounding.

#### Experiment 1: Qualia Coercion Test
**Purpose:** Validate Axis 3 (Internal Decomposition) claims using Pustejovsky's canonical examples

**Test Cases:**
```
"Mary began the book" → began [reading] the book (TELIC quale accessed)
"The author finished the book" → finished [writing] the book (AGENTIVE quale accessed)
"The book is long" → physical object (CONSTITUTIVE) vs content (TELIC)
"fast car" → TELIC (motion capability)
"fast food" → AGENTIVE (preparation time)
"fast driver" → behavioral property
```

**Method:**
1. Feed these sentences to trained ASA model
2. Extract attention patterns
3. Compare to baseline transformer attention
4. Measure: Does ASA attention align with quale-appropriate interpretations?

**Hypothesis:** ASA's predetermined structure should show different attention patterns for different coercion types — if not, Axis 3 needs refinement.

**Timeline:** 1 week
**Priority:** HIGH — This directly addresses Pustejovsky's expertise

---

#### Experiment 2: VerbNet Selectional Restriction Validation
**Purpose:** Validate Axis 2 (Valence Structure) using Palmer's VerbNet

**Test Cases:** Use VerbNet's selectional restrictions to predict attention blocking
```
"eat" requires: [+animate] Agent, [+concrete, +comestible] Patient
  ✓ "The cat ate the fish" — all restrictions satisfied
  ✗ "The idea ate the fish" — [+animate] violated
  ? "The company ate its competitors" — metaphorical extension

"think" requires: [+animate] Agent, [+abstract/+proposition] Theme
  ✓ "Mary thought about the problem"
  ✗ "*Mary thought the sandwich"
```

**Method:**
1. Expand VerbNet coverage from 468 to 1,000+ verbs
2. For each verb class, generate test sentences with:
   - Satisfied restrictions (should attend normally)
   - Violated restrictions (should show reduced attention or different pattern)
   - Metaphorical extensions (measure what happens)
3. Compare ASA vs baseline attention on violated vs satisfied

**Hypothesis:** ASA should show measurably different attention patterns when selectional restrictions are violated.

**Deliverable:** Quantitative analysis showing correlation between VerbNet predictions and attention behavior

**Timeline:** 2 weeks
**Priority:** HIGH — This directly addresses Palmer's expertise

---

#### Experiment 3: Cross-Axis Consistency Test
**Purpose:** Test whether the 5 axes are truly orthogonal or show dependencies

**Method:**
1. Take 200 diverse words
2. Annotate along all 5 axes:
   - Ontological Type (Thing/Event/State/Property)
   - Valence Structure (argument count, role types)
   - Qualia (Formal, Constitutive, Telic, Agentive scores)
   - Force Dynamics (Agonist/Antagonist/Neutral)
   - Geometric Position (abstraction level, semantic category)
3. Run correlation analysis across dimensions
4. Identify dependencies that violate orthogonality claim

**Hypothesis:** Some correlations are expected (verbs cluster with force dynamics, nouns cluster with qualia), but dimensions should not be fully redundant.

**Timeline:** 2 weeks
**Priority:** MEDIUM — Important for theoretical rigor

---

#### Experiment 4: Predetermined vs Learned Embedding Comparison
**Purpose:** Direct test of the "semantic periodic table" feasibility

**Method:**
1. Create small vocabulary (100 words) with manually assigned coordinates along all 5 axes
2. Train three models on toy corpus:
   - Learned embeddings (baseline)
   - Predetermined embeddings (from periodic table)
   - Hybrid (predetermined + learned residual)
3. Compare on:
   - Next-word prediction accuracy
   - Similarity structure (does predetermined cluster sensibly?)
   - Compositional generalization (novel combinations)

**Critical Question:** Can predetermined embeddings capture enough meaning to compete with learned?

**Timeline:** 3 weeks
**Priority:** HIGH — Core feasibility test for radical path

---

### 1.2 Prototype Implementations to Demonstrate Feasibility

#### Prototype 1: 100-Word Semantic Periodic Table
**Purpose:** Concrete demonstration of the framework

**Design:**
```
Word Selection (100 tokens):
- 20 common nouns (concrete: dog, table, water; abstract: idea, justice)
- 20 common verbs (action: run, eat, break; mental: think, know, want)
- 15 adjectives (physical: red, tall; psychological: happy, angry)
- 15 function words (determiners, prepositions, pronouns)
- 10 quantifiers/numbers
- 10 discourse markers
- 10 edge cases (polysemous: bank, light, run)

Coordinate Assignment:
Axis 1 (Ontological Type): 8-dimensional one-hot [Thing, Event, State, Place, Path, Property, Time, Amount]
Axis 2 (Valence): 3 features [arg_count, primary_role_type, selectional_strictness]
Axis 3 (Qualia): 4 features [formal, constitutive, telic, agentive] (0-1 scales)
Axis 4 (Force): 3 features [agonist_potential, antagonist_potential, neutral]
Axis 5 (Geometric): 2 features [abstraction_level, semantic_angle]

Total dimensions: 8 + 3 + 4 + 3 + 2 = 20
```

**Deliverable:** JSON file with 100 words, full coordinate annotations, design document explaining choices

**Timeline:** 2 weeks
**Priority:** HIGH — Foundation for all subsequent work

---

#### Prototype 2: Attention Visualization Tool
**Purpose:** Show academics exactly how ASA constraints manifest in attention

**Features:**
- Input any sentence
- Display attention heatmap (baseline vs ASA)
- Highlight which constraint axis is responsible for each blocking decision
- Color-code: POS compatibility (blue), VerbNet restriction (green), Binding Theory (orange), WordNet hierarchy (purple)
- Export publication-quality figures

**Technical Approach:**
- Build on existing ASA v2.2 codebase
- Add attribution tracking to bonding mask computation
- Create web interface or Jupyter notebook

**Timeline:** 2 weeks
**Priority:** HIGH — Essential for collaboration discussions

---

#### Prototype 3: Qualia-Aware Coercion Predictor
**Purpose:** Demonstrate Axis 3 in action

**Functionality:**
- Input: noun + context verb (e.g., "begin" + "book")
- Output: Predicted quale being accessed (Telic: reading)
- Show reasoning chain: verb "begin" requires eventive complement, "book" has Telic quale (reading), therefore coercion to "reading"

**Implementation:**
1. Hand-annotate qualia for 100 nouns from periodic table
2. Define coercion rules from Pustejovsky (1995)
3. Build inference engine
4. Evaluate on standard coercion test sets

**Timeline:** 3 weeks
**Priority:** MEDIUM-HIGH — Directly relevant for Pustejovsky discussion

---

## Part 2: Collaboration Preparation

### 2.1 Questions for James Pustejovsky (Qualia/Generative Lexicon Expert)

**Background:** Pustejovsky's Generative Lexicon (1995) is the foundation for Axis 3 (Internal Decomposition). He invented qualia structure.

#### Strategic Questions:

**On Qualia Coverage:**
1. "We've encoded 4 qualia roles (Formal, Constitutive, Telic, Agentive). Are there cases where these four are insufficient? Should we add granularity (sub-qualia) or additional roles?"

2. "How should we handle nouns where qualia values are highly context-dependent? For example, 'window' can highlight physical object (constitutive), aperture (formal), or interface element (functional)."

**On Coercion Mechanisms:**
3. "In our predetermined constraint framework, we need to decide: should coercion rules be hardcoded, or should we allow the model to learn coercion patterns given qualia annotations? What's your intuition on which would generalize better?"

4. "Your work distinguishes logical polysemy from contrastive ambiguity. How should we encode this distinction in the embedding space? Different coordinates, or different constraint behaviors?"

**On Computational Implementation:**
5. "We're experimenting with predetermined embeddings where qualia values are fixed coordinates. Have you seen any prior work attempting this, and what pitfalls should we anticipate?"

6. "The Generative Lexicon uses type composition and coercion operators (e.g., *THEME, *AGENTIVE). Could these be encoded as tensor operations in neural architectures, or is symbolic treatment necessary?"

**On Validation:**
7. "What benchmark datasets or phenomena would best test whether our qualia-aware attention captures the patterns your theory predicts?"

8. "Are there specific linguistic phenomena where you'd expect qualia-based constraints to dramatically outperform distributional approaches?"

---

### 2.2 Questions for Martha Palmer (VerbNet Creator)

**Background:** Palmer created VerbNet, co-developed PropBank, and has worked extensively on semantic role labeling and cross-linguistic verb semantics.

#### Strategic Questions:

**On VerbNet Integration:**
1. "We're using VerbNet selectional restrictions to predetermine attention masks. For the ~300 verb classes, we extract role requirements and type constraints. What coverage issues should we anticipate for real-world text?"

2. "VerbNet classes share syntactic alternations (causative, locative, etc.). Should these alternations be encoded as transformations in our embedding space, or is class membership sufficient?"

**On Selectional Restrictions:**
3. "How strict should selectional restriction enforcement be? Your work shows violations are often meaningful (metaphor, coercion). Should we allow 'soft' blocking that reduces attention rather than eliminating it?"

4. "VerbNet restrictions use features like [+animate], [+concrete]. What additional features would most improve coverage? Are there critical distinctions VerbNet doesn't capture?"

**On Thematic Roles:**
5. "We currently use ~10 thematic roles from VerbNet. Some theories propose 20+, others argue for 3-4 proto-roles. What granularity do you recommend for computational purposes?"

6. "How should we handle role ambiguity? For 'The door opened,' is 'door' Patient (affected) or Theme (moved)? Does this distinction matter computationally?"

**On Cross-Linguistic Validity:**
7. "Our constraint framework claims cross-linguistic validity based on convergent role inventories. VerbNet is English-focused. What evidence suggests these patterns transfer to typologically diverse languages?"

8. "Are there verb classes or semantic domains where VerbNet's organization is controversial or disputed? Where should we be cautious in relying on its structure?"

**On Future Directions:**
9. "If you were designing a 'periodic table' for verbs, what dimensions would you use beyond class membership? Event structure? Aspect? Force dynamics?"

10. "What would excite you most about a collaboration on this framework?"

---

### 2.3 Demonstration Materials to Prepare

#### For Both Collaborators:

**Demo 1: H6 Correlation Visualization**
- Show the 73.9% overlap between learned attention and linguistic constraints
- Highlight that this was measured on a baseline model NOT trained with constraints
- Implication: Transformers naturally learn what linguistic theory predicts

**Demo 2: Convergence Acceleration**
- Show training curves: ASA reaches baseline perplexity 21% faster
- Explain: Linguistic priors reduce what the model must learn from scratch

**Demo 3: Attention Heatmap Comparison**
- Side-by-side: Baseline attention vs ASA attention on same sentences
- Highlight differences and explain which constraint caused each

#### For Pustejovsky Specifically:

**Demo 4: Coercion Pattern Analysis**
- Run ASA on Generative Lexicon's canonical coercion examples
- Show how attention differs between:
  - "begin the book" (telic coercion)
  - "write the book" (no coercion)
  - "heavy book" (constitutive)
  - "interesting book" (telic/content)

#### For Palmer Specifically:

**Demo 5: VerbNet Class Behavior**
- Take verbs from different classes (motion, psych, consumption)
- Show how selectional restrictions affect attention
- Demonstrate handling of alternations (causative, locative)

---

## Part 3: Research Direction Prioritization

### 3.1 Which Constraint Dimension Needs Deepest Investigation?

#### Ranking (by urgency and value):

**1. Axis 3: Internal Decomposition (Qualia) — HIGHEST PRIORITY**

*Why:*
- Most theoretically novel contribution
- Directly addresses polysemy challenge
- Pustejovsky is THE expert and potential collaborator
- Current ASA implementation has NO qualia integration
- Could explain the ~26% of attention NOT captured by current constraints

*Key Questions:*
- How to computationally encode 4 qualia values?
- How to handle context-dependent quale activation?
- Can qualia predict type coercion outcomes?
- Does qualia-aware attention improve compositional generalization?

*Experiments Needed:*
- Qualia annotation for vocabulary subset
- Coercion prediction accuracy test
- Ablation: ASA + Qualia vs ASA without
- Comparison to Pustejovsky's test cases

---

**2. Axis 2: Valence Structure — HIGH PRIORITY**

*Why:*
- Already partially implemented (VerbNet ~468 verbs)
- Palmer is THE expert and potential collaborator
- Clear path to full implementation (6,800 VerbNet verbs)
- Highest current impact on H6 correlation

*Key Questions:*
- Does full VerbNet coverage improve H6 correlation?
- How to handle verbs not in VerbNet?
- Should selectional violations be hard-blocked or soft-penalized?
- Can we predict alternation patterns from embeddings?

*Experiments Needed:*
- Expand VerbNet coverage, measure correlation improvement
- Test on selectional restriction violation detection
- Cross-validate with FrameNet frames

---

**3. Axis 5: Geometric Position — MEDIUM-HIGH PRIORITY**

*Why:*
- Foundation for "semantic periodic table" vision
- Hyperbolic geometry could dramatically reduce dimensionality
- Determines how other axes combine
- Important for scaling to full vocabulary

*Key Questions:*
- Euclidean vs hyperbolic vs hybrid geometry?
- How to position abstract vs concrete concepts?
- Does radial position (abstraction) correlate with attention depth?
- Can geometry encode semantic domains?

*Experiments Needed:*
- Hyperbolic embedding of WordNet subtree
- Compare link prediction: Euclidean vs Poincare
- Test if abstraction level predicts attention layer

---

**4. Axis 1: Ontological Type — MEDIUM PRIORITY**

*Why:*
- Already implemented via POS tags and SUMO categories
- Well-understood from ontology research
- Current implementation is reasonably complete
- Less theoretical novelty

*Key Questions:*
- Is 8-category Jackendoff system optimal, or should we use SUMO's finer distinctions?
- How to handle systematic category shifts (noun-verb conversion)?
- Does ontological type interact with other axes?

*Experiments Needed:*
- Compare POS-only vs SUMO-enhanced ontology
- Test category shift handling
- Measure axis interaction

---

**5. Axis 4: Force Dynamics — MEDIUM-LOW PRIORITY (for now)**

*Why:*
- No computational implementation exists
- Talmy's framework is linguistically important but less computationally specified
- Most relevant for event semantics (subset of language)
- Harder to validate empirically

*Key Questions:*
- How to encode force vectors computationally?
- What vocabulary subset is force-dynamic?
- Does force encoding improve causative understanding?
- Can we derive force from VerbNet + aspect?

*Experiments Needed:*
- Annotate force features for verb subset
- Test on causative alternation prediction
- Compare to Talmy's original examples

---

### 3.2 Highest-Value Experiments

#### Tier 1: Must Do (Impact academic collaboration)

| Experiment | Purpose | Timeline | Owner |
|------------|---------|----------|-------|
| Qualia Coercion Test | Validate Axis 3 for Pustejovsky | 1 week | Researcher |
| VerbNet Coverage Expansion | Validate Axis 2 for Palmer | 2 weeks | Implementer |
| 100-Word Periodic Table | Concrete demonstrator | 2 weeks | Researcher |
| Attention Visualization Tool | Collaboration material | 2 weeks | Implementer |

#### Tier 2: Should Do (Strengthen theoretical claims)

| Experiment | Purpose | Timeline | Owner |
|------------|---------|----------|-------|
| Cross-Axis Consistency | Test orthogonality claim | 2 weeks | Critic |
| Predetermined vs Learned | Core feasibility test | 3 weeks | Researcher |
| Hyperbolic Embedding Test | Geometry decision | 2 weeks | Researcher |

#### Tier 3: Could Do (If resources allow)

| Experiment | Purpose | Timeline | Owner |
|------------|---------|----------|-------|
| Force Dynamics Prototype | Explore Axis 4 | 4 weeks | Researcher |
| FrameNet Integration | Additional validation | 3 weeks | Implementer |
| Cross-Lingual Test | Universality claim | 6 weeks | Researcher |

---

### 3.3 Gaps: Theoretical Work vs Computational Validation

| Gap | Type | Priority | Notes |
|-----|------|----------|-------|
| Qualia-to-embedding mapping | Theoretical + Computational | HIGH | Need theory for how qualia become coordinates |
| Polysemy resolution | Theoretical | HIGH | Core challenge; multiple sense vectors? |
| Cross-axis interaction | Theoretical | MEDIUM | Are axes truly independent? |
| Full VerbNet integration | Computational | HIGH | Engineering task, theory exists |
| Force dynamics formalization | Theoretical | MEDIUM | Talmy is qualitative, need quantitative |
| Hyperbolic composition | Computational | MEDIUM | How to compose in hyperbolic space? |
| Coercion rule derivation | Theoretical + Computational | MEDIUM | Can we derive rules from qualia? |
| Subword handling | Computational | LOW | How to assign coordinates to BPE tokens? |

---

## Part 4: Contingency Planning

### If Pustejovsky Responds Positively
- Offer collaboration on qualia annotation project
- Request access to Generative Lexicon test sets
- Propose joint paper on qualia-aware attention
- Ask for feedback on Axis 3 formalization

### If Palmer Responds Positively
- Offer collaboration on VerbNet expansion for neural models
- Request guidance on cross-lingual VerbNet resources
- Propose joint paper on selectional restriction attention
- Ask for feedback on Axis 2 formalization

### If Neither Responds (30-day timeout)
- Proceed with independent validation work
- Publish results on arXiv
- Seek alternative collaborators (Fillmore's students for FrameNet, NSM researchers)
- Continue building demonstrators

### If Response is Critical/Skeptical
- Address specific criticisms empirically
- Revise theoretical claims as needed
- Document which claims hold and which need modification
- Value honest feedback over agreement

---

## Part 5: Timeline Summary

```
Week 1-2 (Jan 2-15):
├── Qualia coercion experiment design and execution
├── VerbNet coverage expansion (468 → 1000+ verbs)
├── 100-word periodic table design (draft)
└── Attention visualization tool (prototype)

Week 3-4 (Jan 16-30):
├── Complete 100-word periodic table
├── Cross-axis consistency analysis
├── Finish attention visualization tool
├── Prepare Pustejovsky/Palmer question sets
└── Build coercion demonstrator

Week 5-6 (Jan 31 - Feb 13):
├── Predetermined vs learned embedding experiment
├── Hyperbolic embedding test
├── Refine demonstrators based on results
└── Prepare collaboration materials

Week 7-8 (Feb 14-28):
├── Analyze all experiment results
├── Update theoretical documents if needed
├── Finalize collaboration presentations
└── Await/respond to academic feedback

Decision Point (March 1):
├── If positive response: initiate collaboration
├── If no response: proceed independently
└── If critical response: revise and re-engage
```

---

## Part 6: Success Criteria

### For Validation Work:
- [ ] Qualia coercion test shows measurable attention difference across coercion types
- [ ] VerbNet expansion to 1000+ verbs maintains or improves H6 correlation
- [ ] 100-word periodic table coherently assigns coordinates to all tokens
- [ ] Cross-axis analysis shows low correlation (<0.3) between axes

### For Collaboration Preparation:
- [ ] Question sets reviewed and refined
- [ ] All demonstrators functional and visually clear
- [ ] Materials organized for 30-minute presentation
- [ ] Technical backup for any questions

### For Academic Engagement:
- [ ] Response received from at least one academic
- [ ] Feedback incorporated into research direction
- [ ] Path to publication or collaboration established

---

## Appendix A: Key Resources

### Pustejovsky's Work:
- "The Generative Lexicon" (1995) — Core text on qualia
- "The Syntax of Event Structure" (1991) — Event types and aspect
- "Type Theory and Lexical Decomposition" (2001) — Formal foundations

### Palmer's Work:
- "VerbNet: A Broad-Coverage, Comprehensive Verb Lexicon" (2005) — Core resource
- "PropBank: A Corpus for Semantic Role Labeling" (2005) — Role annotation
- "The Syntax and Semantics of the English Verb" (2017) — Recent synthesis

### Relevant Benchmarks:
- COGS (Compositional Generalization) — Tests systematic combination
- SCAN (Simplified Component Analysis) — Compositional reasoning
- SemCor (Sense-tagged Corpus) — Polysemy evaluation
- CoNLL SRL (Semantic Role Labeling) — Role prediction

---

## Appendix B: Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Academics don't respond | Medium | Medium | Independent validation path |
| Qualia encoding proves intractable | Medium | High | Fall back to learned qualia features |
| VerbNet expansion breaks existing results | Low | High | Incremental integration with testing |
| Predetermined embeddings fail feasibility test | Medium | High | Hybrid approach (predetermined + learned) |
| Cross-axis analysis shows redundancy | Medium | Medium | Merge axes, revise theory |
| Academic feedback is highly critical | Low | Medium | Revise claims, value honesty |

---

*Document created January 2, 2025*
*For ASA Research Swarm — Academic Collaboration Strategy*
