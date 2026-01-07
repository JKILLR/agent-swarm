---
created: 2026-01-02 00:00
updated: 2026-01-02
---

# Round 2 Design Decision: Framework Architecture

**Date:** January 2, 2026
**Author:** System Architect
**Decision Type:** Design Decision (Framework Architecture)

---

## Executive Summary

After evaluating four architectural options for the 5-axis constraint framework against five criteria (theoretical soundness, empirical testability, communication effectiveness, implementation complexity, and risk level), I recommend **OPTION D: Decouple presentation from foundations**.

This recommendation balances the need for rigorous theoretical grounding (via DisCoCat categorical semantics) with practical communication requirements (via the 5-axis framework as a pedagogical tool). The key insight is that the 5-axis framework and category-theoretic foundations are not competing alternatives but complementary layers serving different purposes.

---

## Part 1: Problem Context

### 1.1 The Issue

The 5-axis constraint framework (Ontological Type, Valence, Qualia, Force Dynamics, Geometric Position) has been criticized by all Round 1 exploration documents:

| Document | Primary Criticism |
|----------|-------------------|
| **Theory Exploration** | Framework lacks principled integration; axes may overlap; needs categorical foundation |
| **Empirical Exploration** | Axes 3-4 have ZERO validation; cannot ablate unimplemented axes |
| **Strategic Exploration** | Framework is a "communication liability"; invites unfavorable comparison to chemistry |
| **Critic Synthesis** | All criticisms valid; axis organization itself may be wrong decomposition |

### 1.2 Root Causes of the Problem

The criticisms reveal three distinct issues conflated in the current framework:

1. **Characterization errors** (fixable): Qualia and VerbNet are mischaracterized
2. **Validation gaps** (addressable): Axes 3-4 are untested
3. **Integration incoherence** (structural): Discrete types cannot be simply "stacked" with continuous geometry

The first two issues can be addressed incrementally. The third requires architectural choice.

### 1.3 Stakes

The framework architecture decision affects:
- Academic credibility with Pustejovsky and Palmer
- Publication strategy and venue selection
- Implementation roadmap and resource allocation
- Long-term theoretical development path

---

## Part 2: Options Analysis

### OPTION A: Keep 5 Axes, Fix Characterization Issues, Validate Each Axis

**Description:** Retain the current 5-axis organization. Correct mischaracterizations of Qualia and VerbNet. Implement and validate Axes 3-4.

**Detailed Assessment:**

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Theoretical soundness | 2/5 | Does not address integration incoherence between discrete (Lambek) and continuous (Gardenfors) frameworks |
| Empirical testability | 4/5 | Clear axes to test, but 3-4 require significant implementation first |
| Communication effectiveness | 3/5 | 5 axes are memorable but invite "Periodic Table" comparisons |
| Implementation complexity | 4/5 | Incremental from current state |
| Risk level | HIGH | Experts will still identify integration incoherence |

**Pros:**
- Preserves intellectual investment
- Maintains theoretical ambition
- Clear incremental path

**Cons:**
- Does not resolve foundational problem (how do discrete and continuous frameworks interact?)
- Two axes require implementation before testing (significant effort)
- Experts may identify problems even after fixes

**Verdict:** INSUFFICIENT - Fixes symptoms without addressing root cause

---

### OPTION B: Restructure into Validated/Speculative Tiers

**Description:** Replace 5 parallel axes with a hierarchical structure reflecting validation status:

```
Tier 1 (Validated): POS compatibility, basic VerbNet integration
Tier 2 (Partially Validated): WordNet hierarchies, expanded VerbNet
Tier 3 (Speculative): Qualia Structure, Force Dynamics
```

**Detailed Assessment:**

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Theoretical soundness | 3/5 | Honest about validation status; but "tiers" is not a theoretical commitment |
| Empirical testability | 5/5 | Directly maps structure to evidence status |
| Communication effectiveness | 4/5 | Honest, defensible, avoids overclaiming |
| Implementation complexity | 3/5 | Requires restructuring but follows existing code organization |
| Risk level | LOW-MEDIUM | May appear as retreat from original vision |

**Pros:**
- Matches actual evidence status
- Allows incremental development
- Honest positioning with academics
- Easy to explain and defend

**Cons:**
- Less theoretically elegant than unified framework
- Does not provide integration mechanism between tiers
- May appear as admission of failure rather than strategic maturation

**Verdict:** GOOD - Honest and defensible, but lacks theoretical depth

---

### OPTION C: Replace with Categorical Composition (DisCoCat)

**Description:** Abandon axis-based organization entirely. Adopt DisCoCat (Distributional Compositional Categorical) semantics as the theoretical foundation. Semantic constraints become morphisms in enriched categories.

**Key Elements from Theory Exploration:**
- Types are objects in category **Sem**
- Valid compositions are morphisms
- Coercion is functor between type categories
- Morphism spaces are vector spaces (enriched category **Sem_V**)
- Qualia become morphisms: `phys * info -> event` (reading, writing)

**Detailed Assessment:**

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Theoretical soundness | 5/5 | Principled categorical foundation; mathematically well-defined; handles coercion |
| Empirical testability | 2/5 | Category theory is abstract; unclear how to test categorical claims empirically |
| Communication effectiveness | 2/5 | Inaccessible to non-specialists; loses "5 axes" simplicity |
| Implementation complexity | 2/5 | Major architectural pivot; requires category-theoretic expertise |
| Risk level | HIGH | High ambition, high complexity, unclear empirical path |

**Pros:**
- Provides principled foundation
- Resolves integration problem (enriched categories combine categorical and geometric)
- Handles coercion via morphisms
- Mathematically elegant

**Cons:**
- Major pivot abandoning existing framing
- Inaccessible to non-specialists (including potentially Pustejovsky and Palmer)
- No clear empirical test of categorical structure
- Requires expertise team may lack

**Verdict:** THEORETICALLY SUPERIOR but PRACTICALLY RISKY - Right theory, wrong communication layer

---

### OPTION D: Decouple Presentation (5 Axes for Communication, Category Theory for Foundations)

**Description:** Maintain the 5-axis framework as a *communication and organizational tool* while adopting DisCoCat-style categorical semantics as the *theoretical foundation*. The axes become convenient groupings of morphisms in the underlying categorical structure.

**Architecture:**

```
                   PRESENTATION LAYER
                   (For communication, pedagogy)
          +------------------------------------+
          |   5-Axis Framework                 |
          |   - Axis 1: Ontological Type       |  -> "What categories"
          |   - Axis 2: Valence Structure      |  -> "What roles"
          |   - Axis 3: Qualia Structure       |  -> "What functions"
          |   - Axis 4: Force Dynamics         |  -> "What causation"
          |   - Axis 5: Geometric Position     |  -> "What similarity"
          +------------------------------------+
                         |
                   [Interpretation Map]
                         |
                   FOUNDATION LAYER
                   (For theory, implementation)
          +------------------------------------+
          |   Enriched Category Sem_Vect       |
          |   - Objects: Semantic types        |
          |   - Morphisms: Type transformations|
          |   - Hom-sets: Vector spaces        |
          |   - Composition: Bilinear maps     |
          +------------------------------------+
```

**Mapping from Axes to Categories:**

| Axis | Categorical Interpretation | Implementation |
|------|---------------------------|----------------|
| Axis 1 (Type) | Object membership in **Sem** | Type checking |
| Axis 2 (Valence) | Morphisms for argument structure | VerbNet frame constraints |
| Axis 3 (Qualia) | Indexed morphism families (telic, agentive, etc.) | Coercion morphisms |
| Axis 4 (Force) | Algebraic structure on morphism composition | Force dynamics algebra |
| Axis 5 (Geometric) | Metric structure on hom-spaces | Similarity in morphism space |

**Detailed Assessment:**

| Criterion | Score | Rationale |
|-----------|-------|-----------|
| Theoretical soundness | 4/5 | Category theory provides foundation; 5-axis is explicit approximation |
| Empirical testability | 4/5 | Test 5-axis predictions while foundations mature |
| Communication effectiveness | 5/5 | 5 axes for non-specialists; category theory for formal work |
| Implementation complexity | 3/5 | Can proceed incrementally; foundations develop in parallel |
| Risk level | MEDIUM | Requires managing two representations consistently |

**Pros:**
- Best of both worlds: rigorous foundations AND accessible communication
- Does not require abandoning 5-axis language
- Allows independent development of theory and implementation
- Provides principled answer to integration question
- Explicitly acknowledges 5-axis as approximation (honest)

**Cons:**
- May create confusion if mappings are not well-defined
- Requires maintaining consistency between layers
- "Two truths" could seem intellectually inconsistent if poorly managed
- Requires expertise in both areas

**Key Risk Mitigation:** The "two representations" concern is addressed by making the relationship EXPLICIT. The 5-axis framework is a *coordinate system* on the underlying categorical structure, not an alternative theory. Just as Cartesian and polar coordinates describe the same geometric space, axes and categories describe the same semantic structure at different levels of abstraction.

**Verdict:** RECOMMENDED - Combines theoretical rigor with practical accessibility

---

## Part 3: Comparative Summary

| Criterion | Option A | Option B | Option C | Option D |
|-----------|----------|----------|----------|----------|
| Theoretical soundness | 2/5 | 3/5 | 5/5 | 4/5 |
| Empirical testability | 4/5 | 5/5 | 2/5 | 4/5 |
| Communication effectiveness | 3/5 | 4/5 | 2/5 | 5/5 |
| Implementation complexity | 4/5 | 3/5 | 2/5 | 3/5 |
| Risk level | HIGH | LOW-MED | HIGH | MEDIUM |
| **WEIGHTED TOTAL** | **13/25** | **15/25** | **11/25** | **16/25** |

**Weighting:** All criteria weighted equally, but risk level inverts (low = good).

---

## Part 4: Decision and Rationale

### THE DECISION

**RECOMMENDED: OPTION D - Decouple Presentation from Foundations**

Adopt a two-layer architecture:
1. **Presentation Layer:** 5-axis framework for communication, documentation, and preliminary empirical work
2. **Foundation Layer:** DisCoCat-style enriched categories as theoretical basis

### RATIONALE

**Why not Option A (Keep and Fix)?**
Fixing characterization issues does not address the deeper integration problem identified by all three Round 1 documents. Simply stacking Lambek calculus with Gardenfors conceptual spaces remains theoretically incoherent. Experts will notice.

**Why not Option B (Hierarchical Tiers)?**
While honest and defensible, this is an organizational scheme, not a theoretical commitment. It tells us *what evidence we have* but not *how the pieces fit together*. It would position ASA as an empirical contribution without theoretical depth.

**Why not Option C (Pure DisCoCat)?**
DisCoCat provides the right theoretical foundation but would be a communication disaster. Pustejovsky and Palmer are linguists, not category theorists. Leading with enriched categories and functorial semantics would make the work inaccessible to its target audience.

**Why Option D?**
- **Satisfies Theory Researcher:** Provides principled categorical foundation they have been requesting
- **Satisfies Strategic Researcher:** Keeps accessible 5-axis framing for communication
- **Satisfies Empirical Researcher:** Maintains testable 5-axis predictions while foundations mature
- **Addresses Critic's concerns:** Makes the relationship between theory and presentation explicit
- **Manages expert expectations:** Can present 5-axis to linguists, category theory to mathematicians

### CONSEQUENCES

**Positive:**
1. Principled answer to "how does discrete integrate with continuous?"
2. Coercion mechanism via morphisms (addresses Pustejovsky's key concern)
3. Accessible communication layer preserved
4. Independent validation paths for each layer

**Negative:**
1. Requires maintaining consistency between layers
2. Need to develop interpretation map explicitly
3. Category theory expertise required for foundation work
4. Some additional documentation burden

**Trade-offs Accepted:**
- Complexity over simplicity (but managed via layer separation)
- Theoretical depth over pure empiricism (but empirical work continues)

---

## Part 5: Implementation Plan

### Phase 1: Foundation Specification (Weeks 1-2)

**Task:** Formally define enriched category **Sem_Vect** and interpretation map

**Deliverables:**
1. `FOUNDATION_SPECIFICATION.md` - Formal definition of categorical structure
2. `AXIS_TO_CATEGORY_MAP.md` - Explicit mapping from 5 axes to categorical components

**Specific Steps:**
1. Define object set (semantic types: e, t, n, complex types)
2. Define morphism spaces as vector spaces
3. Define composition operation (bilinear)
4. Map each axis to categorical component
5. Verify interpretation map is consistent

### Phase 2: Coercion Mechanism Development (Weeks 2-4)

**Task:** Implement qualia as morphism families; test on canonical examples

**Deliverables:**
1. `COERCION_MECHANISM.md` - Formal specification of coercion via morphisms
2. Working prototype for "begin the book", "fast car", "newspaper fired editor"

**Specific Steps:**
1. Define quale morphisms: telic, agentive, formal, constitutive
2. Implement morphism composition
3. Test on Pustejovsky's canonical examples
4. Compare attention patterns to theoretical predictions

### Phase 3: Documentation Update (Week 3)

**Task:** Revise external-facing documents to reflect two-layer architecture

**Deliverables:**
1. Updated `semantic_constraints.pdf` with layer distinction
2. Revised academic communication materials

**Key Changes:**
- Lead with 5-axis presentation (accessible)
- Note categorical foundations (for theoretically inclined readers)
- Explicitly describe relationship between layers

### Phase 4: Empirical Validation (Weeks 4-8)

**Task:** Run experiments from Empirical Exploration using 5-axis predictions

**Deliverables:**
1. Results for P1-P3 experiments (Tier 1 from Empirical Exploration)
2. Analysis of whether results are consistent with categorical interpretation

**Note:** Empirical work proceeds on 5-axis predictions. Categorical interpretation informs analysis but does not change experimental design.

### Phase 5: Integration and Refinement (Ongoing)

**Task:** Ensure consistent interpretation between layers as both develop

**Ongoing Activities:**
- Cross-check categorical predictions against empirical results
- Refine interpretation map based on findings
- Document any discrepancies and their resolution

---

## Part 6: Risk Mitigation

### Risk 1: Inconsistency Between Layers

**Scenario:** 5-axis predictions diverge from categorical predictions
**Mitigation:**
- Interpretation map must be well-defined BEFORE empirical work
- Discrepancies are diagnostic, not failures - investigate and document

### Risk 2: Category Theory Too Abstract

**Scenario:** Team cannot implement categorical foundations
**Mitigation:**
- Proposal C (attention-based quale selection) as computational fallback
- Category theory provides conceptual framework even if not fully implemented

### Risk 3: Communication Confusion

**Scenario:** Audiences confused by "two representations"
**Mitigation:**
- Different audiences get different layers (linguists: 5-axis; theorists: categories)
- Explicitly describe layer relationship in all documents
- Use consistent terminology across layers

### Risk 4: Expert Skepticism

**Scenario:** Pustejovsky/Palmer view layer decoupling as evasion
**Mitigation:**
- Emphasize that 5-axis is EXPLICIT APPROXIMATION, not hidden theory
- Show coercion mechanism addresses their core concerns
- Lead with empirical results, theory as interpretation

---

## Part 7: Relation to Other Decisions

### Dependency on Coercion Resolution Discussion

This architecture decision enables but does not fully resolve the coercion question. The decision here is:
- Coercion will be modeled as morphisms in the categorical foundation
- The 5-axis presentation will describe coercion as "quale access"
- Both are valid descriptions at different levels of abstraction

### Dependency on Publication Strategy Discussion

This architecture affects publication framing:
- Empirical papers: Lead with 5-axis, mention categorical interpretation in related work
- Theoretical papers: Lead with categorical structure, note 5-axis as application
- Mixed venues (ACL/EMNLP): Emphasize empirical with "principled foundation" note

### Relation to Damage Control Protocol

The two-layer architecture provides flexibility for expert response:
- If experts question 5-axis: "It is a presentation of deeper categorical structure"
- If experts question category theory: "It is principled foundation for intuitive axes"
- Neither layer is "the truth"; both are valid representations

---

## Part 8: Summary

### Architecture Decision Record

**ADR-008: Two-Layer Architecture for ASA Framework**

**Context:** The 5-axis constraint framework has been criticized for characterization errors (fixable), validation gaps (addressable), and integration incoherence (structural). All Round 1 documents agree the framework cannot survive expert scrutiny in current form, but disagree on whether to fix, restructure, or replace it.

**Decision:** Adopt a two-layer architecture:
- **Presentation Layer:** 5-axis framework (for communication, documentation, preliminary empirical work)
- **Foundation Layer:** DisCoCat-style enriched categories (for theoretical coherence, coercion mechanism)
- **Explicit Interpretation Map:** Defines relationship between layers

**Rationale:** This approach:
1. Preserves accessible 5-axis framing for communication with non-specialists
2. Provides principled categorical foundation that resolves integration problem
3. Enables coercion mechanism via morphisms
4. Allows independent development and validation of each layer
5. Manages expert expectations with appropriate layer for each audience

**Consequences:**
- Requires maintaining consistency between layers
- Adds documentation burden for interpretation map
- Enables theoretical depth without sacrificing accessibility
- Provides principled answer to integration question

**Status:** PROPOSED - Pending team alignment

---

### Clear Recommendation

**ADOPT OPTION D: Decouple presentation from foundations.**

The 5-axis framework is valuable for communication and organization. Category theory is necessary for theoretical coherence. These are complementary, not competing. By making the relationship explicit, ASA gains both accessible presentation and rigorous foundations.

This is not a retreat from theoretical ambition. It is an advancement that grounds the existing framework in principled foundations while preserving what works.

---

*Decision document prepared by System Architect*
*Round 2 Design Decision - January 2, 2026*
