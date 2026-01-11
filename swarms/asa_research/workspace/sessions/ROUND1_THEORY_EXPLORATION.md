---
created: 2026-01-02 00:00
updated: 2026-01-02
---

# Round 1: Theoretical Exploration for ASA
## New Directions for Atomic Semantic Architecture

**Date:** January 2, 2026
**Author:** Theory Researcher
**Status:** Initial Exploration - Round 1

---

## Executive Summary

This document explores theoretical directions to address the critical issues identified in ASA's current formulation: (1) the coercion mechanism gap, (2) framework integration incoherence between discrete and continuous representations, and (3) the need for rigorous mathematical foundations.

**Key Findings:**

1. **Coercion Mechanism**: Category-theoretic approaches, specifically functorial semantics and enriched categories, offer a principled way to model coercion as morphisms between types rather than static feature access. The DisCoCat framework provides a template for combining categorical structure with vector space semantics.

2. **Framework Integration**: The tension between discrete Lambek calculus and continuous Gardenfors spaces can be resolved through *graded type theory* or *fuzzy categorical semantics*, where typing judgments carry continuous confidence values and categorical composition preserves graded structure.

3. **Alternative Foundations**: Abstract Meaning Representation (AMR) extensions and Conceptual Role Semantics offer complementary perspectives that may strengthen specific ASA axes, particularly for argument structure (Axis 2) and inferential relations.

4. **Mathematical Formalization**: ASA would benefit from proving (a) axis orthogonality conditions, (b) coercion compositionality, and (c) constraint satisfaction soundness relative to natural language acceptability judgments.

**Recommended Priority:** Develop a category-theoretic coercion mechanism first, as this addresses both the immediate Pustejovsky concern and provides a principled foundation for framework integration.

---

## Part 1: Coercion Mechanism Proposals

### 1.1 The Core Problem

The current ASA framework treats Qualia as a "4D feature vector" with static coordinates along Formal, Constitutive, Telic, and Agentive dimensions. This fundamentally mischaracterizes Pustejovsky's Generative Lexicon, where qualia are *generative operators* that enable three types of coercion:

| Coercion Type | Mechanism | Example |
|--------------|-----------|---------|
| **Type Coercion** | Verb requires event type; noun supplies via quale | "begin the book" -> begin [reading] |
| **Selective Binding** | Adjective selects quale from noun | "fast car" -> motion quale; "fast food" -> preparation quale |
| **Co-composition** | Verb and noun mutually constrain interpretation | "bake a potato" vs "bake a cake" |

The question is: How can predetermined embeddings support these dynamic operations?

### 1.2 Proposal A: Functorial Coercion via Enriched Categories

**Theoretical Foundation:** Category theory provides a natural language for compositional semantics. In this framework:
- Types are objects in a category
- Valid compositions are morphisms between types
- Coercion is a *functor* between type categories

**Formalization:**

Let **Sem** be the category of semantic types with:
- Objects: Types including atomic types (e, t, n) and complex types (e.g., phys * info for "book")
- Morphisms: Type coercion operations

Define an *enriched category* **Sem_V** where hom-sets are not mere sets but vector spaces:

```
Hom_V(A, B) = V   (a vector space of coercion possibilities)
```

For a noun like "book" with dot type `phys * info`:
- The Telic quale provides a morphism: `phys * info -> event` (reading)
- The Agentive quale provides: `phys * info -> event` (writing)

**Crucially:** These morphisms live in a vector space, so we can represent *which* coercion is preferred in context as a weighted combination:

```
coerce(book, begin) = alpha * telic_morphism + beta * agentive_morphism
```

where alpha >> beta because "begin" typically selects telic interpretation.

**How Predetermined Embeddings Support This:**

Each noun embedding encodes not just a position but a *bundle of morphisms*:

```python
class QualiaEmbedding:
    formal: Vector      # What category does this belong to?
    constitutive: Vector  # What is it made of?
    telic: Morphism     # Function from self-type to event-type
    agentive: Morphism  # Function from self-type to event-type
```

The morphisms are represented as *matrices* (linear maps) that transform the noun's representation into an event representation when composed with an eventive verb.

**Mathematical Details:**

Given:
- Noun n with embedding v_n in V (dimension d)
- Telic quale represented as matrix T_n: V -> V_event
- Verb "begin" requiring event-type argument

The coercion operation is:
```
coerced(n) = T_n @ v_n
```

This is computable from predetermined structure: T_n is part of the lexical entry for n.

### 1.3 Proposal B: Type-Theoretic Coercion with Subtyping

**Theoretical Foundation:** Rather than treating coercion as external type-shifting, model it as *subtype polymorphism* with explicit coercion paths.

In Pustejovsky's dot objects, "book" has type `phys.info` where the dot indicates a complex type that can project to either component:

```
phys.info <: phys   (via pi_1)
phys.info <: info   (via pi_2)
```

**Formalization:**

Extend ASA's type system with:

1. **Dot types**: `A.B` denotes complex types
2. **Projection coercions**: `pi_1: A.B -> A` and `pi_2: A.B -> B`
3. **Quale-indexed projections**: Each quale provides a specific coercion path

```
book : phys.info

FORMAL(book) = physical_object
CONSTITUTIVE(book) = {pages, binding, ink}
TELIC(book) = lambda x. read(x)    -- a function!
AGENTIVE(book) = lambda x. write(x)
```

**Coercion Rule:**

When a verb V requires type T and receives argument of type A.B where A.B is not subtype of T:

1. Check if TELIC(A.B) has result type compatible with T
2. If yes, apply: V(TELIC(arg)(arg))
3. If no, check AGENTIVE, then fail

**Implementation in Predetermined Embeddings:**

```python
def coerce(noun_embedding, required_type, context_verb):
    """
    Attempt to coerce noun to required type using qualia.
    """
    if subtype(noun_embedding.type, required_type):
        return noun_embedding  # No coercion needed

    # Try telic coercion
    telic_result = noun_embedding.telic.apply(noun_embedding)
    if subtype(telic_result.type, required_type):
        return telic_result

    # Try agentive coercion
    agentive_result = noun_embedding.agentive.apply(noun_embedding)
    if subtype(agentive_result.type, required_type):
        return agentive_result

    # Coercion fails - composition is anomalous
    return AnomalyMarker(noun_embedding, required_type)
```

### 1.4 Proposal C: Attention-Based Quale Selection

**Theoretical Foundation:** Rather than computing coercion through explicit type operations, use the attention mechanism itself to implement quale selection.

**Key Insight:** In a transformer, attention weights determine which aspects of a representation contribute to the output. If qualia are represented as separate attention heads or subspaces, the context can "attend to" the relevant quale.

**Implementation Sketch:**

1. **Multi-Head Qualia Representation**: For nouns, use 4 dedicated attention heads corresponding to the 4 qualia:
   - Head 1: Formal quale
   - Head 2: Constitutive quale
   - Head 3: Telic quale
   - Head 4: Agentive quale

2. **Context-Driven Selection**: The verb embedding contains a "quale selector" that gates which heads are active:
   ```
   "begin" has selector: [0.1, 0.1, 0.7, 0.1]  -- strongly prefers telic
   "finish" has selector: [0.1, 0.1, 0.4, 0.4]  -- telic or agentive depending on subject
   ```

3. **Composition**: When "begin" attends to "book":
   ```
   attended_book = sum(selector_i * head_i(book) for i in 1..4)
   ```
   This implements selective binding through attention gating.

**Advantages:**
- Integrates naturally with transformer architecture
- Quale selection is differentiable and learnable
- Predetermined structure (which heads map to which qualia) combines with learned selection weights

**Disadvantages:**
- Less formally principled than category-theoretic approach
- Requires architectural modifications
- May not generalize beyond trained distribution

### 1.5 Comparative Analysis of Coercion Proposals

| Proposal | Formalism | Predetermined? | Handles Dot Objects? | Implementable? |
|----------|-----------|----------------|---------------------|----------------|
| **A: Functorial** | Category theory | Yes (morphisms in embedding) | Yes | Medium difficulty |
| **B: Subtyping** | Type theory | Yes (type projections) | Yes | Medium difficulty |
| **C: Attention-Based** | Neural attention | Partially (head assignment) | Partially | Easy |

**Recommendation:** Pursue Proposal A (Functorial Coercion) as the primary theoretical direction, with Proposal C as a computational approximation for implementation.

---

## Part 2: Framework Integration Theory

### 2.1 The Incompatibility Problem

ASA currently "stacks" incompatible frameworks:

| Framework | Ontology | Operations | Key Property |
|-----------|----------|------------|--------------|
| Lambek Calculus | Discrete types | Logical derivation | Proofs are programs |
| Gardenfors Conceptual Spaces | Continuous geometry | Vector operations | Convexity = naturalness |
| Talmy Force Dynamics | Image-schematic | Force interaction | Embodied grounding |

These have different foundational commitments:
- Lambek: Compositional meaning is logical type inference
- Gardenfors: Meaning is geometric position in quality spaces
- Talmy: Meaning is grounded in bodily force experience

Simply concatenating representations from each framework does not produce a coherent theory.

### 2.2 Reconciliation Strategy 1: Graded Type Theory

**Core Idea:** Replace discrete typing judgments with graded judgments that carry continuous confidence values.

In standard type theory:
```
x : A   (x has type A - binary)
```

In graded type theory:
```
x :_r A   (x has type A with grade r in [0,1])
```

**Application to ASA:**

A word like "furniture" might have graded type:
```
furniture :_0.9 COUNT
furniture :_0.7 MASS
```

This captures that "furniture" is predominantly count but exhibits mass-like behavior in some contexts ("much furniture" is acceptable).

**Composition in Graded Types:**

When composing f: A -> B with x :_r A:
```
f(x) :_s B   where s = g(r, type_strictness(f))
```

The grade propagates through composition, allowing continuous "compatibility scores" rather than binary well-formedness.

**Benefits:**
- Captures gradient acceptability judgments
- Preserves categorical structure while adding continuous dimensions
- Aligns with psycholinguistic evidence of graded grammaticality

### 2.3 Reconciliation Strategy 2: Enriched Categories over Vector Spaces

**Core Idea:** Use enriched category theory to have categorical structure where morphisms live in vector spaces.

In an ordinary category:
```
Hom(A, B) is a set
```

In a category enriched over Vect:
```
Hom(A, B) is a vector space
```

**How This Reconciles Lambek + Gardenfors:**

- **Objects** are semantic types (from Lambek)
- **Morphisms** are vectors encoding how one type transforms to another (from Gardenfors)
- **Composition** is bilinear (respects both categorical and geometric structure)

**Formal Definition:**

Let **Sem_Vect** be the category where:
- Objects: Semantic types {e, t, n, e->t, (e->t)->t, ...}
- Hom(A,B) = R^d (d-dimensional vector space)
- Composition: Hom(A,B) x Hom(B,C) -> Hom(A,C) via bilinear map

A lexical item with type A/B (Lambek functor type) is assigned:
- Its type A/B (discrete)
- A vector in Hom(B, A) (continuous)

**Composition Example:**

"the" : NP/N with vector v_the in Hom(N, NP)
"cat" : N with vector v_cat in Hom(1, N)  [1 is terminal object]

"the cat" : NP with vector = v_the o v_cat (morphism composition)

This is essentially DisCoCat (Coecke, Sadrzadeh, Clark 2010) but with richer type structure.

### 2.4 Reconciliation Strategy 3: Dual-Space Architecture

**Core Idea:** Maintain separate spaces for discrete and continuous representations, with explicit interface operations.

**Architecture:**

```
                    +------------------+
                    |  Discrete Space  |  (Lambek types, logical composition)
                    +--------+---------+
                             |
                      [Interface Layer]
                             |
                    +--------v---------+
                    | Continuous Space |  (Gardenfors geometry, similarity)
                    +------------------+
```

**Interface Operations:**

1. **Discretize**: Map continuous position to nearest discrete type
   ```
   discretize: V -> Type
   discretize(v) = argmax_T P(T | v)
   ```

2. **Embed**: Map discrete type to prototypical continuous position
   ```
   embed: Type -> V
   embed(T) = centroid(all instances of T)
   ```

3. **Compose_hybrid**: Combine discrete type-checking with continuous similarity
   ```
   compose(a, b) =
     if type_compatible(a.type, b.type):
       return continuous_compose(a.vector, b.vector)
     else:
       return coerce_and_compose(a, b)  # Attempt coercion
   ```

**Benefits:**
- Preserves each framework's strengths in its native representation
- Makes interface explicit and inspectable
- Allows gradual refinement of interface operations

**Drawbacks:**
- Introduces complexity at the interface
- Must ensure interface operations are semantically meaningful
- May lose unified theoretical elegance

### 2.5 Recommendation for Framework Integration

**Primary Approach:** Enriched categories (Strategy 2) provide the most principled reconciliation:
- Categorical structure handles compositionality (Lambek's insight)
- Vector-space enrichment handles similarity and gradience (Gardenfors' insight)
- The combination is mathematically well-defined

**Secondary Approach:** Dual-space architecture (Strategy 3) as an implementation strategy that approximates the enriched category structure while being more tractable computationally.

---

## Part 3: Alternative Theoretical Foundations

### 3.1 DisCoCat (Distributional Compositional Categorical Semantics)

**Source:** Coecke, Sadrzadeh, Clark (2010), "Mathematical Foundations for a Compositional Distributional Model of Meaning"

**Core Ideas:**
- Grammatical types from pregroup grammar determine composition structure
- Word meanings are vectors (nouns) or tensors (relational words)
- Composition is tensor contraction guided by grammatical types

**Relevance to ASA:**

DisCoCat already solves some of ASA's problems:
1. **Type-driven composition**: Grammatical types determine how meanings compose
2. **Geometric semantics**: Meanings live in vector spaces
3. **Principled interaction**: Category theory unifies structure and content

**What DisCoCat Lacks That ASA Addresses:**
- No explicit qualia structure
- No force dynamics
- No selectional restrictions beyond type compatibility
- Limited treatment of coercion

**Synthesis Opportunity:** ASA could adopt DisCoCat's categorical compositional framework while enriching it with:
- Qualia-indexed morphisms (from Pustejovsky)
- Force-dynamic vectors (from Talmy)
- Hierarchical type structure (from VerbNet/FrameNet)

### 3.2 Abstract Meaning Representation (AMR) Extensions

**Source:** Banarescu et al. (2013), "Abstract Meaning Representation for Sembanking"

**Core Ideas:**
- Sentence meaning as rooted, directed, acyclic graphs
- Nodes are concepts; edges are semantic relations
- Abstracts away from syntactic variation

**AMR Example:**
```
"The boy wants to go"
(w / want-01
   :ARG0 (b / boy)
   :ARG1 (g / go-02
            :ARG0 b))
```

**Relevance to ASA Axis 2 (Valence Structure):**

AMR's frame-based argument structure aligns well with ASA's valence axis:
- PropBank-style numbered arguments (ARG0, ARG1, ...)
- Explicit role labels for semantic relations
- Compositional graph structure

**Potential Enhancement:** Extend AMR with:
- Qualia annotations on concept nodes
- Force-dynamic edge types
- Type constraints on argument positions

**AMR + Qualia Example:**
```
"John began the book"
(b / begin-01
   :ARG0 (j / john)
   :ARG1 (b2 / book
            :QUALIA-ACCESS telic
            :COERCED-TO read-01))
```

### 3.3 Conceptual Role Semantics

**Source:** Block (1986), Harman (1982)

**Core Ideas:**
- Meaning is determined by inferential role in a network
- A concept's meaning = its connections to other concepts
- Holistic: meaning depends on entire network

**Relevance to ASA:**

Conceptual Role Semantics suggests that ASA's "constraint dimensions" might reduce to inferential patterns:
- Ontological type = which inferences are licensed (physical -> has location)
- Qualia = which functional inferences hold (book.telic = readable)
- Force dynamics = which causal inferences apply

**Potential Formalization:**

Model each concept as a node in an inference graph:
```
concept(book):
  infers: [physical_object, artifact, readable, writable, purchasable]
  inferred_by: [novel, textbook, manual, ...]
  mutually_exclusive: [event, property, ...]
```

The "embedding" of a concept is its pattern of inferential connections, which can be represented as a vector over possible inferences.

**Synthesis with ASA:**
- Axes 1-4 define *which inferential patterns* are tracked
- Axis 5 (geometry) represents *similarity of inferential patterns*
- Coercion = temporary modification of inferential connections

### 3.4 Dynamic Semantics

**Source:** Heim (1982), Kamp & Reyle (1993), Groenendijk & Stokhof (1991)

**Core Ideas:**
- Meaning is not static truth conditions but context change potential
- Sentence meaning = function from input context to output context
- Handles anaphora, presupposition, discourse dynamics

**Relevance to ASA:**

ASA currently treats meanings as static constraint bundles. Dynamic semantics suggests meanings should be *context update functions*:

```
[[sentence]] : Context -> Context
```

**Potential Integration:**

Model ASA constraints as affecting context update:
```
[[the book]] = lambda c. c + {x : book(x) & accessible(x)}
[[begin]] = lambda c. lambda x.
    let y = coerce_to_event(x, c) in
    c + {e : begin(e) & theme(e, y)}
```

Here, coercion is context-dependent: the function `coerce_to_event` consults the current context to determine which quale to access.

**Benefits:**
- Naturally handles context-dependent coercion
- Provides compositional framework for discourse
- Aligns with psycholinguistic processing models

**Drawbacks:**
- More complex than static semantics
- Harder to implement in neural architectures
- May be overkill for ASA's initial goals

### 3.5 Comparative Assessment of Alternative Foundations

| Framework | Strengths for ASA | Weaknesses | Integration Effort |
|-----------|-------------------|------------|-------------------|
| **DisCoCat** | Categorical composition, geometric semantics | No qualia, no force dynamics | Low (closest match) |
| **AMR Extensions** | Explicit argument structure, graph representation | No continuous geometry | Medium |
| **Conceptual Role** | Inferential patterns, network structure | Holism problems, no clear geometry | Medium |
| **Dynamic Semantics** | Context-sensitivity, update semantics | Complexity, implementation difficulty | High |

**Recommendation:**
- **Primary**: Adopt DisCoCat as categorical foundation, enrich with ASA-specific structure
- **Secondary**: Use AMR-style representations for explicit argument structure annotation
- **Tertiary**: Incorporate dynamic semantics insights for coercion mechanism

---

## Part 4: Mathematical Formalization Opportunities

### 4.1 Axis Orthogonality Theorem

**Claim (to prove or refute):** The five constraint axes are orthogonal in the sense that knowing a concept's value on one axis provides no information about its value on another.

**Formal Statement:**

Let X_1, ..., X_5 be random variables representing a concept's position on each axis. Orthogonality holds if:
```
I(X_i; X_j) = 0   for all i != j
```
where I is mutual information.

**Evidence Needed:**
- Large-scale annotation of concepts on all 5 axes
- Statistical analysis of axis correlations
- Potential counter-examples (Qualia and Ontological Type seem correlated)

**Why This Matters:** If axes are not orthogonal, the framework is either redundant (axes can be reduced) or the axes need refinement.

### 4.2 Coercion Compositionality Theorem

**Claim:** Coercion operations compose in predictable ways.

**Formal Statement:**

Let C be the category of coercion operations. For any composable pair of coercions c1: A -> B and c2: B -> C:
```
c2 o c1 is a valid coercion: A -> C
```

And the semantic effect is compositional:
```
[[c2 o c1]](x) = [[c2]]([[c1]](x))
```

**Evidence Needed:**
- Identification of coercion operation inventory
- Empirical testing of composed coercions
- Formal proof that composition preserves meaning

**Example Test Case:**
```
"begin the long book"
1. "long" selects constitutive quale (long in pages)
2. "begin" coerces to telic quale (reading)

Does "begin the long book" = begin(coerce_telic(modify_constitutive(book, long)))?
```

### 4.3 Constraint Satisfaction Soundness

**Claim:** If ASA predicts a combination is well-formed, human judgments agree (soundness). If ASA predicts anomaly, humans judge it anomalous (completeness).

**Formal Statement:**

Let J: Sentence -> {acceptable, anomalous} be human judgment function.
Let A: Sentence -> {satisfies_constraints, violates_constraints} be ASA's prediction.

**Soundness:** A(s) = satisfies -> J(s) = acceptable
**Completeness:** J(s) = acceptable -> A(s) = satisfies

**Evaluation Method:**
- Use acceptability judgment datasets (CoLA, BLiMP)
- Compute ASA's constraint satisfaction
- Measure precision (soundness) and recall (completeness)

### 4.4 Hierarchy Embedding Theorem

**Claim:** The hyperbolic embedding component of Axis 5 can represent arbitrary taxonomic hierarchies with bounded distortion.

**Formal Statement:**

For any tree T with n nodes and depth d, there exists an embedding f: T -> H^k (k-dimensional hyperbolic space) such that:
```
(1 - epsilon) * d_T(u,v) <= d_H(f(u), f(v)) <= (1 + epsilon) * d_T(u,v)
```

where d_T is tree distance and d_H is hyperbolic distance, and k = O(log n).

**Status:** This is essentially proven in Nickel & Kiela (2017). What remains is to verify that:
- ASA's specific taxonomies (WordNet, SUMO) satisfy the theorem's conditions
- The embedding dimension is sufficient for practical use
- The distortion bounds are acceptable for downstream tasks

### 4.5 Force Dynamics Algebra

**Opportunity:** Formalize Talmy's force dynamics as an algebraic structure.

**Proposed Algebra:**

Define a *force dynamics algebra* FD with:
- Carrier set: {compel, block, resist, enable, divert, ...}
- Binary operation: force interaction (+)
- Identity: null_force
- Properties: (FD, +) is a commutative monoid

**Axioms:**
```
compel + resist = struggle
compel + enable = amplified_compel
block + divert = redirected_block
```

**Application:** Force dynamics vectors in ASA embeddings could be elements of this algebra, with composition defined by the + operation.

**Why This Matters:** Currently, Force Dynamics (Axis 4) has no computational implementation. An algebraic formalization would enable:
- Formal inference of force-dynamic consequences
- Compositional computation of event structure
- Testable predictions for causative alternations

---

## Part 5: Risks and Open Questions

### 5.1 Theoretical Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Category theory is too abstract for implementation | Medium | High | Develop concrete computational instantiation |
| Graded types lose compositionality | Low | High | Prove grade preservation under composition |
| Framework integration introduces inconsistency | Medium | Medium | Formal consistency proof for combined system |
| Alternative foundations are incompatible with ASA | Low | Medium | Identify specific integration points |

### 5.2 Open Questions Requiring Further Research

**Q1: What is the right level of type granularity?**
- Too coarse: Loses predictive power
- Too fine: Combinatorial explosion, sparse coverage
- Need empirical study to find optimal granularity

**Q2: How do metaphorical extensions interact with coercion?**
- "The company devoured its competitors"
- This involves both metaphor and argument coercion
- No current theory handles both uniformly

**Q3: Can predetermined structure support novel compositions?**
- ASA assumes constraints are predetermined
- But language constantly produces novel combinations
- Need theory of productive coercion, not just stored coercions

**Q4: How do the 5 axes interact in composition?**
- Current framework treats axes independently
- But real composition involves coordination across axes
- Need multi-axis composition theory

**Q5: What is lost in vectorization?**
- Any fixed-dimensional vector is a lossy representation
- What aspects of meaning are necessarily lost?
- Can we characterize the information loss formally?

### 5.3 Potential Failures and Contingencies

**If category-theoretic coercion proves intractable:**
- Fall back to Proposal C (attention-based quale selection)
- Acknowledge as engineering approximation, not principled theory

**If axis orthogonality fails:**
- Merge correlated axes
- Develop theory of axis interaction
- May strengthen rather than weaken framework

**If no framework integration strategy works:**
- Commit to single foundational framework
- Treat others as useful heuristics, not theoretical components
- Loss of synthesis claim, but gain in coherence

---

## Part 6: Recommended Priorities for Further Research

### Priority 1: Functorial Coercion Development (Weeks 1-2)

**Task:** Fully specify the enriched category semantics for coercion
**Deliverable:** Formal definition + worked examples for canonical cases
**Success Criterion:** Can explain "begin the book" mathematically

**Specific Steps:**
1. Define the category **Sem_Vect** formally
2. Specify morphism spaces for core type pairs
3. Define composition operation
4. Work through Pustejovsky's canonical examples
5. Identify what is predetermined vs. contextual

### Priority 2: Graded Type System Specification (Weeks 2-3)

**Task:** Develop graded type theory for ASA
**Deliverable:** Type system definition with grade propagation rules
**Success Criterion:** Can handle gradient acceptability judgments

**Specific Steps:**
1. Review existing graded type theory literature
2. Adapt to ASA's specific types
3. Define grade combination rules
4. Test on known gradient constructions
5. Integrate with coercion mechanism

### Priority 3: DisCoCat Integration Study (Week 3)

**Task:** Determine how DisCoCat can serve as ASA's categorical foundation
**Deliverable:** Mapping document: DisCoCat <-> ASA correspondences
**Success Criterion:** Clear path to implementation

**Specific Steps:**
1. Detailed DisCoCat review
2. Identify enrichments needed for ASA
3. Propose hybrid architecture
4. Estimate implementation effort

### Priority 4: Mathematical Formalization (Weeks 3-4)

**Task:** Prove or refute axis orthogonality; define force dynamics algebra
**Deliverable:** Technical report with proofs/counterexamples
**Success Criterion:** Clear status of orthogonality claim; FD algebra defined

**Specific Steps:**
1. Gather axis annotation data (or design annotation study)
2. Compute axis correlations
3. Formalize force dynamics operations
4. Develop algebra and verify properties

### Priority 5: Synthesis Document (Week 4)

**Task:** Integrate findings into coherent theoretical proposal
**Deliverable:** ASA Theoretical Foundations v2.0
**Success Criterion:** Addresses Pustejovsky and Palmer concerns proactively

---

## Appendix A: Key References

### Category Theory and Semantics
- Coecke, B., Sadrzadeh, M., & Clark, S. (2010). Mathematical Foundations for a Compositional Distributional Model of Meaning. *Linguistic Analysis*, 36, 345-384.
- Lambek, J. (1958). The Mathematics of Sentence Structure. *American Mathematical Monthly*, 65, 154-170.
- Moortgat, M. (2012). Typelogical Grammar. *Stanford Encyclopedia of Philosophy*.

### Generative Lexicon and Coercion
- Pustejovsky, J. (1995). *The Generative Lexicon*. MIT Press.
- Pustejovsky, J. (2011). Coercion in a General Theory of Argument Selection. *Linguistics*, 49(6), 1401-1431.
- Asher, N. (2011). *Lexical Meaning in Context*. Cambridge University Press.

### Conceptual Spaces
- Gardenfors, P. (2000). *Conceptual Spaces*. MIT Press.
- Gardenfors, P. (2014). *The Geometry of Meaning*. MIT Press.

### Type Theory
- Martin-Lof, P. (1984). *Intuitionistic Type Theory*. Bibliopolis.
- Luo, Z. (2012). Formal Semantics in Modern Type Theories. *Linguistics and Philosophy*, 35, 491-513.

### Force Dynamics
- Talmy, L. (2000). *Toward a Cognitive Semantics*. MIT Press.
- Wolff, P. (2007). Representing Causation. *Journal of Experimental Psychology: General*, 136, 82-111.

### Hyperbolic Embeddings
- Nickel, M. & Kiela, D. (2017). Poincare Embeddings for Learning Hierarchical Representations. *NeurIPS*.
- Nickel, M. & Kiela, D. (2018). Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry. *ICML*.

---

## Appendix B: Glossary of Technical Terms

**Enriched Category:** A category where hom-sets are objects in another category (e.g., vector spaces) rather than mere sets.

**Functor:** A structure-preserving map between categories.

**Graded Type:** A typing judgment carrying a continuous confidence value.

**DisCoCat:** Distributional Compositional Categorical semantics, a framework combining distributional and compositional approaches.

**Dot Object:** In Generative Lexicon, a complex type representing systematic polysemy (e.g., book = phys.info).

**Type Coercion:** The process of shifting a type to satisfy compositional requirements.

**Quale (pl. Qualia):** In Pustejovsky's theory, one of four aspects of a concept's meaning (Formal, Constitutive, Telic, Agentive).

**Force Dynamics:** Talmy's framework for representing force interactions underlying causation and modality.

---

*Document prepared by Theory Researcher for ASA Research Swarm*
*Round 1 Theoretical Exploration - January 2, 2026*
