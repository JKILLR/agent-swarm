---
created: 2025-01-02 00:00
updated: 2026-01-02
---

# Semantic Periodic Table: Theoretical Foundations for ASA
## Comprehensive Research Compilation

*Version 1.0 — January 2, 2025*

---

## Executive Summary

This document synthesizes research from 12+ major linguistic, cognitive, and computational frameworks to establish the theoretical foundations for a **Semantic Periodic Table** — a principled organization of tokens by their semantic/syntactic properties that could enable predetermined embeddings for ASA (Atomic Semantic Attention).

The core insight: Just as the periodic table organizes elements by electron configuration (which determines bonding behavior), a semantic periodic table would organize tokens by their linguistic "valence configuration" — the properties that determine what they can meaningfully combine with.

**Key Finding:** Multiple independent research traditions converge on similar primitives and structures, suggesting a genuine underlying organization to semantic space that could be formalized and predetermined rather than learned.

---

## Table of Contents

1. [Semantic Primitives: The Atomic Layer](#1-semantic-primitives-the-atomic-layer)
2. [Ontological Categories](#2-ontological-categories)
3. [Thematic Roles and Selectional Restrictions](#3-thematic-roles-and-selectional-restrictions)
4. [Frame Semantics](#4-frame-semantics)
5. [Qualia Structure](#5-qualia-structure)
6. [Force Dynamics](#6-force-dynamics)
7. [Conceptual Spaces](#7-conceptual-spaces)
8. [Hyperbolic Geometry](#8-hyperbolic-geometry)
9. [Cross-Linguistic Universals](#9-cross-linguistic-universals)
10. [Compositional Semantics](#10-compositional-semantics)
11. [Type-Logical Grammar](#11-type-logical-grammar)
12. [Synthesis: Integrated Framework](#12-synthesis-integrated-framework)
13. [Implementation Strategy](#13-implementation-strategy)
14. [Open Questions](#14-open-questions)
15. [References](#15-references)

---

## 1. Semantic Primitives: The Atomic Layer

### Natural Semantic Metalanguage (NSM)
**Source:** Wierzbicka (1972-present), Goddard & Wierzbicka (2002)

NSM proposes that all human languages share a small set of **semantic primes** — irreducible concepts that cannot be defined in simpler terms and are lexicalized in all languages.

#### The 65 Semantic Primes (Current List)

| Category | Primes |
|----------|--------|
| **Substantives** | I, YOU, SOMEONE/PERSON, SOMETHING/THING, PEOPLE, BODY |
| **Relational** | KIND, PART |
| **Determiners** | THIS, THE SAME, OTHER/ELSE/ANOTHER |
| **Quantifiers** | ONE, TWO, SOME, ALL, MUCH/MANY, LITTLE/FEW |
| **Evaluators** | GOOD, BAD |
| **Descriptors** | BIG, SMALL |
| **Mental Predicates** | THINK, KNOW, WANT, DON'T WANT, FEEL, SEE, HEAR |
| **Speech** | SAY, WORDS, TRUE |
| **Actions/Events** | DO, HAPPEN, MOVE |
| **Existence/Possession** | THERE IS/EXIST, BE (SOMEONE/SOMETHING), HAVE, BE (SOMEWHERE) |
| **Life/Death** | LIVE, DIE |
| **Time** | WHEN/TIME, NOW, BEFORE, AFTER, A LONG TIME, A SHORT TIME, FOR SOME TIME, MOMENT |
| **Space** | WHERE/PLACE, HERE, ABOVE, BELOW, FAR, NEAR, SIDE, INSIDE, TOUCH (CONTACT) |
| **Logical** | NOT, MAYBE, CAN, BECAUSE, IF |
| **Intensifier** | VERY, MORE |
| **Similarity** | LIKE/WAY |

#### Key Properties
- **Cross-linguistically verified** in 30+ languages from 16+ families
- **Universal combinatorial grammar** — primes combine via shared syntactic patterns
- **Non-circular definitions** — primes define complex concepts without circularity
- Represent "the atoms of human thought"

#### Relevance to ASA
NSM primes could form the **innermost layer** of the semantic periodic table — the fundamental dimensions from which all other meanings are composed. Every token could be characterized by its decomposition into these primes.

---

## 2. Ontological Categories

### SUMO (Suggested Upper Merged Ontology)
**Source:** Niles & Pease (2001), IEEE Standard Upper Ontology Working Group

SUMO provides a comprehensive formal ontology with 20,000+ terms organized hierarchically.

#### Top-Level Categories

```
Entity
├── Physical
│   ├── Object
│   │   ├── SelfConnectedObject
│   │   ├── CorpuscularObject
│   │   └── Region
│   └── Process
│       ├── Motion
│       ├── InternalChange
│       └── DualObjectProcess
└── Abstract
    ├── Quantity
    ├── Attribute
    ├── SetOrClass
    ├── Relation
    └── Proposition
```

#### Key Distinctions
- **Physical vs Abstract**
- **Object vs Process**
- **Continuant vs Occurrent** (things that persist vs things that happen)
- **Independent vs Dependent** (substances vs modes)

### Jackendoff's Conceptual Semantics
**Source:** Jackendoff (1983, 1990, 2002)

Proposes ontological categories as semantic primes:

| Category | Examples | Feature |
|----------|----------|---------|
| **Thing** | dog, idea, water | [±BOUNDED] distinguishes count/mass |
| **Event** | explosion, concert | Bounded in time |
| **State** | being happy, knowing | Unbounded in time |
| **Place** | in the room, here | Spatial region |
| **Path** | to the store, across | Trajectory through space |
| **Property** | red, tall, angry | Attributes |
| **Time** | yesterday, noon | Temporal reference |
| **Amount** | three, much, half | Quantification |

#### Key Insight
The **[±BOUNDED]** feature is crucial — it distinguishes:
- Count nouns (a dog) vs mass nouns (water)
- Telic events (build a house) vs atelic activities (run)
- This feature could be encoded in predetermined embeddings

---

## 3. Thematic Roles and Selectional Restrictions

### VerbNet
**Source:** Kipper-Schuler (2005), based on Levin (1993)

VerbNet organizes ~6,800 English verbs into ~300 classes based on shared syntactic and semantic properties.

#### Thematic Roles

| Role | Definition | Example |
|------|-----------|---------|
| **Agent** | Intentional causer | *John* kicked the ball |
| **Patient** | Entity affected/changed | John kicked *the ball* |
| **Theme** | Entity moved/located | Put *the book* on the shelf |
| **Experiencer** | Entity experiencing state | *Mary* felt sad |
| **Instrument** | Means of action | Cut with *a knife* |
| **Goal** | Endpoint of motion | Walk to *the store* |
| **Source** | Starting point | Come from *Paris* |
| **Beneficiary** | Entity benefiting | Buy a gift for *her* |
| **Location** | Place of event | Live *in London* |
| **Stimulus** | Cause of experience | *The movie* frightened John |

#### Selectional Restrictions

Each verb class specifies what semantic features its arguments must have:

```
Example: "eat" (VerbNet class consume-66)
- Agent: [+animate]
- Patient: [+concrete], [+comestible]

Example: "think" (VerbNet class consider-29.9)  
- Agent: [+animate]
- Theme: [+abstract] or [+proposition]
```

#### Relevance to ASA
- Thematic roles define the **valence structure** of verbs
- Selectional restrictions define **bonding compatibility**
- This directly maps to ASA's predetermined attention masks

---

## 4. Frame Semantics

### FrameNet
**Source:** Fillmore (1982), Baker et al. (1998)

FrameNet catalogs 1,200+ semantic frames — schematic representations of situations with participant roles.

#### Example Frame: Commerce_buy

```
Frame: Commerce_buy
Definition: A Buyer gives Money to a Seller in exchange for Goods

Frame Elements:
- Buyer [Core]: The person obtaining goods
- Seller [Core]: The person giving goods for money  
- Goods [Core]: What is obtained
- Money [Core]: What is exchanged
- Purpose [Non-Core]: Why buyer wants goods
- Manner [Non-Core]: How transaction occurs

Example: [Buyer Chuck] bought [Goods a car] from [Seller a dealer] for [Money $20,000]
```

#### Frame Relations

| Relation | Meaning | Example |
|----------|---------|---------|
| **Inheritance** | Child inherits parent roles | Getting ⊂ Commerce_buy |
| **Using** | Frame presupposes another | Commerce uses Transfer |
| **Subframe** | Part of larger frame | Hiring subframe of Employment_start |
| **Perspective** | Same event, different viewpoint | Commerce_sell ↔ Commerce_buy |
| **Precedes** | Temporal ordering | Employment_start → Working |
| **Causative_of** | Causal relation | Killing causative_of Death |

#### Relevance to ASA
Frames define **which tokens can meaningfully co-occur** in a context. A frame-aware system knows:
- What roles a sentence requires
- What semantic types can fill each role
- This is precisely the bonding structure ASA needs

---

## 5. Qualia Structure

### Generative Lexicon
**Source:** Pustejovsky (1991, 1995)

Pustejovsky proposes that nouns have internal structure captured by four **qualia** (from Aristotle's four causes):

#### The Four Qualia

| Quale | Question Answered | Example: "book" |
|-------|-------------------|-----------------|
| **Formal** | What IS it? | artifact, physical_object |
| **Constitutive** | What is it MADE OF? | pages, binding, ink |
| **Telic** | What is it FOR? | reading, reference |
| **Agentive** | How did it COME TO BE? | writing, printing |

#### Applications

**Type Coercion:**
- "Mary began the book" → began [reading] the book (telic)
- "The author finished the book" → finished [writing] the book (agentive)

**Systematic Polysemy:**
- "book" = physical object (constitutive) OR information (telic)
- "window" = physical object OR opening OR software element

**Noun-Verb Relations:**
- "bottle" (N) → "bottle" (V) uses agentive quale
- "hammer" (N) → "hammer" (V) uses telic quale

#### Relevance to ASA
Qualia provide a **4-dimensional feature space** for ALL nouns:
- Every nominal can be characterized along [Formal, Constitutive, Telic, Agentive]
- This structure could be directly encoded in predetermined embeddings
- Enables predicting coercion and polysemy patterns

---

## 6. Force Dynamics

### Talmy's Cognitive Semantics
**Source:** Talmy (1988, 2000)

Force dynamics extends the notion of physical force to abstract domains, providing primitives for causation, modality, and discourse.

#### Force-Dynamic Primitives

| Primitive | Description | Example |
|-----------|-------------|---------|
| **Compulsion** | Force applied | "push", "make" |
| **Blockage** | Force resisted | "prevent", "block" |
| **Counterforce** | Force opposed | "resist", "withstand" |
| **Diversion** | Force redirected | "deflect", "channel" |
| **Removal of restraint** | Barrier removed | "let", "allow" |
| **Enablement** | Support provided | "help", "enable" |

#### Agonist/Antagonist Framework

Every force-dynamic event involves:
- **Agonist**: Entity whose tendency is at issue
- **Antagonist**: Entity opposing/enabling the agonist
- **Intrinsic tendency**: Toward action or rest
- **Resultant state**: Whether agonist's tendency is realized

```
"The ball kept rolling despite the wind"
- Agonist: ball (tendency: motion)
- Antagonist: wind (tendency: stop ball)
- Result: Agonist's tendency wins
```

#### Extended to Causation

| Causal Type | Force Pattern | Example |
|-------------|--------------|---------|
| **Cause** | A makes B do X | "The wind blew the door open" |
| **Let** | A removes restraint on B | "Mary let the bird go" |
| **Hinder** | A impedes B | "The rain hindered our progress" |
| **Help** | A assists B's tendency | "The wind helped the fire spread" |

#### Relevance to ASA
Force dynamics is essential for **verb semantics**:
- Agent/Patient map to Agonist/Antagonist
- Causation types determine argument structure
- Force vectors could be encoded in verb embeddings

---

## 7. Conceptual Spaces

### Gärdenfors' Theory
**Source:** Gärdenfors (2000, 2014)

Conceptual Spaces provides a **geometric formalization** of meaning — concepts are regions in multi-dimensional quality spaces.

#### Core Architecture

**Quality Dimensions:**
- Measurable properties forming the axes of conceptual space
- Examples: hue, brightness, saturation (for color); pitch, loudness (for sound)

**Domains:**
- Sets of **integral dimensions** that are separable from other dimensions
- Color domain = {hue, saturation, brightness}
- Spatial domain = {x, y, z coordinates}
- Dimensions within a domain are perceived holistically

**Criterion P (Convexity):**
> Natural properties correspond to CONVEX regions of a single domain

A region is convex if, for any two points in it, all points between them are also in it. This explains:
- Why we have "red" but not a word for "red or green but not orange"
- Why natural categories have prototype structure

#### Single-Domain Hypotheses

**Adjectives** (Gärdenfors 2000):
- Map to convex regions within a SINGLE domain
- "red" = convex region in color domain
- "tall" = convex region in height dimension

**Nouns** (Gärdenfors 2000):
- Map to regions across MULTIPLE domains
- "apple" = intersection of shape-region × color-region × taste-region × texture-region
- More dimensions = more specific concept

**Verbs** (Gärdenfors 2014):
- Map to **force vectors** and **result vectors**
- Agent/Patient located in property domains
- Events = mappings from force vectors to result vectors

```
Event structure:
- Agent properties (property domains)
- Patient properties (property domains)  
- Force vector (3D force space)
- Result vector (change in patient properties)
```

#### Relevance to ASA
This is the most directly applicable framework:
- Quality dimensions = **axes of embedding space**
- Domains = **subspaces** (e.g., color, shape, force)
- Convex regions = **natural concepts**
- Distance = **similarity**

**Could directly implement:**
- Each token gets coordinates in relevant domain spaces
- Nouns: positions in multiple domains
- Adjectives: regions in single domains
- Verbs: force + result vectors

---

## 8. Hyperbolic Geometry

### Poincaré Embeddings
**Source:** Nickel & Kiela (2017)

Hyperbolic space naturally represents hierarchical data — trees embed with near-zero distortion.

#### Why Hyperbolic?

**The Problem with Euclidean Space:**
- Trees have exponentially growing nodes: level ℓ has (b+1)b^(ℓ-1) nodes (branching factor b)
- Euclidean dimensions needed to embed tree with low distortion grows exponentially with depth
- Flat space cannot efficiently represent hierarchy

**Hyperbolic Space Properties:**
- Space **expands exponentially** toward boundary
- Regular trees embed with **constant distortion** regardless of depth
- Distance from origin naturally encodes **hierarchy level**

#### Poincaré Ball Model

The n-dimensional Poincaré ball is:
```
B^n = {x ∈ ℝ^n : ||x|| < 1}
```

With distance metric:
```
d(u,v) = arcosh(1 + 2 * ||u-v||² / ((1-||u||²)(1-||v||²)))
```

Key properties:
- **Origin = root** of hierarchy (most general)
- **Boundary = leaves** (most specific)
- **Angular position** encodes category
- Geodesics are circles orthogonal to boundary

#### Results on WordNet

| Embedding | Dimensions | Reconstruction F1 |
|-----------|------------|-------------------|
| Euclidean | 200 | 0.870 |
| Poincaré | 10 | 0.879 |
| Poincaré | 5 | 0.823 |

5D hyperbolic ≈ 200D Euclidean for hierarchical data!

#### Relevance to ASA
Hyperbolic space provides the **geometry** for the semantic periodic table:
- **Distance from origin** = level of abstraction
  - Near origin: ENTITY, THING, EVENT (NSM primes)
  - Middle: CAR, PERSON, WALK (basic level)
  - Near boundary: TOYOTA_CAMRY_2024, USAIN_BOLT (instances)
- **Angular position** = semantic category (SUMO taxonomy)
- Natural fit for taxonomic structure from WordNet/SUMO

---

## 9. Cross-Linguistic Universals

### Berlin & Kay Color Universals
**Source:** Berlin & Kay (1969), Regier et al. (2005-2018)

Color terms across languages show constrained variation:

#### Universal Focal Colors
- All languages have terms for BLACK, WHITE
- If 3 terms: add RED
- If 4-5 terms: add YELLOW and/or GREEN
- If 6 terms: add BLUE
- If 7+ terms: add BROWN, then PURPLE, PINK, ORANGE, GREY

#### Explanation
- **Perceptual universals**: Human color perception is structured
- **Efficient communication**: Categories optimize informativeness
- **Not arbitrary**: Languages carve up the same perceptual space

### Semantic Typology Findings

Research reveals constrained variation across languages:

| Domain | Universal Pattern |
|--------|------------------|
| **Color** | 11 basic terms, evolutionary sequence |
| **Kinship** | Limited set of distinguishing features |
| **Body parts** | Core terms stable across languages |
| **Spatial relations** | Universal primitives (TOP, BOTTOM, etc.) |
| **Number** | Universal sequence (1, 2, 3, some, many) |
| **Motion events** | Limited framing types (verb vs satellite) |

#### NSM Cross-Linguistic Validation
- 65 semantic primes verified in 30+ languages
- Each prime has translational equivalent
- Universal combinatorial grammar

#### Relevance to ASA
Cross-linguistic universals **validate** that semantic structure is NOT arbitrary:
- Universal focal colors = anchor points in color space
- NSM primes = universal semantic atoms
- Constrained variation = underlying structure to discover
- The semantic periodic table must encode these universals

---

## 10. Compositional Semantics

### Compositional Distributional Semantics
**Source:** Baroni et al. (2014), Mitchell & Lapata (2010), Coecke et al. (2010)

How to compute phrase/sentence meaning from word vectors?

#### Three Approaches

**1. Vector Mixture (Mitchell & Lapata)**
```python
phrase_vector = word1_vector ⊙ word2_vector  # element-wise operation
# ⊙ = addition, multiplication, weighted combination
```
- Simple but surprisingly effective baseline
- Ignores word order and syntax

**2. Tensor-Based (Baroni & Zamparelli 2010)**
```python
# "Nouns are vectors, adjectives are matrices"
noun_vector = [d1, d2, ..., dn]        # n-dimensional vector
adjective_matrix = [[a11, a12, ...],   # n×n matrix
                    [a21, a22, ...], 
                    ...]
adjective_noun = adjective_matrix @ noun_vector  # matrix-vector product
```
- Respects grammatical types
- Adjectives transform nouns
- Verbs are higher-order tensors

**3. Neural Composition (Socher et al.)**
```python
# Recursive neural networks
phrase_vector = f(W @ [word1; word2] + b)  # learned transformation
```
- Learned composition function
- Uses parse tree structure

#### Key Insight: "Frege in Space"

Different word classes have different **algebraic types**:
- Nouns: vectors (n-dimensional)
- Adjectives: matrices (n×n, transform nouns)
- Transitive verbs: order-3 tensors (take two noun arguments)
- Intransitive verbs: matrices (take one noun argument)

This aligns with type-logical grammar and VerbNet!

#### Relevance to ASA
Compositional semantics informs how **predetermined embeddings should interact**:
- Type-matched composition
- Grammatical structure guides tensor operations
- Could derive composition operations from linguistic theory rather than learning

---

## 11. Type-Logical Grammar

### Lambek Calculus and Categorial Grammar
**Source:** Lambek (1958), Steedman (1996), Moortgat (2012)

Type-logical grammar treats syntax as type inference — words have types, and combination is type-valid function application.

#### Basic Idea

Assign each word a **type** that specifies what it combines with:

| Word | Type | Meaning |
|------|------|---------|
| John | NP | Noun phrase |
| runs | NP\S | Takes NP on left, yields S |
| quickly | (NP\S)\(NP\S) | Modifies verb |
| the | NP/N | Takes N on right, yields NP |

#### Composition Rules

**Forward Application (>):**
```
A/B + B → A
"the" + "cat" → NP
NP/N   N
```

**Backward Application (<):**
```
B + B\A → A  
"John" + "runs" → S
NP       NP\S
```

#### Curry-Howard Correspondence

Types = Propositions, Derivations = Proofs, Composition = Computation

This means:
- Syntactic derivation = semantic computation
- Type assignment = semantic type
- **Compositionality is built into the grammar**

#### Relevance to ASA
Type-logical grammar provides:
- **Principled type system** for word classes
- **Composition operations** derived from logic, not learned
- **Syntax-semantics interface** is automatic
- Could assign types to tokens that determine bonding

---

## 12. Synthesis: Integrated Framework

### Dimensional Structure

Based on the research, a semantic periodic table would have multiple interacting components:

#### Layer 1: Atomic Core (NSM Primes)
- 65 semantic primes as fundamental dimensions
- Every concept decomposable into these primitives
- Cross-linguistically universal

#### Layer 2: Ontological Categories (SUMO + Jackendoff)
- Major axes: Physical ↔ Abstract, Object ↔ Process
- Feature: [±BOUNDED] for mass/count, telic/atelic
- Taxonomic hierarchy encoded in hyperbolic space

#### Layer 3: Qualia Structure (Pustejovsky)
- 4D feature vector: [Formal, Constitutive, Telic, Agentive]
- Applies to all nominals
- Enables polysemy and coercion prediction

#### Layer 4: Force Dynamics (Talmy)
- Agonist/Antagonist framework
- Force tendency vectors
- Causation primitives

#### Layer 5: Thematic Roles (VerbNet)
- Valence structure for verbs
- Selectional restrictions as feature constraints
- ~20 thematic roles

#### Layer 6: Frame Membership (FrameNet)
- Binary vector over 1,200 frames
- Encodes situational context
- Frame-to-frame relations

#### Layer 7: Conceptual Domains (Gärdenfors)
- Quality dimensions for perceptual domains
- Color: [hue, saturation, brightness]
- Shape: [geometric features]
- Force: [3D force vectors]

### Geometric Encoding

**Hyperbolic Core:**
- Radial dimension = abstraction level (SUMO hierarchy)
- Angular dimensions = semantic category
- NSM primes cluster near origin

**Embedded Euclidean Subspaces:**
- Each domain is a Euclidean subspace
- Natural metrics for similarity within domain
- Distances meaningful only within domain

**Type Assignment:**
- Nouns: vectors in multi-domain product space
- Adjectives: regions in single domain
- Verbs: force vectors + result vectors + role slots

### Predetermined Embedding Algorithm (Conceptual)

```python
def compute_predetermined_embedding(token):
    """
    Compute fixed embedding for token based on linguistic theory.
    """
    embedding = {}
    
    # 1. NSM decomposition
    embedding['nsm'] = decompose_into_primes(token)
    
    # 2. SUMO category → hyperbolic position
    category = lookup_sumo_category(token)
    embedding['hyperbolic'] = category_to_poincare_position(category)
    
    # 3. Qualia structure (for nouns)
    if is_noun(token):
        embedding['qualia'] = extract_qualia(token)
    
    # 4. VerbNet class (for verbs)
    if is_verb(token):
        vn_class = lookup_verbnet_class(token)
        embedding['thematic_roles'] = get_role_structure(vn_class)
        embedding['selectional_restrictions'] = get_restrictions(vn_class)
    
    # 5. FrameNet frames
    embedding['frames'] = get_frame_membership(token)
    
    # 6. Conceptual domains
    embedding['domains'] = get_domain_positions(token)
    
    # 7. Type assignment
    embedding['type'] = assign_categorial_type(token)
    
    return combine_components(embedding)
```

### Bonding Rules Derived from Structure

Given two tokens with embeddings, compatibility is determined by:

1. **Type compatibility** (Lambek calculus)
   - Does A/B + B → A work?
   
2. **Frame compatibility** (FrameNet)
   - Are tokens in same/compatible frames?
   
3. **Selectional restriction satisfaction** (VerbNet)
   - Does argument have required features?
   
4. **Domain overlap** (Gärdenfors)
   - Are tokens in compatible conceptual domains?

This gives **predetermined sparsity** — we know which pairs can bond before any attention computation.

---

## 13. Implementation Strategy

### Phase 1: Resource Assembly
- Download and parse: VerbNet, FrameNet, WordNet, SUMO
- Build lookup tables: word → VerbNet class, word → frames, word → SUMO category
- Create NSM decomposition database (manual curation needed)

### Phase 2: Feature Extraction
- POS tags: from spaCy/Stanza
- VerbNet class: from verb lemma lookup
- FrameNet frames: from lexical unit lookup
- WordNet features: hypernym traversal (current ASA approach)
- SUMO category: from WordNet-SUMO mapping

### Phase 3: Geometric Encoding
- Train/derive hyperbolic positions from taxonomy
- Define domain subspaces
- Assign type-logical types

### Phase 4: Composition Rules
- Implement Lambek-style type inference
- Define composition operations per type pair
- Derive bonding masks from type compatibility

### Phase 5: Validation
- Compare predetermined embeddings to learned embeddings
- Measure H6-style correlation
- Test on downstream tasks

### Dimensionality Estimate

| Component | Dimensions | Notes |
|-----------|-----------|-------|
| Hyperbolic (hierarchy) | 50-100 | SUMO taxonomy |
| Qualia | 4 | Per-quale scores |
| Thematic roles | 10-20 | Role slot features |
| Selectional features | 15-20 | Binary features |
| FrameNet | 100-200 | Compressed frame membership |
| Domain positions | 50-100 | Per-domain coordinates |
| Type | 10-20 | Categorial grammar type encoding |
| **Total** | **250-500** | Without redundancy |

---

## 14. Open Questions

### Theoretical

1. **Coverage gaps**: How to handle words not in lexical resources?
   - Option: Interpolation from similar words
   - Option: Fallback to distributional features

2. **Polysemy**: One position or multiple?
   - Option: Sense vectors as offsets from core position
   - Option: Blended position in ambiguous region

3. **Abstract/technical terms**: How to encode domain-specific vocabulary?
   - Option: Hierarchical extension from nearest SUMO category
   - Option: Compositional derivation from component meanings

4. **Pragmatic meaning**: How to handle context-dependent interpretation?
   - Answer: This is what the LEARNED components handle
   - Predetermined = type/structure; Learned = contextual adjustment

### Computational

5. **Efficiency**: Can we compute hyperbolic operations fast enough?
   - Poincaré ball operations are well-understood
   - GPU implementations exist

6. **Sparsity**: What sparsity level can we achieve?
   - Current ASA: ~31% on WikiText-2
   - With full type system: potentially 50-70%

7. **Training**: How to fine-tune while preserving structure?
   - Constrained optimization: stay in predetermined region
   - Hybrid: fixed structure + learned residuals

### Empirical

8. **Validation**: Do predetermined embeddings capture linguistic intuitions?
   - Psycholinguistic experiments needed
   - Comparison to human similarity judgments

9. **Downstream**: Do they improve task performance?
   - Language modeling (perplexity)
   - Compositional generalization (COGS, SCAN)
   - Semantic role labeling

10. **Cross-lingual**: Do they generalize across languages?
    - NSM claims universality
    - Need multilingual evaluation

---

## 15. References

### Semantic Primitives
- Wierzbicka, A. (1996). *Semantics: Primes and Universals*. Oxford University Press.
- Goddard, C., & Wierzbicka, A. (2002). *Meaning and Universal Grammar*. John Benjamins.

### Ontologies
- Niles, I., & Pease, A. (2001). Towards a standard upper ontology. *FOIS*.
- Jackendoff, R. (1990). *Semantic Structures*. MIT Press.

### Thematic Roles
- Kipper-Schuler, K. (2005). *VerbNet: A broad-coverage, comprehensive verb lexicon*. PhD thesis, UPenn.
- Levin, B. (1993). *English Verb Classes and Alternations*. University of Chicago Press.

### Frame Semantics
- Fillmore, C. J. (1982). Frame semantics. *Linguistics in the Morning Calm*.
- Baker, C. F., Fillmore, C. J., & Lowe, J. B. (1998). The Berkeley FrameNet project. *COLING-ACL*.

### Qualia Structure
- Pustejovsky, J. (1995). *The Generative Lexicon*. MIT Press.

### Force Dynamics
- Talmy, L. (2000). *Toward a Cognitive Semantics* (2 vols). MIT Press.

### Conceptual Spaces
- Gärdenfors, P. (2000). *Conceptual Spaces*. MIT Press.
- Gärdenfors, P. (2014). *The Geometry of Meaning*. MIT Press.

### Hyperbolic Embeddings
- Nickel, M., & Kiela, D. (2017). Poincaré embeddings for learning hierarchical representations. *NeurIPS*.

### Cross-Linguistic Universals
- Berlin, B., & Kay, P. (1969). *Basic Color Terms*. University of California Press.
- Regier, T., Kay, P., & Khetarpal, N. (2007). Color naming reflects optimal partitions of color space. *PNAS*.

### Compositional Distributional Semantics
- Baroni, M., Bernardi, R., & Zamparelli, R. (2014). Frege in space: A program for compositional distributional semantics. *LiLT*.
- Coecke, B., Sadrzadeh, M., & Clark, S. (2010). Mathematical foundations for a compositional distributional model of meaning. *Linguistic Analysis*.

### Type-Logical Grammar
- Lambek, J. (1958). The mathematics of sentence structure. *American Mathematical Monthly*.
- Moortgat, M. (2012). Typelogical grammar. *Stanford Encyclopedia of Philosophy*.

---

## Appendix A: Mapping to Current ASA

The current ASA v2.2 implementation uses a subset of this framework:

| Current ASA | Theoretical Foundation | Potential Expansion |
|-------------|----------------------|---------------------|
| POS tags | Coarse type system | Full categorial types |
| WordNet hypernyms | Partial SUMO hierarchy | Full SUMO + hyperbolic |
| VerbNet restrictions | 300 verbs | Full 6,800 verbs |
| 15 binary features | Partial selectional features | Full qualia + roles |
| — | — | FrameNet integration |
| — | — | Force dynamics vectors |
| — | — | Conceptual domain spaces |

The current ~31% sparsity could potentially increase to 50-70% with the full framework.

---

## Appendix B: The Semantic Periodic Table Vision

Just as the periodic table organizes elements by:
- **Atomic number** (protons = identity)
- **Electron configuration** (determines bonding)
- **Period** (shell count = size)
- **Group** (valence electrons = bonding type)

A semantic periodic table would organize tokens by:
- **NSM decomposition** (semantic "atomic number")
- **Thematic valence** (argument structure = bonding)
- **Abstraction level** (hyperbolic radius)
- **Semantic category** (hyperbolic angle)

The goal: Every token has a **fixed position** determined by theory, not learned from data. Training only learns how to PROCESS these representations, not how to CREATE them.

This is the path from "attention with a sparsity mask" to "true atomic semantic processing."

---

*Document compiled January 2, 2025*
*For ASA Project — Atomic Semantic Attention*
