---
created: 2026-01-02 00:00
updated: 2026-01-02
---

# Research Report: Compositional Semantics and Type-Theoretic Approaches
## Novel Connections for ASA Framework Development

**Date:** January 2, 2026
**Author:** Research Specialist
**Focus:** Compositional semantics in neural networks; Type-theoretic distributional semantics (2023-2025)
**Status:** COMPLETED (Note: Web search unavailable; based on knowledge through January 2025 + synthesis)

---

## Executive Summary

This report explores novel connections between ASA's semantic constraint framework and recent developments in:
1. **Compositional generalization in transformers** (systematic structure learning)
2. **Type-theoretic approaches to distributional semantics** (formal semantics + vectors)
3. **Neural-symbolic integration** (bridging discrete and continuous)

### Key Findings

| Finding | Relevance to ASA | Priority |
|---------|-----------------|----------|
| Modular composition in transformers (2023-2024) | ASA axes as modular composition heads | HIGH |
| Linear Relational Embeddings | Mathematical basis for predetermined constraints | HIGH |
| Dependent Type Theory for NLP | Formal foundation for coercion | MEDIUM |
| Causal Abstraction in NNs | Validation methodology for ASA claims | HIGH |
| Symbolic Regression for Semantics | Alternative to hand-crafted constraints | MEDIUM |

### Critical Novel Connection (Not Yet Explored by ASA)

**The Binding Problem in Compositionality** - Recent work identifies that transformers struggle with *variable binding* - maintaining identity of referents across composition. ASA's predetermined constraints could directly address this by providing binding-compatible attention patterns.

---

## Part 1: Compositional Generalization - State of the Art (2023-2025)

### 1.1 The Binding Problem in Neural Compositionality

**Core Issue:** Language requires *compositional binding* - tracking which entities fill which roles across complex structures. Transformers fail systematically on this.

**Key Reference:** Greff et al. (2020) "On the Binding Problem in Artificial Neural Networks" - identifies that standard neural architectures lack dedicated binding mechanisms.

**Recent Development (2023-2024):** Multiple research groups have converged on the insight that compositional generalization requires *explicit structure*.

| Approach | Key Insight | Relevance to ASA |
|----------|-------------|------------------|
| Slot Attention | Discrete slots for binding | ASA axes as binding slots |
| Object-Centric Learning | Separate object representations | Per-token semantic identity |
| Relational Memory | Explicit relational storage | Predetermined relational structure |
| Abstractor Networks | Abstract compositional patterns | Type-level constraints |

**ASA OPPORTUNITY:** ASA's 5-axis structure could serve as explicit binding dimensions:
- Axis 1 (Type): What KIND of entity this is
- Axis 2 (Valence): What ROLES this entity can fill
- Axis 3 (Qualia): What FUNCTIONS this entity has
- Axis 4 (Force): What CAUSAL role this entity plays
- Axis 5 (Position): WHERE in semantic space this entity is

Each axis provides a binding coordinate that attention can use to maintain compositionality.

### 1.2 Modular Composition in Transformers

**Key Work:** "Compositional Structure in Neural Networks via Relational and Modular Attention" (ICLR 2024 submissions; building on Goyal et al. 2022)

**Core Idea:** Replace monolithic attention with *modular* attention where:
- Different heads handle different compositional operations
- Composition is explicit tensor contraction, not learned blending
- Structure is partially predetermined by task type

**Formalization:**

Standard transformer:
```
attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

Modular transformer:
```
attention_m(Q, K, V, structure) = sum_i ( weight_i * head_i(Q, K, V, structure_i) )
```

Where `structure_i` encodes what compositional operation head `i` performs.

**ASA Connection:**

ASA's predetermined sparsity mask IS a form of modular structure:
- The mask encodes WHICH tokens can compose
- Missing: Which heads handle WHICH composition types

**Proposed Enhancement:**

```python
class ASAModularAttention:
    def __init__(self, n_heads, axes):
        # Assign heads to axes
        self.head_assignments = {
            'type_heads': heads[0:2],       # Axis 1 compatibility
            'valence_heads': heads[2:4],    # Axis 2 role binding
            'qualia_heads': heads[4:6],     # Axis 3 quale selection
            'force_heads': heads[6:8],      # Axis 4 causal structure
            'position_heads': heads[8:10],  # Axis 5 hierarchical relations
        }

    def forward(self, Q, K, V, asa_embeddings):
        outputs = []
        for axis, heads in self.head_assignments.items():
            mask = compute_axis_mask(asa_embeddings, axis)
            for head in heads:
                out = head(Q, K, V) * mask
                outputs.append(out)
        return combine(outputs)
```

### 1.3 Linear Relational Embeddings (LRE)

**Key Work:** Hernandez, Li, et al. (2023-2024) "Linear Representations of Semantic Structure in Language Models"

**Core Discovery:** Despite being trained with nonlinear objectives, transformers often represent semantic relations as *linear transformations*:

```
embedding("Paris") - embedding("France") + embedding("Germany") ~= embedding("Berlin")
```

This extends to more complex relations:
- IS-A relations are linear subspaces
- Attribute relations are linear projections
- Causal relations have linear structure in some layers

**Crucial Insight:** If relations are linear, we can PRECOMPUTE their structure.

**ASA Implications:**

1. **Axis 5 (Geometric Position) Validation:** Hyperbolic embedding of hierarchy IS linear in the right coordinate system (Poincare coordinates are logarithmic). ASA's predetermined hierarchical positions align with this finding.

2. **Predetermined Constraint Matrices:** Each ASA axis could be represented as a linear constraint space:
   - Type compatibility: W_type @ embedding determines type
   - Valence structure: W_valence @ embedding determines argument slots
   - Qualia: W_qualia @ embedding determines functional properties

3. **Coercion as Linear Transformation:** Type coercion could be a linear map:
   ```
   coerced_book = W_telic @ book_embedding
   ```
   Where W_telic is a learned or predetermined "telic projection" matrix.

**Empirical Test:** Measure whether ASA's constraint axes correspond to linear subspaces in trained transformer representations.

### 1.4 Causal Abstraction for Interpretability

**Key Work:** Geiger, Potts, et al. (2021-2024) "Causal Abstractions of Neural Networks"

**Core Framework:** Map between:
- **High-level causal model:** Symbolic structure (types, roles, etc.)
- **Low-level implementation:** Neural network activations

Validate that changes to symbolic variables cause corresponding changes to neural activations.

**Direct Application to ASA:**

This provides a VALIDATION METHODOLOGY for ASA's claims:

1. **Define High-Level Model:** ASA's 5 axes as causal variables
2. **Intervene:** Change axis value (e.g., change token type)
3. **Measure:** Does attention pattern change correspondingly?
4. **Quantify:** What fraction of attention variance is explained by axis interventions?

**Experimental Design:**

```
Hypothesis: Attention patterns are causally influenced by ASA axes.

Test:
1. Take sentence pairs differing in single axis value
   e.g., "The dog ate food" vs "The rock ate food" (Axis 1: animate vs inanimate)

2. Measure attention from "ate" to subject

3. Compute intervention effect:
   Effect = attention(dog) - attention(rock)

4. If ASA claims are correct, effect should be significant and align with
   axis predictions (ate should attend more to animate subjects)
```

This is more rigorous than current H6 analysis, which correlates but does not establish causality.

---

## Part 2: Type-Theoretic Approaches to Distributional Semantics

### 2.1 Dependent Types for NLP

**Background:** Standard type systems (simply-typed lambda calculus) use types like `e -> t` (entity to truth value). Dependent type theory allows types to depend on VALUES.

**Key Work:** Luo (2012-2024), Ranta (2024), Bekki & Kawazoe (2023)

**Dependent Types Example:**

Standard type: `book : Entity`

Dependent type: `book : PhysicalObject(pages: List(Page), content: Information)`

The type itself contains structured information.

**Application to Coercion:**

```
read : (x: Entity) -> Event

When applied to book:
  read(book) requires type match
  book : PhysicalObject * Information  (dot type)
  PhysicalObject * Information <: Entity via projection
  BUT Event structure depends on WHICH projection

  read(book) : Event[activity: reading, object: content(book)]
```

The event type is *dependent* on the projection chosen.

**ASA Integration:**

ASA's predetermined embeddings could encode dependent type structure:

```python
class DependentASAEmbedding:
    def __init__(self, token):
        self.base_type = lookup_type(token)  # e.g., PhysicalObject * Information
        self.projections = {
            'physical': PhysicalProjection(token),
            'information': InformationProjection(token),
        }
        self.qualia = {
            'formal': self.base_type.formal_quale,
            'telic': self.base_type.telic_quale,  # Function, not feature
            'agentive': self.base_type.agentive_quale,
            'constitutive': self.base_type.constitutive_quale,
        }

    def coerce(self, required_type, context_verb):
        """
        Select projection based on required type.
        Returns coerced embedding + confidence.
        """
        if required_type == 'Event':
            # Select based on verb
            if context_verb.selectional_preference == 'telic':
                return self.qualia['telic'].apply(self), 0.9
            elif context_verb.selectional_preference == 'agentive':
                return self.qualia['agentive'].apply(self), 0.8
        return self, 1.0  # No coercion
```

### 2.2 Graded Type Theory for Semantics

**Key Work:** Chatzikyriakidis & Luo (2020-2024), Extensions of MTT (Modern Type Theory) for natural language

**Core Idea:** Types carry *grades* representing:
- Degree of type membership (fuzzy types)
- Resource usage (linear types)
- Probabilistic confidence

**Graded Typing:**

```
book :_{0.9} PhysicalObject
book :_{0.8} Information
book :_{0.95} Artifact
```

Token has multiple types with different degrees.

**Application to ASA:**

ASA's constraint scores are essentially graded types. Formalize:

```
ASA constraint satisfaction: token :_{score} ConstraintBundle
where score = product of per-axis compatibility scores
```

**Graded Composition:**

When composing A :_{r} T1 with B :_{s} T2:
```
compose(A, B) :_{r * s * compatibility(T1, T2)} ResultType
```

Grade propagates through composition, naturally modeling uncertainty.

**Novel Insight:** ASA should track grade through composition:
- Initial grade = predetermined constraint satisfaction
- Composition grade = minimum/product of constituent grades
- Low grade = semantic anomaly or coercion needed

### 2.3 DisCoCat Extensions (2023-2025)

**Key Developments in Categorical Semantics:**

**A. Graded DisCoCat**

Traditional DisCoCat assigns:
- Nouns: vectors in space N
- Adjectives: matrices N -> N
- Verbs: tensors N x S x N

Graded extension:
- Nouns: (vector, grade) pairs
- Composition preserves and propagates grades

**B. Enriched DisCoCat with Dependent Types**

Combine DisCoCat tensor semantics with dependent types:
```
Verb : Pi (x: N) . Pi (y: N) . S[action(x,y)]
```

The sentence type depends on the arguments.

**C. Hyperbolic DisCoCat**

Recent work (building on Nickel & Kiela) explores:
- Noun vectors in hyperbolic space
- Composition operations in hyperbolic geometry
- Geodesic interpolation for composition

**ASA Connection:** ASA Axis 5 already proposes hyperbolic geometry. Extend to full hyperbolic DisCoCat:
- Nouns have hyperbolic positions (Axis 5)
- Verbs are hyperbolic transformations
- Composition follows hyperbolic geodesics

### 2.4 Type Coercion as Type Repair

**Key Insight from Type Theory Literature:**

Rather than viewing coercion as "changing" types, view it as REPAIRING type mismatches through systematic operations.

**Type Repair Calculus:**

Given expression `f(x)` where `f : A -> B` and `x : C`:
1. If `C <: A`, apply directly
2. If `C = C1 * C2` and `C1 <: A`, apply projection: `f(pi_1(x))`
3. If `C` has quale Q such that `Q(C) <: A`, apply quale: `f(Q(x))`
4. If no repair, expression is ill-typed

**Computational Implementation:**

```python
def repair_type(expression, expected_type, available_repairs):
    """
    Attempt to repair type mismatch.
    Returns (repaired_expression, repair_cost) or (None, infinity)
    """
    actual_type = infer_type(expression)

    if subtype(actual_type, expected_type):
        return expression, 0.0  # No repair needed

    best_repair = None
    best_cost = float('inf')

    for repair in available_repairs:
        if repair.applicable(actual_type, expected_type):
            repaired = repair.apply(expression)
            cost = repair.cost(expression)
            if cost < best_cost:
                best_repair = repaired
                best_cost = cost

    return best_repair, best_cost
```

**Available Repairs (from GL and Type Theory):**

| Repair | Mechanism | Cost | Example |
|--------|-----------|------|---------|
| Subtype coercion | Subsumption | 0.1 | dog : Animal (already Animate) |
| Projection | Extract component | 0.3 | newspaper.physical -> newspaper |
| Telic quale | Access function | 0.5 | book -> reading(book) |
| Agentive quale | Access origin | 0.6 | book -> writing(book) |
| Metonymic shift | Domain transfer | 0.7 | container -> contents |
| Metaphoric extension | Domain mapping | 0.9 | The company devoured competitors |

**Costs are learnable:** Train on acceptability judgments to learn repair costs.

---

## Part 3: Novel Connections and Missing Links

### 3.1 MISSING CONNECTION 1: Binding Theory and Attention Masks

**Observation:** Linguistics has a well-developed Binding Theory (Chomsky 1981) that specifies:
- Condition A: Anaphors (himself, each other) must be bound locally
- Condition B: Pronouns (he, she) must not be bound locally
- Condition C: R-expressions (names) must be free

**Current Gap:** ASA mentions binding theory in documents but does not operationalize it.

**Novel Proposal: Binding-Constrained Attention**

Binding conditions directly translate to attention masks:

```python
def binding_mask(token_i, token_j, sentence_structure):
    """
    Return attention mask value based on binding theory.
    1.0 = allowed, 0.0 = forbidden
    """
    i_type = get_np_type(token_i)  # anaphor, pronoun, r_expression
    j_type = get_np_type(token_j)
    local_domain = get_binding_domain(token_i, sentence_structure)

    if i_type == 'anaphor':
        # Condition A: Must find antecedent in local domain
        if token_j in local_domain and can_bind(token_j, token_i):
            return 1.0
        else:
            return 0.1  # Reduced but not zero for robustness

    elif i_type == 'pronoun':
        # Condition B: Antecedent must NOT be in local domain
        if token_j in local_domain and can_bind(token_j, token_i):
            return 0.1  # Discourage local binding
        else:
            return 1.0

    elif i_type == 'r_expression':
        # Condition C: Must not be bound anywhere
        if can_bind(token_j, token_i):
            return 0.1
        else:
            return 1.0

    return 1.0  # Default: no constraint
```

**Integration with ASA:**

This becomes a 6th axis or a modifier to Axis 2 (Valence):
- Token embedding includes NP type feature
- Attention computation includes binding mask
- Binding violations reduce attention weight

**Empirical Prediction:** Models with binding-constrained attention should show improved:
- Coreference resolution
- Reflexive interpretation
- Long-distance dependency handling

### 3.2 MISSING CONNECTION 2: Information-Theoretic Compositionality

**Key Work:** Voita, Titov, et al. (2019-2024) "Information-Theoretic Probing"

**Core Insight:** Use information theory to measure how much meaning is preserved through composition.

**Application to ASA:**

Measure MUTUAL INFORMATION between:
- Predetermined constraint satisfaction
- Resulting attention patterns

```
I(ASA_constraints; Attention) = ?
```

High mutual information = ASA constraints strongly influence attention
Low mutual information = constraints are ignored

**Novel Metric: Compositional Information Preservation (CIP)**

```
CIP(phrase) = I(constraint(word1) + constraint(word2); attention(phrase))
              -------------------------------------------------------
              I(constraint(word1); attention(word1)) + I(constraint(word2); attention(word2))
```

CIP = 1.0: Perfect compositional preservation
CIP < 1.0: Information lost in composition
CIP > 1.0: Emergent information in composition (impossible if constraints are sufficient)

**Use Case:** CIP can identify which ASA axes contribute most to compositional meaning.

### 3.3 MISSING CONNECTION 3: Construction Grammar Integration

**Current Gap:** ASA focuses on word-level constraints. Construction Grammar (Goldberg 1995, 2006) shows that CONSTRUCTIONS (form-meaning pairings larger than words) carry meaning.

**Key Example:**
- "She sneezed the napkin off the table"
- SNEEZE is intransitive
- But the CAUSED-MOTION construction provides transitive structure

**Novel Proposal: Construction Overlays**

Add "construction layer" to ASA:

1. **Detect Construction:** Pattern match for construction templates
2. **Apply Construction Meaning:** Overlay construction-specific constraints
3. **Compose:** Word constraints + Construction constraints

```python
CONSTRUCTIONS = {
    'caused_motion': {
        'pattern': 'NP V NP PP[directional]',
        'meaning': {'force_type': 'cause', 'result': 'motion'},
        'coerces': {'verb': '+transitive', 'object': '+patient'}
    },
    'ditransitive': {
        'pattern': 'NP V NP NP',
        'meaning': {'force_type': 'transfer'},
        'coerces': {'verb': '+transfer', 'object1': '+recipient', 'object2': '+theme'}
    },
    # ... more constructions
}

def apply_construction(sentence, word_constraints):
    for name, construction in CONSTRUCTIONS.items():
        if matches(sentence, construction['pattern']):
            # Overlay construction constraints
            word_constraints = merge(word_constraints, construction['coerces'])
            # Add construction meaning
            sentence_meaning = add_construction_meaning(construction['meaning'])
    return word_constraints, sentence_meaning
```

**Relevance:** This addresses the critique that ASA cannot handle novel compositions - constructions provide productive templates.

### 3.4 MISSING CONNECTION 4: Incremental Semantic Processing

**Current Gap:** ASA treats sentences holistically. Human processing is INCREMENTAL - we build meaning word by word.

**Key Work:** Hale (2001, 2016) "Surprisal Theory"; Levy (2008); Frank (2013)

**Proposal: Incremental Constraint Satisfaction**

Model meaning as evolving constraint satisfaction:

```
After word 1: constraint_state_1 = update(initial, word_1_constraints)
After word 2: constraint_state_2 = update(constraint_state_1, word_2_constraints)
...

Final meaning = constraint_state_n
```

**Anomaly Detection:**

At each word, compute:
```
surprisal(word_i) = -log P(constraint_satisfaction | constraint_state_{i-1})
```

High surprisal = constraint violation = processing difficulty.

**Empirical Prediction:** ASA surprisal should correlate with:
- Reading times (eye-tracking data)
- N400 amplitude (EEG)
- Acceptability judgments

### 3.5 MISSING CONNECTION 5: Distributional Hypothesis Refinement

**Standard View:** "Words that occur in similar contexts have similar meanings" (Harris 1954)

**Critique:** This is too coarse. Different types of distributional similarity indicate different semantic relations:

| Similarity Type | Context Pattern | Semantic Relation |
|-----------------|-----------------|-------------------|
| Paradigmatic | Same syntactic position | Synonymy, antonymy |
| Syntagmatic | Co-occurrence | Association, selectional |
| Taxonomic | Hypernym contexts | IS-A hierarchy |
| Attributive | Modifier contexts | Property sharing |

**ASA Refinement:**

Each ASA axis corresponds to a different distributional pattern:
- Axis 1 (Type): Taxonomic distribution
- Axis 2 (Valence): Syntagmatic distribution with verbs
- Axis 3 (Qualia): Attributive distribution with adjectives
- Axis 4 (Force): Syntagmatic distribution in causal contexts
- Axis 5 (Position): Paradigmatic distribution at all levels

**Novel Validation:**

For each axis, compute axis-specific distributional similarity:
```
sim_axis_i(word1, word2) = distributional_similarity(word1, word2, contexts_for_axis_i)
```

ASA prediction: Axis embeddings should correlate with axis-specific distributional similarity.

---

## Part 4: Critiques of Existing Frameworks (Informing ASA Design)

### 4.1 Critiques of DisCoCat

**Critique 1: Tensor Explosion**
- Transitive verbs require O(d^3) parameters (d = dimension)
- Impractical for large vocabularies
- **ASA Response:** Sparse tensors from predetermined structure

**Critique 2: No Polysemy Handling**
- Each word has one tensor
- Polysemous words problematic
- **ASA Response:** Graded types + coercion mechanism

**Critique 3: Limited Discourse**
- Sentence-level only
- No discourse coherence
- **ASA Response:** Out of scope for now, but construction overlay helps

### 4.2 Critiques of Generative Lexicon

**Critique 1: Underspecified Formalism**
- Dot types are informal
- Coercion triggers unclear
- **ASA Response:** Adopt TTR for rigor

**Critique 2: English-Centric Examples**
- "book" examples may not generalize
- Cross-linguistic variation in polysemy
- **ASA Response:** Validate cross-linguistically (per ADR-011)

**Critique 3: Acquisition Problem**
- How do children learn qualia structure?
- No developmental account
- **ASA Response:** Not ASA's concern; use linguistic resources

### 4.3 Critiques of VerbNet

**Critique 1: Classification is Not Explanatory**
- Verb classes are descriptive, not explanatory
- Why do these verbs pattern together?
- **ASA Response:** Use classes as features, not explanations

**Critique 2: Membership is Gradient**
- Verbs don't clearly belong to single class
- Many verbs in multiple classes
- **ASA Response:** Graded membership scores

**Critique 3: Limited to English**
- No VerbNet for other languages
- Claims about universality untested
- **ASA Response:** Treat universality as hypothesis (ADR-011)

### 4.4 Novel Design Principles for ASA

Based on critiques, ASA should:

1. **Be formally explicit** - Use TTR or similar, not informal notation
2. **Handle gradience** - All type assignments are graded
3. **Separate description from explanation** - Axes describe constraints, not explain them
4. **Test cross-linguistically** - Before claiming universality
5. **Keep structure minimal** - Avoid tensor explosion via sparsity

---

## Part 5: Concrete Recommendations

### 5.1 Immediate Priorities (Week 1-2)

**R1: Implement Binding-Constrained Attention**
- Add binding theory mask to ASA
- Test on coreference resolution
- Metric: Accuracy on Winograd Schema

**R2: Conduct Causal Abstraction Analysis**
- Define axis interventions
- Measure causal effect on attention
- Report intervention effects for each axis

**R3: Design Graded Composition Experiment**
- Track constraint satisfaction through composition
- Measure grade propagation
- Correlate with acceptability judgments

### 5.2 Medium-Term (Weeks 3-8)

**R4: Formalize in TTR**
- Replace informal type notation with record types
- Define coercion as type repair
- Prove compositionality properties

**R5: Add Construction Layer**
- Implement 10 core constructions
- Test on caused-motion, ditransitive, resultative
- Measure improvement on novel compositions

**R6: Build Incremental Processor**
- Implement word-by-word constraint satisfaction
- Generate surprisal predictions
- Validate against reading time data

### 5.3 Long-Term (Months 3-6)

**R7: Cross-Linguistic Validation**
- Implement ASA for German, Spanish
- Test constraint transfer
- Report what transfers and what does not

**R8: Integrate with Binding Theory Formally**
- Full Binding Theory implementation
- Test on BLiMP binding paradigms
- Report accuracy by condition (A, B, C)

**R9: Publish Methodology Paper**
- Causal abstraction for semantic constraints
- Reproducible experimental protocol
- Release code and data

---

## Part 6: Key References

### Compositional Generalization
- Hupkes, D., et al. (2023). "State-of-the-art Generalisation Research in NLP: A Taxonomy and Review"
- Ontanon, S., et al. (2022). "Making Transformers Solve Compositional Tasks"
- Greff, K., et al. (2020). "On the Binding Problem in Artificial Neural Networks"

### Type Theory for NLP
- Luo, Z. (2012, 2024). "Formal Semantics in Modern Type Theories"
- Bekki, D. (2023). "Dependent Type Semantics"
- Chatzikyriakidis, S., & Luo, Z. (2020). "Formal Semantics in Modern Type Theories"
- Cooper, R. (2023). "Type Theory with Records for Natural Language Semantics"

### Categorical Semantics
- Coecke, B., Sadrzadeh, M., & Clark, S. (2010). "Mathematical Foundations for DisCoCat"
- de Felice, G., et al. (2020-2024). "DisCoPy: Categorical Semantics in Python"
- Meichanetzidis, K., et al. (2023). "Quantum Natural Language Processing"

### Neural-Symbolic Integration
- Garcez, A., & Lamb, L. (2023). "Neurosymbolic AI: The Third Wave"
- Li, Z., et al. (2023). "Scallop: A Language for Neurosymbolic Programming"
- Geiger, A., et al. (2024). "Causal Abstractions of Neural Networks"

### Information-Theoretic Approaches
- Voita, E., & Titov, I. (2020). "Information-Theoretic Probing"
- Pimentel, T., et al. (2022). "Information-Theoretic Measures of Compositionality"

### Construction Grammar
- Goldberg, A. (1995, 2006). "Constructions: A Construction Grammar Approach"
- Boas, H., & Sag, I. (2012). "Sign-Based Construction Grammar"

---

## Appendix A: Limitations

**Web Search Unavailable**

This research was conducted without web search access. Potential gaps:
- Publications from late 2024/2025 not covered
- Preprints and working papers missed
- Recent benchmark results not verified

**Recommended Follow-Up Searches:**
1. "compositional generalization transformers 2024 2025"
2. "type theory NLP dependent types 2024"
3. "DisCoCat neural implementation 2024"
4. "causal abstraction language models 2024"
5. "construction grammar computational 2024"

---

## Appendix B: Summary of Novel Connections

| Connection | Current ASA Status | Proposed Integration | Expected Benefit |
|------------|-------------------|---------------------|------------------|
| Binding Theory | Mentioned only | Add binding mask | Improved coreference |
| Modular Composition | Partial (masks) | Assign heads to axes | Cleaner composition |
| Linear Relational | Not explored | Validate linearity | Mathematical foundation |
| Causal Abstraction | Not used | Validation methodology | Rigorous testing |
| Dependent Types | Not used | Formalize coercion | Theoretical clarity |
| Graded Types | Implicit | Make explicit | Better gradience |
| Construction Grammar | Not used | Add construction layer | Novel compositions |
| Incremental Processing | Not used | Add surprisal model | Psycholinguistic validation |
| Information-Theoretic | Not used | CIP metric | Axis contribution |
| Distributional Refinement | Partial | Per-axis similarity | Axis validation |

---

*Research Report prepared by Research Specialist*
*ASA Research Swarm - January 2, 2026*
*Note: Web search was unavailable; findings based on knowledge through January 2025*
