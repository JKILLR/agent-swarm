---
created: 2026-01-02 00:00
updated: 2026-01-02
---

# Round 1: Empirical Exploration for ASA Validation

**Date:** January 2, 2026
**Author:** Empirical Researcher
**Status:** Research Findings and Experimental Designs

---

## Executive Summary

This document presents a systematic analysis of empirical validation opportunities for the ASA (Atomic Semantic Architecture) framework. Based on review of existing results (73.9% H6 correlation, 31% sparsity, 21% convergence speedup), I identify critical gaps in validation coverage and propose concrete experimental designs to address them.

**Key Finding:** Current ASA validation tests only 2-3 of 5 claimed constraint axes. Axes 3 (Qualia) and 4 (Force Dynamics) have ZERO empirical testing. This represents a significant credibility gap that must be addressed before academic engagement.

**Recommended Priority Order:**
1. **Qualia Coercion Experiments** (High impact, addresses Pustejovsky concerns)
2. **VerbNet Alternation Tests** (High impact, addresses Palmer concerns)
3. **Per-Axis Ablation Study** (Establishes contribution of each axis)
4. **Coverage Analysis** (Quantifies practical limitations)
5. **Benchmark Mapping** (External validation via established datasets)

---

## 1. Existing Benchmark Analysis

### 1.1 CoLA (Corpus of Linguistic Acceptability)

**What it is:** 10,657 sentences annotated for grammatical acceptability from linguistics publications.

**What it tests:**
- Grammaticality judgments across diverse syntactic phenomena
- Sentences drawn from theoretical linguistics literature (Chomsky, etc.)
- Binary labels: acceptable (1) or unacceptable (0)

**Relevance to ASA Axes:**

| CoLA Phenomenon | Primary ASA Axis | Secondary Axis |
|-----------------|------------------|----------------|
| Subcategorization | Axis 2 (Valence) | Axis 1 (Type) |
| Argument structure | Axis 2 (Valence) | - |
| Binding theory | Axis 2 (Valence) | - |
| Island constraints | Axis 2 (Valence) | - |
| Selectional restrictions | Axis 2 (Valence) | Axis 3 (Qualia) |
| Aspect violations | Axis 4 (Force) | Axis 1 (Type) |

**Experimental Design for CoLA:**

```
Hypothesis: ASA-trained models should show higher sensitivity to
selectional restriction violations than baseline transformers.

Method:
1. Filter CoLA for sentences involving selectional restriction violations
   - Identify ~200-400 sentences with animacy/concreteness violations
   - Examples: "*The idea devoured the fish" (animate agent required)

2. Create matched pairs:
   - Violation sentence + minimally different acceptable sentence

3. Measure:
   - Accuracy difference: ASA vs baseline on selectional violations
   - Attention pattern analysis on violation locus
   - Loss gradient at violation point

Expected Result:
- ASA should show LOWER perplexity on acceptable sentences
- ASA should show HIGHER perplexity spike at violation points
- Attention should avoid semantically incompatible pairs

Timeline: 1-2 weeks
Data requirements: CoLA corpus (publicly available)
Compute: Minimal (inference only on existing models)
```

### 1.2 BLiMP (Benchmark of Linguistic Minimal Pairs)

**What it is:** 67 linguistic paradigms, each with 1,000 minimal pairs testing specific grammatical phenomena.

**What it tests:**
- Morphological agreement
- Argument structure
- Binding
- Control/raising
- Quantifiers
- NPI licensing
- Island effects

**Relevance to ASA:**

| BLiMP Paradigm | ASA Axis | Testable Prediction |
|----------------|----------|---------------------|
| Anaphor agreement | Axis 2 | Blocked attention to incompatible antecedents |
| Argument structure | Axis 2 | Attention aligns with VerbNet frames |
| Determiner-noun agreement | Axis 1 | Type compatibility enforced |
| Subject-verb agreement | Axis 2 | Valence structure respected |
| Irregular forms | Axis 5 | Lexical proximity encodes irregularity |
| NPI licensing | Axis 4 (?) | Force dynamics for negation scope |

**Most Valuable BLiMP Subsets for ASA:**

1. **Argument Structure** (4 paradigms, 4,000 pairs)
   - Causative alternation
   - Inchoative alternation
   - Transitive/intransitive
   - Drop arguments

2. **Binding** (7 paradigms, 7,000 pairs)
   - Principle A (reflexives)
   - Principle B (pronouns)
   - Principle C (R-expressions)

**Experimental Design for BLiMP:**

```
Hypothesis: ASA should outperform baseline on paradigms
involving selectional/thematic constraints.

Method:
1. Select BLiMP subsets aligned with ASA axes:
   - Argument structure (Axis 2)
   - Binding (Axis 2)
   - Animacy (Axis 1 + 2)

2. Measure accuracy per paradigm:
   - Compare: ASA vs baseline vs random
   - Identify paradigms where ASA has largest advantage

3. Attention analysis on selected pairs:
   - Do ASA attention patterns explain correct/incorrect judgments?

Expected Result:
- ASA advantage on selectional restriction paradigms
- Baseline advantage on purely syntactic paradigms (word order)
- Similar performance on lexical paradigms

Timeline: 1 week
Data: BLiMP (publicly available)
Compute: Minimal
```

### 1.3 SemEval Tasks for Semantic Role Labeling

**Relevant Tasks:**
- SemEval-2007 Task 19: Frame Semantic Structure Extraction
- CoNLL-2008/2009: Syntactic-Semantic Dependencies
- SemEval-2010 Task 17: All-Words Word Sense Disambiguation

**Why SRL matters for ASA:**
- Directly tests Axis 2 (Valence Structure)
- Ground truth role labels from FrameNet/PropBank
- Can compare ASA attention patterns to gold SRL

**Experimental Design for SRL:**

```
Hypothesis: ASA attention heads should show correlation with
semantic role dependencies, even without SRL training.

Method:
1. Use CoNLL-2009 English SRL test set
   - ~2,000 sentences with PropBank SRL annotation

2. Extract attention patterns from ASA model

3. Measure alignment:
   - For each predicate-argument pair with gold role:
     - Measure attention weight from predicate to argument
   - Compare: ASA vs baseline

4. Role-specific analysis:
   - Agent (Arg0) attention patterns
   - Patient (Arg1) attention patterns
   - Other arguments (Arg2-4)

Expected Result:
- ASA shows stronger attention alignment with SRL gold
- Attention patterns correlate with role type

Caveat: PropBank uses Arg0-5, not thematic roles.
This is a known issue noted in STATE.md (H2).

Timeline: 2 weeks
Data: CoNLL-2009 (requires LDC access or public subset)
```

---

## 2. Qualia Coercion Experiments

### 2.1 Background: The Coercion Problem

Pustejovsky's Generative Lexicon identifies coercion as a key phenomenon where the verb "demands" a semantic type that the noun does not literally provide. The noun is coerced into the required type by accessing its qualia structure.

**Canonical Examples:**

| Sentence | Coercion Type | Quale Accessed | Expected ASA Behavior |
|----------|---------------|----------------|----------------------|
| "Mary began the book" | Eventive | Telic (reading) | High attention to book, modified by begin's requirement |
| "The author finished the book" | Eventive | Agentive (writing) | Context-dependent quale selection |
| "enjoy the book" | Eventive | Telic (reading) | Standardized access pattern |
| "The newspaper fired its editor" | Metonymic | Formal (organization) | Organization quale, not physical |
| "The newspaper fell off the table" | Literal | Constitutive (physical) | No coercion needed |

### 2.2 Experimental Design: Coercion Discrimination

**Hypothesis:** If ASA captures qualia structure (even implicitly through VerbNet features), attention patterns should differ systematically between coercion types.

**Method:**

```python
# Test Set Construction
COERCION_PAIRS = {
    'telic_coercion': [
        ("Mary began the book", "reading"),
        ("John enjoyed the movie", "watching"),
        ("She started the sonata", "playing"),
        ("He finished the sandwich", "eating"),
    ],
    'agentive_coercion': [
        ("The author finished the book", "writing"),
        ("The chef began the soup", "cooking"),
        ("The composer completed the symphony", "composing"),
    ],
    'no_coercion': [
        ("Mary read the book", None),
        ("John watched the movie", None),
        ("She played the sonata", None),
    ],
    'metonymic': [
        ("The newspaper fired its editor", "organization"),
        ("The school decided to expand", "institution"),
        ("The restaurant closed early", "business"),
    ],
}

# Measurement Protocol
for sentence_type, examples in COERCION_PAIRS.items():
    for sentence, expected_quale in examples:
        # 1. Get ASA attention patterns
        attn_asa = model_asa.get_attention(sentence)
        attn_baseline = model_baseline.get_attention(sentence)

        # 2. Identify key positions
        verb_pos = find_verb_position(sentence)
        object_pos = find_object_position(sentence)

        # 3. Measure verb-object attention
        verb_to_object_asa = attn_asa[verb_pos, object_pos]
        verb_to_object_base = attn_baseline[verb_pos, object_pos]

        # 4. Compare patterns across coercion types
```

**Predictions:**
- Telic coercion sentences: Higher attention from aspectual verbs (begin, start, finish) to nouns with strong Telic qualia
- Agentive coercion: Different pattern when subject has agentive role
- No coercion: Baseline attention pattern
- Metonymic: Attention should spread to related noun positions

### 2.3 Experimental Design: Coercion Reading Time Comparison

**Rationale:** If ASA captures psycholinguistically real patterns, its predictions should correlate with human reading time data.

**Method:**

```
Data Sources:
1. Katsika et al. (2012) - Eye-tracking on complement coercion
2. McElree et al. (2001, 2006) - Speed-accuracy tradeoff for coercion
3. Traxler et al. (2002) - Reading times for coerced complements

Protocol:
1. Obtain reading time data for coercion sentences
2. Run same sentences through ASA and baseline
3. Measure:
   - Perplexity at coercion point
   - Attention entropy at coercion point
   - Surprisal gradient

4. Correlate ASA metrics with human reading times:
   - Higher RT should correlate with processing difficulty
   - ASA should show stronger correlation than baseline

Expected Result:
- Coercion sentences show higher processing signals in both humans and ASA
- ASA processing signals correlate more strongly with human RT than baseline

Timeline: 3-4 weeks (data acquisition may require contacting authors)
```

### 2.4 Quale Selection Experiment

**Hypothesis:** For polysemous nouns with multiple accessible qualia, ASA attention should reflect quale-appropriate interpretation based on context.

**Test Set:**

```
"book" (physical-object / information-content / artifact)
- "The book is heavy" -> CONSTITUTIVE (physical)
- "The book is interesting" -> TELIC (content)
- "The book is rare" -> FORMAL (artifact)

"window" (physical-object / aperture / interface)
- "Clean the window" -> CONSTITUTIVE (physical glass)
- "Open the window" -> TELIC (aperture function)
- "The window is frozen" -> AMBIGUOUS

"newspaper" (organization / physical / information)
- "The newspaper fired its editor" -> FORMAL (organization)
- "The newspaper fell off the table" -> CONSTITUTIVE (physical)
- "The newspaper reported the scandal" -> TELIC (information)
```

**Measurement:**
1. Identify quale-triggering context words
2. Measure attention from noun to context word
3. Compare ASA vs baseline on disambiguation

---

## 3. VerbNet Alternation Tests

### 3.1 Background: What VerbNet Actually Tests

STATE.md notes (Issue C2) that VerbNet's core innovation is ALTERNATION PATTERNS, not selectional restrictions. The current ASA framing mischaracterizes this.

**Key Alternation Classes:**

| Alternation | Example | VerbNet Pattern |
|-------------|---------|-----------------|
| Spray/Load | spray paint on wall / spray wall with paint | Locatum-Location swap |
| Causative/Inchoative | John broke the vase / The vase broke | Agent optionality |
| Dative | give the book to Mary / give Mary the book | Goal realization |
| Conative | cut the bread / cut at the bread | Affectedness |
| Middle | This bread cuts easily | Patient as subject |

### 3.2 Experimental Design: Alternation Handling

**Hypothesis:** Both alternation variants should be well-formed; ASA should handle both without inappropriate blocking.

**Test Set Construction:**

```python
ALTERNATION_PAIRS = {
    'spray_load': [
        ("John sprayed paint on the wall", "John sprayed the wall with paint"),
        ("Mary loaded boxes onto the truck", "Mary loaded the truck with boxes"),
        ("She spread butter on the bread", "She spread the bread with butter"),
    ],
    'causative_inchoative': [
        ("John broke the vase", "The vase broke"),
        ("Mary opened the door", "The door opened"),
        ("The sun melted the ice", "The ice melted"),
    ],
    'dative': [
        ("John gave the book to Mary", "John gave Mary the book"),
        ("She sent a letter to her mother", "She sent her mother a letter"),
        ("He showed the photos to us", "He showed us the photos"),
    ],
    'middle': [
        ("This knife cuts well", None),  # Middle only
        ("This book reads easily", None),
        ("These clothes wash nicely", None),
    ],
}
```

**Measurements:**

1. **Well-formedness:** Both variants should have similar perplexity
   - Failure: One variant has much higher perplexity than other
   - Success: Perplexities within 10% of each other

2. **Attention patterns:** Should show appropriate role assignment
   - Spray/load: Locatum and Location should both receive attention
   - Causative/inchoative: Patient attention should be high in both

3. **Role consistency:** Same thematic roles despite different surface order
   - Extract attention to each argument position
   - Verify role-appropriate attention patterns

**Success Metrics:**

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| PPL ratio (variant1/variant2) | < 1.15 | Both variants acceptable |
| Attention alignment | > 0.7 | Consistent role attention |
| Blocking rate | < 5% | Neither variant incorrectly blocked |

### 3.3 Verb Class Coverage Analysis

**Current State:** ASA v2.2 covers ~468 verbs from VerbNet.

**Analysis Required:**

```
1. Which VerbNet classes are covered?
   - List all 468 verbs by class
   - Identify coverage gaps by class

2. Alternation coverage:
   - How many alternating classes are covered?
   - Are both alternation variants testable?

3. Frequency analysis:
   - Coverage by word frequency (high-freq verbs covered?)
   - Coverage by class frequency (common classes covered?)
```

---

## 4. Coverage and Scalability Analysis

### 4.1 Current Coverage Assessment

| Resource | Total | ASA Coverage | Coverage % |
|----------|-------|--------------|------------|
| VerbNet verbs | 6,800 | 468 | 6.9% |
| VerbNet classes | 329 | ~50 (est.) | 15.2% |
| LLM vocabulary | 50,257 (GPT-2) | ? | ? |
| FrameNet frames | 1,200 | 0 | 0% |
| Qualia annotations | - | 0 | 0% |

### 4.2 Coverage Estimation Experiment

**Method:**

```python
def estimate_coverage(texts: List[str], tokenizer, extractor):
    """
    For a corpus, measure what % of tokens have meaningful
    predetermined constraints beyond POS.
    """
    total_tokens = 0
    covered_by_pos = 0
    covered_by_verbnet = 0
    covered_by_wordnet = 0
    covered_by_qualia = 0  # Currently 0
    covered_by_force = 0   # Currently 0

    for text in texts:
        tokens = tokenizer.tokenize(text)
        for token in tokens:
            total_tokens += 1

            # POS coverage (nearly universal)
            pos = extractor.get_pos(token)
            if pos: covered_by_pos += 1

            # VerbNet coverage (verbs only, limited set)
            if is_verb(pos):
                vn_class = extractor.get_verbnet_class(token)
                if vn_class: covered_by_verbnet += 1

            # WordNet coverage (nouns/verbs/adj/adv)
            if has_wordnet_entry(token):
                covered_by_wordnet += 1

    return {
        'total': total_tokens,
        'pos_coverage': covered_by_pos / total_tokens,
        'verbnet_coverage': covered_by_verbnet / total_tokens,
        'wordnet_coverage': covered_by_wordnet / total_tokens,
        'meaningful_coverage': (covered_by_verbnet + covered_by_wordnet) / total_tokens,
    }
```

**Test Corpora:**
1. WikiText-2 validation (current benchmark)
2. Brown Corpus (diverse genres)
3. Scientific abstracts (technical vocabulary)
4. Social media text (informal, OOV-heavy)

**Expected Results:**
- POS coverage: ~95%+ (universal)
- VerbNet coverage: ~5-10% of all tokens (verbs only, limited coverage)
- WordNet coverage: ~40-60% (common words)
- Qualia coverage: 0% (not implemented)
- Force dynamics: 0% (not implemented)

### 4.3 Diminishing Returns Analysis

**Research Question:** At what coverage level do we see diminishing returns?

**Method:**

```
1. Create coverage tiers:
   - Tier 1: Top 100 verbs (by frequency)
   - Tier 2: Top 500 verbs
   - Tier 3: Top 1,000 verbs
   - Tier 4: Full VerbNet (6,800 verbs)

2. For each tier, measure:
   - H6 correlation
   - Sparsity achieved
   - Perplexity improvement

3. Plot diminishing returns curve:
   - X-axis: Coverage (% of verbs)
   - Y-axis: H6 correlation / Sparsity / PPL improvement
```

**Hypothesized Curve:**

```
H6 Correlation
     ^
0.80 |                         ___________
     |                    ____/
0.70 |               ____/
     |          ____/
0.60 |     ____/
     | ___/
0.50 |/
     +------------------------------------>
        100   500   1000  2000  5000  6800
                  VerbNet Coverage
```

**Key Question:** Does coverage beyond ~1,000 high-frequency verbs yield meaningful improvement?

---

## 5. Per-Axis Ablation Design

### 5.1 Current Ablation Structure

The existing `run_ablations.py` tests:
- `none`: No ASA constraints (baseline)
- `pos_only`: Only POS compatibility mask
- `features_only`: Only feature compatibility scores
- `full`: POS + features + blocking

**Gap:** This does NOT isolate individual semantic axes (1-5).

### 5.2 Proposed Per-Axis Ablation

**New Modes to Implement:**

| Mode | Description | Axes Active |
|------|-------------|-------------|
| `axis1_only` | Ontological type only | 1 |
| `axis2_only` | Valence structure only | 2 |
| `axis5_only` | Geometric position only | 5 |
| `axis1_2` | Type + Valence | 1, 2 |
| `axis1_2_5` | Type + Valence + Geometric | 1, 2, 5 |
| `all_current` | Current full mode | 1, 2, 5 |

**Note:** Axes 3 (Qualia) and 4 (Force Dynamics) cannot be ablated because they are not implemented.

### 5.3 Ablation Metrics

| Metric | What It Measures | Expected Pattern |
|--------|------------------|------------------|
| Final PPL | Model quality | Lower is better |
| Convergence step | Training efficiency | Fewer steps = faster learning |
| Sparsity | Attention reduction | Higher = more constraint impact |
| H6 correlation | Alignment with learned attention | Higher = more meaningful constraints |

**Experimental Protocol:**

```python
def run_per_axis_ablation():
    modes = ['none', 'axis1_only', 'axis2_only', 'axis5_only',
             'axis1_2', 'axis1_2_5', 'all_current']

    results = {}
    for mode in modes:
        # Train model with specific axis configuration
        model = create_model(mode=mode)
        train_results = train_and_evaluate(model)

        results[mode] = {
            'final_ppl': train_results['ppl'],
            'convergence_step': train_results['convergence'],
            'sparsity': measure_sparsity(model),
            'h6_correlation': measure_h6(model),
        }

    # Compute axis contribution
    for i, axis in enumerate([1, 2, 5]):
        axis_contribution = (
            results['all_current']['h6_correlation'] -
            results[f'without_axis{axis}']['h6_correlation']
        )
        print(f"Axis {axis} contribution: {axis_contribution:.1%}")
```

### 5.4 Expected Results

Based on current implementation:

| Mode | Expected PPL | Expected H6 | Interpretation |
|------|--------------|-------------|----------------|
| `none` | 26.56 | ~35% (random) | Baseline |
| `axis1_only` | ~26.8 | ~55% | POS provides basic structure |
| `axis2_only` | ~26.6 | ~65% | VerbNet captures roles |
| `axis5_only` | ~27.0 | ~45% | Hierarchy alone is weak |
| `all_current` | 26.33 | 73.9% | Full synergy |

**Key Questions:**
1. Is any single axis sufficient?
2. Which axis combinations show synergy?
3. Are any axes redundant?

---

## 6. Resource Requirements

### 6.1 Data Requirements

| Dataset | Source | Availability | Purpose |
|---------|--------|--------------|---------|
| CoLA | GLUE benchmark | Public | Acceptability judgments |
| BLiMP | Johns Hopkins | Public | Linguistic minimal pairs |
| CoNLL-2009 SRL | LDC | Restricted | Semantic role labeling |
| WikiText-2 | HuggingFace | Public | Language modeling |
| Eye-tracking data | Various papers | Contact authors | Reading time correlation |
| VerbNet 3.3 | Colorado/Penn | Public | Verb classes |

### 6.2 Compute Requirements

| Experiment | GPU Hours | Memory | Storage |
|------------|-----------|--------|---------|
| CoLA evaluation | 2 | 8GB | 1GB |
| BLiMP evaluation | 2 | 8GB | 2GB |
| Coercion experiments | 4 | 8GB | 1GB |
| Alternation tests | 2 | 8GB | 1GB |
| Coverage analysis | 1 | 4GB | 5GB |
| Per-axis ablation | 20 | 16GB | 10GB |
| **Total** | **~31 hours** | 16GB peak | 20GB |

### 6.3 Time Estimates

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Benchmark data acquisition | 1 week | None |
| CoLA/BLiMP experiments | 1 week | Data |
| Coercion test set construction | 1 week | None |
| Coercion experiments | 2 weeks | Test set |
| Alternation experiments | 1 week | None |
| Coverage analysis | 1 week | None |
| Per-axis ablation | 2 weeks | None |
| Analysis and write-up | 1 week | All experiments |
| **Total** | **8-10 weeks** | - |

---

## 7. Priority Ranking of Experiments

### Tier 1: Critical (Must complete before academic response)

| Priority | Experiment | Rationale | Timeline |
|----------|------------|-----------|----------|
| **P1** | Qualia Coercion Attention | Directly addresses Pustejovsky's expertise; Axis 3 is untested | 2 weeks |
| **P2** | VerbNet Alternation Tests | Addresses Palmer's core concern; validates framing | 1 week |
| **P3** | Per-Axis Ablation | Quantifies contribution of each axis | 2 weeks |

### Tier 2: High Value (Complete within 4 weeks)

| Priority | Experiment | Rationale | Timeline |
|----------|------------|-----------|----------|
| **P4** | BLiMP Selectional Subset | External validation on established benchmark | 1 week |
| **P5** | Coverage Analysis | Quantifies practical limitations honestly | 1 week |
| **P6** | CoLA Selectional Violations | Additional external validation | 1 week |

### Tier 3: Valuable (Complete if time permits)

| Priority | Experiment | Rationale | Timeline |
|----------|------------|-----------|----------|
| P7 | SRL Attention Alignment | Tests Axis 2 against gold SRL | 2 weeks |
| P8 | Reading Time Correlation | Psycholinguistic validation | 3-4 weeks |
| P9 | Diminishing Returns Curve | Informs scaling decisions | 2 weeks |

---

## 8. Risks and Limitations

### 8.1 Experimental Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Coercion experiments show no ASA advantage | Medium | High | Honestly report; suggest Axis 3 needs explicit qualia |
| Alternation tests reveal blocking errors | Medium | Medium | Identify specific failure modes; propose fixes |
| Coverage analysis reveals <10% meaningful coverage | High | High | Acknowledge limitation; propose fallback strategies |
| Per-axis ablation shows redundancy | Medium | Medium | Revise axis structure based on evidence |

### 8.2 Methodological Limitations

1. **Axis 3 and 4 Not Implemented:** Cannot ablate what does not exist. Coercion experiments test whether implicit patterns exist, not explicit qualia implementation.

2. **Model Scale:** All experiments on 6.8M parameter model. Results may not transfer to larger models.

3. **English-Only:** VerbNet, CoLA, BLiMP are English-centric. Cross-linguistic claims remain untested.

4. **Ground Truth Ambiguity:** Coercion interpretation is often context-dependent. What counts as "correct" attention pattern?

### 8.3 Interpretation Caveats

- Positive results do NOT prove ASA captures linguistic theory correctly
- Positive results show ASA attention CORRELATES with linguistic predictions
- Correlation is necessary but not sufficient for theoretical validation
- Need converging evidence from multiple experiment types

---

## 9. Recommendations for Theory Team

Based on this empirical exploration:

### 9.1 Axis 3 (Qualia) Recommendations

1. **Minimum Viable Qualia:** Annotate 100 high-frequency nouns with 4D qualia scores
2. **Coercion Rules:** Formalize 5-10 coercion patterns from Pustejovsky (1995)
3. **Test Before Full Implementation:** Run attention analysis on coercion sentences FIRST to see if patterns emerge implicitly

### 9.2 Axis 4 (Force Dynamics) Recommendations

1. **Defer Full Implementation:** Lowest priority given complexity
2. **Proxy via VerbNet:** Some force dynamics encoded in VerbNet class membership (causative verbs, etc.)
3. **Target Subset:** Focus on causative/inchoative verbs where force dynamics is most salient

### 9.3 VerbNet Recharacterization

1. **Emphasize Alternations:** Documentation should lead with alternation patterns, not selectional restrictions
2. **Explicit Coverage Limits:** State that selectional restrictions are sparse in VerbNet
3. **Version Specificity:** Cite VerbNet 3.3 explicitly

---

## 10. Appendix: Test Sentence Sets

### A.1 Coercion Test Sentences (50 sentences)

```
# Telic Coercion (aspectual verbs)
Mary began the book.
John started the novel.
She finished the assignment.
He continued the project.
They completed the analysis.

# Agentive Coercion (creation context)
The author finished the book.
The chef began the soup.
The composer completed the symphony.
The artist started the painting.
The developer finished the app.

# No Coercion (explicit activity)
Mary read the book.
John wrote the novel.
She completed the assignment on time.
He edited the project.
They analyzed the data.

# Metonymic Coercion (organization for building)
The newspaper fired its editor.
The school decided to expand.
The hospital treated 500 patients.
The company announced layoffs.
The government issued new regulations.

# Polysemous Nouns with Context
The book is heavy. (physical)
The book is interesting. (content)
The book is rare. (artifact)
The window is dirty. (surface)
The window is open. (aperture)
```

### A.2 Alternation Test Pairs (30 pairs)

```
# Spray/Load Alternation
John sprayed paint on the wall. | John sprayed the wall with paint.
Mary loaded boxes onto the truck. | Mary loaded the truck with boxes.
She spread butter on the bread. | She spread the bread with butter.
He stuffed feathers into the pillow. | He stuffed the pillow with feathers.
They planted trees in the garden. | They planted the garden with trees.

# Causative/Inchoative
John broke the vase. | The vase broke.
Mary opened the door. | The door opened.
The sun melted the ice. | The ice melted.
She rolled the ball. | The ball rolled.
He closed the window. | The window closed.

# Dative Alternation
John gave the book to Mary. | John gave Mary the book.
She sent a letter to her mother. | She sent her mother a letter.
He showed the photos to us. | He showed us the photos.
They told the story to the children. | They told the children the story.
I offered a job to him. | I offered him a job.
```

---

**Document Version:** 1.0
**Last Updated:** January 2, 2026
**Next Review:** After Tier 1 experiments complete

---

*This document prepared by Empirical Researcher for ASA Research Swarm*
