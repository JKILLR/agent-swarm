---
created: 2026-01-02 00:00
updated: 2026-01-02
---

# Round 2 Debate: The Coercion Mechanism Question

## Thesis-Antithesis-Synthesis on Build-First vs. Test-First

**Date:** January 2, 2026
**Facilitated by:** Research Specialist
**Topic:** Should we build principled coercion theory before testing, or test for implicit patterns before building?

---

## Executive Summary

This document presents a structured debate on the coercion mechanism question, arguing both positions vigorously before developing a synthesis. The core tension: Theory Researcher advocates for category-theoretic foundations before experimentation; Empirical Researcher advocates for testing whether implicit patterns exist before committing to theoretical machinery.

**Synthesis Conclusion:** Both positions contain essential insights but also blind spots. The RIGHT answer is neither "build first" nor "test first" but rather a **coupled minimal probe** approach: run the minimum experiment that can falsify the theoretical hypothesis, while specifying the minimum theory that would make experimental results interpretable.

**Key Resolution:** The experiment IS the theory test. We do not need full functorial coercion to have a testable prediction. If ASA already captures coercion patterns implicitly (through VerbNet and attention), that CONSTRAINS which theories are viable. If it does not, that tells us what theoretical machinery we need.

---

## Part 1: THESIS (Theory Position)

### 1.1 The Argument for Principled Foundations First

**Core Claim:** Coercion requires principled categorical foundations. Without functorial semantics, we cannot explain HOW type-shifting occurs, and any empirical patterns we observe will be theoretically uninterpretable.

**Argument Structure:**

**Premise 1: Coercion is not merely pattern-matching; it is structured type transformation.**

Pustejovsky's Generative Lexicon identifies three distinct coercion operations:
- Type Coercion: "begin the book" requires event type; noun supplies via quale
- Selective Binding: "fast car" vs "fast food" accesses different qualia
- Co-composition: "bake a potato" vs "bake a cake" involves mutual constraint

These are not surface phenomena. They involve structured access to lexical content. A model that merely "correlates" with coercion patterns without representing the underlying mechanism is not capturing coercion at all--it is capturing statistical shadows.

**Premise 2: Without theory, we cannot distinguish success from failure.**

Consider two possible experimental outcomes:
- Outcome A: ASA attention patterns correlate with coercion type at r=0.6
- Outcome B: ASA attention patterns correlate with coercion type at r=0.4

Which is success? Without a theoretical model specifying what correlation we SHOULD see, we cannot interpret these numbers. A correlation of 0.6 might be:
- Evidence that ASA implicitly captures qualia structure (optimistic reading)
- Evidence that ASA captures surface correlates of coercion but not the mechanism (skeptical reading)
- Noise from confounds we have not controlled (pessimistic reading)

**Premise 3: Category theory provides the right level of abstraction.**

Coercion is fundamentally about type relationships--how one semantic type can transform into another under compositional pressure. Category theory is the mathematics of structure-preserving transformations. Specifically:

- Types are objects in a category
- Coercion operations are morphisms
- Compositional coercion is morphism composition
- The quale structure provides the morphism content

This is not abstract machinery for its own sake. It is the MINIMAL mathematical structure that captures what coercion does.

**Premise 4: Building theory first prevents wasted experimentation.**

Without theoretical grounding, we risk:
1. Testing the wrong thing (surface patterns vs underlying mechanism)
2. Building experiments that cannot distinguish between interpretations
3. Generating "results" that experts like Pustejovsky will immediately dismiss

The Theory Researcher's functorial coercion proposal (ROUND1_THEORY_EXPLORATION.md, Section 1.2) provides:
- A formal definition of coercion as morphism composition
- A concrete representation of qualia as matrices
- Testable predictions about how coercion should manifest in attention

### 1.2 Strongest Version of the Thesis

**The strongest version:** We should not run coercion experiments until we can state precisely:
1. What mathematical object represents a quale
2. How coercion is computed (morphism application)
3. What attention pattern we predict for each coercion type
4. What experimental outcome would FALSIFY our theory

Without (1)-(4), we are doing exploratory data analysis, not hypothesis testing. Exploratory analysis is valuable, but it should not be confused with validation.

### 1.3 Evidence Supporting the Thesis

**Historical precedent:** Major advances in linguistic theory came from principled frameworks:
- Chomsky's generative grammar preceded psycholinguistic testing
- Montague semantics formalized compositionality before corpus work
- Lambek calculus established type-logical foundations before computational implementation

**ASA-specific evidence:** The existing empirical results (H6 correlation, convergence) are currently uninterpretable because we lack theory to explain them. The 73.9% H6 correlation is presented as success, but we do not know:
- Why some pairs align and others do not
- What the 26.1% non-alignment represents
- Whether the alignment is semantically meaningful or syntactically confounded

**Expert expectations:** Pustejovsky WILL ask "Where is the unification mechanism?" (STATE.md, anticipated question). Arriving with empirical patterns but no mechanistic account will not satisfy him.

---

## Part 2: ANTITHESIS (Empirical Position)

### 2.1 The Argument for Testing Before Building

**Core Claim:** Building theory before testing risks constructing elaborate machinery for phenomena that may not emerge. We should first establish whether ASA captures coercion patterns implicitly, then theorize about mechanisms.

**Argument Structure:**

**Premise 1: We do not know if there is anything to theorize about.**

The fundamental question is empirical: Does ASA capture coercion-relevant patterns AT ALL?

If ASA attention shows no systematic difference between:
- "Mary began the book" (coercion)
- "Mary read the book" (no coercion)

Then building functorial coercion theory is premature--there is nothing to explain. Conversely, if attention DOES differ systematically, we have something concrete to theorize about.

**Premise 2: Theory without data tends toward unfalsifiable abstraction.**

The history of linguistics is littered with elegant theories that collapsed on contact with data:
- Deep structure transformations (empirically inadequate)
- Theta-role inventories (varies across frameworks)
- Feature geometry in phonology (contested)

Category-theoretic semantics in particular has struggled with empirical grounding. DisCoCat is mathematically elegant but has shown mixed results in NLP applications. We risk building another beautiful theory that does not connect to reality.

**Premise 3: The experiment is informative regardless of theory.**

Consider what we learn from the proposed coercion experiment (ROUND1_EMPIRICAL_EXPLORATION.md, Section 2.2):

| Outcome | What We Learn | Theoretical Implication |
|---------|--------------|-------------------------|
| ASA attention differs by coercion type | Implicit pattern exists | Theory should explain what ASA is capturing |
| No difference from baseline | Nothing implicit | Need explicit qualia implementation |
| ASA worse than baseline | Current approach interferes | Rethink entire approach |

Each outcome informs theory. But you cannot get these outcomes without running the experiment.

**Premise 4: Minimum viable experiments are fast and cheap.**

The proposed coercion experiments require:
- 50 test sentences (already constructed)
- Attention pattern extraction (existing infrastructure)
- Statistical comparison (standard methods)
- Timeline: 2 weeks

By contrast, functorial coercion theory requires:
- Formal category definition
- Morphism specification for qualia
- Integration with ASA architecture
- Implementation and testing
- Timeline: 4+ weeks

Why commit 4+ weeks to theory when 2 weeks of empirical work could tell us whether theory is needed?

### 2.2 Strongest Version of the Antithesis

**The strongest version:** Theory development without empirical grounding is speculation dressed in mathematics. We should:

1. Run minimum viable coercion experiment (10-20 sentences, attention analysis)
2. Determine whether systematic patterns exist (YES/NO)
3. Only IF patterns exist, develop theory to explain them
4. Only IF no patterns exist, develop theory to create them

This is not anti-theoretical. It is empirically constrained theory development.

### 2.3 Evidence Supporting the Antithesis

**Deep learning precedent:** BERT, GPT, and modern transformers work WITHOUT principled semantic theory. Attention patterns capture meaningful linguistic structure despite having no explicit semantic formalization. ASA might already capture coercion implicitly.

**ASA-specific evidence:** The existing H6 correlation (73.9%) was discovered empirically. No one predicted it from theory. The empirical discovery then became a phenomenon to explain. Coercion patterns might work the same way.

**Sunk cost concern:** 4+ weeks of theoretical development is significant investment. If the experimental result shows no implicit patterns, the theory was built for a phenomenon that does not manifest in the model. If the experimental result shows strong implicit patterns, the theory might be unnecessary--the model already captures it.

**Practical urgency:** Experts may respond soon. We need SOMETHING to report. Empirical findings (even negative) are reportable. Theoretical frameworks-in-progress are not.

---

## Part 3: SYNTHESIS

### 3.1 What Each Position Gets Right

**Theory gets right:**
1. Coercion is genuinely structured, not just statistical correlation
2. Interpretation of results requires theoretical framework
3. Category theory captures the right level of abstraction for type operations
4. Without theory, we risk producing uninterpretable numbers

**Empirical gets right:**
1. We do not know if implicit patterns exist to be explained
2. Experiments are faster and cheaper than theory development
3. Historical examples of unfounded theory warn against speculation
4. Any outcome is informative, even without theory

### 3.2 What Each Position Overlooks

**Theory overlooks:**
1. The experiment can BE the theory test if properly designed
2. Theoretical elegance is not the same as correctness
3. We might be solving a problem that does not exist
4. Full functorial coercion is not needed for testable predictions

**Empirical overlooks:**
1. Without ANY theory, we cannot design discriminative experiments
2. Statistical patterns without mechanism do not answer Pustejovsky
3. "Implicit patterns" is itself a theoretical commitment
4. The baseline comparison requires specifying what counts as "baseline"

### 3.3 The Core Insight: These Are Not Independent Questions

The thesis and antithesis present a false dichotomy: build theory THEN test, or test THEN build theory.

**The synthesis:** Theory and experiment must be COUPLED. The minimum experiment is one that CAN test a theoretical prediction. The minimum theory is one that MAKES a testable prediction.

Neither pure theory development nor pure empirical exploration serves our goals. What we need is:

**Prediction-Driven Experimentation:** Specify theoretical predictions at a level that experiments can test or falsify, WITHOUT requiring the full theoretical apparatus.

### 3.4 Specific Resolution

**Q1: What is the MINIMUM experiment that would inform theory?**

**Answer:** Test whether ASA attention patterns distinguish coercion types BEFORE implementing qualia.

Specifically:
1. Select 20 sentences: 5 telic coercion, 5 agentive coercion, 5 metonymic coercion, 5 no coercion
2. Extract attention from verb to object position
3. Compute: Does attention pattern cluster by coercion type?

**Theoretical prediction to test:** If qualia structure is implicitly encoded (via VerbNet, WordNet, or learned representations), attention should show TYPE-SPECIFIC patterns. Specifically:
- Telic coercion (aspectual verbs): Higher attention to nouns with clear telic function
- Agentive coercion (creator subjects): Attention modulated by subject-object interaction
- Metonymic: Attention spread across semantically related positions

**Falsification criterion:** If attention patterns show NO systematic difference by coercion type (statistical test p > 0.1), then:
- Implicit encoding is absent
- Explicit qualia mechanism is required
- Theory development becomes warranted

**Confirmation criterion:** If attention patterns DO show systematic difference:
- Implicit encoding exists
- Theory should explain WHAT is being encoded
- But we do not need to BUILD the mechanism--it already exists implicitly

**Timeline:** 1 week

---

**Q2: What is the MINIMUM theory that would guide experimentation?**

**Answer:** We do not need full functorial coercion. We need the PREDICTION level of theory:

**Minimal theoretical commitments:**
1. Coercion types are distinct (telic vs agentive vs metonymic)
2. These types access different aspects of noun meaning
3. Different aspects should manifest as different attention patterns
4. The direction of difference is: coercion sentences show MODIFIED attention compared to non-coercion sentences

**What we do NOT need to specify (yet):**
- The categorical structure of qualia
- Morphism composition rules
- How coercion integrates with ASA's mask mechanism
- Mathematical formalization of qualia as matrices

**Theoretical output:** A one-page prediction document stating:
- For each coercion type, what attention pattern we expect
- What statistical test discriminates between patterns
- What outcome falsifies vs confirms implicit encoding

**Timeline:** 2-3 days (can be written while preparing experiment)

---

**Q3: How do we proceed if the experiment shows implicit patterns exist?**

**If implicit patterns are found:**

1. **Document the patterns:** What specific attention differences characterize each coercion type?

2. **Characterize what ASA is capturing:** Is it:
   - VerbNet class membership (verbs that take coercion have particular classes)
   - WordNet hypernym relations (nouns with rich qualia have particular positions)
   - Learned correlations from training data
   - Something else

3. **Theory development shifts:** Instead of building coercion mechanism, we need theory that EXPLAINS what existing mechanisms capture. This is lighter lift.

4. **Communication strategy:** We can tell Pustejovsky: "ASA implicitly captures coercion patterns. Here is what we observe. Our theoretical question is now: what explains this implicit encoding?"

**Timeline for post-confirmation work:** 2-3 weeks for pattern characterization and explanatory theory

---

**Q4: How do we proceed if the experiment shows NO implicit patterns?**

**If no implicit patterns are found:**

1. **Confirm the negative:** Re-run with expanded test set (50 sentences) to ensure result is robust.

2. **Theory becomes necessary:** Now we know explicit qualia mechanism is required. The functorial coercion proposal (ROUND1_THEORY_EXPLORATION.md, Proposal A) becomes the development target.

3. **Scope the implementation:**
   - Start with Proposal C (Attention-Based Quale Selection) as faster-to-implement approximation
   - Use Proposal A (Functorial Coercion) as theoretical framework for what we are approximating

4. **Communication strategy:** We tell Pustejovsky: "Current ASA does not capture coercion implicitly. We are developing an explicit mechanism based on category-theoretic foundations. Here is our proposal."

**Timeline for post-negative work:** 4-6 weeks for qualia implementation, following theory development

---

## Part 4: Concrete Next Steps

### 4.1 Immediate Actions (This Week)

| Action | Owner | Deliverable | Timeline |
|--------|-------|-------------|----------|
| Write prediction document | Theory Researcher | 1-page prediction spec | Day 1-2 |
| Prepare test sentences | Empirical Researcher | 20 sentences, 4 coercion types | Day 1-2 |
| Run attention extraction | Implementer | Attention matrices for test sentences | Day 3-4 |
| Statistical analysis | Empirical Researcher | Pattern comparison, p-values | Day 5 |
| Result interpretation | Theory + Empirical | Joint assessment meeting | Day 6-7 |

### 4.2 Decision Tree Based on Results

```
                        Run Minimum Experiment
                                |
                    +-----------+-----------+
                    |                       |
            Patterns Found           No Patterns Found
                    |                       |
            +-------+-------+               |
            |               |               |
       Strong (p<0.01)  Weak (p<0.1)    Build Theory
            |               |               |
       Characterize     Expand Test    Implement Qualia
       What ASA         Set (50        (Proposal A or C)
       Captures         sentences)          |
            |               |               |
       Explanatory     Re-analyze      Test Implementation
       Theory              |               |
                    +------+------+        |
                    |             |        |
              Confirm        Negative -> Build Theory
              Pattern
```

### 4.3 Success Criteria for the Synthesis Approach

**We will know this approach worked if:**

1. **Experiment completes in 1 week** (not 8-10 weeks for full validation suite)
2. **Results are interpretable** (prediction document specifies what counts as success/failure)
3. **Theory development is scoped appropriately** (elaborate theory only if warranted)
4. **Expert communication is possible** (we have something concrete to report)

**We will know this approach failed if:**

1. **Results are ambiguous** (cannot distinguish patterns from noise)
2. **Theory predictions were wrong** (need to revise theoretical framework)
3. **Experiment was poorly designed** (confounds prevent interpretation)

---

## Part 5: Addressing the Underlying Tension

### 5.1 Why This Debate Exists

The thesis and antithesis represent different research temperaments:
- **Theory-first:** Values explanatory depth over coverage; prefers principled foundations
- **Empirical-first:** Values evidence over speculation; prefers iterative refinement

Both temperaments have produced important science. The debate is not about which is "right" in general, but which is right FOR THIS SPECIFIC PROBLEM AT THIS SPECIFIC TIME.

### 5.2 What Makes Coercion Different

Coercion is a phenomenon where:
- **Theory is well-developed** (Pustejovsky 1995 is 30 years old)
- **Computational implementation is limited** (few NLP systems handle coercion well)
- **Empirical evidence is mixed** (psycholinguistic results vary)
- **Our system might already capture it implicitly** (via VerbNet, attention)

This makes coercion a good candidate for the coupled approach: we have enough theory to make predictions, but not enough evidence to know if our system captures the phenomenon.

### 5.3 The Deeper Resolution

**The debate resolves when we recognize:** Theory and evidence are not sequential stages but MUTUAL CONSTRAINTS.

- Theory without evidence is speculation
- Evidence without theory is uninterpretable data
- Theory + evidence jointly determine what we can claim

For coercion specifically:
- We have ENOUGH theory (Pustejovsky) to make predictions
- We have ENOUGH infrastructure (ASA, attention extraction) to test
- We do NOT have evidence yet about what ASA captures
- We do NOT need MORE theory until we know what ASA captures

**Therefore:** Test first, but test WITH theoretical predictions. Build theory second, but only the theory that evidence demands.

---

## Conclusion

**The Right Answer:** Neither pure theory-first nor pure empirical-first. The synthesis is a **coupled minimal probe** where:

1. Minimum theory: Prediction-level commitments (coercion types differ, attention should reflect this)
2. Minimum experiment: 20 sentences, attention analysis, statistical test
3. Interpretation framework: Decision tree for what each outcome implies
4. Contingent development: Elaborate theory only if warranted by results

**This approach:**
- Respects the Theory position by grounding experiments in predictions
- Respects the Empirical position by testing before building
- Is faster than either approach alone (1 week vs 4+ or 8-10 weeks)
- Produces actionable results regardless of outcome

**The core insight:** The question "Should we build theory or run experiments?" is malformed. The right question is: "What is the minimum theory needed to make experiments interpretable, and what is the minimum experiment needed to constrain theory?"

---

*Document prepared by Research Specialist*
*Round 2 Structured Debate - January 2, 2026*
