---
created: 2026-01-02 00:00
updated: 2026-01-02
---

# Next Direction Recommendation
## Executive Synthesis and Strategic Priorities

**Date:** January 2, 2026
**Author:** Research Team (Final Synthesis)
**Status:** ACTIONABLE — Immediate priorities identified

---

## Executive Summary

Three independent analyses (Researcher, Architect, Critic) have reviewed the theoretical framework sent to James Pustejovsky and Martha Palmer. The consensus is clear: **the framework has genuine strengths but contains critical vulnerabilities that experts will immediately identify**.

The good news: These vulnerabilities are addressable before feedback arrives. The framework's core insight — that semantic constraints from independent research traditions converge on similar structures — remains valid. The problems lie in how specific theories are characterized and operationalized.

**Bottom Line:** Shift from theoretical expansion to empirical grounding and precise characterization. The most damaging criticism will be that the framework misrepresents the theories it synthesizes.

---

## Part 1: Key Strengths

| Strength | Evidence | Why It Matters |
|----------|----------|----------------|
| **Appropriate epistemic humility** | "Hypotheses to test, not defend" framing | Invites collaboration rather than triggering defensive response |
| **Convergent evidence** | 12+ independent frameworks pointing to similar structures | This is genuinely compelling if characterizations are accurate |
| **Preliminary empirical validation** | 74% attention on compatible pairs, 31% sparsity, 21% faster convergence | Shows the approach captures something real |
| **Clear scope limitations** | Explicitly excludes pragmatics, discourse, social meaning | Demonstrates theoretical sophistication |
| **Mathematical grounding** | Type-logical grammar via Lambek calculus | Provides formal foundation others can engage with |

---

## Part 2: Critical Gaps (Must Address)

### Gap 1: Qualia Mischaracterization [SEVERITY: CRITICAL]

**The Problem:** Documents treat qualia as a "4D feature vector" — static coordinates along Formal, Constitutive, Telic, Agentive dimensions.

**What Pustejovsky Actually Means:** Qualia are GENERATIVE OPERATORS that enable:
- **Type Coercion** — "begin the book" coerces book to its Telic quale (reading)
- **Selective Binding** — "fast car" vs "fast food" access different qualia dynamically
- **Co-composition** — verb and noun mutually constrain interpretation

**What Is Missing:**
- Dot objects (complex types): "newspaper" = organization * physical * information
- Coercion MECHANISM — how does the system decide which quale to access?
- Dynamic selection based on context

**Expert Risk:** Pustejovsky will immediately see this as a fundamental misunderstanding of his 30+ years of work.

---

### Gap 2: VerbNet Framing Error [SEVERITY: CRITICAL]

**The Problem:** VerbNet is presented as providing selectional restrictions (semantic features required by arguments).

**What Palmer Actually Created:** VerbNet's core innovation is SYNTACTIC ALTERNATION PATTERNS (from Levin 1993):
- Spray/load alternation: "spray paint on wall" vs "spray wall with paint"
- Causative/inchoative: "John broke the vase" vs "The vase broke"
- Classes defined by SHARED ALTERNATION BEHAVIOR, not semantic features

**What Is Missing:**
- Alternation patterns as defining characteristic
- Subclass hierarchy (329 main classes with extensive subclasses)
- Version specificity (VerbNet has evolved significantly)
- Acknowledgment that selectional restrictions in VerbNet are SPARSE

**Expert Risk:** Palmer will question why her own work is framed incorrectly.

---

### Gap 3: Framework Integration Incoherent [SEVERITY: HIGH]

**The Problem:** The synthesis stacks incompatible theoretical frameworks:
- **Type-logical grammar** (discrete categories, symbolic)
- **Conceptual Spaces** (continuous geometry, graded)
- **Lambek calculus** (formal/logical)
- **Talmy's Force Dynamics** (embodied/image-schematic)

These have different ontological commitments. You cannot simply add continuous Gardenfors coordinates to discrete Lambek types without a principled integration theory.

**Expert Risk:** Both academics will ask how incompatible foundations are reconciled.

---

### Gap 4: Zero Empirical Validation for Axes 3 and 4 [SEVERITY: HIGH]

**The Problem:** Current ASA experiments use only:
- POS tags (Axis 1)
- VerbNet restrictions (Axis 2, partial)
- WordNet hypernyms (Axis 5, partial)

**Reality:** Qualia (Axis 3) and Force Dynamics (Axis 4) have NO empirical validation in ASA. The 74% attention correlation does not test these axes at all.

**Expert Risk:** Framework claims 5 axes but only tests 2-3.

---

### Gap 5: Thematic Role Claim Factually Incorrect [SEVERITY: MEDIUM-HIGH]

**The Problem:** Documents claim "8-12 roles regardless of framework."

**Reality:**
- PropBank uses numbered arguments (Arg0-Arg5), NOT thematic roles
- FrameNet uses frame-specific roles, NOT universal inventory
- VerbNet and PropBank have systematic mismatches documented in SemLink

**Expert Risk:** Palmer co-developed PropBank. She will notice this immediately.

---

## Part 3: Top 3 Priority Actions

### PRIORITY 1: Develop Coercion Mechanism Proposal
**Timeline:** 1-2 weeks
**Rationale:** This is the SINGLE BIGGEST gap. Pustejovsky will ask "How does coercion work?" and there is currently no answer.

**What to Do:**
1. Review Pustejovsky (1995) Chapter 6 on coercion operations
2. Propose explicit mechanism for how predetermined embeddings could support:
   - Type coercion (eventive complement requirement triggers quale access)
   - Selective binding (adjective-noun combination selects quale)
   - Co-composition (mutual constraint between verb and argument)
3. Build small demonstrator on canonical examples:
   - "begin the book" (Telic)
   - "finish the book" (Agentive for author, Telic for reader)
   - "fast car" vs "fast food" vs "fast driver"
4. EXPLICITLY ACKNOWLEDGE that a 4D vector is a lossy approximation of full qualia structure, and characterize what is lost

**Success Criterion:** Can answer "Where is the unification mechanism in your vector representation?" with a concrete proposal.

---

### PRIORITY 2: Correct VerbNet Characterization
**Timeline:** 1 week
**Rationale:** Cannot present Palmer's own work incorrectly to her.

**What to Do:**
1. Reframe VerbNet contribution around ALTERNATION PATTERNS as core innovation
2. Acknowledge that selectional restrictions are sparse and often absent in VerbNet
3. Specify VerbNet version being used (3.3 is current)
4. Address subclass hierarchy — which level is being used?
5. Prepare demonstration on Palmer's canonical examples:
   - Spray/load alternation
   - Causative/inchoative alternation
   - Verbs appearing in multiple classes

**Success Criterion:** Can explain how framework handles alternation patterns, not just selectional features.

---

### PRIORITY 3: Run Targeted Empirical Validation
**Timeline:** 2-3 weeks
**Rationale:** Claims about 5 axes must be backed by tests on all 5 axes.

**What to Do:**
1. **Qualia Coercion Test:**
   - Feed Pustejovsky's canonical coercion sentences to ASA
   - Measure whether attention patterns differ by coercion type
   - Compare to baseline transformer

2. **VerbNet Alternation Test:**
   - Test whether ASA handles alternation pairs appropriately
   - "John sprayed paint on the wall" vs "John sprayed the wall with paint"
   - Both should be well-formed; attention patterns may differ

3. **Coverage Analysis:**
   - Actual percentage of vocabulary assignable to predetermined features
   - Current estimate: VerbNet covers 6,800 verbs; LLM vocabulary is 100,000+ (93% uncovered)

4. **Ablation by Axis:**
   - Remove each constraint type and measure degradation
   - This reveals which axes actually contribute

**Success Criterion:** Can report empirical results for each of the 5 axes, even if preliminary.

---

## Part 4: Recommended Research Direction

### Primary Focus: EMPIRICAL VALIDATION (60%)

**Rationale:** The theoretical synthesis is already expansive. What is missing is evidence that the framework captures what it claims. More theory without validation will not convince academics.

**Specific Focus Areas:**
- Coercion behavior in attention patterns
- Alternation handling
- Coverage estimation
- Per-axis ablation

### Secondary Focus: PRECISE CHARACTERIZATION (30%)

**Rationale:** The framework mischaracterizes key theories. This must be corrected before academics read it closely.

**Specific Focus Areas:**
- Qualia as generative operators (not static features)
- VerbNet as alternation classes (not selectional restrictions)
- Framework integration theory (how do discrete and continuous combine?)

### Tertiary Focus: PROTOTYPE IMPLEMENTATION (10%)

**Rationale:** A 100-word "semantic periodic table" demonstrator would make abstract claims concrete. But this should not consume resources until characterization issues are resolved.

**Defer:**
- Large-scale VerbNet expansion (6,800 verbs)
- Hyperbolic geometry exploration
- Cross-linguistic validation
- Force dynamics formalization

---

## Part 5: Timeline Estimate

| Priority | Action | Duration | Deliverable |
|----------|--------|----------|-------------|
| **P1** | Coercion mechanism proposal | Week 1-2 | Document + small demo |
| **P2** | VerbNet recharacterization | Week 1 | Revised framing document |
| **P3a** | Qualia coercion experiment | Week 2-3 | Attention analysis report |
| **P3b** | Alternation handling test | Week 2-3 | Test results |
| **P3c** | Coverage analysis | Week 2 | Coverage statistics |
| **P3d** | Per-axis ablation | Week 3-4 | Ablation report |

**Decision Point (Week 4):**
- If empirical results support framework: Prepare collaboration materials
- If empirical results challenge framework: Revise theory before expert feedback

---

## Part 6: Risks and Mitigations

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Coercion mechanism proves intractable | Medium | Explicitly acknowledge as open problem; propose hybrid approach |
| Experts respond before corrections complete | Medium | Prepare honest acknowledgment of gaps identified since submission |
| Empirical tests fail to support framework | Medium | Treat as valuable negative result; revise theory accordingly |
| Coverage too low for practical use | High | Propose learned fallback for uncovered vocabulary |

---

## Part 7: What NOT to Do

1. **Do not expand theoretical claims** — The framework is already ambitious. More synthesis without validation weakens credibility.

2. **Do not build large-scale implementations** — Until characterization issues are resolved, engineering effort is premature.

3. **Do not defend the framework as-is** — The documents explicitly invite feedback. Acknowledge gaps proactively.

4. **Do not wait for expert response** — Use this time productively. Proactive improvement demonstrates research maturity.

---

## Conclusion

The semantic constraints framework has a compelling core insight: independent research traditions converge on similar constraint structures. This convergence is real and worth investigating.

However, the current presentation mischaracterizes key theories (Qualia, VerbNet), lacks empirical validation for claimed axes, and does not explain HOW constraints operate (especially coercion).

**The path forward is clear:**
1. Fix the characterization errors (Qualia, VerbNet)
2. Develop a concrete coercion mechanism proposal
3. Run targeted empirical tests on all claimed axes
4. Acknowledge gaps honestly when experts respond

This is not a setback — it is the normal process of research refinement. The framework's epistemic humility ("hypotheses to test, not defend") is appropriate. Now demonstrate that humility through rigorous self-correction before external critique arrives.

---

*Document synthesizes findings from Researcher, Architect, and Critic analyses.*
*Prepared January 2, 2026 for ASA Research Swarm*
