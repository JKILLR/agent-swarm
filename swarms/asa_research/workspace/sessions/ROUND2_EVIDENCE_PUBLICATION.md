---
created: 2026-01-02 00:00
updated: 2026-01-02
---

# Round 2 Evidence Review: Publication Strategy for ASA

**Date:** January 2, 2026
**Author:** Research Specialist (Evidence Review)
**Role:** Evaluate claims about publication readiness with supporting and counter-evidence
**Format:** Evidence Review (per ROUND1_CRITIC_SYNTHESIS.md Discussion Topic 3)

---

## Executive Summary

After systematic evaluation of three claims about publication readiness, the evidence supports a **hybrid approach**: publish the empirical H6 results NOW as a sparse attention contribution, while continuing Tier 1 validation work in parallel. Neither waiting for full theoretical foundations nor waiting for all empirical validation is warranted by current evidence.

**VERDICT:** CLAIM 1 is best supported, but with modifications.

---

## CLAIM 1 EVALUATION

### "We have sufficient evidence to publish H6 + convergence results NOW"

**The Claim in Full:**
- The 73.9% H6 correlation and 21% convergence speedup are real findings
- A sparse attention paper could be submitted in 1-3 months
- The theoretical framework could come in a follow-up paper

---

### SUPPORTING EVIDENCE

**E1.1: The H6 result is robust and well-documented**
- Source: `asa_results_v2.2.md`
- Finding: 73.9% mean mass overlap (range 53.6%-86.5%)
- Per-layer breakdown: Layer 0 = 71.7%, Layer 1 = 76.1%
- Count overlap: 71.8%, Top-10 overlap: 72.3%
- Sample size: 100 samples
- Confidence: HIGH that this is a real finding, not noise

**E1.2: The convergence speedup is measurable**
- Source: `asa_results_v2.2.md`
- Finding: ASA reaches baseline PPL 26.56 in 43,000 steps vs 54,500 baseline
- Speedup: 21% fewer training steps
- Final performance: ASA 26.33 vs baseline 26.56 (0.9% improvement)
- Confidence: MEDIUM-HIGH (single experiment, but clear effect)

**E1.3: Strategic Exploration recommends this path**
- Source: `ROUND1_STRATEGIC_EXPLORATION.md` Part 7
- Quote: "Position ASA as sparse attention contribution with linguistic sparsity"
- Quote: "Lead academic conversations with H6 empirical results"
- Venue recommendation: ACL/EMNLP (empirical focus)
- Confidence: HIGH that this is the lowest-risk publication path

**E1.4: Core findings are decoupled from problematic theory**
- Source: `ROUND1_STRATEGIC_EXPLORATION.md` Section 1.2
- Sunk cost analysis shows empirical results retain HIGH value
- H6 correlation does not require predetermined embeddings claim
- Convergence result stands independently of 5-axis framework
- Confidence: HIGH that results can be published without theoretical baggage

**E1.5: Competitive landscape is favorable**
- Source: `ROUND1_STRATEGIC_EXPLORATION.md` Part 3
- ASA's linguistic sparsity approach is distinctive
- Not duplicated by Longformer, BigBird (geometric sparsity)
- Not duplicated by ERNIE, K-BERT (knowledge enhancement)
- Confidence: MEDIUM-HIGH that there is publication novelty

---

### COUNTER-EVIDENCE

**C1.1: Only 2-3 of 5 claimed axes are tested**
- Source: `ROUND1_EMPIRICAL_EXPLORATION.md` Table in Section 4.1
- Axis 1 (Type): Partial (via POS mask)
- Axis 2 (Valence): Partial (468 verbs)
- Axis 3 (Qualia): ZERO
- Axis 4 (Force Dynamics): ZERO
- Axis 5 (Geometric): Partial (WordNet hypernyms)
- Risk: Cannot claim "linguistically-derived constraints" without testing all claimed constraint types
- Confidence: HIGH that this is a real limitation

**C1.2: Scale is too small for credible claims**
- Source: `asa_results_v2.2.md` Section 5: Limitations
- Model: 6.8M parameters (tiny)
- Corpus: WikiText-2 (small)
- Risk: Effects may disappear at scale
- Confidence: MEDIUM that this undermines generalizability claims

**C1.3: No downstream task validation**
- Source: `asa_results_v2.2.md` Section 3: "What This Does Not Validate"
- Quote: "That ASA-trained models outperform baselines on downstream tasks"
- Risk: Reviewers will ask "so what?" for language modeling improvement
- Confidence: MEDIUM-HIGH that reviewers will require downstream tasks

**C1.4: VerbNet coverage is only 6.9%**
- Source: `ROUND1_EMPIRICAL_EXPLORATION.md` Section 4.1
- Coverage: 468 / 6,800 VerbNet verbs
- Implication: Most verbs have no semantic constraints
- Risk: Reviewers may question how meaningful "linguistic constraints" are with 7% coverage
- Confidence: HIGH that this needs honest acknowledgment

**C1.5: True sparse attention not implemented**
- Source: `asa_results_v2.2.md` Section 5
- Current: Dense compute + masking (still O(N^2))
- Risk: Cannot claim efficiency gains without true sparsity
- Confidence: HIGH that efficiency claims require implementation work

---

### CONFIDENCE ASSESSMENT FOR CLAIM 1

| Aspect | Confidence | Notes |
|--------|------------|-------|
| Results are real | HIGH | Well-documented, reasonable methodology |
| Results are publishable | MEDIUM-HIGH | Needs framing adjustment |
| 1-3 month timeline is realistic | MEDIUM | Depends on scope of paper |
| Follow-up paper strategy works | MEDIUM | Risk of getting scooped on theory |

**OVERALL CONFIDENCE: MEDIUM-HIGH**

The H6 and convergence results ARE publishable, but the paper must be carefully scoped to avoid claims that the evidence does not support.

---

### WHAT WOULD CHANGE THIS ASSESSMENT

**Would INCREASE confidence:**
- One downstream task showing ASA advantage (subject-verb agreement, coreference)
- Replication at 100M+ scale with similar effects
- Random mask control showing 35% overlap (proving linguistic structure matters)

**Would DECREASE confidence:**
- Discovery that H6 correlation is spurious (e.g., due to sentence length confound)
- Similar results achievable with purely syntactic constraints (no semantics needed)
- Expert feedback indicating fundamental flaw in methodology

---

## CLAIM 2 EVALUATION

### "We need Tier 1 empirical validation (Qualia coercion, VerbNet alternations, per-axis ablation) BEFORE any publication"

**The Claim in Full:**
- Current results only test 2-3 of 5 claimed axes
- Publishing prematurely invites criticism when experts review
- 5 additional weeks of validation would strengthen paper significantly

---

### SUPPORTING EVIDENCE

**E2.1: Axes 3 and 4 have zero empirical testing**
- Source: `ROUND1_EMPIRICAL_EXPLORATION.md` Section 1
- Qualia (Axis 3): 0% coverage, no experiments
- Force Dynamics (Axis 4): 0% coverage, no experiments
- Quote: "This represents a significant credibility gap"
- Confidence: HIGH that this gap exists

**E2.2: Pustejovsky will ask about qualia**
- Source: `STATE.md` Section "Anticipated Expert Questions"
- Expected question: "GL coercion requires typed feature structures with unification. Where is the unification mechanism?"
- Risk: Document already sent to Pustejovsky
- Confidence: HIGH that this question will come

**E2.3: VerbNet characterization needs correction**
- Source: `ROUND1_CRITIC_SYNTHESIS.md` Issue C2
- Problem: VerbNet presented as selectional restrictions
- Reality: Core innovation is ALTERNATION PATTERNS
- Risk: Palmer will question framing of her own work
- Confidence: HIGH that recharacterization is needed

**E2.4: Per-axis ablation would quantify contributions**
- Source: `ROUND1_EMPIRICAL_EXPLORATION.md` Section 5
- Current ablations: none/pos_only/features_only/full
- Missing: Individual axis isolation
- Benefit: Would prove each constraint type contributes
- Confidence: HIGH that this would strengthen the paper

**E2.5: Empirical Researcher timeline is 5 weeks for Tier 1**
- Source: `ROUND1_EMPIRICAL_EXPLORATION.md` Section 7
- P1: Qualia Coercion Attention = 2 weeks
- P2: VerbNet Alternation Tests = 1 week
- P3: Per-Axis Ablation = 2 weeks
- Total: 5 weeks for priority experiments
- Confidence: HIGH that this is feasible

---

### COUNTER-EVIDENCE

**C2.1: Tier 1 validation may FAIL**
- Source: `ROUND1_EMPIRICAL_EXPLORATION.md` Section 8.1 Risks
- Risk: "Coercion experiments show no ASA advantage" = Medium probability
- Risk: "Alternation tests reveal blocking errors" = Medium probability
- Implication: 5 weeks may reveal problems, not strengths
- Confidence: MEDIUM that results could be negative

**C2.2: Axes 3-4 are not implemented, not just untested**
- Source: `STATE.md` Issue C4
- Reality: Cannot run per-axis ablation on axes that do not exist
- Qualia: Not implemented
- Force Dynamics: Not implemented
- Implication: Testing requires implementation first
- Confidence: HIGH that validation requires development work

**C2.3: 5 weeks may become 10-12 weeks**
- Source: `ROUND1_EMPIRICAL_EXPLORATION.md` Section 6.3
- Full validation suite: 8-10 weeks
- Analysis and write-up: 1 week additional
- Risk: Scope creep, implementation delays
- Confidence: MEDIUM that timeline will slip

**C2.4: Experts may respond BEFORE validation is complete**
- Source: `ROUND1_CRITIC_SYNTHESIS.md` Gap 1
- Reality: Documents already sent to Pustejovsky and Palmer
- Quote: "Experts could respond any day"
- Risk: Waiting 5 weeks may not be an option
- Confidence: HIGH that timing is not entirely in our control

**C2.5: H6 paper does not require Axes 3-4 claims**
- Source: `ROUND1_STRATEGIC_EXPLORATION.md` Alternative Architectures
- Strategy: Publish sparse attention contribution only
- No claim about qualia or force dynamics
- No need to validate what you do not claim
- Confidence: HIGH that scoped paper avoids this requirement

---

### CONFIDENCE ASSESSMENT FOR CLAIM 2

| Aspect | Confidence | Notes |
|--------|------------|-------|
| Validation gap exists | HIGH | Clear documentation |
| 5 weeks would help | MEDIUM | Could reveal problems or strengths |
| 5 weeks is feasible | MEDIUM | Implementation required for Axes 3-4 |
| Publication must wait | LOW | Only if paper claims all 5 axes |

**OVERALL CONFIDENCE: MEDIUM**

The validation gap is real, but it is only blocking if the paper claims to validate all 5 axes. A scoped sparse attention paper can proceed without Axes 3-4 validation.

---

### WHAT WOULD CHANGE THIS ASSESSMENT

**Would INCREASE confidence (wait for validation):**
- Decision to publish comprehensive framework paper (not just H6)
- Expert response demanding evidence for all claims
- Discovery that Axes 1-2 results depend on Axes 3-4 somehow

**Would DECREASE confidence (proceed now):**
- Confirmation that sparse attention paper can avoid 5-axis claims entirely
- Evidence that quick experiments (1-2 weeks) can provide Tier 1 coverage
- Expert response that is positive/curious rather than critical

---

## CLAIM 3 EVALUATION

### "We need theoretical foundations resolved before empirical work is interpretable"

**The Claim in Full:**
- Without understanding HOW coercion works, we cannot interpret attention patterns
- Publishing empirical results without theory is just pattern-finding
- Category-theoretic foundations would make results more meaningful

---

### SUPPORTING EVIDENCE

**E3.1: Theory Researcher proposes principled coercion mechanism**
- Source: `ROUND1_THEORY_EXPLORATION.md` Part 1
- Proposal A: Functorial Coercion via Enriched Categories
- Quote: "Most principled" approach
- Would provide mathematical basis for interpretation
- Confidence: HIGH that this would strengthen theoretical claims

**E3.2: Current framework is "theoretically incoherent"**
- Source: `STATE.md` Issue C3
- Problem: Lambek (discrete) + Gardenfors (continuous) incompatible
- Quote: "Different ontological commitments"
- Risk: Reviewers may reject synthesis on theoretical grounds
- Confidence: HIGH that incoherence is a real problem

**E3.3: Coercion mechanism is described as CRITICAL**
- Source: `STATE.md` Issue C1
- Problem: Qualia treated as static features, not generative operators
- Missing: Type Coercion, Selective Binding, Co-composition
- Expected question: "Where is the unification mechanism?"
- Confidence: HIGH that this gap is serious

**E3.4: DisCoCat integration could unify approach**
- Source: `ROUND1_THEORY_EXPLORATION.md` Section 3.1
- DisCoCat provides: Type-driven composition, geometric semantics
- ASA could add: Qualia morphisms, force dynamics, selectional restrictions
- Confidence: MEDIUM that this synthesis is achievable

**E3.5: Theory development timeline is 4 weeks**
- Source: `ROUND1_THEORY_EXPLORATION.md` Part 6
- Priority 1: Functorial Coercion = Weeks 1-2
- Priority 2: Graded Type System = Weeks 2-3
- Priority 3: DisCoCat Integration = Week 3
- Priority 4: Mathematical Formalization = Weeks 3-4
- Total: 4 weeks for theoretical foundations
- Confidence: MEDIUM that this is achievable

---

### COUNTER-EVIDENCE

**C3.1: Empirical NLP routinely proceeds without theoretical foundations**
- Source: `ROUND1_CRITIC_SYNTHESIS.md` Assumption 3
- Quote: "BERT does not have a coherent theoretical foundation; it works anyway"
- Reality: ACL/EMNLP accept empirical contributions without theory
- Precedent: Most neural NLP papers are atheoretical
- Confidence: HIGH that theory is not required for empirical publication

**C3.2: Theory development is SLOW**
- Source: `ROUND1_CRITIC_SYNTHESIS.md` Tension 1
- Theory Researcher: 4+ weeks of development
- Strategic Researcher: 2 weeks to prepare for expert response
- Gap: 2+ weeks of misalignment
- Confidence: HIGH that theory delays publication

**C3.3: Theory may be WRONG**
- Source: `ROUND1_THEORY_EXPLORATION.md` Section 5.1 Risks
- Risk: "Category theory is too abstract for implementation" = Medium probability
- Risk: "Framework integration introduces inconsistency" = Medium probability
- Implication: 4 weeks could produce untested theory
- Confidence: MEDIUM that theory work may not pay off

**C3.4: H6 result is interpretable WITHOUT coercion theory**
- Source: `asa_results_v2.2.md` Section 6
- Conservative claim: "Substantial attention mass can be predicted by external linguistic structure"
- This interpretation requires no coercion mechanism
- This interpretation is empirically defensible
- Confidence: HIGH that empirical interpretation is sufficient

**C3.5: Strategic Researcher recommends empirical focus**
- Source: `ROUND1_STRATEGIC_EXPLORATION.md` Recommendations
- Quote: "Lead academic conversations with empirical results, not grand theory"
- Recommendation: 60% empirical, 30% characterization, 10% implementation
- Confidence: HIGH that this is the consensus direction

**C3.6: Theoretical elegance may be unnecessary**
- Source: `ROUND1_CRITIC_SYNTHESIS.md` Assumption 3
- Quote: "Maybe theoretical elegance is a luxury, not a requirement"
- Reality: Practical utility may matter more to reviewers
- Confidence: MEDIUM that reviewers care less about theory than we assume

---

### CONFIDENCE ASSESSMENT FOR CLAIM 3

| Aspect | Confidence | Notes |
|--------|------------|-------|
| Theory would strengthen results | MEDIUM-HIGH | Would provide deeper interpretation |
| Theory is required for publication | LOW | Empirical NLP precedent disagrees |
| 4 weeks is sufficient | MEDIUM | Theory work often expands |
| Results are uninterpretable without theory | LOW | Conservative interpretation available |

**OVERALL CONFIDENCE: LOW**

The theoretical work is valuable but NOT required before publication. The H6 results can be interpreted empirically without category-theoretic foundations. Theory development should proceed in parallel, not as a blocker.

---

### WHAT WOULD CHANGE THIS ASSESSMENT

**Would INCREASE confidence (wait for theory):**
- Expert response demanding theoretical justification
- Discovery that H6 patterns are only meaningful with coercion interpretation
- Requirement to submit to theoretical venue (Computational Linguistics journal)

**Would DECREASE confidence (proceed now):**
- Confirmation of ACL/EMNLP as target venue (empirical focus)
- Evidence that DisCoCat integration is months away, not weeks
- Success of conservative empirical interpretation in internal review

---

## VERDICT

### Which claim is best supported by current evidence?

**VERDICT: CLAIM 1 (with modifications)**

The evidence strongly supports publishing H6 + convergence results, but with critical modifications:

1. **Scope the paper carefully**: Do NOT claim to validate all 5 axes. Claim linguistic sparsity based on POS + VerbNet + WordNet (Axes 1, 2, 5 only).

2. **Acknowledge limitations explicitly**: State that Qualia and Force Dynamics are future work, not current claims.

3. **Add one downstream validation**: Subject-verb agreement or coreference resolution would significantly strengthen the paper.

4. **Run random mask control**: This is critical to proving linguistic structure matters (not just sparsity).

**Why not Claim 2?**
- Waiting 5 weeks only matters if paper claims all 5 axes
- Axes 3-4 require IMPLEMENTATION, not just testing
- Scoped paper avoids this requirement entirely

**Why not Claim 3?**
- Theory is valuable but not required
- Empirical NLP precedent does not require theory
- H6 result has valid empirical interpretation without coercion mechanism
- Theory can be follow-up paper

---

## RECOMMENDED PUBLICATION PATH

### Paper Outline

**Title:** "Linguistic Structure Predicts Transformer Attention: Sparse Attention via Semantic Constraints"

*Alternative:* "From Linguistic Theory to Efficient Attention: Validating Predetermined Sparsity Patterns"

### What to Include

**INCLUDE:**
1. **H6 Correlation Finding** (Section 3 of paper)
   - 73.9% attention mass overlap with linguistic constraints
   - Per-layer breakdown
   - Comparison to random baseline (NEED TO RUN)

2. **Convergence Speedup** (Section 4)
   - 21% fewer training steps
   - Learning curve comparison

3. **Methodology** (Section 2)
   - POS compatibility matrix
   - VerbNet selectional restrictions (468 verbs)
   - WordNet hypernym constraints
   - Explicit coverage disclosure (6.9% verb coverage)

4. **One Downstream Task** (Section 5) [REQUIRES NEW EXPERIMENT]
   - Subject-verb agreement OR
   - Coreference resolution OR
   - Semantic Role Labeling

5. **Limitations Section** (Section 6)
   - Scale: 6.8M parameters only
   - Coverage: 6.9% VerbNet, partial WordNet
   - True sparsity: Not yet implemented
   - Languages: English only

### What to Exclude

**EXCLUDE (defer to follow-up):**
1. "Semantic Periodic Table" framing
2. Predetermined embeddings claims
3. Qualia structure / coercion mechanism
4. Force Dynamics (Axis 4)
5. 5-axis framework as primary contribution
6. Category-theoretic foundations
7. Any claims about generative operators

### Target Venue

**Primary:** ACL 2026 or EMNLP 2026
- Empirical focus
- Sparse attention is timely
- 8-page limit forces discipline

**Secondary:** NeurIPS 2026 (if scale validation achieved)

**Avoid:**
- Computational Linguistics journal (requires theory)
- CogSci (requires theoretical framework)

### Timeline

| Week | Activity | Deliverable |
|------|----------|-------------|
| 1 | Random mask control experiment | Baseline comparison data |
| 2 | Downstream task experiment | SRL or S-V agreement results |
| 3-4 | Paper draft | Full manuscript |
| 5 | Internal review | Revised manuscript |
| 6 | Submission preparation | Camera-ready |

**Target:** ACL 2026 submission (deadline typically February)

### Key Risks

| Risk | Probability | Mitigation |
|------|-------------|------------|
| Random mask shows similar overlap | LOW | Would require major reframing |
| Downstream task shows no advantage | MEDIUM | Emphasize correlation over performance |
| Reviewer asks about Axes 3-4 | MEDIUM | Explicit future work statement |
| Expert responds critically before submission | MEDIUM | Prepare damage control response |
| Scale validation fails | MEDIUM | Acknowledge as limitation |

---

## DAMAGE CONTROL PROTOCOL

### If Experts (Pustejovsky/Palmer) Respond Critically Before We Are Ready

#### Draft Response Template

```
Dear Professor [Pustejovsky/Palmer],

Thank you for your thoughtful response to our materials. Your feedback
is extremely valuable and highlights important areas where our
presentation needs refinement.

[ACKNOWLEDGE SPECIFIC CRITICISM]
You are correct that [specific issue]. This was an overreach in our
framing that we are actively correcting.

[REFRAME CONTRIBUTION]
Our core empirical finding is more modest than our theoretical framing
suggested: we observe that ~74% of learned attention in baseline
transformers aligns with linguistically-derived compatibility constraints.
This suggests that linguistic structure predicts a substantial fraction
of what transformers learn, with implications for efficient attention.

[CLARIFY CLAIMS]
To be clear about what we are and are not claiming:
- We ARE claiming: Linguistic constraints predict attention patterns
- We are NOT claiming: A complete theory of semantic composition
- We ARE claiming: 21% training speedup with predetermined structure
- We are NOT claiming: All 5 axes are empirically validated

[EXPRESS INTEREST IN GUIDANCE]
We would greatly value your guidance on [specific question]. Your
expertise in [qualia structure / verb alternations] would help us
avoid further mischaracterizations.

[OFFER COLLABORATION]
If you have interest in collaborating on a more rigorous treatment
of these questions, we would be honored to work with you.

Best regards,
[Team]
```

#### Claims That CAN Be Walked Back

| Claim | Walk-Back Statement |
|-------|---------------------|
| Qualia as 4D feature vector | "We oversimplified GL; qualia are generative operators, not static features. Our current approach does not implement true coercion." |
| VerbNet for selectional restrictions | "We acknowledge VerbNet's core innovation is alternation patterns. Our current implementation uses only a subset of VerbNet's capabilities." |
| 5-axis synthesis | "The 5-axis framework is a research hypothesis, not a validated theory. Axes 3-4 lack empirical testing." |
| Semantic Periodic Table | "The periodic table analogy was overreaching. We are pursuing a more modest claim about linguistic sparsity." |
| Framework integration | "We have not reconciled the theoretical foundations of Lambek and Gardenfors. This remains an open problem." |

#### Claims That MUST Be Defended

| Claim | Defense | Evidence |
|-------|---------|----------|
| H6 correlation is real | This is empirical data with clear methodology | 73.9% mean, 100 samples, per-layer breakdown |
| Convergence speedup is real | Measurable training curve difference | 43K vs 54.5K steps to target PPL |
| Linguistic constraints capture attention | Comparison to random baseline (NEED THIS) | [Will show random = ~35%] |
| Core approach is novel | Linguistic sparsity vs geometric sparsity | Literature review shows no direct precedent |

#### Communication Owner

**Recommendation:** Assign ONE person to handle expert communication to ensure consistent messaging.

**Responsibilities:**
- All email responses to Pustejovsky/Palmer
- Framing decisions for acknowledgments
- Coordination with research team before responding
- Documentation of all exchanges

**Timeline:** Respond within 48 hours of any expert message, even if just to acknowledge receipt.

---

## SUMMARY TABLE

| Claim | Support | Counter | Confidence | Recommendation |
|-------|---------|---------|------------|----------------|
| 1: Publish H6 NOW | Strong empirical results, clear novelty | Scale, coverage, no downstream task | MEDIUM-HIGH | PROCEED with modifications |
| 2: Need Tier 1 validation first | Real gaps in Axes 3-4 | Scoped paper avoids this | MEDIUM | DO NOT WAIT if paper is scoped |
| 3: Need theory first | Would strengthen interpretation | Not required for empirical publication | LOW | DEVELOP IN PARALLEL |

**Final Recommendation:** Pursue Claim 1 strategy with two quick experiments (random mask control, one downstream task) and careful scoping. Begin paper draft in Week 1.

---

## PROGRESS LOG UPDATE FOR STATE.md

```
### 2026-01-02 RESEARCHER - Round 2 Evidence Review: Publication Strategy

**Context:** Evidence review of three claims about publication readiness

**Work Completed:**
- Evaluated CLAIM 1 (publish now) with 5 supporting, 5 counter-evidence points
- Evaluated CLAIM 2 (wait for Tier 1 validation) with 5 supporting, 5 counter points
- Evaluated CLAIM 3 (wait for theory) with 5 supporting, 6 counter points
- Produced verdict: CLAIM 1 best supported with modifications
- Defined complete publication path with timeline
- Created damage control protocol with response template

**Files Created:**
- `ROUND2_EVIDENCE_PUBLICATION.md` - Full evidence review with verdict and recommendations

**KEY FINDINGS:**

**Verdict:** CLAIM 1 (publish H6 + convergence NOW) is best supported, but requires:
1. Scope paper to avoid 5-axis claims
2. Add random mask control experiment
3. Add one downstream task validation
4. Explicit acknowledgment of limitations

**Recommended Publication Path:**
- Title: "Linguistic Structure Predicts Transformer Attention"
- Target: ACL/EMNLP 2026
- Timeline: 6 weeks to submission
- Exclude: Semantic Periodic Table, predetermined embeddings, Axes 3-4 claims

**Damage Control Protocol:**
- Response template drafted for expert criticism
- Identified walk-back claims vs. must-defend claims
- Recommended single communication owner

**Outcome:** success - Publication strategy defined with evidence-based justification
```

---

*Document prepared by Research Specialist*
*Round 2 Evidence Review - January 2, 2026*
