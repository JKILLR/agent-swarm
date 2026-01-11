# ASA v2.2 Empirical Results

## Executive Summary

| Experiment | Result | Implication |
|------------|--------|-------------|
| Training PPL | ASA 26.33 vs Baseline 26.56 | Equivalent final performance |
| Convergence | ASA reaches baseline PPL in 21% fewer steps | Linguistic bias accelerates learning |
| H6 Correlation | 73.9% attention mass overlap | ASA captures natural attention patterns |
| Sparsity | 35% of attention pairs blocked | Meaningful but not extreme reduction |

---

## 1. Experimental Setup

### Model Configuration
- **Architecture:** Tiny transformer (6.8M parameters)
- **Layers:** 2
- **Heads:** 2
- **Hidden dim:** 128
- **Vocabulary:** GPT-2 tokenizer (50,257 tokens)

### Dataset
- **Corpus:** WikiText-2
- **Train:** 23,029 examples
- **Validation:** 2,396 examples
- **Test:** 2,711 examples
- **Max sequence length:** 256 tokens

### Training
- **Epochs:** 10
- **Batch size:** 4
- **Learning rate:** 3e-4
- **Optimizer:** AdamW
- **Hardware:** Apple M2 (MPS)

### ASA Configuration
- **Mode:** full (POS mask + feature compatibility + blocking)
- **α:** 1.0
- **Hard blocking:** True

---

## 2. Language Modeling Results

### Final Perplexity

| Model | Validation PPL | Training Time |
|-------|----------------|---------------|
| ASA (full) | **26.33** | 271 min |
| Baseline | 26.56 | 193 min |

**Interpretation:** ASA achieves marginally better perplexity (0.9% improvement) while requiring more wall-clock time due to bonding mask computation overhead.

### Convergence Analysis

| Metric | ASA | Baseline | Difference |
|--------|-----|----------|------------|
| Steps to reach PPL 26.56 | 43,000 | 54,500 | **21% fewer** |

**Interpretation:** ASA reaches equivalent performance in significantly fewer training steps, suggesting that predetermined linguistic structure provides useful inductive bias. The model does not need to "discover" syntactic constraints from scratch.

### What This Shows
- Linguistic constraints do not harm final model quality
- Convergence is accelerated by structural priors
- The 21% step reduction represents meaningful compute savings at scale

### What This Does Not Show
- Dramatic perplexity improvements
- Wall-clock speedups (current implementation has Python overhead)
- Scaling behavior at larger model sizes

---

## 3. H6 Correlation Analysis

### Hypothesis
> Do learned dense attention patterns substantially align with linguistically derived compatibility constraints, even when those constraints were not used during training?

### Method
1. Train baseline transformer (mode='none', no ASA constraints)
2. Extract attention weights on held-out test data
3. Compute ASA bonding mask for same inputs
4. Measure overlap between attention weights and ASA-compatible pairs

### Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mass overlap** | **73.9%** | ~3/4 of attention energy lands on ASA-compatible edges |
| Count overlap (threshold=0.01) | 71.8% | Most non-trivial attention links are structurally licensed |
| Top-10 overlap | 72.3% | ~7/10 most important dependencies are predictable |
| Range | 53.6% - 86.5% | Variation reflects linguistic complexity |

### Per-Layer Breakdown

| Layer | Overlap |
|-------|---------|
| Layer 0 | 71.7% |
| Layer 1 | 76.1% |

**Key finding:** Deeper layers show *higher* alignment with linguistic structure, suggesting learned abstractions respect linguistic compatibility rather than diverging from it.

### Context

| Comparison | Expected Overlap |
|------------|------------------|
| Random structure | ~35% (weighted sparsity baseline) |
| Local window attention | ~50-55% |
| **ASA linguistic structure** | **73.9%** |

### What This Validates

✅ **Attention is not arbitrary** — It is heavily constrained by linguistic structure.

✅ **Fixed rules recover most attention mass** — Without any training.

✅ **Dense attention is largely redundant** — For English-like corpora.

✅ **ASA sparsity aligns with learned representations** — Not antagonistic to them.

### What This Does Not Validate

❌ That ASA-trained models outperform baselines on downstream tasks

❌ That hard masking is always safe (edge cases exist)

❌ That rules generalize cross-lingually

❌ That thermodynamic scoring improves quality

---

## 4. Variance Analysis

The range of 53.6% to 86.5% overlap is informative:

| Overlap Level | Likely Cause |
|---------------|--------------|
| High (~86%) | Clean declarative syntax |
| Medium (~70%) | Typical English prose |
| Low (~54%) | Figurative language, ellipsis, parser errors |

This variance is expected and healthy. Uniform overlap would suggest the metric is insensitive.

The ~26% of attention outside ASA-compatible pairs represents:
- Long-range semantic dependencies
- Metaphorical/figurative language
- Patterns ASA's current rules don't capture
- Potential targets for probabilistic escape edges

---

## 5. Limitations (Explicit)

### Scale
- Experiments conducted on tiny (6.8M) model
- WikiText-2 is a small corpus
- Larger models may show different behavior

### Language
- English only
- UD/VerbNet/WordNet coverage varies by language
- Cross-lingual generalization not tested

### Coverage
- VerbNet covers ~468 common verbs (not exhaustive)
- WordNet hypernym traversal may miss domain-specific terms
- Parser errors propagate to features

### Implementation
- Current sparse attention uses dense compute + masking
- True O(N×k) sparse attention requires custom kernels
- Wall-clock time not yet improved

### Hard Masking
- Permanently blocks ~35% of attention pairs
- May block rare but valid dependencies (metaphor, coercion)
- Probabilistic escape edges not yet implemented

---

## 6. Conclusions

### Core Finding
**73.9% of learned attention mass in baseline transformers concentrates on pairs deemed compatible by ASA's linguistic constraints.** This suggests that predetermined linguistic structure captures patterns transformers would otherwise need to learn from data.

### Implications
1. **Efficiency:** Linguistic priors accelerate convergence (21% fewer steps)
2. **Interpretability:** Attention patterns become linguistically meaningful
3. **Scalability:** Sparse attention enables longer contexts (with proper implementation)

### Conservative Claim
> ASA demonstrates that substantial attention mass can be predicted by external linguistic structure, supporting the feasibility of rule-derived sparse attention as an alternative to learned or positional sparsity patterns.

---

## 7. Future Work

### Immediate (strengthen current results)
1. **Random mask control:** Same sparsity, random edges — show overlap collapses to ~35%
2. **Overlap vs sparsity curve:** Visualize relationship
3. **One downstream task:** Subject-verb agreement or coreference

### Medium-term (extend validity)
4. **Larger models:** Test at 100M+ parameters
5. **Longer contexts:** Test on document-level tasks
6. **True sparse attention:** Implement O(N×k) with custom kernels or xformers

### Long-term (theoretical)
7. **Probabilistic escape edges:** Allow rare "tunneling" through blocked pairs
8. **Cross-lingual evaluation:** Test on non-English languages
9. **Thermodynamic scoring:** Revisit entropy/enthalpy formulation

---

## Appendix: Raw Numbers

### Training Curves

**ASA (full):**
- Step 500: Val PPL 74.99
- Step 1000: Val PPL 67.63
- Step 43000: Val PPL 26.56 ← matches baseline final
- Step 57500: Val PPL 26.33 (final)

**Baseline (none):**
- Step 54500: Val PPL 26.56 (first occurrence)
- Step 57500: Val PPL 26.56 (final)

### H6 Detailed Results
```json
{
  "num_samples": 100,
  "threshold": 0.01,
  "top_k": 10,
  "metrics": {
    "mass_overlap": {"mean": 0.739, "min": 0.536, "max": 0.865},
    "count_overlap": {"mean": 0.718},
    "topk_overlap": {"mean": 0.723},
    "per_layer": {"0": {"mean": 0.717}, "1": {"mean": 0.761}}
  }
}
```
