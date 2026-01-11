# ASA (Atomic Semantic Architecture) - Core Concepts

## The Core Insight
"The relationship IS the position in vector space."

ASA treats concepts like atoms:
- **Inner shells**: The concept itself (core meaning)
- **Outer shells**: Context and relationships
- **Physics-based bonding rules**: How concepts combine

## Why This Matters
Could differentiate MYND by solving semantic representation at a deeper level than competitors.

## Five-Axis Constraint Framework

### Axis 1: Ontological Type (Type-Logical Grammar)
- Lambek calculus foundation
- Tracks syntactic categories that predict semantic roles

### Axis 2: Valence Structure (VerbNet/PropBank)
- Argument structure constraints
- Based on Levin verb classes and syntactic alternations
- **Critical**: VerbNet's core is ALTERNATION PATTERNS, not selectional restrictions

### Axis 3: Qualia Structure (Generative Lexicon)
- Four qualia: FORMAL, CONSTITUTIVE, TELIC, AGENTIVE
- **Critical**: Qualia are GENERATIVE OPERATORS, not static features
- Enables type coercion (e.g., "begin the book" → reading event)

### Axis 4: Force Dynamics (Talmy)
- Agonist/Antagonist force interactions
- Causation, prevention, enablement patterns
- **Status**: Zero empirical validation

### Axis 5: Geometric Position (Conceptual Spaces)
- Gärdenfors conceptual spaces
- Hyperbolic geometry for hierarchy
- **Status**: Partial validation via WordNet hypernyms

## Key Empirical Results (v2.2)
- **H6 Result**: 73.9% attention overlap with trained attention (vs 47% random)
- **Sparsity**: 31% reduction while maintaining performance
- **Convergence**: 21% faster convergence

## Known Issues (Must Address)

### Critical Gaps
1. **Qualia Mechanism**: Treated as static 4D vector, should be generative operators
2. **VerbNet Framing**: Presented as selectional restrictions, core is alternations
3. **Framework Integration**: Discrete (Lambek) vs continuous (Gärdenfors) not reconciled
4. **Axes 3-4**: Zero empirical validation

### Coverage Reality
- VerbNet: 468/6,800 verbs = 6.9% coverage
- Qualia annotations: 0%
- Force Dynamics: 0%

## Connection to Lottery Ticket Hypothesis
- LTH: Sparse "winning" subnetworks exist within overparameterized networks
- ASA: ~74% of attention patterns are predictable from linguistic structure
- Both claim: Sparsity patterns matter more than raw parameter count
- **Key**: ASA could identify winning tickets WITHOUT iterative pruning (O(1) vs O(n))

## Academic Contacts
- **James Pustejovsky** (Brandeis): Generative Lexicon expert - Axis 3
- **Martha Palmer** (CU Boulder): VerbNet creator - Axis 2
- **Status**: Emails sent seeking feedback on theoretical foundations
