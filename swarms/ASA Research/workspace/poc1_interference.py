#!/usr/bin/env python3
"""
POC-1: Complex Embedding Interference Test
==========================================

Goal: Prove phases in complex embeddings DO something functionally.

Method:
1. Generate complex embeddings z = r·e^(iθ) for word pairs
2. Compute interference: I = |z₁ + z₂|² = r₁² + r₂² + 2r₁r₂cos(θ₁-θ₂)
3. Train: interference pattern → semantic similarity (ground truth: SimLex-999)
4. Compare: Does interference term (2r₁r₂cos) add predictive value over magnitude-only baseline?

Null Hypothesis: cos(θ₁-θ₂) term contributes ≤0 additional R² over magnitude baseline.
Pass Condition: p < 0.01 for phase contribution.

Usage: python poc1_interference.py
"""

import os
import urllib.request
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

SIMLEX_URL = "https://fh295.github.io/SimLex-999.zip"
SIMLEX_FILE = "SimLex-999/SimLex-999.txt"
DATA_DIR = Path(__file__).parent / "data"
EMBEDDING_DIM = 50
RANDOM_SEED = 42
P_THRESHOLD = 0.01  # Statistical significance threshold

# ============================================================================
# Data Loading
# ============================================================================

def download_simlex():
    """Download SimLex-999 dataset if not present."""
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / "SimLex-999.zip"
    txt_path = DATA_DIR / "SimLex-999.txt"

    if txt_path.exists():
        return txt_path

    print("Downloading SimLex-999 dataset...")
    try:
        urllib.request.urlretrieve(SIMLEX_URL, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Extract the txt file
            for name in z.namelist():
                if name.endswith('.txt') and 'SimLex' in name:
                    content = z.read(name)
                    with open(txt_path, 'wb') as f:
                        f.write(content)
                    break
        zip_path.unlink()  # Clean up zip
        print(f"Downloaded to {txt_path}")
        return txt_path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Using synthetic data instead...")
        return None


def load_simlex(path):
    """Load SimLex-999 word pairs and similarity scores."""
    pairs = []
    scores = []
    with open(path, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                word1, word2 = parts[0], parts[1]
                sim_score = float(parts[3])  # SimLex999 score (0-10)
                pairs.append((word1, word2))
                scores.append(sim_score / 10.0)  # Normalize to 0-1
    return pairs, np.array(scores)


def generate_synthetic_data(n_pairs=500):
    """Generate synthetic word pairs with known similarity structure."""
    print("Generating synthetic evaluation data...")
    np.random.seed(RANDOM_SEED)

    # Create synthetic "words" as indices
    pairs = [(f"word_{i}_a", f"word_{i}_b") for i in range(n_pairs)]

    # Generate similarity scores with structure:
    # Some pairs are highly similar, some are not
    scores = np.random.beta(2, 5, n_pairs)  # Skewed towards lower similarity
    scores[:n_pairs//4] = np.random.beta(5, 2, n_pairs//4)  # Some high similarity pairs

    return pairs, scores

# ============================================================================
# Complex Embedding Generation
# ============================================================================

def create_word_vocabulary(pairs):
    """Create vocabulary from word pairs."""
    vocab = set()
    for w1, w2 in pairs:
        vocab.add(w1)
        vocab.add(w2)
    return {word: idx for idx, word in enumerate(sorted(vocab))}


def initialize_complex_embeddings(vocab_size, dim, seed=RANDOM_SEED):
    """
    Initialize complex embeddings as z = r·e^(iθ).

    Returns:
        magnitudes: (vocab_size, dim) - r values
        phases: (vocab_size, dim) - θ values in radians
    """
    np.random.seed(seed)

    # Initialize magnitudes from uniform distribution
    magnitudes = np.random.uniform(0.5, 1.5, (vocab_size, dim))

    # Initialize phases uniformly in [0, 2π)
    phases = np.random.uniform(0, 2 * np.pi, (vocab_size, dim))

    return magnitudes, phases


def get_complex_embedding(word_idx, magnitudes, phases):
    """Get complex embedding z = r·e^(iθ) for a word."""
    r = magnitudes[word_idx]
    theta = phases[word_idx]
    return r * np.exp(1j * theta)

# ============================================================================
# Interference Computation
# ============================================================================

def compute_interference_features(z1, z2):
    """
    Compute interference between two complex embeddings.

    For z₁ = r₁·e^(iθ₁) and z₂ = r₂·e^(iθ₂):
    I = |z₁ + z₂|² = r₁² + r₂² + 2r₁r₂cos(θ₁-θ₂)

    Returns:
        magnitude_term: r₁² + r₂² (baseline - no phase)
        interference_term: 2r₁r₂cos(θ₁-θ₂) (phase contribution)
        full_interference: |z₁ + z₂|² (complete)
    """
    r1 = np.abs(z1)
    r2 = np.abs(z2)
    theta1 = np.angle(z1)
    theta2 = np.angle(z2)

    # Magnitude-only baseline (sum across dimensions)
    magnitude_term = np.sum(r1**2 + r2**2)

    # Phase-dependent interference term
    interference_term = np.sum(2 * r1 * r2 * np.cos(theta1 - theta2))

    # Full interference
    full_interference = magnitude_term + interference_term

    return magnitude_term, interference_term, full_interference


def compute_all_interference_features(pairs, vocab, magnitudes, phases):
    """Compute interference features for all word pairs."""
    magnitude_features = []
    interference_features = []
    full_features = []

    for w1, w2 in pairs:
        if w1 not in vocab or w2 not in vocab:
            # Handle OOV with zero features
            magnitude_features.append(0)
            interference_features.append(0)
            full_features.append(0)
            continue

        idx1, idx2 = vocab[w1], vocab[w2]
        z1 = get_complex_embedding(idx1, magnitudes, phases)
        z2 = get_complex_embedding(idx2, magnitudes, phases)

        mag, interf, full = compute_interference_features(z1, z2)
        magnitude_features.append(mag)
        interference_features.append(interf)
        full_features.append(full)

    return (np.array(magnitude_features),
            np.array(interference_features),
            np.array(full_features))

# ============================================================================
# Training: Learn embeddings that predict similarity
# ============================================================================

def train_embeddings(pairs, scores, vocab, dim=EMBEDDING_DIM, n_epochs=100, lr=0.01):
    """
    Train complex embeddings to predict semantic similarity via interference.

    Objective: minimize MSE between interference-based prediction and ground truth similarity.
    """
    vocab_size = len(vocab)
    np.random.seed(RANDOM_SEED)

    # Initialize embeddings
    magnitudes = np.random.uniform(0.5, 1.5, (vocab_size, dim))
    phases = np.random.uniform(0, 2 * np.pi, (vocab_size, dim))

    # Get pair indices
    pair_indices = []
    valid_mask = []
    for w1, w2 in pairs:
        if w1 in vocab and w2 in vocab:
            pair_indices.append((vocab[w1], vocab[w2]))
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    valid_scores = scores[valid_mask]

    print(f"Training on {len(pair_indices)} valid pairs...")

    def compute_predictions(mags, phs):
        """Compute interference-based similarity predictions."""
        preds = []
        for idx1, idx2 in pair_indices:
            z1 = mags[idx1] * np.exp(1j * phs[idx1])
            z2 = mags[idx2] * np.exp(1j * phs[idx2])
            # Normalized interference as similarity
            interference = np.abs(z1 + z2)**2
            # Normalize by max possible interference
            max_interf = (np.abs(z1) + np.abs(z2))**2
            sim = np.mean(interference / (max_interf + 1e-8))
            preds.append(sim)
        return np.array(preds)

    def loss(params):
        """MSE loss function."""
        mags = params[:vocab_size * dim].reshape(vocab_size, dim)
        phs = params[vocab_size * dim:].reshape(vocab_size, dim)
        preds = compute_predictions(mags, phs)
        return np.mean((preds - valid_scores)**2)

    # Pack parameters
    params = np.concatenate([magnitudes.ravel(), phases.ravel()])

    print("Optimizing embeddings...")
    # Use L-BFGS-B for efficient optimization
    result = minimize(
        loss,
        params,
        method='L-BFGS-B',
        options={'maxiter': n_epochs, 'disp': False}
    )

    # Unpack optimized parameters
    opt_params = result.x
    opt_magnitudes = opt_params[:vocab_size * dim].reshape(vocab_size, dim)
    opt_phases = opt_params[vocab_size * dim:].reshape(vocab_size, dim)

    print(f"Final loss: {result.fun:.4f}")

    return opt_magnitudes, opt_phases

# ============================================================================
# Statistical Testing
# ============================================================================

def test_phase_contribution(magnitude_features, interference_features, scores):
    """
    Test whether phase contributes predictive value beyond magnitude.

    Uses hierarchical regression:
    Model 1: similarity ~ magnitude_term (baseline)
    Model 2: similarity ~ magnitude_term + interference_term (full)

    Tests: Does adding interference_term significantly improve R²?
    """
    # Remove any invalid entries
    valid = ~(np.isnan(magnitude_features) | np.isnan(interference_features) | np.isnan(scores))
    mag = magnitude_features[valid]
    interf = interference_features[valid]
    y = scores[valid]

    if len(y) < 10:
        return None, None, None, "Insufficient data"

    # Normalize features
    mag_norm = (mag - np.mean(mag)) / (np.std(mag) + 1e-8)
    interf_norm = (interf - np.mean(interf)) / (np.std(interf) + 1e-8)

    # Model 1: Magnitude only
    slope1, intercept1, r1, p1, se1 = stats.linregress(mag_norm, y)
    r2_baseline = r1**2

    # Model 2: Magnitude + Interference (multiple regression)
    # Using matrix formulation: y = Xβ + ε
    X = np.column_stack([np.ones(len(y)), mag_norm, interf_norm])
    beta, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    # Compute R² for full model
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2_full = 1 - (ss_res / ss_tot)

    # F-test for significance of adding interference term
    # F = ((R²_full - R²_reduced) / df_num) / ((1 - R²_full) / df_denom)
    n = len(y)
    df_num = 1  # One additional parameter (interference)
    df_denom = n - 3  # n - (number of parameters in full model)

    if df_denom > 0 and r2_full < 1:
        f_stat = ((r2_full - r2_baseline) / df_num) / ((1 - r2_full) / df_denom)
        p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)
    else:
        f_stat = 0
        p_value = 1.0

    # Compute Spearman correlation for full model predictions
    spearman_rho, spearman_p = stats.spearmanr(y_pred, y)

    return {
        'r2_baseline': r2_baseline,
        'r2_full': r2_full,
        'r2_delta': r2_full - r2_baseline,
        'f_statistic': f_stat,
        'p_value': p_value,
        'spearman_rho': spearman_rho,
        'spearman_p': spearman_p,
        'n_samples': n,
        'interference_coefficient': beta[2] if len(beta) > 2 else 0
    }

# ============================================================================
# Main Experiment
# ============================================================================

def run_poc1():
    """Run the complete POC-1 experiment."""
    print("=" * 70)
    print("POC-1: Complex Embedding Interference Test")
    print("=" * 70)
    print()

    # Step 1: Load or generate data
    print("Step 1: Loading evaluation data...")
    simlex_path = download_simlex()

    if simlex_path and simlex_path.exists():
        pairs, scores = load_simlex(simlex_path)
        print(f"Loaded {len(pairs)} word pairs from SimLex-999")
    else:
        pairs, scores = generate_synthetic_data(500)
        print(f"Generated {len(pairs)} synthetic word pairs")

    print()

    # Step 2: Create vocabulary
    print("Step 2: Creating vocabulary...")
    vocab = create_word_vocabulary(pairs)
    print(f"Vocabulary size: {len(vocab)}")
    print()

    # Step 3: Initialize random embeddings (baseline)
    print("Step 3: Testing random embeddings (baseline)...")
    rand_mags, rand_phases = initialize_complex_embeddings(len(vocab), EMBEDDING_DIM)
    rand_mag_feats, rand_interf_feats, rand_full_feats = compute_all_interference_features(
        pairs, vocab, rand_mags, rand_phases
    )

    rand_results = test_phase_contribution(rand_mag_feats, rand_interf_feats, scores)
    print(f"  Random R² (magnitude only): {rand_results['r2_baseline']:.4f}")
    print(f"  Random R² (with phase):     {rand_results['r2_full']:.4f}")
    print(f"  Random Δ R²:                {rand_results['r2_delta']:.4f}")
    print()

    # Step 4: Train embeddings
    print("Step 4: Training complex embeddings...")
    trained_mags, trained_phases = train_embeddings(
        pairs, scores, vocab,
        dim=EMBEDDING_DIM,
        n_epochs=200
    )
    print()

    # Step 5: Evaluate trained embeddings
    print("Step 5: Evaluating trained embeddings...")
    train_mag_feats, train_interf_feats, train_full_feats = compute_all_interference_features(
        pairs, vocab, trained_mags, trained_phases
    )

    trained_results = test_phase_contribution(train_mag_feats, train_interf_feats, scores)
    print()

    # Step 6: Compute phase statistics
    print("Step 6: Analyzing phase variance...")
    phase_variance = np.var(trained_phases)
    phase_std = np.std(trained_phases)

    # Check if phases have learned structure (not uniform)
    # Under null (uniform), variance should be ~(2π)²/12 ≈ 3.29
    expected_uniform_var = (2 * np.pi)**2 / 12
    phase_variance_ratio = phase_variance / expected_uniform_var

    print(f"  Phase variance: {phase_variance:.4f}")
    print(f"  Expected uniform variance: {expected_uniform_var:.4f}")
    print(f"  Variance ratio: {phase_variance_ratio:.4f}")
    print()

    # Step 7: Report results
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("Model Performance:")
    print(f"  R² (magnitude only):     {trained_results['r2_baseline']:.4f}")
    print(f"  R² (magnitude + phase):  {trained_results['r2_full']:.4f}")
    print(f"  Δ R² (phase contribution): {trained_results['r2_delta']:.4f}")
    print()
    print("Statistical Test (F-test for phase contribution):")
    print(f"  F-statistic: {trained_results['f_statistic']:.4f}")
    print(f"  p-value:     {trained_results['p_value']:.6f}")
    print(f"  Threshold:   {P_THRESHOLD}")
    print()
    print(f"Spearman Correlation: ρ = {trained_results['spearman_rho']:.4f} (p = {trained_results['spearman_p']:.6f})")
    print(f"Interference Coefficient: {trained_results['interference_coefficient']:.4f}")
    print()

    # Step 8: Pass/Fail determination
    print("=" * 70)
    phase_significant = trained_results['p_value'] < P_THRESHOLD
    phase_positive = trained_results['r2_delta'] > 0
    spearman_valid = trained_results['spearman_rho'] > 0.1

    passed = phase_significant and phase_positive

    if passed:
        print("RESULT: ✓ PASS")
        print()
        print("Phase contribution is statistically significant (p < 0.01).")
        print(f"The interference term adds {trained_results['r2_delta']*100:.2f}% additional")
        print("variance explained beyond magnitude alone.")
        print()
        print("CONCLUSION: Phases in complex embeddings encode semantic information")
        print("that improves similarity prediction. Proceed to POC-2.")
    else:
        print("RESULT: ✗ FAIL")
        print()
        if not phase_significant:
            print(f"Phase contribution not significant (p = {trained_results['p_value']:.4f} >= {P_THRESHOLD})")
        if not phase_positive:
            print(f"Phase does not improve prediction (ΔR² = {trained_results['r2_delta']:.4f} <= 0)")
        print()
        print("CONCLUSION: Cannot reject null hypothesis. Phases do not contribute")
        print("meaningful predictive value beyond magnitude.")
        print()
        print("RECOMMENDATION: Reassess complex embedding approach before proceeding.")

    print("=" * 70)

    return {
        'passed': passed,
        'trained_results': trained_results,
        'random_results': rand_results,
        'phase_variance': phase_variance,
        'phase_variance_ratio': phase_variance_ratio
    }


if __name__ == "__main__":
    results = run_poc1()

    # Exit with appropriate code for CI/automation
    import sys
    sys.exit(0 if results['passed'] else 1)
