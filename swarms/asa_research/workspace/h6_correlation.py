#!/usr/bin/env python3
"""
H6 Correlation Experiment
=========================

Tests the core ASA hypothesis: Do transformers naturally attend along
linguistically valid pathways?

Method:
1. Load trained baseline model (mode='none')
2. Extract attention weights on test data
3. Compute ASA bonding mask for same inputs
4. Measure: What % of baseline attention falls on ASA-compatible pairs?

If overlap is high (60-75%), ASA is making explicit what transformers learn anyway.
If overlap is low (<50%), ASA might be blocking important learned patterns.

Usage:
    python h6_correlation.py --baseline-checkpoint ./asa_output/none_tiny/best.pt
"""

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add parent dir to path for imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_model_and_config(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    from asa_v2_2 import create_model
    from transformers import AutoTokenizer
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        size=config['size'],
        mode=config['mode'],
        alpha=config.get('alpha', 1.0),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config, tokenizer


def extract_attention_weights(
    model,
    input_ids: torch.Tensor,
    pos_ids: torch.Tensor,
    features: torch.Tensor,
    requirements: torch.Tensor,
    attention_mask: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Extract attention weights from all layers.
    
    Returns list of [batch, heads, seq, seq] tensors, one per layer.
    """
    # We need to modify forward pass to return attention weights
    # For now, we'll hook into the attention layers
    
    attention_weights = []
    hooks = []
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            # ASAAttention returns (output, weights) when return_attention=True
            # But we're using the model's forward pass, so we need to recompute
            pass
        return hook
    
    # Simpler approach: manually run attention with return_attention=True
    with torch.no_grad():
        x = model.token_embedding(input_ids)
        x = model.pos_encoding(x)
        
        for layer in model.layers:
            # Get attention weights by calling attention directly
            attn_out, weights = layer.attention(
                x,
                pos_ids=pos_ids,
                features=features,
                requirements=requirements,
                attention_mask=attention_mask,
                causal=True,
                return_attention=True,
            )
            attention_weights.append(weights)
            
            # Continue forward pass
            x = layer.norm1(x + attn_out)
            x = layer.norm2(x + layer.ff(x))
    
    return attention_weights


def compute_random_mask(
    seq: int,
    batch: int,
    sparsity: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate random mask with same sparsity as ASA.
    
    Control experiment: If random mask achieves similar overlap,
    ASA's linguistic structure adds no value.
    """
    # Random mask with target sparsity
    mask = torch.rand(batch, seq, seq, device=device) > sparsity
    
    # Ensure diagonal is always valid (self-attention)
    diag = torch.eye(seq, dtype=torch.bool, device=device)
    mask = mask | diag.unsqueeze(0)
    
    return mask


def compute_asa_mask(
    pos_ids: torch.Tensor,
    features: torch.Tensor,
    requirements: torch.Tensor,
) -> torch.Tensor:
    """Compute ASA bonding mask."""
    from asa_v2_2 import BondingComputer
    
    device = pos_ids.device
    bonding = BondingComputer(mode='full')
    bonding.to(device)  # Move pos_matrix to same device
    mask, _ = bonding.compute_bonding_mask(pos_ids, features, requirements)
    
    return mask


def compute_overlap(
    attention_weights: torch.Tensor,
    asa_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold: float = 0.0,
    top_k: int = None,
) -> Dict[str, float]:
    """
    Compute overlap between attention weights and ASA mask.
    
    Args:
        attention_weights: [batch, heads, seq, seq]
        asa_mask: [batch, seq, seq] boolean
        attention_mask: [batch, seq] boolean (valid tokens)
        threshold: Only count attention weights above this value
        top_k: If set, only consider top-k attended positions per query
    
    Returns:
        Dictionary with overlap metrics
    """
    batch, heads, seq, _ = attention_weights.shape
    device = attention_weights.device
    
    # Expand masks
    asa_mask_expanded = asa_mask.unsqueeze(1).expand(-1, heads, -1, -1)
    
    # Create valid pair mask (both positions are real tokens)
    valid_tokens = attention_mask.float()
    valid_pairs = valid_tokens.unsqueeze(-1) * valid_tokens.unsqueeze(-2)
    valid_pairs = valid_pairs.unsqueeze(1).expand(-1, heads, -1, -1)
    
    # Apply causal mask (we only look at positions that could be attended)
    causal_mask = torch.tril(torch.ones(seq, seq, device=device)).unsqueeze(0).unsqueeze(0)
    valid_pairs = valid_pairs * causal_mask
    
    # Total attention mass on ASA-compatible pairs
    attention_on_asa = (attention_weights * asa_mask_expanded.float() * valid_pairs).sum()
    total_attention = (attention_weights * valid_pairs).sum()
    
    mass_overlap = (attention_on_asa / total_attention).item() if total_attention > 0 else 0
    
    # Count-based overlap (what fraction of "significant" attention is on ASA pairs)
    if threshold > 0:
        significant = attention_weights > threshold
        significant_on_asa = (significant & asa_mask_expanded & (valid_pairs > 0)).sum()
        total_significant = (significant & (valid_pairs > 0)).sum()
        count_overlap = (significant_on_asa / total_significant).float().item() if total_significant > 0 else 0
    else:
        count_overlap = mass_overlap
    
    # Top-k overlap
    if top_k is not None:
        # For each query, check if top-k attended positions are ASA-compatible
        topk_hits = 0
        topk_total = 0
        
        for b in range(batch):
            for h in range(heads):
                for i in range(seq):
                    if not attention_mask[b, i]:
                        continue
                    
                    # Get attention weights for this query
                    attn_row = attention_weights[b, h, i, :i+1]  # Causal: only up to position i
                    if len(attn_row) == 0:
                        continue
                    
                    # Get top-k indices
                    k = min(top_k, len(attn_row))
                    _, topk_indices = attn_row.topk(k)
                    
                    # Check how many are ASA-compatible
                    for j in topk_indices:
                        if attention_mask[b, j]:
                            topk_total += 1
                            if asa_mask[b, i, j]:
                                topk_hits += 1
        
        topk_overlap = topk_hits / topk_total if topk_total > 0 else 0
    else:
        topk_overlap = None
    
    return {
        'mass_overlap': mass_overlap,
        'count_overlap': count_overlap,
        'topk_overlap': topk_overlap,
        'total_attention_mass': total_attention.item(),
    }


def run_h6_experiment(
    checkpoint_path: str,
    cache_dir: str = './asa_cache',
    dataset: str = 'wikitext-2',
    num_samples: int = 100,
    threshold: float = 0.01,
    top_k: int = 10,
) -> Dict:
    """
    Run the full H6 correlation experiment.
    """
    from train_asa import ASADataset, collate_fn
    from torch.utils.data import DataLoader
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Running H6 Correlation Experiment")
    print(f"=" * 50)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Samples: {num_samples}")
    print()
    
    # Load model
    print("Loading model...")
    model, config, tokenizer = load_model_and_config(checkpoint_path, device)
    print(f"  Mode: {config['mode']}")
    print(f"  Size: {config['size']}")
    
    # Load test data
    print("\nLoading test data...")
    test_path = os.path.join(cache_dir, f"{dataset}_test_asa.pt")
    test_data = ASADataset.load(test_path)
    print(f"  Total samples: {len(test_data)}")
    
    # Take subset
    subset_data = ASADataset(test_data.data[:num_samples])
    loader = DataLoader(subset_data, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Run experiment
    print(f"\nExtracting attention and computing overlap...")
    
    all_results = {
        'mass_overlap': [],
        'count_overlap': [],
        'topk_overlap': [],
        'per_layer': {i: [] for i in range(len(model.layers))},
        # Random mask control
        'random_mass_overlap': [],
        'random_count_overlap': [],
    }
    
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= num_samples:
            break
        
        if batch_idx % 20 == 0:
            print(f"  Processing sample {batch_idx}/{num_samples}...")
        
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Extract attention weights
        attention_weights = extract_attention_weights(
            model,
            batch['input_ids'],
            batch['pos_ids'],
            batch['features'],
            batch['requirements'],
            batch['attention_mask'],
        )
        
        # Compute ASA mask
        asa_mask = compute_asa_mask(
            batch['pos_ids'],
            batch['features'],
            batch['requirements'],
        )
        
        # Apply causal mask to ASA mask
        seq = asa_mask.shape[1]
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=device))
        asa_mask = asa_mask & causal.unsqueeze(0)
        
        # Compute overlap for each layer
        for layer_idx, layer_attn in enumerate(attention_weights):
            layer_results = compute_overlap(
                layer_attn,
                asa_mask,
                batch['attention_mask'],
                threshold=threshold,
                top_k=top_k,
            )
            all_results['per_layer'][layer_idx].append(layer_results['mass_overlap'])
        
        # Aggregate across layers (average attention)
        avg_attention = torch.stack(attention_weights).mean(dim=0)
        sample_results = compute_overlap(
            avg_attention,
            asa_mask,
            batch['attention_mask'],
            threshold=threshold,
            top_k=top_k,
        )
        
        all_results['mass_overlap'].append(sample_results['mass_overlap'])
        all_results['count_overlap'].append(sample_results['count_overlap'])
        if sample_results['topk_overlap'] is not None:
            all_results['topk_overlap'].append(sample_results['topk_overlap'])
        
        # CONTROL: Random mask with same sparsity
        asa_sparsity = 1 - (asa_mask.float().mean().item())
        random_mask = compute_random_mask(seq, 1, asa_sparsity, device)
        random_mask = random_mask & causal.unsqueeze(0)  # Apply causal
        
        random_results = compute_overlap(
            avg_attention,
            random_mask,
            batch['attention_mask'],
            threshold=threshold,
            top_k=None,
        )
        all_results['random_mass_overlap'].append(random_results['mass_overlap'])
        all_results['random_count_overlap'].append(random_results['count_overlap'])
    
    # Aggregate results
    results = {
        'config': config,
        'num_samples': num_samples,
        'threshold': threshold,
        'top_k': top_k,
        'metrics': {
            'mass_overlap': {
                'mean': sum(all_results['mass_overlap']) / len(all_results['mass_overlap']),
                'min': min(all_results['mass_overlap']),
                'max': max(all_results['mass_overlap']),
            },
            'count_overlap': {
                'mean': sum(all_results['count_overlap']) / len(all_results['count_overlap']),
                'min': min(all_results['count_overlap']),
                'max': max(all_results['count_overlap']),
            },
            'topk_overlap': {
                'mean': sum(all_results['topk_overlap']) / len(all_results['topk_overlap']) if all_results['topk_overlap'] else None,
            },
            'per_layer': {
                layer: {
                    'mean': sum(overlaps) / len(overlaps),
                }
                for layer, overlaps in all_results['per_layer'].items()
            },
            # Random mask control
            'random_mass_overlap': {
                'mean': sum(all_results['random_mass_overlap']) / len(all_results['random_mass_overlap']),
                'min': min(all_results['random_mass_overlap']),
                'max': max(all_results['random_mass_overlap']),
            },
            'random_count_overlap': {
                'mean': sum(all_results['random_count_overlap']) / len(all_results['random_count_overlap']),
            },
        },
    }
    
    return results


def print_results(results: Dict):
    """Pretty print experiment results."""
    print()
    print("=" * 60)
    print("H6 CORRELATION EXPERIMENT RESULTS")
    print("=" * 60)
    print()
    
    metrics = results['metrics']
    
    print(f"Overall Attention-ASA Overlap:")
    print(f"-" * 40)
    print(f"  Mass overlap:  {metrics['mass_overlap']['mean']:.1%}")
    print(f"    (range: {metrics['mass_overlap']['min']:.1%} - {metrics['mass_overlap']['max']:.1%})")
    print()
    print(f"  Count overlap (threshold={results['threshold']}): {metrics['count_overlap']['mean']:.1%}")
    print()
    if metrics['topk_overlap']['mean'] is not None:
        print(f"  Top-{results['top_k']} overlap: {metrics['topk_overlap']['mean']:.1%}")
        print()
    
    print(f"Per-Layer Breakdown:")
    print(f"-" * 40)
    for layer, layer_metrics in metrics['per_layer'].items():
        print(f"  Layer {layer}: {layer_metrics['mean']:.1%}")
    
    print()
    print(f"=" * 60)
    print("CONTROL: Random Mask (Same Sparsity)")
    print(f"=" * 60)
    print()
    print(f"  Random mass overlap:  {metrics['random_mass_overlap']['mean']:.1%}")
    print(f"    (range: {metrics['random_mass_overlap']['min']:.1%} - {metrics['random_mass_overlap']['max']:.1%})")
    print()
    print(f"  Random count overlap: {metrics['random_count_overlap']['mean']:.1%}")
    print()
    
    # Compute improvement over random
    asa_overlap = metrics['mass_overlap']['mean']
    random_overlap = metrics['random_mass_overlap']['mean']
    improvement = (asa_overlap - random_overlap) / random_overlap * 100
    
    print(f"  ASA vs Random: {asa_overlap:.1%} vs {random_overlap:.1%}")
    print(f"  Improvement:   +{improvement:.1f}% above random baseline")
    
    print()
    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    
    if improvement >= 50:
        interpretation = "STRONG SUPPORT"
        detail = f"ASA achieves {improvement:.0f}% higher overlap than random mask. Linguistic structure captures meaningful attention patterns."
    elif improvement >= 25:
        interpretation = "MODERATE SUPPORT"
        detail = f"ASA achieves {improvement:.0f}% higher overlap than random mask. Linguistic structure provides useful bias."
    elif improvement >= 10:
        interpretation = "WEAK SUPPORT"
        detail = f"ASA achieves {improvement:.0f}% higher overlap than random mask. Some value from linguistic structure."
    else:
        interpretation = "NOT SUPPORTED"
        detail = f"ASA achieves only {improvement:.0f}% higher overlap than random mask. Linguistic structure may not be capturing meaningful patterns."
    
    print(f"\n  {interpretation}")
    print(f"\n  {detail}")
    print()
    print(f"  Summary:")
    print(f"    - ASA overlap:    {asa_overlap:.1%}")
    print(f"    - Random overlap: {random_overlap:.1%}")
    print(f"    - Difference:     {asa_overlap - random_overlap:.1%} absolute")
    print()


def main():
    parser = argparse.ArgumentParser(description='H6 Correlation Experiment')
    parser.add_argument('--baseline-checkpoint', type=str, 
                        default='./asa_output/none_tiny/best.pt',
                        help='Path to trained baseline checkpoint')
    parser.add_argument('--cache-dir', type=str, default='./asa_cache')
    parser.add_argument('--dataset', type=str, default='wikitext-2')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--output', type=str, default='h6_results.json')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_h6_experiment(
        args.baseline_checkpoint,
        args.cache_dir,
        args.dataset,
        args.num_samples,
        args.threshold,
        args.top_k,
    )
    
    # Print results
    print_results(results)
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        # Convert non-serializable items
        results_json = {
            'num_samples': results['num_samples'],
            'threshold': results['threshold'],
            'top_k': results['top_k'],
            'metrics': {
                'mass_overlap': results['metrics']['mass_overlap'],
                'count_overlap': results['metrics']['count_overlap'],
                'topk_overlap': results['metrics']['topk_overlap'],
                'per_layer': {str(k): v for k, v in results['metrics']['per_layer'].items()},
                'random_mass_overlap': results['metrics']['random_mass_overlap'],
                'random_count_overlap': results['metrics']['random_count_overlap'],
            },
        }
        json.dump(results_json, f, indent=2)
    
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
