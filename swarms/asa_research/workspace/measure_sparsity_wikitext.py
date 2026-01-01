#!/usr/bin/env python3
"""
ASA Sparsity Measurement on WikiText-2
======================================

Run this to get the headline sparsity number for the paper.

Usage:
    python3 measure_sparsity_wikitext.py          # Quick (500 texts)
    python3 measure_sparsity_wikitext.py --full   # Full validation set (~2400 texts)

Output:
    - Total pairs analyzed (off-diagonal only)
    - Blocked pairs (incompatible)
    - Sparsity % (the headline number)
    - Top blocked POS pairs
    - Sequence length statistics
    - Extraction timing

Requirements:
    - asa_v2_2.py (or asa_v2_2_fixed.py) in same directory
    - pip install spacy datasets nltk
    - python -m spacy download en_core_web_sm
    - python -c "import nltk; nltk.download('wordnet')"
"""

import sys
import os
import argparse
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='ASA Sparsity Measurement')
    parser.add_argument('--full', action='store_true', 
                        help='Use full validation set (~2400 texts) instead of 500')
    parser.add_argument('--max-texts', type=int, default=500,
                        help='Max texts to process (default: 500, ignored if --full)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("ASA SPARSITY MEASUREMENT")
    print("=" * 60)
    print()
    
    # Import ASA
    try:
        from asa_v2_2_fixed import (
            measure_sparsity, 
            measure_pos_matrix_sparsity,
            PropertyExtractor,
            BondingComputer,
            POS_TO_ID,
        )
        print("✅ Loaded asa_v2_2_fixed.py")
    except ImportError:
        try:
            from asa_v2_2 import (
                measure_sparsity, 
                measure_pos_matrix_sparsity,
                PropertyExtractor,
                BondingComputer,
                POS_TO_ID,
            )
            print("✅ Loaded asa_v2_2.py")
        except ImportError as e:
            print(f"❌ Could not import ASA: {e}")
            print("   Make sure asa_v2_2.py is in the same directory")
            return
    
    # Load WikiText-2
    print("\n📊 Loading WikiText-2 validation set...")
    try:
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
        texts = [t for t in ds['text'] if t.strip() and len(t.split()) >= 5]
        print(f"   Loaded {len(texts)} non-empty texts (≥5 words)")
    except Exception as e:
        print(f"❌ Could not load WikiText-2: {e}")
        print("   Using sample texts instead...")
        texts = [
            "The cat sat on the mat and watched the birds outside.",
            "Scientists discovered a new species of butterfly in the Amazon rainforest.",
            "The company announced record profits for the third quarter.",
            "She walked quickly through the crowded streets of the city.",
            "The president addressed the nation about the economic crisis.",
            "Children learn best when they are engaged and curious.",
            "The old house stood empty at the end of the long dirt road.",
            "Technology continues to transform how we communicate with each other.",
            "The musician played a beautiful melody on the grand piano.",
            "Researchers found evidence of water on the surface of Mars.",
        ] * 10  # Repeat for more data
    
    # Apply text limit
    if args.full:
        print(f"   Using FULL dataset: {len(texts)} texts")
    else:
        max_texts = args.max_texts
        if len(texts) > max_texts:
            print(f"   Using first {max_texts} texts (use --full for all {len(texts)})")
            texts = texts[:max_texts]
    
    # Sequence length statistics
    seq_lengths = [len(t.split()) for t in texts]
    print(f"\n📏 Sequence length statistics:")
    print(f"   Texts:    {len(texts)}")
    print(f"   Avg len:  {sum(seq_lengths)/len(seq_lengths):.1f} words")
    print(f"   Median:   {sorted(seq_lengths)[len(seq_lengths)//2]} words")
    print(f"   Min:      {min(seq_lengths)} words")
    print(f"   Max:      {max(seq_lengths)} words")
    
    # Note about diagonal exclusion
    print(f"\n📝 Note: Sparsity measured on OFF-DIAGONAL pairs only (i ≠ j)")
    print(f"   Self-attention (diagonal) is always allowed.")
    
    # Initialize extractor
    print("\n🔧 Initializing property extractor...")
    extractor = PropertyExtractor()
    
    # Measure theoretical POS matrix sparsity first
    print("\n" + "=" * 60)
    print("1. THEORETICAL POS MATRIX SPARSITY")
    print("=" * 60)
    pos_stats = measure_pos_matrix_sparsity()
    print(f"   Raw sparsity (uniform POS dist):    {pos_stats['raw_sparsity']:.1%}")
    print(f"   Weighted sparsity (English dist):   {pos_stats['weighted_sparsity']:.1%}")
    
    # Measure actual sparsity for each mode
    print("\n" + "=" * 60)
    print("2. ACTUAL SPARSITY ON WIKITEXT-2")
    print("=" * 60)
    
    modes = ['full', 'pos_only', 'features_only']
    results = {}
    
    for mode in modes:
        print(f"\n   Measuring mode='{mode}'...")
        
        start_time = time.time()
        result = measure_sparsity(
            texts, 
            extractor=extractor, 
            mode=mode,
            show_blocked_pairs=(mode == 'full')  # Only show for full mode
        )
        elapsed = time.time() - start_time
        
        results[mode] = result
        results[mode]['time_sec'] = elapsed
        
        print(f"   Total pairs:    {result['total_pairs']:,}")
        print(f"   Blocked pairs:  {result['blocked_pairs']:,}")
        print(f"   Sparsity:       {result['sparsity']:.1%}")
        print(f"   Time:           {elapsed:.1f}s ({len(texts)/elapsed:.0f} texts/sec)")
        
        if 'top_blocked_pos_pairs' in result:
            print(f"\n   Top blocked POS pairs:")
            for pair, count in result['top_blocked_pos_pairs'][:5]:
                print(f"      {pair[0]:6} → {pair[1]:6}: {count:,} blocked")
    
    # Summary table
    print("\n" + "=" * 60)
    print("3. SUMMARY TABLE (for paper)")
    print("=" * 60)
    print()
    print("| Mode          | Total Pairs | Blocked   | Sparsity | Time (s) |")
    print("|---------------|-------------|-----------|----------|----------|")
    print(f"| none          | -           | 0         | 0.0%     | -        |")
    for mode in modes:
        r = results[mode]
        print(f"| {mode:13} | {r['total_pairs']:>11,} | {r['blocked_pairs']:>9,} | {r['sparsity']:>7.1%}  | {r['time_sec']:>7.1f}  |")
    
    # The headline number
    print("\n" + "=" * 60)
    print("📰 HEADLINE NUMBER FOR PAPER")
    print("=" * 60)
    full_sparsity = results['full']['sparsity']
    print(f"\n   ASA achieves {full_sparsity:.1%} sparsity on WikiText-2 validation")
    print(f"   ({results['full']['blocked_pairs']:,} of {results['full']['total_pairs']:,} off-diagonal pairs blocked)")
    print(f"   Measured on {len(texts)} texts, avg {sum(seq_lengths)/len(seq_lengths):.0f} words each")
    print()
    
    # Save results to JSON
    import json
    output_file = 'sparsity_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'pos_matrix': pos_stats,
            'wikitext2': {mode: {k: v for k, v in r.items() if k != 'top_blocked_pos_pairs'} 
                         for mode, r in results.items()},
            'top_blocked_pairs': results['full'].get('top_blocked_pos_pairs', []),
            'num_texts': len(texts),
            'sequence_stats': {
                'mean': sum(seq_lengths)/len(seq_lengths),
                'median': sorted(seq_lengths)[len(seq_lengths)//2],
                'min': min(seq_lengths),
                'max': max(seq_lengths),
            },
            'note': 'Sparsity measured on off-diagonal pairs only (i != j)',
        }, f, indent=2)
    print(f"💾 Results saved to {output_file}")
    
    return results


if __name__ == '__main__':
    main()
