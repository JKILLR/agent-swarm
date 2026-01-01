#!/usr/bin/env python3
"""
ASA Ablation Experiments (v2)
=============================

Runs all 4 ablation modes and produces the comparison table for the paper.

Usage:
    # Quick test (1 epoch)
    python3 run_ablations.py --epochs 1 --quick
    
    # Full run (overnight)
    python3 run_ablations.py --epochs 10

Output:
    - Checkpoints for each mode
    - ablation_results.json with all metrics
    - Learning curves for paper figures
    - Formatted table for paper

Modes tested:
    - none:          Standard transformer (baseline)
    - pos_only:      Only POS compatibility mask
    - features_only: Only feature compatibility scores  
    - full:          POS mask + features + blocking

Requirements:
    - Preprocessed WikiText-2 data in ./asa_cache/
    - Run: python3 train_asa.py preprocess --dataset wikitext-2
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    import torch
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For MPS, manual_seed covers it
    
    # Make cudnn deterministic (slower but reproducible)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"🎲 Random seed set to {seed}")


def run_ablations(epochs: int = 10, batch_size: int = 4, size: str = 'tiny', 
                  quick: bool = False, seed: int = 42):
    """Run all ablation experiments."""
    
    print("=" * 70)
    print("ASA ABLATION EXPERIMENTS v2")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Model size: {size}")
    print(f"Seed: {seed}")
    print()
    
    # Set seed first
    set_seed(seed)
    
    # Check for cached data
    cache_dir = './asa_cache'
    if not Path(cache_dir).exists():
        print("❌ No cached data found!")
        print("   Run: python3 train_asa.py preprocess --dataset wikitext-2")
        return None
    
    # Import training components
    try:
        from train_asa import ASADataset, collate_fn, TrainConfig
        from asa_v2_2_fixed import create_model, count_parameters, measure_sparsity, PropertyExtractor
    except ImportError:
        try:
            from train_asa import ASADataset, collate_fn, TrainConfig
            from asa_v2_2 import create_model, count_parameters, measure_sparsity, PropertyExtractor
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return None
    
    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from transformers import AutoTokenizer
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print("\n📊 Loading data...")
    train_data = ASADataset.load(os.path.join(cache_dir, 'wikitext-2_train_asa.pt'))
    val_data = ASADataset.load(os.path.join(cache_dir, 'wikitext-2_validation_asa.pt'))
    print(f"   Train: {len(train_data)} examples")
    print(f"   Val: {len(val_data)} examples")
    
    if quick:
        # Use subset for quick testing
        train_data = ASADataset(train_data.data[:1000])
        val_data = ASADataset(val_data.data[:200])
        print(f"   [QUICK MODE] Using {len(train_data)} train, {len(val_data)} val")
    
    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
    
    # Modes to test - run baseline FIRST to get target PPL
    modes = ['none', 'pos_only', 'features_only', 'full']
    
    results = {}
    baseline_final_ppl = None  # Will be set after 'none' mode
    
    for mode in modes:
        print("\n" + "=" * 70)
        print(f"MODE: {mode.upper()}")
        print("=" * 70)
        
        # Reset seed for each mode for fair comparison
        set_seed(seed)
        
        # Config
        output_dir = f'./asa_ablation/{mode}_{size}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Model - fresh instance each time
        model = create_model(
            vocab_size=tokenizer.vocab_size,
            size=size,
            mode=mode,
            alpha=1.0,
            hard_block=True,
            pad_token_id=tokenizer.pad_token_id,
        )
        model.to(device)
        
        n_params = count_parameters(model)
        print(f"Parameters: {n_params:,}")
        
        # Fresh optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
        total_steps = len(train_loader) * epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training state - explicitly reset
        global_step = 0
        best_val_loss = float('inf')
        
        # Track convergence to baseline PPL (dynamic target)
        convergence_step = None
        target_ppl = baseline_final_ppl if baseline_final_ppl else None
        
        # Learning curves for paper
        train_losses = []  # (step, loss)
        val_ppls = []      # (step, ppl)
        
        start_time = time.time()
        
        print(f"\nTraining {mode} for {epochs} epochs...")
        if target_ppl:
            print(f"   Target PPL (baseline): {target_ppl:.2f}")
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                result = model(
                    input_ids=batch['input_ids'],
                    pos_ids=batch['pos_ids'],
                    features=batch['features'],
                    requirements=batch['requirements'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids'],
                )
                
                loss = result['loss']
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1
                
                # Log training loss every 100 steps
                if global_step % 100 == 0:
                    train_losses.append((global_step, loss.item()))
                
                # Evaluate every 500 steps
                if global_step % 500 == 0:
                    model.eval()
                    val_loss = 0
                    val_batches = 0
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_batch = {k: v.to(device) for k, v in val_batch.items()}
                            val_result = model(
                                input_ids=val_batch['input_ids'],
                                pos_ids=val_batch['pos_ids'],
                                features=val_batch['features'],
                                requirements=val_batch['requirements'],
                                attention_mask=val_batch['attention_mask'],
                                labels=val_batch['input_ids'],
                            )
                            val_loss += val_result['loss'].item()
                            val_batches += 1
                    
                    avg_val_loss = val_loss / val_batches
                    val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
                    val_ppls.append((global_step, val_ppl))
                    
                    # Check convergence to baseline
                    if target_ppl and convergence_step is None and val_ppl <= target_ppl:
                        convergence_step = global_step
                        print(f"  ✓ Reached baseline PPL {target_ppl:.2f} at step {convergence_step}")
                    
                    # Track best
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        # Save best checkpoint
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'global_step': global_step,
                            'best_val_loss': best_val_loss,
                            'mode': mode,
                        }, os.path.join(output_dir, 'best.pt'))
                    
                    model.train()
            
            # End of epoch eval
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = {k: v.to(device) for k, v in val_batch.items()}
                    val_result = model(
                        input_ids=val_batch['input_ids'],
                        pos_ids=val_batch['pos_ids'],
                        features=val_batch['features'],
                        requirements=val_batch['requirements'],
                        attention_mask=val_batch['attention_mask'],
                        labels=val_batch['input_ids'],
                    )
                    val_loss += val_result['loss'].item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
            print(f"  Epoch {epoch+1}/{epochs}: Val PPL = {val_ppl:.2f}")
        
        elapsed = time.time() - start_time
        
        # Final evaluation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                val_result = model(
                    input_ids=val_batch['input_ids'],
                    pos_ids=val_batch['pos_ids'],
                    features=val_batch['features'],
                    requirements=val_batch['requirements'],
                    attention_mask=val_batch['attention_mask'],
                    labels=val_batch['input_ids'],
                )
                val_loss += val_result['loss'].item()
                val_batches += 1
        
        final_loss = val_loss / val_batches
        final_ppl = torch.exp(torch.tensor(final_loss)).item()
        
        # Save final checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
            'final_loss': final_loss,
            'mode': mode,
        }, os.path.join(output_dir, 'final.pt'))
        
        # Set baseline target for subsequent modes
        if mode == 'none':
            baseline_final_ppl = final_ppl
            print(f"\n  📊 Baseline PPL set to {baseline_final_ppl:.2f}")
        
        # Record results
        results[mode] = {
            'final_ppl': final_ppl,
            'final_loss': final_loss,
            'best_val_loss': best_val_loss,
            'convergence_step': convergence_step,
            'total_steps': global_step,
            'training_time_sec': elapsed,
            'training_time_min': elapsed / 60,
            'parameters': n_params,
            'train_losses': train_losses,  # Learning curve data
            'val_ppls': val_ppls,          # Validation curve data
        }
        
        print(f"\n  Final PPL: {final_ppl:.2f}")
        print(f"  Convergence step: {convergence_step if convergence_step else 'N/A'}")
        print(f"  Training time: {elapsed/60:.1f} min")
    
    # Measure sparsity for each mode (500 texts for consistency)
    print("\n" + "=" * 70)
    print("MEASURING SPARSITY (500 texts)")
    print("=" * 70)
    
    # Get sample texts for sparsity measurement
    from datasets import load_dataset
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    sample_texts = [t for t in ds['text'] if t.strip() and len(t.split()) >= 5][:500]
    print(f"   Using {len(sample_texts)} texts for sparsity measurement")
    
    extractor = PropertyExtractor()
    
    for mode in modes:
        if mode == 'none':
            results[mode]['sparsity'] = 0.0
        else:
            sparsity_result = measure_sparsity(sample_texts, extractor=extractor, mode=mode)
            results[mode]['sparsity'] = sparsity_result['sparsity']
            print(f"   {mode}: {sparsity_result['sparsity']:.1%} sparsity")
    
    # Print summary table
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)
    print()
    print("| Mode          | Sparsity | Final PPL | Steps to Baseline | Time (min) |")
    print("|---------------|----------|-----------|-------------------|------------|")
    
    for mode in modes:
        r = results[mode]
        sparsity = f"{r['sparsity']:.1%}" if r['sparsity'] > 0 else "0.0%"
        conv = str(r['convergence_step']) if r['convergence_step'] else "-"
        print(f"| {mode:13} | {sparsity:>8} | {r['final_ppl']:>9.2f} | {conv:>17} | {r['training_time_min']:>10.1f} |")
    
    # Compute speedups
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)
    
    baseline_steps = results['none']['total_steps']
    print(f"\nBaseline (none) final PPL: {results['none']['final_ppl']:.2f}")
    print(f"Baseline total steps: {baseline_steps}")
    
    for mode in ['pos_only', 'features_only', 'full']:
        r = results[mode]
        if r['convergence_step']:
            speedup = (baseline_steps - r['convergence_step']) / baseline_steps * 100
            print(f"\n{mode}:")
            print(f"  Reached baseline PPL at step {r['convergence_step']}")
            print(f"  Speedup: {speedup:.1f}% fewer steps")
        else:
            print(f"\n{mode}:")
            print(f"  Did not reach baseline PPL {results['none']['final_ppl']:.2f}")
            print(f"  Final PPL: {r['final_ppl']:.2f}")
    
    # Save results (exclude large curve data for JSON, save separately)
    output_file = 'ablation_results.json'
    results_summary = {}
    for mode, r in results.items():
        results_summary[mode] = {k: v for k, v in r.items() 
                                  if k not in ['train_losses', 'val_ppls']}
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'size': size,
                'device': str(device),
                'seed': seed,
            },
            'baseline_final_ppl': baseline_final_ppl,
            'results': results_summary,
        }, f, indent=2)
    
    # Save learning curves separately
    curves_file = 'ablation_curves.json'
    with open(curves_file, 'w') as f:
        json.dump({
            mode: {
                'train_losses': r['train_losses'],
                'val_ppls': r['val_ppls'],
            }
            for mode, r in results.items()
        }, f, indent=2)
    
    print(f"\n💾 Results saved to {output_file}")
    print(f"📈 Learning curves saved to {curves_file}")
    print(f"⏱️  Total time: {sum(r['training_time_min'] for r in results.values()):.1f} min")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='ASA Ablation Experiments')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--size', default='tiny', choices=['tiny', 'small', 'medium'])
    parser.add_argument('--quick', action='store_true', help='Quick test with subset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set memory optimization for MPS
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    run_ablations(
        epochs=args.epochs,
        batch_size=args.batch_size,
        size=args.size,
        quick=args.quick,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
