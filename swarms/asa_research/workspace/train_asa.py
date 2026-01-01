#!/usr/bin/env python3
"""
ASA Training Script
===================

Trains ASA and baseline models on WikiText-2 for comparison.

Usage:
    # Preprocess (once)
    python train_asa.py preprocess --dataset wikitext-2
    
    # Train ASA
    python train_asa.py train --mode full --size small
    
    # Train baseline
    python train_asa.py train --mode none --size small
    
    # Compare results
    python train_asa.py evaluate --checkpoint asa_full.pt --checkpoint baseline.pt

Requirements:
    pip install torch transformers datasets tqdm
    pip install spacy nltk
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('wordnet')"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    size: str = 'small'
    mode: str = 'full'  # full, pos_only, features_only, none
    alpha: float = 1.0
    hard_block: bool = True
    soft_penalty: float = 10.0
    
    # Data
    dataset: str = 'wikitext-2'
    max_length: int = 256
    
    # Training
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    epochs: int = 10
    warmup_steps: int = 500
    grad_clip: float = 1.0
    
    # Logging
    log_interval: int = 100
    eval_interval: int = 500
    save_interval: int = 1000
    
    # Paths
    output_dir: str = './asa_output'
    cache_dir: str = './asa_cache'
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainConfig':
        with open(path) as f:
            return cls(**json.load(f))


# =============================================================================
# DATA LOADING
# =============================================================================

class ASADataset(Dataset):
    """Dataset with precomputed ASA properties."""
    
    def __init__(self, data: List[Dict[str, torch.Tensor]]):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def save(self, path: str):
        torch.save(self.data, path)
    
    @classmethod
    def load(cls, path: str) -> 'ASADataset':
        return cls(torch.load(path))


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding."""
    max_len = max(item['input_ids'].shape[0] for item in batch)
    
    input_ids = []
    pos_ids = []
    features = []
    requirements = []
    attention_mask = []
    
    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_len - seq_len
        
        # Pad each tensor
        input_ids.append(torch.cat([
            item['input_ids'],
            torch.zeros(pad_len, dtype=torch.long)
        ]))
        pos_ids.append(torch.cat([
            item['pos_ids'],
            torch.full((pad_len,), 16, dtype=torch.long)  # 'X' POS
        ]))
        features.append(torch.cat([
            item['features'],
            torch.zeros(pad_len, item['features'].shape[-1])
        ]))
        requirements.append(torch.cat([
            item['requirements'],
            torch.zeros(pad_len, item['requirements'].shape[-1])
        ]))
        
        mask = torch.zeros(max_len, dtype=torch.bool)
        mask[:seq_len] = True
        attention_mask.append(mask)
    
    return {
        'input_ids': torch.stack(input_ids),
        'pos_ids': torch.stack(pos_ids),
        'features': torch.stack(features),
        'requirements': torch.stack(requirements),
        'attention_mask': torch.stack(attention_mask),
    }


def preprocess_dataset(
    dataset_name: str,
    tokenizer,
    extractor,
    aligner,
    max_length: int,
    cache_dir: str,
    split: str = 'train',
) -> ASADataset:
    """Preprocess dataset with ASA properties."""
    from datasets import load_dataset
    from tqdm import tqdm
    
    cache_path = Path(cache_dir) / f"{dataset_name}_{split}_asa.pt"
    
    if cache_path.exists():
        print(f"Loading cached {split} data from {cache_path}")
        return ASADataset.load(str(cache_path))
    
    print(f"Preprocessing {dataset_name} {split} split...")
    
    # Load dataset
    if dataset_name == 'wikitext-2':
        ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        texts = [t for t in ds['text'] if t.strip()]
    elif dataset_name == 'wikitext-103':
        ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        texts = [t for t in ds['text'] if t.strip()]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Process texts
    data = []
    skipped = 0
    
    for text in tqdm(texts, desc=f"Processing {split}"):
        try:
            # Skip very short texts
            if len(text.split()) < 5:
                skipped += 1
                continue
            
            # Extract properties
            props = extractor.extract(text)
            
            if len(props) < 2:
                skipped += 1
                continue
            
            # Align to subwords
            result = aligner.align(text, props, max_length)
            
            if len(result.input_ids) < 2:
                skipped += 1
                continue
            
            data.append({
                'input_ids': result.input_ids,
                'pos_ids': result.pos_ids,
                'features': result.features,
                'requirements': result.requirements,
            })
            
        except Exception as e:
            skipped += 1
            continue
    
    print(f"Processed {len(data)} examples, skipped {skipped}")
    
    # Save cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = ASADataset(data)
    dataset.save(str(cache_path))
    
    return dataset


# =============================================================================
# TRAINING
# =============================================================================

class Trainer:
    """ASA Trainer."""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        total_steps = len(train_loader) * config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
        )
        
        # State
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Output dir
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward
            result = self.model(
                input_ids=batch['input_ids'],
                pos_ids=batch['pos_ids'],
                features=batch['features'],
                requirements=batch['requirements'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids'],
            )
            
            loss = result['loss']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
            
            # Update
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            if self.global_step % self.config.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.scheduler.get_last_lr()[0]
                ppl = torch.exp(torch.tensor(avg_loss)).item()
                print(f"Step {self.global_step} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}")
            
            # Evaluation
            if self.global_step % self.config.eval_interval == 0:
                val_loss, val_ppl = self.evaluate()
                print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best.pt')
                
                self.model.train()
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(f'step_{self.global_step}.pt')
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            result = self.model(
                input_ids=batch['input_ids'],
                pos_ids=batch['pos_ids'],
                features=batch['features'],
                requirements=batch['requirements'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids'],
            )
            
            total_loss += result['loss'].item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, ppl
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config),
        }, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from {path}")
    
    def train(self):
        """Full training loop."""
        print(f"Training {self.config.mode} model ({self.config.size})")
        print(f"  α = {self.config.alpha}")
        print(f"  hard_block = {self.config.hard_block}")
        print(f"  epochs = {self.config.epochs}")
        print(f"  batch_size = {self.config.batch_size}")
        print(f"  learning_rate = {self.config.learning_rate}")
        print()
        
        start_time = time.time()
        
        for epoch in range(self.config.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.epochs}")
            print(f"{'='*50}")
            
            train_loss = self.train_epoch(epoch)
            val_loss, val_ppl = self.evaluate()
            
            print(f"\nEpoch {epoch + 1} complete:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val PPL: {val_ppl:.2f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'epoch_{epoch + 1}.pt')
        
        elapsed = time.time() - start_time
        print(f"\nTraining complete in {elapsed/60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Final evaluation
        final_loss, final_ppl = self.evaluate()
        print(f"Final validation PPL: {final_ppl:.2f}")
        
        return final_ppl


# =============================================================================
# MAIN
# =============================================================================

def cmd_preprocess(args):
    """Preprocess dataset."""
    from transformers import AutoTokenizer
    
    # Import ASA
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from asa_v2_2 import PropertyExtractor, SubwordAligner
    
    print(f"Preprocessing {args.dataset}...")
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    extractor = PropertyExtractor()
    aligner = SubwordAligner(tokenizer)
    
    # Process each split
    for split in ['train', 'validation', 'test']:
        print(f"\nProcessing {split}...")
        preprocess_dataset(
            args.dataset,
            tokenizer,
            extractor,
            aligner,
            args.max_length,
            args.cache_dir,
            split,
        )
    
    print(f"\nPreprocessing complete. Cached in {args.cache_dir}")


def cmd_train(args):
    """Train model."""
    from transformers import AutoTokenizer
    
    # Import ASA
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from asa_v2_2 import create_model, count_parameters
    
    # Config
    config = TrainConfig(
        size=args.size,
        mode=args.mode,
        alpha=args.alpha,
        hard_block=args.hard_block,
        dataset=args.dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
    )
    
    # Update output dir with mode
    config.output_dir = os.path.join(args.output_dir, f"{args.mode}_{args.size}")
    
    # Device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load cached data
    train_data = ASADataset.load(os.path.join(args.cache_dir, f"{args.dataset}_train_asa.pt"))
    val_data = ASADataset.load(os.path.join(args.cache_dir, f"{args.dataset}_validation_asa.pt"))
    
    print(f"Train: {len(train_data)} examples")
    print(f"Val: {len(val_data)} examples")
    
    # DataLoaders
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    # Model
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        size=config.size,
        mode=config.mode,
        alpha=config.alpha,
        hard_block=config.hard_block,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    print(f"\nModel: {config.mode} ({config.size})")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Save config
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    config.save(os.path.join(config.output_dir, 'config.json'))
    
    # Train
    trainer = Trainer(model, config, train_loader, val_loader, device)
    final_ppl = trainer.train()
    
    return final_ppl


def cmd_evaluate(args):
    """Evaluate and compare checkpoints."""
    from transformers import AutoTokenizer
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from asa_v2_2 import create_model
    
    # Device (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load test data
    test_data = ASADataset.load(os.path.join(args.cache_dir, f"{args.dataset}_test_asa.pt"))
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    print(f"Evaluating on {len(test_data)} test examples\n")
    
    results = []
    
    for checkpoint_path in args.checkpoints:
        print(f"Loading {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = TrainConfig(**checkpoint['config'])
        
        model = create_model(
            vocab_size=tokenizer.vocab_size,
            size=config.size,
            mode=config.mode,
            alpha=config.alpha,
            hard_block=config.hard_block,
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        # Evaluate
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                result = model(
                    input_ids=batch['input_ids'],
                    pos_ids=batch['pos_ids'],
                    features=batch['features'],
                    requirements=batch['requirements'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['input_ids'],
                )
                total_loss += result['loss'].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        ppl = torch.exp(torch.tensor(avg_loss)).item()
        
        results.append({
            'checkpoint': checkpoint_path,
            'mode': config.mode,
            'size': config.size,
            'alpha': config.alpha,
            'test_loss': avg_loss,
            'test_ppl': ppl,
        })
        
        print(f"  Mode: {config.mode}, Size: {config.size}")
        print(f"  Test Loss: {avg_loss:.4f}")
        print(f"  Test PPL: {ppl:.2f}")
        print()
    
    # Comparison
    if len(results) > 1:
        print("\n" + "=" * 50)
        print("COMPARISON")
        print("=" * 50)
        print(f"{'Mode':<15} {'Size':<10} {'α':<6} {'PPL':<10}")
        print("-" * 45)
        for r in sorted(results, key=lambda x: x['test_ppl']):
            print(f"{r['mode']:<15} {r['size']:<10} {r['alpha']:<6.1f} {r['test_ppl']:<10.2f}")
        
        best = min(results, key=lambda x: x['test_ppl'])
        print(f"\nBest: {best['mode']} with PPL = {best['test_ppl']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='ASA Training')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Preprocess command
    p_preprocess = subparsers.add_parser('preprocess', help='Preprocess dataset')
    p_preprocess.add_argument('--dataset', default='wikitext-2', choices=['wikitext-2', 'wikitext-103'])
    p_preprocess.add_argument('--tokenizer', default='gpt2')
    p_preprocess.add_argument('--max-length', type=int, default=256)
    p_preprocess.add_argument('--cache-dir', default='./asa_cache')
    
    # Train command
    p_train = subparsers.add_parser('train', help='Train model')
    p_train.add_argument('--mode', default='full', choices=['full', 'pos_only', 'features_only', 'none'])
    p_train.add_argument('--size', default='small', choices=['tiny', 'small', 'medium', 'large'])
    p_train.add_argument('--alpha', type=float, default=1.0)
    p_train.add_argument('--hard-block', action='store_true', default=True)
    p_train.add_argument('--soft-block', action='store_true')
    p_train.add_argument('--dataset', default='wikitext-2')
    p_train.add_argument('--tokenizer', default='gpt2')
    p_train.add_argument('--max-length', type=int, default=256)
    p_train.add_argument('--batch-size', type=int, default=16)
    p_train.add_argument('--lr', type=float, default=3e-4)
    p_train.add_argument('--epochs', type=int, default=10)
    p_train.add_argument('--output-dir', default='./asa_output')
    p_train.add_argument('--cache-dir', default='./asa_cache')
    
    # Evaluate command
    p_eval = subparsers.add_parser('evaluate', help='Evaluate checkpoints')
    p_eval.add_argument('--checkpoints', nargs='+', required=True, help='Checkpoint paths')
    p_eval.add_argument('--dataset', default='wikitext-2')
    p_eval.add_argument('--tokenizer', default='gpt2')
    p_eval.add_argument('--batch-size', type=int, default=16)
    p_eval.add_argument('--cache-dir', default='./asa_cache')
    
    args = parser.parse_args()
    
    # Handle soft-block flag
    if hasattr(args, 'soft_block') and args.soft_block:
        args.hard_block = False
    
    if args.command == 'preprocess':
        cmd_preprocess(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)


if __name__ == '__main__':
    main()
