"""
Transformer Policy Training Script
===================================
IDENTICAL training procedure to CT-EFT-20.

Training:
- Step-based (325K steps, 50 epochs)
- Gradient accumulation (effective batch 2048)
- Vaswani LR schedule
- Label-smoothed cross-entropy
- Mixed precision (BF16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import time
import sys
import math
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from transformer_policy.model import ChessTransformerEncoderFT
from transformer_policy.dataset import ChessDatasetFT
from transformer_policy.config import TransformerConfig as Config


def get_vaswani_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """
    Vaswani learning rate schedule (IDENTICAL to CT-EFT-20).
    
    LR = d_model^(-0.5) × min(step^(-0.5), step × warmup^(-1.5))
    """
    step = max(step, 1)
    lr = (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    return lr


class LabelSmoothedCE(nn.Module):
    """
    Label-Smoothed Cross Entropy Loss.
    IDENTICAL to CT-EFT-20's loss function.
    """
    
    def __init__(self, eps, n_predictions=1):
        super().__init__()
        self.eps = eps
        self.n_predictions = n_predictions
        if n_predictions > 1:
            self.indices = torch.arange(n_predictions).unsqueeze(0).to(Config.DEVICE)
            self.indices.requires_grad = False
    
    def forward(self, predicted, targets, lengths=None):
        """
        Forward prop - IDENTICAL to CT-EFT-20.
        
        Args:
            predicted: (N, n_predictions, vocab_size) or (N, vocab_size)
            targets: (N, n_predictions) or (N,)
            lengths: (N, 1) for masking (optional)
        """
        # Handle single prediction case
        if len(predicted.shape) == 2:
            # (N, vocab_size) - single prediction
            predicted = predicted.unsqueeze(1)  # (N, 1, vocab_size)
            targets = targets.unsqueeze(1) if len(targets.shape) == 1 else targets  # (N, 1)
        
        # Remove pad-positions if needed
        if self.n_predictions > 1 and lengths is not None:
            predicted = predicted[self.indices < lengths]
            targets = targets[self.indices < lengths]
        else:
            # Flatten for single prediction
            N, n_pred, vocab_size = predicted.shape
            predicted = predicted.view(-1, vocab_size)
            targets = targets.view(-1)
        
        # Create smoothed one-hot targets
        target_vector = torch.zeros_like(predicted).scatter(
            dim=1, index=targets.unsqueeze(1), value=1.0
        )
        target_vector = target_vector * (1.0 - self.eps) + self.eps / target_vector.size(1)
        
        # Compute loss
        loss = (-1 * target_vector * F.log_softmax(predicted, dim=1)).sum(dim=1)
        return torch.mean(loss)


class Trainer:
    """Trainer for Transformer policy network."""
    
    def __init__(self, config):
        self.config = config
        
        # Create directories
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device(config.DEVICE)
        print(f"\n🖥️  Using device: {self.device}")
        
        # Model
        print("\n📦 Initializing Transformer model...")
        self.model = ChessTransformerEncoderFT(config).to(self.device)
        
        # Count parameters
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"   Parameters: {param_count:,}")
        
        # Loss function (IDENTICAL to CT-EFT-20)
        self.criterion_from = LabelSmoothedCE(eps=config.LABEL_SMOOTHING, n_predictions=1)
        self.criterion_to = LabelSmoothedCE(eps=config.LABEL_SMOOTHING, n_predictions=1)
        
        # Optimizer (IDENTICAL to CT-EFT-20)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-7,
            betas=config.BETAS,
            eps=config.EPSILON
        )
        
        # Mixed Precision
        self.use_amp = getattr(config, 'USE_AMP', False)
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print(f"   🚀 Mixed Precision: Enabled (BF16)")
        
        # Data loaders
        print("\n📂 Loading datasets...")
        h5_path = config.DATA_PATH / 'raw/LE22ct/LE22ct.h5'
        
        if not h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        
        self.train_dataset = ChessDatasetFT(
            data_folder=str(config.DATA_PATH),
            h5_file='raw/LE22ct/LE22ct.h5',
            split='train'
        )
        self.val_dataset = ChessDatasetFT(
            data_folder=str(config.DATA_PATH),
            h5_file='raw/LE22ct/LE22ct.h5',
            split='val'
        )
        
        # Configure DataLoader (matching CNN setup)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        print(f"   Train batches: {len(self.train_loader):,}")
        print(f"   Val batches:   {len(self.val_loader):,}")
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(config.LOG_DIR / 'tensorboard'))
        
        # Training state
        self.current_step = 0
        self.best_val_accuracy = 0.0
        self.training_history = []
        self.patience_counter = 0
        self.best_step = 0
        self.start_time = time.time()
        self.recent_checkpoints = []
    
    def train_step(self) -> Tuple[float, float, float]:
        """
        Train for one step with gradient accumulation.
        IDENTICAL procedure to CT-EFT-20!
        """
        self.model.train()
        
        total_from_loss = 0.0
        total_to_loss = 0.0
        total_combined_loss = 0.0
        
        self.optimizer.zero_grad()
        
        for acc_step in range(Config.BATCHES_PER_STEP):
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)
            
            # Move to device
            for key in batch:
                batch[key] = batch[key].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp, dtype=torch.bfloat16):
                from_logits, to_logits = self.model(batch)
                
                # from_logits: (N, 1, 64), to_logits: (N, 1, 64)
                # Squeeze to (N, 64)
                from_logits = from_logits.squeeze(1)
                to_logits = to_logits.squeeze(1)
                
                # Compute losses
                from_loss = self.criterion_from(
                    from_logits, batch["from_squares"].squeeze(1), batch["lengths"]
                )
                to_loss = self.criterion_to(
                    to_logits, batch["to_squares"].squeeze(1), batch["lengths"]
                )
                loss = (from_loss + to_loss) / Config.BATCHES_PER_STEP
            
            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            total_from_loss += from_loss.item()
            total_to_loss += to_loss.item()
            total_combined_loss += (from_loss.item() + to_loss.item())
        
        # Update weights
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        return (
            total_from_loss / Config.BATCHES_PER_STEP,
            total_to_loss / Config.BATCHES_PER_STEP,
            total_combined_loss / Config.BATCHES_PER_STEP
        )
    
    def validate(self) -> Tuple[float, float, float]:
        """Validate model."""
        self.model.eval()
        
        total_from_correct = 0
        total_to_correct = 0
        total_both_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move to device
                for key in batch:
                    batch[key] = batch[key].to(self.device)
                
                # Forward pass
                from_logits, to_logits = self.model(batch)
                from_logits = from_logits.squeeze(1)  # (N, 64)
                to_logits = to_logits.squeeze(1)      # (N, 64)
                
                # Predictions
                from_preds = from_logits.argmax(dim=1)
                to_preds = to_logits.argmax(dim=1)
                
                # Targets
                from_targets = batch["from_squares"].squeeze(1)
                to_targets = batch["to_squares"].squeeze(1)
                
                # Accuracy
                total_from_correct += (from_preds == from_targets).sum().item()
                total_to_correct += (to_preds == to_targets).sum().item()
                total_both_correct += ((from_preds == from_targets) & (to_preds == to_targets)).sum().item()
                total_samples += from_targets.size(0)
        
        from_accuracy = total_from_correct / total_samples
        to_accuracy = total_to_correct / total_samples
        combined_accuracy = total_both_correct / total_samples
        
        return from_accuracy, to_accuracy, combined_accuracy
    
    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        checkpoint = {
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'training_history': self.training_history
        }
        
        path = Config.CHECKPOINT_DIR / filename
        torch.save(checkpoint, path)
        
        self.recent_checkpoints.append(path)
        if len(self.recent_checkpoints) > Config.CHECKPOINT_AVERAGE_LAST_N:
            self.recent_checkpoints.pop(0)
    
    def average_checkpoints(self):
        """Average last N checkpoints (like CT-EFT-20)."""
        print(f"\n📊 Averaging last {len(self.recent_checkpoints)} checkpoints...")
        
        if not self.recent_checkpoints:
            return
        
        state_dicts = []
        for ckpt_path in self.recent_checkpoints:
            ckpt = torch.load(ckpt_path)
            state_dicts.append(ckpt['model_state_dict'])
        
        averaged_state_dict = {}
        for key in state_dicts[0].keys():
            averaged_state_dict[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
        
        final_checkpoint = {
            'step': self.current_step,
            'model_state_dict': averaged_state_dict,
            'best_val_accuracy': self.best_val_accuracy,
            'note': f'Averaged from last {len(self.recent_checkpoints)} checkpoints'
        }
        
        final_path = Config.CHECKPOINT_DIR / 'averaged_final.pth'
        torch.save(final_checkpoint, final_path)
        print(f"✅ Saved averaged model: {final_path}")
    
    def train(self):
        """Main training loop (IDENTICAL procedure to CT-EFT-20)."""
        print("\n" + "=" * 70)
        print("🚀 Starting Transformer Training (CT-EFT-20 Replica)")
        print("=" * 70)
        print(f"\nTotal steps: {Config.N_STEPS:,}")
        print(f"Effective batch: {Config.BATCH_SIZE * Config.BATCHES_PER_STEP}")
        print()
        
        self.train_iter = iter(self.train_loader)
        
        # For tracking steps/sec
        step_times = []
        last_log_time = time.time()
        
        try:
            with tqdm(total=Config.N_STEPS, desc="Training") as pbar:
                for step in range(1, Config.N_STEPS + 1):
                    step_start = time.time()
                    self.current_step = step
                    
                    # Update learning rate (Vaswani schedule)
                    current_lr = get_vaswani_lr(step, Config.D_MODEL, Config.WARMUP_STEPS)
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = current_lr
                    
                    # Train step
                    from_loss, to_loss, combined_loss = self.train_step()
                    
                    # Track step time
                    step_time = time.time() - step_start
                    step_times.append(step_time)
                    if len(step_times) > 100:
                        step_times.pop(0)
                    
                    # Update progress
                    pbar.update(1)
                    avg_step_time = sum(step_times) / len(step_times)
                    steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
                    pbar.set_postfix({
                        'loss': f'{combined_loss:.3f}',
                        'lr': f'{current_lr:.6f}',
                        'step/s': f'{steps_per_sec:.2f}'
                    })
                    
                    # Log to TensorBoard
                    self.writer.add_scalar('Loss/from', from_loss, step)
                    self.writer.add_scalar('Loss/to', to_loss, step)
                    self.writer.add_scalar('Loss/combined', combined_loss, step)
                    self.writer.add_scalar('LR', current_lr, step)
                    
                    # Periodic updates every 1000 steps
                    if step % 1000 == 0:
                        elapsed = time.time() - self.start_time
                        eta = (Config.N_STEPS - step) * avg_step_time
                        elapsed_h, elapsed_m = int(elapsed // 3600), int((elapsed % 3600) // 60)
                        eta_h, eta_m = int(eta // 3600), int((eta % 3600) // 60)
                        
                        # Get GPU memory
                        gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                        
                        print(f"\n⏱️  Step {step:,}/{Config.N_STEPS:,} ({100*step/Config.N_STEPS:.1f}%)")
                        print(f"   Loss: {combined_loss:.4f} | LR: {current_lr:.6f} | Speed: {steps_per_sec:.2f} steps/s")
                        print(f"   Time: {elapsed_h}h{elapsed_m:02d}m | ETA: {eta_h}h{eta_m:02d}m | GPU: {gpu_mem:.1f}GB")
                    
                    # Validate periodically
                    if step % 5000 == 0 or step == Config.N_STEPS:
                        from_acc, to_acc, move_acc = self.validate()
                        
                        # Calculate time info
                        elapsed = time.time() - self.start_time
                        eta = (Config.N_STEPS - step) * avg_step_time if step < Config.N_STEPS else 0
                        elapsed_h, elapsed_m = int(elapsed // 3600), int((elapsed % 3600) // 60)
                        eta_h, eta_m = int(eta // 3600), int((eta % 3600) // 60)
                        
                        print(f"\n{'='*70}")
                        print(f"📊 VALIDATION - Step {step:,}/{Config.N_STEPS:,} ({100*step/Config.N_STEPS:.1f}%)")
                        print(f"{'='*70}")
                        print(f"   From accuracy:  {from_acc:.4f} ({from_acc*100:.2f}%)")
                        print(f"   To accuracy:    {to_acc:.4f} ({to_acc*100:.2f}%)")
                        print(f"   Move accuracy:  {move_acc:.4f} ({move_acc*100:.2f}%) ⭐ PRIMARY METRIC")
                        print(f"   Combined loss:  {combined_loss:.4f}")
                        print(f"   Learning rate:  {current_lr:.6f}")
                        print(f"   Time elapsed:   {elapsed_h}h {elapsed_m}m")
                        print(f"   ETA:            {eta_h}h {eta_m}m")
                        print(f"   Speed:          {steps_per_sec:.2f} steps/sec")
                        
                        # Check against CNN baseline
                        cnn_baseline = 0.5081
                        if move_acc > cnn_baseline:
                            diff = (move_acc - cnn_baseline) * 100
                            print(f"   🎉 BEATS CNN by {diff:.2f}%!")
                        else:
                            diff = (cnn_baseline - move_acc) * 100
                            print(f"   📊 CNN leads by {diff:.2f}%")
                        
                        # Log validation
                        self.writer.add_scalar('Accuracy/from', from_acc, step)
                        self.writer.add_scalar('Accuracy/to', to_acc, step)
                        self.writer.add_scalar('Accuracy/move', move_acc, step)
                        
                        # Save history
                        self.training_history.append({
                            'step': step,
                            'from_acc': from_acc,
                            'to_acc': to_acc,
                            'move_acc': move_acc,
                            'lr': current_lr,
                            'elapsed_hours': elapsed / 3600
                        })
                        
                        # Save CSV log
                        df = pd.DataFrame(self.training_history)
                        df.to_csv(Config.LOG_DIR / 'training_log.csv', index=False)
                        
                        # Check if best
                        if move_acc > self.best_val_accuracy + Config.EARLY_STOPPING_MIN_DELTA:
                            self.best_val_accuracy = move_acc
                            self.best_step = step
                            self.patience_counter = 0
                            print(f"   🎯 NEW BEST ACCURACY: {move_acc*100:.2f}%")
                        else:
                            self.patience_counter += 1
                            print(f"   No improvement (patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE})")
                        
                        # Early stopping check
                        if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                            print(f"\n   ⚠️  Early stopping triggered!")
                            print(f"   No improvement for {Config.EARLY_STOPPING_PATIENCE} validations")
                            print(f"   Best accuracy: {self.best_val_accuracy*100:.2f}% at step {self.best_step:,}")
                            print(f"   Continuing training... (stop with Ctrl+C if desired)")
                        
                        # Save checkpoint
                        print(f"\n   💾 Saving checkpoint...")
                        self.save_checkpoint(f'checkpoint_step_{step}.pth')
                        print(f"   ✅ Checkpoint saved: checkpoint_step_{step}.pth")
                        print(f"{'='*70}\n")
        
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted!")
        
        finally:
            self.finalize()
    
    def finalize(self):
        """Finalize training."""
        print("\n" + "=" * 70)
        print("🏁 Training Complete!")
        print("=" * 70)
        
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print(f"\n📊 Training Summary:")
        print(f"   Total steps:        {self.current_step:,}")
        print(f"   Best move accuracy: {self.best_val_accuracy:.2%}")
        print(f"   Total time:         {hours}h {minutes}m")
        
        # Average checkpoints
        self.average_checkpoints()
        
        # Close TensorBoard
        self.writer.close()
        
        print(f"\n💾 Final model: {Config.CHECKPOINT_DIR / 'averaged_final.pth'}")
        print("\n" + "=" * 70)


def main():
    """Main function."""
    Config.print_config()
    
    # Check data
    h5_path = Config.DATA_PATH / 'raw/LE22ct/LE22ct.h5'
    if not h5_path.exists():
        print(f"❌ H5 file not found: {h5_path}")
        print("\n   Download with:")
        print("   python scripts/download_le22ct.py")
        sys.exit(1)
    
    print(f"\n✅ Found H5 dataset: {h5_path}")
    
    # Create trainer
    trainer = Trainer(Config)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()


