# Created by: Rami Abu Mukh, Omar Gharra, Ameer Khalaily

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

from cnn_policy.model import ChessCNNPolicy
from cnn_policy.dataset import ChessPolicyDataset
from cnn_policy.dataset_h5_proper import ChessPolicyDatasetH5Proper
from cnn_policy.config import TrainingConfig as Config

# chatgpt generated - vaswani learning rate schedule
def get_vaswani_lr(step: int, d_model: int, warmup_steps: int) -> float:
    step = max(step, 1)  # Avoid division by zero
    
    lr = (d_model ** -0.5) * min(
        step ** -0.5,
        step * (warmup_steps ** -1.5)
    )
    
    return lr

# alphazero label smoothing
class LabelSmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # Create smooth labels
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(self.smoothing / (num_classes - 1))
        smooth_labels.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        # KL divergence
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        
        return loss


class Trainer:
    def __init__(self, config):
        self.config = config
        
        # Create directories
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device(config.DEVICE)
        print(f"\nUsing device: {self.device}")
        
        # Model
        print("\nInitializing model...")
        self.model = ChessCNNPolicy(
            num_input_channels=config.NUM_INPUT_CHANNELS,
            num_filters=config.NUM_FILTERS,
            num_blocks=config.NUM_BLOCKS,
            dropout_rate=config.DROPOUT_RATE
        ).to(self.device)

        self.criterion = LabelSmoothedCrossEntropyLoss(smoothing=config.LABEL_SMOOTHING)
        

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=1e-7,  # Will be overridden by scheduler
            betas=config.BETAS,
            eps=config.EPSILON,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Mixed Precision for H100 optimization
        self.use_amp = getattr(config, 'USE_MIXED_PRECISION', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Data loaders
        print("\n📂 Loading datasets...")
        
        h5_path = config.DATA_PATH / 'raw/LE22ct/LE22ct.h5'
        
        if h5_path.exists():
            self.train_dataset = ChessPolicyDatasetH5Proper(str(h5_path), split='train', augment=False)
            self.val_dataset = ChessPolicyDatasetH5Proper(str(h5_path), split='val', augment=False)
        else:
            raise FileNotFoundError(f"No dataset found!\n")
        
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
        
        # Early stopping
        self.patience_counter = 0
        self.best_step = 0
        
        # Timing
        self.start_time = time.time()
        
        # Checkpoint list for averaging
        self.recent_checkpoints = []
    
    # train for one step  - returns the average loss for from, to and combined loss
    def train_step(self) -> Tuple[float, float, float]:
        self.model.train()
        
        total_from_loss = 0.0
        total_to_loss = 0.0
        total_combined_loss = 0.0
        
        # Gradient accumulation
        self.optimizer.zero_grad()
        
        for acc_step in range(Config.GRADIENT_ACCUMULATION_STEPS):
            try:
                positions, (from_targets, to_targets) = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                positions, (from_targets, to_targets) = next(self.train_iter)
            
            # Move to device
            positions = positions.to(self.device)
            from_targets = from_targets.to(self.device)
            to_targets = to_targets.to(self.device)
            
            # Forward pass with mixed precision - got warning fix later
            with autocast(enabled=self.use_amp, dtype=torch.bfloat16):
                from_logits, to_logits = self.model(positions)
                
                # Compute losses
                from_loss = self.criterion(from_logits, from_targets)
                to_loss = self.criterion(to_logits, to_targets)
                loss = (from_loss + to_loss) / Config.GRADIENT_ACCUMULATION_STEPS
            
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
            total_from_loss / Config.GRADIENT_ACCUMULATION_STEPS,
            total_to_loss / Config.GRADIENT_ACCUMULATION_STEPS,
            total_combined_loss / Config.GRADIENT_ACCUMULATION_STEPS
        )
    
    def validate(self) -> Tuple[float, float, float]:
        self.model.eval()
        
        total_from_correct = 0
        total_to_correct = 0
        total_both_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for positions, (from_targets, to_targets) in tqdm(self.val_loader, desc="Validation", leave=False):
                positions = positions.to(self.device)
                from_targets = from_targets.to(self.device)
                to_targets = to_targets.to(self.device)
                
                # Forward pass
                from_logits, to_logits = self.model(positions)
                
                # Predictions
                from_preds = from_logits.argmax(dim=1)
                to_preds = to_logits.argmax(dim=1)
                
                # Accuracy
                total_from_correct += (from_preds == from_targets).sum().item()
                total_to_correct += (to_preds == to_targets).sum().item()
                total_both_correct += ((from_preds == from_targets) & (to_preds == to_targets)).sum().item()
                total_samples += positions.size(0)
        
        from_accuracy = total_from_correct / total_samples
        to_accuracy = total_to_correct / total_samples
        combined_accuracy = total_both_correct / total_samples  # Full move correct
        
        return from_accuracy, to_accuracy, combined_accuracy
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'training_history': self.training_history
        }
        
        path = Config.CHECKPOINT_DIR / filename
        torch.save(checkpoint, path)
        
        # Track for averaging
        self.recent_checkpoints.append(path)
        if len(self.recent_checkpoints) > Config.CHECKPOINT_AVERAGE_LAST_N:
            self.recent_checkpoints.pop(0)
    
    def average_checkpoints(self):
        
        state_dicts = []
        for ckpt_path in self.recent_checkpoints:
            ckpt = torch.load(ckpt_path)
            state_dicts.append(ckpt['model_state_dict'])
        
        # Average weights
        averaged_state_dict = {}
        for key in state_dicts[0].keys():
            averaged_state_dict[key] = torch.stack([sd[key].float() for sd in state_dicts]).mean(dim=0)
        
        # Save averaged model
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
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"\nTotal steps: {Config.TOTAL_STEPS:,}")
        print()
        
        # Create infinite iterator
        self.train_iter = iter(self.train_loader)
        
         # training loop - prints progress bar and logs to tensorboard from chatgpt
        with tqdm(total=Config.TOTAL_STEPS, desc="Training") as pbar:
            for step in range(1, Config.TOTAL_STEPS + 1):
                self.current_step = step
                
                # Update learning rate (Vaswani schedule)
                current_lr = get_vaswani_lr(step, Config.D_MODEL, Config.WARMUP_STEPS)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # Train step
                from_loss, to_loss, combined_loss = self.train_step()
                
                # Update progress
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{combined_loss:.3f}',
                    'lr': f'{current_lr:.6f}'
                })
                
                # Log to TensorBoard
                self.writer.add_scalar('Loss/from', from_loss, step)
                self.writer.add_scalar('Loss/to', to_loss, step)
                self.writer.add_scalar('Loss/combined', combined_loss, step)
                self.writer.add_scalar('LR', current_lr, step)
                
                # Validate periodically
                if step % 5000 == 0 or step == Config.TOTAL_STEPS:
                    from_acc, to_acc, move_acc = self.validate()
                    
                    print(f"\nStep {step:,}/{Config.TOTAL_STEPS:,}:")
                    print(f"   From accuracy:  {from_acc:.2%}")
                    print(f"   To accuracy:    {to_acc:.2%}")
                    print(f"   Move accuracy:  {move_acc:.2%} (both correct)")
                    print(f"   LR:             {current_lr:.6f}")
                    
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
                        'lr': current_lr
                    })
                    
                    # Save CSV log
                    df = pd.DataFrame(self.training_history)
                    df.to_csv(Config.LOG_DIR / 'training_log.csv', index=False)
                    
                    # Check if best
                    if move_acc > self.best_val_accuracy + Config.EARLY_STOPPING_MIN_DELTA:
                        self.best_val_accuracy = move_acc
                        self.best_step = step
                        self.patience_counter = 0
                        print(f"   New best accuracy: {move_acc:.2%}")
                    else:
                        self.patience_counter += 1
                        print(f"   No improvement (patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE})")
                    
                    # Early stopping check
                    if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                        print(f"\n  Early stopping triggered!")
                        print(f"   No improvement for {Config.EARLY_STOPPING_PATIENCE} validations")
                        print(f"   Best accuracy: {self.best_val_accuracy:.2%} at step {self.best_step:,}")
                        
                    
                    # Save checkpoint
                    self.save_checkpoint(f'checkpoint_step_{step}.pth')
                    
                    print()
        


def main():
    h5_path = Config.DATA_PATH / 'raw/LE22ct/LE22ct.h5'
    if h5_path.exists():
        print(f"\n Found H5 dataset: {h5_path}")
    trainer = Trainer(Config)
    trainer.train()


if __name__ == "__main__":
    main()

