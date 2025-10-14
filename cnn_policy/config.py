"""
CNN Policy Training Configuration
==================================
Matches CT-EFT-20 training protocol for fair comparison.
"""

import torch
from pathlib import Path


class TrainingConfig:
    """Training configuration matching CT-EFT-20."""
    
    # Model Architecture
    NUM_INPUT_CHANNELS = 18
    NUM_FILTERS = 256
    NUM_BLOCKS = 15
    DROPOUT_RATE = 0.1
    
    # Training Parameters (OPTIMIZED FOR H100)
    BATCH_SIZE = 2048                   # Optimized: 4x larger (was 512)
    GRADIENT_ACCUMULATION_STEPS = 1     # Optimized: No accumulation needed! (was 4)
    TOTAL_STEPS = 325_000               # Extended: ~50 epochs (was 100K for 15 epochs)
    WARMUP_STEPS = 8_000                # Match CT-EFT-20
    # Note: Effective batch still 2048 (2048×1 = 512×4), same learning!
    
    # Learning Rate
    LR_SCHEDULE = 'vaswani'             # Match CT-EFT-20
    D_MODEL = 256                       # For LR calculation
    BASE_LR = None                      # Calculated by Vaswani schedule
    
    # Optimizer (MATCH CT-EFT-20!)
    OPTIMIZER = 'Adam'
    BETAS = (0.9, 0.98)                 # Match CT-EFT-20 exactly!
    EPSILON = 1e-9                      # Match CT-EFT-20
    WEIGHT_DECAY = 0.0                  # CT-EFT-20 uses 0
    
    # Loss Function
    LABEL_SMOOTHING = 0.1               # Match CT-EFT-20
    
    # Data
    TRAIN_SPLIT = 0.9                   # Estimate (check CT-EFT-20's actual split)
    VAL_SPLIT = 0.1
    NUM_WORKERS = 8                     # Optimized: More workers for faster loading (was 4)
    PIN_MEMORY = True
    
    # H100 Optimizations
    USE_MIXED_PRECISION = True          # BF16 for 2x speedup on H100
    PRECISION = 'bf16'                  # BF16 better than FP16 for training
    
    # Checkpointing (EXTENDED)
    SAVE_EVERY_N_STEPS = 5_000          # Save every 5K steps
    CHECKPOINT_AVERAGE_LAST_N = 10      # Average last 10 checkpoints (like CT-EFT-20)
    
    # Early Stopping (Optional - can stop if converged)
    EARLY_STOPPING_PATIENCE = 10        # Stop if no improvement for 10 validations (50K steps)
    EARLY_STOPPING_MIN_DELTA = 0.001    # Minimum improvement threshold (0.1%)
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Paths
    DATA_PATH = Path('dataset')
    H5_PATH = Path('dataset/raw/LE22ct/LE22ct.h5')  # Direct H5 path
    CSV_PATH = Path('dataset/processed')             # CSV fallback
    CHECKPOINT_DIR = Path('cnn_policy/checkpoints')
    LOG_DIR = Path('cnn_policy/logs')
    
    @classmethod
    def print_config(cls):
        """Print configuration."""
        print("\n" + "=" * 70)
        print("CNN Policy Training Configuration")
        print("(Matching CT-EFT-20 Protocol)")
        print("=" * 70)
        print("\n📊 Model Architecture:")
        print(f"   Input channels:      {cls.NUM_INPUT_CHANNELS}")
        print(f"   Filters:             {cls.NUM_FILTERS}")
        print(f"   Residual blocks:     {cls.NUM_BLOCKS}")
        print(f"   Dropout rate:        {cls.DROPOUT_RATE}")
        
        print("\n🎯 Training Parameters (CT-EFT-20 Matched):")
        print(f"   Batch size:          {cls.BATCH_SIZE}")
        print(f"   Grad accumulation:   {cls.GRADIENT_ACCUMULATION_STEPS}")
        print(f"   Effective batch:     {cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION_STEPS}")
        print(f"   Total steps:         {cls.TOTAL_STEPS:,}")
        print(f"   Warmup steps:        {cls.WARMUP_STEPS:,}")
        
        print("\n📉 Learning Rate:")
        print(f"   Schedule:            {cls.LR_SCHEDULE} (like CT-EFT-20)")
        print(f"   Warmup:              Linear over {cls.WARMUP_STEPS:,} steps")
        print(f"   Peak LR:             ~{(cls.D_MODEL ** -0.5):.6f}")
        
        print("\n⚙️  Optimizer (CT-EFT-20 Matched):")
        print(f"   Type:                {cls.OPTIMIZER}")
        print(f"   Betas:               {cls.BETAS}")
        print(f"   Epsilon:             {cls.EPSILON}")
        print(f"   Weight decay:        {cls.WEIGHT_DECAY}")
        
        print("\n💾 Data:")
        print(f"   Dataset:             LE22ct (13.3M positions)")
        print(f"   Train/Val split:     {cls.TRAIN_SPLIT}/{cls.VAL_SPLIT}")
        print(f"   Label smoothing:     {cls.LABEL_SMOOTHING}")
        
        print("\n🖥️  Hardware:")
        print(f"   Device:              {cls.DEVICE}")
        
        print("\n" + "=" * 70)
        
        # Calculate expected epochs
        dataset_size = 13_287_522  # LE22ct size
        total_samples = cls.TOTAL_STEPS * cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION_STEPS
        epochs = total_samples / dataset_size
        
        print(f"\n📊 Training Scope:")
        print(f"   Total samples:       {total_samples:,}")
        print(f"   Expected epochs:     ~{epochs:.1f}")
        
        print(f"\n⚡ H100 Optimizations:")
        print(f"   Batch size:          {cls.BATCH_SIZE} (4x larger, no grad accumulation!)")
        print(f"   Effective batch:     {cls.BATCH_SIZE * cls.GRADIENT_ACCUMULATION_STEPS}")
        print(f"   Mixed precision:     {cls.PRECISION.upper()} (2x faster on H100)")
        print(f"   Workers:             {cls.NUM_WORKERS}")
        print(f"   Expected speedup:    ~2x vs baseline")
        
        print(f"\n🛡️  Early Stopping:")
        print(f"   Patience:            {cls.EARLY_STOPPING_PATIENCE} validations (no improvement)")
        print(f"   Min improvement:     {cls.EARLY_STOPPING_MIN_DELTA:.1%}")
        print(f"   (Training will notify but continue - you can stop manually)")
        
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    TrainingConfig.print_config()

