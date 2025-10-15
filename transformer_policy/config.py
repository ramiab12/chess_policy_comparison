"""
Transformer Policy Configuration (CT-EFT-20 Replica)
====================================================
IDENTICAL to CT-EFT-20, with H100 optimizations applied.

Changes from original:
- BATCH_SIZE: 512 → 2048 (H100 optimization)
- BATCHES_PER_STEP: 4 → 1 (no grad accumulation needed)
- N_STEPS: 100,000 → 325,000 (50 epochs)
- Everything else IDENTICAL!
"""

import torch
from pathlib import Path

# Vocabulary sizes (from chess-transformers)
PIECES_VOCAB = [
    "<empty>",
    "<pad>",
    "bP",
    "wP",
    "bR",
    "wR",
    "bN",
    "wN",
    "bB",
    "wB",
    "bQ",
    "wQ",
    "bK",
    "wK",
]

TURN_VOCAB = ["w", "b"]

BOOL_VOCAB = [False, True]

# UCI moves - all possible chess moves
def generate_uci_moves():
    """Generate all possible UCI moves."""
    moves = ["<move>"]  # Special token
    files = 'abcdefgh'
    ranks = '12345678'
    
    # Regular moves
    for from_file in files:
        for from_rank in ranks:
            for to_file in files:
                for to_rank in ranks:
                    if from_file != to_file or from_rank != to_rank:
                        moves.append(f"{from_file}{from_rank}{to_file}{to_rank}")
    
    # Promotions
    for from_file in files:
        for to_file in files:
            for piece in ['q', 'r', 'b', 'n']:
                moves.append(f"{from_file}7{to_file}8{piece}")
                moves.append(f"{from_file}2{to_file}1{piece}")
    
    return moves

UCI_MOVES = generate_uci_moves()


class TransformerConfig:
    """Configuration for CT-EFT-20 Transformer (H100 optimized)."""
    
    # Model name
    NAME = "Transformer-Policy-50epochs"
    
    # Vocabulary sizes (IDENTICAL to CT-EFT-20)
    VOCAB_SIZES = {
        "moves": len(UCI_MOVES),
        "turn": len(TURN_VOCAB),
        "white_kingside_castling_rights": len(BOOL_VOCAB),
        "white_queenside_castling_rights": len(BOOL_VOCAB),
        "black_kingside_castling_rights": len(BOOL_VOCAB),
        "black_queenside_castling_rights": len(BOOL_VOCAB),
        "board_position": len(PIECES_VOCAB),
    }
    
    # Model Architecture (IDENTICAL to CT-EFT-20)
    D_MODEL = 512          # Vector size throughout transformer
    N_HEADS = 8            # Number of attention heads
    D_QUERIES = 64         # Query vector size
    D_VALUES = 64          # Value vector size
    D_INNER = 2048         # FFN intermediate size
    N_LAYERS = 6           # Number of transformer layers
    DROPOUT = 0.1          # Dropout probability
    BOARD_STATUS_LENGTH = 70  # Input sequence length
    
    # Training Parameters (OPTIMIZED FOR H100)
    BATCH_SIZE = 2048                # Optimized: 4x larger (was 512)
    BATCHES_PER_STEP = 1             # Optimized: No accumulation (was 4)
    N_STEPS = 325_000                # Extended: 50 epochs (was 100K)
    WARMUP_STEPS = 8_000             # IDENTICAL to CT-EFT-20
    
    # Learning Rate (IDENTICAL to CT-EFT-20)
    LR_SCHEDULE = 'vaswani'
    
    # Optimizer (IDENTICAL to CT-EFT-20)
    BETAS = (0.9, 0.98)
    EPSILON = 1e-9
    
    # Loss (IDENTICAL to CT-EFT-20)
    LABEL_SMOOTHING = 0.1
    
    # H100 Optimization
    USE_AMP = True                   # Mixed precision
    PRECISION = 'bf16'               # BF16 for H100
    
    # Data
    NUM_WORKERS = 8
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    
    # Checkpointing
    SAVE_EVERY_N_STEPS = 5_000
    CHECKPOINT_AVERAGE_LAST_N = 10
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Paths
    DATA_PATH = Path('dataset')
    CHECKPOINT_DIR = Path('transformer_policy/checkpoints')
    LOG_DIR = Path('transformer_policy/logs')
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def print_config(cls):
        """Print configuration."""
        print("\n" + "=" * 70)
        print("Transformer Policy Configuration (CT-EFT-20 Replica)")
        print("=" * 70)
        print("\n📊 Model Architecture (IDENTICAL to CT-EFT-20):")
        print(f"   d_model:             {cls.D_MODEL}")
        print(f"   n_layers:            {cls.N_LAYERS}")
        print(f"   n_heads:             {cls.N_HEADS}")
        print(f"   d_inner (FFN):       {cls.D_INNER}")
        print(f"   dropout:             {cls.DROPOUT}")
        
        print("\n🎯 Training Parameters (H100 Optimized):")
        print(f"   Batch size:          {cls.BATCH_SIZE}")
        print(f"   Grad accumulation:   {cls.BATCHES_PER_STEP}")
        print(f"   Effective batch:     {cls.BATCH_SIZE * cls.BATCHES_PER_STEP}")
        print(f"   Total steps:         {cls.N_STEPS:,}")
        print(f"   Warmup steps:        {cls.WARMUP_STEPS:,}")
        
        print("\n📉 Learning Rate (IDENTICAL to CT-EFT-20):")
        print(f"   Schedule:            {cls.LR_SCHEDULE}")
        print(f"   Peak LR:             ~{(cls.D_MODEL ** -0.5):.6f}")
        
        print("\n⚙️  Optimizer (IDENTICAL to CT-EFT-20):")
        print(f"   Type:                Adam")
        print(f"   Betas:               {cls.BETAS}")
        print(f"   Epsilon:             {cls.EPSILON}")
        
        print("\n💾 Data:")
        print(f"   Label smoothing:     {cls.LABEL_SMOOTHING}")
        print(f"   Workers:             {cls.NUM_WORKERS}")
        
        print("\n⚡ H100 Optimizations:")
        print(f"   Batch size:          {cls.BATCH_SIZE} (4x larger)")
        print(f"   Mixed precision:     {cls.PRECISION.upper()}")
        print(f"   Expected speedup:    ~2x vs baseline")
        
        # Calculate epochs
        dataset_size = 13_287_522
        total_samples = cls.N_STEPS * cls.BATCH_SIZE * cls.BATCHES_PER_STEP
        epochs = total_samples / dataset_size
        
        print(f"\n📊 Training Scope:")
        print(f"   Total samples:       {total_samples:,}")
        print(f"   Expected epochs:     ~{epochs:.1f}")
        print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    TransformerConfig.print_config()
    
    # Calculate parameter count
    print(f"Expected parameters: ~20M")
    print(f"Comparison: CNN has 17.8M params")

