# Created by: Rami Abu Mukh, Omar Gharra, Ameer Khalaily

import torch
from pathlib import Path

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
    
    # Model name
    NAME = "Transformer-Policy"
    VOCAB_SIZES = {
        "moves": len(UCI_MOVES),
        "turn": len(TURN_VOCAB),
        "white_kingside_castling_rights": len(BOOL_VOCAB),
        "white_queenside_castling_rights": len(BOOL_VOCAB),
        "black_kingside_castling_rights": len(BOOL_VOCAB),
        "black_queenside_castling_rights": len(BOOL_VOCAB),
        "board_position": len(PIECES_VOCAB),
    }
    
    #architecture
    D_MODEL = 512          # Vector size throughout transformer
    N_HEADS = 8            # Number of attention heads
    D_QUERIES = 64         # Query vector size
    D_VALUES = 64          # Value vector size
    D_INNER = 2048         # FFN intermediate size
    N_LAYERS = 6           # Number of transformer layers
    DROPOUT = 0.1          # Dropout probability
    BOARD_STATUS_LENGTH = 70  # Input sequence length
    
    # training parameters - exactly the same as the cnn policy
    BATCH_SIZE = 2048                
    BATCHES_PER_STEP = 1             # No grad accum needed with large batch
    N_STEPS = 55_000                 
    WARMUP_STEPS = 8_000             # was suggested by chatgpt 
    
    # Learning Rate - the same as the cnn policy
    LR_SCHEDULE = 'vaswani'
    
    # optimizer - the same as the cnn policy
    BETAS = (0.9, 0.98)
    EPSILON = 1e-9
    
    # loss - the same as the cnn policy
    LABEL_SMOOTHING = 0.1
    
    # H100 Optimization - 2x speedup on H100 GPUs - BF16 better than FP16 for training on H100 GPUs - the same as the cnn policy
    USE_AMP = True                   # Mixed precision
    PRECISION = 'bf16'               # BF16 for H100
    
    # a bit different than the cnn policy - to make it faster in training
    NUM_WORKERS = 24                 
    PIN_MEMORY = True
    PREFETCH_FACTOR = 4              
    PERSISTENT_WORKERS = True        
    
    # checkpointing - save every 5k steps
    SAVE_EVERY_N_STEPS = 5_000
    CHECKPOINT_AVERAGE_LAST_N = 10
    
    # early Stopping
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    BASE_DIR = Path(__file__).parent.parent.absolute()
    DATA_PATH = BASE_DIR / 'dataset'
    CHECKPOINT_DIR = BASE_DIR / 'transformer_policy' / 'checkpoints'
    LOG_DIR = BASE_DIR / 'transformer_policy' / 'logs'
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'