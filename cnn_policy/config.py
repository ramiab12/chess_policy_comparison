# Created by: Rami Abu Mukh, Omar Gharra, Ameer Khalaily

import torch
from pathlib import Path


class TrainingConfig:
    
    # model architecture
    NUM_INPUT_CHANNELS = 18 #number of input channels
    NUM_FILTERS = 256 #number of filters in the convolutional layers
    NUM_BLOCKS = 15 #number of blocks in the residual layers - increased by rami 8 -> 15
    DROPOUT_RATE = 0.1 #dropout rate - decreased by rami 0.3 -> 0.1 - the previous dropout rate was too high , model was underfitting
    
    # training parameters 
    BATCH_SIZE = 2048                   # the same as the transformer model - 2048
    GRADIENT_ACCUMULATION_STEPS = 1     # when increased the batch size, no gradient accumulation is needed
    TOTAL_STEPS = 55000                 # 55k steps, the same as the transformer model
    WARMUP_STEPS = 8_000                # no warmup needed, but we need to set it to 8000 to match the transformer model
    
    
    # Learning Rate
    LR_SCHEDULE = 'vaswani'             # the same a the transformer model
    D_MODEL = 256                       # for LR calculation                    # was chatgpt's suggestion
    BASE_LR = None                      # Calculated by Vaswani schedule
    
    # optimizer 
    OPTIMIZER = 'Adam'
    BETAS = (0.9, 0.98)                 # matches the transformer model
    EPSILON = 1e-9                      # matches the transformer model
    WEIGHT_DECAY = 0.0                  # matches the transformer model 
    
    # loss function - label smoothing - matches the transformer model
    LABEL_SMOOTHING = 0.1               # matches the transformer model
    
    # data
    TRAIN_SPLIT = 0.9                   # 90% for training, 10% for validation
    VAL_SPLIT = 0.1
    NUM_WORKERS = 8                     # Optimized (was 4)
    PIN_MEMORY = True
    
    # H100 Optimizations - 2x speedup on H100 GPUs - BF16 better than FP16 for training on H100 GPUs
    USE_MIXED_PRECISION = True          # BF16 for 2x speedup on H100
    PRECISION = 'bf16'                  # BF16 better than FP16 for training
    
    # checkpointing - save every 5k steps
    SAVE_EVERY_N_STEPS = 5_000          
    CHECKPOINT_AVERAGE_LAST_N = 10      # turned out to be more effective, by omar
    
    # early stopping - stop if no improvement for 10 validations 
    EARLY_STOPPING_PATIENCE = 10        
    EARLY_STOPPING_MIN_DELTA = 0.001    # 0.1%
    
    # device - inference on cpu, training on gpu - by omar
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # paths
    DATA_PATH = Path('dataset')
    H5_PATH = Path('dataset/raw/LE22ct/LE22ct.h5')  # raw data path - no conversion needed
    CSV_PATH = Path('dataset/processed')             # CSV fallback - not used, worked with h5 file
    CHECKPOINT_DIR = Path('cnn_policy/checkpoints')
    LOG_DIR = Path('cnn_policy/logs')