# Technical Documentation: Chess Policy Comparison

**Created by: Rami Abu Mukh, Omar Gharra, Ameer Khalaily**

**Project:** CNN vs Transformer Architecture Comparison for Chess Move Prediction  
**Date:** 2025  
**Framework:** PyTorch 2.x  
**Hardware:** NVIDIA H100 GPU

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Specification](#2-dataset-specification)
3. [CNN Architecture](#3-cnn-architecture)
4. [Transformer Architecture](#4-transformer-architecture)
5. [Training Configuration](#5-training-configuration)
6. [Implementation Details](#6-implementation-details)
7. [Training Results](#7-training-results)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Model Checkpoints](#9-model-checkpoints)
10. [Reproducibility](#10-reproducibility)

---

## 1. Project Overview

### 1.1 Research Question

Compare CNN and Transformer architectures for chess move prediction under controlled conditions:
- Same task (from-square and to-square prediction)
- Same dataset (LE22ct: Lichess Elite 2400+ games)
- Same training budget (55,000 steps × 2048 batch size)
- Same optimization strategy (Vaswani LR schedule, Adam optimizer)
- Similar parameter count (~18M parameters)

### 1.2 Task Definition

**Input:** Chess position (board state + metadata)  
**Output:** Two probability distributions:
- P(from_square): 64-class classification (which square to move from)
- P(to_square): 64-class classification (which square to move to)

**Loss Function:** Cross-entropy with label smoothing (ε = 0.1)

---

## 2. Dataset Specification

### 2.1 Source

**Name:** LE22ct (Lichess Elite 2022 Checkmate Training)  
**Description:** Chess positions extracted from high-level games ending in checkmate  
**Player Rating:** 2400+ Elo (white) vs 2200+ Elo (black)  
**Format:** HDF5 file with structured arrays

### 2.2 Dataset Statistics

```
Total positions: 13,287,522
Train split: 11,958,769 positions (90%)
Validation split: 1,328,753 positions (10%)

Training samples per epoch: 11,958,769
Effective epochs (55K steps, batch 2048): ~9.4 epochs
Total training samples seen: 112,640,000
```

### 2.3 Data Format

**HDF5 Structure:**
```python
encoded_data: structured array with fields:
  - board_position: (64,) uint8 array
      Encoding: 0=empty, 2-3=pawn(b/w), 4-5=rook(b/w), 
                6-7=knight(b/w), 8-9=bishop(b/w), 
                10-11=queen(b/w), 12-13=king(b/w)
  
  - turn: uint8 (0=white, 1=black)
  
  - white_kingside_castling_rights: bool
  - white_queenside_castling_rights: bool
  - black_kingside_castling_rights: bool
  - black_queenside_castling_rights: bool
  
  - from_square: uint8 (0-63, a1=0, h8=63)
  - to_square: uint8 (0-63)
```

### 2.4 Data Loading

**CNN Pipeline:**
```python
1. Read position from H5 file
2. Decode to FEN string
3. Convert FEN to 18-channel (8×8) tensor:
   - Channels 0-11: Piece positions (6 white + 6 black)
   - Channel 12: Turn (1.0=white, 0.0=black)
   - Channels 13-16: Castling rights (broadcast to 8×8)
   - Channel 17: En passant square (one-hot 8×8)
4. Return: (18, 8, 8) float32 tensor
```

**Transformer Pipeline:**
```python
1. Read position from H5 file directly
2. Extract structured fields
3. Return: dictionary with int tensors
   {
     'turns': (1,),
     'castling_rights': 4 × (1,),
     'board_positions': (64,),
     'from_squares': (1,),
     'to_squares': (1,),
     'lengths': (1,)
   }
```

---

## 3. CNN Architecture

### 3.1 Model: ChessCNNPolicy

**Architecture Type:** Residual Convolutional Neural Network  
**Inspiration:** AlphaZero policy network  
**Design:** Spatial feature extraction with skip connections

### 3.2 Architecture Diagram

```
Input: (B, 18, 8, 8)
    ↓
┌─────────────────────────────────────┐
│ Initial Convolution                  │
│   Conv2d(18 → 256, kernel=3×3,      │
│           padding=1, bias=False)     │
│   BatchNorm2d(256)                   │
│   ReLU                               │
└─────────────────────────────────────┘
    ↓ (B, 256, 8, 8)
┌─────────────────────────────────────┐
│ Residual Block × 15                  │
│   ┌─────────────────────────────┐   │
│   │ Conv2d(256→256, 3×3, pad=1) │   │
│   │ BatchNorm2d(256)            │   │
│   │ ReLU                        │   │
│   │ Conv2d(256→256, 3×3, pad=1) │   │
│   │ BatchNorm2d(256)            │   │
│   │ Dropout2d(0.1)              │   │
│   └─────────────────────────────┘   │
│            ↓                         │
│     + skip connection                │
│            ↓                         │
│          ReLU                        │
└─────────────────────────────────────┘
    ↓ (B, 256, 8, 8)
┌──────────────────┬──────────────────┐
│  From-Square     │   To-Square      │
│  Head            │   Head           │
│                  │                  │
│ Conv2d(256→2,    │ Conv2d(256→2,    │
│   kernel=1×1)    │   kernel=1×1)    │
│ BatchNorm2d(2)   │ BatchNorm2d(2)   │
│ ReLU             │ ReLU             │
│ Take channel 0   │ Take channel 0   │
│ Reshape(B, 64)   │ Reshape(B, 64)   │
└──────────────────┴──────────────────┘
    ↓                    ↓
from_logits (B, 64)  to_logits (B, 64)
```

### 3.3 Layer-by-Layer Specification

#### Initial Convolution
```python
Conv2d(in_channels=18, out_channels=256, 
       kernel_size=3, stride=1, padding=1, bias=False)
BatchNorm2d(256)
ReLU(inplace=False)
```

#### Residual Block (×15)
```python
class ResidualBlock:
    conv1: Conv2d(256, 256, kernel_size=3, stride=1, 
                  padding=1, bias=False)
    bn1: BatchNorm2d(256)
    
    conv2: Conv2d(256, 256, kernel_size=3, stride=1, 
                  padding=1, bias=False)
    bn2: BatchNorm2d(256)
    
    dropout: Dropout2d(p=0.1)
    
    def forward(x):
        identity = x
        out = F.relu(bn1(conv1(x)))
        out = dropout(bn2(conv2(out)))
        out = out + identity  # skip connection
        out = F.relu(out)
        return out
```

#### Policy Heads
```python
# From-square head
from_conv: Conv2d(256, 2, kernel_size=1, bias=False)
from_bn: BatchNorm2d(2)
from_output = F.relu(from_bn(from_conv(features)))
from_logits = from_output[:, 0, :, :].reshape(batch_size, 64)

# To-square head (identical structure)
to_conv: Conv2d(256, 2, kernel_size=1, bias=False)
to_bn: BatchNorm2d(2)
to_output = F.relu(to_bn(to_conv(features)))
to_logits = to_output[:, 0, :, :].reshape(batch_size, 64)
```

### 3.4 Parameter Count

```
Component                      Shape              Parameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Initial Conv                   [256,18,3,3]       41,472
Initial BN                     [256]+[256]        512
Residual Block ×15:
  Conv1 ×15                    [256,256,3,3]      8,847,360
  BN1 ×15                      [256]+[256]        7,680
  Conv2 ×15                    [256,256,3,3]      8,847,360
  BN2 ×15                      [256]+[256]        7,680
From-square head:
  Conv                         [2,256,1,1]        512
  BN                           [2]+[2]            4
To-square head:
  Conv                         [2,256,1,1]        512
  BN                           [2]+[2]            4
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total                                             17,753,096
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Memory (FP32):                                    67.72 MB
Memory (BF16):                                    33.86 MB
```

### 3.5 Weight Initialization

```python
def _initialize_weights():
    for module in self.modules():
        if isinstance(module, nn.Conv2d):
            # He initialization for ReLU activations
            nn.init.kaiming_normal_(module.weight, 
                                    mode='fan_out', 
                                    nonlinearity='relu')
        
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
```

### 3.6 Receptive Field Analysis

```
Layer              Receptive Field    Coverage
─────────────────────────────────────────────────
Input              1×1                Single square
Initial Conv       3×3                3×3 neighborhood
After Block 1      5×5                Local tactics
After Block 3      9×9                Extended area
After Block 5      13×13              ~Half board
After Block 10     23×23              Most of board
After Block 15     33×33              Full board + margin
```

Effective receptive field at output: Entire 8×8 board

---

## 4. Transformer Architecture

### 4.1 Model: ChessTransformer

**Architecture Type:** Encoder-only Transformer  
**Inspiration:** BERT-style architecture adapted for chess  
**Design:** Self-attention over sequential board representation

### 4.2 Architecture Diagram

```
Input: Dictionary of integer tensors
    ↓
┌──────────────────────────────────────────┐
│ Embedding Layer                          │
│   turn → Embedding(2, 512)               │
│   castling_rights (×4) → Embedding(2,512)│
│   board_squares (×64) → Embedding(14,512)│
└──────────────────────────────────────────┘
    ↓ Concatenate → (B, 69, 512)
┌──────────────────────────────────────────┐
│ Add Positional Embeddings                │
│   Learned: Embedding(69, 512)            │
│   Scale by sqrt(d_model) = sqrt(512)     │
│   Apply Dropout(0.1)                     │
└──────────────────────────────────────────┘
    ↓ (B, 69, 512)
┌──────────────────────────────────────────┐
│ Transformer Encoder Layer × 6            │
│   ┌────────────────────────────────────┐ │
│   │ Multi-Head Self-Attention          │ │
│   │   heads=8, d_q=d_k=64, d_v=64      │ │
│   │   LayerNorm (pre-norm)             │ │
│   │   Residual connection              │ │
│   │   Dropout(0.1)                     │ │
│   └────────────────────────────────────┘ │
│            ↓                              │
│   ┌────────────────────────────────────┐ │
│   │ Position-wise Feed-Forward         │ │
│   │   512 → 2048 → 512                 │ │
│   │   LayerNorm (pre-norm)             │ │
│   │   Residual connection              │ │
│   │   Dropout(0.1)                     │ │
│   └────────────────────────────────────┘ │
└──────────────────────────────────────────┘
    ↓ (B, 69, 512)
┌──────────────────────────────────────────┐
│ Final Layer Normalization                │
└──────────────────────────────────────────┘
    ↓ Take tokens 5:69 (board squares)
    ↓ (B, 64, 512)
┌────────────────┬─────────────────────────┐
│ From-Square    │  To-Square              │
│ Head           │  Head                   │
│ Linear(512→1)  │  Linear(512→1)          │
│ Squeeze        │  Squeeze                │
└────────────────┴─────────────────────────┘
    ↓ (B, 1, 64)        ↓ (B, 1, 64)
```

### 4.3 Detailed Component Specifications

#### 4.3.1 Embedding Layer

```python
# Token embeddings
turn_embeddings: Embedding(2, 512)
white_kingside_castling_rights_embeddings: Embedding(2, 512)
white_queenside_castling_rights_embeddings: Embedding(2, 512)
black_kingside_castling_rights_embeddings: Embedding(2, 512)
black_queenside_castling_rights_embeddings: Embedding(2, 512)
board_position_embeddings: Embedding(14, 512)

# Positional embeddings
positional_embeddings: Embedding(69, 512)

# Forward pass
embeddings = torch.cat([
    turn_embeddings(turns),                    # (B, 1, 512)
    wk_embeddings(white_kingside_rights),      # (B, 1, 512)
    wq_embeddings(white_queenside_rights),     # (B, 1, 512)
    bk_embeddings(black_kingside_rights),      # (B, 1, 512)
    bq_embeddings(black_queenside_rights),     # (B, 1, 512)
    board_embeddings(board_positions),         # (B, 64, 512)
], dim=1)  # → (B, 69, 512)

# Add positional info and scale
output = (embeddings + positional_embeddings.weight) * sqrt(512)
output = dropout(output)
```

#### 4.3.2 Multi-Head Attention Layer

```python
class MultiHeadAttention(nn.Module):
    d_model = 512
    n_heads = 8
    d_queries = 64  # 512 / 8
    d_values = 64   # 512 / 8
    d_keys = 64
    
    # Linear projections
    cast_queries: Linear(512, 512)      # 8 heads × 64 dims
    cast_keys_values: Linear(512, 1024) # 8 heads × (64+64)
    cast_output: Linear(512, 512)
    
    layer_norm: LayerNorm(512)
    dropout: Dropout(0.1)
    
    def forward(query_seq, kv_seq, kv_lengths):
        # Pre-layer norm
        query_seq = layer_norm(query_seq)
        if self_attention:
            kv_seq = layer_norm(kv_seq)
        
        # Project to Q, K, V
        Q = cast_queries(query_seq)  # (B, 69, 512)
        K, V = cast_keys_values(kv_seq).split(512, dim=-1)
        
        # Reshape for multi-head: (B, 69, 8, 64)
        Q = Q.view(B, 69, 8, 64).permute(0, 2, 1, 3)
        K = K.view(B, 69, 8, 64).permute(0, 2, 1, 3)
        V = V.view(B, 69, 8, 64).permute(0, 2, 1, 3)
        # → (B, 8, 69, 64)
        
        # Merge batch and heads: (B*8, 69, 64)
        Q = Q.contiguous().view(B*8, 69, 64)
        K = K.contiguous().view(B*8, 69, 64)
        V = V.contiguous().view(B*8, 69, 64)
        
        # Attention scores
        scores = Q @ K.transpose(-2, -1)  # (B*8, 69, 69)
        scores = scores / sqrt(64)
        
        # Mask padding (all tokens are valid, no padding)
        # Apply softmax
        attn_weights = softmax(scores, dim=-1)
        attn_weights = dropout(attn_weights)
        
        # Apply attention to values
        output = attn_weights @ V  # (B*8, 69, 64)
        
        # Reshape back: (B, 69, 512)
        output = output.view(B, 8, 69, 64)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(B, 69, 512)
        
        # Output projection
        output = cast_output(output)
        output = dropout(output)
        
        # Residual connection
        output = output + input_to_add
        
        return output
```

**Attention Pattern:**
- Each of 69 tokens attends to all 69 tokens
- 8 attention heads learn different patterns
- Attention matrix: (69 × 69) per head
- Total attention computations: 8 × 69² = 38,088 per layer

#### 4.3.3 Position-Wise Feed-Forward Network

```python
class PositionWiseFCNetwork(nn.Module):
    d_model = 512
    d_inner = 2048
    
    layer_norm: LayerNorm(512)
    fc1: Linear(512, 2048)
    relu: ReLU()
    fc2: Linear(2048, 512)
    dropout: Dropout(0.1)
    
    def forward(sequences):
        input_to_add = sequences.clone()
        
        # Pre-layer norm
        sequences = layer_norm(sequences)
        
        # Two-layer MLP with ReLU
        sequences = fc1(sequences)          # (B, 69, 2048)
        sequences = relu(sequences)
        sequences = dropout(sequences)
        sequences = fc2(sequences)          # (B, 69, 512)
        sequences = dropout(sequences)
        
        # Residual connection
        sequences = sequences + input_to_add
        
        return sequences
```

#### 4.3.4 Policy Heads

```python
# Take only board square tokens (skip first 5 metadata tokens)
board_representations = encoder_output[:, 5:, :]  # (B, 64, 512)

# From-square head
from_squares = Linear(512, 1)(board_representations)  # (B, 64, 1)
from_squares = from_squares.squeeze(2).unsqueeze(1)  # (B, 1, 64)

# To-square head
to_squares = Linear(512, 1)(board_representations)    # (B, 64, 1)
to_squares = to_squares.squeeze(2).unsqueeze(1)      # (B, 1, 64)
```

### 4.4 Parameter Count

```
Component                           Shape              Parameters
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Embeddings:
  turn                              [2, 512]           1,024
  castling_rights (×4)              [2, 512] ×4        4,096
  board_positions                   [14, 512]          7,168
  positional                        [69, 512]          35,328
  
Encoder Layer ×6:
  Multi-Head Attention:
    cast_queries                    [512, 512] ×6      1,572,864
    cast_queries.bias               [512] ×6           3,072
    cast_keys_values                [1024, 512] ×6     3,145,728
    cast_keys_values.bias           [1024] ×6          6,144
    cast_output                     [512, 512] ×6      1,572,864
    cast_output.bias                [512] ×6           3,072
    layer_norm (weight+bias)        [512]+[512] ×6     6,144
    
  Feed-Forward:
    fc1                             [2048, 512] ×6     6,291,456
    fc1.bias                        [2048] ×6          12,288
    fc2                             [512, 2048] ×6     6,291,456
    fc2.bias                        [512] ×6           3,072
    layer_norm (weight+bias)        [512]+[512] ×6     6,144
    
Final layer_norm                    [512]+[512]        1,024

Policy Heads:
  from_squares                      [1, 512]           512
  to_squares                        [1, 512]           512
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total                                                  18,963,970
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Memory (FP32):                                         72.34 MB
Memory (BF16):                                         36.17 MB
```

### 4.5 Computational Complexity

```
Component                    Complexity per Layer
───────────────────────────────────────────────────
Multi-Head Attention:
  Q, K, V projections        O(n × d²) = O(69 × 512²)
  Attention computation      O(n² × d) = O(69² × 512)
  Output projection          O(n × d²) = O(69 × 512²)
  
Feed-Forward:
  FC1                        O(n × d × d_ff) = O(69 × 512 × 2048)
  FC2                        O(n × d_ff × d) = O(69 × 2048 × 512)

Total per layer: ~38M FLOPs
Total (6 layers): ~228M FLOPs
```

### 4.6 Weight Initialization

```python
def init_weights():
    # Xavier uniform for linear layers
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param, gain=1.0)
    
    # Normal initialization for embeddings
    std = math.pow(d_model, -0.5)  # 1/sqrt(512) ≈ 0.044
    
    nn.init.normal_(board_position_embeddings.weight, 
                    mean=0.0, std=std)
    nn.init.normal_(turn_embeddings.weight, 
                    mean=0.0, std=std)
    # ... same for all embeddings
```

---

## 5. Training Configuration

### 5.1 Hyperparameters

```yaml
# Common (both models)
TASK: from-to square prediction
BATCH_SIZE: 2048
TRAINING_STEPS: 55000
TOTAL_SAMPLES: 112640000 (55000 × 2048)
EPOCHS: ~9.4 (112.6M / 11.96M)

OPTIMIZER: Adam
BETAS: (0.9, 0.98)
EPSILON: 1e-9
WEIGHT_DECAY: 0.0

LR_SCHEDULE: Vaswani
WARMUP_STEPS: 8000
D_MODEL_FOR_LR: 
  - CNN: 256
  - Transformer: 512

LABEL_SMOOTHING: 0.1

MIXED_PRECISION: BF16 (brain float 16)
GRADIENT_ACCUMULATION:
  - CNN: 1 step (effective batch = 2048)
  - Transformer: 1 step (effective batch = 2048)

CHECKPOINT_FREQUENCY: every 5000 steps
VALIDATION_FREQUENCY: every 5000 steps
CHECKPOINT_AVERAGING: last 10 checkpoints

EARLY_STOPPING:
  PATIENCE: 10 validations
  MIN_DELTA: 0.001 (0.1% improvement)
```

### 5.2 Learning Rate Schedule

**Vaswani Schedule (from "Attention is All You Need"):**

```python
def get_vaswani_lr(step: int, d_model: int, warmup_steps: int) -> float:
    step = max(step, 1)
    lr = (d_model ** -0.5) * min(
        step ** -0.5,
        step * (warmup_steps ** -1.5)
    )
    return lr
```

**CNN Learning Rate:**
```
d_model = 256
Peak LR = 256^(-0.5) = 0.0625 at step ~8000
Warmup: steps 1-8000, linear increase
Decay: steps 8000+, follows 1/sqrt(step)

Example schedule:
  Step 1000:  0.00195
  Step 4000:  0.00781
  Step 8000:  0.06250 (peak)
  Step 16000: 0.04419
  Step 32000: 0.03125
  Step 55000: 0.01976
```

**Transformer Learning Rate:**
```
d_model = 512
Peak LR = 512^(-0.5) = 0.04419 at step ~8000
Warmup: steps 1-8000, linear increase
Decay: steps 8000+, follows 1/sqrt(step)

Example schedule:
  Step 1000:  0.00138
  Step 4000:  0.00552
  Step 8000:  0.04419 (peak)
  Step 16000: 0.03125
  Step 32000: 0.02209
  Step 55000: 0.01397
```

### 5.3 Loss Function

**Label Smoothed Cross-Entropy:**

```python
class LabelSmoothedCrossEntropy:
    epsilon = 0.1
    
    def forward(logits, targets):
        # logits: (B, num_classes)
        # targets: (B,) integer labels
        
        num_classes = logits.size(-1)  # 64 for chess
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smooth label distribution
        confidence = 1.0 - epsilon  # 0.9
        smooth_value = epsilon / (num_classes - 1)  # 0.1/63 ≈ 0.00159
        
        smooth_labels = torch.zeros_like(log_probs)
        smooth_labels.fill_(smooth_value)
        smooth_labels.scatter_(1, targets.unsqueeze(1), confidence)
        
        # KL divergence between smooth labels and predictions
        loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        
        return loss
```

**Total Loss:**
```python
from_loss = LabelSmoothedCE(from_logits, from_targets)
to_loss = LabelSmoothedCE(to_logits, to_targets)
total_loss = from_loss + to_loss
```

### 5.4 Hardware Configuration

```
GPU: NVIDIA H100 80GB
CUDA Version: 12.1+
PyTorch Version: 2.1+
Mixed Precision: BF16 (brain float 16)

Dataloader:
  NUM_WORKERS: 
    - CNN: 8
    - Transformer: 24
  PIN_MEMORY: True
  PREFETCH_FACTOR: 4 (Transformer only)
  PERSISTENT_WORKERS: True (Transformer only)

Memory Usage:
  CNN Model: ~34 MB (BF16)
  Transformer Model: ~36 MB (BF16)
  Batch (2048): ~1.5 GB
  Optimizer States: ~150 MB
  Total VRAM: ~2 GB
```

### 5.5 Training Time

```
CNN:
  Step time: ~0.35 seconds
  55000 steps: ~5.3 hours
  Cost (H100 @ $2.70/hr): ~$14.31

Transformer:
  Step time: ~0.40 seconds
  55000 steps: ~6.1 hours
  Cost (H100 @ $2.70/hr): ~$16.47
```

---

## 6. Implementation Details

### 6.1 File Structure

```
chess_policy_comparison/
├── cnn_policy/
│   ├── model.py                    # ChessCNNPolicy definition
│   ├── train.py                    # Training loop
│   ├── config.py                   # Hyperparameters
│   ├── position_encoder.py        # FEN → 18×8×8 tensor
│   ├── dataset_loader.py           # H5 data loading (deprecated)
│   ├── dataset_h5_proper.py        # Current H5 loader
│   ├── inference.py                # Move prediction
│   ├── checkpoints/
│   │   └── checkpoint_step_55000.pth
│   └── logs/
│       ├── training_log.csv
│       └── plots.png
│
├── transformer_policy/
│   ├── model.py                    # ChessTransformer definition
│   ├── train.py                    # Training loop
│   ├── config.py                   # Hyperparameters
│   ├── dataset.py                  # H5 data loading
│   ├── inference.py                # Move prediction
│   ├── checkpoints/
│   │   └── checkpoint_step_55000.pth
│   └── logs/
│       ├── training_log.csv
│       ├── plots.png
│       └── loss_analysis.png
│
├── dataset/
│   └── raw/LE22ct/
│       └── LE22ct.h5               # 13.3M positions
│
├── ARCHITECTURE_ANALYSIS.md        # Conceptual comparison
├── TECHNICAL_DOCUMENTATION.md      # This file
└── README.md                       # Project overview
```

### 6.2 Key Implementation Choices

#### 6.2.1 CNN-specific

1. **No global pooling:** Spatial structure preserved throughout
2. **Dropout placement:** After second conv in each residual block
3. **Policy heads:** 1×1 convolution to reduce channels, then flatten
4. **Batch normalization:** Applied after every convolution
5. **Activation:** ReLU (not LeakyReLU or GELU)

#### 6.2.2 Transformer-specific

1. **Pre-layer normalization:** Layer norm before attention/FFN (more stable)
2. **Positional embeddings:** Learned, not sinusoidal (only 69 tokens)
3. **No decoder:** Encoder-only architecture (single-step prediction)
4. **No causal masking:** All tokens see all tokens
5. **Attention dropout:** Applied to attention weights

#### 6.2.3 Common Choices

1. **From-to factorization:** Separate heads for from/to squares
2. **Direct square prediction:** Not UCI move strings
3. **No value head:** Policy network only (no game outcome prediction)
4. **Single-step prediction:** No move sequences
5. **Promotion handling:** Always promote to queen at inference time

### 6.3 Training Loop Pseudocode

```python
def train():
    model.train()
    optimizer = Adam(model.parameters(), lr=1e-7, 
                     betas=(0.9, 0.98), eps=1e-9)
    scaler = GradScaler()  # for BF16
    
    for step in range(1, 55001):
        # Update learning rate
        lr = get_vaswani_lr(step, d_model, warmup_steps=8000)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Get batch
        batch = next(train_loader)
        
        # Forward pass (with mixed precision)
        with autocast(dtype=torch.bfloat16):
            from_logits, to_logits = model(batch['positions'])
            from_loss = criterion(from_logits, batch['from_squares'])
            to_loss = criterion(to_logits, batch['to_squares'])
            loss = from_loss + to_loss
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        if step % 100 == 0:
            log_metrics(step, loss, lr)
        
        # Validation
        if step % 5000 == 0:
            val_acc = validate(model, val_loader)
            save_checkpoint(model, optimizer, step, val_acc)
        
        # Early stopping
        if no_improvement_for(10):
            break
    
    # Checkpoint averaging
    final_model = average_last_n_checkpoints(n=10)
    save(final_model, "final_averaged.pth")
```

### 6.4 Validation Procedure

```python
def validate(model, val_loader):
    model.eval()
    
    total_from_correct = 0
    total_to_correct = 0
    total_both_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            from_logits, to_logits = model(batch['positions'])
            
            from_pred = from_logits.argmax(dim=-1)
            to_pred = to_logits.argmax(dim=-1)
            
            from_correct = (from_pred == batch['from_squares']).sum()
            to_correct = (to_pred == batch['to_squares']).sum()
            both_correct = ((from_pred == batch['from_squares']) & 
                           (to_pred == batch['to_squares'])).sum()
            
            total_from_correct += from_correct
            total_to_correct += to_correct
            total_both_correct += both_correct
            total_samples += len(batch)
    
    from_acc = total_from_correct / total_samples
    to_acc = total_to_correct / total_samples
    move_acc = total_both_correct / total_samples
    
    return {
        'from_acc': from_acc,
        'to_acc': to_acc,
        'move_acc': move_acc
    }
```

---

## 7. Training Results

### 7.1 CNN Training Results

**Training completed:** Step 105,000 (continued past 55K)  
**Best checkpoint:** Step 55,000

```
Training Progression:
─────────────────────────────────────────────────────────────
Step    From_Acc  To_Acc    Move_Acc   Learning_Rate
─────────────────────────────────────────────────────────────
5000    54.86%    44.95%    35.62%     0.000437
10000   60.12%    50.81%    43.83%     0.000625
15000   62.31%    53.28%    47.07%     0.000510
20000   63.25%    54.41%    48.73%     0.000442
25000   63.77%    55.09%    49.52%     0.000395
30000   63.92%    55.35%    49.94%     0.000361
35000   64.01%    55.44%    50.14%     0.000334
40000   64.07%    55.49%    50.34%     0.000312
45000   64.10%    55.52%    50.49%     0.000294
50000   64.11%    55.53%    50.56%     0.000278
55000   64.12%    55.44%    50.53%     0.000265 ← Selected
60000   64.12%    55.45%    50.66%     0.000255
...
100000  63.73%    55.03%    50.53%     0.000198
105000  63.74%    55.13%    50.55%     0.000193
─────────────────────────────────────────────────────────────

Final Performance (Step 55000):
  From-square accuracy: 64.12%
  To-square accuracy:   55.44%
  Full move accuracy:   50.53%
  
Training time: ~5.3 hours (55K steps)
```

**Observations:**
- Rapid improvement in first 15K steps (warmup + early training)
- Plateau around 40K steps
- From-square prediction easier than to-square (64% vs 55%)
- Full move accuracy ~50% (both squares correct)

### 7.2 Transformer Training Results

**Training completed:** Step 55,000  
**Best checkpoint:** Step 55,000

```
Training Progression:
─────────────────────────────────────────────────────────────
Step    From_Acc  To_Acc    Move_Acc   Learning_Rate  Hours
─────────────────────────────────────────────────────────────
5000    54.98%    44.20%    36.18%     0.000309      4.78
10000   59.86%    50.54%    43.75%     0.000442      9.61
15000   62.20%    53.27%    47.12%     0.000361      14.36
20000   63.33%    54.64%    48.71%     0.000313      19.10
25000   63.99%    55.38%    49.77%     0.000280      23.87
30000   64.42%    55.94%    50.53%     0.000256      28.64
35000   64.87%    56.51%    51.22%     0.000236      33.32
40000   65.09%    56.72%    51.51%     0.000221      37.86
45000   65.33%    57.02%    51.78%     0.000208      42.39
50000   65.45%    57.18%    51.95%     0.000198      47.18
55000   65.56%    57.29%    52.14%     0.000188      52.06 ← Final
─────────────────────────────────────────────────────────────

Final Performance (Step 55000):
  From-square accuracy: 65.56%
  To-square accuracy:   57.29%
  Full move accuracy:   52.14%
  
Training time: ~52 hours (55K steps)
```

**Observations:**
- Slower per-step training (more compute per attention layer)
- Continued improvement throughout training (no early plateau)
- Slightly better final accuracy than CNN
- Higher to-square accuracy (57.3% vs 55.4%)

### 7.3 Comparative Analysis

```
Metric              CNN        Transformer   Difference
───────────────────────────────────────────────────────────
From Accuracy      64.12%     65.56%        +1.44%
To Accuracy        55.44%     57.29%        +1.85%
Move Accuracy      50.53%     52.14%        +1.61%
Training Time      5.3 hrs    52.0 hrs      +9.8×
Steps to Plateau   ~40K       >55K          -
Final LR           0.000265   0.000188      -
───────────────────────────────────────────────────────────
```

**Key Findings:**
1. Transformer achieves marginally higher accuracy (+1.6% move accuracy)
2. CNN trains much faster (5.3 hours vs 52 hours for 55K steps)
3. CNN plateaus earlier, Transformer continues improving
4. Both models find from-square easier than to-square
5. Full move accuracy ~50%, meaning ~50% of positions have correct move predicted

### 7.4 Loss Curves

Both models show:
- Sharp loss decrease in first 10K steps (warmup phase)
- Gradual convergence 10K-55K steps
- No overfitting (validation loss tracks training loss)
- Stable training (no divergence or collapse)

---

## 8. Inference Pipeline

### 8.1 CNN Inference

```python
def predict_move_cnn(model, board, encoder, device, k=1):
    """
    Predict next move using CNN model.
    
    Args:
        model: ChessCNNPolicy instance
        board: chess.Board object
        encoder: PositionEncoder instance
        device: torch.device
        k: top-k sampling (1 = greedy, >1 = stochastic)
    
    Returns:
        chess.Move object
    """
    model.eval()
    
    # 1. Encode position to 18×8×8 tensor
    position = encoder.fen_to_tensor(board.fen())
    position = position.unsqueeze(0).to(device)  # (1, 18, 8, 8)
    
    # 2. Forward pass
    with torch.no_grad():
        from_logits, to_logits = model(position)  # (1, 64), (1, 64)
    
    # 3. Combine probabilities (log space)
    from_log_probs = F.log_softmax(from_logits, dim=-1)  # (1, 64)
    to_log_probs = F.log_softmax(to_logits, dim=-1)      # (1, 64)
    
    # Outer product: P(from, to) = P(from) * P(to)
    from_log_probs = from_log_probs.unsqueeze(2)  # (1, 64, 1)
    to_log_probs = to_log_probs.unsqueeze(1)      # (1, 1, 64)
    combined = (from_log_probs + to_log_probs).view(1, -1)  # (1, 4096)
    
    # 4. Filter to legal moves
    legal_moves = [move.uci() for move in board.legal_moves]
    legal_moves_no_promo = list(set([m[:4] for m in legal_moves]))
    
    legal_indices = []
    for move_str in legal_moves_no_promo:
        from_sq = chess.SQUARE_NAMES.index(move_str[:2])
        to_sq = chess.SQUARE_NAMES.index(move_str[2:4])
        move_idx = from_sq * 64 + to_sq
        legal_indices.append(move_idx)
    
    legal_predictions = combined[:, legal_indices]  # (1, num_legal)
    
    # 5. Sample move
    if k == 1:
        best_idx = legal_predictions.argmax().item()
    else:
        # Top-k sampling
        topk_logits, topk_indices = torch.topk(legal_predictions, k, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        sampled = torch.multinomial(topk_probs, 1)
        best_idx = topk_indices.gather(-1, sampled).item()
    
    move_str = legal_moves_no_promo[best_idx]
    
    # 6. Handle pawn promotion (always queen)
    from_sq = chess.SQUARE_NAMES.index(move_str[:2])
    to_sq = chess.SQUARE_NAMES.index(move_str[2:4])
    piece = board.piece_at(from_sq)
    
    if piece and piece.piece_type == chess.PAWN:
        to_rank = to_sq // 8
        if (piece.color == chess.WHITE and to_rank == 7) or \
           (piece.color == chess.BLACK and to_rank == 0):
            move_str = move_str + "q"
    
    return chess.Move.from_uci(move_str)
```

### 8.2 Transformer Inference

```python
def predict_move_transformer(model, board, device, k=1):
    """
    Predict next move using Transformer model.
    
    Args:
        model: ChessTransformer instance
        board: chess.Board object
        device: torch.device
        k: top-k sampling
    
    Returns:
        chess.Move object
    """
    model.eval()
    
    # 1. Encode position to dictionary of tensors
    batch = {
        'turns': torch.tensor([[0 if board.turn == chess.WHITE else 1]]).to(device),
        'white_kingside_castling_rights': torch.tensor(
            [[int(board.has_kingside_castling_rights(chess.WHITE))]]
        ).to(device),
        'white_queenside_castling_rights': torch.tensor(
            [[int(board.has_queenside_castling_rights(chess.WHITE))]]
        ).to(device),
        'black_kingside_castling_rights': torch.tensor(
            [[int(board.has_kingside_castling_rights(chess.BLACK))]]
        ).to(device),
        'black_queenside_castling_rights': torch.tensor(
            [[int(board.has_queenside_castling_rights(chess.BLACK))]]
        ).to(device),
        'board_positions': encode_board_to_indices(board).to(device)  # (1, 64)
    }
    
    # 2. Forward pass
    with torch.no_grad():
        from_logits, to_logits = model(batch)
        from_logits = from_logits.squeeze(1)  # (1, 64)
        to_logits = to_logits.squeeze(1)      # (1, 64)
    
    # 3-6. Same as CNN (combine probs, filter legal, sample, handle promotion)
    # [Identical logic to CNN inference from step 3 onwards]
    
    return chess.Move.from_uci(move_str)
```

### 8.3 Performance Metrics

```
Inference Speed (single position, CPU):
  CNN:         ~15 ms
  Transformer: ~25 ms

Inference Speed (batch=32, GPU):
  CNN:         ~8 ms  (0.25 ms per position)
  Transformer: ~12 ms (0.38 ms per position)

Memory (inference, BF16):
  CNN:         ~50 MB
  Transformer: ~55 MB
```

---

## 9. Model Checkpoints

### 9.1 Checkpoint Format

```python
checkpoint = {
    'step': int,                    # Training step number
    'model_state_dict': OrderedDict,  # Model parameters
    'optimizer_state_dict': dict,     # Optimizer state
    'scaler_state_dict': dict,        # GradScaler state (for BF16)
    'training_history': list,         # List of metrics per step
    'config': dict,                   # Hyperparameters
    'val_accuracy': {                 # Validation metrics
        'from_acc': float,
        'to_acc': float,
        'move_acc': float
    }
}
```

### 9.2 Loading Checkpoints

```python
def load_checkpoint(checkpoint_path, model, optimizer=None, device='cuda'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    step = checkpoint['step']
    val_accuracy = checkpoint['val_accuracy']
    
    return model, optimizer, step, val_accuracy
```

### 9.3 Checkpoint Averaging

```python
def average_checkpoints(checkpoint_paths):
    """Average parameters from multiple checkpoints."""
    
    # Load first checkpoint to get structure
    avg_state = torch.load(checkpoint_paths[0])['model_state_dict']
    
    # Average parameters
    for key in avg_state.keys():
        # Sum parameters from all checkpoints
        param_sum = avg_state[key].clone()
        for path in checkpoint_paths[1:]:
            checkpoint = torch.load(path)
            param_sum += checkpoint['model_state_dict'][key]
        
        # Take average
        avg_state[key] = param_sum / len(checkpoint_paths)
    
    return avg_state
```

**Applied:** Last 10 checkpoints averaged for final models

---

## 10. Reproducibility

### 10.1 Random Seeds

```python
import random
import numpy as np
import torch

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### 10.2 Environment Setup

```bash
# Create conda environment
conda create -n chess_policy python=3.10
conda activate chess_policy

# Install dependencies
pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
pip install python-chess==1.999
pip install h5py==3.9.0
pip install numpy==1.24.3
pip install pandas==2.0.3
pip install tqdm==4.66.1
pip install tensorboard==2.14.0
```

### 10.3 Complete Requirements

```
torch==2.1.0
python-chess==1.999
h5py==3.9.0
numpy==1.24.3
pandas==2.0.3
tqdm==4.66.1
tensorboard==2.14.0
matplotlib==3.7.2
scipy==1.11.2
```

### 10.4 Training Commands

```bash
# Train CNN
cd chess_policy_comparison
python cnn_policy/train.py

# Train Transformer
python transformer_policy/train.py

# Monitor with TensorBoard
tensorboard --logdir cnn_policy/logs/tensorboard
tensorboard --logdir transformer_policy/logs/tensorboard
```

### 10.5 Deterministic Data Loading

```python
# Fix dataloader randomness
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    dataset,
    batch_size=2048,
    shuffle=True,
    num_workers=8,
    worker_init_fn=seed_worker,
    generator=g
)
```

---

## Appendix A: Notation Glossary

```
B:          Batch size
N:          Sequence length (69 for transformer)
d_model:    Model dimensionality (512 for transformer, 256 for CNN)
d_k, d_q:   Key/query dimensions (64)
d_v:        Value dimensions (64)
d_ff:       Feed-forward inner dimension (2048)
n_heads:    Number of attention heads (8)
n_layers:   Number of transformer layers (6)
n_blocks:   Number of residual blocks (15)
```

---

## Appendix B: Mathematical Formulations

### Multi-Head Attention

```
Q = XW_Q    K = XW_K    V = XW_V

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

MultiHead(X) = Concat(head_1, ..., head_h)W_O
where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
```

### Residual Block (CNN)

```
H(x) = F(x) + x

where F(x) = Dropout(BN(Conv(ReLU(BN(Conv(x))))))
```

### Label Smoothing

```
y_smooth = (1 - ε) * y_hard + ε / K

where:
  y_hard: one-hot target
  ε: smoothing factor (0.1)
  K: number of classes (64)
```

---

**Document Version:** 1.0  
**Last Updated:** 2025-01-20  
**Contact:** [Project Repository]

---

