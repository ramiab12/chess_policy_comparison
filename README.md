# Chess Policy Comparison: CNN vs Transformer

**Created by: Rami Abu Mukh, Omar Garah, Ameer Khalaily**

**A controlled comparison of CNN and Transformer architectures for chess move prediction**

---

##  Project Overview

This project implements and compares two neural network architectures for chess move prediction:
- **CNN-based policy network** (inspired by AlphaZero)
- **Transformer-based policy network** (encoder-only architecture)

Both models are trained on the same dataset with identical hyperparameters to isolate the impact of architectural choices on chess playing ability.

### Research Question

*How do architectural inductive biases (spatial CNN vs attention-based Transformer) affect chess move prediction performance when all other variables are controlled?*

---

##  Project Objectives

### Primary Objectives

1. **Fair Comparison**: Train both CNN and Transformer models under identical conditions
   - Same dataset (LE22ct: 13.3M elite chess positions)
   - Same training budget (55,000 steps × 2048 batch size)
   - Same optimization (Vaswani LR schedule, Adam, label smoothing)
   - Similar parameter count (~18M parameters)

2. **Architecture Analysis**: Understand how each architecture processes chess positions
   - CNN: Spatial 2D convolutions with residual connections
   - Transformer: Sequence-based self-attention mechanism

3. **Behavioral Comparison**: Compare models through gameplay evaluation
   - Playing strength (Elo rating estimation)
   - Tactical accuracy
   - Strategic understanding
   - Error patterns and failure modes

### Implementation Goals

- Clean, well-documented code
- Reproducible training pipeline
- Comprehensive logging and visualization
- Easy model comparison interface

---

##  Implementation Details

### Task Definition

**Input:** Chess position (board state + metadata)  
**Output:** Two probability distributions:
- P(from_square): 64-class classification (which square to move from)
- P(to_square): 64-class classification (which square to move to)

**Loss:** Cross-entropy with label smoothing (ε = 0.1)

### Dataset

**Source:** LE22ct (Lichess Elite 2022 Checkmate Training)
- **Size:** 13,287,522 positions
- **Quality:** Games between 2400+ Elo players
- **Format:** HDF5 with structured arrays
- **Split:** 90% train, 10% validation

### CNN Architecture

**Model:** ChessCNNPolicy (ResNet-style)

```
Input: (18, 8, 8) tensor
  ├─ 18 channels: pieces (12) + turn (1) + castling (4) + en passant (1)
  └─ 8×8 spatial grid matching chess board

Architecture:
  ├─ Initial Conv: 18 → 256 filters, 3×3 kernel
  ├─ 15 Residual Blocks (256 filters each)
  │   ├─ Conv 3×3 + BatchNorm + ReLU
  │   ├─ Conv 3×3 + BatchNorm + Dropout
  │   └─ Skip connection + ReLU
  └─ Policy Heads (two separate 1×1 convolutions)
      ├─ From-square: 256 → 64 logits
      └─ To-square: 256 → 64 logits

Parameters: 17,753,096
Memory (BF16): 33.86 MB
```

**Key Features:**
- Preserves spatial structure throughout
- Translation equivariance (same conv filters across board)
- Local receptive fields growing to full board
- Batch normalization for training stability

### Transformer Architecture

**Model:** ChessTransformer (Encoder-only)

```
Input: Sequence of 69 tokens
  ├─ Token 0: Turn (white/black)
  ├─ Tokens 1-4: Castling rights
  └─ Tokens 5-68: Board squares (a1 to h8)

Architecture:
  ├─ Token Embeddings (learned, 512-dim)
  ├─ Positional Embeddings (learned, 512-dim)
  ├─ 6 Transformer Encoder Layers
  │   ├─ Multi-Head Self-Attention (8 heads, 64-dim each)
  │   │   ├─ Pre-layer normalization
  │   │   ├─ Residual connection
  │   │   └─ Dropout (0.1)
  │   └─ Position-wise FFN (512 → 2048 → 512)
  │       ├─ Pre-layer normalization
  │       ├─ Residual connection
  │       └─ Dropout (0.1)
  └─ Policy Heads (two separate linear layers)
      ├─ From-square: 512 → 1 per square → 64 logits
      └─ To-square: 512 → 1 per square → 64 logits

Parameters: 18,963,970
Memory (BF16): 36.17 MB
```

**Key Features:**
- Global attention (every token sees every token)
- No spatial inductive bias
- Learned positional relationships
- Pre-layer normalization for stability

---

##  Training Configuration

### Hyperparameters (Identical for Both Models)

```yaml
Optimizer: Adam
  - Beta1: 0.9
  - Beta2: 0.98
  - Epsilon: 1e-9
  - Weight decay: 0.0

Learning Rate: Vaswani Schedule
  - Warmup: 8,000 steps
  - CNN Peak LR: 0.0625 (d_model=256)
  - Transformer Peak LR: 0.0442 (d_model=512)
  - Decay: Inverse square root after warmup

Training:
  - Steps: 55,000
  - Batch size: 2048
  - Total samples: 112,640,000 (~9.4 epochs)
  - Label smoothing: 0.1
  - Mixed precision: BF16

Hardware:
  - GPU: NVIDIA H100 80GB
  - Training time: CNN ~5.3 hours, Transformer ~52 hours
```

### Training Results

**CNN Model (Step 55,000):**
- From-square accuracy: 64.12%
- To-square accuracy: 55.44%
- Full move accuracy: 50.53%

**Transformer Model (Step 55,000):**
- From-square accuracy: 65.56%
- To-square accuracy: 57.29%
- Full move accuracy: 52.14%

Both models show:
- Rapid improvement during warmup (0-10K steps)
- Steady convergence through training
- No overfitting (validation tracks training)
- From-square prediction easier than to-square prediction

---

##  Project Structure

```
chess_policy_comparison/
├── cnn_policy/
│   ├── model.py                    # CNN architecture
│   ├── train.py                    # Training loop
│   ├── inference.py                # Move prediction
│   ├── position_encoder.py        # FEN → 18×8×8 tensor
│   ├── dataset_h5_proper.py        # Data loading
│   ├── config.py                   # Hyperparameters
│   ├── checkpoints/
│   │   └── checkpoint_step_55000.pth
│   └── logs/
│       ├── training_log.csv
│       └── plots.png
│
├── transformer_policy/
│   ├── model.py                    # Transformer architecture
│   ├── train.py                    # Training loop
│   ├── inference.py                # Move prediction
│   ├── dataset.py                  # Data loading
│   ├── config.py                   # Hyperparameters
│   ├── checkpoints/
│   │   └── checkpoint_step_55000.pth
│   └── logs/
│       ├── training_log.csv
│       └── plots.png
│
├── dataset/
│   └── raw/LE22ct/
│       └── LE22ct.h5               # Training data
│
├── predict_move.py                 # Unified inference script
├── ARCHITECTURE_ANALYSIS.md        # Conceptual comparison
├── TECHNICAL_DOCUMENTATION.md      # Complete technical reference
└── README.md                       # This file
```

---

##  Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/chess_policy_comparison.git
cd chess_policy_comparison

# Create environment
conda create -n chess_policy python=3.10
conda activate chess_policy

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install python-chess h5py numpy pandas tqdm tensorboard matplotlib
```

### Download Dataset

The LE22ct dataset (13.3M positions) should be placed in:
```
dataset/raw/LE22ct/LE22ct.h5
```

### Training

```bash
# Train CNN model
python cnn_policy/train.py

# Train Transformer model
python transformer_policy/train.py

# Monitor with TensorBoard
tensorboard --logdir cnn_policy/logs/tensorboard
tensorboard --logdir transformer_policy/logs/tensorboard
```

### Inference

```bash
# Predict move with CNN
python predict_move.py --model cnn --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Predict move with Transformer
python predict_move.py --model transformer --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Top-3 sampling (stochastic)
python predict_move.py --model cnn --k 3

# Custom checkpoint
python predict_move.py --model cnn --checkpoint path/to/checkpoint.pth
```

---

##  Documentation

### Complete Documentation Files

1. **[ARCHITECTURE_ANALYSIS.md](ARCHITECTURE_ANALYSIS.md)** (335 lines)
   - Conceptual comparison of architectures
   - How each model "thinks" about chess
   - Inductive biases and their implications
   - Predictions for behavioral differences

2. **[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)** (1,350 lines)
   - Complete architectural specifications
   - Layer-by-layer breakdowns
   - Parameter counts and memory footprints
   - Training procedures and results
   - Inference pipelines with code
   - Reproducibility guide

### Key Architectural Differences

| Aspect | CNN | Transformer |
|--------|-----|-------------|
| **Input** | 18×8×8 grid | 69-token sequence |
| **Inductive Bias** | Spatial locality, translation equivariance | None (learned through attention) |
| **Information Flow** | Local → global through layers | Global from first layer |
| **Receptive Field** | Grows with depth (3×3 → full board) | Full board at all layers |
| **Parameters** | 17.8M | 19.0M |
| **Computation** | O(n²d) per conv layer | O(n²d) per attention layer |

---

##  Evaluation (Coming Soon)

### Planned Evaluation Methods

1. **Lichess Bot Integration**
   - Deploy both models as Lichess bots
   - Play rated games against human players
   - Estimate Elo ratings

2. **Head-to-Head Matches**
   - CNN vs Transformer (100+ games)
   - Different time controls
   - Various opening positions

3. **Tactical Puzzles**
   - Standard chess puzzle sets
   - Measure tactical accuracy
   - Compare pattern recognition

4. **Positional Analysis**
   - Strategic positions without immediate tactics
   - Endgame technique
   - Opening knowledge

5. **Error Analysis**
   - Blunder rate comparison
   - Failure mode identification
   - Position type strengths/weaknesses

---

##  Requirements

```
Python >= 3.10
PyTorch >= 2.1.0
python-chess >= 1.999
h5py >= 3.9.0
numpy >= 1.24.0
pandas >= 2.0.0
tqdm >= 4.66.0
tensorboard >= 2.14.0
matplotlib >= 3.7.0
```

Full requirements in `requirements.txt`

---

##  Key Implementation Features

### CNN-Specific
- AlphaZero-style residual architecture
- Preserves 8×8 spatial structure
- Position encoder: FEN → multi-channel tensor
- Batch normalization after every convolution
- Dropout2d in residual blocks

### Transformer-Specific
- Encoder-only (no decoder needed)
- Pre-layer normalization
- Learned positional embeddings
- Multi-head self-attention (8 heads)
- Position-wise feed-forward (512 → 2048 → 512)

### Common Features
- From-to square factorization
- Mixed precision training (BF16)
- Vaswani learning rate schedule
- Label smoothing (ε=0.1)
- Checkpoint averaging (last 10)
- Comprehensive logging
- TensorBoard integration

---

##  Research Context

### Inspiration

- **AlphaZero** (DeepMind, 2017): CNN-based policy and value networks
- **"Attention is All You Need"** (Vaswani et al., 2017): Transformer architecture

### Novel Contributions

1. **Controlled comparison** with matched training conditions
2. **From-to factorization** instead of full UCI move prediction
3. **Dual implementation** showing both architectural approaches
4. **Comprehensive documentation** of design decisions

---

##  License

This project is released under MIT License. See LICENSE file for details.

---

##  Acknowledgments

- **Lichess.org** - Elite game database
- **python-chess library** - Chess logic and utilities
- **PyTorch team** - Deep learning framework
- **DeepMind AlphaZero team** - CNN architecture inspiration

- [ ] Extended documentation with results

---

**Last Updated:** January 2025

