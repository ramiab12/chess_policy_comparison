# 🤖 AI AGENT CONTEXT - RUNPOD TRAINING SESSION

**CRITICAL: Read this ENTIRE file before doing anything!**

This document provides complete context for an AI agent helping with transformer training on RunPod.

---

## 📋 PROJECT OVERVIEW

### **Goal**
Scientifically compare CNN vs Transformer architectures for chess move prediction.

### **Status**
- ✅ CNN trained (50.81% accuracy at step 55K, beats CT-EFT-20 baseline)
- ⏳ Transformer ready to train (CT-EFT-20 replica, 100K steps)
- ✅ All code tested and optimized for A6000
- ✅ Git repository: https://github.com/ramiab12/chess_policy_comparison
- ✅ Switched from H100 to A6000 for cost efficiency

---

## 🎯 CURRENT TASK

**TRAIN THE TRANSFORMER MODEL ON RUNPOD RTX A6000 GPU**

The transformer model is an **EXACT REPLICA** of CT-EFT-20 (from chess-transformers paper).
It must be trained with IDENTICAL settings to ensure fair comparison with CNN.

**Why A6000 instead of H100?**
- 19M parameter model is too small to utilize H100 (only 5-10% GPU usage)
- A6000 provides same speed (~0.50-0.55 steps/sec) at **82% lower cost**
- H100: $2.69/hr → A6000: $0.49/hr
- Estimated savings: $121 for 100K steps!

---

## 📁 PROJECT STRUCTURE

```
chess_policy_comparison/
├── cnn_policy/                    # ✅ TRAINED (50.81% accuracy)
│   ├── config.py
│   ├── model.py
│   ├── train.py
│   ├── dataset_h5_proper.py       # Direct H5 loading
│   ├── position_encoder.py
│   ├── inference.py
│   └── checkpoints/
│       ├── checkpoint_step_55000.pth   (213 MB, best: 50.81%)
│       └── checkpoint_step_105000.pth  (213 MB, final)
│
├── transformer_policy/            # ⏳ READY TO TRAIN
│   ├── config.py                  # Hyperparameters (100K steps, A6000 optimized)
│   ├── model.py                   # ChessTransformerEncoderFT (~19M params)
│   ├── dataset.py                 # ChessDatasetFT (H5 loader, uses h5py)
│   ├── train.py                   # Training loop (Vaswani LR, label smoothing)
│   └── inference.py               # Move prediction
│
├── dataset/                       # DATA LOCATION
│   └── raw/LE22ct/
│       └── LE22ct.h5              # 13.3M positions (4.5 GB)
│
├── scripts/
│   ├── download_le22ct.py         # Downloads LE22ct.h5
│   └── predict_move.py            # CNN inference script
│
├── evaluation/
│   ├── play_vs_stockfish.py       # Evaluation protocol
│   └── compare_to_ct_eft_20.py    # Compare to baseline
│
├── requirements.txt               # All dependencies
├── README.md                      # Main documentation
├── FINAL_RUNPOD_GUIDE.md         # RunPod setup guide
├── PROJECT_CONTEXT.md            # Technical details
├── START_HERE.md                 # Quick start
└── HOW_CT_EFT_20_USES_DATA.md   # Dataset usage explanation
```

---

## 🔧 TRANSFORMER MODEL DETAILS

### **Architecture**
- **Type**: Encoder-only Transformer (like BERT, not GPT)
- **Input**: 70-token sequence (5 metadata + 64 board squares + 1 padding)
- **Embedding**: Each token → 512-dim vector
- **Layers**: 6 transformer encoder blocks
- **Attention**: 8 heads per layer, 64-dim Q/K/V per head
- **FFN**: 512 → 2048 → 512 with ReLU
- **Output**: From-square (64 logits) + To-square (64 logits)
- **Parameters**: ~19M (18,963,970)

### **Training Configuration**
```python
# Model Architecture (IDENTICAL to CT-EFT-20)
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
D_QUERIES = 64
D_VALUES = 64
D_INNER = 2048
DROPOUT = 0.1

# Training (A6000 Optimized)
BATCH_SIZE = 512            # Optimal for A6000 (tested fastest)
BATCHES_PER_STEP = 4        # Gradient accumulation for effective batch 2048
N_STEPS = 100_000           # Match CT-EFT-20 baseline (fair comparison)
WARMUP_STEPS = 8_000        # IDENTICAL to CT-EFT-20

# Learning Rate (Vaswani Schedule)
lr(step) = 512^(-0.5) × min(step^(-0.5), step × 8000^(-1.5))
Peak LR ≈ 0.044194 (at step 8000)

# Optimizer (IDENTICAL to CT-EFT-20)
Adam(β1=0.9, β2=0.98, ε=1e-9)

# Loss (IDENTICAL to CT-EFT-20)
Label Smoothed Cross-Entropy (ε=0.1)
Total Loss = From-Loss + To-Loss

# A6000 Optimizations
Mixed Precision: BF16 (autocast + GradScaler)
Num Workers: 8
Pin Memory: True
Expected Speed: ~0.50-0.55 steps/sec
```

### **Dataset**
- **Name**: LE22ct (Lichess Elite 2022, checkmate games)
- **File**: `dataset/raw/LE22ct/LE22ct.h5` (4.5 GB)
- **Format**: HDF5 with PyTables
- **Samples**: 13,287,522 positions
- **Split**: Train/Val at index 11,958,770 (90/10 split)
- **Encoding**: Pre-encoded (pieces as indices 0-12, castling as 0-1)

### **Data Fields**
```python
{
    "turns": (N, 1)                               # 0=white, 1=black
    "white_kingside_castling_rights": (N, 1)      # 0=no, 1=yes
    "white_queenside_castling_rights": (N, 1)
    "black_kingside_castling_rights": (N, 1)
    "black_queenside_castling_rights": (N, 1)
    "board_positions": (N, 64)                    # Piece indices 0-12
    "from_squares": (N, 1)                        # Target: 0-63
    "to_squares": (N, 1)                          # Target: 0-63
    "lengths": (N, 1)                             # Always 1
}
```

---

## 🚀 RUNPOD TRAINING WORKFLOW

### **Step 1: Setup RunPod Pod**
```bash
# GPU: RTX A6000 (48 GB VRAM)
# Cost: $0.49/hr
# Disk: 80 GB minimum

# Clone repository
cd /workspace
git clone https://github.com/ramiab12/chess_policy_comparison.git
cd chess_policy_comparison

# Install dependencies
pip install -r requirements.txt
# Key packages: torch, tables (PyTables), python-chess, pandas, tqdm
```

### **Step 2: Download Dataset**
```bash
# Download LE22ct.h5 (4.5 GB)
python scripts/download_le22ct.py

# Should create: dataset/raw/LE22ct/LE22ct.h5
# Verify: ls -lh dataset/raw/LE22ct/LE22ct.h5
```

### **Step 3: Verify Setup**
```bash
# Test transformer loads correctly
python -c "from transformer_policy.config import TransformerConfig; TransformerConfig.print_config()"

# Test model initializes
python -c "from transformer_policy.model import ChessTransformerEncoderFT; from transformer_policy.config import TransformerConfig; import torch; model = ChessTransformerEncoderFT(TransformerConfig); print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')"

# Test dataset loads
python -c "from transformer_policy.dataset import ChessDatasetFT; ds = ChessDatasetFT('dataset', 'raw/LE22ct/LE22ct.h5', 'train'); print(f'Dataset size: {len(ds):,}'); sample = ds[0]; print('Sample keys:', list(sample.keys()))"
```

### **Step 4: Start Training**
```bash
cd transformer_policy
python train.py
```

### **Expected Output**
```
======================================================================
Transformer Policy Configuration (CT-EFT-20 Replica)
======================================================================
...

🖥️  Using device: cuda

📦 Initializing Transformer model...
   Parameters: 18,963,970

📂 Loading datasets...
   Train samples: 11,958,770
   Val samples: 1,328,752
   Train batches: 5,839
   Val batches: 649

🚀 Starting Transformer Training (CT-EFT-20 Replica)
======================================================================
Total steps: 325,000
Effective batch: 2048

Training: 100%|████████| 325000/325000 [XX:XX:XX<00:00, X.XXit/s]
```

### **Step 5: Monitor Training**
```bash
# In another terminal/tab
tensorboard --logdir transformer_policy/logs/tensorboard --host 0.0.0.0
```

Metrics tracked:
- From-square loss
- To-square loss
- Combined loss
- Learning rate
- From-square accuracy
- To-square accuracy
- **Move accuracy** (both from AND to correct) ← **PRIMARY METRIC**

### **Step 6: Checkpoints**
Saved every 5,000 steps to: `transformer_policy/checkpoints/`
- `checkpoint_step_5000.pth`
- `checkpoint_step_10000.pth`
- ...
- `checkpoint_step_325000.pth`
- `averaged_final.pth` (average of last 10 checkpoints)

### **Step 7: Early Stopping**
Training will notify when early stopping is triggered:
```
⚠️  Early stopping triggered!
   No improvement for 10 validations
   Best accuracy: XX.XX% at step XXX,XXX
   You can stop training now or let it continue.
```

**Decision**: Stop training manually (Ctrl+C) or let it continue to 325K steps.

---

## 📊 EXPECTED RESULTS

### **CNN Baseline (Already Trained)**
- Step 55,000: **50.81% move accuracy**
- Step 105,000: **50.88% move accuracy**
- Training time: ~3.5 hours on H100

### **Transformer Target**
- **Goal**: Match or beat CNN performance (>50.81%)
- **CT-EFT-20 Baseline**: ~48% (from paper, 100K steps)
- **Our Training**: 100K steps (fair comparison with CT-EFT-20)
- **Expected**: ~48-50% based on CT-EFT-20 paper results

### **Training Time Estimate**
- Total steps: 100,000
- Batch size: 512 (effective 2048 with grad accum)
- A6000 with BF16: ~0.50-0.55 steps/sec
- **Estimated time**: ~50-55 hours (~2.1-2.3 days)
- **Cost**: ~52 hours × $0.49/hr = **~$25-27**
- **Comparison**: H100 would cost $140+ for same training!

---

## 🔍 VERIFICATION CHECKLIST

Before training, verify:

### **Dependencies**
```bash
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import tables; print('✅ PyTables:', tables.__version__)"
python -c "import chess; print('✅ python-chess:', chess.__version__)"
python -c "import torch; print('✅ CUDA:', torch.cuda.is_available())"
```

### **Dataset**
```bash
ls -lh dataset/raw/LE22ct/LE22ct.h5
# Should show: ~4.5 GB file
```

### **Model**
```bash
python -c "from transformer_policy.model import ChessTransformerEncoderFT; from transformer_policy.config import TransformerConfig; import torch; model = ChessTransformerEncoderFT(TransformerConfig).cuda(); print('✅ Model on GPU'); print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')"
```

### **Disk Space**
```bash
df -h /workspace
# Need: 4.5 GB (dataset) + 10 GB (checkpoints) + 5 GB (logs) ≈ 20 GB minimum
```

---

## ⚠️ IMPORTANT NOTES

### **IDENTICAL to CT-EFT-20**
The transformer implementation is an **EXACT COPY** of CT-EFT-20 from the chess-transformers repository:
- Same model architecture (6 layers, 512 d_model, 8 heads)
- Same loss function (label-smoothed CE)
- Same optimizer (Adam with β=(0.9, 0.98))
- Same LR schedule (Vaswani)
- Same dataset (LE22ct H5)
- **ONLY DIFFERENCES**: More steps (325K vs 100K), H100 optimizations (batch size 2048, BF16)

### **Fair Comparison**
Both CNN and Transformer:
- Train on same dataset (LE22ct, 13.3M positions)
- Same total steps (325,000)
- Same effective batch size (2048)
- Same LR schedule (Vaswani)
- Same loss function (label-smoothed CE, ε=0.1)
- Same evaluation (vs Stockfish)
- Same optimizations (BF16, H100)

### **H5 Dataset Loading**
- Uses **PyTables** (`tables` library), not `h5py`
- Reads from `encoded_data` table (pre-encoded)
- Already has piece indices (0-12), castling (0-1)
- Directly reads `from_square` and `to_square` targets (0-63)

### **Git LFS**
Checkpoint files use Git LFS:
- Local files are LFS pointers (134 bytes)
- Actual files on GitHub (213 MB each)
- Download with: `git lfs pull`

---

## 🐛 TROUBLESHOOTING

### **"ModuleNotFoundError: No module named 'tables'"**
```bash
pip install tables
# Or: conda install pytables
```

### **"FileNotFoundError: H5 file not found"**
```bash
python scripts/download_le22ct.py
# Verify: ls dataset/raw/LE22ct/LE22ct.h5
```

### **"CUDA out of memory"**
- Reduce batch size in `transformer_policy/config.py`
- Current: 2048 (uses ~50 GB VRAM on H100)
- Try: 1024 or 512

### **Training too slow**
- Verify BF16 is enabled: `USE_AMP = True` in config
- Check GPU utilization: `nvidia-smi`
- Should see ~90%+ GPU utilization

### **Loss is NaN**
- Usually due to learning rate too high
- Transformer uses Vaswani schedule (starts low, warms up)
- Should NOT happen with current config

---

## 📈 MONITORING METRICS

### **Primary Metric: Move Accuracy**
- Both from_square AND to_square must be correct
- Reported every 5,000 steps during validation
- **Target**: >50.81% (to beat CNN)

### **Secondary Metrics**
- From-square accuracy: Usually ~70-75%
- To-square accuracy: Usually ~70-75%
- Move accuracy = from_acc × to_acc ≈ 50% (combined probability)

### **Loss Values**
- Initial loss: ~4.0-4.5
- Converged loss: ~2.5-3.0
- Loss should decrease smoothly (no spikes)

---

## 🎯 SUCCESS CRITERIA

Training is successful if:
1. ✅ Completes 325K steps (or early stops with good accuracy)
2. ✅ Move accuracy > 50% on validation set
3. ✅ Loss converges smoothly without NaN
4. ✅ Checkpoints saved every 5K steps
5. ✅ `averaged_final.pth` created at end

**GOAL**: Transformer move accuracy **>50.81%** to beat CNN baseline!

---

## 💬 COMMUNICATION WITH USER

When reporting progress:
1. **Every 5K steps**: Report move accuracy
2. **Significant milestones**: 50%, 75%, 100% completion
3. **Best accuracy updates**: When validation accuracy improves
4. **Early stopping**: When patience counter increases
5. **Errors**: Immediately report any errors with context

Example update:
```
📊 Step 50,000/325,000 (15.4% complete)
   From accuracy:  72.34%
   To accuracy:    73.12%
   Move accuracy:  52.89% ⭐ (NEW BEST! Beats CNN!)
   LR:             0.019748
   Time elapsed:   1h 32m
```

---

## 📚 REFERENCE FILES

If you need more details, check these files in the repository:

1. **FINAL_RUNPOD_GUIDE.md** - Complete RunPod setup guide
2. **PROJECT_CONTEXT.md** - Technical architecture details
3. **HOW_CT_EFT_20_USES_DATA.md** - Dataset format explanation
4. **README.md** - Project overview
5. **transformer_policy/config.py** - All hyperparameters
6. **transformer_policy/train.py** - Training loop implementation

---

## 🔐 CRITICAL RULES

1. **DO NOT** modify model architecture (must match CT-EFT-20)
2. **DO NOT** change hyperparameters (except debugging)
3. **DO NOT** use different dataset
4. **DO NOT** skip validation or checkpointing
5. **DO** monitor training closely
6. **DO** report progress regularly
7. **DO** save all checkpoints
8. **DO** verify setup before starting

---

## ✅ QUICK START (TL;DR)

```bash
# 1. Setup
cd /workspace
git clone https://github.com/ramiab12/chess_policy_comparison.git
cd chess_policy_comparison
pip install -r requirements.txt

# 2. Download data
python scripts/download_le22ct.py

# 3. Verify
python -c "from transformer_policy.config import TransformerConfig; TransformerConfig.print_config()"

# 4. Train
cd transformer_policy
python train.py

# 5. Monitor (separate terminal)
tensorboard --logdir logs/tensorboard --host 0.0.0.0
```

**Expected time**: 10-12 hours  
**Expected cost**: ~$27-32 on H100  
**Target accuracy**: >50.81% (to beat CNN)

---

## 🎓 BACKGROUND KNOWLEDGE

### **Why Transformer vs CNN?**
- **CNN**: Treats chess as spatial/image problem (8×8 grid)
- **Transformer**: Treats chess as sequence/relational problem (70 tokens)
- **Key difference**: Transformer has global view from layer 1, CNN builds up receptive field gradually
- **Question**: Which is better for chess move prediction?

### **CT-EFT-20**
- Published model from chess-transformers paper
- "CT" = Chess Transformer
- "EFT" = Encoder From-To (predicts from/to squares separately)
- "20" = ~20M parameters
- Baseline: ~48% move accuracy (100K steps)
- Our replica: Same code, more training (325K steps)

### **LE22ct Dataset**
- Lichess Elite 2022, checkmate games only
- Elite players (Elo > 2000)
- Checkmate games (high-quality, decisive)
- 13.3M positions
- Pre-encoded in H5 format

---

**END OF CONTEXT FILE**

**Last Updated**: October 15, 2025  
**Author**: Rami AB  
**Repository**: https://github.com/ramiab12/chess_policy_comparison  
**Contact**: [Your contact if needed]

Good luck with training! 🚀

