# 🎯 Chess Policy Comparison: CNN vs CT-EFT-20 Transformer

Fair scientific comparison of CNN and Transformer architectures for chess move prediction.

---

## 📊 Project Overview

###  **Research Question:**
Do CNNs outperform Transformers at chess move prediction when given equal resources?

### **Comparison:**

| Model | Architecture | Parameters | Dataset | Task |
|-------|-------------|------------|---------|------|
| **CT-EFT-20** (baseline) | Transformer (6×512) | 20M | LE22ct (13.3M) | From-To prediction |
| **CNN Policy** (ours) | ResNet (15×256) | 18M | LE22ct (13.3M) | From-To prediction |

**Baseline Performance (CT-EFT-20):**
- Move accuracy: ~48%
- Playing strength: 1750-1850 ELO (measured vs Stockfish)

**Expected CNN Performance:**
- Move accuracy: 54-60%
- Playing strength: 1900-2100 ELO

---

## ✅ **Implementation Status**

### **Completed:**
- [x] CNN policy architecture (from-to prediction)
- [x] Training configuration (matching CT-EFT-20)
- [x] Vaswani LR schedule implementation
- [x] Gradient accumulation (4 batches, effective=2048)
- [x] Label smoothing (0.1)
- [x] Dataset downloaders
- [x] Position encoder (18-channel)

### **To Do:**
- [ ] Download LE22ct dataset OR generate from Lichess Elite
- [ ] Preprocess to CSV format
- [ ] Train CNN (100K steps, ~33 hours on H100)
- [ ] Evaluate vs Stockfish (match CT-EFT-20 protocol)
- [ ] Compare results

---

## 🔧 **Fair Comparison Protocol**

### **MUST Match CT-EFT-20:**

✅ **Dataset:**
- Source: LE22ct (Lichess Elite 2400+ vs 2200+, checkmate games)
- Size: 13,287,522 positions
- Split: Same train/val split

✅ **Training:**
- Steps: 100,000
- Batch: 512
- Gradient accumulation: 4 (effective batch = 2,048)
- Total samples: 204.8M (~15.4 epochs)
- LR schedule: Vaswani (d_model^-0.5)
- Warmup: 8,000 steps
- Optimizer: Adam (β1=0.9, β2=0.98, ε=1e-9)
- Label smoothing: 0.1
- Checkpoint averaging: Last 10 checkpoints

✅ **Evaluation:**
- Metrics: Top-1/3/5 accuracy
- Games: 1000 vs Fairy Stockfish per level (levels 1-6)
- Hardware: Document CPU specs
- Settings: Same as CT-EFT-20

---

## 🚀 **Quick Start**

### **1. Download Dataset**

**Option A: Official LE22ct (Best)**
```bash
python scripts/download_le22ct.py
python scripts/convert_le22ct_to_csv.py
```

**Option B: Generate Equivalent**
```bash
# Download Lichess Elite PGNs from https://database.nikonoel.fr/
python scripts/create_elite_moves_dataset.py \
  --pgn-dir dataset/raw/lichess_elite \
  --target 13000000
```

### **2. Train CNN**

```bash
# On RunPod H100
python cnn_policy/train.py

# Expected time: ~33 hours
# Expected cost: ~$89
```

### **3. Evaluate**

```bash
python evaluation/compare_to_ct_eft_20.py
```

---

## 📊 **Training Details**

### **CNN Architecture:**

```
Input (18×8×8)
├─ Conv (18→256, 3×3) + BN + ReLU
├─ ResBlock × 15 (256→256)
├─ From-head: Conv1×1 (256→2) → 64 logits
└─ To-head: Conv1×1 (256→2) → 64 logits

Parameters: ~18M
Task: Predict from-square (64 classes) + to-square (64 classes)
Loss: Cross-Entropy with label smoothing 0.1
```

### **Training Configuration:**

```python
Steps: 100,000
Effective batch: 2,048 (512 × 4 accumulation)
LR schedule: Vaswani
  - Warmup: 0 → peak over 8K steps
  - Decay: sqrt after warmup
  - Peak LR: ~0.0625 (256^-0.5)
Optimizer: Adam (β1=0.9, β2=0.98)
Label smoothing: 0.1
Checkpointing: Every 5K steps
Final model: Average of last 10 checkpoints
```

---

## 🎯 **Expected Results**

### **Performance Prediction:**

| Metric | CNN Policy | CT-EFT-20 | Difference |
|--------|------------|-----------|------------|
| **Top-1 Accuracy** | 54-60% | 48% | +6-12% |
| **Top-3 Accuracy** | 74-80% | ~70% | +4-10% |
| **vs Stockfish L3** | 98%+ | 96.6% | +1-2% |
| **vs Stockfish L5** | 68-75% | 55.2% | +13-20% |
| **Estimated ELO** | 1950-2100 | 1750-1850 | +150-300 |

### **Research Conclusion:**

*CNNs outperform Transformers by X% in move prediction accuracy due to spatial inductive bias, achieving approximately Y ELO advantage on chess tasks.*

---

## 💰 **Cost Estimate**

### **On RunPod H100 ($2.70/hr):**

| Phase | Time | Cost |
|-------|------|------|
| Dataset download | 1h | $3 |
| Dataset preprocessing | 4h | $11 |
| CNN training | 33h | $89 |
| Evaluation games | 10h | $27 |
| **Total** | **48h** | **$130** |

---

## 📁 **Project Structure**

```
chess_policy_comparison/
├── cnn_policy/
│   ├── model.py              ✅ CNN architecture
│   ├── train.py              ✅ Training script
│   ├── config.py             ✅ Configuration
│   ├── dataset.py            ✅ Data loader
│   ├── position_encoder.py   ✅ FEN encoding
│   ├── checkpoints/          📂 Model weights
│   └── logs/                 📂 Training logs
│
├── evaluation/
│   ├── compare_to_ct_eft_20.py  📝 Comparison script
│   ├── play_vs_stockfish.py     📝 Game playing
│   └── results/                 📂 Results
│
├── dataset/
│   ├── raw/                  📂 LE22ct or Elite PGNs
│   └── processed/            📂 train.csv, val.csv
│
├── scripts/
│   ├── download_le22ct.py          ✅ Download LE22ct
│   ├── convert_le22ct_to_csv.py    ✅ H5 → CSV
│   └── create_elite_moves_dataset.py ✅ Alternative dataset
│
└── README.md                 ✅ This file
```

---

## 🔬 **Ensuring Fair Comparison**

### **Matched Variables:**

- ✅ Task: From-to square prediction
- ✅ Dataset: LE22ct (or equivalent elite games)
- ✅ Dataset size: ~13.3M positions
- ✅ Training compute: 100K steps × 2048 effective batch
- ✅ LR schedule: Vaswani with 8K warmup
- ✅ Label smoothing: 0.1
- ✅ Evaluation: 1000 games vs Stockfish per level

### **Tested Variables (Research Question):**

- ❓ Architecture: CNN vs Transformer
- ❓ Representation: 18×8×8 grid vs 70-token sequence
- ❓ Inductive bias: Spatial (CNN) vs None (Transformer)

---

## 📖 **Documentation**

- `README.md` - This file
- `FAIR_COMPARISON_GUIDE.md` - Detailed fairness protocol
- `CT_EFT_20_ANALYSIS.md` - Baseline model analysis
- `cnn_policy/model.py` - Architecture documentation

---

## 🎓 **Citation**

If you use this comparison in research:

```bibtex
@misc{chess_policy_comparison_2025,
  author = {Your Name},
  title = {CNN vs Transformer for Chess Move Prediction: A Fair Comparison},
  year = {2025},
  note = {Comparing against CT-EFT-20 baseline}
}
```

---

## 🙏 **Acknowledgments**

- **CT-EFT-20** by sgrvinod - Baseline transformer model
- **Lichess Elite Database** by nikonoel - Source of elite games
- **Lichess.org** - Platform and data

---

**Status:** Implementation complete, ready for dataset and training!

