# 📦 Checkpoint Files Notice

## ⚠️ Git LFS Required

The checkpoint files are stored using **Git LFS** (Large File Storage).

### **Checkpoint Info:**
- `checkpoint_step_55000.pth` - Best model (50.81% accuracy)
- Size: 213 MB
- Currently: Git LFS pointer (134 bytes)

### **To Download Actual Checkpoint:**

**Option 1: Git LFS (if you have it)**
```bash
git lfs pull
```

**Option 2: Download from RunPod/Server**
The actual trained checkpoint should be downloaded from where you trained it.

**Option 3: Retrain**
```bash
python cnn_policy/train.py
# Will create checkpoints in cnn_policy/checkpoints/
```

### **For Inference Script:**

The `predict_move.py` script will automatically check if checkpoint exists.

```bash
# If you have the checkpoint:
python scripts/predict_move.py "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# If checkpoint missing:
# Download it first or retrain the model
```
