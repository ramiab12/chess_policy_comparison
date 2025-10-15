# 📥 Download Trained CNN Checkpoint

## ⚠️ Checkpoint Not in Git (Too Large)

The trained CNN checkpoint (`checkpoint_step_55000.pth`) is **213 MB** - too large for regular Git.

**Current status:** Git LFS pointer only (134 bytes)

---

## 🔧 How to Get the Checkpoint

### **Option 1: Download from Your Training Server (RunPod)**

```bash
# On RunPod (where you trained):
cd /workspace/chess_policy_comparison/cnn_policy/checkpoints

# Copy to your local machine:
scp -P <PORT> root@<RUNPOD_IP>:/workspace/chess_policy_comparison/cnn_policy/checkpoints/checkpoint_step_55000.pth \
    ~/chess_policy_comparison/cnn_policy/checkpoints/
```

### **Option 2: Git LFS**

If you set up Git LFS:
```bash
cd chess_policy_comparison
git lfs pull
```

### **Option 3: Upload to GitHub Releases**

Since file is 213 MB:
1. Go to https://github.com/ramiab12/chess_policy_comparison/releases
2. Create new release
3. Upload `checkpoint_step_55000.pth` as asset
4. Others can download from releases page

### **Option 4: Use Cloud Storage**

Upload to Google Drive/Dropbox/etc and share link.

---

## 📊 Checkpoint Info

**Best Checkpoint:**
- File: `checkpoint_step_55000.pth`
- Step: 55,000
- Accuracy: 50.81% (peak performance)
- Size: 213 MB
- Parameters: 17,753,096

**Alternative:**
- File: `checkpoint_step_105000.pth`  
- Step: 105,000
- Accuracy: 50.55%
- Size: ~213 MB

---

## 🚀 Once You Have the Checkpoint

### **Test the inference script:**

```bash
# With starting position
python scripts/predict_move.py

# With custom position
python scripts/predict_move.py "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"

# With complex position
python scripts/predict_move.py "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
```

---

## 💡 For Distribution

**If you want to share your trained model:**

1. **Create GitHub Release** with checkpoint as asset
2. **Or use Hugging Face Hub** for model hosting
3. **Or Google Drive** with public link

Then update `predict_move.py` to auto-download if checkpoint missing!


