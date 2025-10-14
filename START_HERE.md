# 🚀 START HERE - For New AI Agent

**Project:** CNN vs CT-EFT-20 Chess Policy Comparison  
**Status:** ✅ Production Ready, Optimized for H100  
**Location:** RunPod H100 SXM

---

## 📖 Read These Files First

1. **THIS FILE** - Quick overview
2. **FINAL_RUNPOD_GUIDE.md** - Complete setup guide ⭐
3. **PROJECT_CONTEXT.md** - Full technical context
4. **HOW_CT_EFT_20_USES_DATA.md** - Fairness guarantee

---

## ⚡ Quick Summary

### **What This Is:**
Scientific comparison of CNN vs Transformer for chess move prediction.

### **Current State:**
- ✅ All code complete and tested
- ✅ H5 direct loading (matches CT-EFT-20 exactly!)
- ✅ H100 optimizations applied (BF16, batch 2048)
- ✅ 50 epochs training (325K steps)
- ✅ Early stopping implemented
- ✅ Ready to transfer to RunPod

### **What User Needs:**
Transfer to RunPod H100 and train the model.

---

## 🎯 Key Facts

| Aspect | Value |
|--------|-------|
| **Model** | CNN, 17.8M params, 15 ResBlocks |
| **Dataset** | LE22ct H5 (13.3M positions) |
| **Training** | 325K steps, BF16, batch 2048 |
| **Expected Time** | ~19h (likely early stop at 150K) |
| **Expected Cost** | ~$51 on H100 |
| **Expected Accuracy** | ~57-58% (vs CT-EFT-20's 48%) |
| **Fairness** | Uses moves[0] like CT-EFT-20 ✅ |

---

## 🚀 RunPod Commands (Copy-Paste)

```bash
# Setup
cd /workspace
tar -xzf chess_policy_comparison_CLEAN.tar.gz
cd chess_policy_comparison
pip install -r requirements.txt

# Download dataset
python scripts/download_le22ct.py

# Train!
python cnn_policy/train.py

# Monitor (another terminal)
tail -f cnn_policy/logs/training_log.csv
```

---

## 📂 File Structure

```
chess_policy_comparison/
├── cnn_policy/          # 7 Python files - model code
├── evaluation/          # 2 Python files - evaluation
├── scripts/             # 5 Python files - dataset tools
├── README.md
├── FINAL_RUNPOD_GUIDE.md       ⭐ READ THIS
├── PROJECT_CONTEXT.md          ⭐ CONTEXT
├── HOW_CT_EFT_20_USES_DATA.md  ⭐ FAIRNESS
├── START_HERE.md               (this file)
└── requirements.txt
```

**Clean: 18 files total**

---

## ✅ What's Configured

**Optimizations:**
- BF16 mixed precision (2x faster on H100)
- Batch size 2048 (no gradient accumulation)
- 8 workers (fast data loading)
- Early stopping (stop when converged)

**Training:**
- 325K steps max (50 epochs)
- Likely stops at 150K (~19 hours)
- Validates every 5K steps
- Saves checkpoints every 5K steps

**Dataset:**
- Reads LE22ct.h5 directly
- Uses moves[0] (first move) as target
- Same as CT-EFT-20!

---

## 🎯 User's Goal

Get a trained CNN model that:
1. Beats CT-EFT-20 in accuracy (~57% vs ~48%)
2. Fair scientific comparison
3. Train on RunPod H100
4. Cost ~$60 total
5. Time ~1 day

**Everything is ready!**

---

**Next:** Read FINAL_RUNPOD_GUIDE.md for complete instructions! 🚀

