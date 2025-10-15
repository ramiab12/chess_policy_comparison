# RTX A6000 Setup Guide

## ✅ Configuration Complete!

The repository is now optimized for **RTX A6000** training.

## 🎯 Quick Start on A6000

### 1. Setup Pod
```bash
cd /workspace
git clone https://github.com/ramiab12/chess_policy_comparison.git
cd chess_policy_comparison
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python scripts/download_le22ct.py
# OR if you have it: upload LE22ct.h5 to dataset/raw/LE22ct/
```

### 3. Verify Setup
```bash
python -c "from transformer_policy.config import TransformerConfig; TransformerConfig.print_config()"
```

### 4. Start Training
```bash
cd transformer_policy
screen -dmS transformer_training python train.py
screen -r transformer_training  # Attach to see progress
# Ctrl+A then D to detach
```

## 📊 A6000 Configuration

**Optimized Settings:**
- Batch size: 512
- Gradient accumulation: 4 (effective batch 2048)
- Workers: 8
- Mixed precision: BF16
- Total steps: 100,000

**Expected Performance:**
- Speed: ~0.50-0.55 steps/sec
- Time: ~50-55 hours
- Cost: ~$25-27

**GPU Utilization:**
- Memory: ~15-20GB / 48GB
- Utilization: ~30-40% (normal for this model size)

## 🔍 Why These Settings?

Tested on H100 and found:
- **Batch 512** is optimal (fastest per-step time)
- **Grad accum 4** maintains effective batch of 2048 (same as CT-EFT-20)
- **8 workers** provides good data pipeline without overhead
- **h5py** for dataset loading (better multiprocessing than PyTables)

## 📈 Monitoring

```bash
# Check progress
screen -r transformer_training

# View logs
tail -f training_output.log

# Check checkpoints
ls -lth transformer_policy/checkpoints/

# GPU usage
nvidia-smi
```

## 🎯 Success Criteria

- ✅ Loss decreases smoothly
- ✅ Speed: ~0.50-0.55 steps/sec
- ✅ Checkpoints every 5K steps
- ✅ Target: >48% move accuracy (beat CT-EFT-20 baseline)
- 🎯 Stretch goal: >50.81% (beat CNN!)

## 💡 Cost Comparison

| GPU | Cost/hr | 100K steps | Savings |
|-----|---------|------------|---------|
| H100 | $2.69 | ~$140 | Baseline |
| **A6000** | **$0.49** | **~$27** | **Save $113!** |
| RTX 5090 | $0.89 | ~$48 | Save $92 |

## 🚨 Troubleshooting

**OOM Error?**
- Reduce batch size to 256
- Increase grad accum to 8

**Slow speed?**
- Check `nvidia-smi` - should see ~15-20GB usage
- Verify h5py installed: `pip list | grep h5py`

**Dataset not found?**
- Run: `python scripts/download_le22ct.py`
- Verify: `ls dataset/raw/LE22ct/LE22ct.h5`

---

**Ready to train!** 🚀

