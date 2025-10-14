# 🎯 How CT-EFT-20 Uses LE22ct Data (EXACT Match)

## ✅ Based on H5 Structure Analysis

### **H5 File Contains:**
```python
encoded table (13,287,522 rows):
  - board_position: Encoded board (64 values, piece at each square)
  - turn: 0 (white) or 1 (black)
  - white_kingside_castling_rights: boolean
  - white_queenside_castling_rights: boolean
  - black_kingside_castling_rights: boolean
  - black_queenside_castling_rights: boolean
  - moves: Array of 10 future moves (as indices 0-4095)
  - length: Number of valid moves in sequence
```

---

## 🎯 How CT-EFT-20 Uses This for Policy Learning

### **Input (Position):**

CT-EFT-20 converts H5 encoding to their 70-token sequence:
```python
# For Transformer:
1. Take board_position (64 squares)
2. Take turn, castling_rights
3. Convert to 70-token sequence
4. Feed to transformer
```

### **Target (Move to Predict):**

For policy (from-to) prediction:
```python
# Critical: They use moves[0] only!
target_move_index = row['moves'][0]  # First move in the sequence

# Convert to from/to squares:
move_uci = UCI_MOVES[target_move_index]  # e.g., "e2e4"
from_square = parse_uci(move_uci).from_square  # e.g., 12 (e2)
to_square = parse_uci(move_uci).to_square      # e.g., 28 (e4)

# Training targets:
target = (from_square, to_square)
```

### **What They Ignore:**
- ❌ `moves[1:9]` - Future moves (only for sequence models, not policy)
- ❌ `length` - Not needed for single move prediction

---

## ✅ How We Should Match CT-EFT-20

### **For Fair Comparison, We Must:**

**1. Use EXACT Same Data:**
- ✅ Same H5 file (LE22ct)
- ✅ Same rows (all 13.3M positions)
- ✅ Same train/val split

**2. Use EXACT Same Target:**
- ✅ Use `moves[0]` only (first move)
- ✅ Convert to from/to squares
- ✅ Ignore future moves

**3. Different Input Encoding (THIS IS THE RESEARCH QUESTION!):**
- CT-EFT-20: Converts board → 70-token sequence → Transformer
- Our CNN: Converts board → 18×8×8 grid → CNN
- **Both use same board_position data, different representations!**

---

## 🔧 Implementation Strategy

### **Option A: Direct H5 Reading** ⭐ **FAIREST**

Read H5 directly, decode to our 18-channel tensor:

```python
class ChessPolicyDatasetH5Proper(Dataset):
    def __getitem__(self, idx):
        row = h5_data[idx]
        
        # Decode board_position to FEN
        fen = decode_board(row['board_position'], row['turn'], 
                          row['castling_rights'])
        
        # Get first move (same as CT-EFT-20!)
        move_idx = row['moves'][0]
        move_uci = UCI_LABELS[move_idx]
        
        # Parse to from/to
        move = chess.Move.from_uci(move_uci)
        from_square = move.from_square
        to_square = move.to_square
        
        # Convert FEN to our 18-channel tensor
        position = self.encoder.fen_to_tensor(fen)
        
        return position, (from_square, to_square)
```

**Advantages:**
- ✅ Uses EXACT same H5 file as CT-EFT-20
- ✅ Uses EXACT same rows
- ✅ Uses EXACT same move targets (moves[0])
- ✅ Most fair comparison possible!

---

### **Option B: Convert to CSV Once**

Convert H5 → CSV for easier handling:

```python
# One-time conversion:
for row in h5_data:
    fen = decode_board(row)
    move = UCI_LABELS[row['moves'][0]]  # First move only!
    from_sq = parse(move).from_square
    to_sq = parse(move).to_square
    
    csv_rows.append({
        'fen': fen,
        'move': move,
        'from_square': from_sq,
        'to_square': to_sq
    })

# Save to train.csv, val.csv
```

**Advantages:**
- ✅ Same data as Option A
- ✅ Easier to verify
- ✅ Simpler code

---

## 🎯 Key Insight

### **CT-EFT-20's Policy Training:**

```python
# They do:
position_encoding = encode_for_transformer(
    board_position,  # From H5
    turn,            # From H5
    castling_rights  # From H5
)

target_move = moves[0]  # FIRST MOVE ONLY!
target = (from_square, to_square)  # Extracted from move

# Train transformer
loss = cross_entropy(
    model(position_encoding),
    target
)
```

### **Our CNN Training (SHOULD BE):**

```python
# We do:
position_encoding = encode_for_cnn(
    board_position,  # SAME H5 data
    turn,            # SAME H5 data
    castling_rights  # SAME H5 data
) → 18×8×8 grid

target_move = moves[0]  # SAME: FIRST MOVE ONLY!
target = (from_square, to_square)  # SAME extraction

# Train CNN
loss = cross_entropy(
    model(position_encoding),
    target
)
```

**Only difference: Input representation (70-token vs 18×8×8)!**

**This is THE research question!** ✅

---

## ✅ What I've Created

**File: `cnn_policy/dataset_h5_proper.py`**

This loader:
- ✅ Reads LE22ct H5 directly
- ✅ Decodes board_position to FEN
- ✅ Uses moves[0] as target (EXACT CT-EFT-20 approach)
- ✅ Ignores moves[1:9] (like CT-EFT-20)
- ✅ Converts to our 18-channel tensor
- ✅ Returns (position, (from_square, to_square))

---

## 🚀 How to Use It

### **Option 1: Use H5 directly** (FAIREST)

Edit `cnn_policy/train.py`:

```python
# Change import:
from cnn_policy.dataset_h5_proper import ChessPolicyDatasetH5Proper

# Change dataset loading:
h5_path = 'dataset/raw/LE22ct/LE22ct.h5'
self.train_dataset = ChessPolicyDatasetH5Proper(h5_path, split='train')
self.val_dataset = ChessPolicyDatasetH5Proper(h5_path, split='val')
```

**This is the EXACT same data usage as CT-EFT-20!**

### **Option 2: Convert to CSV once** (EASIER)

Use the converter script, then use current code.

**Both are fair, H5 is more direct!**

---

## 🎯 My Recommendation

**Use Option 1: Direct H5 reading**

**Why:**
1. ✅ **Most fair:** Uses CT-EFT-20's exact H5 data
2. ✅ **No conversion:** One less transformation step
3. ✅ **Same move targets:** Uses moves[0] exactly as they do
4. ✅ **Truly comparable:** Only difference is input representation

**The dataset_h5_proper.py I just created does this!**

**Want me to update train.py to use this H5 loader?** 

This will be the FAIREST possible comparison to CT-EFT-20! 🎯

