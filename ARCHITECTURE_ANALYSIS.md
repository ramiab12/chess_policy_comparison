# Architecture Deep Dive: CNN vs Transformer for Chess

## Overview

After going through every line of code in both implementations, I want to lay out exactly what's different between these two approaches and why it matters for chess. This isn't just about "CNN vs Transformer" in the abstract – it's about two fundamentally different ways of *seeing* a chess position.

Both models are trying to solve the same problem: given a chess position, predict which square a piece should move from (the "from square") and which square it should move to (the "to square"). Same data, same training setup, same loss function. The only thing that's different is how they process the information.

---

## The CNN Architecture: Thinking Spatially

### How It Works

The CNN model treats a chess board exactly like what it is – an 8×8 grid. The input is 18 channels stacked on top of each other:
- Channels 0-5: Where the white pieces are (P, N, B, R, Q, K)
- Channels 6-11: Where the black pieces are  
- Channel 12: Whose turn it is (filled with 1.0 for white, 0.0 for black)
- Channels 13-16: Castling rights for both sides
- Channel 17: En passant square

So it's literally a (18, 8, 8) tensor. You could visualize each channel as a heatmap where 1.0 means "something is here" and 0.0 means "nothing here."

The architecture is:
```
Input: (18, 8, 8)
→ Initial Conv: 18 → 256 filters, 3×3 kernel
→ 15 Residual Blocks (each doing Conv-BN-ReLU-Conv-BN-Dropout + skip connection)
→ Final feature map: (256, 8, 8)
→ Two separate heads:
    ├─ From-square head: Conv 256→2, then take first channel → (64,) logits
    └─ To-square head: Conv 256→2, then take first channel → (64,) logits
```

**Key insight:** The 8×8 spatial structure is *preserved* throughout the entire network. When the model looks at square e4, it can directly see what's on d5, e5, f5, d4, f4, etc. because they're literally neighbors in the tensor. A 3×3 convolution kernel at position e4 sees a 3×3 neighborhood around that square.

The 15 residual blocks keep refining this spatial understanding. Each convolution is learning patterns like:
- "If there's a knight here and an empty square there, that's a potential knight move"
- "If there's a pawn on the 7th rank with no pieces blocking, promotion is coming"
- "This diagonal is controlled by a bishop"

The skip connections in each residual block let information flow easily through all 15 layers, so the model can learn both low-level patterns (piece positions) and high-level patterns (tactical themes).

### The Inductive Bias

Here's what makes CNNs special for chess: **translation equivariance**. 

A knight fork in the center of the board looks exactly like a knight fork on the side of the board, just shifted. The CNN learns this pattern *once* because the same convolutional kernel sweeps across the entire board. This is huge for chess because chess rules don't care about absolute position – a bishop on a1 moves the same way as a bishop on h8.

Also, **local connectivity** matters. In chess, most tactics happen in local clusters. A pin involves 3 pieces in a line. A fork involves a knight and 2 targets nearby. The CNN's receptive field grows with depth (after 15 layers, each position "sees" essentially the entire board), but the early layers naturally capture local patterns.

### Stats
- **Parameters:** 17.7M
- **Architecture depth:** 15 residual blocks
- **Receptive field:** Grows from 3×3 to full board
- **Spatial structure:** Preserved (8×8 throughout)

---

## The Transformer Architecture: Thinking Sequentially  

### How It Works

The Transformer takes a completely different approach. Instead of a 2D grid, it treats the chess position as a *sequence* of 69 tokens:

```
Sequence: [turn, castling_WK, castling_WQ, castling_BK, castling_BQ, sq0, sq1, sq2, ..., sq63]
```

So token 0 is whose turn it is, tokens 1-4 are the four castling rights, and tokens 5-68 are the 64 squares of the board (a1, b1, c1, ..., h8).

Each token gets embedded into a 512-dimensional vector. The board squares get embedded based on what piece is on them (using a vocabulary of 14: empty, pad, white/black P/N/B/R/Q/K). 

Then positional embeddings are added – these are learned vectors that tell the model "this is position 5 in the sequence" (which happens to be square a1).

After embeddings, the sequence goes through 6 transformer layers. Each layer has:
1. **Multi-head attention** (8 heads, 64-dim queries/keys/values each)
2. **Feed-forward network** (512 → 2048 → 512)

The attention mechanism is where the magic happens. Every token attends to every other token. When processing square e4, the model computes attention weights to all 68 other tokens. It learns things like:
- "When I'm a white knight on e4, I should pay attention to d2 (where the white king might be) and f6 (where I could jump to)"
- "The turn token is important because it tells me if I'm even allowed to move"
- "That bishop on c1 and the square e3 are related because it's a diagonal"

**Key insight:** There's no built-in notion of spatial proximity. The model has to *learn* that e4 and e5 are neighbors, that c1-e3-g5 is a diagonal, that knights move in an L-shape. It can learn these things (and it does), but it's not baked into the architecture.

### Multi-Head Attention Breakdown

Let's zoom in on the attention mechanism since it's the core difference:

For each token, the model computes:
1. **Query:** "What am I looking for?"
2. **Key:** "What information do I have?"  
3. **Value:** "What information do I give?"

The attention weight between token i and token j is: `softmax(Q_i · K_j / sqrt(64))`

This weight says "how much should token i pay attention to token j?"

With 8 heads, each head learns different attention patterns. Maybe:
- Head 1 learns piece-to-piece interactions
- Head 2 learns king safety patterns  
- Head 3 learns pawn structure
- etc.

After attention, each token's representation is updated to be a weighted sum of all other tokens' values. So every token becomes context-aware.

The 6 layers stack these attention operations, so by layer 6, every token has seen every other token through 6 different "lenses."

### Output

After the 6 transformer layers, we have 69 tokens, each a 512-d vector. The model:
- Takes tokens 5-68 (the 64 board squares)
- Passes each through a linear layer (512 → 1) to get "from square" logits
- Passes each through another linear layer (512 → 1) to get "to square" logits

So we end up with:
- From-square logits: (64,)
- To-square logits: (64,)

Same output format as the CNN, just computed totally differently.

### Stats
- **Parameters:** 19.0M (slightly more than CNN)
- **Architecture depth:** 6 transformer layers
- **Sequence length:** 69 tokens
- **Hidden dimension:** 512
- **Attention heads:** 8
- **Spatial structure:** None (learned through attention)

---

## Direct Comparison

| Aspect | CNN | Transformer |
|--------|-----|-------------|
| **Input representation** | 8×8 grid with 18 channels | Sequence of 69 tokens |
| **Spatial awareness** | Built-in (2D convolutions) | Learned (through attention) |
| **Receptive field** | Local → global across layers | Global from layer 1 |
| **Parameters** | 17.7M | 19.0M |
| **Hidden dimensions** | 256 channels | 512 dimensions |
| **Depth** | 15 residual blocks | 6 transformer layers |
| **Inductive bias** | Translation equivariance, locality | None (sequence processing) |
| **Computation** | O(n²d) per conv layer | O(n²d) per attention layer |

### Training Differences (kept identical for fairness)

Both models use:
- Same dataset: LE22ct (13.3M positions from 2400+ Elo games)
- Same batch size: 2048
- Same optimizer: Adam with (β1=0.9, β2=0.98)  
- Same learning rate schedule: Vaswani schedule with 8000 warmup steps
- Same loss: Cross-entropy with 0.1 label smoothing
- Same training steps: 55,000
- Same hardware: H100 GPU with BF16 mixed precision

The *only* difference is the model architecture.

---

## What This Means for Chess

### CNN's Advantages

**1. Spatial inductive bias**

Chess is fundamentally spatial. Pieces move in geometric patterns (rook = straight lines, bishop = diagonals, knight = L-shape). The CNN's 2D structure captures this naturally. A 3×3 convolution kernel can learn "if there's a knight here and empty squares in an L-pattern, that's a legal move."

The transformer has to learn that square 27 (d4) and square 19 (d3) are vertically adjacent, that 27 and 36 (e5) are diagonal, etc. It can learn this (attention weights will encode it), but it's extra work.

**2. Parameter efficiency on local patterns**

Early chess tactics are local. A pin involves 3 pieces. A fork involves 3 pieces. A discovered attack involves pieces on the same line. CNNs excel at local patterns because early layers have small receptive fields.

The transformer sees everything from layer 1, which is powerful but means it has to learn to *ignore* most things. With 69 tokens, each token has to compute attention to 68 others – that's a lot of noise for local patterns.

**3. Translation equivariance**

A knight fork is a knight fork whether it happens on the queenside or kingside. The CNN learns this once because the same conv filter slides across the board. The transformer has to learn each position separately (though positional embeddings help).

### Transformer's Advantages  

**1. Long-range dependencies**

Some chess concepts are genuinely non-local. King safety might depend on where the opponent's pieces are across the board. Pawn structures on the queenside matter for kingside attacks. Endgame piece coordination requires seeing the whole board.

The transformer's global attention means every square "sees" every other square directly. No need to pass information through 15 layers of local convolutions.

**2. Flexibility with non-spatial information**

Castling rights and turn information are not spatial – they're just global facts about the position. The CNN has to broadcast these across all 64 squares as separate channels. The transformer treats them as first-class tokens that can attend to board squares directly.

**3. Expressiveness**

With no inductive bias, the transformer is theoretically more flexible. If there's some weird chess pattern that doesn't fit spatial locality, the transformer can learn it through attention. The CNN might struggle because it's constrained by the conv filter shape.

---

## Predictions for Playing Behavior

Based on the architectural differences, here's what I expect to see when we test them on Lichess:

### Tactical play (short-term, local patterns)

**Prediction: CNN should be stronger**

Tactics like forks, pins, skewers, discovered attacks – these are spatial patterns. The CNN's convolution filters should capture them more efficiently. I expect the CNN to:
- Spot knight forks more reliably
- See pins earlier  
- Calculate short forced sequences better

The transformer will get these too (it has enough parameters), but might be slightly less consistent.

### Strategic play (long-term planning)

**Prediction: Transformer might be stronger, but maybe not**

Strategy requires understanding global position features: pawn structure, piece coordination, king safety considering all pieces. The transformer's global attention should help here.

But honestly, with only 13M training positions and 55k training steps, neither model will be grandmaster-level strategist. Strategy requires deep search, which neither model does. So this might be a wash.

### Opening play

**Prediction: Similar**

Both models are essentially doing pattern matching in the opening. The transformer's positional embeddings should let it distinguish "this is move 5" from "this is move 20," but that information isn't in our input. We only give the current position, not move history.

So both will play reasonable openings based on what they've seen in the training data.

### Endgame play

**Prediction: Transformer might be stronger**

Endgames are about precise calculation and global piece coordination. A rook endgame requires knowing where both kings are, where all pawns are, and calculating zugzwang. The transformer's global view could help.

But again, without search, both models will struggle in endgames that require deep calculation.

### Blunder rate

**Prediction: CNN should blunder less**

This is speculative, but: the CNN's spatial structure might make it harder to suggest completely illegal moves or moves that hang pieces obviously. The local connectivity means hanging a piece in one part of the board is visible to the local conv filters.

The transformer's attention could get confused – maybe it pays too much attention to one side of the board and misses a hanging piece on the other side.

### Time management

Both should be fast at inference (single forward pass, no search), but:
- CNN: More memory-efficient (local convolutions)  
- Transformer: More compute-intensive (attention is O(n²))

For real-time play on Lichess, the CNN might be slightly faster.

---

## The Core Question

Your paper is asking: "Same dataset, same training, same task – what does the architecture choice matter?"

My hypothesis:

**The CNN will play better chess, maybe 100-150 Elo higher.**

Why? Because chess is spatial. The CNN's inductive biases align with the structure of the problem. The transformer is more general-purpose, which is usually good, but here it means learning from scratch things that the CNN gets for free.

The transformer might show interesting failure modes though – maybe it's better at some positions and worse at others. Maybe it develops weird strategic preferences because it's not constrained by locality.

The really interesting finding would be if they're *equal* – that would mean the transformer's flexibility compensates perfectly for its lack of spatial bias. Or if the transformer is *better* – that would suggest chess is less spatial than we think, or that global attention is crucial.

---

## Implementation Quality Notes

Both implementations are clean and well-matched:

**CNN:**
- Clean AlphaZero-style architecture with residual blocks
- Position encoder properly handles all board features (castling, en passant, turn)
- Training loop has proper BF16 mixed precision for H100
- Vaswani LR schedule correctly implemented

**Transformer:**
- Standard encoder-only architecture (no decoder needed for single move prediction)
- Embeddings for each board feature (pieces, castling rights, turn)  
- Proper positional embeddings (learned, not sinusoidal – good choice since only 69 tokens)
- Multi-head attention with masking (though masking unused since encoder-only)
- Same training setup as CNN

Both use the same data pipeline (reading from LE22ct H5 file), same loss function (cross-entropy with label smoothing), same evaluation (from-square and to-square accuracy).

---

## What to Look For in Evaluation

When you test these on Lichess, here's what I'd track:

**Quantitative:**
- Overall win rate vs various opponents
- Elo rating estimate  
- Tactical puzzle accuracy
- Blunder rate (moves that lose >2 pawns of material)
- Time per move

**Qualitative:**
- Which model plays more "human-like"?
- Do they make the same mistakes or different mistakes?  
- Are there position types where one consistently beats the other?
- Opening repertoire differences
- Endgame technique

**Specific positions to test:**
1. Tactical puzzles (forks, pins, etc.) – should favor CNN
2. Quiet strategic positions – might favor Transformer
3. Complex middlegames – will show which model handles chaos better
4. King safety scenarios – interesting to see who recognizes threats
5. Endgames with global coordination – might favor Transformer

---

## Conclusion

The CNN and Transformer represent two philosophies:

**CNN:** "Chess is spatial, so let's build that into the architecture."  
**Transformer:** "Let the model figure out what matters through attention."

Your experiment is perfect because everything else is controlled. Same data, same training, same task. Just the architecture differs.

My money is on the CNN being stronger because chess *is* spatial and local patterns matter a lot. But the transformer might surprise us – maybe the global view is more valuable than we think, or maybe attention can learn spatial relationships efficiently enough to compete.

Either way, the comparison will tell us something real about inductive biases in neural networks for games. That's way more interesting than just "which model is better" – it's about *why* architecture choices matter.

Looking forward to seeing the results!

