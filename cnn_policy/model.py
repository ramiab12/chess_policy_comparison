"""
CNN Policy Network for Chess Move Prediction
=============================================
Matches CT-EFT-20 Transformer for fair comparison.

Architecture:
- Input: 18×8×8 (pieces + metadata)
- Conv tower: 15 ResBlocks, 256 filters
- Policy heads: From-square + To-square prediction
- Output: (from_logits, to_logits) each (B, 64)

Task: Move prediction (classification)
Comparison: vs CT-EFT-20 (Transformer, 20M params, 1750-1850 ELO)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Residual block for CNN tower."""
    
    def __init__(self, channels: int, dropout_rate: float = 0.1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        self.dropout = nn.Dropout2d(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out = out + identity
        out = F.relu(out)
        
        return out


class ChessCNNPolicy(nn.Module):
    """
    CNN for chess move prediction (policy network).
    
    Matches CT-EFT-20 Transformer for fair comparison:
    - Similar parameter count (~18-20M)
    - Same task (from-to square prediction)
    - Same dataset (LE22ct)
    - Same training protocol
    
    Architecture:
        Input (18×8×8)
        ↓
        Initial Conv (18→256) + BN + ReLU
        ↓
        15× Residual Blocks (256→256)
        ↓ (B, 256, 8, 8) - KEEP SPATIAL!
        ├─→ From-Square Head → (B, 64)
        └─→ To-Square Head → (B, 64)
    """
    
    def __init__(
        self,
        num_input_channels: int = 18,
        num_filters: int = 256,
        num_blocks: int = 15,
        dropout_rate: float = 0.1
    ):
        super(ChessCNNPolicy, self).__init__()
        
        self.num_input_channels = num_input_channels
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        
        # Initial convolution
        self.conv_initial = nn.Conv2d(
            num_input_channels, num_filters,
            kernel_size=3, padding=1, bias=False
        )
        self.bn_initial = nn.BatchNorm2d(num_filters)
        
        # Residual blocks (shared feature extractor)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters, dropout_rate=dropout_rate)
            for _ in range(num_blocks)
        ])
        
        # Policy heads (from-to prediction)
        # NO global pooling! Keep spatial information!
        
        # From-square head
        self.from_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.from_bn = nn.BatchNorm2d(2)
        
        # To-square head
        self.to_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.to_bn = nn.BatchNorm2d(2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 18, 8, 8)
            
        Returns:
            from_logits: (B, 64) - logits for from-square
            to_logits: (B, 64) - logits for to-square
        """
        batch_size = x.size(0)
        
        # Initial convolution
        x = self.conv_initial(x)
        x = self.bn_initial(x)
        x = F.relu(x)
        
        # Residual blocks (shared feature extractor)
        for res_block in self.res_blocks:
            x = res_block(x)
        # x: (B, 256, 8, 8) - Keep spatial information!
        
        # From-square head
        from_out = self.from_conv(x)  # (B, 2, 8, 8)
        from_out = self.from_bn(from_out)
        from_out = F.relu(from_out)
        # Take first channel and reshape to (B, 64)
        from_logits = from_out[:, 0, :, :].reshape(batch_size, 64)
        
        # To-square head
        to_out = self.to_conv(x)  # (B, 2, 8, 8)
        to_out = self.to_bn(to_out)
        to_out = F.relu(to_out)
        # Take first channel and reshape to (B, 64)
        to_logits = to_out[:, 0, :, :].reshape(batch_size, 64)
        
        return from_logits, to_logits
    
    def predict_move(self, x: torch.Tensor, legal_moves: list) -> torch.Tensor:
        """
        Predict best legal move.
        
        Args:
            x: Board position (B, 18, 8, 8)
            legal_moves: List of legal chess.Move objects
            
        Returns:
            Best legal move index
        """
        from_logits, to_logits = self.forward(x)
        
        # Get probabilities
        from_probs = F.softmax(from_logits, dim=1)  # (B, 64)
        to_probs = F.softmax(to_logits, dim=1)  # (B, 64)
        
        # Score legal moves
        move_scores = []
        for move in legal_moves:
            from_idx = move.from_square
            to_idx = move.to_square
            score = from_probs[0, from_idx] * to_probs[0, to_idx]
            move_scores.append(score.item())
        
        best_idx = torch.tensor(move_scores).argmax()
        return best_idx
    
    def get_model_size(self) -> Tuple[int, float]:
        """Calculate model size."""
        param_count = sum(p.numel() for p in self.parameters())
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return param_count, size_mb
    
    def summary(self):
        """Print model summary."""
        param_count, size_mb = self.get_model_size()
        
        print("=" * 70)
        print("CNN Policy Network - Move Prediction")
        print("=" * 70)
        print(f"Input channels:        {self.num_input_channels}")
        print(f"Residual blocks:       {self.num_blocks}")
        print(f"Filters per block:     {self.num_filters}")
        print(f"Total parameters:      {param_count:,}")
        print(f"Model size:            {size_mb:.2f} MB")
        print("=" * 70)
        print("Architecture:")
        print(f"  Input (18×8×8)")
        print(f"  ├─ Conv2d(18→{self.num_filters}, 3×3) + BN + ReLU")
        for i in range(self.num_blocks):
            print(f"  ├─ ResBlock {i+1:2d} ({self.num_filters}→{self.num_filters})")
        print(f"  ├─ From-head: Conv1×1({self.num_filters}→2) → (B, 64)")
        print(f"  └─ To-head: Conv1×1({self.num_filters}→2) → (B, 64)")
        print("=" * 70)
        print("Task: Move prediction (from-to squares)")
        print("Comparison baseline: CT-EFT-20 (1750-1850 ELO)")
        print("=" * 70)


def test_model():
    """Test model architecture."""
    print("\n🧪 Testing CNN Policy Model\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ChessCNNPolicy()
    model = model.to(device)
    
    model.summary()
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 18, 8, 8).to(device)
    
    print(f"\nTesting forward pass:")
    print(f"  Input shape:  {dummy_input.shape}")
    
    with torch.no_grad():
        from_logits, to_logits = model(dummy_input)
    
    print(f"  From logits shape: {from_logits.shape}")
    print(f"  To logits shape:   {to_logits.shape}")
    print(f"  Expected: (4, 64) for each")
    
    # Test softmax
    from_probs = F.softmax(from_logits, dim=1)
    to_probs = F.softmax(to_logits, dim=1)
    
    print(f"\n  From probs sum: {from_probs[0].sum():.4f} (should be 1.0)")
    print(f"  To probs sum:   {to_probs[0].sum():.4f} (should be 1.0)")
    
    print("\n✅ Model test passed!")
    print(f"   Device: {device}")
    print(f"   Ready for training!\n")


if __name__ == "__main__":
    test_model()

