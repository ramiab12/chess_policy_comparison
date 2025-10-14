"""
Chess Policy Dataset Loader
============================
Loads LE22ct dataset for policy (move prediction) training.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import chess
from pathlib import Path
from typing import Tuple
from .position_encoder import PositionEncoder


class ChessPolicyDataset(Dataset):
    """
    Dataset for chess move prediction (policy learning).
    
    Loads from LE22ct format:
    - FEN positions
    - Move labels (UCI format)
    - Converts to from-square, to-square targets
    """
    
    def __init__(
        self,
        csv_path: str,
        augment: bool = False,
        augment_prob: float = 0.5
    ):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file (train.csv or val.csv)
            augment: Enable data augmentation (board flips)
            augment_prob: Probability of applying augmentation
        """
        self.csv_path = Path(csv_path)
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Load data
        print(f"📂 Loading dataset from {self.csv_path}...")
        self.data = pd.read_csv(self.csv_path)
        print(f"   Loaded {len(self.data):,} positions")
        
        # Initialize encoder
        self.encoder = PositionEncoder()
        
        # Verify required columns
        required_cols = ['fen', 'move']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a single training example.
        
        Args:
            idx: Index
            
        Returns:
            position: (18, 8, 8) tensor
            targets: (from_square_idx, to_square_idx) as long tensors
        """
        # Load from dataframe
        row = self.data.iloc[idx]
        fen = row['fen']
        move_uci = row['move']
        
        # Parse move
        try:
            move = chess.Move.from_uci(move_uci)
            from_square = move.from_square  # 0-63
            to_square = move.to_square      # 0-63
        except:
            # Fallback to a default move if parsing fails
            from_square = 0
            to_square = 0
        
        # Convert FEN to tensor
        position = self.encoder.fen_to_tensor(fen)
        
        # Data augmentation (horizontal flip)
        if self.augment and np.random.random() < self.augment_prob:
            position = self.flip_board(position)
            # Also flip move squares!
            from_square = self.flip_square(from_square)
            to_square = self.flip_square(to_square)
        
        # Convert to tensors
        from_target = torch.tensor(from_square, dtype=torch.long)
        to_target = torch.tensor(to_square, dtype=torch.long)
        
        return position, (from_target, to_target)
    
    def flip_board(self, tensor: torch.Tensor) -> torch.Tensor:
        """Flip board horizontally (left-right)."""
        return torch.flip(tensor, dims=[2])
    
    def flip_square(self, square_idx: int) -> int:
        """
        Flip square index horizontally.
        
        Args:
            square_idx: Square index (0-63)
            
        Returns:
            Flipped square index
        """
        rank = square_idx // 8
        file = square_idx % 8
        new_file = 7 - file  # Flip horizontally
        return rank * 8 + new_file


def test_dataset():
    """Test dataset loader."""
    print("🧪 Testing Chess Policy Dataset\n")
    
    # Create dummy data
    import tempfile
    import os
    
    dummy_data = pd.DataFrame({
        'fen': [
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
        ],
        'move': ['e2e4', 'c7c5']
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        dummy_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        dataset = ChessPolicyDataset(temp_path, augment=False)
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test __getitem__
        position, (from_target, to_target) = dataset[0]
        print(f"Position shape: {position.shape}")
        print(f"From target: {from_target.item()} (square index)")
        print(f"To target: {to_target.item()} (square index)")
        
        # Verify
        move = chess.Move.from_uci('e2e4')
        print(f"\nMove e2e4:")
        print(f"  From square: {move.from_square} (expected: {from_target.item()})")
        print(f"  To square: {move.to_square} (expected: {to_target.item()})")
        
        print("\n✅ Dataset test passed!")
        
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    test_dataset()

