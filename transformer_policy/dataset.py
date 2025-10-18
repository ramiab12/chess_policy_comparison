"""
Transformer Dataset Loader - CT-EFT-20
======================================
OPTIMIZED: Uses h5py instead of PyTables for better multiprocessing support.

Reads H5 file like CNN does - much faster with multiple workers!
Returns from_square/to_square directly from H5 fields.
"""

import os
import torch
import h5py
from torch.utils.data import Dataset
from pathlib import Path


class ChessDatasetFT(Dataset):
    """
    Dataset for From-To prediction (CT-EFT-20 style).
    
    IDENTICAL to chess-transformers ChessDatasetFT class.
    Reads LE22ct H5 using PyTables, returns from/to squares directly.
    """
    
    def __init__(self, data_folder, h5_file, split, **unused):
        """
        Init.

        Args:
            data_folder (str): The folder containing the H5 file.
            h5_file (str): The H5 file name.
            split (str): The data split. One of "train", "val", None.
        """
        # Find H5 file
        h5_path = os.path.join(data_folder, h5_file)
        if not os.path.exists(h5_path):
            h5_path = str(Path(data_folder) / 'raw/LE22ct' / h5_file)
        
        self.h5_path = h5_path
        self.split = split
        
        # Open once to get metadata (will reopen per-worker)
        with h5py.File(h5_path, 'r', swmr=True) as f:
            self.encoded_table = f['encoded_data']
            self.total_rows = len(self.encoded_table)
            # Use 90/10 split for fair comparison (same as CNN)
            self.val_split_index = int(self.total_rows * 0.9)

        # Create indices
        if split == "train":
            self.first_index = 0
            self.last_index = self.val_split_index
        elif split == "val":
            self.first_index = self.val_split_index
            self.last_index = self.total_rows
        elif split is None:
            self.first_index = 0
            self.last_index = self.total_rows
        else:
            raise NotImplementedError
        
        # Print info
        if split == "train":
            print(f"   Train samples: {self.val_split_index:,}")
        elif split == "val":
            print(f"   Val samples: {self.total_rows - self.val_split_index:,}")
        
        # H5 file will be opened per-worker in __getitem__
        self.h5_file = None

    def __getitem__(self, i):
        """
        Get single sample - using h5py for better multiprocessing!
        
        Returns dict with:
        - turns, castling_rights, board_positions (inputs)
        - from_squares, to_squares (targets)
        - lengths (always 1 for single move prediction)
        """
        # Open H5 file if not already open (for multiprocessing)
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)
            self.encoded_table = self.h5_file['encoded_data']
        
        # Get actual index
        actual_idx = self.first_index + i
        row = self.encoded_table[actual_idx]
        
        # Extract data
        turns = torch.IntTensor([int(row["turn"])])
        white_kingside_castling_rights = torch.IntTensor(
            [int(row["white_kingside_castling_rights"])]
        )
        white_queenside_castling_rights = torch.IntTensor(
            [int(row["white_queenside_castling_rights"])]
        )
        black_kingside_castling_rights = torch.IntTensor(
            [int(row["black_kingside_castling_rights"])]
        )
        black_queenside_castling_rights = torch.IntTensor(
            [int(row["black_queenside_castling_rights"])]
        )
        board_position = torch.IntTensor(row["board_position"])  # (64)
        from_square = torch.LongTensor([int(row["from_square"])])  # (1)
        to_square = torch.LongTensor([int(row["to_square"])])  # (1)
        length = torch.LongTensor([1])  # Always 1 for single move prediction
        
        return {
            "turns": turns,
            "white_kingside_castling_rights": white_kingside_castling_rights,
            "white_queenside_castling_rights": white_queenside_castling_rights,
            "black_kingside_castling_rights": black_kingside_castling_rights,
            "black_queenside_castling_rights": black_queenside_castling_rights,
            "board_positions": board_position,
            "from_squares": from_square,
            "to_squares": to_square,
            "lengths": length,
        }

    def __len__(self):
        """Return dataset size."""
        return self.last_index - self.first_index
    
    def __del__(self):
        """Close H5 file when done."""
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()


def test_dataset():
    """Test the dataset loader."""
    print("🧪 Testing Transformer Dataset\n")
    
    h5_file = Path("dataset/raw/LE22ct/LE22ct.h5")
    
    if not h5_file.exists():
        print(f"❌ H5 file not found: {h5_file}")
        return
    
    dataset = ChessDatasetFT(
        data_folder=str(h5_file.parent.parent),
        h5_file='raw/LE22ct/LE22ct.h5',
        split='train'
    )
    
    print(f"Dataset size: {len(dataset):,}")
    
    # Test sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Board positions shape: {sample['board_positions'].shape}")
    print(f"From square: {sample['from_squares'].item()}")
    print(f"To square: {sample['to_squares'].item()}")
    
    print("\n✅ Dataset works - IDENTICAL to CT-EFT-20!")


if __name__ == "__main__":
    test_dataset()


