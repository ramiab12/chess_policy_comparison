"""
Transformer Dataset Loader - CT-EFT-20
======================================
IDENTICAL code from chess-transformers/train/datasets.py

Uses ChessDatasetFT class - reads H5 with PyTables (tables library).
Returns from_square/to_square directly from H5 fields.
"""

import os
import torch
import tables as tb
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
        # Open table in H5 file
        h5_path = os.path.join(data_folder, h5_file)
        if not os.path.exists(h5_path):
            h5_path = str(Path(data_folder) / 'raw/LE22ct' / h5_file)
        
        self.h5_file = tb.open_file(h5_path, mode="r")
        self.encoded_table = self.h5_file.root.encoded_data
        self.split = split

        # Create indices
        if split == "train":
            self.first_index = 0
        elif split == "val":
            self.first_index = self.encoded_table.attrs.val_split_index
        elif split is None:
            self.first_index = 0
        else:
            raise NotImplementedError
        
        # Print info
        if split == "train":
            print(f"   Train samples: {self.encoded_table.attrs.val_split_index:,}")
        elif split == "val":
            total = self.encoded_table.nrows
            val_start = self.encoded_table.attrs.val_split_index
            print(f"   Val samples: {total - val_start:,}")

    def __getitem__(self, i):
        """
        Get single sample - IDENTICAL to CT-EFT-20!
        
        Returns dict with:
        - turns, castling_rights, board_positions (inputs)
        - from_squares, to_squares (targets)
        - lengths (always 1 for single move prediction)
        """
        turns = torch.IntTensor([self.encoded_table[self.first_index + i]["turn"]])
        white_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["white_kingside_castling_rights"]]
        )
        white_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["white_queenside_castling_rights"]]
        )
        black_kingside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["black_kingside_castling_rights"]]
        )
        black_queenside_castling_rights = torch.IntTensor(
            [self.encoded_table[self.first_index + i]["black_queenside_castling_rights"]]
        )
        board_position = torch.IntTensor(
            self.encoded_table[self.first_index + i]["board_position"]
        )  # (64)
        from_square = torch.LongTensor(
            [self.encoded_table[self.first_index + i]["from_square"]]
        )  # (1)
        to_square = torch.LongTensor(
            [self.encoded_table[self.first_index + i]["to_square"]]
        )  # (1)
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
        if self.split == "train":
            return self.encoded_table.attrs.val_split_index
        elif self.split == "val":
            return self.encoded_table.nrows - self.encoded_table.attrs.val_split_index
        elif self.split is None:
            return self.encoded_table.nrows
        else:
            raise NotImplementedError


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


