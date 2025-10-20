# Created by: Rami Abu Mukh, Omar Gharra, Ameer Khalaily

import os
import torch
import h5py
from torch.utils.data import Dataset
from pathlib import Path


class ChessDatasetFT(Dataset):
    def __init__(self, data_folder, h5_file, split, **unused):
        h5_path = os.path.join(data_folder, h5_file)
        if not os.path.exists(h5_path):
            h5_path = str(Path(data_folder) / 'raw/LE22ct' / h5_file)
        self.h5_path = h5_path
        self.split = split
        with h5py.File(h5_path, 'r', swmr=True) as f:
            self.encoded_table = f['encoded_data']
            self.total_rows = len(self.encoded_table)
            self.val_split_index = int(self.total_rows * 0.9)

        if split == "train":
            self.first_index = 0
            self.last_index = self.val_split_index
        elif split == "val":
            self.first_index = self.val_split_index
            self.last_index = self.total_rows
        self.h5_file = None
# open if not already open - for multiprocessing
    def __getitem__(self, i):
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
        return self.last_index - self.first_index
    
    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            self.h5_file.close()