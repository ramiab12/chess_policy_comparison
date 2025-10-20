# Created by: Rami Abu Mukh, Omar Gharra, Ameer Khalaily

import torch
from torch.utils.data import Dataset
import h5py
import chess
import numpy as np
from pathlib import Path
from typing import Tuple
from .position_encoder import PositionEncoder


# Generate all possible UCI moves 
def generate_uci_labels():
    labels = []
    squares = [f"{f}{r}" for r in '12345678' for f in 'abcdefgh']
    for from_sq in squares:
        for to_sq in squares:
            if from_sq != to_sq:
                labels.append(f"{from_sq}{to_sq}")
    
    # pawn promotions 
    for from_file in 'abcdefgh':
        for to_file in 'abcdefgh':
            # White 
            for piece in ['q', 'r', 'b', 'n']:
                labels.append(f"{from_file}7{to_file}8{piece}")
            # Black 
            for piece in ['q', 'r', 'b', 'n']:
                labels.append(f"{from_file}2{to_file}1{piece}")
    
    return labels


UCI_LABELS = generate_uci_labels()


def decode_board_from_h5(row) -> str:
    
    # Create empty board
    board = chess.Board.empty()
    # Piece vocabulary (VERIFIED from LE22ct H5 file):
    # 0 = empty
    # Even numbers (2,4,6,8,10,12) = BLACK pieces
    # Odd numbers (3,5,7,9,11,13) = WHITE pieces
    # Order: P, R, N, B, Q, K (not the typical P,N,B,R,Q,K!) - this was the problem in inference, now fixed was not matching
    piece_map = {
        2: (chess.PAWN, chess.BLACK),
        3: (chess.PAWN, chess.WHITE),
        4: (chess.ROOK, chess.BLACK),
        5: (chess.ROOK, chess.WHITE),
        6: (chess.KNIGHT, chess.BLACK),
        7: (chess.KNIGHT, chess.WHITE),
        8: (chess.BISHOP, chess.BLACK),
        9: (chess.BISHOP, chess.WHITE),
        10: (chess.QUEEN, chess.BLACK),
        11: (chess.QUEEN, chess.WHITE),
        12: (chess.KING, chess.BLACK),
        13: (chess.KING, chess.WHITE),
    }
    
    
    board_encoding = row['board_position']
    
    # Decode each square
    for square_idx in range(64):
        piece_code = board_encoding[square_idx]
        if piece_code > 0 and piece_code in piece_map:
            piece_type, color = piece_map[piece_code]
            board.set_piece_at(square_idx, chess.Piece(piece_type, color))
    
    # Set turn
    board.turn = chess.WHITE if row['turn'] == 0 else chess.BLACK
    
    # Set castling rights (use direct indexing for numpy structured arrays)
    board.castling_rights = 0
    if row['white_kingside_castling_rights']:
        board.castling_rights |= chess.BB_H1
    if row['white_queenside_castling_rights']:
        board.castling_rights |= chess.BB_A1
    if row['black_kingside_castling_rights']:
        board.castling_rights |= chess.BB_H8
    if row['black_queenside_castling_rights']:
        board.castling_rights |= chess.BB_A8
    
    return board.fen()

# the new dataset loader - matches the transformer model - no conversion needed
class ChessPolicyDatasetH5Proper(Dataset):
    
    def __init__(
        self,
        h5_path: str,
        split: str = 'train',
        train_ratio: float = 0.9,
        augment: bool = False
    ):
        
        self.h5_path = Path(h5_path)
        self.split = split
        self.augment = augment
        
        
        print(f"Loading H5 dataset: {self.h5_path}")
        
        # random access to the h5 file - for faster loading
        self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)
        
        # Get encoded table (try both possible names)
        if 'encoded_data' in self.h5_file:
            self.data = self.h5_file['encoded_data']
        elif 'encoded' in self.h5_file:
            self.data = self.h5_file['encoded']
        else:
            raise ValueError(f"'encoded_data' or 'encoded' table not found in H5 file! Available: {list(self.h5_file.keys())}")
        
        # train/val split - 90/10 
        total_size = len(self.data)
        split_idx = int(total_size * 0.9)
        print(f"   Using 90/10 split: {split_idx:,} (90% train, 10% val)")
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = split_idx
        else:  # val
            self.start_idx = split_idx
            self.end_idx = total_size
        
        self.size = self.end_idx - self.start_idx
        
        print(f"   Split: {split}")
        print(f"   Samples: {self.size:,} (indices {self.start_idx:,} to {self.end_idx:,})")
        
        # initialize encoder
        self.encoder = PositionEncoder()
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        actual_idx = self.start_idx + idx
        
        # Read from H5
        row = self.data[actual_idx]
        
        # the h5 file already has from_square and to_square directly! no need for conversion
        from_square = int(row['from_square'])
        to_square = int(row['to_square'])
        
        # decode
        fen = decode_board_from_h5(row)
        
        # encode the board to a tensor
        position = self.encoder.fen_to_tensor(fen)
        
        # Convert targets to tensors
        from_target = torch.tensor(from_square, dtype=torch.long)
        to_target = torch.tensor(to_square, dtype=torch.long)
        
        return position, (from_target, to_target)
    
    def __del__(self):
        """Close H5 file when done."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()

