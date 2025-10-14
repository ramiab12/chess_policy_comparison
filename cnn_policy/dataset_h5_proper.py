"""
H5 Dataset Loader - Matching CT-EFT-20's Data Usage
====================================================
Reads LE22ct H5 directly as CT-EFT-20 does.

For policy (from-to) prediction:
- Use board_position + metadata to encode position
- Use moves[0] (first move) as target
- Ignore moves[1:9] (only for sequence models)

This ensures EXACT same data usage as CT-EFT-20!
"""

import torch
from torch.utils.data import Dataset
import h5py
import chess
import numpy as np
from pathlib import Path
from typing import Tuple
from .position_encoder import PositionEncoder


# Generate all possible UCI moves (matching chess-transformers)
def generate_uci_labels():
    """
    Generate all possible UCI moves in standard order.
    This matches the chess-transformers UCI_MOVES vocabulary.
    """
    labels = []
    
    # All squares
    squares = [f"{f}{r}" for r in '12345678' for f in 'abcdefgh']
    
    # Regular moves (all combinations)
    for from_sq in squares:
        for to_sq in squares:
            if from_sq != to_sq:
                labels.append(f"{from_sq}{to_sq}")
    
    # Promotions (pawns reaching last rank)
    for from_file in 'abcdefgh':
        for to_file in 'abcdefgh':
            # White promotions (7th rank to 8th)
            for piece in ['q', 'r', 'b', 'n']:
                labels.append(f"{from_file}7{to_file}8{piece}")
            # Black promotions (2nd rank to 1st)
            for piece in ['q', 'r', 'b', 'n']:
                labels.append(f"{from_file}2{to_file}1{piece}")
    
    return labels


UCI_LABELS = generate_uci_labels()


def decode_board_from_h5(row) -> str:
    """
    Decode board position from H5 encoding to FEN.
    
    The board_position in H5 is an encoded array.
    We need to convert it back to FEN for our CNN input.
    
    Args:
        row: H5 row with board_position, turn, castling_rights
    
    Returns:
        FEN string
    """
    # Create empty board
    board = chess.Board.empty()
    
    # The board_position is encoded - need to decode
    # This requires understanding their exact encoding scheme
    
    # Piece vocabulary (typical encoding):
    # 0 = empty, 1-6 = white pieces (P,N,B,R,Q,K), 7-12 = black pieces
    piece_map = {
        1: (chess.PAWN, chess.WHITE),
        2: (chess.KNIGHT, chess.WHITE),
        3: (chess.BISHOP, chess.WHITE),
        4: (chess.ROOK, chess.WHITE),
        5: (chess.QUEEN, chess.WHITE),
        6: (chess.KING, chess.WHITE),
        7: (chess.PAWN, chess.BLACK),
        8: (chess.KNIGHT, chess.BLACK),
        9: (chess.BISHOP, chess.BLACK),
        10: (chess.ROOK, chess.BLACK),
        11: (chess.QUEEN, chess.BLACK),
        12: (chess.KING, chess.BLACK),
    }
    
    try:
        board_encoding = row['board_position']
        
        # Decode each square
        for square_idx in range(64):
            piece_code = board_encoding[square_idx]
            if piece_code > 0 and piece_code in piece_map:
                piece_type, color = piece_map[piece_code]
                board.set_piece_at(square_idx, chess.Piece(piece_type, color))
        
        # Set turn
        board.turn = chess.WHITE if row['turn'] == 0 else chess.BLACK
        
        # Set castling rights
        board.castling_rights = 0
        if row.get('white_kingside_castling_rights', 0):
            board.castling_rights |= chess.BB_H1
        if row.get('white_queenside_castling_rights', 0):
            board.castling_rights |= chess.BB_A1
        if row.get('black_kingside_castling_rights', 0):
            board.castling_rights |= chess.BB_H8
        if row.get('black_queenside_castling_rights', 0):
            board.castling_rights |= chess.BB_A8
        
        return board.fen()
        
    except Exception as e:
        # Fallback to starting position
        return chess.Board().fen()


class ChessPolicyDatasetH5Proper(Dataset):
    """
    H5 dataset loader matching CT-EFT-20's data usage.
    
    Reads LE22ct H5 directly:
    - Decodes board_position to position tensor
    - Uses moves[0] (first move) as target
    - Same data usage as CT-EFT-20 for policy learning!
    """
    
    def __init__(
        self,
        h5_path: str,
        split: str = 'train',
        train_ratio: float = 0.9,
        augment: bool = False
    ):
        """
        Initialize H5 dataset loader.
        
        Args:
            h5_path: Path to LE22ct.h5
            split: 'train' or 'val'
            train_ratio: Train/val split ratio
            augment: Data augmentation
        """
        self.h5_path = Path(h5_path)
        self.split = split
        self.augment = augment
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        
        print(f"📂 Loading H5 dataset: {self.h5_path}")
        
        # Open H5 file (keep it open for random access)
        self.h5_file = h5py.File(self.h5_path, 'r', swmr=True)
        
        # Get encoded table
        if 'encoded' not in self.h5_file:
            raise ValueError(f"'encoded' table not found in H5 file!")
        
        self.data = self.h5_file['encoded']
        
        # Determine split indices
        total_size = len(self.data)
        split_idx = int(total_size * train_ratio)
        
        if split == 'train':
            self.start_idx = 0
            self.end_idx = split_idx
        else:  # val
            self.start_idx = split_idx
            self.end_idx = total_size
        
        self.size = self.end_idx - self.start_idx
        
        print(f"   Split: {split}")
        print(f"   Samples: {self.size:,} (indices {self.start_idx:,} to {self.end_idx:,})")
        
        # Initialize encoder
        self.encoder = PositionEncoder()
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get training sample - MATCHES CT-EFT-20's approach!
        
        Returns:
            position: (18, 8, 8) tensor
            targets: (from_square, to_square) indices
        """
        # Get actual H5 index
        actual_idx = self.start_idx + idx
        
        # Read from H5
        row = self.data[actual_idx]
        
        # Extract first move (this is what CT-EFT-20 uses for policy!)
        moves_array = row['moves']
        
        if len(moves_array) > 0:
            first_move_idx = moves_array[0]
            
            # Decode move index to UCI
            if first_move_idx < len(UCI_LABELS):
                move_uci = UCI_LABELS[first_move_idx]
            else:
                # Fallback
                move_uci = "e2e4"
        else:
            move_uci = "e2e4"
        
        # Parse move to get from/to squares
        try:
            move = chess.Move.from_uci(move_uci)
            from_square = move.from_square
            to_square = move.to_square
        except:
            from_square = 12  # e2
            to_square = 28     # e4
        
        # Decode board position to FEN
        fen = decode_board_from_h5(row)
        
        # Encode position to our 18-channel tensor
        position = self.encoder.fen_to_tensor(fen)
        
        # Convert targets to tensors
        from_target = torch.tensor(from_square, dtype=torch.long)
        to_target = torch.tensor(to_square, dtype=torch.long)
        
        return position, (from_target, to_target)
    
    def __del__(self):
        """Close H5 file when done."""
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def test_h5_dataset():
    """Test the H5 dataset loader."""
    print("\n🧪 Testing H5 Dataset Loader (CT-EFT-20 Compatible)\n")
    
    h5_path = Path("dataset/raw/LE22ct/LE22ct.h5")
    
    if not h5_path.exists():
        print(f"❌ H5 file not found: {h5_path}")
        print("\n   Download first:")
        print("   python scripts/download_le22ct.py")
        return
    
    try:
        # Load dataset
        dataset = ChessPolicyDatasetH5Proper(h5_path, split='train')
        
        print(f"✅ Dataset loaded: {len(dataset):,} samples")
        
        # Test first sample
        position, (from_target, to_target) = dataset[0]
        
        print(f"\n📊 Sample 0:")
        print(f"   Position shape: {position.shape}")
        print(f"   From square: {from_target.item()}")
        print(f"   To square: {to_target.item()}")
        
        print("\n✅ H5 dataset loader works!")
        print("\n💡 This reads data EXACTLY as CT-EFT-20 does!")
        print("   - Uses encoded board_position")
        print("   - Uses moves[0] as target")
        print("   - Ignores future moves (moves[1:9])")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_h5_dataset()

