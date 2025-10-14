"""
Position Encoder for CNN
=========================
Encodes chess positions into 18-channel tensors.

Channels:
- 0-5:   White pieces (P, N, B, R, Q, K)
- 6-11:  Black pieces (p, n, b, r, q, k)
- 12:    Side to move (1.0 if white, 0.0 if black)
- 13:    White kingside castling available
- 14:    White queenside castling available
- 15:    Black kingside castling available
- 16:    Black queenside castling available
- 17:    En passant target square
"""

import torch
import chess
import numpy as np
from typing import List


class PositionEncoder:
    """Encode chess positions as 18×8×8 tensors."""
    
    def __init__(self):
        """Initialize encoder with piece-to-channel mapping."""
        # Piece to channel mapping
        self.piece_to_channel = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black
        }
    
    def fen_to_tensor(self, fen: str) -> torch.Tensor:
        """
        Convert FEN string to 18×8×8 tensor.
        
        Args:
            fen: FEN string representing chess position
            
        Returns:
            torch.Tensor of shape (18, 8, 8)
        """
        board = chess.Board(fen)
        tensor = torch.zeros(18, 8, 8, dtype=torch.float32)
        
        # Channels 0-11: Piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_symbol = piece.symbol()
                channel = self.piece_to_channel[piece_symbol]
                row, col = divmod(square, 8)
                tensor[channel, row, col] = 1.0
        
        # Channel 12: Side to move (1.0 for white, 0.0 for black)
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0
        else:
            tensor[12, :, :] = 0.0
        
        # Channels 13-16: Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[13, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[14, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[15, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[16, :, :] = 1.0
        
        # Channel 17: En passant target square
        if board.ep_square is not None:
            row, col = divmod(board.ep_square, 8)
            tensor[17, row, col] = 1.0
        
        return tensor
    
    def fen_to_tensor_batch(self, fens: List[str]) -> torch.Tensor:
        """
        Convert list of FEN strings to batch of tensors.
        
        Args:
            fens: List of FEN strings
            
        Returns:
            torch.Tensor of shape (N, 18, 8, 8)
        """
        tensors = [self.fen_to_tensor(fen) for fen in fens]
        return torch.stack(tensors)
    
    def get_num_channels(self) -> int:
        """Return number of input channels."""
        return 18
    
    def describe_channels(self) -> dict:
        """Return description of all channels."""
        return {
            '0-5': 'White pieces (P, N, B, R, Q, K)',
            '6-11': 'Black pieces (p, n, b, r, q, k)',
            '12': 'Side to move (1.0=White, 0.0=Black)',
            '13': 'White kingside castling',
            '14': 'White queenside castling',
            '15': 'Black kingside castling',
            '16': 'Black queenside castling',
            '17': 'En passant target square'
        }


def test_encoder():
    """Test the position encoder."""
    encoder = PositionEncoder()
    
    # Test starting position
    start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    tensor = encoder.fen_to_tensor(start_fen)
    
    print("✅ Position Encoder Test")
    print(f"   Input channels: {encoder.get_num_channels()}")
    print(f"   Tensor shape: {tensor.shape}")
    print(f"   Channel 12 (side to move): {tensor[12, 0, 0].item()} (should be 1.0 for white)")
    print(f"   Channel 13 (WK castling): {tensor[13, 0, 0].item()} (should be 1.0)")
    
    # Test batch conversion
    fens = [start_fen, start_fen]
    batch = encoder.fen_to_tensor_batch(fens)
    print(f"   Batch shape: {batch.shape} (should be [2, 18, 8, 8])")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_encoder()

