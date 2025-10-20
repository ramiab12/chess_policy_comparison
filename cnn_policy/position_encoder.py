"""
Position Encoder for CNN - as in AlphaZero paper 
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
    def __init__(self):
        self.piece_to_channel = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black
        }
    
    def fen_to_tensor(self, fen: str) -> torch.Tensor:
        board = chess.Board(fen)
        tensor = torch.zeros(18, 8, 8, dtype=torch.float32)
        
        # channels 0-11: piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                piece_symbol = piece.symbol()
                channel = self.piece_to_channel[piece_symbol]
                row, col = divmod(square, 8)
                tensor[channel, row, col] = 1.0
        
        # channel 12: side to move (1.0 for white, 0.0 for black)
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0
        else:
            tensor[12, :, :] = 0.0
        
        # channels 13-16: castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[13, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[14, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[15, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[16, :, :] = 1.0
        
        # channel 17: En passant target square
        if board.ep_square is not None:
            row, col = divmod(board.ep_square, 8)
            tensor[17, row, col] = 1.0
        
        return tensor
    
    def fen_to_tensor_batch(self, fens: List[str]) -> torch.Tensor:
        
        tensors = [self.fen_to_tensor(fen) for fen in fens]
        return torch.stack(tensors)
    
    def get_num_channels(self) -> int:
        return 18

