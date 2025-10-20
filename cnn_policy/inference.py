import torch
import torch.nn.functional as F
import chess
import numpy as np
from typing import Optional


# Square name to index mapping
SQUARES = {
    'a1': 0, 'b1': 1, 'c1': 2, 'd1': 3, 'e1': 4, 'f1': 5, 'g1': 6, 'h1': 7,
    'a2': 8, 'b2': 9, 'c2': 10, 'd2': 11, 'e2': 12, 'f2': 13, 'g2': 14, 'h2': 15,
    'a3': 16, 'b3': 17, 'c3': 18, 'd3': 19, 'e3': 20, 'f3': 21, 'g3': 22, 'h3': 23,
    'a4': 24, 'b4': 25, 'c4': 26, 'd4': 27, 'e4': 28, 'f4': 29, 'g4': 30, 'h4': 31,
    'a5': 32, 'b5': 33, 'c5': 34, 'd5': 35, 'e5': 36, 'f5': 37, 'g5': 38, 'h5': 39,
    'a6': 40, 'b6': 41, 'c6': 42, 'd6': 43, 'e6': 44, 'f6': 45, 'g6': 46, 'h6': 47,
    'a7': 48, 'b7': 49, 'c7': 50, 'd7': 51, 'e7': 52, 'f7': 53, 'g7': 54, 'h7': 55,
    'a8': 56, 'b8': 57, 'c8': 58, 'd8': 59, 'e8': 60, 'f8': 61, 'g8': 62, 'h8': 63,
}

# used for sampling k best moves, to make kind of a stochastic policy - by omar
def topk_sampling(logits: torch.Tensor, k: int = 1) -> torch.Tensor: 
    if k == 1:
        # Deterministic - pick best
        return logits.argmax(dim=-1)
    else:
        # Stochastic top-k sampling
        topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        
        # Sample from top-k
        sampled_idx = torch.multinomial(topk_probs, 1)
        return topk_indices.gather(-1, sampled_idx).squeeze(-1)


def is_pawn_promotion(board: chess.Board, move_uci: str) -> bool:
    if len(move_uci) < 4:
        return False
    
    from_square = chess.SQUARE_NAMES.index(move_uci[:2])
    to_square = chess.SQUARE_NAMES.index(move_uci[2:4])
    
    piece = board.piece_at(from_square)
    
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    
    # Check if moving to last rank
    to_rank = to_square // 8
    
    if piece.color == chess.WHITE and to_rank == 7:
        return True
    if piece.color == chess.BLACK and to_rank == 0:
        return True
    
    return False


def predict_move(
    model,
    board: chess.Board,
    encoder,
    device: torch.device,
    k: int = 1
) -> chess.Move:

    model.eval()
    
    with torch.no_grad():
        # get legal moves
        legal_moves = [move.uci() for move in board.legal_moves]
        
        # encode position
        position = encoder.fen_to_tensor(board.fen())
        position = position.unsqueeze(0).to(device)  # (1, 18, 8, 8)
        
        # get predictions
        predicted_from_squares, predicted_to_squares = model(position)
        # (1, 64), (1, 64)
        
        # convert to log probabilities
        predicted_from_log_probs = F.log_softmax(
            predicted_from_squares, dim=-1
        ).unsqueeze(2)  # (1, 64, 1)
        
        predicted_to_log_probs = F.log_softmax(
            predicted_to_squares, dim=-1
        ).unsqueeze(1)  # (1, 1, 64)
        
        # Combine: log P(from) + log P(to) = log P(from, to)
        predicted_moves = (
            predicted_from_log_probs + predicted_to_log_probs
        ).view(1, -1)  # (1, 4096)
        
        # Remove promotion suffixes for matching
        legal_moves_no_promotion = list(set([m[:4] for m in legal_moves]))
        
        # Convert legal moves to indices
        legal_move_indices = []
        for move_str in legal_moves_no_promotion:
            from_square = move_str[:2]
            to_square = move_str[2:4]
            
            from_idx = SQUARES[from_square]
            to_idx = SQUARES[to_square]
            
            # Combine to single index
            move_idx = from_idx * 64 + to_idx
            legal_move_indices.append(move_idx)
        
        # Filter predictions to legal moves only
        legal_predictions = predicted_moves[:, legal_move_indices]
        
        # Top-k sampling (k=1 for best move)
        best_legal_idx = topk_sampling(logits=legal_predictions, k=k).item()
        model_move_str = legal_moves_no_promotion[best_legal_idx]
        
        # Handle pawn promotion - always queen for simplicity - after research almost 90% of the time the best move is a queen promotion
        if is_pawn_promotion(board, model_move_str):
            model_move_str = model_move_str + "q"
        
        # Convert to Move object
        move = chess.Move.from_uci(model_move_str)
        
        return move

# for visualization purposes - to see the probabilities of each move
def get_move_probabilities(
    model,
    board: chess.Board,
    encoder,
    device: torch.device
) -> dict:
    
    model.eval()
    
    with torch.no_grad():
        # Encode position
        position = encoder.fen_to_tensor(board.fen())
        position = position.unsqueeze(0).to(device)
        
        # Get predictions
        from_logits, to_logits = model(position)
        
        # Log probabilities
        from_log_probs = F.log_softmax(from_logits, dim=-1)
        to_log_probs = F.log_softmax(to_logits, dim=-1)
        
        # Combine
        from_log_probs = from_log_probs.unsqueeze(2)
        to_log_probs = to_log_probs.unsqueeze(1)
        combined_log_probs = (from_log_probs + to_log_probs).view(1, -1)
        
        # Convert to probabilities
        combined_probs = torch.exp(combined_log_probs)
        
        # Get legal moves and their probabilities
        move_probs = {}
        legal_moves = list(board.legal_moves)
        
        for move in legal_moves:
            move_str = move.uci()[:4]  # Remove promotion
            from_idx = move.from_square
            to_idx = move.to_square
            move_idx = from_idx * 64 + to_idx
            
            prob = combined_probs[0, move_idx].item()
            move_probs[move.uci()] = prob
        
        # Normalize (should sum to legal move probability mass)
        total = sum(move_probs.values())
        if total > 0:
            move_probs = {k: v/total for k, v in move_probs.items()}
        
        return move_probs