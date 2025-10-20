import torch
import torch.nn.functional as F
import chess
from typing import Optional

# used for sampling k best moves, to make kind of a stochastic policy - by omar
def topk_sampling(logits: torch.Tensor, k: int = 1) -> torch.Tensor:
    if k == 1:
        return logits.argmax(dim=-1)
    else:
        topk_logits, topk_indices = torch.topk(logits, k, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)
        sampled_idx = torch.multinomial(topk_probs, 1)
        return topk_indices.gather(-1, sampled_idx).squeeze(-1)


def predict_move(
    model,
    board: chess.Board,
    device: torch.device,
    k: int = 1
) -> chess.Move:

    model.eval()
    
    with torch.no_grad():
        
        # Get predictions
        from_logits, to_logits = model(batch)
        from_logits = from_logits.squeeze(1)  # (1, 64)
        to_logits = to_logits.squeeze(1)      # (1, 64)
        
        # Combine log probabilities
        from_log_probs = F.log_softmax(from_logits, dim=-1).unsqueeze(2)  # (1, 64, 1)
        to_log_probs = F.log_softmax(to_logits, dim=-1).unsqueeze(1)      # (1, 1, 64)
        combined = (from_log_probs + to_log_probs).view(1, -1)  # (1, 4096)
        
        # Get legal moves
        legal_moves = [move.uci() for move in board.legal_moves]
        legal_moves_no_promo = list(set([m[:4] for m in legal_moves]))
        
        # Map to indices
        legal_indices = []
        for move_str in legal_moves_no_promo:
            from_idx = chess.SQUARE_NAMES.index(move_str[:2])
            to_idx = chess.SQUARE_NAMES.index(move_str[2:4])
            legal_indices.append(from_idx * 64 + to_idx)
        
        # Filter to legal moves
        legal_predictions = combined[:, legal_indices]
        
        # Sample
        best_idx = topk_sampling(legal_predictions, k=k).item()
        move_str = legal_moves_no_promo[best_idx]
        
        # Handle promotions (always queen)
        from_sq = chess.SQUARE_NAMES.index(move_str[:2])
        to_sq = chess.SQUARE_NAMES.index(move_str[2:4])
        piece = board.piece_at(from_sq)
        
        if piece and piece.piece_type == chess.PAWN:
            to_rank = to_sq // 8
            if (piece.color == chess.WHITE and to_rank == 7) or \
               (piece.color == chess.BLACK and to_rank == 0):
                move_str = move_str + "q"
        
        return chess.Move.from_uci(move_str)