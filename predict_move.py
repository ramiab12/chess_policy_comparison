#!/usr/bin/env python3
# Created by: Rami Abu Mukh, Omar Gharra, Ameer Khalaily
"""
Chess Move Predictor - Supports both CNN and Transformer models

Usage:
    python predict_move.py --model cnn --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    python predict_move.py --model transformer --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    python predict_move.py --model cnn --k 3  # Top-3 sampling
"""

import argparse
import sys
import torch
import chess
from pathlib import Path

# Import CNN components
from cnn_policy.model import ChessCNNPolicy
from cnn_policy.position_encoder import PositionEncoder
from cnn_policy.inference import predict_move as predict_move_cnn

# Import Transformer components
from transformer_policy.model import ChessTransformer
from transformer_policy.config import TransformerConfig
from transformer_policy.inference import predict_move as predict_move_transformer


def load_cnn_model(checkpoint_path: str, device: torch.device):
    """Load CNN model from checkpoint."""
    model = ChessCNNPolicy(
        num_input_channels=18,
        num_filters=256,
        num_blocks=15,
        dropout_rate=0.1
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def load_transformer_model(checkpoint_path: str, device: torch.device):
    """Load Transformer model from checkpoint."""
    config = TransformerConfig()
    model = ChessTransformer(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def encode_board_to_transformer_input(board: chess.Board, device: torch.device) -> dict:
    """Encode chess board to transformer input format."""
    # Piece encoding: 0=empty, 2-13=pieces (matching dataset encoding)
    piece_map = {
        (chess.PAWN, chess.BLACK): 2,
        (chess.PAWN, chess.WHITE): 3,
        (chess.ROOK, chess.BLACK): 4,
        (chess.ROOK, chess.WHITE): 5,
        (chess.KNIGHT, chess.BLACK): 6,
        (chess.KNIGHT, chess.WHITE): 7,
        (chess.BISHOP, chess.BLACK): 8,
        (chess.BISHOP, chess.WHITE): 9,
        (chess.QUEEN, chess.BLACK): 10,
        (chess.QUEEN, chess.WHITE): 11,
        (chess.KING, chess.BLACK): 12,
        (chess.KING, chess.WHITE): 13,
    }
    
    board_positions = torch.zeros(64, dtype=torch.long)
    for square in range(64):
        piece = board.piece_at(square)
        if piece:
            board_positions[square] = piece_map[(piece.piece_type, piece.color)]
    
    return {
        'turns': torch.tensor([[0 if board.turn == chess.WHITE else 1]], dtype=torch.long).to(device),
        'white_kingside_castling_rights': torch.tensor(
            [[int(board.has_kingside_castling_rights(chess.WHITE))]], dtype=torch.long
        ).to(device),
        'white_queenside_castling_rights': torch.tensor(
            [[int(board.has_queenside_castling_rights(chess.WHITE))]], dtype=torch.long
        ).to(device),
        'black_kingside_castling_rights': torch.tensor(
            [[int(board.has_kingside_castling_rights(chess.BLACK))]], dtype=torch.long
        ).to(device),
        'black_queenside_castling_rights': torch.tensor(
            [[int(board.has_queenside_castling_rights(chess.BLACK))]], dtype=torch.long
        ).to(device),
        'board_positions': board_positions.unsqueeze(0).to(device),
    }


def predict_move_transformer_wrapper(model, board: chess.Board, device: torch.device, k: int = 1):
    """Wrapper for transformer prediction that matches CNN interface."""
    model.eval()
    
    with torch.no_grad():
        # Encode board
        batch = encode_board_to_transformer_input(board, device)
        
        # Get predictions
        from_logits, to_logits = model(batch)
        from_logits = from_logits.squeeze(1)  # (1, 64)
        to_logits = to_logits.squeeze(1)      # (1, 64)
        
        # Combine log probabilities
        import torch.nn.functional as F
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
        if k == 1:
            best_idx = legal_predictions.argmax().item()
        else:
            topk_logits, topk_indices = torch.topk(legal_predictions, k, dim=-1)
            topk_probs = F.softmax(topk_logits, dim=-1)
            sampled_idx = torch.multinomial(topk_probs, 1)
            best_idx = topk_indices.gather(-1, sampled_idx).squeeze(-1).item()
        
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


def main():
    parser = argparse.ArgumentParser(
        description='Chess Move Predictor - Compare CNN vs Transformer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_move.py --model cnn
  python predict_move.py --model transformer --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
  python predict_move.py --model cnn --k 3
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['cnn', 'transformer'],
        default='cnn',
        help='Model to use for prediction (default: cnn)'
    )
    
    parser.add_argument(
        '--fen',
        type=str,
        default='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
        help='FEN string of the position (default: starting position)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=1,
        help='Top-k sampling (1=greedy, >1=stochastic) (default: 1)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file (default: auto-detect from model type)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # Print header
    print("\n" + "=" * 70)
    print(f"🎯 Chess Move Predictor - {args.model.upper()} Model")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Sampling: {'Greedy' if args.k == 1 else f'Top-{args.k}'}")
    
    # Load model
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        if args.model == 'cnn':
            checkpoint_path = Path('cnn_policy/checkpoints/checkpoint_step_55000.pth')
        else:
            checkpoint_path = Path('transformer_policy/checkpoints/checkpoint_step_55000.pth')
    
    if not checkpoint_path.exists():
        print(f"\n❌ Error: Checkpoint not found at {checkpoint_path}")
        print(f"   Please train the model first or specify --checkpoint path")
        sys.exit(1)
    
    print(f"Loading model from: {checkpoint_path}")
    
    try:
        if args.model == 'cnn':
            model = load_cnn_model(str(checkpoint_path), device)
            encoder = PositionEncoder()
            print("✓ CNN model loaded")
        else:
            model = load_transformer_model(str(checkpoint_path), device)
            encoder = None
            print("✓ Transformer model loaded")
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        sys.exit(1)
    
    # Parse board
    print(f"\nPosition (FEN):")
    print(f"   {args.fen}")
    
    try:
        board = chess.Board(args.fen)
    except Exception as e:
        print(f"\n❌ Invalid FEN: {e}")
        sys.exit(1)
    
    # Show current position
    print(f"\nCurrent Position:")
    print(board)
    print(f"\nTurn: {'White' if board.turn == chess.WHITE else 'Black'}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    
    # Predict move
    print(f"\nPredicting move...")
    
    try:
        if args.model == 'cnn':
            move = predict_move_cnn(model, board, encoder, device, k=args.k)
        else:
            move = predict_move_transformer_wrapper(model, board, device, k=args.k)
        
        print(f"\n✅ Predicted Move: {move.uci()}")
        print(f"   From: {chess.square_name(move.from_square)}")
        print(f"   To: {chess.square_name(move.to_square)}")
        
        # Verify legality
        if move in board.legal_moves:
            print(f"   Status: ✓ Legal")
        else:
            print(f"   Status: ✗ Illegal (shouldn't happen!)")
        
        # Show board after move
        board.push(move)
        print(f"\nAfter Move {move.uci()}:")
        print(board)
        
    except Exception as e:
        print(f"\n❌ Error predicting move: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

