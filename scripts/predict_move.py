#!/usr/bin/env python3
"""
CNN Policy Move Prediction Script
==================================
Load trained CNN model and predict best move for a given position.

Usage:
    python scripts/predict_move.py "fen_string"
    python scripts/predict_move.py  # Uses starting position

Example:
    python scripts/predict_move.py "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
"""

import sys
import torch
import chess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cnn_policy.model import ChessCNNPolicy
from cnn_policy.position_encoder import PositionEncoder
from cnn_policy.inference import predict_move


def load_model(checkpoint_path: str, device: torch.device):
    """
    Load trained CNN model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device
    
    Returns:
        Loaded model
    """
    print(f"📦 Loading model from: {checkpoint_path}")
    
    # Create model
    model = ChessCNNPolicy(
        num_input_channels=18,
        num_filters=256,
        num_blocks=15,
        dropout_rate=0.1
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Print info
    if 'step' in checkpoint:
        print(f"   Checkpoint step: {checkpoint['step']:,}")
    if 'best_val_accuracy' in checkpoint:
        print(f"   Best accuracy: {checkpoint['best_val_accuracy']:.2%}")
    
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"✅ Model loaded successfully!")
    
    return model


def predict_move_for_position(fen: str, model, encoder, device, k: int = 1):
    """
    Predict best move for a given position.
    
    Args:
        fen: FEN string representing position
        model: Trained CNN model
        encoder: Position encoder
        device: torch device
        k: Top-k sampling (1 = best move)
    
    Returns:
        Best move (chess.Move object)
    """
    # Parse FEN
    board = chess.Board(fen)
    
    # Predict move
    move = predict_move(model, board, encoder, device, k=k)
    
    return move, board


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("🎯 CNN Policy Move Predictor")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load model
    # Default to best checkpoint (step 55K)
    checkpoint_path = Path(__file__).parent.parent / 'cnn_policy/checkpoints/checkpoint_step_55000.pth'
    
    if not checkpoint_path.exists():
        # Try other checkpoints
        checkpoints_dir = Path(__file__).parent.parent / 'cnn_policy/checkpoints'
        available = list(checkpoints_dir.glob('*.pth'))
        
        if not available:
            print(f"❌ No checkpoints found in {checkpoints_dir}")
            print("\nPlease ensure you have a trained model checkpoint.")
            print("\nOptions:")
            print("  1. Download checkpoint from your training server")
            print("  2. Use git lfs pull (if checkpoint is in Git LFS)")
            print("  3. Retrain: python cnn_policy/train.py")
            sys.exit(1)
        
        # Check if it's a Git LFS pointer
        checkpoint_path = available[0]
        if checkpoint_path.stat().st_size < 1000:  # Less than 1KB = likely LFS pointer
            print(f"⚠️  Checkpoint appears to be a Git LFS pointer")
            print(f"   File size: {checkpoint_path.stat().st_size} bytes (expected ~68 MB)")
            print(f"\n   Download actual checkpoint with:")
            print(f"   git lfs pull")
            print(f"\n   Or copy from your training server.")
            sys.exit(1)
        
        print(f"⚠️  Using {checkpoint_path.name} (step 55K not found)")
    
    model = load_model(str(checkpoint_path), device)
    encoder = PositionEncoder()
    
    # Get FEN from command line or use default
    if len(sys.argv) > 1:
        fen = sys.argv[1]
    else:
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        print(f"\n💡 No FEN provided, using starting position")
    
    print(f"\n📋 Position (FEN):")
    print(f"   {fen}")
    
    # Predict move
    print(f"\n🎯 Predicting move...")
    
    try:
        move, board = predict_move_for_position(fen, model, encoder, device, k=1)
        
        print(f"\n✅ Predicted Move: {move.uci()}")
        print(f"   From: {chess.square_name(move.from_square)}")
        print(f"   To: {chess.square_name(move.to_square)}")
        
        # Show board
        print(f"\n📊 Current Position:")
        print(board)
        
        # Make move and show result
        board.push(move)
        print(f"\n📊 After Move {move.uci()}:")
        print(board)
        
        # Legal move check
        board = chess.Board(fen)  # Reset
        if move in board.legal_moves:
            print(f"\n✅ Move is legal!")
        else:
            print(f"\n❌ Move is illegal! (This shouldn't happen)")
        
    except Exception as e:
        print(f"\n❌ Error predicting move: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ Prediction complete!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

