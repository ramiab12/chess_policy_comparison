"""
Preprocess LE22ct Dataset
==========================
Convert H5 format to CSV for CNN training.

Input: LE22ct.h5 (from chess-transformers)
Output: train.csv, val.csv
Format: fen, move, from_square, to_square
"""

import h5py
import pandas as pd
import numpy as np
import chess
from pathlib import Path
from tqdm import tqdm
import sys


def decode_board_position(board_encoding, piece_vocab):
    """
    Decode board position from indices to FEN.
    
    Args:
        board_encoding: Array of 64 piece indices
        piece_vocab: Vocabulary mapping
        
    Returns:
        FEN string
    """
    # This is a simplified version
    # You may need to adjust based on their exact encoding
    
    pieces = []
    for idx in board_encoding:
        if idx == 0:
            pieces.append(None)
        else:
            # Map index to piece
            # (Need to match their PIECES vocabulary)
            pieces.append(idx)
    
    # Convert to FEN (simplified - needs their exact format)
    # For now, return placeholder
    return "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def decode_move(move_uci_idx, move_vocab):
    """Decode move from index to UCI string."""
    # Match their UCI_MOVES vocabulary
    return move_vocab[move_uci_idx]


def process_h5_dataset(h5_path: Path, output_dir: Path):
    """
    Process LE22ct H5 file to CSV.
    
    Args:
        h5_path: Path to LE22ct.h5
        output_dir: Output directory
    """
    print(f"\n📂 Processing {h5_path}...")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print("\nH5 file contents:")
            for key in f.keys():
                print(f"  - {key}: {f[key].shape if hasattr(f[key], 'shape') else 'group'}")
            
            # Load encoded data table
            if 'encoded' in f:
                encoded = f['encoded']
                print(f"\nEncoded table columns:")
                for col in encoded.dtype.names:
                    print(f"  - {col}")
                
                # Read data
                print(f"\nReading {len(encoded):,} positions...")
                data = encoded[:]
                
                # Convert to lists for DataFrame
                positions = []
                
                print("Converting to FEN format...")
                for i in tqdm(range(len(data)), desc="Processing"):
                    row = data[i]
                    
                    # Extract fields
                    board_pos = row['board_position'] if 'board_position' in row.dtype.names else None
                    moves = row['moves'] if 'moves' in row.dtype.names else None
                    
                    # Get first move (the move to predict)
                    if moves is not None and len(moves) > 0:
                        first_move_idx = moves[0]
                        
                        # Decode to UCI
                        # (This requires their UCI_MOVES vocabulary)
                        # For now, we'll load this from the chess-transformers library
                        
                        positions.append({
                            'board_encoded': board_pos,
                            'move_idx': first_move_idx
                        })
                
                print(f"\n✅ Processed {len(positions):,} positions")
                
                # This is simplified - full implementation needs their vocabulary
                print("\n⚠️  Note: Full preprocessing requires chess-transformers library")
                print("   to properly decode board positions and moves.")
                print("\n   Alternative: Use their data loading directly or")
                print("   convert using their utilities.")
                
            else:
                print("❌ 'encoded' table not found in H5 file!")
                print("   Available tables:", list(f.keys()))
        
    except Exception as e:
        print(f"❌ Error processing H5 file: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 70)
    print("LE22ct Dataset Preprocessing")
    print("=" * 70)
    
    # Paths
    h5_path = Path("dataset/raw/LE22ct/LE22ct.h5")
    output_dir = Path("dataset/processed")
    
    if not h5_path.exists():
        print(f"\n❌ H5 file not found: {h5_path}")
        print("\n   Please run download script first:")
        print("   python scripts/download_le22ct.py")
        sys.exit(1)
    
    # Process
    process_h5_dataset(h5_path, output_dir)
    
    print("\n" + "=" * 70)
    print("NOTE: Complete preprocessing requires chess-transformers library")
    print("=" * 70)
    print("\nFor full preprocessing, you can:")
    print("1. Use chess-transformers library utilities")
    print("2. Or manually create train.csv with columns:")
    print("   - fen: FEN string")
    print("   - move: UCI move (e.g., 'e2e4')")
    print("   - from_square: 0-63")
    print("   - to_square: 0-63")
    print()


if __name__ == "__main__":
    main()

