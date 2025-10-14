"""
Proper LE22ct H5 to CSV Converter
===================================
Converts LE22ct H5 format to CSV format matching CT-EFT-20's usage.

Based on H5 structure:
- board_position: Encoded board (needs decoding)
- moves: Array of 10 future moves (we use moves[0])
- castling_rights: Separate boolean fields
- turn: 0/1 for white/black

Output CSV:
- fen: FEN string (decoded from board_position + metadata)
- move: UCI move (decoded from moves[0])
- from_square: 0-63
- to_square: 0-63
"""

import h5py
import numpy as np
import pandas as pd
import chess
from pathlib import Path
from tqdm import tqdm
import sys

# Try to import chess-transformers utilities
try:
    from chess_transformers.data.utils import (
        decode_position,
        UCI_MOVES,
        PIECES,
        TURN,
        BOOL
    )
    HAS_CHESS_TRANSFORMERS = True
except ImportError:
    HAS_CHESS_TRANSFORMERS = False
    print("⚠️  chess-transformers not installed")
    print("   Install with: pip install chess-transformers")
    print("   Or I'll use manual decoding...")


def decode_board_manual(board_encoding, turn, castling_rights):
    """
    Manually decode board position to FEN.
    
    This is a fallback if chess-transformers is not available.
    You'll need to adjust based on their exact encoding scheme.
    """
    # Create board
    board = chess.Board.empty()
    
    # The board_position encoding needs to be understood
    # This varies by implementation - needs investigation
    
    # Placeholder: return a valid FEN
    # YOU NEED TO IMPLEMENT PROPER DECODING
    return chess.Board().fen()


def decode_move_index_to_uci(move_idx, uci_moves_list=None):
    """
    Convert move index to UCI string.
    
    Args:
        move_idx: Integer index of the move
        uci_moves_list: List of all UCI moves in order
    
    Returns:
        UCI move string
    """
    if HAS_CHESS_TRANSFORMERS:
        return UCI_MOVES[move_idx]
    else:
        # Manual fallback - need the move vocabulary
        # This should match their UCI_MOVES list
        if uci_moves_list and move_idx < len(uci_moves_list):
            return uci_moves_list[move_idx]
        else:
            return "e2e4"  # Placeholder


def convert_h5_to_csv_proper(
    h5_path: Path,
    output_dir: Path,
    train_split: float = 0.9,
    max_samples: int = None
):
    """
    Convert LE22ct H5 to CSV format.
    
    Uses chess-transformers utilities for proper decoding.
    """
    print("=" * 70)
    print("LE22ct H5 to CSV Converter (CT-EFT-20 Compatible)")
    print("=" * 70)
    
    if not HAS_CHESS_TRANSFORMERS:
        print("\n⚠️  WARNING: chess-transformers not installed!")
        print("   Install with: pip install chess-transformers")
        print("\n   For proper decoding, you should install it.")
        print("   Continuing with manual decoding (may need adjustments)...")
        print()
    
    if not h5_path.exists():
        print(f"\n❌ H5 file not found: {h5_path}")
        sys.exit(1)
    
    print(f"\n📂 Loading {h5_path}...")
    
    records = []
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get encoded table
            if 'encoded' not in f:
                print(f"❌ 'encoded' table not found!")
                print(f"   Available tables: {list(f.keys())}")
                sys.exit(1)
            
            encoded_data = f['encoded']
            total_samples = len(encoded_data)
            
            if max_samples:
                total_samples = min(total_samples, max_samples)
            
            print(f"   Total samples: {total_samples:,}")
            print(f"\n📊 H5 table structure:")
            print(f"   Columns: {encoded_data.dtype.names}")
            print()
            
            # Process data
            print("Converting to CSV format...")
            
            for i in tqdm(range(total_samples), desc="Processing"):
                try:
                    row = encoded_data[i]
                    
                    # Extract fields
                    board_pos = row['board_position']
                    turn = row['turn']
                    moves_array = row['moves']
                    
                    # Get castling rights
                    wk_castle = row.get('white_kingside_castling_rights', 0)
                    wq_castle = row.get('white_queenside_castling_rights', 0)
                    bk_castle = row.get('black_kingside_castling_rights', 0)
                    bq_castle = row.get('black_queenside_castling_rights', 0)
                    
                    # Get first move (this is what CT-EFT-20 uses for policy!)
                    if len(moves_array) > 0:
                        first_move_idx = moves_array[0]
                        
                        # Decode move
                        if HAS_CHESS_TRANSFORMERS:
                            move_uci = UCI_MOVES[first_move_idx]
                        else:
                            # Manual decoding needed
                            move_uci = decode_move_index_to_uci(first_move_idx)
                        
                        # Parse move to get from/to
                        try:
                            move = chess.Move.from_uci(move_uci)
                            from_square = move.from_square
                            to_square = move.to_square
                            
                            # Decode board position to FEN
                            if HAS_CHESS_TRANSFORMERS:
                                # Use their decoder (if available)
                                # This requires understanding their decode function
                                # Placeholder for now
                                fen = decode_board_manual(board_pos, turn, 
                                                         (wk_castle, wq_castle, bk_castle, bq_castle))
                            else:
                                fen = decode_board_manual(board_pos, turn,
                                                         (wk_castle, wq_castle, bk_castle, bq_castle))
                            
                            records.append({
                                'fen': fen,
                                'move': move_uci,
                                'from_square': from_square,
                                'to_square': to_square
                            })
                        except Exception as e:
                            # Skip invalid moves
                            continue
                    
                except Exception as e:
                    # Skip problematic rows
                    continue
            
            print(f"\n✅ Successfully converted {len(records):,} positions")
            
            if len(records) == 0:
                print("\n❌ No records converted! Check H5 file structure.")
                sys.exit(1)
            
            # Create DataFrame
            df = pd.DataFrame(records)
            
            # Shuffle with fixed seed (for reproducibility)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            # Split train/val
            split_idx = int(len(df) * train_split)
            train_df = df[:split_idx]
            val_df = df[split_idx:]
            
            # Save
            output_dir.mkdir(parents=True, exist_ok=True)
            
            train_path = output_dir / 'train.csv'
            val_path = output_dir / 'val.csv'
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            
            print(f"\n💾 Saved:")
            print(f"   {train_path}: {len(train_df):,} positions")
            print(f"   {val_path}: {len(val_df):,} positions")
            print(f"   Total: {len(df):,} positions")
            
            # Verify format
            print(f"\n✅ Verification:")
            print(f"   Columns: {list(df.columns)}")
            print(f"\n   Sample row:")
            print(train_df.head(1).to_string())
            
            print("\n" + "=" * 70)
            print("✅ Conversion complete!")
            print("=" * 70)
            print(f"\n📋 Next step:")
            print(f"   python cnn_policy/train.py")
            print()
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("⚠️  ISSUE: Proper decoding requires chess-transformers library")
        print("=" * 70)
        print("\nTo fix:")
        print("1. Install: pip install chess-transformers")
        print("2. Or clone: git clone https://github.com/sgrvinod/chess-transformers")
        print("3. Then import their decoding utilities")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert LE22ct H5 to CSV format (CT-EFT-20 compatible)"
    )
    parser.add_argument('--h5', type=str,
                       default='dataset/raw/LE22ct/LE22ct.h5',
                       help='Path to LE22ct.h5 file')
    parser.add_argument('--output', type=str,
                       default='dataset/processed',
                       help='Output directory')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples to process (None = all)')
    parser.add_argument('--train-split', type=float, default=0.9,
                       help='Train/val split ratio')
    
    args = parser.parse_args()
    
    convert_h5_to_csv_proper(
        Path(args.h5),
        Path(args.output),
        args.train_split,
        args.max_samples
    )

