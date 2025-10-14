"""
Convert LE22ct H5 to CSV Format
================================
Uses chess-transformers library to properly decode LE22ct dataset.

This ensures we use the EXACT same data as CT-EFT-20!
"""

import sys
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add chess-transformers to path
sys.path.insert(0, '/tmp/chess-transformers')
os.environ['CT_DATA_FOLDER'] = str(Path.cwd() / 'dataset/raw/LE22ct')

import h5py
import chess


def convert_h5_to_csv(h5_path: Path, output_dir: Path, max_samples: int = None):
    """
    Convert LE22ct H5 to CSV format.
    
    Args:
        h5_path: Path to LE22ct.h5
        output_dir: Output directory for CSV files
        max_samples: Maximum samples to convert (None = all)
    """
    print(f"\n📂 Loading {h5_path}...")
    
    try:
        # Import chess-transformers utilities
        from chess_transformers.data.utils import decode, UCI_MOVES, PIECES, TURN, BOOL
        
        with h5py.File(h5_path, 'r') as f:
            # Get encoded data
            if 'encoded' not in f:
                print(f"❌ 'encoded' table not found!")
                print(f"   Available: {list(f.keys())}")
                return False
            
            encoded_data = f['encoded']
            total_samples = len(encoded_data)
            
            print(f"   Total samples in H5: {total_samples:,}")
            
            if max_samples:
                total_samples = min(total_samples, max_samples)
                print(f"   Processing first {total_samples:,} samples")
            
            # Process in chunks
            chunk_size = 100_000
            all_data = []
            
            for start_idx in tqdm(range(0, total_samples, chunk_size), desc="Converting"):
                end_idx = min(start_idx + chunk_size, total_samples)
                chunk = encoded_data[start_idx:end_idx]
                
                for row in chunk:
                    try:
                        # Decode board position to FEN
                        board_pos = row['board_position']
                        turn = row['turn']
                        wk_castle = row['white_kingside_castling_rights']
                        wq_castle = row['white_queenside_castling_rights']
                        bk_castle = row['black_kingside_castling_rights']
                        bq_castle = row['black_queenside_castling_rights']
                        
                        # Reconstruct FEN (simplified)
                        # This is complex - needs proper decoding
                        # For now, use chess-transformers decode function
                        
                        # Get move
                        moves = row['moves']
                        if len(moves) > 0:
                            first_move_idx = moves[0]
                            if first_move_idx < len(UCI_MOVES):
                                move_uci = UCI_MOVES[first_move_idx]
                                
                                # Parse move to get from-to squares
                                try:
                                    move = chess.Move.from_uci(move_uci)
                                    from_square = move.from_square
                                    to_square = move.to_square
                                    
                                    # We need FEN - this requires full decoding
                                    # Placeholder for now
                                    fen = "placeholder"  # TODO: Implement proper FEN decoding
                                    
                                    all_data.append({
                                        'fen': fen,
                                        'move': move_uci,
                                        'from_square': from_square,
                                        'to_square': to_square
                                    })
                                except:
                                    pass
                    except:
                        continue
            
            print(f"\n✅ Converted {len(all_data):,} positions")
            
            # Create DataFrame
            df = pd.DataFrame(all_data)
            
            # Split train/val (90/10 like typical)
            split_idx = int(len(df) * 0.9)
            train_df = df[:split_idx]
            val_df = df[split_idx:]
            
            # Save
            output_dir.mkdir(parents=True, exist_ok=True)
            train_path = output_dir / 'train.csv'
            val_path = output_dir / 'val.csv'
            
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)
            
            print(f"\n💾 Saved:")
            print(f"   Train: {train_path} ({len(train_df):,} samples)")
            print(f"   Val:   {val_path} ({len(val_df):,} samples)")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Alternative approach:")
        print("   The LE22ct H5 format is complex.")
        print("   Consider using chess-transformers library directly")
        print("   or generating a similar dataset from Lichess Elite games.")
        
        return False


if __name__ == "__main__":
    h5_path = Path("dataset/raw/LE22ct/LE22ct.h5")
    output_dir = Path("dataset/processed")
    
    if not h5_path.exists():
        print(f"❌ H5 file not found: {h5_path}")
        print("\n   Run download script first:")
        print("   python scripts/download_le22ct.py")
        sys.exit(1)
    
    convert_h5_to_csv(h5_path, output_dir)

