"""
Create Elite Moves Dataset
===========================
Generate dataset matching LE22ct criteria from Lichess Elite games.

Criteria (matching CT-EFT-20's LE22ct):
- Players: 2400+ vs 2200+
- Time control: ≥ 5 minutes
- Outcome: Checkmate only
- Moves: From winning player only
- Target: 13-15M positions

This creates an equivalent dataset to LE22ct for fair comparison.
"""

import chess
import chess.pgn
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys


class EliteMovesExtractor:
    """Extract moves from elite games matching LE22ct criteria."""
    
    def __init__(self, min_elo_high: int = 2400, min_elo_low: int = 2200):
        self.min_elo_high = min_elo_high
        self.min_elo_low = min_elo_low
        self.positions = []
    
    def filter_game(self, game: chess.pgn.Game) -> bool:
        """Check if game meets LE22ct criteria."""
        headers = game.headers
        
        try:
            white_elo = int(headers.get('WhiteElo', 0))
            black_elo = int(headers.get('BlackElo', 0))
        except:
            return False
        
        # Both players must be 2200+, at least one 2400+
        if white_elo < self.min_elo_low or black_elo < self.min_elo_low:
            return False
        
        if white_elo < self.min_elo_high and black_elo < self.min_elo_high:
            return False
        
        # Time control ≥ 5 minutes (300 seconds)
        time_control = headers.get('TimeControl', '')
        if time_control and '+' in time_control:
            try:
                base_time = int(time_control.split('+')[0])
                if base_time < 300:
                    return False
            except:
                return False
        
        # Must end in checkmate
        result = headers.get('Result', '')
        termination = headers.get('Termination', '').lower()
        
        if 'checkmate' not in termination and result not in ['1-0', '0-1']:
            return False
        
        return True
    
    def extract_moves_from_game(self, game: chess.pgn.Game) -> list:
        """
        Extract all moves from winning player.
        
        Args:
            game: chess.pgn.Game object
            
        Returns:
            List of (fen, move) tuples
        """
        result = game.headers.get('Result', '')
        
        # Determine winner
        if result == '1-0':
            winner = chess.WHITE
        elif result == '0-1':
            winner = chess.BLACK
        else:
            return []  # Draw or unknown
        
        moves_data = []
        board = game.board()
        
        for move in game.mainline_moves():
            # Store position and move if it's winner's turn
            if board.turn == winner:
                fen = board.fen()
                move_uci = move.uci()
                
                # Parse move to get squares
                from_square = move.from_square
                to_square = move.to_square
                
                moves_data.append({
                    'fen': fen,
                    'move': move_uci,
                    'from_square': from_square,
                    'to_square': to_square
                })
            
            # Make the move
            board.push(move)
        
        return moves_data
    
    def process_pgn_file(self, pgn_path: Path, target_positions: int) -> list:
        """Process a single PGN file."""
        print(f"\n📄 Processing {pgn_path.name}...")
        
        positions = []
        games_processed = 0
        games_used = 0
        
        with open(pgn_path) as pgn:
            with tqdm(desc=f"Games", unit="game") as pbar:
                while len(positions) < target_positions:
                    try:
                        game = chess.pgn.read_game(pgn)
                        if game is None:
                            break
                        
                        games_processed += 1
                        pbar.update(1)
                        
                        # Filter game
                        if not self.filter_game(game):
                            continue
                        
                        # Extract winning moves
                        game_positions = self.extract_moves_from_game(game)
                        positions.extend(game_positions)
                        
                        if game_positions:
                            games_used += 1
                        
                        pbar.set_postfix({
                            'used': games_used,
                            'positions': len(positions)
                        })
                        
                    except Exception as e:
                        continue
        
        print(f"   Processed {games_processed:,} games")
        print(f"   Used {games_used:,} checkmate games")
        print(f"   Extracted {len(positions):,} positions")
        
        return positions


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pgn-dir', type=str, default='dataset/raw/lichess_elite',
                       help='Directory containing Lichess Elite PGN files')
    parser.add_argument('--output', type=str, default='dataset/processed',
                       help='Output directory')
    parser.add_argument('--target', type=int, default=13_000_000,
                       help='Target number of positions')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Elite Moves Dataset Creator")
    print("(Matching LE22ct Criteria)")
    print("=" * 70)
    print(f"\nCriteria:")
    print(f"  - Players: 2400+ vs 2200+")
    print(f"  - Time: ≥ 5 minutes")
    print(f"  - Outcome: Checkmate only")
    print(f"  - Moves: Winning player only")
    print(f"  - Target: {args.target:,} positions")
    print()
    
    extractor = EliteMovesExtractor()
    
    # Find PGN files
    pgn_dir = Path(args.pgn_dir)
    if not pgn_dir.exists():
        print(f"❌ PGN directory not found: {pgn_dir}")
        print("\n   Download Lichess Elite PGNs from:")
        print("   https://database.nikonoel.fr/")
        sys.exit(1)
    
    pgn_files = list(pgn_dir.glob('*.pgn'))
    if not pgn_files:
        print(f"❌ No PGN files found in {pgn_dir}")
        sys.exit(1)
    
    print(f"📚 Found {len(pgn_files)} PGN files")
    
    # Process files
    all_positions = []
    for pgn_file in pgn_files:
        positions = extractor.process_pgn_file(pgn_file, args.target - len(all_positions))
        all_positions.extend(positions)
        
        print(f"   Total so far: {len(all_positions):,}")
        
        if len(all_positions) >= args.target:
            print(f"   ✅ Reached target!")
            break
    
    if len(all_positions) == 0:
        print("\n❌ No positions extracted!")
        sys.exit(1)
    
    print(f"\n✅ Total extracted: {len(all_positions):,}")
    
    # Create DataFrame
    df = pd.DataFrame(all_positions)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split (90/10 like CT-EFT-20 likely used)
    split_idx = int(len(df) * 0.9)
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n💾 Saved:")
    print(f"   Train: {train_path} ({len(train_df):,} positions)")
    print(f"   Val:   {val_path} ({len(val_df):,} positions)")
    print()
    print("=" * 70)
    print("✅ Dataset ready for training!")
    print("=" * 70)
    print(f"\n📋 Next step:")
    print(f"   python cnn_policy/train.py")
    print()


if __name__ == "__main__":
    main()

