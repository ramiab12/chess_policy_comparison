"""
Play Games vs Stockfish
========================
Matches CT-EFT-20's evaluation protocol EXACTLY.

Plays games against Fairy Stockfish at levels 1-6 (same as CT-EFT-20):
- 500 games as White
- 500 games as Black
- Total: 1000 games per level

Reports same metrics as CT-EFT-20:
- Win/Loss/Draw counts
- Win ratio
- Elo difference (vs engine level)
"""

import chess
import chess.engine
import torch
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import math

sys.path.append(str(Path(__file__).parent.parent))

from cnn_policy.model import ChessCNNPolicy
from cnn_policy.position_encoder import PositionEncoder
from cnn_policy.inference import predict_move


# Stockfish levels (matching CT-EFT-20's evaluation)
# From: https://github.com/lichess-org/fishnet/blob/dc4be23256e3e5591578f0901f98f5835a138d73/src/api.rs#L224
STOCKFISH_LEVELS = {
    1: {"skill": 0, "depth": 1, "time": 0.05},
    2: {"skill": 1, "depth": 1, "time": 0.05},
    3: {"skill": 2, "depth": 3, "time": 0.1},
    4: {"skill": 5, "depth": 5, "time": 0.15},
    5: {"skill": 10, "depth": 10, "time": 0.2},
    6: {"skill": 15, "depth": 15, "time": 0.3},
}


def play_game(
    model,
    encoder,
    device,
    engine,
    model_color: chess.Color,
    level: int,
    k: int = 1
) -> str:
    """
    Play a single game.
    
    Args:
        model: CNN policy model
        encoder: Position encoder
        device: torch device
        engine: Stockfish engine
        model_color: Color the model plays (WHITE or BLACK)
        level: Stockfish strength level (1-6)
        k: Top-k sampling (1 = best move)
        
    Returns:
        Game result: "1-0", "0-1", or "1/2-1/2"
    """
    board = chess.Board()
    
    # Set engine parameters for this level
    engine_config = STOCKFISH_LEVELS[level]
    
    max_moves = 200  # Prevent infinite games
    
    for move_num in range(max_moves):
        if board.is_game_over():
            break
        
        if board.turn == model_color:
            # Model's turn
            try:
                move = predict_move(model, board, encoder, device, k=k)
                board.push(move)
            except Exception as e:
                # If model fails, it loses
                return "0-1" if model_color == chess.WHITE else "1-0"
        else:
            # Engine's turn
            try:
                result = engine.play(
                    board,
                    chess.engine.Limit(
                        depth=engine_config["depth"],
                        time=engine_config["time"]
                    ),
                    options={"Skill Level": engine_config["skill"]}
                )
                board.push(result.move)
            except Exception as e:
                # If engine fails, model wins
                return "1-0" if model_color == chess.WHITE else "0-1"
    
    # Determine result
    if board.is_checkmate():
        return "0-1" if board.turn else "1-0"
    elif board.is_stalemate() or board.is_insufficient_material():
        return "1/2-1/2"
    elif board.can_claim_draw():
        return "1/2-1/2"
    else:
        # Max moves reached
        return "1/2-1/2"


def evaluate_level(
    model,
    encoder,
    device,
    engine,
    level: int,
    games_per_color: int = 500,
    k: int = 1
) -> dict:
    """
    Evaluate against a single Stockfish level.
    Matches CT-EFT-20 protocol: 500 games per color.
    
    Args:
        model: CNN policy model
        encoder: Position encoder
        device: torch device
        engine: Stockfish engine
        level: Stockfish level (1-6)
        games_per_color: Games to play as each color
        k: Top-k sampling
        
    Returns:
        Results dictionary
    """
    print(f"\n{'='*70}")
    print(f"Evaluating vs Stockfish Level {level}")
    print(f"{'='*70}")
    
    results = {
        'level': level,
        'white_wins': 0,
        'white_losses': 0,
        'white_draws': 0,
        'black_wins': 0,
        'black_losses': 0,
        'black_draws': 0
    }
    
    # Play as White
    print(f"\n🎮 Playing {games_per_color} games as WHITE...")
    for game_num in tqdm(range(games_per_color), desc="White games"):
        result = play_game(model, encoder, device, engine, chess.WHITE, level, k)
        
        if result == "1-0":
            results['white_wins'] += 1
        elif result == "0-1":
            results['white_losses'] += 1
        else:
            results['white_draws'] += 1
    
    # Play as Black
    print(f"\n🎮 Playing {games_per_color} games as BLACK...")
    for game_num in tqdm(range(games_per_color), desc="Black games"):
        result = play_game(model, encoder, device, engine, chess.BLACK, level, k)
        
        if result == "0-1":
            results['black_wins'] += 1
        elif result == "1-0":
            results['black_losses'] += 1
        else:
            results['black_draws'] += 1
    
    # Calculate totals
    total_wins = results['white_wins'] + results['black_wins']
    total_losses = results['white_losses'] + results['black_losses']
    total_draws = results['white_draws'] + results['black_draws']
    total_games = total_wins + total_losses + total_draws
    
    results['total_wins'] = total_wins
    results['total_losses'] = total_losses
    results['total_draws'] = total_draws
    results['total_games'] = total_games
    
    # Win ratio (matching CT-EFT-20 calculation)
    win_ratio = (total_wins + 0.5 * total_draws) / total_games
    results['win_ratio'] = win_ratio
    
    # Elo difference (matching CT-EFT-20)
    if win_ratio >= 0.99:
        elo_diff = 800  # Cap at +800
    elif win_ratio <= 0.01:
        elo_diff = -800  # Cap at -800
    else:
        elo_diff = 400 * math.log10(win_ratio / (1 - win_ratio))
    
    results['elo_difference'] = elo_diff
    
    # Standard error (for confidence intervals)
    n = total_games
    p = win_ratio
    se = math.sqrt(p * (1 - p) / n) if n > 0 else 0
    elo_se = 400 / (math.log(10) * p * (1 - p)) * se if 0 < p < 1 else 0
    
    results['elo_std_error'] = elo_se
    
    # Print results
    print(f"\n📊 Results for Level {level}:")
    print(f"   Wins:   {total_wins}/{total_games} ({100*total_wins/total_games:.1f}%)")
    print(f"   Losses: {total_losses}/{total_games} ({100*total_losses/total_games:.1f}%)")
    print(f"   Draws:  {total_draws}/{total_games} ({100*total_draws/total_games:.1f}%)")
    print(f"   Win ratio: {win_ratio:.2%}")
    print(f"   Elo difference: {elo_diff:+.1f} (± {elo_se:.1f})")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CNN policy model vs Stockfish (matching CT-EFT-20 protocol)"
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--stockfish', type=str, default='stockfish',
                       help='Path to Stockfish binary')
    parser.add_argument('--levels', type=str, default='1,2,3,4,5,6',
                       help='Comma-separated Stockfish levels to test')
    parser.add_argument('--games-per-color', type=int, default=500,
                       help='Games to play as each color (default: 500, like CT-EFT-20)')
    parser.add_argument('--k', type=int, default=1,
                       help='Top-k sampling (1=best move)')
    parser.add_argument('--threads', type=int, default=8,
                       help='Stockfish threads (default: 8, like CT-EFT-20)')
    parser.add_argument('--hash', type=int, default=8,
                       help='Stockfish hash size in GB (default: 8, like CT-EFT-20)')
    parser.add_argument('--output', type=str, default='evaluation/results.csv',
                       help='Output CSV file')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("CNN POLICY EVALUATION vs STOCKFISH")
    print("(Matching CT-EFT-20 Protocol)")
    print("=" * 70)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Stockfish: {args.stockfish}")
    print(f"Levels: {args.levels}")
    print(f"Games per color: {args.games_per_color}")
    print(f"Engine threads: {args.threads}")
    print(f"Engine hash: {args.hash} GB")
    print()
    
    # Load model
    print("📦 Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = ChessCNNPolicy().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    encoder = PositionEncoder()
    
    print(f"✅ Model loaded on {device}")
    
    # Initialize Stockfish
    print(f"\n🔧 Initializing Stockfish...")
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"Threads": args.threads, "Hash": args.hash * 1024})
    
    print(f"✅ Engine ready: {engine.id['name']}")
    
    # Parse levels
    levels = [int(l.strip()) for l in args.levels.split(',')]
    
    # Evaluate each level
    all_results = []
    
    for level in levels:
        results = evaluate_level(
            model, encoder, device, engine,
            level, args.games_per_color, args.k
        )
        all_results.append(results)
    
    # Close engine
    engine.quit()
    
    # Save results
    df = pd.DataFrame(all_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    # Print final summary table (like CT-EFT-20)
    print("\n" + "=" * 70)
    print("FINAL RESULTS (CT-EFT-20 Format)")
    print("=" * 70)
    print()
    print("| Level | Games | Wins | Losses | Draws | Win Ratio | Elo Diff |")
    print("|-------|-------|------|--------|-------|-----------|----------|")
    
    for _, row in df.iterrows():
        level = int(row['level'])
        games = int(row['total_games'])
        wins = int(row['total_wins'])
        losses = int(row['total_losses'])
        draws = int(row['total_draws'])
        win_ratio = row['win_ratio']
        elo_diff = row['elo_difference']
        elo_se = row['elo_std_error']
        
        print(f"| **{level}** | {games} | {wins} | {losses} | {draws} | "
              f"**{win_ratio:.2%}** | {elo_diff:+.1f} (± {elo_se:.1f}) |")
    
    print()
    print("=" * 70)
    print("\n✅ Evaluation complete!")
    print("\nCompare these results to CT-EFT-20's published numbers:")
    print("https://github.com/sgrvinod/chess-transformers#ct-eft-20")


if __name__ == "__main__":
    main()

