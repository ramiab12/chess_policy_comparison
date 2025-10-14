"""
Compare CNN Results to CT-EFT-20 Baseline
==========================================
Creates side-by-side comparison table.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# CT-EFT-20 Published Results
# From: https://github.com/sgrvinod/chess-transformers
CT_EFT_20_RESULTS = {
    1: {"games": 1000, "wins": 994, "losses": 0, "draws": 6, "win_ratio": 0.997, "elo_diff": 1008.63},
    2: {"games": 1000, "wins": 988, "losses": 0, "draws": 12, "win_ratio": 0.994, "elo_diff": 887.69},
    3: {"games": 1000, "wins": 942, "losses": 11, "draws": 47, "win_ratio": 0.9655, "elo_diff": 578.77},
    4: {"games": 1000, "wins": 697, "losses": 192, "draws": 111, "win_ratio": 0.7525, "elo_diff": 193.17},
    5: {"games": 1000, "wins": 482, "losses": 379, "draws": 139, "win_ratio": 0.5515, "elo_diff": 35.91},
    6: {"games": 1000, "wins": 61, "losses": 872, "draws": 67, "win_ratio": 0.0945, "elo_diff": -392.58},
}


def compare_results(cnn_results_csv: str):
    """
    Compare CNN results to CT-EFT-20 baseline.
    
    Args:
        cnn_results_csv: Path to CNN evaluation results CSV
    """
    # Load CNN results
    cnn_df = pd.read_csv(cnn_results_csv)
    
    print("=" * 70)
    print("COMPARISON: CNN POLICY vs CT-EFT-20 TRANSFORMER")
    print("=" * 70)
    print()
    
    # Create comparison table
    print("| Level | Model | Wins | Losses | Draws | Win Ratio | Elo Diff |")
    print("|-------|-------|------|--------|-------|-----------|----------|")
    
    improvements = []
    
    for level in [1, 2, 3, 4, 5, 6]:
        # CT-EFT-20 baseline
        ct_data = CT_EFT_20_RESULTS[level]
        print(f"| **{level}** | CT-EFT-20 | {ct_data['wins']} | {ct_data['losses']} | "
              f"{ct_data['draws']} | {ct_data['win_ratio']:.2%} | {ct_data['elo_diff']:+.1f} |")
        
        # CNN results (if available)
        cnn_row = cnn_df[cnn_df['level'] == level]
        if not cnn_row.empty:
            cnn_wins = int(cnn_row['total_wins'].values[0])
            cnn_losses = int(cnn_row['total_losses'].values[0])
            cnn_draws = int(cnn_row['total_draws'].values[0])
            cnn_win_ratio = cnn_row['win_ratio'].values[0]
            cnn_elo_diff = cnn_row['elo_difference'].values[0]
            
            print(f"| **{level}** | **CNN** | {cnn_wins} | {cnn_losses} | "
                  f"{cnn_draws} | **{cnn_win_ratio:.2%}** | **{cnn_elo_diff:+.1f}** |")
            
            # Calculate improvement
            elo_improvement = cnn_elo_diff - ct_data['elo_diff']
            improvements.append({
                'level': level,
                'elo_improvement': elo_improvement,
                'win_ratio_improvement': cnn_win_ratio - ct_data['win_ratio']
            })
        else:
            print(f"| **{level}** | CNN | - | - | - | - | - |")
    
    print()
    
    # Summary
    if improvements:
        print("=" * 70)
        print("IMPROVEMENT SUMMARY")
        print("=" * 70)
        print()
        
        for imp in improvements:
            level = imp['level']
            elo_imp = imp['elo_improvement']
            wr_imp = imp['win_ratio_improvement']
            
            direction = "✅" if elo_imp > 0 else "❌"
            print(f"Level {level}: {direction} CNN {elo_imp:+.1f} Elo "
                  f"({wr_imp:+.1%} win ratio)")
        
        # Average improvement at key levels (4-5)
        key_improvements = [imp for imp in improvements if imp['level'] in [4, 5]]
        if key_improvements:
            avg_elo = sum(imp['elo_improvement'] for imp in key_improvements) / len(key_improvements)
            avg_wr = sum(imp['win_ratio_improvement'] for imp in key_improvements) / len(key_improvements)
            
            print()
            print(f"Average improvement (Levels 4-5): {avg_elo:+.1f} Elo, {avg_wr:+.1%} win ratio")
            
            if avg_elo > 50:
                print("\n✅ CNN shows SIGNIFICANT advantage over Transformer!")
            elif avg_elo > 20:
                print("\n✅ CNN shows MODERATE advantage over Transformer")
            elif avg_elo > 0:
                print("\n✅ CNN shows SLIGHT advantage over Transformer")
            else:
                print("\n⚠️  CNN does not outperform Transformer significantly")
        
        print()
    
    print("=" * 70)


def plot_comparison(cnn_results_csv: str):
    """Create visualization comparing CNN to CT-EFT-20."""
    cnn_df = pd.read_csv(cnn_results_csv)
    
    # Prepare data
    levels = []
    ct_elos = []
    cnn_elos = []
    
    for level in [1, 2, 3, 4, 5, 6]:
        ct_data = CT_EFT_20_RESULTS[level]
        cnn_row = cnn_df[cnn_df['level'] == level]
        
        if not cnn_row.empty:
            levels.append(level)
            ct_elos.append(ct_data['elo_diff'])
            cnn_elos.append(cnn_row['elo_difference'].values[0])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(levels))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], ct_elos, width, label='CT-EFT-20 (Transformer)', alpha=0.8)
    ax.bar([i + width/2 for i in x], cnn_elos, width, label='CNN Policy', alpha=0.8)
    
    ax.set_xlabel('Stockfish Level')
    ax.set_ylabel('Elo Difference vs Engine')
    ax.set_title('CNN vs CT-EFT-20: Playing Strength Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('evaluation/comparison_plot.png', dpi=300)
    print("📊 Saved plot: evaluation/comparison_plot.png")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn-results', type=str, required=True,
                       help='Path to CNN results CSV')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plot')
    
    args = parser.parse_args()
    
    compare_results(args.cnn_results)
    
    if args.plot:
        plot_comparison(args.cnn_results)

