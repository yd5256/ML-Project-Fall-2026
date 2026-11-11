"""Plot exploratory data analysis (EDA) for Pokémon TCG game states.

Run from the repository root:
    python3 src/plot_eda.py

Reads data/processed/game_states.csv and saves graphs to figures/eda/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "processed" / "game_states.csv"
FIGURES_DIR = ROOT / "figures" / "eda"


def plot_games_by_turn(df):
    """
    Show how many distinct games have a snapshot at each turn.
    """
    counts = (
        df[df["turn"].between(1, 15)]
        .groupby("turn")["game_id"]
        .nunique()
        .reindex(range(1, 16), fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index, counts.values, color="steelblue")
    ax.set(title="Games Reaching Each Turn", xlabel="Turn", ylabel="Number of games")
    ax.set_xticks(range(1, 16))
    ax.set_ylim(0, max(counts.max() * 1.12, 1))

    for turn, count in counts.items():
        ax.text(turn, count + counts.max() * 0.015, str(count),
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "games_by_turn.png", dpi=300)
    plt.close(fig)


def plot_prize_advantage(df, turn=8):
    """
    Compare P1's eventual win rate against its Prize-card advantage.
    """
    turn_data = df[df["turn"] == turn].copy()
    if turn_data.empty:
        print(f"No data for Turn {turn}; skipping prize advantage graph.")
        return

    # Fewer prizes remaining means a player is closer to winning.
    # Positive advantage means Player 1 has fewer prizes remaining.
    turn_data["prize_advantage"] = (
        turn_data["p2_prizes_remaining"] - turn_data["p1_prizes_remaining"]
    )
    stats = turn_data.groupby("prize_advantage")["p1_wins"].agg(
        win_rate="mean", games="count"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["seagreen" if x > 0 else "salmon" if x < 0 else "gray"
              for x in stats.index]
    ax.bar(stats.index, stats["win_rate"] * 100, color=colors)
    ax.set(
        title=f"Player 1 Win Rate by Prize Advantage (Turn {turn})",
        xlabel="P1 Prize advantage (positive = P1 ahead)",
        ylabel="Player 1 eventual win rate (%)",
    )
    ax.set_xticks(stats.index)
    ax.set_ylim(0, 110)
    ax.axhline(50, linestyle="--", color="black", alpha=0.35)

    for advantage, row in stats.iterrows():
        ax.text(advantage, row["win_rate"] * 100 + 2, f'n={int(row["games"])}',
                ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prize_advantage.png", dpi=300)
    plt.close(fig)


def plot_advantage_win_rate(df, p1_col, p2_col, label, filename, turn=8):
    """
    Plot Player 1's eventual win rate by feature advantage.
    """
    turn_data = df[df["turn"] == turn].copy()
    if turn_data.empty:
        print(f"No data for Turn {turn}; skipping {label} advantage graph.")
        return

    # Positive advantage means Player 1 has more of the feature.
    turn_data["advantage"] = turn_data[p1_col] - turn_data[p2_col]

    # Group HP differences into 200-HP intervals to reduce noise.
    if label == "Total HP":
        turn_data["advantage"] = (
            (turn_data["advantage"] // 200) * 200
        )

    summary = turn_data.groupby("advantage")["p1_wins"].agg(
        ["mean", "count"]
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        summary["advantage"],
        summary["mean"] * 100,
        s=summary["count"] * 2,
        alpha=0.7
    )

    fig.text(
        0.5, 0.01,
        "Bubble size represents the number of games at each advantage value",
        ha="center",
        fontsize=9
    )

    ax.axhline(50, color="gray", linestyle="--", alpha=0.6)

    if label == "Total HP":
        ax.set_xlabel("Player 1 Total HP advantage (200-HP bin lower bound)")
    else:
        ax.set_xlabel(f"Player 1 {label} advantage (P1 - P2)")

    ax.set_ylabel("Player 1 eventual win rate (%)")
    ax.set_title(f"Win Rate by {label} Advantage (Turn {turn})")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close(fig)


def main():
    df = pd.read_csv(DATA_FILE)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_games_by_turn(df)
    plot_prize_advantage(df, turn=8)

    plot_advantage_win_rate(
        df,
        "p1_total_hp",
        "p2_total_hp",
        "Total HP",
        "hp_advantage.png"
    )

    plot_advantage_win_rate(
        df,
        "p1_total_energy",
        "p2_total_energy",
        "Energy",
        "energy_advantage.png"
    )

    print("Saved figures/eda/games_by_turn.png")
    print("Saved figures/eda/prize_advantage.png")
    print("Saved figures/eda/hp_advantage.png")
    print("Saved figures/eda/energy_advantage.png")


if __name__ == "__main__":
    main()
