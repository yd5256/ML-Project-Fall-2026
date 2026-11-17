"""
error_analysis.py — Analyze prediction errors in Pokémon TCG games.

Run from the repository root:
  python3 src/error_analysis.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from models import train_l1_logistic

RESULTS_DIR = Path("results")


def main():
    # Load game states and existing game-level splits
    df = pd.read_csv("data/processed/game_states.csv")
    splits = pd.read_csv("data/processed/game_splits.csv")

    df = df.merge(splits, on="game_id", validate="many_to_one")

    # Use the same features as models.py
    exclude = ["game_id", "source_date", "turn", "p1_wins", "split"]
    features = [col for col in df.columns if col not in exclude]

    results = []
    combined_results = []

    for turn in [4, 8, 12]:
        turn_data = df[df["turn"] == turn]
        train = turn_data[turn_data["split"] == "train"]
        val = turn_data[turn_data["split"] == "validation"].copy()

        # Train using training games only
        model = train_l1_logistic(
            train[features],
            train["p1_wins"]
        )

        # Predict on validation games
        val["prediction"] = model.predict(val[features])
        val["correct"] = (
            val["prediction"] == val["p1_wins"]
        ).astype(int)

        # Positive means Player 1 has fewer Prize cards remaining
        val["prize_advantage"] = (
            val["p2_prizes_remaining"] -
            val["p1_prizes_remaining"]
        )

        # Group by whether either player is ahead
        val["prize_state"] = np.select(
            [
                val["prize_advantage"] > 0,
                val["prize_advantage"] < 0
            ],
            [
                "P1 ahead",
                "P2 ahead"
            ],
            default="Tied"
        )

        summary = (
            val.groupby("prize_state")["correct"]
            .agg(["mean", "count"])
            .reset_index()
        )

        for _, row in summary.iterrows():
            results.append({
                "turn": turn,
                "prize_state": row["prize_state"],
                "accuracy": row["mean"],
                "games": row["count"]
            })
            
        # Compare games with a Prize-card leader against tied games
        val["prize_group"] = np.where(
            val["prize_advantage"] == 0,
            "Tied",
            "Either player ahead"
        )
        
        combined_summary = (
            val.groupby("prize_group")["correct"]
            .agg(["mean", "count"])
            .reset_index()
        )
        
        for _, row in combined_summary.iterrows():
            combined_results.append({
                "turn": turn,
                "prize_group": row["prize_group"],
                "accuracy": row["mean"],
                "games": row["count"]
            })

    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(
        RESULTS_DIR / "error_by_prize_state.csv",
        index=False
    )
    
    pd.DataFrame(combined_results).to_csv(
        RESULTS_DIR / "error_by_prize_group.csv",
        index=False
    )
    
    print("\nAccuracy by Prize-card state:")
    print(pd.DataFrame(results).round(3).to_string(index=False))

    print("\nAccuracy: Either player ahead vs. tied:")
    print(pd.DataFrame(combined_results).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
