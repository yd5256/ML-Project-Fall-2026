
"""
plot_results.py — Visualize model experiments for Pokémon TCG winner prediction.

Run from the repository root:
  python3 src/plot_results.py

Figures are saved in figures/.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"


def plot_accuracy():
    # Load validation accuracy results
    df = pd.read_csv(RESULTS_DIR / "model_accuracy.csv")

    plt.figure(figsize=(10, 6))

    # Compare models across game turns
    models = {
        "prize_baseline": "Prize baseline",
        "l2_logistic": "L2 logistic regression",
        "l1_logistic": "L1 logistic regression",
        "decision_tree": "Decision tree",
        "neural_network": "Neural Network"
    }

    for column, label in models.items():
        plt.plot(df["turn"], df[column], marker="o", label=label)

    plt.xlabel("Game turn")
    plt.ylabel("Validation accuracy")
    plt.title("Winner Prediction Accuracy by Turn")
    plt.xticks(df["turn"])
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "accuracy_by_turn.png", dpi=300)
    plt.close()


def plot_ablation():
    # Load feature ablation results
    df = pd.read_csv(RESULTS_DIR / "feature_ablation.csv")

    # Convert accuracy drops to percentage points
    df["accuracy_drop_pp"] = df["accuracy_drop"] * 100

    # Organize results into a matrix
    matrix = df.pivot(
        index="removed_group",
        columns="turn",
        values="accuracy_drop_pp"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                      vmin=-10, vmax=10)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)

    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    ax.set_xlabel("Game turn")
    ax.set_ylabel("Removed feature group")
    ax.set_title("Feature Ablation: Change in Validation Accuracy")

    fig.colorbar(image, ax=ax, label="Accuracy drop (percentage points)")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_ablation.png", dpi=300)
    plt.close()


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    plot_accuracy()
    plot_ablation()

    print("Saved figures to figures/")


if __name__ == "__main__":
    main()
