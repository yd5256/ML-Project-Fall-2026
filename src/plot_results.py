
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
FIGURES_DIR = REPO_ROOT / "figures" / "model_results"


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
    

def plot_cross_validation():
    """
    Compare mean cross-validation accuracy across model configurations.
    """
    df = pd.read_csv(RESULTS_DIR / "cross_validation.csv")

    # Calculate mean accuracy and standard deviation across folds
    summary = (
        df.groupby(["turn", "model", "configuration"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Make readable labels for each model configuration
    summary["label"] = summary.apply(
        lambda row: (
            f"NN ({row['configuration'].split('=')[1]} neurons)"
            if row["model"] == "neural_network"
            else {
                "l1_logistic": "L1 Logistic",
                "l2_logistic": "L2 Logistic",
                "decision_tree": "Decision Tree"
            }[row["model"]]
        ),
        axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, group in summary.groupby("label"):
        group = group.sort_values("turn")

        ax.errorbar(
            group["turn"],
            group["mean"],
            yerr=group["std"],
            marker="o",
            capsize=4,
            linewidth=2,
            label=label
        )

    ax.set_xlabel("Game turn")
    ax.set_ylabel("Mean cross-validation accuracy")
    ax.set_title("5-Fold Cross-Validation: Model Comparison")
    ax.set_xticks(sorted(df["turn"].unique()))
    ax.set_ylim(0.45, 0.85)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cross_validation.png", dpi=300)
    plt.close(fig)



def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_accuracy()
    plot_ablation()
    plot_cross_validation()

    print("Saved figures to figures/model_results/")


if __name__ == "__main__":
    main()
