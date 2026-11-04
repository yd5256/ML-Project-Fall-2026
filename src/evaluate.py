"""
evaluate.py — evaluation code for the Pokémon TCG winner-prediction project.
Owner: Jason

Currently implemented:
  1. Game-level train/validation/test split (70/15/15), saved as game_id,split
  2. Classification metrics: confusion matrix, accuracy, precision, recall, F1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent  # so paths work from any directory
MOCK_STATES = str(REPO_ROOT / "data/mock/mock_game_states.csv")
MOCK_SPLITS = str(REPO_ROOT / "data/mock/mock_game_splits.csv")

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_NAMES = ["train", "validation", "test"]
SEED = 42  # fixed so everyone gets the same split every time

# game splits
def make_game_split(states, ratios=SPLIT_RATIOS, seed=SEED):
    # assign each game to train validate or test
    game_ids = np.sort(states["game_id"].unique())  # sort first so the seed is reproducible
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(game_ids)

    n = len(shuffled)
    n_train = int(round(ratios["train"] * n))
    n_val = int(round(ratios["validation"] * n))
    n_test = n - n_train - n_val

    labels = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
    splits = pd.DataFrame({"game_id": shuffled, "split": labels})
    return splits.sort_values("game_id").reset_index(drop=True)


def check_states(states):
    # error checking incoming data for usability
    required = {"game_id", "turn", "p1_wins", "p1_prizes_remaining", "p2_prizes_remaining"}
    missing = required - set(states.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if states.isna().any().any():
        raise ValueError("Game states contain NaN values.")
    if states.duplicated(["game_id", "turn"]).any():
        raise ValueError("Duplicate (game_id, turn) rows.")
    if (states.groupby("game_id")["p1_wins"].nunique() > 1).any():
        raise ValueError("p1_wins changes within a game; the label must be constant per game.")
    if not states["p1_wins"].isin([0, 1]).all():
        raise ValueError("p1_wins must be 0 or 1.")
    for col in ("p1_prizes_remaining", "p2_prizes_remaining"):
        if not states[col].between(0, 6).all():
            raise ValueError(f"{col} must be between 0 and 6.")
        ordered = states.sort_values(["game_id", "turn"])
        if (ordered.groupby("game_id")[col].diff() > 0).any():
            raise ValueError(f"{col} increases within a game.")
    turns = states.groupby("game_id")["turn"].agg(["min", "max", "count"])
    if ((turns["min"] != 1) | (turns["max"] != turns["count"])).any():
        raise ValueError("Turns must be contiguous and start at 1 within each game.")


def check_split(states, splits, ratios=SPLIT_RATIOS):
    # error checking invalid splits
    bad = set(splits["split"]) - set(SPLIT_NAMES)
    if bad:
        raise ValueError(f"Unknown split names: {sorted(bad)}")

    # every game appears exactly once in the split file
    if splits["game_id"].duplicated().any():
        raise ValueError("A game_id appears more than once in the split file.")

    # every game in the data has a split, and the split file has no extra games
    data_games = set(states["game_id"])
    split_games = set(splits["game_id"])
    if data_games != split_games:
        missing = data_games - split_games
        extra = split_games - data_games
        raise ValueError(f"Split/data mismatch. Missing: {sorted(missing)[:5]} Extra: {sorted(extra)[:5]}")

    # after joining, every row of a game must land in a single split
    merged = states.merge(splits, on="game_id")
    if (merged.groupby("game_id")["split"].nunique() > 1).any():
        raise ValueError("Rows from the same game ended up in different splits.")

    # group sizes match 70/15/15
    n = len(splits)
    counts = splits["split"].value_counts()
    for name in SPLIT_NAMES:
        if abs(counts.get(name, 0) - ratios[name] * n) > 1:
            raise ValueError(f"Split '{name}' has {counts.get(name, 0)} games; expected about {ratios[name] * n:.1f}.")


def summarize_split(states, splits):
    # per split - number of games, number of rows, and Player 1 win rate
    merged = states.merge(splits, on="game_id")
    per_game = merged.groupby("game_id").agg(split=("split", "first"), p1_wins=("p1_wins", "first"))
    summary = pd.DataFrame({
        "games": per_game.groupby("split").size(),
        "rows": merged.groupby("split").size(),
        "p1_win_rate": per_game.groupby("split")["p1_wins"].mean().round(3),
    })
    return summary.reindex(["train", "validation", "test"])

# metrics
def confusion_counts(y_true, y_pred, positive=1):
    # counts of TP, FP, FN, TN, given a positive = p1 win
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true has shape {y_true.shape} but y_pred has shape {y_pred.shape}")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.isin(arr, [0, 1]).all():
            raise ValueError(f"{name} must contain only 0/1 labels (threshold probabilities first).")

    actual_pos = y_true == positive
    pred_pos = y_pred == positive
    return {
        "tp": int(np.sum(actual_pos & pred_pos)),
        "fp": int(np.sum(~actual_pos & pred_pos)),
        "fn": int(np.sum(actual_pos & ~pred_pos)),
        "tn": int(np.sum(~actual_pos & ~pred_pos)),
    }


def _safe_divide(numerator, denominator):
    # precision is undefined if the model never predicts positive - report NaN
    # so that undefined != always wrong
    return numerator / denominator if denominator > 0 else float("nan")


def classification_metrics(y_true, y_pred, positive=1):
    # accuracy, precision, recall, and F1
    c = confusion_counts(y_true, y_pred, positive)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]

    accuracy = _safe_divide(tp + tn, tp + tn + fp + fn)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)  # equals 2PR/(P+R); 0 (not NaN) when P=R=0

    # Swapping the positive class: TP<->TN and FP<->FN
    precision_other = _safe_divide(tn, tn + fn)
    recall_other = _safe_divide(tn, tn + fp)
    f1_other = _safe_divide(2 * tn, 2 * tn + fn + fp)

    return {
        **c,
        "n": tp + fp + fn + tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_other_class": f1_other,
        "macro_f1": (f1 + f1_other) / 2,  # NaN if either F1 is undefined
    }


def _check(condition, message):
    # explicit raise instead of assert, so it still runs under python -O
    if not condition:
        raise AssertionError(message)


# run: python src/evaluate.py  (creates the game-level split)
# for the real data, change MOCK_STATES / MOCK_SPLITS at the top to the data/processed/ paths
if __name__ == "__main__":
    states = pd.read_csv(MOCK_STATES)
    check_states(states)
    splits = make_game_split(states)
    check_split(states, splits)
    splits.to_csv(MOCK_SPLITS, index=False)
    print(f"Wrote {MOCK_SPLITS}\n")
    print(summarize_split(states, splits))
