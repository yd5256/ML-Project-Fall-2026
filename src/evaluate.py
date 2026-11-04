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


# ---------------------------------------------------------------------------
# 1. Game-level split
# ---------------------------------------------------------------------------

def make_game_split(states, ratios=SPLIT_RATIOS, seed=SEED):
    """
    Assign every *game* (not every row) to train, validation, or test.

    All snapshots from one game share a winner and are strongly correlated,
    so splitting by row would leak information about test games into training.
    Returns a DataFrame with columns: game_id, split
    """
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
    """Raise an error if the game-state table breaks the data contract."""
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
    """Raise an error if the split is invalid; otherwise return nothing."""
    # only known split names
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

    # group sizes must match the requested ratios (within one game of rounding)
    n = len(splits)
    counts = splits["split"].value_counts()
    for name in SPLIT_NAMES:
        if abs(counts.get(name, 0) - ratios[name] * n) > 1:
            raise ValueError(f"Split '{name}' has {counts.get(name, 0)} games; expected about {ratios[name] * n:.1f}.")


def summarize_split(states, splits):
    """Per split: number of games, number of rows, and Player 1 win rate."""
    merged = states.merge(splits, on="game_id")
    per_game = merged.groupby("game_id").agg(split=("split", "first"), p1_wins=("p1_wins", "first"))
    summary = pd.DataFrame({
        "games": per_game.groupby("split").size(),
        "rows": merged.groupby("split").size(),
        "p1_win_rate": per_game.groupby("split")["p1_wins"].mean().round(3),
    })
    return summary.reindex(["train", "validation", "test"])


# ---------------------------------------------------------------------------
# 2. Metrics (confusion matrix, accuracy, precision, recall, F1)
# ---------------------------------------------------------------------------

def confusion_counts(y_true, y_pred, positive=1):
    """
    Count TP, FP, FN, TN. By default 'positive' = Player 1 wins (p1_wins == 1).
    """
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
    """
    Accuracy, precision, recall, and F1 with 'positive' as the positive class.

    Also returns F1 with the *other* class as positive and the average of the
    two (macro F1), because P1/P2 are symmetric outcomes and precision/recall
    for one side alone only describe how well we predict that side's wins.
    """
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


def selftest():
    """Check the metric code against Lecture 5's imbalanced example (and sklearn if installed)."""
    # Lecture 5, "An Imbalanced Example": TP=5, TN=90, FP=0, FN=5
    y_true = np.array([1] * 10 + [0] * 90)
    y_pred = np.array([1] * 5 + [0] * 5 + [0] * 90)
    m = classification_metrics(y_true, y_pred)

    _check((m["tp"], m["tn"], m["fp"], m["fn"]) == (5, 90, 0, 5), "(m['tp'], m['tn'], m['fp'], m['fn']) == (5, 90, 0, 5)")
    _check(np.isclose(m["accuracy"], 0.95), "np.isclose(m['accuracy'], 0.95)")
    _check(np.isclose(m["precision"], 1.0), "np.isclose(m['precision'], 1.0)")
    _check(np.isclose(m["recall"], 0.5), "np.isclose(m['recall'], 0.5)")
    _check(np.isclose(m["f1"], 2 / 3), "np.isclose(m['f1'], 2 / 3)")
    print("Lecture 5 example: OK  ", {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})

    # Random predictions: compare against sklearn if it is installed
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except ImportError:
        raise SystemExit("sklearn not installed, so the cross-check cannot run (pip install scikit-learn)")

    # undefined precision must be NaN, not 0
    m = classification_metrics([1, 0], [0, 0])
    _check(np.isnan(m["precision"]) and m["recall"] == 0.0, "undefined precision should be NaN")
    try:
        classification_metrics([1, 0], [0.9, 0.2])
    except ValueError:
        pass
    else:
        raise AssertionError("probabilities should be rejected as y_pred")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_pred = rng.integers(0, 2, 500)
    m = classification_metrics(y_true, y_pred)
    _check(np.isclose(m["accuracy"], accuracy_score(y_true, y_pred)), "np.isclose(m['accuracy'], accuracy_score(y_true, y_pred))")
    _check(np.isclose(m["precision"], precision_score(y_true, y_pred)), "np.isclose(m['precision'], precision_score(y_true, y_pred))")
    _check(np.isclose(m["recall"], recall_score(y_true, y_pred)), "np.isclose(m['recall'], recall_score(y_true, y_pred))")
    _check(np.isclose(m["f1"], f1_score(y_true, y_pred)), "np.isclose(m['f1'], f1_score(y_true, y_pred))")
    _check(np.isclose(m["macro_f1"], f1_score(y_true, y_pred, average="macro")), "np.isclose(m['macro_f1'], f1_score(y_true, y_pred, average='macro'))")
    print("sklearn cross-check: OK")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluation tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_split = sub.add_parser("split", help="create the game-level train/validation/test split")
    p_split.add_argument("--states", default=MOCK_STATES)
    p_split.add_argument("--out", default=MOCK_SPLITS)
    p_split.add_argument("--seed", type=int, default=SEED)

    sub.add_parser("selftest", help="check metric functions")

    args = parser.parse_args(argv)

    if args.command == "split":
        states = pd.read_csv(args.states)
        check_states(states)
        splits = make_game_split(states, seed=args.seed)
        check_split(states, splits)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        splits.to_csv(args.out, index=False)
        print(f"Wrote {args.out}\n")
        print(summarize_split(states, splits).to_string())
    elif args.command == "selftest":
        selftest()


if __name__ == "__main__":
    sys.exit(main())
