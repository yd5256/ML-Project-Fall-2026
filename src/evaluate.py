"""
evaluate.py — evaluation code for the Pokémon TCG winner-prediction project.
Owner: Jason

What is in here:
  1. Game-level train/validation/test split (70/15/15) and checks on the data and split
  2. Classification metrics: confusion matrix, accuracy, precision, recall, F1
  3. Data summary (game lengths, how many games reach each turn) and checkpoint choice
  4. Results table: every model scored on the VALIDATION split at each checkpoint turn

Run from the repo root:
  python3 src/evaluate.py                     check data + split, print summaries
  python3 src/evaluate.py results             build and save the results tables
  python3 src/evaluate.py results --turns 4 8 use your own checkpoint turns
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent  # so paths work from any directory
sys.path.insert(0, str(REPO_ROOT / "src"))  # so "import models" works from any directory
import models  # Dominic's models: prize_baseline, train_logistic, train_tree

# Real data (from Yavuz's preprocess.py) and the shared split file (do NOT regenerate it)
STATES_PATH = str(REPO_ROOT / "data/processed/game_states.csv")
SPLITS_PATH = str(REPO_ROOT / "data/processed/game_splits.csv")

OUTPUT_DIR = REPO_ROOT / "outputs"

SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_NAMES = ["train", "validation", "test"]
SEED = 42  # fixed so everyone gets the same split every time

# Checkpoint turns for the results table. None = use choose_checkpoints() suggestion.
# To override, hard-code a list here, e.g. CHECKPOINT_TURNS = [2, 4, 6, 8, 10]
CHECKPOINT_TURNS = None

# Feature lists: copied from models.main() so we use exactly the same features as Dominic
PRIZE_FEATURES = ["p1_prizes_remaining", "p2_prizes_remaining"]
NOT_FEATURES = ["game_id", "source_date", "turn", "p1_wins", "split"]  # ids, label, split name
TREE_DEPTH = 3  # models.main() calls train_tree() with its default depth, which is 3


# game splits
def make_game_split(states, ratios=SPLIT_RATIOS, seed=SEED):
    # assign each game to train validate or test
    # this is how data/processed/game_splits.csv was made (seed 42 gives the same file again).
    # it is not called anywhere: the split file is shared with Dominic and must not be rewritten
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


# data summary and checkpoints
def summarize_data(states):
    # one row per game: its label, who went first, and its last turn (= game length)
    per_game = states.groupby("game_id").agg(
        p1_wins=("p1_wins", "first"),
        p1_went_first=("p1_went_first", "first"),
        length=("turn", "max"),
    )
    n_games = len(per_game)

    # the first player wins if (P1 went first and P1 won) or (P2 went first and P2 won)
    first_player_won = per_game["p1_went_first"] == per_game["p1_wins"]

    summary = {
        "games": n_games,
        "rows": len(states),
        "p1_win_pct": 100 * per_game["p1_wins"].mean(),
        "first_player_win_pct": 100 * first_player_won.mean(),
        "length_min": int(per_game["length"].min()),
        "length_median": float(per_game["length"].median()),
        "length_mean": float(per_game["length"].mean()),
        "length_max": int(per_game["length"].max()),
    }

    # for each turn: how many games are still going (reach that turn)
    all_turns = np.arange(1, summary["length_max"] + 1)
    games_reaching = [(per_game["length"] >= t).sum() for t in all_turns]
    coverage = pd.DataFrame({"turn": all_turns, "games": games_reaching})
    coverage["pct"] = (100 * coverage["games"] / n_games).round(1)
    return summary, coverage


def choose_checkpoints(states, min_fraction=0.8):
    # even turns that at least min_fraction of all games reach
    # (later turns only contain long games, so we stop where too many games have ended)
    lengths = states.groupby("game_id")["turn"].max()
    max_turn = int(lengths.max())
    return [t for t in range(2, max_turn + 1, 2) if (lengths >= t).mean() >= min_fraction]


def suggest_checkpoints(candidates, max_count=5):
    # pick up to max_count turns spread from the earliest to the latest candidate
    if len(candidates) <= max_count:
        return list(candidates)
    picks = np.linspace(0, len(candidates) - 1, max_count).round().astype(int)
    return [candidates[i] for i in sorted(set(picks))]


# models scored on validation
def majority_class(y_train):
    # the label that is more common in the training data (ties go to 1 = P1 wins)
    return 1 if np.mean(y_train) >= 0.5 else 0


def full_features(df):
    # every column except ids, the label and the split name (same rule as models.main())
    return [col for col in df.columns if col not in NOT_FEATURES]


def validation_predictions(train, val):
    # train every model on the training rows and return its predictions on the validation rows
    y_train = train["p1_wins"]
    features = full_features(train)

    prize_lr = models.train_logistic(train[PRIZE_FEATURES], y_train)
    full_lr = models.train_logistic(train[features], y_train)
    tree = models.train_tree(train[features], y_train, max_depth=TREE_DEPTH)

    return {
        "Majority": np.full(len(val), majority_class(y_train)),
        "Prize baseline": models.prize_baseline(val["p1_prizes_remaining"], val["p2_prizes_remaining"]),
        "Prize-only LR": prize_lr.predict(val[PRIZE_FEATURES]),
        "Full LR": full_lr.predict(val[features]),
        f"Tree (depth {TREE_DEPTH})": tree.predict(val[features]),
    }


def build_results_table(states, splits, turns):
    # for each checkpoint turn: train on train games, score on validation games
    # (the test split is never used here - it is saved for the final report)
    merged = states.merge(splits, on="game_id", validate="many_to_one")
    rows = []
    for turn in turns:
        at_turn = merged[merged["turn"] == turn]  # one row per game that reached this turn
        train = at_turn[at_turn["split"] == "train"]
        val = at_turn[at_turn["split"] == "validation"]
        if train["p1_wins"].nunique() < 2 or len(val) == 0:
            print(f"Turn {turn}: not enough data, skipped")
            continue

        for model_name, y_pred in validation_predictions(train, val).items():
            m = classification_metrics(val["p1_wins"], y_pred)
            rows.append({
                "turn": turn,
                "model": model_name,
                "n": m["n"],
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "macro_f1": m["macro_f1"],
            })

    long_table = pd.DataFrame(rows)

    # wide accuracy table: one row per turn, one column per model
    model_order = list(dict.fromkeys(long_table["model"]))  # keep the order models were added
    accuracy = long_table.pivot(index="turn", columns="model", values="accuracy")[model_order]
    games = long_table.groupby("turn")["n"].first().rename("Games")
    accuracy_table = pd.concat([games, accuracy], axis=1)
    return accuracy_table, long_table


def to_markdown(df, digits=3):
    # simple markdown table (avoids needing the extra "tabulate" package)
    def fmt(x):
        if isinstance(x, float):
            return "–" if np.isnan(x) else f"{x:.{digits}f}"
        return str(x)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider] + body)


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

    # F1 with P2 wins as the positive class: swap TP<->TN and FP<->FN
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


# command line
def load_and_check():
    # load the real data and the EXISTING split file, and stop if anything looks wrong
    states = pd.read_csv(STATES_PATH)
    check_states(states)
    splits = pd.read_csv(SPLITS_PATH)
    check_split(states, splits)
    return states, splits


def run_summary():
    states, splits = load_and_check()
    print("Data and split checks passed.\n")
    print("Split summary:")
    print(summarize_split(states, splits), "\n")

    summary, coverage = summarize_data(states)
    print("Data summary:")
    print(f"  games: {summary['games']}, rows: {summary['rows']}")
    print(f"  Player 1 wins: {summary['p1_win_pct']:.1f}%")
    print(f"  Player who went first wins: {summary['first_player_win_pct']:.1f}%")
    print(f"  game length (final turn): min {summary['length_min']}, median {summary['length_median']:.0f}, "
          f"mean {summary['length_mean']:.1f}, max {summary['length_max']}\n")

    # the full turn-by-turn table is long (one very long game), so save it and print the start
    OUTPUT_DIR.mkdir(exist_ok=True)
    coverage.to_csv(OUTPUT_DIR / "turn_coverage.csv", index=False)
    print("Games reaching each turn (first 20 turns; full table in outputs/turn_coverage.csv):")
    print(coverage.head(20).to_string(index=False), "\n")

    candidates = choose_checkpoints(states)
    print(f"Even turns reached by at least 80% of games: {candidates}")
    print(f"Suggested checkpoints: {suggest_checkpoints(candidates)}")
    print("(override with CHECKPOINT_TURNS at the top of this file or --turns)")


def run_results(turns=None):
    states, splits = load_and_check()
    if turns is None:
        turns = CHECKPOINT_TURNS or suggest_checkpoints(choose_checkpoints(states))
    print(f"Checkpoint turns: {turns}  (all scores on the VALIDATION split)\n")

    accuracy_table, long_table = build_results_table(states, splits, turns)

    OUTPUT_DIR.mkdir(exist_ok=True)
    accuracy_table.to_csv(OUTPUT_DIR / "results_accuracy.csv")
    long_table.to_csv(OUTPUT_DIR / "results_full_metrics.csv", index=False)

    accuracy_md = to_markdown(accuracy_table.reset_index().rename(columns={"turn": "Turn"}))
    full_md = to_markdown(long_table)
    with open(OUTPUT_DIR / "results.md", "w") as f:
        f.write("## Validation accuracy by turn\n\n" + accuracy_md + "\n\n")
        f.write("## Validation metrics (positive class = Player 1 wins)\n\n" + full_md + "\n")

    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print("Validation accuracy:")
        print(accuracy_table, "\n")
        print("All validation metrics:")
        print(long_table.to_string(index=False))
    print("\nSaved outputs/results_accuracy.csv, outputs/results_full_metrics.csv, outputs/results.md")


def main():
    parser = argparse.ArgumentParser(description="Evaluation for the Pokémon TCG winner-prediction project.")
    sub = parser.add_subparsers(dest="command")
    results = sub.add_parser("results", help="build and save the validation results tables")
    results.add_argument("--turns", type=int, nargs="+", help="checkpoint turns, e.g. --turns 2 4 6 8 10")
    args = parser.parse_args()

    if args.command == "results":
        run_results(args.turns)
    else:
        run_summary()


if __name__ == "__main__":
    main()
