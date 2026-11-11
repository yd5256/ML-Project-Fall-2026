# Pokémon TCG Winner Prediction

**CSCI 3345: Machine Learning | Boston College | Fall 2026**

**Team:** Yavuz Damkaci, Dominic Chen, Jason Jung

## Overview

This project explores whether machine learning models can predict the winner of a Pokémon Trading Card Game (TCG) match from the current state of the game.

Our main research question is:

**How early in a Pokémon TCG game can machine learning models accurately predict the eventual winner?**

We treat this as a binary classification problem. At different turns in a match, the models use information about the current game state to predict whether Player 1 or Player 2 will eventually win.

## Dataset

We use publicly available [Pokémon TCG AI Battle episode datasets](https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-index) from Kaggle. These datasets contain full AI-vs-AI game replays and final match results.

For the current experiments, we sampled **1,000 games** evenly across five daily datasets from **August 18–22, 2026**. One drawn game was excluded because the project uses a binary winner label, leaving:

- **999 usable games**
- **13,800 turn-level game-state snapshots**
- **493 Player 1 wins and 506 Player 2 wins**

Each replay is converted into one snapshot for each game turn. Extracted features include:

- Prize cards remaining
- Hand size
- Deck size
- Number of Pokémon in play
- Active Pokémon HP
- Total Pokémon HP
- Attached Energy
- Active Pokémon status conditions
- Which player went first

The target variable, `p1_wins`, is 1 when Player 1 eventually wins and 0 when Player 2 wins.

The raw replay files are not included in the repository because of their size. The processed dataset used for the current experiments is included in `data/processed/`.

## Data Split and Evaluation

Games are split by complete match rather than by individual turn snapshots so that snapshots from the same game never appear in multiple splits.

The current split is:

- **70% training**
- **15% validation**
- **15% testing**

The test set is kept untouched during model and feature selection.

Our primary metric is **accuracy**. We also calculate precision, recall, F1 score, and macro-F1.

For the main checkpoint analysis, we evaluate models at turns **2, 4, 6, 8, and 10**. These are the even-numbered turns reached by at least 80% of games, which allows us to compare early and middle stages of a match without relying on very small late-game samples.

## Models and Experiments

The project currently includes:

1. **Majority baseline**  
   Predicts the most common winner in the training data.

2. **Prize-card baseline**  
   Predicts that the player with fewer Prize cards remaining will eventually win.

3. **Prize-only logistic regression**  
   Uses only the two players' remaining Prize-card counts.

4. **Full logistic regression**  
   Uses the full set of extracted game-state features.

5. **L1-regularized logistic regression**  
   Tests a sparse version of logistic regression and provides another way to examine feature importance.

6. **Decision tree classifier**  
   Tests whether nonlinear relationships between game-state features improve prediction.

Additional experiments include:

- Prediction accuracy across game turns
- Logistic-regression coefficient analysis
- Decision-tree feature importance
- Decision-tree depth comparisons
- L1 regularization comparisons
- Feature-group ablation experiments

## Preliminary Results

The table below shows validation accuracy for several of the main models at the checkpoint turns.

| Turn | Validation Games | Prize Baseline | Full Logistic Regression | Decision Tree |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 150 | 50.7% | 44.7% | 49.3% |
| 4 | 149 | 51.0% | 55.0% | 51.7% |
| 6 | 148 | 56.8% | **65.5%** | 55.4% |
| 8 | 144 | 55.6% | **69.4%** | 64.6% |
| 10 | 119 | 57.1% | **68.9%** | 63.9% |

The early turns are difficult to predict from the available game-state information. Performance begins to improve more clearly around turns 6–8. At turn 8, the full logistic regression model reaches **69.4% validation accuracy**, compared with **55.6%** for the Prize-card baseline.

Extended experiments also show an L1 logistic-regression accuracy of **70.1% at turn 8**.

These results are preliminary and are based only on the validation split. The test set has not yet been evaluated.

## Repository Structure

```text
ML-Project-Fall-2026/
├── data/
│   ├── mock/                # Synthetic data used during early development
│   ├── processed/           # Processed game states and shared game split
│   └── raw/                 # Raw Kaggle replays (ignored by Git)
│
├── figures/                 # Generated experiment figures
├── outputs/                 # Main evaluation tables and summaries
├── results/                 # Extended model experiment results
│
├── src/
│   ├── preprocess.py        # Replay preprocessing and feature extraction
│   ├── models.py            # Model training and model experiments
│   ├── evaluate.py          # Splitting, metrics, and checkpoint evaluation
│   └── plot_results.py      # Result visualizations
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Install the required Python packages from the repository root:

```bash
python3 -m pip install -r requirements.txt
```

## Reproducing the Current Results

The processed dataset and fixed game-level split are already included in the repository, so downloading the raw replay files is not necessary to reproduce the current model results.

From the repository root, run:

```bash
python3 src/evaluate.py
```

This checks the processed dataset and game-level split and prints summary information.

To generate the main validation result tables:

```bash
python3 src/evaluate.py results
```

The main evaluation outputs are saved in:

```text
outputs/
```

To run the extended model experiments:

```bash
python3 src/models.py
```

These results are saved in:

```text
results/
```

To regenerate the experiment figures:

```bash
python3 src/plot_results.py
```

Figures are saved in:

```text
figures/
```

### Re-running Preprocessing

The raw Kaggle replay files are not stored in GitHub. If the raw files are available locally under `data/raw/`, preprocessing can be rerun with:

```bash
python3 src/preprocess.py
```

This generates:

```text
data/processed/game_states.csv
```

## Team Responsibilities

- **Yavuz Damkaci:** data collection, replay preprocessing, and game-state feature extraction
- **Dominic Chen:** model training, model comparisons, feature importance, and ablation experiments
- **Jason Jung:** game-level splitting, evaluation metrics, checkpoint selection, and result reporting

All team members contribute to testing, interpretation of results, and preparation of the final report and presentation.

## Current Status

As of the November 10 baseline checkpoint:

- Data collection and preprocessing are complete for the current 1,000-game sample.
- The game-level train/validation/test split is fixed.
- The baseline and comparison models are working.
- Initial validation experiments and feature comparisons have been completed.
- The test set remains untouched.
- Future work will include additional model analysis, error analysis, possible dataset expansion, and final evaluation on the held-out test set.

## References

- [Pokémon TCG AI Battle Episodes — Kaggle](https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-index)
- [PTCGABC Engine Documentation](https://matsuoinstitute.github.io/cabt/)