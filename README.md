# Pokémon TCG Winner Prediction

**CSCI 3345: Machine Learning | Boston College | Fall 2026**

**Team:** Yavuz Damkaci, Dominic Chen, Jason Jung

## Overview

This project explores whether machine learning models can predict the winner of a Pokémon Trading Card Game (TCG) match based on the current game state.

Our main research question is:

**How early in a Pokémon TCG game can machine learning models accurately predict the eventual winner?**

We approach this as a binary classification problem, using information available at different stages of a match to predict whether Player 1 or Player 2 will win.

## Dataset

We use publicly available [Pokémon TCG AI Battle episode datasets](https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-index) from Kaggle, which contain game replays and final match results.

We plan to preprocess approximately 5,000 games into structured game-state snapshots with features including:

- Prize cards remaining
- Hand and deck sizes
- Number of Pokémon in play
- Pokémon HP and Energy
- Status conditions
- Turn order
- Differences in resources between players

## Methods

We plan to compare several approaches:

1. **Prize Card Baseline:** Predict the player currently ahead in Prize cards.
2. **Logistic Regression:** Our primary machine learning baseline.
3. **Decision Tree:** Capture nonlinear relationships between game-state features.
4. **Neural Network (optional):** Explore whether a small multilayer network improves prediction accuracy.

We will also compare models trained on basic Prize card information against models using the full game state.

## Evaluation

We will split the dataset by complete games into:

- **70%** training
- **15%** validation
- **15%** testing

Models will be evaluated using accuracy, precision, recall, and F1 score.

Our main experiment will measure how prediction accuracy changes across early, middle, and late stages of a match.

## Project Status

Currently in development. Initial work focuses on data preprocessing, feature extraction, and baseline model implementation.

## References

- [Pokémon TCG AI Battle Episodes — Kaggle](https://www.kaggle.com/datasets/kaggle/pokemon-tcg-ai-battle-episodes-index)
- [PTCGABC Engine Documentation](https://matsuoinstitute.github.io/cabt/)
