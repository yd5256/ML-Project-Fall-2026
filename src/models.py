"""
models.py — Model training and experiments for Pokémon TCG winner prediction.
Owner: Dom

What is in here:
  1. Prize-card baseline
  2. Logistic regression (L2 and L1 regularization)
  3. Decision tree classifier
  4. Per-turn validation accuracy comparisons
  5. Feature coefficient and importance analysis
  6. Decision tree depth and L1 regularization experiments
  7. Feature ablation experiments

Run from the repository root:
  python3 src/models.py

Results are saved in results/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

def prize_baseline(p1_prizes, p2_prizes):
     """
     Predict the winner based on Prize cards remaining.
     Returns 1 if Player 1 is predicted to win, 0 otherwise.
     """
     return np.where(p1_prizes <= p2_prizes, 1, 0)

def train_logistic(X_train, y_train):
     """
     Train a logistic regression model using game-state features. 
     """
     model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
     model.fit(X_train, y_train)
     
     return model

def train_l1_logistic(X_train, y_train, C=1.0):
     """
     Train logistic regression using L1 regularization
     """
     model = make_pipeline(
          StandardScaler(),
          LogisticRegression(
               l1_ratio=1.0,
               solver="liblinear",
               C=C,
               max_iter=1000,
               random_state=42
          )
     )
     model.fit(X_train, y_train)
     
     return model

def train_tree(X_train, y_train, max_depth=3):
     """
     Train a decision tree classifier using game-state features.
     """
     model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
     model.fit(X_train, y_train)
     
     return model


def train_neural_network(X_train, y_train, hidden_units=16):
    """
    Train a neural network with scaling and early stopping.
    """
    tf.keras.utils.set_random_seed(42)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)

    # Separate data for training and early stopping
    X_fit, X_stop, y_fit, y_stop = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    # Fit scaler only on actual training examples
    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit)
    X_stop_scaled = scaler.transform(X_stop)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_fit_scaled,
        y_fit,
        validation_data=(X_stop_scaled, y_stop),
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True
            )
        ],
        verbose=0
    )

    return model, scaler


def cross_validate_models(train, full_features, turn):
     """
     Compare model configurations using 5-fold cross-validation
     """
     X = train[full_features].to_numpy()
     y = train["p1_wins"].to_numpy()
     
     kfold = StratifiedKFold(
          n_splits=5,
          shuffle=True,
          random_state=42
     )
     
     results = []
     
     for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
          X_fold_train = X[train_idx]
          X_fold_val = X[val_idx]
          y_fold_train = y[train_idx]
          y_fold_val = y[val_idx]
          
          # Scale using only the current fold's training data
          scaler = StandardScaler()
          X_train_scaled = scaler.fit_transform(X_fold_train)
          X_val_scaled = scaler.transform(X_fold_val)
          
          # Logistic Regression (L2)
          l2_model = LogisticRegression(max_iter=1000)
          l2_model.fit(X_train_scaled, y_fold_train)
          l2_accuracy = l2_model.score(X_val_scaled, y_fold_val)
          
          results.append({
               "turn": turn,
               "fold": fold,
               "model": "l2_logistic",
               "configuration": "C=1.0",
               "accuracy": l2_accuracy
          })
     
          # Logistic regression (L1)
          l1_model = LogisticRegression(
               l1_ratio=1.0,
               solver="liblinear",
               max_iter=1000,
               random_state=42
          )
          l1_model.fit(X_train_scaled, y_fold_train)
          l1_accuracy = l1_model.score(X_val_scaled, y_fold_val)

          results.append({
               "turn": turn,
               "fold": fold,
               "model": "l1_logistic",
               "configuration": "C=1.0",
               "accuracy": l1_accuracy
          })

          # Decision tree
          tree_model = DecisionTreeClassifier(
               max_depth=3,
               random_state=42
          )
          tree_model.fit(X_fold_train, y_fold_train)
          tree_accuracy = tree_model.score(X_fold_val, y_fold_val)

          results.append({
               "turn": turn,
               "fold": fold,
               "model": "decision_tree",
               "configuration": "depth=3",
               "accuracy": tree_accuracy
          })

          # Neural networks with different hidden layer sizes
          for hidden_units in [8, 16, 32]:
               nn_model, nn_scaler = train_neural_network(
               X_fold_train,
               y_fold_train,
               hidden_units=hidden_units
          )

               X_nn_val = nn_scaler.transform(X_fold_val)

               probabilities = nn_model.predict(X_nn_val, verbose=0)
               predictions = (probabilities.ravel() >= 0.5).astype(int)
               nn_accuracy = np.mean(predictions == y_fold_val)

               results.append({
                    "turn": turn,
                    "fold": fold,
                    "model": "neural_network",
                    "configuration": f"hidden_units={hidden_units}",
                    "accuracy": nn_accuracy
               })

     return results
 
def main():
     # Load processed game states
     df = pd.read_csv("data/processed/game_states.csv")
     splits = pd.read_csv("data/processed/game_splits.csv")
     
     # Assign each game to its train/validation/test split
     df = df.merge(splits, on="game_id", validate="many_to_one")
     
     # Get all available turns
     turns = list(range(1, 16))
     
     # Define feature sets
     prize_features = ["p1_prizes_remaining", "p2_prizes_remaining"]
     
     exclude = ["game_id", "source_date", "turn", "p1_wins", "split"]
     full_features = [col for col in df.columns if col not in exclude]
     
     # Store experiment results
     accuracy_rows = []
     cv_rows = []
     coefficent_rows = []
     importance_rows = []
     depth_rows = []
     l1_rows = []
     
     # Turns used for detailed analysis
     selected_turns = [2, 4, 6, 8, 10, 12, 14]
     
     # Feature groups for ablation experiments
     feature_groups = {
          "prizes": ["p1_prizes_remaining", "p2_prizes_remaining"],
          "hp": ["p1_active_hp", "p2_active_hp",
                 "p1_total_hp", "p2_total_hp"],
          "energy": ["p1_total_energy", "p2_total_energy"],
          "hand_deck": ["p1_hand_size", "p2_hand_size",
                        "p1_deck_size", "p2_deck_size"],
          "pokemon_count": ["p1_pokemon_in_play", "p2_pokemon_in_play"],
          "status": ["p1_active_has_status", "p2_active_has_status"],
          "went_first": ["p1_went_first"]
     }

     ablation_rows = []
     
     # Evaluate each turn
     for turn in turns:
          turn_data = df[df["turn"] == turn]
          
          train = turn_data[turn_data["split"] == "train"]
          val = turn_data[turn_data["split"] == "validation"]
          
          if train["p1_wins"].nunique() < 2 or len(val) == 0:
               print(f"Turn {turn}: Not enough data")
               continue

          # Labels
          y_train = train["p1_wins"]
          y_val = val["p1_wins"]
          
          # Cross-validation on selected turns
          if turn in [4, 8, 12]:
               cv_rows.extend(cross_validate_models(train, full_features, turn))
          
          # Train Prize-only logistic regression
          prize_model = train_logistic(train[prize_features], y_train)
          prize_predictions = prize_model.predict(val[prize_features])
          
          # Train full-feature logistic regression
          full_model = train_logistic(train[full_features], y_train)
          full_predictions = full_model.predict(val[full_features])
          
          # Train L1-regularized logistic regression
          l1_model = train_l1_logistic(train[full_features], y_train)
          l1_predictions = l1_model.predict(val[full_features])
          
          # Train decision tree
          tree_model = train_tree(train[full_features], y_train)
          tree_predictions = tree_model.predict(val[full_features])
          
          # Train neural network (scaling handled inside function)
          nn_model, nn_scaler = train_neural_network(train[full_features],y_train)

          # Scale validation features using the training scaler
          X_val_scaled = nn_scaler.transform(val[full_features].to_numpy())
          
          # Predict winner
          nn_probabilities = nn_model.predict(X_val_scaled, verbose=0)
          nn_predictions = (nn_probabilities.ravel() >= 0.5).astype(int)
 
          # Store logistic regression coefficients
          if turn in selected_turns:
               coefficients = full_model.named_steps["logisticregression"].coef_[0]
               
               for feature, weight in zip(full_features, coefficients):
                    coefficent_rows.append({
                         "turn": turn,
                         "feature": feature,
                         "coefficient": weight
                    })
               
          # Prize-card baseline predictions
          baseline_predictions = prize_baseline(
          val["p1_prizes_remaining"],
          val["p2_prizes_remaining"]
          )
               
          # Compare validation accuracy
          baseline_accuracy = np.mean(baseline_predictions == y_val)
          prize_accuracy = np.mean(prize_predictions == y_val)
          full_accuracy = np.mean(full_predictions == y_val)
          tree_accuracy = np.mean(tree_predictions == y_val)
          l1_accuracy = np.mean(l1_predictions == y_val)
          nn_accuracy = np.mean(nn_predictions == y_val)
          
          # Store accuracy results
          accuracy_rows.append({
               "turn": turn,
               "train_games": len(train),
               "validation_games": len(val),
               "prize_baseline": baseline_accuracy,
               "prize_logistic": prize_accuracy,
               "l2_logistic": full_accuracy,
               "l1_logistic": l1_accuracy,
               "decision_tree": tree_accuracy,
               "neural_network": nn_accuracy
          })
          
          # Feature ablation experiments
          if turn in selected_turns:
               for group, removed_features in feature_groups.items():

                    # Keep all features except the selected group
                    remaining_features = [
                         feature for feature in full_features
                         if feature not in removed_features
                    ]

                    # Retrain without the selected features
                    ablated_model = train_logistic(
                         train[remaining_features], y_train
                    )

                    ablated_accuracy = ablated_model.score(
                         val[remaining_features], y_val
                    )

                    ablation_rows.append({
                         "turn": turn,
                         "removed_group": group,
                         "full_accuracy": full_accuracy,
                         "ablated_accuracy": ablated_accuracy,
                         "accuracy_drop": full_accuracy - ablated_accuracy
                    })
          
          # Compare decision tree depths
          if turn in selected_turns:
               for depth in [2, 3, 4, 5, 6]:
                    depth_model = train_tree(train[full_features], y_train, max_depth=depth)
                    
                    train_accuracy = depth_model.score(train[full_features], y_train)
                    val_accuracy = depth_model.score(val[full_features], y_val)
                    
                    depth_rows.append({
                         "turn": turn,
                         "depth": depth,
                         "train_accuracy": train_accuracy,
                         "validation_accuracy": val_accuracy
                    })
                    
          # Store decision tree feature importance
          if turn in selected_turns:
               for feature, importance in zip(full_features, tree_model.feature_importances_):
                    importance_rows.append({
                         "turn": turn,
                         "feature": feature,
                         "importance": importance
                    })
                    
          # Compare L1 regularization strengths
          if turn in selected_turns:
               for C in [0.01, 0.1, 1.0, 10.0]:
                    model = train_l1_logistic(train[full_features], y_train, C=C)
                    
                    val_accuracy = model.score(val[full_features], y_val)
                    
                    coefficients = model.named_steps["logisticregression"].coef_[0]
                    
                    zero_features = [
                         feature
                         for feature, weight in zip(full_features, coefficients)
                         if weight == 0
                    ]
                    
                    l1_rows.append({
                         "turn": turn,
                         "C": C,
                         "validation_accuracy": val_accuracy,
                         "nonzero_features": np.count_nonzero(coefficients),
                         "zero_features": "; ".join(zero_features)
                    })
                    
     # Create results directory
     results_dir = Path("results")
     results_dir.mkdir(exist_ok=True)
          
     # Save accuracy results
     accuracy_df = pd.DataFrame(accuracy_rows)
     accuracy_df.to_csv(results_dir / "model_accuracy.csv", index=False)
     
     # Save cross-validation results
     pd.DataFrame(cv_rows).to_csv(results_dir / "cross_validation.csv", index=False)
     
     # Save detailed experiment results
     pd.DataFrame(coefficent_rows).to_csv(results_dir / "logistic_coefficients.csv", index=False)
     pd.DataFrame(importance_rows).to_csv(results_dir / "tree_importance.csv", index=False)
     pd.DataFrame(depth_rows).to_csv(results_dir / "tree_depth_comparison.csv", index=False)
     pd.DataFrame(l1_rows).to_csv(results_dir / "l1_regularization.csv", index=False)
     pd.DataFrame(ablation_rows).to_csv(results_dir / "feature_ablation.csv", index=False)
     
     # Print accuracy summary
     print("\nValidation accuracy by turn:")
     print(accuracy_df.round(3).to_string(index=False))
     print("\nSaved to results/model_accuracy.csv")
          
          
if __name__ == "__main__":
    main()
