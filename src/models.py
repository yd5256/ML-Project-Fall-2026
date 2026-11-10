from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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
     coefficent_rows = []
     importance_rows = []
     depth_rows = []
     l1_rows = []
     
     # Turns used for detailed analysis
     selected_turns = [2, 4, 6, 8, 10, 12, 14]
     
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
          
          # Store accuracy results
          accuracy_rows.append({
               "turn": turn,
               "train_games": len(train),
               "validation_games": len(val),
               "prize_baseline": baseline_accuracy,
               "prize_logistic": prize_accuracy,
               "l2_logistic": full_accuracy,
               "l1_logistic": l1_accuracy,
               "decision_tree": tree_accuracy
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
                         "train_accuracy": val_accuracy,
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
     
     # Save detailed experiment results
     pd.DataFrame(coefficent_rows).to_csv(results_dir / "logistic_coefficients.csv", index=False)
     pd.DataFrame(importance_rows).to_csv(results_dir / "tree_importance.csv", index=False)
     pd.DataFrame(depth_rows).to_csv(results_dir / "tree_depth_comparison.csv", index=False)
     pd.DataFrame(l1_rows).to_csv(results_dir / "l1_regularization.csv", index=False)
     
     # Print accuracy summary
     print("\nValidation accuracy by turn:")
     print(accuracy_df.round(3).to_string(index=False))
     print("\nSaved to results/model_accuracy.csv")
          
               
               
if __name__ == "__main__":
    main()
