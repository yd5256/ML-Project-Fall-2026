import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
 
def main():
     turns = [2, 4, 6, 8]
    
     # Load mock data and game splits
     df = pd.read_csv("data/mock/mock_game_states.csv")
     splits = pd.read_csv("data/mock/mock_game_splits.csv")
     
     # Assign each game to its train/validation/test split
     df = df.merge(splits, on="game_id", validate="many_to_one")
     
     # Define feature sets
     prize_features = ["p1_prizes_remaining", "p2_prizes_remaining"]
     
     exclude = ["game_id", "source_date", "turn", "p1_wins", "split"]
     full_features = [col for col in df.columns if col not in exclude]
     
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
          
          # Prize-card baseline predictions
          baseline_predictions = prize_baseline(
          val["p1_prizes_remaining"],
          val["p2_prizes_remaining"]
          )
               
          # Compare validation accuracy
          baseline_accuracy = np.mean(baseline_predictions == y_val)
          prize_accuracy = np.mean(prize_predictions == y_val)
          full_accuracy = np.mean(full_predictions == y_val)
          
          print(f"Turn: {turn}")
          print(f"Validation examples: {len(val)}")
          print(f"Prize baseline accuracy: {baseline_accuracy:.3f}")
          print(f"Prize-only logistic accuracy: {prize_accuracy:.3f}")
          print(f"Full-feature logistic accuracy: {full_accuracy:.3f}")
          print()
          

if __name__ == "__main__":
    main()
