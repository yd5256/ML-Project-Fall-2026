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
     turn = 4
    
     # Load mock data and game splits
     df = pd.read_csv("data/mock/mock_game_states.csv")
     splits = pd.read_csv("data/mock/mock_game_splits.csv")
     
     # Assign each game to its train/validation/test split
     df = df.merge(splits, on="game_id")
     
     train = df[df["split"] == "train"]
     val = df[df["split"] == "validation"]
     test = df[df["split"] == "test"]

     # Select game states from the chosen turn
     train = train[train["turn"] == turn]
     val = val[val["turn"] == turn]
     test = test[test["turn"] == turn]
     
     # Separate features and labels
     exclude = ["game_id", "source_date", "turn", "p1_wins", "split"]
     features = [col for col in df.columns if col not in exclude]
     
     X_train = train[features]
     y_train = train["p1_wins"]

     X_val = val[features]
     y_val = val["p1_wins"]

     # Train logistic regression
     model = train_logistic(X_train, y_train)

     # Make predictions
     predictions = model.predict(X_val)

     # Prize-card baseline predictions
     baseline_predictions = prize_baseline(
          val["p1_prizes_remaining"],
          val["p2_prizes_remaining"]
     )

     print("Training examples:", len(train))
     print("Validation examples:", len(val))
     print("Test examples:", len(test))

     # Compare validation accuracy
     print("Turn:", turn)
     print("Logistic regression accuracy:", np.mean(predictions == y_val))
     print("Prize baseline accuracy:", np.mean(baseline_predictions == y_val))
    

if __name__ == "__main__":
    main()
