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
    
     # Load mock data
     df = pd.read_csv("data/mock/mock_game_states.csv")
    
     # Split by game
     games = df["game_id"].unique().to_numpy()
     rng = np.random.default_rng(42)
     rng.shuffle(games)

     n_train = int(0.7 * len(games))
     n_val = int(0.15 * len(games))

     train_games = games[:n_train]
     val_games = games[n_train:n_train + n_val]
     test_games = games[n_train + n_val:]

     train = df[df["game_id"].isin(train_games)]
     val = df[df["game_id"].isin(val_games)]
     test = df[df["game_id"].isin(test_games)]   
      
      
     # Select game states from the chosen turn
     train = train[train["turn"] == turn]
     val = val[val["turn"] == turn]
     test = test[test["turn"] == turn] 
     
     # Separate features and labels
     exclude = ["game_id", "source_date", "turn", "p1_wins"]
     features = [col for col in df.columns if col not in exclude]
     
     X_train = train[features]
     y_train = train["p1_wins"]
     
     X_val = val[features]
     y_val = val["p1_wins"]
     
     # Train Logistic regression
     model = train_logistic(X_train, y_train)
     
     # Make predictions
     predictions = model.predict(X_val)
     
     # Prize-card baseline predictions
     baseline_predictions = prize_baseline(
         val["p1_prizes_remaining"],
         val["p2_prizes_remaining"]
     )
     
     # Debugging statements 
     # print("Training examples:", len(X_train))
     # print("Validation examples:", len(X_val))
     # print("Actual winners:", y_val.to_numpy())
     # print("Logistic predictions:", predictions)
     # print("Baseline predictions:", baseline_predictions)
     # print(val[["p1_prizes_remaining", "p2_prizes_remaining"]])
     # print("Features used:", features)
     
     # Compare validation accuracy
     print("Turn:", turn)
     print("Logistic regression accuracy:", np.mean(predictions == y_val))
     print("Prize baseline accuracy:", np.mean(baseline_predictions == y_val))
    

if __name__ == "__main__":
    main()
