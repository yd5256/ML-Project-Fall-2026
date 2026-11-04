import numpy as np
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
 
 
    
