import numpy as np

def prize_baseline(p1_prizes, p2_prizes):
     """
     Predict the winner based on Prize cards remaining.
     Returns 1 if Player 1 wins, 0 if Player 2 wins.
     """
     return np.where(p1_prizes <= p2_prizes, 1, 0)
