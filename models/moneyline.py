from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import nflreadpy as nfl
import pandas as pd

FEATURES = [
    "home_off_avg",
    "away_off_avg",
    "home_def_avg",
    "away_def_avg",
    "home_off_season_avg",    # NEW: Season-to-date
    "away_off_season_avg",
    "home_def_season_avg",
    "away_def_season_avg",
    "is_division_game",
]

# Buffer % whether to make the bet
threshold = 0.01

def ml_conversion(moneyline):
    """Convert moneyline to implied probability"""
    if moneyline > 0:
        return 100 / (100 + moneyline)
    elif moneyline < 0:
        return abs(moneyline) / (abs(moneyline) + 100)
    else:
        return 0.5

class Moneyline:
    def __init__(self):
        
        '''self.model = XGBClassifier(
            n_estimators=100,
            eval_metric='logloss'
            max_depth=3,
            learning_rate=0.03,
            eval_metric='logloss',
            random_state=42
        )'''
        
        self.model = LogisticRegression(max_iter=2000)

    def train(self, df):
        self.model.fit(df[FEATURES], df["home_win"])

    def predict(self, df):
        X = df[FEATURES]

        probs = self.model.predict_proba(X)
        home_probs = probs[:, 1]
        away_probs = probs[:, 0]

        # Convert Vegas odds to probabilities
        home_ml_probs = df['home_moneyline'].apply(ml_conversion)
        away_ml_probs = df['away_moneyline'].apply(ml_conversion)

        # Pick winner - convert to Series first
        picks = pd.Series(home_probs > 0.5).map({True: "HOME", False: "AWAY"})

        # Determine if there's betting edge - convert to Series first
        home_ml_bet = pd.Series(threshold > (home_ml_probs.values - home_probs)).map({True: "Bet", False: "No Bet"})
        away_ml_bet = pd.Series(threshold > (away_ml_probs.values - away_probs)).map({True: "Bet", False: "No Bet"})

        return pd.DataFrame({
            "game_id": df["game_id"].values,
            "home_win_prob": home_probs,
            "away_win_prob": away_probs,
            "moneyline_pick": picks.values,
            "home_ml_bet": home_ml_bet.values,
            "away_ml_bet": away_ml_bet.values,
            "home_moneyline": df["home_moneyline"],
            "away_moneyline": df["away_moneyline"],
        })