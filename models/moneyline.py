from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import nflreadpy as nfl
import pandas as pd

FEATURES = [
    "home_off_avg",
    "away_off_avg",
    "home_def_avg",
    "away_def_avg",
    "is_division_game",
]

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
        
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            eval_metric='logloss',
            random_state=42
        )
        
        #self.model = LogisticRegression(max_iter=2000)
        teams_df = nfl.load_teams().to_pandas()
        self.team_divisions = teams_df.set_index('team_abbr')['team_division'].to_dict()
        
    def add_division_features(self, df):
        """Add division and conference game indicators"""
        data = df.copy()
        
        # Add division/conference for each team
        data['home_division'] = data['home_team'].map(self.team_divisions)
        data['away_division'] = data['away_team'].map(self.team_divisions)
        data['is_division_game'] = (data['home_division'] == data['away_division']).astype(int)
          
        return data

    def train(self, df):
        df_with_features = self.add_division_features(df)
        self.model.fit(df_with_features[FEATURES], df["home_win"])

    def predict(self, df):
        df_with_features = self.add_division_features(df)
        X = df_with_features[FEATURES]

        probs = self.model.predict_proba(X)
        home_probs = probs[:, 1]
        away_probs = probs[:, 0]

        # Convert Vegas odds to probabilities
        home_ml_probs = df['home_moneyline'].apply(ml_conversion)
        away_ml_probs = df['away_moneyline'].apply(ml_conversion)

        # Pick winner - convert to Series first
        picks = pd.Series(home_probs > 0.5).map({True: "HOME", False: "AWAY"})

        # Determine if there's betting edge - convert to Series first
        home_ml_bet = pd.Series(home_probs > (home_ml_probs.values + 0.01)).map({True: "Bet", False: "No Bet"})
        away_ml_bet = pd.Series(away_probs > (away_ml_probs.values + 0.01)).map({True: "Bet", False: "No Bet"})

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