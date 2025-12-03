import pandas as pd

class Moneyline:
    def predict(self, df):
        return pd.DataFrame({
            "game_id": df["game_id"],
            "moneyline_pick": ["HOME"] * len(df),
            "home_win_prob": [0.55] * len(df),
            "away_win_prob": [0.45] * len(df)
        })