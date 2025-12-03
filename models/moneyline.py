from sklearn.linear_model import LogisticRegression
import pandas as pd

FEATURES = [
    "home_off_avg",
    "away_off_avg",
    "home_def_avg",
    "away_def_avg",
]


class Moneyline:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, df):
        self.model.fit(df[FEATURES], df["home_win"])

    def predict(self, df):
        X = df[FEATURES]

        probs = self.model.predict_proba(X)

        home_probs = probs[:, 1]
        away_probs = probs[:, 0]

        picks = (pd.Series(home_probs) > 0.5).map(
            {True: "HOME", False: "AWAY"}
        )

        return pd.DataFrame({
            "game_id": df["game_id"].values[: len(home_probs)],
            "home_win_prob": home_probs,
            "away_win_prob": away_probs,
            "moneyline_pick": picks,
        })