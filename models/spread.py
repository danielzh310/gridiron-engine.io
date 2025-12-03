from sklearn.linear_model import LinearRegression
import pandas as pd

FEATURES = [
    "home_off_avg",
    "away_off_avg",
    "home_def_avg",
    "away_def_avg",
]


class Spread:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, df):
        self.model.fit(df[FEATURES], df["margin"])

    def predict(self, df):
        X = df[FEATURES]

        preds = self.model.predict(X)
        edges = preds - df["spread_line"].values[: len(preds)]

        picks = (pd.Series(edges) > 0).map(
            {True: "HOME", False: "AWAY"}
        )

        return pd.DataFrame({
            "game_id": df["game_id"].values[: len(preds)],
            "predicted_margin": preds,
            "spread_edge": edges,
            "spread_pick": picks,
        })