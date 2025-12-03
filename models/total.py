from sklearn.linear_model import LinearRegression
import pandas as pd

FEATURES = [
    "home_off_avg",
    "away_off_avg",
    "home_def_avg",
    "away_def_avg",
]


class Total:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, df):
        self.model.fit(df[FEATURES], df["total_pts"])

    def predict(self, df):
        X = df[FEATURES]

        preds = self.model.predict(X)
        edges = preds - df["total_line"].values[: len(preds)]

        picks = (pd.Series(edges) > 0).map(
            {True: "OVER", False: "UNDER"}
        )

        return pd.DataFrame({
            "game_id": df["game_id"].values[: len(preds)],
            "predicted_total": preds,
            "total_edge": edges,
            "total_pick": picks,
        })