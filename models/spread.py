from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd

FEATURES = [
    "home_off_avg",
    "away_off_avg",
    "home_def_avg",
    "away_def_avg",
]


class Spread:
    def __init__(self):
        # tuned lasso on residual target (edge)
        self.model = Lasso(alpha=1.0, max_iter=10000)
        self.scaler = StandardScaler()

    def _build_model_matrix(self, df):
        X = df[FEATURES].copy()

        # aggregated features:
        
        # spread line for residual modeling
        X["spread_line"] = df["spread_line"].values

        # matchup advantages
        X["home_attack_adv"] = X["home_off_avg"] - X["away_def_avg"]
        X["away_attack_adv"] = X["away_off_avg"] - X["home_def_avg"]
        X["net_advantage"] = X["home_attack_adv"] - X["away_attack_adv"]

        # overall pace and defensive context
        X["pace"] = X["home_off_avg"] + X["away_off_avg"]
        X["def_total"] = X["home_def_avg"] + X["away_def_avg"]
        X["def_diff"] = X["away_def_avg"] - X["home_def_avg"]

        # interactions
        X["pace_def_interaction"] = X["pace"] * X["def_total"]
        X["attack_def_interaction"] = X["net_advantage"] * X["def_diff"]

        # home-field scaling
        X["home_boost"] = X["home_off_avg"] * 0.12

        return X

    def train(self, df):
        X = self._build_model_matrix(df)
        # residual target: predict edge directly
        y = df["margin"] - df["spread_line"]
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)

    def predict(self, df):
        X = self._build_model_matrix(df)
        Xs = self.scaler.transform(X)
        # model predicts edge; reconstruct margin
        pred_edge = self.model.predict(Xs)
        spread = df["spread_line"].values[: len(pred_edge)]
        pred_margin = pred_edge + spread
        edges = pred_edge

        picks = (pd.Series(edges) > 0).map(
            {True: "HOME", False: "AWAY"}
        )

        return pd.DataFrame({
            "game_id": df["game_id"].values[: len(pred_edge)],
            "predicted_margin": pred_margin,
            "spread_edge": edges,
            "spread_pick": picks,
        })