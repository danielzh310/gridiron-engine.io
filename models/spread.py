import pandas as pd

# Placeholder Test
class Spread:
    def predict(self, df):
        return pd.DataFrame({
            "game_id": df["game_id"],
            "spread_pick": ["HOME"] * len(df),
            "spread_edge": [3.2] * len(df)
        })