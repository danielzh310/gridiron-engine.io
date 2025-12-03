import pandas as pd

# Placeholder Test
class Total:
    def predict(self, df):
        return pd.DataFrame({
            "game_id": df["game_id"],
            "total_pick": ["OVER"] * len(df),
            "total_edge": [4.1] * len(df)
        })