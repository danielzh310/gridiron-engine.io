"""
Weekly prediction pipeline:

Loads NFL data, builds features, runs Moneyline, Spread, 
and Total models, and exporting weekly picks to CSV.
"""

from data.loaders import load_weekly_data
from data.preprocessing import build_features

from models.moneyline import Moneyline
from models.spread import Spread
from models.total import Total

import pandas as pd


OUTPUT_PATH = "outputs/weekly_picks.csv"


def run_weekly():
    print("Loading weekly data...")
    df = load_weekly_data()

    print("Building features...")
    features = build_features(df)

    print("Running models...")
    ml = Moneyline()
    sp = Spread()
    tot = Total()

    ml_preds = ml.predict(features)
    sp_preds = sp.predict(features)
    tot_preds = tot.predict(features)

    print("Combining output...")
    results = features[["game_id", "home_team", "away_team"]].copy()

    results = results.merge(ml_preds, on="game_id", how="left")
    results = results.merge(sp_preds, on="game_id", how="left")
    results = results.merge(tot_preds, on="game_id", how="left")

    print("Saving output...")
    results.to_csv(OUTPUT_PATH, index=False)

    print("Weekly predictions generated")