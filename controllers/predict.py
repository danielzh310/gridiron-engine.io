from data.loaders import load_weekly_data
from data.preprocessing import build_features

from models.moneyline import Moneyline
from models.spread import Spread
from models.total import Total


OUTPUT_PATH = "outputs/weekly_picks.csv"


def run_weekly(season=2025, week=13):
    print("Loading NFL data...")
    df = load_weekly_data()

    print("Building rolling features...")
    feats = build_features(df)

    if season is not None and week is not None:
        train_df = feats[
            (feats["season"] < season) |
            ((feats["season"] == season) & (feats["week"] < week))
        ].copy()

        predict_df = feats[
            (feats["season"] == season) &
            (feats["week"] == week)
        ].copy()
    else:
        train_df = feats[feats["home_score"].notna()].copy()
        predict_df = feats[feats["home_score"].isna()].copy()

    print(f"Training on {len(train_df)} completed games")

    if predict_df.empty:
        print("You suck at coding")
        return

    print(f"Predicting {len(predict_df)} upcoming games")

    ml = Moneyline()
    sp = Spread()
    tot = Total()

    print("Training models...")
    ml.train(train_df)
    sp.train(train_df)
    tot.train(train_df)

    print("Generating predictions...")
    ml_preds = ml.predict(predict_df)
    sp_preds = sp.predict(predict_df)
    tot_preds = tot.predict(predict_df)

    print("Merging outputs...")
    base = predict_df[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")

    results = base.merge(ml_preds, on="game_id")
    results = results.merge(sp_preds, on="game_id")
    results = results.merge(tot_preds, on="game_id")

    print("Cleaning and formatting CSV output...")

    pretty = results.rename(columns={
        "moneyline_pick": "ml_pick",
        "home_win_prob": "ml_home_prob",
        "predicted_margin": "model_spread_margin",
        "predicted_total": "projected_total",
        "total_edge": "vegas_total_edge",
    })

    round_cols = [
        "ml_home_prob",
        "model_spread_margin",
        "spread_edge",
        "projected_total",
        "vegas_total_edge",
        "total_low_q",
        "total_high_q",
        "total_buffer",
    ]

    pretty[round_cols] = pretty[round_cols].round(2)

    ordered_cols = [
        "game_id",
        "home_team",
        "away_team",

        "ml_pick",
        "ml_home_prob",
        "home_ml_bet",
        "away_ml_bet",

        "spread_pick",
        "model_spread_margin",
        "spread_edge",

        "total_pick",
        "projected_total",
        "vegas_total_edge",

        "total_low_q",
        "total_high_q",
        "total_buffer",
    ]

    pretty[ordered_cols].to_csv(OUTPUT_PATH, index=False)

    print("Weekly predictions generated.")