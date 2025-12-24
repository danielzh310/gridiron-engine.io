from pathlib import Path

from data.loaders import load_weekly_data
from data.preprocessing import build_features

from models.moneyline import Moneyline
from models.spread import Spread
from models.total import Total


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "outputs" / "weekly_picks.csv"


def run_weekly(
    season=2025,
    week=16,
    export=True,
    output_path=None,
    verbose=True,
):
    if verbose:
        print("Loading NFL data...")
    df = load_weekly_data()

    if verbose:
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

    if verbose:
        print(f"Training on {len(train_df)} completed games")

    if predict_df.empty:
        print("You suck at coding")
        return None

    if verbose:
        print(f"Predicting {len(predict_df)} upcoming games")

    ml = Moneyline()
    sp = Spread()
    tot = Total()

    if verbose:
        print("Training models...")
    ml.train(train_df)
    sp.train(train_df)
    tot.train(train_df)

    if verbose:
        print("Generating predictions...")
    ml_preds = ml.predict(predict_df)
    sp_preds = sp.predict(predict_df)
    tot_preds = tot.predict(predict_df)

    if verbose:
        print("Merging outputs...")
    base = predict_df[["game_id", "home_team", "away_team"]].drop_duplicates("game_id")

    results = base.merge(ml_preds, on="game_id")
    results = results.merge(sp_preds, on="game_id")
    results = results.merge(tot_preds, on="game_id")

    if verbose:
        print("Cleaning and formatting output...")

    pretty = results.rename(columns={
        "moneyline_pick": "ml_pick",
        "home_win_prob": "ml_home_prob",
        "away_win_prob": "ml_away_prob",
        "predicted_margin": "model_spread_margin",
        "predicted_total": "projected_total",
        "total_edge": "vegas_total_edge",
    })

    round_cols = [
        "ml_home_prob",
        "ml_away_prob",
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
        "ml_away_prob",
        "home_ml_bet",
        "away_ml_bet",
        "home_moneyline",
        "away_moneyline",

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

    out_df = pretty[ordered_cols]

    if export:
        path = Path(output_path) if output_path else DEFAULT_OUTPUT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(path, index=False)

        if verbose:
            print(f"Wrote predictions to {path}")

    if verbose:
        print("Weekly predictions generated.")

    return out_df