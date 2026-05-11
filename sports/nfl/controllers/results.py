from pathlib import Path

import pandas as pd

from sports.nfl.data.loaders import load_weekly_data
from utils.betting import payout_profit_per_dollar


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs"


def prediction_snapshot_path(season: int, week: int) -> Path:
    return OUTPUT_DIR / f"nfl_predictions_{int(season)}_wk{int(week)}.csv"


def results_snapshot_path(season: int, week: int) -> Path:
    return OUTPUT_DIR / f"nfl_results_{int(season)}_wk{int(week)}.csv"


def save_prediction_snapshot(df: pd.DataFrame, season: int, week: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = prediction_snapshot_path(season, week)
    df.to_csv(path, index=False)
    return path


def load_prediction_snapshot(season: int, week: int) -> pd.DataFrame:
    path = prediction_snapshot_path(season, week)
    if not path.exists():
        raise FileNotFoundError(f"No saved prediction snapshot found at {path}")
    return pd.read_csv(path)


def _moneyline_result(row):
    if row["home_score"] > row["away_score"]:
        return "HOME"
    if row["away_score"] > row["home_score"]:
        return "AWAY"
    return "PUSH"


def _spread_result(row):
    edge = (row["home_score"] - row["away_score"]) - row["spread_line"]
    if edge > 0:
        return "HOME"
    if edge < 0:
        return "AWAY"
    return "PUSH"


def _total_result(row):
    actual_total = row["home_score"] + row["away_score"]
    if actual_total > row["total_line"]:
        return "OVER"
    if actual_total < row["total_line"]:
        return "UNDER"
    return "PUSH"


def _moneyline_net(row):
    stake = float(row.get("kelly_stake_ml", 0.0) or 0.0)
    if stake <= 0 or row["ml_result"] == "PUSH":
        return 0.0
    if row["ml_pick"] != row["ml_result"]:
        return -stake

    odds_col = "home_moneyline" if row["ml_pick"] == "HOME" else "away_moneyline"
    return stake * payout_profit_per_dollar(row[odds_col])


def grade_saved_predictions(season: int, week: int):
    predictions = load_prediction_snapshot(season, week)
    schedule = load_weekly_data(seasons=[int(season)])
    actuals = schedule[
        (schedule["season"] == int(season)) &
        (schedule["week"] == int(week)) &
        schedule["home_score"].notna() &
        schedule["away_score"].notna()
    ].copy()

    if actuals.empty:
        return None, {
            "message": "No completed games found for that season/week yet.",
            "results_path": None,
        }

    actual_cols = ["game_id", "home_score", "away_score"]
    if "spread_line" not in predictions.columns:
        actual_cols.append("spread_line")
    if "total_line" not in predictions.columns:
        actual_cols.append("total_line")
    graded = predictions.merge(actuals[actual_cols], on="game_id", how="left")
    graded = graded[graded["home_score"].notna() & graded["away_score"].notna()].copy()

    if graded.empty:
        return None, {
            "message": "Saved predictions did not match any completed games.",
            "results_path": None,
        }

    graded["actual_margin"] = graded["home_score"] - graded["away_score"]
    graded["actual_total"] = graded["home_score"] + graded["away_score"]
    graded["ml_result"] = graded.apply(_moneyline_result, axis=1)
    graded["spread_result"] = graded.apply(_spread_result, axis=1)
    graded["total_result"] = graded.apply(_total_result, axis=1)

    graded["ml_hit"] = graded["ml_pick"] == graded["ml_result"]
    graded["spread_hit"] = graded["spread_pick"] == graded["spread_result"]
    graded["total_hit"] = graded["total_pick"] == graded["total_result"]
    graded.loc[graded["total_pick"] == "PASS", "total_hit"] = False

    graded["ml_net"] = graded.apply(_moneyline_net, axis=1).round(2)

    summary = {
        "games_completed": int(len(graded)),
        "ml_hits": int(graded["ml_hit"].sum()),
        "ml_accuracy": float(graded["ml_hit"].mean()),
        "spread_hits": int(graded["spread_hit"].sum()),
        "spread_accuracy": float(graded["spread_hit"].mean()),
        "total_bets": int((graded["total_pick"] != "PASS").sum()),
        "total_hits": int((graded["total_hit"] & (graded["total_pick"] != "PASS")).sum()),
        "ml_net": float(graded["ml_net"].sum().round(2)),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = results_snapshot_path(season, week)
    graded.to_csv(results_path, index=False)
    summary["results_path"] = str(results_path)
    summary["message"] = "Results graded."

    return graded, summary
