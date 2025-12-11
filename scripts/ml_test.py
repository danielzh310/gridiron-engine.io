"""
MONEYLINE MODEL TEST HARNESS

Loads weekly data, builds rolling features, creates a train/test
split, evaluates multiple models (logistic, random forest, tuned XGBoost),
and computes betting-weighted profit metrics.

Run as: `python scripts/ml_test.py --models logistic,xgboost_tuned --threshold 0.02 --save-bets`
"""

import os
import sys
import argparse

# Ensure the repository root is on sys.path so `data` imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.loaders import load_weekly_data
from data.preprocessing import build_features

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
import nflreadpy as nfl
import pandas as pd
import numpy as np


# ---- Feature set ----
ML_FEATURES = [
    "home_off_avg",
    "away_off_avg",
    "home_def_avg",
    "away_def_avg",
    "home_off_season_avg",    # NEW: Season-to-date
    "away_off_season_avg",
    "home_def_season_avg",
    "away_def_season_avg",
    "is_division_game",
]

# ---- Configuration ----
MODELS_TO_RUN = ["logistic", "random_forest", "xgboost_tuned"]
BET_THRESHOLD = 0.01  # Minimum edge to place bet
SAVE_BETS = False


def build_ml_X(df):
    return df[ML_FEATURES].copy()


def ml_conversion(moneyline):
    """Convert vegas moneyline to implied probability."""
    try:
        ml = float(moneyline)
    except Exception:
        return 0.5
    if ml > 0:
        return 100.0 / (100.0 + ml)
    elif ml < 0:
        return abs(ml) / (abs(ml) + 100.0)
    else:
        return 0.5


def rolling_splits(n, n_folds=5):
    """Simple rolling-origin splits for time-ordered data."""
    if n < (n_folds + 1):
        yield slice(0, int(0.8 * n)), slice(int(0.8 * n), n)
        return
    step = n // (n_folds + 1)
    for k in range(1, n_folds + 1):
        train_end = step * k
        valid_end = step * (k + 1) if k < n_folds else n
        yield slice(0, train_end), slice(train_end, valid_end)


def tune_xgb_classifier(train_df, n_folds=4):
    X_all = build_ml_X(train_df)
    y_all = train_df["home_win"].values

    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.1, 0.03, 0.05],
        "n_estimators": [100, 200],
    }

    best_score = float("inf")
    best_params = None

    for md in param_grid["max_depth"]:
        for lr in param_grid["learning_rate"]:
            for ne in param_grid["n_estimators"]:
                cfg = dict(
                    max_depth=md,
                    learning_rate=lr,
                    n_estimators=ne,
                    eval_metric="logloss",
                    random_state=42,
                    subsample=0.9,
                    colsample_bytree=0.9,
                )

                fold_scores = []
                for tr, va in rolling_splits(len(X_all), n_folds=n_folds):
                    model = XGBClassifier(**cfg)
                    model.fit(X_all.iloc[tr], y_all[tr])
                    pred_proba = model.predict_proba(X_all.iloc[va])[:, 1]
                    score = log_loss(y_all[va], pred_proba)
                    fold_scores.append(score)

                mean_score = float(np.mean(fold_scores))
                if mean_score < best_score:
                    best_score = mean_score
                    best_params = cfg

    return best_params, best_score


def eval_ml_metrics(y_true, prob_home):
    prob_home = np.asarray(prob_home)
    pred = (prob_home > 0.5).astype(int)

    return {
        "logloss": float(log_loss(y_true, prob_home)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob_home)),
    }


def payout_profit_per_dollar(moneyline):
    """Return net profit per $1 stake when bet wins (not including stake)."""
    try:
        ml = float(moneyline)
    except Exception:
        return None
    if ml > 0:
        return ml / 100.0
    else:
        return 100.0 / abs(ml)


def simulate_bets(home_probs, test_df, threshold=0.0):
    """Simulate $1 bets where model has edge over implied odds.

    threshold: minimum edge (model_prob - implied_prob) to place bet.
    Returns dict with metrics and bets dataframe.
    """
    bets = []
    for i, row in test_df.reset_index(drop=True).iterrows():
        model_home = float(home_probs[i])
        implied_home = float(ml_conversion(row.get("home_moneyline", 0)))
        implied_away = float(ml_conversion(row.get("away_moneyline", 0)))

        edge_home = model_home - implied_home
        edge_away = (1 - model_home) - implied_away

        bet_side = None
        if edge_home > threshold and edge_home >= edge_away:
            bet_side = "home"
        elif edge_away > threshold and edge_away > edge_home:
            bet_side = "away"

        if bet_side is None:
            continue

        if bet_side == "home":
            ml = row.get("home_moneyline")
            win = 1 if row.get("home_win") == 1 else 0
            edge = edge_home
        else:
            ml = row.get("away_moneyline")
            win = 1 if row.get("home_win") == 0 else 0
            edge = edge_away

        p_win_profit = payout_profit_per_dollar(ml)
        if p_win_profit is None:
            continue

        profit = p_win_profit if win else -1.0
        bets.append({
            "idx": i,
            "side": bet_side,
            "ml": ml,
            "win": win,
            "profit": profit,
            "edge": edge,
        })

    bets_df = pd.DataFrame(bets)
    if bets_df.empty:
        return {
            "n_bets": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "avg_profit_per_bet": 0.0,
            "profit_weighted_accuracy": 0.0,
            "bets_df": bets_df,
        }

    total_profit = bets_df["profit"].sum()
    n_bets = len(bets_df)
    win_rate = bets_df["win"].mean()
    avg_profit_per_bet = bets_df["profit"].mean()

    positive_profit = bets_df.loc[bets_df["profit"] > 0, "profit"].sum()
    negative_loss = -bets_df.loc[bets_df["profit"] < 0, "profit"].sum()
    profit_weighted_accuracy = (
        positive_profit / (positive_profit + negative_loss)
        if (positive_profit + negative_loss) > 0
        else 0.0
    )

    return {
        "n_bets": n_bets,
        "total_profit": float(total_profit),
        "win_rate": float(win_rate),
        "avg_profit_per_bet": float(avg_profit_per_bet),
        "profit_weighted_accuracy": float(profit_weighted_accuracy),
        "bets_df": bets_df,
    }


def evaluate_model(name, model, X_tr, y_tr, X_te, y_te, test_df, threshold=0.0):
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te)[:, 1]
    ml_metrics = eval_ml_metrics(y_te, probs)
    bet_stats = simulate_bets(probs, test_df, threshold=threshold)

    result = {
        "name": name,
        **ml_metrics,
        "n_bets": bet_stats["n_bets"],
        "total_profit": bet_stats["total_profit"],
        "bet_win_rate": bet_stats["win_rate"],
        "avg_profit_per_bet": bet_stats["avg_profit_per_bet"],
        "profit_weighted_accuracy": bet_stats["profit_weighted_accuracy"],
        "bets_df": bet_stats["bets_df"],
    }
    return result


def main():
    print("\n==== MONEYLINE CLASSIFIER TEST ===\n")
    print(f"Models to run: {', '.join(MODELS_TO_RUN)}")
    print(f"Bet threshold: {BET_THRESHOLD}")
    print(f"Save bets: {SAVE_BETS}\n")

    # Load and build features
    df = load_weekly_data()
    feats = build_features(df)

    # Only completed games for training/eval
    labeled = feats[feats["home_score"].notna()].copy()
    labeled = labeled[labeled['week'] > 8]
    labeled = labeled.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    if labeled.empty:
        raise ValueError("No completed games found in data")

    # 80/20 chronologic split
    split_idx = int(0.8 * len(labeled))
    train_df = labeled.iloc[:split_idx].copy()
    test_df = labeled.iloc[split_idx:].copy()

    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}\n")

    # Build matrices
    X_tr = build_ml_X(train_df)
    X_te = build_ml_X(test_df)
    y_tr = train_df["home_win"].values
    y_te = test_df["home_win"].values

    # Available model constructors
    available = {
        "logistic": lambda: LogisticRegression(max_iter=2000),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        "xgboost_tuned": None,
    }

    models = []

    # If xgboost_tuned is selected, tune first
    if "xgboost_tuned" in MODELS_TO_RUN:
        print("Tuning XGBoost classifier...")
        best_params_ml, best_cv_logloss = tune_xgb_classifier(train_df)
        print(f"Best CV logloss: {best_cv_logloss:.3f}")
        print(f"Best params: {best_params_ml}\n")
        available["xgboost_tuned"] = lambda: XGBClassifier(**best_params_ml)

    for name in MODELS_TO_RUN:
        if name not in available:
            print(f"Warning: Unknown model '{name}', skipping...")
            continue
        models.append((name, available[name]()))

    results = []
    all_bets = {}

    for name, mdl in models:
        print(f"Evaluating model: {name}")
        res = evaluate_model(name, mdl, X_tr, y_tr, X_te, y_te, 
                             test_df, threshold=BET_THRESHOLD)
        results.append(res)
        all_bets[name] = res.pop("bets_df")
        print(
            f"  ML metrics: logloss={res['logloss']:.4f}, acc={res['accuracy']:.4f}, roc_auc={res['roc_auc']:.4f}"
        )
        print(
            f"  Betting: n_bets={res['n_bets']}, total_profit=${res['total_profit']:.2f}, "
            f"win_rate={res['bet_win_rate']:.2%}, profit_weighted_acc={res['profit_weighted_accuracy']:.4f}\n"
        )

    # Summary dataframe
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "bets_df"} for r in results])
    print("\n=== SUMMARY RESULTS ===")
    print(summary.set_index("name").to_string())

    # Optionally save per-model bets
    if SAVE_BETS:
        os.makedirs("outputs", exist_ok=True)
        for name, bets_df in all_bets.items():
            if not bets_df.empty:
                fname = f"outputs/bets_{name}.csv"
                bets_df.to_csv(fname, index=False)
                print(f"\nSaved bets for {name} to {fname}")


if __name__ == "__main__":
    main()