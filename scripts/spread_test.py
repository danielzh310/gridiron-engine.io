from data.loaders import load_weekly_data
from data.preprocessing import build_features
from models.spread import Spread, FEATURES

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# helpers: feature matrices
def build_baseline_X(df, include_spread=False):
    # original linear model features
    X = df[FEATURES].copy()
    if include_spread:
        X["spread_line"] = df["spread_line"].values
    return X


def build_gb_X(df, include_spread=False):
    # engineered features for gb model efficiency
    X = build_baseline_X(df, include_spread=include_spread)
    X["home_attack_adv"] = X["home_off_avg"] - X["away_def_avg"]
    X["away_attack_adv"] = X["away_off_avg"] - X["home_def_avg"]
    X["net_advantage"] = X["home_attack_adv"] - X["away_attack_adv"]
    X["pace"] = X["home_off_avg"] + X["away_off_avg"]
    X["def_total"] = X["home_def_avg"] + X["away_def_avg"]
    X["def_diff"] = X["away_def_avg"] - X["home_def_avg"]
    X["pace_def_interaction"] = X["pace"] * X["def_total"]
    X["attack_def_interaction"] = X["net_advantage"] * X["def_diff"]
    X["home_boost"] = X["home_off_avg"] * 0.12
    return X


# helpers: metrics
def eval_metrics(y_true, y_pred, spread_line):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    pred_cover = (y_pred - spread_line) > 0
    actual_cover = (y_true - spread_line) > 0
    ats_accuracy = float((pred_cover == actual_cover).mean())
    pred_edge = y_pred - spread_line
    actual_edge = y_true - spread_line
    ats_mae = float(np.abs(pred_edge - actual_edge).mean())
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "ats_accuracy": ats_accuracy,
        "ats_mae": ats_mae,
    }

# simple rolling-origin cv for time series
def rolling_splits(n, n_folds=5):
    # create n_folds sequential splits where each fold trains on [0:train_end) and validates on [train_end:valid_end)
    step = n // (n_folds + 1)
    for k in range(1, n_folds + 1):
        train_end = step * k
        valid_end = step * (k + 1) if k < n_folds else n
        yield slice(0, train_end), slice(train_end, valid_end)


# tuner for histgradientboostingregressor on engineered features, residual target
def tune_hgbr(train_df, n_folds=5):
    X_all = build_gb_X(train_df, include_spread=True)
    y_all = (train_df["margin"] - train_df["spread_line"]).values

    param_grid = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.02, 0.03, 0.05],
        "max_iter": [800],
        "l2_regularization": [0.0, 1.0],
        "max_leaf_nodes": [31, 63],
        "min_samples_leaf": [20, 40, 60],
    }

    # expand grid
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    best_score = float("inf")
    best_params = None

    for md in param_grid["max_depth"]:
        for lr in param_grid["learning_rate"]:
            for mi in param_grid["max_iter"]:
                for l2 in param_grid["l2_regularization"]:
                    for mln in param_grid["max_leaf_nodes"]:
                        for msl in param_grid["min_samples_leaf"]:
                            cfg = dict(
                                max_depth=md,
                                learning_rate=lr,
                                max_iter=mi,
                                l2_regularization=l2,
                                max_leaf_nodes=mln,
                                min_samples_leaf=msl,
                                early_stopping=True,
                                validation_fraction=0.1,
                                random_state=42,
                            )
                            fold_scores = []
                            for tr, va in rolling_splits(len(X_all), n_folds=n_folds):
                                model = HistGradientBoostingRegressor(**cfg)
                                model.fit(X_all.iloc[tr], y_all[tr])
                                pred = model.predict(X_all.iloc[va])
                                score = mean_absolute_error(y_all[va], pred)
                                fold_scores.append(score)
                            mean_score = float(np.mean(fold_scores))
                            if mean_score < best_score:
                                best_score = mean_score
                                best_params = cfg

    return best_params, best_score


# tuner for linear models on engineered features, residual target
def tune_linear_residual(train_df, model_type="ridge", alphas=None, n_folds=5):
    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]

    X_all = build_gb_X(train_df, include_spread=True)
    y_all = (train_df["margin"] - train_df["spread_line"]).values

    best_alpha = None
    best_score = float("inf")

    for alpha in alphas:
        fold_scores = []
        for tr, va in rolling_splits(len(X_all), n_folds=n_folds):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_all.iloc[tr])
            Xva = scaler.transform(X_all.iloc[va])
            if model_type == "ridge":
                model = Ridge(alpha=alpha)
            else:
                model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(Xtr, y_all[tr])
            pred = model.predict(Xva)
            score = mean_absolute_error(y_all[va], pred)
            fold_scores.append(score)
        mean_score = float(np.mean(fold_scores))
        if mean_score < best_score:
            best_score = mean_score
            best_alpha = alpha

    return best_alpha, best_score


# load and build
df = load_weekly_data()
feats = build_features(df)

# use only completed games
labeled = feats[feats["home_score"].notna()].copy()
labeled = labeled.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

if labeled.empty:
    raise ValueError("no completed games found")

# split
split_idx = int(0.8 * len(labeled))
train_df = labeled.iloc[:split_idx].copy()
test_df = labeled.iloc[split_idx:].copy()

print(f"train size: {len(train_df)}")
print(f"test size: {len(test_df)}")

# baseline linear regression (old model)
print("training baseline linear regression...")
X_tr_base = build_baseline_X(train_df)
X_te_base = build_baseline_X(test_df)
lr = LinearRegression()
lr.fit(X_tr_base, train_df["margin"].values)
pred_lr = lr.predict(X_te_base)
base_metrics = eval_metrics(
    test_df["margin"].values, pred_lr, test_df["spread_line"].values
)

# linear on engineered features (margin target)
print("training linear on engineered features...")
X_tr_eng = build_gb_X(train_df, include_spread=False)
X_te_eng = build_gb_X(test_df, include_spread=False)
lr_eng = LinearRegression()
lr_eng.fit(X_tr_eng, train_df["margin"].values)
pred_lr_eng = lr_eng.predict(X_te_eng)
lin_eng_metrics = eval_metrics(
    test_df["margin"].values, pred_lr_eng, test_df["spread_line"].values
)

# linear on engineered features, residual target (predict edge)
print("training linear residual (edge) on engineered features...")
X_tr_res = build_gb_X(train_df, include_spread=True)
X_te_res = build_gb_X(test_df, include_spread=True)
y_tr_res = (train_df["margin"] - train_df["spread_line"]).values
lr_res = LinearRegression()
lr_res.fit(X_tr_res, y_tr_res)
pred_edge_lr = lr_res.predict(X_te_res)
pred_lr_res_margin = pred_edge_lr + test_df["spread_line"].values
lin_res_metrics = eval_metrics(
    test_df["margin"].values, pred_lr_res_margin, test_df["spread_line"].values
)

# default gb model (current spread defaults)
print("training default gb model...")
spread_default = Spread()
spread_default.train(train_df)
preds_default_df = spread_default.predict(test_df)
joined_default = preds_default_df.merge(
    test_df[["game_id", "margin", "spread_line"]], on="game_id", how="left"
)
default_metrics = eval_metrics(
    joined_default["margin"].values,
    joined_default["predicted_margin"].values,
    joined_default["spread_line"].values,
)

# tuned gb model (residual target)
print("tuning gb model (rolling cv, residual target)...")
best_params, cv_mae = tune_hgbr(train_df, n_folds=5)
print(f"best cv mae: {cv_mae:.3f} | params: {best_params}")
print("training tuned gb model...")
gb_tuned = HistGradientBoostingRegressor(**best_params)
gb_tuned.fit(build_gb_X(train_df, include_spread=True), y_tr_res)
pred_edge_tuned = gb_tuned.predict(build_gb_X(test_df, include_spread=True))
pred_tuned_margin = pred_edge_tuned + test_df["spread_line"].values
tuned_metrics = eval_metrics(
    test_df["margin"].values, pred_tuned_margin, test_df["spread_line"].values
)

# tuned ridge and lasso (residual target)
print("tuning ridge (residual target)...")
ridge_alpha, ridge_cv_mae = tune_linear_residual(train_df, model_type="ridge", n_folds=5)
print(f"best ridge cv mae: {ridge_cv_mae:.3f} | alpha: {ridge_alpha}")
scaler_ridge = StandardScaler()
Xtr_scaled = scaler_ridge.fit_transform(X_tr_res)
Xte_scaled = scaler_ridge.transform(X_te_res)
ridge = Ridge(alpha=ridge_alpha)
ridge.fit(Xtr_scaled, y_tr_res)
pred_edge_ridge = ridge.predict(Xte_scaled)
pred_ridge_margin = pred_edge_ridge + test_df["spread_line"].values
ridge_metrics = eval_metrics(
    test_df["margin"].values, pred_ridge_margin, test_df["spread_line"].values
)

print("tuning lasso (residual target)...")
lasso_alpha, lasso_cv_mae = tune_linear_residual(train_df, model_type="lasso", n_folds=5)
print(f"best lasso cv mae: {lasso_cv_mae:.3f} | alpha: {lasso_alpha}")
scaler_lasso = StandardScaler()
Xtr_scaled_l = scaler_lasso.fit_transform(X_tr_res)
Xte_scaled_l = scaler_lasso.transform(X_te_res)
lasso = Lasso(alpha=lasso_alpha, max_iter=10000)
lasso.fit(Xtr_scaled_l, y_tr_res)
pred_edge_lasso = lasso.predict(Xte_scaled_l)
pred_lasso_margin = pred_edge_lasso + test_df["spread_line"].values
lasso_metrics = eval_metrics(
    test_df["margin"].values, pred_lasso_margin, test_df["spread_line"].values
)

# report
def print_block(title, m):
    print(f"== {title} ==")
    print(f"mae:         {m['mae']:.3f}")
    print(f"rmse:        {m['rmse']:.3f}")
    print(f"r2:          {m['r2']:.3f}")
    print(f"ats accuracy:{m['ats_accuracy']:.3f}")
    print(f"ats mae:     {m['ats_mae']:.3f}")

def print_thresholds(title, y_pred, spread_line, y_true):
    print(f"-- {title} ats by |edge| threshold --")
    edges = np.abs(y_pred - spread_line)
    for thr in [0.5, 1.0, 1.5, 2.0, 2.5]:
        m = edges >= thr
        if m.any():
            acc = (
                ((y_pred[m] - spread_line[m]) > 0) == ((y_true[m] - spread_line[m]) > 0)
            ).mean()
            print(f"thr>={thr:.1f}: acc={acc:.3f} (n={int(m.sum())})")

print("\ncomparison on holdout test set")
print_block("baseline linear", base_metrics)
print_block("linear engineered", lin_eng_metrics)
print_block("linear engineered residual", lin_res_metrics)
print_block("gb default", default_metrics)
print_block("gb tuned", tuned_metrics)
print_block("ridge tuned residual", ridge_metrics)
print_block("lasso tuned residual", lasso_metrics)

print("\nats accuracy by edge thresholds")
print_thresholds("baseline linear", pred_lr, test_df["spread_line"].values, test_df["margin"].values)
print_thresholds("linear engineered", pred_lr_eng, test_df["spread_line"].values, test_df["margin"].values)
print_thresholds("linear engineered residual", pred_lr_res_margin, test_df["spread_line"].values, test_df["margin"].values)
print_thresholds("gb default", joined_default["predicted_margin"].values, joined_default["spread_line"].values, joined_default["margin"].values)
print_thresholds("gb tuned residual", pred_tuned_margin, test_df["spread_line"].values, test_df["margin"].values)
print_thresholds("ridge tuned residual", pred_ridge_margin, test_df["spread_line"].values, test_df["margin"].values)
print_thresholds("lasso tuned residual", pred_lasso_margin, test_df["spread_line"].values, test_df["margin"].values)

