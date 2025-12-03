import pandas as pd

ROLLING_WINDOW = 5


def build_features(df):
    """
    Build rolling offensive and defensive team features
    for modeling moneyline, spread, and totals.
    """

    data = df.copy()

    # Targets and helper columns
    data["home_win"] = (data["home_score"] > data["away_score"]).astype(int)
    data["margin"] = data["home_score"] - data["away_score"]
    data["total_pts"] = data["home_score"] + data["away_score"]

    home = data[
        ["game_id", "season", "week", "home_team", "home_score", "away_score"]
    ].copy()

    home.rename(
        columns={
            "home_team": "team",
            "home_score": "pts_for",
            "away_score": "pts_against",
        },
        inplace=True,
    )

    away = data[
        ["game_id", "season", "week", "away_team", "away_score", "home_score"]
    ].copy()

    away.rename(
        columns={
            "away_team": "team",
            "away_score": "pts_for",
            "home_score": "pts_against",
        },
        inplace=True,
    )

    teams = (
        pd.concat([home, away])
        .sort_values(["team", "season", "week"])
        .reset_index(drop=True)
    )

    teams["off_avg"] = (
        teams.groupby("team")["pts_for"]
        .rolling(ROLLING_WINDOW)
        .mean()
        .reset_index(0, drop=True)
        .fillna(0)
    )

    teams["def_avg"] = (
        teams.groupby("team")["pts_against"]
        .rolling(ROLLING_WINDOW)
        .mean()
        .reset_index(0, drop=True)
        .fillna(0)
    )

    home_feats = teams[
        ["game_id", "team", "off_avg", "def_avg"]
    ].rename(
        columns={
            "team": "home_team",
            "off_avg": "home_off_avg",
            "def_avg": "home_def_avg",
        }
    )

    away_feats = teams[
        ["game_id", "team", "off_avg", "def_avg"]
    ].rename(
        columns={
            "team": "away_team",
            "off_avg": "away_off_avg",
            "def_avg": "away_def_avg",
        }
    )

    data = data.merge(home_feats, on=["game_id", "home_team"])
    data = data.merge(away_feats, on=["game_id", "away_team"])

    return data