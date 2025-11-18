"""
In-Class Project Progress Check-In
1. Loads NFL schedule and team statistics using nflreadpy.
2. Builds a convex-hull clustered scatterplot of all NFL teams based on
   average points scored and points allowed.
3. Builds a 10-year performance chart for the Pittsburgh Steelers using
   season averages for points and offensive yardage categories.
"""

import nflreadpy as nfl
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
import numpy as np


def load_games():
    """
    Load home and away NFL game results into a unified dataframe.

    Returns:
        pandas.DataFrame: Combined home/away dataset with team, points_for,
                          points_against, season, and week.
    """
    sched = nfl.load_schedules()

    home = sched.select([
        "season",
        "week",
        pl.col("home_team").alias("team"),
        pl.col("home_score").alias("points_for"),
        pl.col("away_score").alias("points_against")
    ])

    away = sched.select([
        "season",
        "week",
        pl.col("away_team").alias("team"),
        pl.col("away_score").alias("points_for"),
        pl.col("home_score").alias("points_against")
    ])

    return pl.concat([home, away]).to_pandas()


def load_team_stats(seasons):
    """
    Load aggregated team-level season statistics.

    Args:
        seasons (list[int]): Seasons to load.

    Returns:
        pandas.DataFrame: Per-team, per-season averages for key yardage stats.
    """
    pbp = nfl.load_team_stats(seasons=True).to_pandas()

    return pbp.groupby(["team", "season"]).agg({
        "passing_yards": "mean",
        "rushing_yards": "mean",
        "misc_yards": "mean"
    }).reset_index()


def plot_clusters(cluster_df, season_min, season_max):
    """
    Plot convex-hull clusters for all 32 NFL teams based on season-averaged
    scoring performance.

    Args:
        cluster_df (pd.DataFrame): Contains team, points_for, points_against, cluster ID.
        season_min (int): First season in the dataset.
        season_max (int): Last season in the dataset.
    """
    plt.figure(figsize=(10, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, cluster_df["cluster"].nunique()))

    for i, cluster_id in enumerate(sorted(cluster_df["cluster"].unique())):
        pts = cluster_df[cluster_df["cluster"] == cluster_id]

        # Scatter plot for cluster points
        plt.scatter(pts["points_for"], pts["points_against"], s=120, color=colors[i])

        # Convex hull for cluster shape
        if len(pts) >= 3:
            hull = ConvexHull(pts[["points_for", "points_against"]].values)
            hull_pts = pts.iloc[hull.vertices][["points_for", "points_against"]].values
            plt.fill(hull_pts[:, 0], hull_pts[:, 1], color=colors[i], alpha=0.2)

        # Team labels
        for _, r in pts.iterrows():
            plt.text(r["points_for"] + 0.15, r["points_against"] + 0.15, r["team"], fontsize=8)

    plt.xlabel("Points For (Avg)")
    plt.ylabel("Points Against (Avg)")
    plt.title(f"NFL Team Clusters (Seasons {season_min}–{season_max})")
    plt.show()


def plot_steelers(pit_final):
    """
    Plot a 10-year trendline chart for Pittsburgh Steelers that includes
    season averages for:
    - Points scored
    - Points allowed
    - Point differential
    - Passing / rushing / miscellaneous yards

    Args:
        pit_final (pd.DataFrame): Combined Steelers points + yardage dataset.
    """
    plt.figure(figsize=(12, 8))

    ax1 = plt.gca()
    ax1.plot(pit_final["season"], pit_final["points_for"], linewidth=2, label="Points For")
    ax1.plot(pit_final["season"], pit_final["points_against"], linewidth=2, label="Points Against")
    ax1.plot(pit_final["season"], pit_final["point_diff"], linewidth=2, label="Point Differential")
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Points")
    ax1.grid(True)

    # Yardage plotted on second Y-axis
    ax2 = ax1.twinx()
    ax2.plot(pit_final["season"], pit_final["passing_yards"], linewidth=2, linestyle="--", label="Passing Yards")
    ax2.plot(pit_final["season"], pit_final["rushing_yards"], linewidth=2, linestyle="--", label="Rushing Yards")
    ax2.plot(pit_final["season"], pit_final["misc_yards"], linewidth=2, linestyle="--", label="Misc Yards")
    ax2.set_ylabel("Yards")

    # Unified legend across both axes
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc="upper left")

    plt.title("Pittsburgh Steelers 10-Year Performance Trends (Season Averages)")
    plt.show()

# Load game results
games = load_games()
seasons = sorted(games["season"].unique())
season_min = min(seasons)
season_max = max(seasons)

# Load per-season yardage stats
pbp_small = load_team_stats(seasons)

# Create clustering features
merged = games.groupby("team").agg({
    "points_for": "mean",
    "points_against": "mean"
}).reset_index()

cluster_df = merged.copy()
kmeans = KMeans(n_clusters=4, n_init=10)
cluster_df["cluster"] = kmeans.fit_predict(cluster_df[["points_for", "points_against"]])

# Plot the cluster graph
plot_clusters(cluster_df, season_min, season_max)

current_season = games["season"].max()

pit_games = games[(games["team"] == "PIT") & (games["season"] >= current_season - 9)]
pit_extra = pbp_small[(pbp_small["team"] == "PIT") & (pbp_small["season"] >= current_season - 9)]

pit_grouped = pit_games.groupby("season").agg({
    "points_for": "mean",
    "points_against": "mean"
}).reset_index()

pit_grouped["team"] = "PIT"
pit_grouped["point_diff"] = pit_grouped["points_for"] - pit_grouped["points_against"]

# Merge with yardage metrics
pit_final = pit_grouped.merge(pit_extra, on=["team", "season"], how="left")
plot_steelers(pit_final)