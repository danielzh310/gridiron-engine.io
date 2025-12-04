import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nflreadpy as nfl


OUTPUT_DIR = "eda/figures"
SEASONS = list(range(2020, 2025))
MAX_WEEKLY_VISUALS = 12

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 140


def load_game_totals():
    print("Loading play-by-play data from nflreadpy...")

    pbp = nfl.load_pbp(SEASONS)

    pbp = pbp.select([
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "season",
        "week"
    ]).to_pandas()

    print("Aggregating to game level...")

    games = pbp.drop_duplicates("game_id").copy()

    games["actual_total"] = games["home_score"] + games["away_score"]

    return games


def plot_weekly_total_distributions(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sample = df.sample(min(MAX_WEEKLY_VISUALS, len(df)), random_state=42)

    for _, row in sample.iterrows():

        game_id = row["game_id"]

        mean = row["actual_total"]
        std = max(0.18 * mean, 4.5)

        x = np.linspace(mean - 4 * std, mean + 4 * std, 400)
        y = np.exp(-(x - mean) ** 2 / (2 * std ** 2))
        y /= y.max()

        plt.figure(figsize=(7, 4))

        plt.plot(
            x, y,
            linewidth=2,
            label="Scoring distribution",
            color="#1f77b4"
        )

        plt.fill_between(x, y, alpha=0.20)

        plt.axvline(mean, color="black", linewidth=2, label="Actual total")

        plt.title(f"Weekly Game Scoring Distribution — {game_id}")
        plt.xlabel("Total Points")
        plt.yticks([])
        plt.legend()

        plt.tight_layout()

        fname = os.path.join(OUTPUT_DIR, f"week_dist_{game_id}.png")
        plt.savefig(fname)
        plt.close()

    print("Weekly bell-curve scoring visuals created.")


def plot_team_defense_heatmap(df):
    print("Creating team-vs-defense heatmap...")

    home = df[["home_team","away_team","home_score"]].rename(
        columns={
            "home_team":"offense",
            "away_team":"defense",
            "home_score":"points"
        }
    )

    away = df[["away_team","home_team","away_score"]].rename(
        columns={
            "away_team":"offense",
            "home_team":"defense",
            "away_score":"points"
        }
    )

    combined = pd.concat([home, away])

    pivot = (
        combined
        .groupby(["offense","defense"])["points"]
        .mean()
        .reset_index()
        .pivot(
            index="offense",
            columns="defense",
            values="points"
        )
    )

    plt.figure(figsize=(13, 11))

    sns.heatmap(
        pivot,
        cmap="coolwarm",
        center=24,
        linewidths=0.2,
        linecolor="gray"
    )

    plt.title("Team Offense vs Defense Scoring Heatmap\n(Average Points Scored)")
    plt.xlabel("Defensive Team")
    plt.ylabel("Offensive Team")

    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "team_defense_heatmap.png"))
    plt.close()

    print("Heatmap saved.")


def plot_scoring_distribution(df):
    plt.figure(figsize=(8, 5))

    sns.kdeplot(df["actual_total"], fill=True, linewidth=3)

    plt.xlabel("Total Game Points")
    plt.ylabel("Density")
    plt.title("NFL Game Total Scoring Distribution")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "scoring_distribution.png"))
    plt.close()

    print("Scoring KDE distribution saved.")


def plot_season_scoring_trends(df):
    season_means = (
        df.groupby("season")["actual_total"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 4))

    plt.plot(
        season_means["season"],
        season_means["actual_total"],
        marker="o",
        linewidth=2
    )

    plt.xlabel("Season")
    plt.ylabel("Average Game Total")
    plt.title("NFL Scoring Trend by Season")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "season_scoring_trend.png"))
    plt.close()

    print("Season trend chart saved.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_game_totals()

    plot_weekly_total_distributions(df)
    plot_team_defense_heatmap(df)
    plot_scoring_distribution(df)
    plot_season_scoring_trends(df)

    print(f"\nEDA visuals generated in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()