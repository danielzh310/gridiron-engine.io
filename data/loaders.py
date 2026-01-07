import nflreadpy as nfl


def load_weekly_data(seasons=None):
    """
    Load NFL schedules for all historical seasons plus the current season
    so that both training games and upcoming games are included.
    """

    if seasons is None:
        current_season = nfl.get_current_season()
        seasons = list(range(2022, current_season + 1))

    schedules = nfl.load_schedules(seasons=seasons)

    if "season_type" in schedules.columns:
        schedules = schedules.with_columns(
            nfl.pl.when(nfl.pl.col("season_type") == "POST")
            .then(nfl.pl.col("week") + 18)
            .otherwise(nfl.pl.col("week"))
            .alias("week")
        )

    keep = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "spread_line",
        "total_line",
        "home_moneyline",
        "away_moneyline",
    ]

    return schedules.select(keep).to_pandas()