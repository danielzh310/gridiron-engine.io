import nflreadpy as nfl


def load_weekly_data(seasons=None):
    """
    Load NFL schedules for all historical seasons plus the current season
    so that both training games and upcoming games are included.
    
    Note: For the presentation, we will need to go back to Week 13
    """

    if seasons is None:
        current_season = nfl.get_current_season()
        seasons = list(range(2020, current_season + 1))

    schedules = nfl.load_schedules(seasons=seasons)

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
    ]

    return schedules.select(keep).to_pandas()