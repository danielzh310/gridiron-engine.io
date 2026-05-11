# QuantPlay Sports Modules

Sport-specific pipelines live here. The current working implementation is under
`sports/nfl`, while `sports/mlb` and `sports/nba` are scaffolds for future models.

Each sport should own its own data loading, feature engineering, and model choices.
Shared betting math, odds conversion, backtesting helpers, and app utilities should live
outside these folders in `utils/`.
