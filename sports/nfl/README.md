# QuantPlay NFL

This is the current working football module.

- `controllers/` orchestrates production-style flows, such as weekly prediction generation.
- `data/` loads sport-specific source data.
- `models/` contains football-specific model classes and feature assumptions.
- `scripts/` contains runnable experiments, backtests, and model comparison harnesses.

Shared math such as odds conversion, payout helpers, and Kelly staking should live in `utils/`
so future `sports/mlb` and `sports/nba` modules can reuse it without copying football models.
