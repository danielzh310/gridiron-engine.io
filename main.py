"""
Main entrypoint for QuantPlay model predictions.

Runs the prediction pipeline:
- loads data
- builds features
- runs models
- exports picks
"""

from sports.nfl.controllers.predict import run_weekly


def main():
    run_weekly(week=17)


if __name__ == "__main__":
    main()
