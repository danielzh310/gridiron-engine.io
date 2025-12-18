"""
Main entrypoint for the NFL Parlay Picker

Runs the prediction pipeline:
- loads data
- builds features
- runs models
- exports picks
"""

from controllers.predict import run_weekly


def main():
    run_weekly(week = 16)


if __name__ == "__main__":
    main()