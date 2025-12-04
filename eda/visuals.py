import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = "outputs/weekly_picks.csv"
OUTPUT_DIR = "eda/figures"
MAX_PLOTS = 8


def plot_total_distributions(df):
    """
    Generates diagnostic visuals comparing predicted scoring distributions
    against sportsbook totals.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, row in df.head(MAX_PLOTS).iterrows():

        game_id = row["game_id"]

        mean = row["projected_total"]
        q20 = row["total_low_q"]
        q80 = row["total_high_q"]
        vegas_line = mean - row["vegas_total_edge"]

        std = max((q80 - q20) / 2.56, 1.0)

        x = np.linspace(mean - 4 * std, mean + 4 * std, 400)
        y = np.exp(-(x - mean) ** 2 / (2 * std ** 2))
        y /= y.max()

        plt.figure(figsize=(7, 4))

        plt.plot(x, y, label="Predicted distribution")

        plt.axvline(mean, linestyle="--", label="Model mean")
        plt.axvline(q20, linestyle=":", label="20% quantile")
        plt.axvline(q80, linestyle=":", label="80% quantile")
        plt.axvline(vegas_line, linestyle="-", label="Vegas total")

        plt.title(f"Total Projection — {game_id}")
        plt.xlabel("Total Points")
        plt.yticks([])
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(OUTPUT_DIR, f"total_visual_{game_id}.png")
        plt.savefig(fname)
        plt.close()

    print(f"EDA visuals to {OUTPUT_DIR}/")


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            "weekly_picks.csv not found. Run main.py before generating visuals."
        )

    df = pd.read_csv(INPUT_PATH)
    plot_total_distributions(df)


if __name__ == "__main__":
    main()