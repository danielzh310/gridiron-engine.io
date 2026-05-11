import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from sports.nfl.controllers.predict import run_weekly
from sports.nfl.controllers.results import (
    grade_saved_predictions,
    prediction_snapshot_path,
    save_prediction_snapshot,
)

from utils.kelly import allocate_kelly

st.set_page_config(page_title="QuantPlay", layout="wide")
st.title("quantplay.io")

col1, col2 = st.columns([1, 1])
with col1:
    seasons = list(range(2020, 2031))
    season = st.selectbox("Season", options=seasons, index=seasons.index(2025))
with col2:
    week_labels = (
        ["Select week..."]
        + [str(i) for i in range(1, 19)]
        + ["Wildcard", "Divisional", "Conference Championship", "Super Bowl"]
    )

    selected_week_label = st.selectbox("Week", options=week_labels, index=0)

    label_to_week = {str(i): i for i in range(1, 19)}
    label_to_week.update({
        "Wildcard": 19,
        "Divisional": 20,
        "Conference Championship": 21,
        "Super Bowl": 22,
    })

    week = label_to_week.get(selected_week_label, None)

st.divider()

ordered_cols = [
    "game_id",
    "home_team",
    "away_team",

    "ml_pick",
    "ml_home_prob",
    "ml_away_prob",
    "home_ml_bet",
    "away_ml_bet",
    "home_moneyline",
    "away_moneyline",

    "spread_line",
    "spread_pick",
    "model_spread_margin",
    "spread_edge",

    "total_line",
    "total_pick",
    "projected_total",
    "vegas_total_edge",

    "total_low_q",
    "total_high_q",
    "total_buffer",
]

default_on = [
    "game_id",
    "home_team",
    "away_team",
    "home_ml_bet",
    "away_ml_bet",
    "home_moneyline",
    "away_moneyline",
]

if "selected_cols" not in st.session_state:
    st.session_state["selected_cols"] = default_on.copy()

st.subheader("Columns to display")

b1, b2, b3, b4 = st.columns([1, 1, 1, 2])

def set_defaults():
    st.session_state["selected_cols"] = default_on.copy()

def select_all():
    st.session_state["selected_cols"] = ordered_cols.copy()

def clear_all():
    st.session_state["selected_cols"] = []

with b1:
    st.button("Select defaults", on_click=set_defaults)
with b2:
    st.button("Select all", on_click=select_all)
with b3:
    st.button("Clear all", on_click=clear_all)
with b4:
    st.caption("Defaults = game/team + ML bet + moneylines")

with st.expander("Filter options", expanded=False):
    st.session_state["selected_cols"] = st.multiselect(
        "Columns",
        options=ordered_cols,
        default=st.session_state["selected_cols"],
    )

st.divider()
st.subheader("Kelly sizing (Moneyline)")

k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
with k1:
    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=160.0, step=10.0)
with k2:
    fraction_of_kelly = st.selectbox("Kelly fraction", options=[0.25, 0.5, 1.0], index=0)
with k3:
    normalize_to_full_bankroll = st.checkbox("Normalize to use full bankroll", value=True)
with k4:
    cap_pct = st.number_input("Max bet cap (% of bankroll)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

cap_fraction_of_bankroll = None
if cap_pct > 0:
    cap_fraction_of_bankroll = float(cap_pct) / 100.0

st.caption(
    "Kelly uses model probability and sportsbook odds to size stakes. "
    "Only positive-edge picks get a stake. If normalization is on, stakes are scaled to sum to the bankroll."
)

st.divider()

run_col, results_col = st.columns([1, 1])
with run_col:
    run = st.button("Run predictions", type="primary", use_container_width=True)
with results_col:
    check_results = st.button("Check results", use_container_width=True)


def add_kelly_columns(df):
    needed = ["ml_pick", "ml_home_prob", "ml_away_prob", "home_moneyline", "away_moneyline"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Kelly sizing skipped, missing columns: {missing}")

    probs = []
    odds = []
    for _, r in df.iterrows():
        pick = str(r["ml_pick"]).upper()
        if pick == "HOME":
            probs.append(float(r["ml_home_prob"]))
            odds.append(float(r["home_moneyline"]))
        elif pick == "AWAY":
            probs.append(float(r["ml_away_prob"]))
            odds.append(float(r["away_moneyline"]))
        else:
            probs.append(0.0)
            odds.append(100.0)

    stakes, meta = allocate_kelly(
        probs=probs,
        odds=odds,
        bankroll=float(bankroll),
        fraction_of_kelly=float(fraction_of_kelly),
        normalize_to_full_bankroll=bool(normalize_to_full_bankroll),
        cap_fraction_of_bankroll=cap_fraction_of_bankroll,
        min_stake=0.0,
    )

    out = df.copy()
    out["kelly_stake_ml"] = [round(s, 2) for s in stakes]
    out["kelly_no_bet_ml"] = [m.is_no_bet for m in meta]
    out["kelly_f_ml"] = [round(float(m.kelly_fraction), 4) for m in meta]
    return out


def selected_view(df):
    selected = st.session_state.get("selected_cols", default_on)
    if not selected:
        st.warning("No columns selected, showing defaults.")
        selected = default_on.copy()

    selected = [c for c in ordered_cols if c in set(selected)]
    view_df = df[selected].copy()

    for col in ["kelly_stake_ml", "kelly_no_bet_ml", "kelly_f_ml"]:
        if col in df.columns:
            view_df[col] = df[col]

    return view_df

if run:
    if week is None:
        st.warning("Please select a week before running predictions.")
    else:
        with st.spinner("Running pipeline..."):
            df = run_weekly(
                season=int(season),
                week=int(week),
                export=False,
                verbose=False,
            )

        if df is None or len(df) == 0:
            st.warning("No games found for that season/week.")
        else:
            try:
                df = add_kelly_columns(df)
                st.caption("Added columns: kelly_stake_ml, kelly_no_bet_ml, kelly_f_ml")
            except Exception as e:
                st.warning(f"Kelly sizing failed: {e}")

            snapshot = save_prediction_snapshot(df, int(season), int(week))
            view_df = selected_view(df)

            st.success(f"Generated and saved {len(df)} picks to {snapshot}.")
            st.dataframe(view_df, use_container_width=True)

            csv_bytes = view_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download displayed CSV",
                data=csv_bytes,
                file_name=f"nfl_weekly_picks_{int(season)}_wk{int(week)}.csv",
                mime="text/csv",
            )

if check_results:
    if week is None:
        st.warning("Please select a week before checking results.")
    else:
        snapshot = prediction_snapshot_path(int(season), int(week))
        if not snapshot.exists():
            st.warning(f"No saved predictions found yet for {int(season)} week {int(week)}.")
        else:
            with st.spinner("Checking saved picks against completed games..."):
                graded, summary = grade_saved_predictions(int(season), int(week))

            if graded is None:
                st.warning(summary["message"])
            else:
                st.success(summary["message"])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Completed games", summary["games_completed"])
                m2.metric("ML hits", f'{summary["ml_hits"]}/{summary["games_completed"]}')
                m3.metric("Spread hits", f'{summary["spread_hits"]}/{summary["games_completed"]}')
                m4.metric("ML net", f'${summary["ml_net"]:.2f}')

                if summary["total_bets"] > 0:
                    st.metric("Total hits", f'{summary["total_hits"]}/{summary["total_bets"]}')

                result_cols = [
                    "game_id",
                    "home_team",
                    "away_team",
                    "ml_pick",
                    "ml_result",
                    "ml_hit",
                    "kelly_stake_ml",
                    "ml_net",
                    "spread_pick",
                    "spread_result",
                    "spread_hit",
                    "total_pick",
                    "total_result",
                    "total_hit",
                    "home_score",
                    "away_score",
                    "actual_total",
                ]
                result_cols = [c for c in result_cols if c in graded.columns]
                st.dataframe(graded[result_cols], use_container_width=True)
                st.caption(f'Saved graded results to {summary["results_path"]}')
