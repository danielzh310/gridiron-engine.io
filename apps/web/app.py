import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
from controllers.predict import run_weekly

st.set_page_config(page_title="Gridiron Engine", layout="wide")
st.title("gridiron-engine.io")

col1, col2 = st.columns([1, 1])
with col1:
    seasons = list(range(2020, 2031))
    season = st.selectbox("Season", options=seasons, index=seasons.index(2025))
with col2:
    weeks = list(range(1, 19))
    week = st.selectbox("Week", options=weeks, index=weeks.index(17))

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

    "spread_pick",
    "model_spread_margin",
    "spread_edge",

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

run = st.button("Run predictions", type="primary")

if run:
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
        selected = st.session_state.get("selected_cols", default_on)
        if not selected:
            st.warning("No columns selected, showing defaults.")
            selected = default_on.copy()

        selected = [c for c in ordered_cols if c in set(selected)]

        view_df = df[selected]

        st.success(f"Generated {len(df)} picks.")
        st.dataframe(view_df, use_container_width=True)

        csv_bytes = view_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download displayed CSV",
            data=csv_bytes,
            file_name=f"weekly_picks_{int(season)}_wk{int(week)}.csv",
            mime="text/csv",
        )
