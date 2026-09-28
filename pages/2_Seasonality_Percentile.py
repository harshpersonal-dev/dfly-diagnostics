"""
View 2 -- Structure Seasonality & Percentile

For one generic curve-structure slot (e.g. "M7 of the 1-Month Double
Fly"), shows where today's reading sits versus the same time of year in
prior years, the historical distribution of magnitudes at that slot, and
how similarly-sized readings tended to resolve over the following
sessions.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from curves.data_loader import STRUCTURE_LABELS, available_structures, data_available, load_structure
from curves.seasonality import (
    WINDOW_DAYS,
    magnitude_distribution,
    seasonal_percentile,
    subsequent_reversion,
    windowed_history,
)

st.set_page_config(page_title="Seasonality & Percentile", layout="wide")
st.title("Structure Seasonality & Percentile")
st.caption("Generic-slot analysis of flies, double-flies, condors and spreads across history.")

if not data_available():
    st.error(
        "Prepared data not found under `data/prepared/`. Run `python prepare_curve_data.py` "
        "once against the source workbook to generate it."
    )
    st.stop()

structures = available_structures()
if not structures:
    st.error("No structure sheets found in data/prepared/structures/.")
    st.stop()

with st.sidebar:
    st.header("Controls")
    structure = st.selectbox(
        "Structure", structures, format_func=lambda s: f"{s} — {STRUCTURE_LABELS.get(s, s)}"
    )
    structure_df = load_structure(structure, field="close")
    max_position = max((int(c[1:]) for c in structure_df.columns if c.startswith("M")), default=1)
    generic_position = st.number_input("Generic position (M#)", min_value=1, max_value=max_position, value=1)
    window = st.selectbox("Lookback window", list(WINDOW_DAYS.keys()), index=4)
    max_date = structure_df.index.max().date() if not structure_df.empty else None
    min_date = structure_df.index.min().date() if not structure_df.empty else None
    as_of = st.date_input("As-of date", value=max_date, min_value=min_date, max_value=max_date)

col = f"M{generic_position}"
if structure_df.empty or col not in structure_df.columns:
    st.warning("No data for this structure/position.")
    st.stop()

series = structure_df[col].dropna()
end_date = pd.Timestamp(as_of)
if end_date not in series.index:
    st.info("No session on the selected date for this structure; pick another date.")
    st.stop()

result = seasonal_percentile(series, end_date, window)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current value", f"{result['current_value']:.3f}" if pd.notna(result["current_value"]) else "—")
m2.metric("Seasonal percentile", f"{result['percentile']:.0f}%" if pd.notna(result["percentile"]) else "—")
m3.metric("Seasonal obs. (±15d, all yrs)", result["n_obs"])
m4.metric("Years covered", result["n_years"])

st.subheader(f"{structure} M{generic_position} history")
hist = windowed_history(series, end_date, window)
hist_fig = go.Figure()
hist_fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode="lines", name=col, line=dict(color="#3b82f6")))
hist_fig.add_trace(go.Scatter(
    x=[end_date], y=[result["current_value"]], mode="markers", name="As-of",
    marker=dict(size=12, color="#ef4444"),
))
hist_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360, yaxis_title="Value ($/bbl)")
st.plotly_chart(hist_fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Magnitude distribution (this slot)")
    mags = magnitude_distribution(series, end_date, window)
    if mags.empty:
        st.info("Not enough history in this window.")
    else:
        dist_fig = go.Figure()
        dist_fig.add_trace(go.Histogram(x=mags.values, nbinsx=40, marker_color="#94a3b8"))
        current_abs = abs(result["current_value"]) if pd.notna(result["current_value"]) else None
        if current_abs is not None:
            dist_fig.add_vline(x=current_abs, line_color="#ef4444", line_dash="dash")
        dist_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10), height=320,
            xaxis_title="|value| ($/bbl)", yaxis_title="Count",
        )
        st.plotly_chart(dist_fig, use_container_width=True)
        st.caption(
            f"Median |value|: {mags.median():.3f}  ·  90th pct: {mags.quantile(0.9):.3f}  ·  "
            f"Today: {current_abs:.3f}" if current_abs is not None else ""
        )

with right:
    st.subheader("Subsequent reversion after similar readings")
    tol = st.slider("Magnitude tolerance (±%)", 5, 100, 25, 5) / 100
    reversion = subsequent_reversion(series, end_date, window, magnitude_tolerance=tol)
    if reversion["n_episodes"] == 0:
        st.info("No comparable historical episodes found at this tolerance/window.")
    else:
        st.caption(f"{reversion['n_episodes']} comparable historical episode(s) found.")
        rows = [
            {"Horizon": f"{h}d", "Avg move": stats["avg"], "Median move": stats["median"]}
            for h, stats in reversion["horizons"].items()
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
