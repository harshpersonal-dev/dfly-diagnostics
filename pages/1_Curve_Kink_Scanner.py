"""
View 1 -- Curve Kink Scanner

Plots the full outright curve (by curve position, M1..M24) on a chosen
date and flags "kinks": points where the curve bends sharply relative
to its neighbours. Entirely additive -- reads only from data/prepared/
(see prepare_curve_data.py) and never touches dflies_data.csv or the
classic diagnostics page.
"""

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from curves.data_loader import available_dates, data_available, get_curve, nearest_date, shift_date
from curves.kink_detection import detect_kinks, kink_table

st.set_page_config(page_title="Curve Kink Scanner", layout="wide")
st.title("Curve Kink Scanner")
st.caption("Outright forward curve by curve position, with automatic kink flagging.")

if not data_available():
    st.error(
        "Prepared data not found under `data/prepared/`. Run `python prepare_curve_data.py` "
        "once against the source workbook to generate it."
    )
    st.stop()

dates = available_dates()
min_date, max_date = dates.min().date(), dates.max().date()

if "kink_scanner_date" not in st.session_state:
    st.session_state.kink_scanner_date = max_date

with st.sidebar:
    st.header("Controls")
    picked = st.date_input(
        "Date", value=st.session_state.kink_scanner_date, min_value=min_date, max_value=max_date
    )
    st.session_state.kink_scanner_date = picked

    nav1, nav2 = st.columns(2)
    if nav1.button("Prev day", use_container_width=True):
        current = nearest_date(pd.Timestamp(st.session_state.kink_scanner_date), dates)
        st.session_state.kink_scanner_date = shift_date(current, dates, -1).date()
        st.rerun()
    if nav2.button("Next day", use_container_width=True):
        current = nearest_date(pd.Timestamp(st.session_state.kink_scanner_date), dates)
        st.session_state.kink_scanner_date = shift_date(current, dates, 1).date()
        st.rerun()

    st.divider()
    threshold = st.slider("Kink threshold (|2nd difference|, $/bbl)", 0.01, 0.50, 0.05, 0.01)
    st.divider()
    st.caption("Optional secondary rules (ANDed onto the threshold rule)")
    use_sign_change = st.toggle("Require local slope sign-flip", value=False)
    use_mad_outlier = st.toggle("Require MAD-robust outlier", value=False)
    mad_k = st.slider("MAD outlier sensitivity (k)", 1.0, 6.0, 3.0, 0.5, disabled=not use_mad_outlier)

target = pd.Timestamp(st.session_state.kink_scanner_date)
snapped = nearest_date(target, dates)
if snapped.date() != target.date():
    st.info(f"No session on {target.date()}; showing nearest trading day {snapped.date()}.")

curve = get_curve(snapped)
if curve.empty:
    st.warning("No curve data for this date.")
    st.stop()

result = detect_kinks(
    curve, threshold=threshold, use_sign_change=use_sign_change,
    use_mad_outlier=use_mad_outlier, mad_k=mad_k,
)

col_left, col_right = st.columns([3, 1])
with col_right:
    st.metric("Curve points", len(curve))
    st.metric("Kinks flagged", len(result.kink_positions))
    st.metric("Front contract", str(curve.iloc[0]["contract"]))

with col_left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["curve_position"], y=curve["close"], mode="lines+markers",
        name="Outright close", line=dict(color="#3b82f6", width=2), marker=dict(size=6),
    ))
    if len(result.kink_positions) > 0:
        kinked = curve[curve["curve_position"].isin(result.kink_positions)]
        fig.add_trace(go.Scatter(
            x=kinked["curve_position"], y=kinked["close"], mode="markers", name="Kink",
            marker=dict(size=14, color="#ef4444", symbol="x", line=dict(width=2, color="#ef4444")),
            text=kinked["contract"], hovertemplate="M%{x} (%{text})<br>%{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        xaxis_title="Curve position (M1 = front month)", yaxis_title="Close ($/bbl)",
        margin=dict(l=10, r=10, t=10, b=10), height=460, hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Curvature (second difference) by curve position")
if len(result.second) > 0:
    curvature_fig = go.Figure()
    x_pos = result.positions[1:-1]
    colors = ["#ef4444" if flagged else "#94a3b8" for flagged in result.combined_mask]
    curvature_fig.add_trace(go.Bar(x=x_pos, y=result.second, marker_color=colors, name="2nd difference"))
    curvature_fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444", opacity=0.6)
    curvature_fig.add_hline(y=-threshold, line_dash="dash", line_color="#ef4444", opacity=0.6)
    curvature_fig.update_layout(
        xaxis_title="Curve position", yaxis_title="2nd difference ($/bbl)",
        margin=dict(l=10, r=10, t=10, b=10), height=280,
    )
    st.plotly_chart(curvature_fig, use_container_width=True)

st.subheader("Flagged kinks")
table = kink_table(curve, result)
if table.empty:
    st.info("No kinks flagged for this date/threshold combination.")
else:
    st.dataframe(table, use_container_width=True, hide_index=True)
