"""
View 3 -- Contract Lifecycle

Traces one *specific* physical structure (outright, spread, or fly --
identified by its fixed delivery-month legs, e.g. "COH24-J24-K24") across
its whole life in the dataset, with an optional overlay of the same
seasonal contract in other years, and an optional user-uploaded OHLC
series (held only in st.session_state, never written to disk).
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from curves.data_loader import available_instruments, data_available, load_ohlc_for_instrument
from curves.lifecycle import align_multi_year, parse_uploaded_ohlc, shift_instrument_year

st.set_page_config(page_title="Contract Lifecycle", layout="wide")
st.title("Contract Lifecycle")
st.caption("Full-life reconstruction of one specific outright, spread or fly contract, with multi-year overlay.")

if not data_available():
    st.error(
        "Prepared data not found under `data/prepared/`. Run `python prepare_curve_data.py` "
        "once against the source workbook to generate it."
    )
    st.stop()

with st.sidebar:
    st.header("Controls")
    structure_type = st.radio("Structure type", ["outright", "spread", "fly"], horizontal=True)
    instruments = available_instruments(structure_type)
    if instruments.empty:
        st.warning("No instruments of this type found.")
        st.stop()
    instrument = st.selectbox("Instrument", instruments["Instrument"].tolist())

    st.divider()
    overlay_years = st.slider("Overlay +/- N seasonal years", 0, 5, 2)
    field = st.selectbox("Field", ["close", "open", "high", "low"], index=0)

    st.divider()
    st.caption("Optional: overlay an uploaded OHLC series (kept in-memory for this session only)")
    uploaded = st.file_uploader("Upload OHLC (CSV or JSON)", type=["csv", "json"])
    if uploaded is not None:
        try:
            st.session_state["lifecycle_upload"] = parse_uploaded_ohlc(uploaded.getvalue(), uploaded.name)
            st.success(f"Loaded {len(st.session_state['lifecycle_upload'])} uploaded rows.")
        except ValueError as e:
            st.error(str(e))
    if st.session_state.get("lifecycle_upload") is not None and st.button("Clear uploaded data"):
        st.session_state["lifecycle_upload"] = None

base = load_ohlc_for_instrument(instrument)
if base.empty:
    st.warning("No OHLC rows for this instrument.")
    st.stop()

base_series = base.set_index("Date")[field].dropna()

m1, m2, m3 = st.columns(3)
m1.metric("First session", base["Date"].min().date().isoformat())
m2.metric("Last session", base["Date"].max().date().isoformat())
m3.metric("Sessions", len(base))

st.subheader(f"{instrument} — {field} lifecycle")
fig = go.Figure()
fig.add_trace(go.Scatter(x=base_series.index, y=base_series.values, mode="lines", name=instrument,
                          line=dict(color="#3b82f6", width=2)))

upload_df = st.session_state.get("lifecycle_upload")
if upload_df is not None:
    field_map = {"close": "Close", "open": "Open", "high": "High", "low": "Low"}
    up_series = upload_df.set_index("Date")[field_map[field]].dropna()
    fig.add_trace(go.Scatter(x=up_series.index, y=up_series.values, mode="lines", name="Uploaded",
                              line=dict(color="#f59e0b", width=2, dash="dot")))

fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420, yaxis_title=field.title())
st.plotly_chart(fig, use_container_width=True)

if overlay_years > 0:
    st.subheader("Multi-year seasonal overlay")
    st.caption(
        "Same delivery-month legs, shifted by whole years, aligned on trading days since each "
        "series' first session so seasonally-equivalent contracts line up on one x-axis."
    )
    series_by_year = {}
    for delta in range(-overlay_years, overlay_years + 1):
        code = instrument if delta == 0 else shift_instrument_year(instrument, delta)
        candidate = load_ohlc_for_instrument(code)
        if candidate.empty:
            continue
        s = candidate.set_index("Date")[field].dropna()
        year_label = s.index.min().year if not s.empty else None
        if year_label is not None:
            series_by_year[year_label] = s

    if len(series_by_year) <= 1:
        st.info("No comparable seasonal contracts found in other years for this instrument.")
    else:
        overlay_df = align_multi_year(series_by_year)
        overlay_fig = go.Figure()
        for col in overlay_df.columns:
            is_current = col == str(base["Date"].min().year)
            overlay_fig.add_trace(go.Scatter(
                x=overlay_df.index, y=overlay_df[col], mode="lines", name=col,
                line=dict(width=3 if is_current else 1.5),
            ))
        overlay_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10), height=420,
            xaxis_title="Trading days since first session", yaxis_title=field.title(),
        )
        st.plotly_chart(overlay_fig, use_container_width=True)
