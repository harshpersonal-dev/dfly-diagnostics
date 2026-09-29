"""Period-wise percentile seasonality. All maths lives in curves/seasonality.py (unchanged);
this page is only the UI around it."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import theme
from curves import data_loader as dl
from curves import seasonality as sz

st.title("Seasonality & Percentile")

if not dl.data_available():
    st.error("Prepared data not found. Put the CurveTerminal workbook in `data/` and run "
             "`python prepare_curve_data.py` once (writes `data/prepared/`).")
    st.stop()

structs = dl.available_structures()
st.sidebar.header("Selection")
structure = st.sidebar.selectbox("Structure", structs, index=structs.index("1MDF") if "1MDF" in structs else 0,
                                 format_func=lambda s: f"{s} — {dl.STRUCTURE_LABELS.get(s, s)}", key="sz_struct")
df = dl.load_structure(structure, "close")
if df.empty:
    st.error(f"No data for {structure}.")
    st.stop()
positions = sorted(int(c[1:]) for c in df.columns if c.startswith("M"))
pos = st.sidebar.selectbox("Generic position (M#)", positions, key="sz_pos",
                           index=positions.index(7) if 7 in positions else 0)
dates = df.index
end_date = pd.Timestamp(st.sidebar.date_input("As-of date", value=dates.max().date(),
                                              min_value=dates.min().date(), max_value=dates.max().date(),
                                              key="sz_end"))
end_date = dl.nearest_date(end_date, dates)
window = st.sidebar.selectbox("Lookback window", list(sz.WINDOW_DAYS), index=len(sz.WINDOW_DAYS) - 1, key="sz_win")
tol = st.sidebar.slider("Reversion: magnitude tolerance", 0.05, 0.75, 0.25, 0.05, key="sz_tol")

series = sz.get_structure_series(df, pos)
if end_date not in series.index:
    st.warning(f"M{pos} has no value on {end_date.date()}. Pick another date or position.")
    st.stop()

res = sz.seasonal_percentile(series, end_date, window)
c = st.columns(4)
c[0].metric(f"{structure} M{pos} on {end_date.date()}", f"{res['current_value']:.4f}")
c[1].metric("Seasonal percentile", f"{res['percentile']:.1f}" if pd.notna(res["percentile"]) else "n/a")
c[2].metric("Obs in seasonal band", f"{res['n_obs']}")
c[3].metric("Years covered", f"{res['n_years']}")
st.caption("Percentile = rank of today's value vs. the same time of year (±15 calendar days) across the lookback window.")

hist = sz.windowed_history(series, end_date, window)
doy = end_date.dayofyear
band = hist[np.abs((hist.index.dayofyear - doy + 183) % 366 - 183) <= 15]

t1, t2, t3, t4 = st.tabs(["History", "Seasonal band", "Magnitude", "Subsequent reversion"])
with t1:
    fig = go.Figure(go.Scatter(x=hist.index, y=hist.values, mode="lines", line=dict(color="#58A6FF", width=1.5), name=f"M{pos}"))
    fig.add_trace(go.Scatter(x=band.index, y=band.values, mode="markers", marker=dict(color=theme.WARN, size=4), name="seasonal band"))
    fig.add_trace(go.Scatter(x=[end_date], y=[res["current_value"]], mode="markers",
                             marker=dict(color=theme.ACCENT, size=11, symbol="diamond"), name="as-of"))
    st.plotly_chart(theme.style_fig(fig), width="stretch")
with t2:
    if band.empty:
        st.info("No observations in the seasonal band for this window.")
    else:
        fig = go.Figure(go.Histogram(x=band.values, nbinsx=30, marker_color="#58A6FF", name="band values"))
        fig.add_vline(x=res["current_value"], line_color=theme.ACCENT, line_width=2,
                      annotation_text=f"now  ({res['percentile']:.0f}th pct)")
        st.plotly_chart(theme.style_fig(fig, 360), width="stretch")
        by_year = band.groupby(band.index.year).agg(["count", "mean", "min", "max"]).round(4)
        by_year.index.name = "year"
        st.dataframe(by_year, width="stretch")
with t3:
    mag = sz.magnitude_distribution(series, end_date, window)
    if mag.empty:
        st.info("No data.")
    else:
        fig = go.Figure(go.Histogram(x=mag.values, nbinsx=40, marker_color="#A371F7"))
        fig.add_vline(x=abs(res["current_value"]), line_color=theme.ACCENT, line_width=2, annotation_text="now")
        st.plotly_chart(theme.style_fig(fig, 340), width="stretch")
        st.dataframe(mag.describe().round(4).to_frame("|value|"), width="stretch")
with t4:
    rev = sz.subsequent_reversion(series, end_date, window, magnitude_tolerance=tol)
    if not rev["n_episodes"]:
        st.info("No comparable past episodes (or not enough forward history).")
    else:
        st.write(f"**{rev['n_episodes']}** past episodes with |value| within ±{tol:.0%} of today's.")
        st.dataframe(pd.DataFrame(rev["horizons"]).T.rename_axis("horizon (days)").round(4), width="stretch")
        with st.expander("Episodes"):
            st.dataframe(rev["episodes"].round(4), width="stretch", hide_index=True)
