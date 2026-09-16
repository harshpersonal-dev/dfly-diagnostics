"""

Dfly Diagnostics App

=====================

Pick a Dfly (generic number) and a date range (preset or custom), and get

the ADF statistic, half-life, quality score, median/max heat, stop-out

rate and reward-to-risk for that exact slice of history.
 
Run:

    streamlit run app.py

"""
 
import streamlit as st

import pandas as pd

import numpy as np

import datetime as dt

from diagnostics import (

    full_diagnostics,

    to_correlation_input,

    compute_correlation_matrix,

    compute_rolling_correlation,

    rank_correlations,

)
 
st.set_page_config(page_title="Dfly Diagnostics", layout="wide")
 
DATA_CSV = "dflies_data.csv"
 
 
@st.cache_data

def load_data(path):

    df = pd.read_csv(path, parse_dates=["Date"])

    df = df.sort_values("Date").set_index("Date")

    return df
 
 
def _corr_color(val):

    """Red-blue diverging background color for a correlation value.

    Pure Python/pandas - no matplotlib dependency required."""

    if pd.isna(val):

        return ""

    v = max(-1.0, min(1.0, val))

    if v >= 0:

        intensity = v

        r, g, b = 255, int(255 * (1 - intensity)), int(255 * (1 - intensity))

    else:

        intensity = -v

        r, g, b = int(255 * (1 - intensity)), int(255 * (1 - intensity)), 255

    return f"background-color: rgb({r},{g},{b})"
 
 
df = load_data(DATA_CSV)

min_date, max_date = df.index.min().date(), df.index.max().date()
 
st.title("Dfly Diagnostics")

st.caption(f"Data loaded: {min_date} to {max_date}  |  {len(df)} rows  |  {len(df.columns)} Dflies")
 
# ----------------------------- SIDEBAR ---------------------------------

st.sidebar.header("Selection")
 
dfly_choice = st.sidebar.selectbox(

    "Generic Dfly number", options=list(df.columns), key="dfly_choice"

)
 
duration_preset = st.sidebar.selectbox(

    "Duration",

    ["Custom range", "20 days", "1 month", "3 months", "6 months", "1 year", "Full history"],

    key="duration_preset",

)
 
if duration_preset == "Custom range":

    start_date = st.sidebar.date_input(

        "Start date",

        value=max_date - dt.timedelta(days=180),

        min_value=min_date,

        max_value=max_date,

        key="start_date_custom",

    )

    end_date = st.sidebar.date_input(

        "End date",

        value=max_date,

        min_value=min_date,

        max_value=max_date,

        key="end_date_custom",

    )

    if start_date > end_date:

        st.sidebar.error("Start date must be on or before end date.")

        st.stop()

elif duration_preset == "Full history":

    start_date, end_date = min_date, max_date

else:

    end_date = st.sidebar.date_input(

        "End date", value=max_date, min_value=min_date, max_value=max_date, key="end_date_preset"

    )

    days_map = {"20 days": 20, "1 month": 30, "3 months": 91, "6 months": 182, "1 year": 365}

    start_date = end_date - dt.timedelta(days=days_map[duration_preset])

    if start_date < min_date:

        start_date = min_date

    st.sidebar.caption(f"Start date: {start_date}")
 
st.sidebar.header("Trade parameters")

tick_size = st.sidebar.number_input(

    "Tick size (price units)", value=0.01, format="%.4f", key="tick_size"

)

tick_value = st.sidebar.number_input("Value per tick ($)", value=10.0, key="tick_value")

stop_ticks = st.sidebar.number_input("Stop distance (ticks)", value=2, step=1, key="stop_ticks")

stop_distance = tick_size * stop_ticks

dollar_risk = tick_value * stop_ticks

st.sidebar.caption(f"Stop distance = {stop_distance:.4f}  |  Dollar risk = ${dollar_risk:.2f}")
 
entry_z = st.sidebar.slider("Entry z-score threshold", 0.5, 3.0, 1.0, 0.1, key="entry_z")

lookback = st.sidebar.slider("Rolling lookback (days)", 5, 60, 20, 1, key="lookback")

adf_regression = st.sidebar.selectbox(

    "ADF regression type", ["c", "ct", "n"],

    format_func=lambda x: {"c": "constant (revert to fixed mean)",

                            "ct": "constant + trend",

                            "n": "none"}[x],

    key="adf_regression",

)
 
st.sidebar.header("Correlations")

show_correlations = st.sidebar.radio(

    "Show Correlations", ["Off", "On"], index=0, key="show_correlations"

) == "On"
 
if show_correlations:

    corr_dflies = st.sidebar.multiselect(

        "Instruments to include (outright / spread / fly / dfly)",

        options=list(df.columns),

        default=list(df.columns),

        key="corr_dflies",

    )

    corr_basis = st.sidebar.radio(

        "Correlate on", ["Daily changes", "Levels"], index=0,

        key="corr_basis",

        help="Daily changes avoids inflated correlation between instruments that share legs "

             "(e.g. CO1-2-3-4 and CO2-3-4-5 share 3 contracts). Use Levels only if the "

             "raw co-movement itself is what you want to see.",

    )

    corr_method = st.sidebar.selectbox(

        "Correlation method", ["pearson", "spearman", "kendall"],

        format_func=lambda x: x.capitalize(),

        key="corr_method",

    )

    corr_window = st.sidebar.slider("Rolling window (days)", 5, 120, 20, 1, key="corr_window")

    filter_threshold = st.sidebar.checkbox(

        "Show only |correlation| above threshold", key="filter_threshold"

    )

    corr_threshold = st.sidebar.slider(

        "Threshold", 0.0, 1.0, 0.5, 0.05, disabled=not filter_threshold, key="corr_threshold"

    )
 
# ----------------------------- MAIN ---------------------------------

mask = (df.index.date >= start_date) & (df.index.date <= end_date)

series = df.loc[mask, dfly_choice]
 
st.subheader(f"{dfly_choice}  |  {start_date} to {end_date}  ({len(series)} obs)")
 
if len(series.dropna()) < 15:

    st.error("Not enough observations in this range to run diagnostics (need at least ~15-30).")

    st.stop()
 
st.line_chart(series, height=280)
 
results = full_diagnostics(

    series, entry_z=entry_z, lookback=lookback,

    stop_distance=stop_distance, adf_regression=adf_regression,

)
 
col1, col2, col3, col4 = st.columns(4)
 
with col1:

    st.metric("ADF statistic", f"{results['adf_stat']:.4f}" if pd.notna(results["adf_stat"]) else "n/a")

    st.metric("ADF p-value", f"{results['adf_pvalue']:.4f}" if pd.notna(results["adf_pvalue"]) else "n/a")

    verdict = "STATIONARY" if results.get("stationary") else "NON-STATIONARY"

    st.markdown(f"**Verdict:** {verdict}")
 
with col2:

    st.metric("Half-life (periods)", f"{results['half_life']:.2f}" if pd.notna(results["half_life"]) else "n/a")

    st.metric("Quality score (theta/sigma)", f"{results['quality_score']:.3f}" if pd.notna(results["quality_score"]) else "n/a")
 
with col3:

    st.metric("Median heat", f"{results['median_heat']:.4f}" if pd.notna(results["median_heat"]) else "n/a")

    st.metric("Max heat", f"{results['max_heat']:.4f}" if pd.notna(results["max_heat"]) else "n/a")
 
with col4:

    st.metric("Stop-out rate", f"{results['stop_out_rate_pct']:.1f}%" if pd.notna(results["stop_out_rate_pct"]) else "n/a")

    st.metric("Reward:Risk", f"{results['reward_to_risk']:.2f}" if pd.notna(results["reward_to_risk"]) else "n/a")
 
with st.expander("Full raw output"):

    st.json({k: (round(v, 5) if isinstance(v, float) else v) for k, v in results.items()})
 
with st.expander("ADF critical values"):

    st.write({

        "1%": results.get("crit_1pct"),

        "5%": results.get("crit_5pct"),

        "10%": results.get("crit_10pct"),

        "lags used": results.get("lags_used"),

        "n_obs used": results.get("n_obs"),

    })
 
st.caption(

    f"n_entries used for heat calc: {results.get('n_entries', 0)}  |  "

    "Heat/entries use a rolling lookback (no look-ahead) -- see the reference doc for details."

)
 
# ----------------------------- CORRELATIONS ---------------------------------

if show_correlations:

    st.divider()

    st.subheader(f"Correlations  |  {start_date} to {end_date}")
 
    if len(corr_dflies) < 2:

        st.warning("Select at least 2 instruments to compute correlations.")

    else:

        raw_slice = df.loc[mask, corr_dflies]

        basis_key = "changes" if corr_basis == "Daily changes" else "levels"

        corr_input = to_correlation_input(raw_slice, on=basis_key)
 
        if len(corr_input.dropna(how="all")) < 10:

            st.warning("Not enough overlapping observations in this range for reliable correlations.")

        else:

            corr_matrix = compute_correlation_matrix(corr_input, method=corr_method, min_periods=10)
 
            st.markdown(f"**Correlation matrix** ({corr_basis.lower()}, {corr_method})")

            styled = (

                corr_matrix.style

                .map(_corr_color)

                .format("{:.2f}", na_rep="—")

            )

            st.dataframe(styled, use_container_width=True)
 
            ranked = rank_correlations(corr_matrix, df_slice=corr_input, top_n=10)

            if filter_threshold:

                ranked = ranked[ranked["abs_correlation"] >= corr_threshold]
 
            st.markdown("**Top correlated pairs**")

            if ranked.empty:

                st.caption("No pairs meet the current threshold.")

            else:

                st.dataframe(

                    ranked.style.format({"correlation": "{:.3f}", "abs_correlation": "{:.3f}"}),

                    use_container_width=True,

                )

                st.caption("n_obs = overlapping non-NaN observations used for that pair — "

                           "treat correlations backed by few points with caution.")
 
        # --- Pick any 2 instruments + rolling window (core requested feature) ---

        # Uses corr_dflies directly rather than requiring the matrix step above,

        # so this works even if the matrix was skipped for insufficient data,

        # as long as the specific pair chosen here has enough overlap.

        st.markdown("### Pairwise rolling correlation")

        roll_a = st.selectbox("Instrument A", options=corr_dflies, key="roll_a")

        roll_b = st.selectbox(

            "Instrument B",

            options=[c for c in corr_dflies if c != roll_a] or corr_dflies,

            key="roll_b",

        )

        if roll_a == roll_b:

            st.caption("Choose two different instruments to compare.")

        else:

            # Rolling correlation should start plotting exactly at start_date,
            # using the corr_window days *before* start_date as the lookback
            # for that first point -- not eating the first corr_window days
            # of the visible window before showing anything. Pull a buffer
            # of full-history rows before start_date (nothing after
            # end_date, so still no look-ahead), compute the rolling
            # correlation across buffer+window, then trim the displayed
            # series back down to [start_date, end_date].

            full_idx = df.index

            in_range = (full_idx.date >= start_date) & (full_idx.date <= end_date)

            if not in_range.any():

                st.warning("No data in the selected date range for this pair.")

            else:

                start_pos = int(np.argmax(in_range))

                end_pos = len(full_idx) - 1 - int(np.argmax(in_range[::-1]))

                ext_start_pos = max(0, start_pos - corr_window)
 
                extended_slice = df.iloc[ext_start_pos:end_pos + 1][[roll_a, roll_b]]

                extended_input = to_correlation_input(extended_slice, on=basis_key)
 
                # informational overlap count, computed on the VISIBLE window only,
                # so the caption reflects what's on screen (same as before)

                visible_input = to_correlation_input(df.loc[mask, [roll_a, roll_b]], on=basis_key)

                pair_overlap = visible_input.dropna().shape[0]
 
                if extended_input[[roll_a, roll_b]].dropna().shape[0] < corr_window:

                    st.warning(

                        f"Not enough combined history (including the {corr_window}-day lookback "

                        f"buffer before {start_date}) to compute this rolling correlation -- likely "

                        f"because {start_date} is too close to the start of the dataset "

                        f"({min_date}). Try a later start date or a smaller rolling window."

                    )

                else:

                    roll_series_full = compute_rolling_correlation(

                        extended_input[roll_a], extended_input[roll_b], window=corr_window,

                    )

                    roll_series = roll_series_full[

                        (roll_series_full.index.date >= start_date)

                        & (roll_series_full.index.date <= end_date)

                    ]

                    st.line_chart(roll_series, height=280)

                    latest = roll_series.dropna()

                    latest_val = f"{latest.iloc[-1]:.3f}" if not latest.empty else "n/a"

                    st.caption(

                        f"{corr_window}-day rolling pearson correlation of {corr_basis.lower()} "

                        f"between {roll_a} and {roll_b}  |  latest value: {latest_val}  |  "

                        f"{pair_overlap} overlapping obs in range  |  "

                        f"chart starts exactly at {start_date}, using the {corr_window} days "

                        f"before it as lookback."

                    )
 
