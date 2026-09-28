"""
Generic-structure seasonality & percentile analysis (View 2 of the new
dashboard). Pure pandas/numpy, no Streamlit imports -- cached wrappers
live in pages/2_Seasonality_Percentile.py.
"""

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

WINDOW_DAYS = {
    "1 year": 365,
    "2 years": 365 * 2,
    "3 years": 365 * 3,
    "5 years": 365 * 5,
    "Full history": None,
}


def get_structure_series(structure_df: pd.DataFrame, generic_position: int) -> pd.Series:
    """Extracts the column for one curve position (e.g. M7) from a wide
    structure DataFrame (index=Date, columns=M1..MN)."""
    col = f"M{generic_position}"
    if col not in structure_df.columns:
        return pd.Series(dtype=float)
    return structure_df[col].dropna()


def windowed_history(series: pd.Series, end_date: pd.Timestamp, window: str) -> pd.Series:
    """Slice of `series` ending on `end_date`, going back `window`
    (one of WINDOW_DAYS keys)."""
    series = series.loc[:end_date]
    days = WINDOW_DAYS.get(window)
    if days is None:
        return series
    start = end_date - pd.Timedelta(days=days)
    return series.loc[series.index >= start]


def seasonal_percentile(series: pd.Series, end_date: pd.Timestamp, window: str) -> dict:
    """Percentile rank of the value on `end_date` against the same
    calendar window in prior years -- i.e. compare this year's
    mid-September reading against every mid-September (+/- ~15 days)
    reading in the lookback window, not the whole year's distribution.
    This isolates "is this unusual for this time of year" from "is this
    unusual overall"."""
    hist = windowed_history(series, end_date, window)
    if end_date not in series.index or hist.empty:
        return {"current_value": np.nan, "percentile": np.nan, "n_obs": 0, "n_years": 0}

    current_value = float(series.loc[end_date])
    doy = end_date.dayofyear
    band = 15  # +/- 15 calendar days around the same day-of-year, across all included years

    def in_seasonal_band(idx: pd.DatetimeIndex) -> np.ndarray:
        diff = np.abs((idx.dayofyear - doy + 183) % 366 - 183)  # wrap-around day-of-year distance
        return diff <= band

    seasonal_hist = hist[in_seasonal_band(hist.index)]
    if seasonal_hist.empty:
        return {"current_value": current_value, "percentile": np.nan, "n_obs": 0, "n_years": 0}

    pct = percentileofscore(seasonal_hist.dropna().values, current_value, kind="mean")
    n_years = seasonal_hist.index.year.nunique()
    return {
        "current_value": current_value,
        "percentile": float(pct),
        "n_obs": int(len(seasonal_hist)),
        "n_years": int(n_years),
    }


def magnitude_distribution(series: pd.Series, end_date: pd.Timestamp, window: str) -> pd.Series:
    """Distribution of |value| ("kink magnitude") for this generic slot
    over the lookback window, for a histogram / summary stats."""
    hist = windowed_history(series, end_date, window)
    return hist.abs().dropna()


def subsequent_reversion(
    series: pd.Series, end_date: pd.Timestamp, window: str,
    magnitude_tolerance: float = 0.25, horizons=(5, 10, 20),
) -> dict:
    """Finds historical dates (within the lookback window, excluding the
    trailing `max(horizons)` days so every episode has a full forward
    path) whose |value| was within `magnitude_tolerance` (relative) of
    today's |value|, then reports the average/median forward move at
    each horizon -- "when this generic slot saw a similarly-sized kink
    before, how did it resolve over the following N days?"""
    hist = windowed_history(series, end_date, window)
    if end_date not in series.index or hist.empty:
        return {"current_value": np.nan, "n_episodes": 0, "horizons": {}}

    current_value = float(series.loc[end_date])
    current_abs = abs(current_value)
    if current_abs == 0:
        return {"current_value": current_value, "n_episodes": 0, "horizons": {}}

    max_h = max(horizons)
    full_idx = series.index
    lo, hi = current_abs * (1 - magnitude_tolerance), current_abs * (1 + magnitude_tolerance)

    episodes = []
    candidate_dates = hist.index[hist.index < end_date]
    for d in candidate_dates:
        val = hist.loc[d]
        if pd.isna(val) or not (lo <= abs(val) <= hi):
            continue
        pos = full_idx.get_loc(d)
        if pos + max_h >= len(full_idx):
            continue  # not enough forward history for this episode
        episode = {"date": d, "value": float(val)}
        for h in horizons:
            fwd_date = full_idx[pos + h]
            fwd_val = series.loc[fwd_date]
            episode[f"move_{h}d"] = float(fwd_val - val) if pd.notna(fwd_val) else np.nan
        episodes.append(episode)

    if not episodes:
        return {"current_value": current_value, "n_episodes": 0, "horizons": {}}

    ep_df = pd.DataFrame(episodes)
    horizon_stats = {}
    for h in horizons:
        col = f"move_{h}d"
        horizon_stats[h] = {
            "avg": float(ep_df[col].mean()),
            "median": float(ep_df[col].median()),
        }
    return {
        "current_value": current_value,
        "n_episodes": len(ep_df),
        "horizons": horizon_stats,
        "episodes": ep_df,
    }
