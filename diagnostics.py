"""
Core diagnostics: ADF stationarity test, half-life, quality score,
and heat (adverse excursion) with a fixed dollar-stop translation.
Same math as dfly_plain.py, packaged as importable functions for the app.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from itertools import combinations


def run_adf(x, regression="c"):
    x = x.dropna()
    if len(x) < 15:
        return {"adf_stat": np.nan, "adf_pvalue": np.nan, "stationary": None, "n_obs": len(x)}
    stat, pval, lags, nobs, crit, _ = adfuller(x, regression=regression, autolag="AIC", result_object=False)
    return {
        "adf_stat": stat,
        "adf_pvalue": pval,
        "stationary": pval < 0.05,
        "n_obs": nobs,
        "lags_used": lags,
        "crit_1pct": crit["1%"],
        "crit_5pct": crit["5%"],
        "crit_10pct": crit["10%"],
    }


def half_life_quality(x):
    """OU-style fit: dx = a + b*x_lag + e
    theta = -b (pull strength), sigma = std of residuals (unexplained noise)
    half_life = ln(2)/theta, quality_score = theta/sigma
    """
    x = x.dropna()
    if len(x) < 10:
        return {"half_life": np.nan, "theta": np.nan, "sigma": np.nan, "quality_score": np.nan}
    dx = x.diff().dropna()
    x_lag = x.shift(1).dropna().loc[dx.index]
    X = np.column_stack([np.ones(len(x_lag)), x_lag.values])
    beta, *_ = np.linalg.lstsq(X, dx.values, rcond=None)
    theta = -beta[1]
    sigma = (dx.values - X @ beta).std()
    if theta <= 0 or sigma == 0:
        return {"half_life": np.nan, "theta": theta, "sigma": sigma, "quality_score": np.nan}
    return {
        "half_life": np.log(2) / theta,
        "theta": theta,
        "sigma": sigma,
        "quality_score": theta / sigma,
    }


def heat_stats(x, entry_z=1.0, lookback=20, stop_distance=0.02):
    """Adverse excursion using a ROLLING mean/std (no look-ahead): on any
    given day, the entry/exit reference only reflects the trailing
    `lookback` days, not the whole selected period.
    """
    x = x.dropna()
    if len(x) < lookback + 10:
        return {
            "median_heat": np.nan, "max_heat": np.nan,
            "stop_out_rate_pct": np.nan, "avg_reversion_size": np.nan,
            "reward_to_risk": np.nan, "n_entries": 0,
        }
    roll_mean = x.rolling(lookback).mean()
    roll_std = x.rolling(lookback).std()
    z = (x - roll_mean) / roll_std

    excursions, reversion_sizes = [], []
    in_trade, direction, worst, entry_price = False, None, None, None
    for t in x.index:
        zt = z.loc[t]
        if pd.isna(zt):
            continue
        if not in_trade and abs(zt) >= entry_z:
            in_trade, direction = True, np.sign(zt)
            entry_price = worst = x.loc[t]
        elif in_trade:
            price = x.loc[t]
            worst = max(worst, price) if direction > 0 else min(worst, price)
            ref_mean = roll_mean.loc[t]
            crossed_back = (direction > 0 and price <= ref_mean) or (direction < 0 and price >= ref_mean)
            if crossed_back:
                excursions.append(abs(worst - entry_price))
                reversion_sizes.append(abs(entry_price - price))
                in_trade = False

    if not excursions:
        return {
            "median_heat": np.nan, "max_heat": np.nan,
            "stop_out_rate_pct": np.nan, "avg_reversion_size": np.nan,
            "reward_to_risk": np.nan, "n_entries": 0,
        }

    ex = pd.Series(excursions)
    avg_reversion = pd.Series(reversion_sizes).mean()
    return {
        "median_heat": ex.median(),
        "max_heat": ex.max(),
        "stop_out_rate_pct": (ex > stop_distance).mean() * 100,
        "avg_reversion_size": avg_reversion,
        "reward_to_risk": avg_reversion / stop_distance if stop_distance else np.nan,
        "n_entries": len(excursions),
    }


def full_diagnostics(series, entry_z=1.0, lookback=20, stop_distance=0.02, adf_regression="c"):
    """Runs all diagnostics on one Dfly series slice and returns one flat dict."""
    out = {}
    out.update(run_adf(series, regression=adf_regression))
    out.update(half_life_quality(series))
    out.update(heat_stats(series, entry_z=entry_z, lookback=lookback, stop_distance=stop_distance))
    return out
def count_potential_trades(x, entry_z=1.0, lookback=20):
    """Counts how many distinct trade entries would have triggered
    (|z| >= entry_z on the same rolling mean/std used by heat_stats),
    using the SAME state machine as heat_stats: once in a trade, no new
    entry is counted until price crosses back through the rolling mean.

    Unlike heat_stats' n_entries (which only counts entries that
    completed a round trip before the window ended), this counts EVERY
    entry that occurred, including one still open at the end of the
    window -- i.e. "how many trades existed / could have been entered"
    in this window, not just how many finished reverting.

    Does not modify or call heat_stats; fully independent, read-only
    pass over the same series.
    """
    x = x.dropna()
    if len(x) < lookback + 10:
        return 0
    roll_mean = x.rolling(lookback).mean()
    roll_std = x.rolling(lookback).std()
    z = (x - roll_mean) / roll_std

    count = 0
    in_trade, direction = False, None
    for t in x.index:
        zt = z.loc[t]
        if pd.isna(zt):
            continue
        if not in_trade and abs(zt) >= entry_z:
            in_trade, direction = True, np.sign(zt)
            count += 1
        elif in_trade:
            price = x.loc[t]
            ref_mean = roll_mean.loc[t]
            crossed_back = (direction > 0 and price <= ref_mean) or (direction < 0 and price >= ref_mean)
            if crossed_back:
                in_trade = False
    return count


def to_correlation_input(df_slice: pd.DataFrame, on: str = "changes") -> pd.DataFrame:
    """Transforms a Dfly panel for correlation analysis.

    on="changes" (default): first-differences each column. Use this for
    the correlation matrix/ranking — Dflies sharing legs (e.g.
    CO1-2-3-4 vs CO2-3-4-5) will show inflated level correlation that
    reflects shared contracts, not an independent relationship.
    on="levels": returns df_slice unchanged, for cases where the raw
    co-movement of levels is explicitly what's wanted.
    """
    if on == "changes":
        return df_slice.diff().iloc[1:]
    if on == "levels":
        return df_slice
    raise ValueError(f"on must be 'changes' or 'levels', got {on!r}")


def compute_correlation_matrix(
    df_slice: pd.DataFrame, method: str = "pearson", min_periods: int = 10
) -> pd.DataFrame:
    """Pairwise correlation matrix across all columns in df_slice.

    method: "pearson", "spearman", or "kendall".
    min_periods: minimum overlapping observations required for a pair;
    pairs with fewer become NaN rather than a correlation computed on
    too few points.
    """
    return df_slice.corr(method=method, min_periods=min_periods)


def compute_rolling_correlation(
    series_a: pd.Series, series_b: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    """Rolling pairwise correlation between two aligned series.

    No look-ahead: each point uses only the trailing `window`
    observations. min_periods defaults to the full window (no partial
    windows) unless explicitly relaxed.
    """
    aligned_a, aligned_b = series_a.align(series_b, join="inner")
    return aligned_a.rolling(window, min_periods=min_periods or window).corr(aligned_b)


def rank_correlations(
    matrix: pd.DataFrame, df_slice: pd.DataFrame | None = None, top_n: int = 10
) -> pd.DataFrame:
    """Flattens a correlation matrix's upper triangle into ranked pairs.

    Returns columns [dfly_a, dfly_b, correlation, abs_correlation, n_obs],
    sorted by abs_correlation descending, truncated to top_n. If
    df_slice is provided, n_obs is the count of overlapping non-NaN
    observations for that pair (lets the viewer judge reliability —
    a 0.9 correlation on 12 points is not the same claim as 0.9 on 600).
    """
    cols = matrix.columns
    rows = []
    for a, b in combinations(cols, 2):
        val = matrix.loc[a, b]
        if pd.isna(val):
            continue
        n_obs = int(df_slice[[a, b]].dropna().shape[0]) if df_slice is not None else None
        rows.append({
            "dfly_a": a, "dfly_b": b,
            "correlation": val, "abs_correlation": abs(val),
            "n_obs": n_obs,
        })

    if not rows:
        return pd.DataFrame(columns=["dfly_a", "dfly_b", "correlation", "abs_correlation", "n_obs"])

    out = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return out.head(top_n)
