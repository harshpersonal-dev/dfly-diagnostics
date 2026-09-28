"""
Pure, deterministic kink-detection functions on a single outright curve
snapshot. No Streamlit imports here -- these are plain numpy/pandas
functions so they're trivially unit-testable and reusable outside the
dashboard.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class KinkResult:
    positions: np.ndarray          # curve positions (1-indexed) of the curve points, length N
    prices: np.ndarray             # close price at each curve position, length N
    delta: np.ndarray              # first differences, length N-1
    second: np.ndarray             # second differences, length N-2
    kink_mask: np.ndarray          # bool, length N-2 -- True where |second| > threshold
    sign_change_mask: np.ndarray   # bool, length N-2 -- True where delta changes sign around that point
    outlier_mask: np.ndarray       # bool, length N-2 -- True where second is a MAD-robust outlier
    combined_mask: np.ndarray      # bool, length N-2 -- kink_mask AND any enabled secondary rule
    kink_positions: np.ndarray     # curve positions flagged as kinks (1-indexed, aligned to `second`)


def detect_kinks(
    curve: pd.DataFrame,
    threshold: float = 0.05,
    use_sign_change: bool = False,
    use_mad_outlier: bool = False,
    mad_k: float = 3.0,
) -> KinkResult:
    """Runs the deterministic kink rules against one curve snapshot.

    curve: DataFrame with columns [curve_position, contract, close], as
    returned by curves.data_loader.get_curve(date). Must already be
    sorted by curve_position (get_curve guarantees this).

    Primary rule: flag any point whose second difference in price
    exceeds `threshold` in absolute value -- i.e. a sharp local bend in
    the curve.

    Secondary rules (both optional, ANDed onto the primary rule when
    enabled -- a point must clear the primary threshold AND satisfy any
    enabled secondary rule to be flagged):
      - sign_change: the curve's slope (first difference) flips sign
        immediately around this point (a literal "V" or "^" kink).
      - mad_outlier: the second difference is a robust (median absolute
        deviation) outlier relative to the rest of the curve's second
        differences on this date, not just an absolute threshold.
    """
    curve = curve.dropna(subset=["close"]).sort_values("curve_position")
    positions = curve["curve_position"].to_numpy()
    prices = curve["close"].to_numpy(dtype=float)

    if len(prices) < 3:
        empty = np.array([], dtype=bool)
        return KinkResult(positions, prices, np.array([]), np.array([]), empty, empty, empty, empty, np.array([]))

    # --- Primary rule ---
    delta = np.diff(prices)              # first differences
    second = np.diff(delta)              # second differences
    kink_mask = np.abs(second) > threshold  # threshold default 0.05, sidebar slider 0.01-0.50

    # --- Optional secondary rule: slope sign flip around the point ---
    sign_change = np.sign(delta[:-1]) != np.sign(delta[1:])

    # --- Optional robust rule: MAD-based outlier among this date's second differences ---
    if len(second) > 0:
        mad = np.median(np.abs(second - np.median(second)))
        outlier = np.abs(second - np.median(second)) > mad_k * mad if mad > 0 else np.zeros_like(second, dtype=bool)
    else:
        outlier = np.array([], dtype=bool)

    combined = kink_mask.copy()
    if use_sign_change:
        combined &= sign_change
    if use_mad_outlier:
        combined &= outlier

    # `second[i]` corresponds to curve position `positions[i + 1]` (the
    # middle point of the 3-point window used for the second difference).
    kink_at = np.where(combined)[0] + 1  # position index within `positions`
    kink_positions = positions[kink_at]

    return KinkResult(
        positions=positions,
        prices=prices,
        delta=delta,
        second=second,
        kink_mask=kink_mask,
        sign_change_mask=sign_change,
        outlier_mask=outlier,
        combined_mask=combined,
        kink_positions=kink_positions,
    )


def kink_table(curve: pd.DataFrame, result: KinkResult) -> pd.DataFrame:
    """Builds a display-ready table of flagged kinks with position,
    contract code, second-difference value and absolute size."""
    if len(result.kink_positions) == 0:
        return pd.DataFrame(columns=["curve_position", "contract", "second_diff", "abs_size"])

    contract_by_pos = curve.set_index("curve_position")["contract"]
    idx = np.where(result.combined_mask)[0]
    rows = []
    for i in idx:
        pos = result.positions[i + 1]
        rows.append({
            "curve_position": int(pos),
            "contract": contract_by_pos.get(pos, None),
            "second_diff": float(result.second[i]),
            "abs_size": float(abs(result.second[i])),
        })
    return pd.DataFrame(rows).sort_values("abs_size", ascending=False).reset_index(drop=True)
