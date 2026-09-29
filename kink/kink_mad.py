"""Kink analysis, pure numpy/pandas (no Streamlit).

Method A  (mad_analysis)      - second difference of the dfly strip, flagged by
                                 a MAD-robust cutoff and/or a fixed tick threshold,
                                 plus sign-change turning points.
Method B  (normalized_kink)   - each historical day's strip is z-normalised across
                                 contracts; today's normalised strip is compared with
                                 each contract's historical normalised mean/SD
                                 (kink z-score), plus a pooled quadratic-fit residual.
combine()                     - joins both per contract and grades the confluence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- Method A
@dataclass
class MadParams:
    tradable_from: int = 1        # strip position (1-indexed) where the tradable zone starts
    fixed_thresh: float = 5.0     # flag |2nd diff| > this many ticks
    borderline_at: float = 5.0    # |2nd diff| == this (and nothing else flagged) -> borderline
    mad_k: float = 3.0            # MAD cutoff = mad_k * MAD
    mad_on: str = "full"          # "full" strip or "tradable" zone only
    min_turn_ticks: float = 2.0   # sign-change turns with amplitude <= this are noise
    count_zeros: bool = False     # count zero plateaus as turning points


def short_code(c) -> str:
    s = str(c).strip().upper()
    return s[2:] if s.startswith("CO") else s


def second_diff(v) -> np.ndarray:
    v = np.asarray(v, float)
    d2 = np.full(len(v), np.nan)
    if len(v) >= 3:
        d2[1:-1] = v[:-2] - 2 * v[1:-1] + v[2:]
    return d2


def _mad(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def mad_analysis(codes, ticks, p: MadParams | None = None):
    """Returns (table, info, turning_points)."""
    p = p or MadParams()
    ticks = np.asarray(ticks, float)
    n = len(ticks)
    short = [short_code(c) for c in codes]
    d2 = second_diff(ticks)
    pos = np.arange(1, n + 1)
    trad = pos >= p.tradable_from

    mad_full, mad_trad = _mad(d2), _mad(d2[trad])
    mad = mad_full if p.mad_on == "full" else mad_trad
    cutoff = p.mad_k * mad if np.isfinite(mad) else np.nan
    # A zero MAD (e.g. mostly-flat strip) would flag every non-zero 2nd diff;
    # in that case the MAD rule is switched off and the fixed rule still applies.
    mad_usable = bool(np.isfinite(cutoff) and cutoff > 0)

    with np.errstate(invalid="ignore"):
        ad = np.abs(d2)
        mad_flag = (ad > cutoff) if mad_usable else np.zeros(n, bool)
        fixed_flag = ad > p.fixed_thresh
        borderline = ~mad_flag & ~fixed_flag & (ad == p.borderline_at)
    mad_flag &= ~np.isnan(d2)
    fixed_flag &= ~np.isnan(d2)

    def label(i):
        f = []
        if mad_flag[i]:
            f.append("MAD")
        if fixed_flag[i]:
            f.append(f">{p.fixed_thresh:g}")
        if not f and borderline[i]:
            f.append("borderline")
        return ", ".join(f)

    table = pd.DataFrame({
        "Pos": pos, "Contract": short, "Dfly": ticks, "2nd diff": d2,
        "MAD": mad_flag, "Fixed": fixed_flag, "Borderline": borderline,
        "Flag": [label(i) for i in range(n)], "Tradable": trad,
    })

    turns = []
    sgn = np.sign(np.diff(ticks))
    for i in range(1, n - 1):
        a, b = sgn[i - 1], sgn[i]
        if a * b < 0:
            amp = min(abs(ticks[i] - ticks[i - 1]), abs(ticks[i] - ticks[i + 1]))
            turns.append((i + 1, short[i], "peak" if a > 0 else "trough", amp,
                          "noise" if amp <= p.min_turn_ticks else "real"))
        elif p.count_zeros and a * b == 0 and (a == 0) != (b == 0):
            turns.append((i + 1, short[i], "plateau edge", 0.0, "noise"))
    tp = pd.DataFrame(turns, columns=["Pos", "Contract", "Type", "Amp (ticks)", "Quality"])
    tp = tp[tp.Pos >= p.tradable_from].reset_index(drop=True)

    info = {"mad_full": mad_full, "mad_trad": mad_trad, "mad_used": mad,
            "cutoff": cutoff, "mad_usable": mad_usable}
    return table, info, tp


# ----------------------------------------------------------------- Method B
def _hist_code(c) -> str:
    """'CO H27 Dfly' -> 'COH27'"""
    return re.sub(r"DFLY", "", re.sub(r"\s+", "", str(c).upper()))


def load_history(src) -> pd.DataFrame:
    """Historical dfly panel (Date + one column per physical contract), price units."""
    df = pd.read_excel(src)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = df.columns[0]
    out = df.drop(columns=[date_col]).apply(pd.to_numeric, errors="coerce")
    out.columns = [_hist_code(c) for c in out.columns]
    out.index = pd.to_datetime(df[date_col], errors="coerce")
    out = out[out.index.notna()].dropna(how="all").sort_index()
    return out


def normalized_kink(hist: pd.DataFrame, current: pd.Series, min_rows: int = 5):
    """current: price-unit dfly values indexed by contract code (e.g. COH27).
    Only contracts present in BOTH history and current (non-NaN) are used, so the
    normalisation set is identical for history and today. Returns (table, info)."""
    cols = [c for c in current.index if c in hist.columns and pd.notna(current[c])]
    info = {"n_contracts": len(cols), "n_rows": 0, "ok": False, "used": cols}
    if len(cols) < 4:
        info["reason"] = "fewer than 4 contracts overlap between live strip and history"
        return pd.DataFrame(), info
    h = hist[cols].dropna(how="all")
    info["n_rows"] = len(h)
    if len(h) < min_rows:
        info["reason"] = f"only {len(h)} historical rows (need >= {min_rows})"
        return pd.DataFrame(), info

    mean_d = h.mean(axis=1)
    sd_d = h.std(axis=1, ddof=1).replace(0, np.nan)
    norm = h.sub(mean_d, axis=0).div(sd_d, axis=0)
    hm, hs = norm.mean(axis=0), norm.std(axis=0, ddof=1).replace(0, np.nan)

    cur = current[cols].astype(float)
    cur_sd = cur.std(ddof=1)
    if not cur_sd or not np.isfinite(cur_sd):
        info["reason"] = "current strip has zero dispersion"
        return pd.DataFrame(), info
    cn = (cur - cur.mean()) / cur_sd
    z = (cn - hm) / hs

    # pooled quadratic fit y = b0 + b1 x + b2 x^2, x = position within `cols`
    x = np.tile(np.arange(1, len(cols) + 1), len(h))
    y = h.to_numpy().reshape(-1)
    ok = np.isfinite(y)
    X = np.column_stack([np.ones(ok.sum()), x[ok], x[ok] ** 2])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    xs = np.arange(1, len(cols) + 1)
    pred = beta[0] + beta[1] * xs + beta[2] * xs ** 2

    table = pd.DataFrame({
        "Contract": cols, "Current": cur.values, "Cur_Norm": cn.values,
        "Hist_Norm_Mean": hm.values, "Hist_Norm_SD": hs.values, "Kink_Z": z.values,
        "Quad_Resid": cur.values - pred,
    })
    info.update(ok=True, cur_mean=float(cur.mean()), cur_sd=float(cur_sd),
                beta=[float(b) for b in beta])
    return table, info


# ---------------------------------------------------------------- combiner
def combine(mad_table: pd.DataFrame, z_table: pd.DataFrame, z_thr: float = 2.0,
            tick: float = 0.01) -> pd.DataFrame:
    """One row per strip contract with both methods and a confluence grade.
    Direction: 2nd diff > 0 = dfly sits BELOW its neighbours (dip / cheap);
               kink z > 0 = dfly sits ABOVE its historical shape (rich)."""
    df = mad_table.copy()
    if z_table is not None and len(z_table):
        zt = z_table[["Contract", "Kink_Z", "Quad_Resid"]].copy()
        zt["Contract"] = zt["Contract"].map(short_code)     # 'COH27' -> 'H27' to match Method A
        zt["Quad_Resid"] = zt["Quad_Resid"] / tick          # -> ticks
        df = df.merge(zt, on="Contract", how="left")
    else:
        df["Kink_Z"], df["Quad_Resid"] = np.nan, np.nan

    a = df["MAD"] | df["Fixed"]
    b = df["Kink_Z"].abs() >= z_thr
    df["Signal A"] = a
    df["Signal B"] = b.fillna(False)
    df["Confluence"] = np.select(
        [a & df["Signal B"], a, df["Signal B"]],
        ["CONFIRMED", "A only", "B only"], default="")
    d2, z = df["2nd diff"], df["Kink_Z"]
    df["A says"] = np.where(a, np.where(d2 > 0, "cheap (dip)", "rich (spike)"), "")
    df["B says"] = np.where(df["Signal B"], np.where(z > 0, "rich", "cheap"), "")
    both = a & df["Signal B"]
    agree = ((d2 > 0) & (z < 0)) | ((d2 < 0) & (z > 0))
    df["Directions agree"] = np.where(both, np.where(agree, "yes", "NO"), "")
    return df
