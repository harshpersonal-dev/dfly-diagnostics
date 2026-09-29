"""Run:  python -m pytest -q   (or: python tests/test_kink_mad.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kink import kink_mad as km  # noqa: E402
from kink.strip import build_strip, dfly_values  # noqa: E402


def test_strip_and_weights():
    names, legs = build_strip("COH27", 16)
    assert names[0] == "COH27" and names[-1] == "COM28" and legs[-1] == "COU28" and len(legs) == 19
    px = {c: 100.0 + i ** 2 for i, c in enumerate(legs)}   # quadratic -> 3rd difference is 0
    assert np.allclose(dfly_values(px, legs), 0.0)
    px["COK27"] = None
    assert np.isnan(dfly_values(px, legs)[:3]).all() and not np.isnan(dfly_values(px, legs)[3])


def test_method_a_known_values():
    raw = pd.read_excel(ROOT / "data" / "CO-1MDF.xlsx", header=None)
    t, info, _ = km.mad_analysis(raw.iloc[0].tolist(), pd.to_numeric(raw.iloc[1]).to_numpy(float))
    assert info["mad_full"] == 3.0 and info["cutoff"] == 9.0
    row = t[t.Contract == "V27"].iloc[0]
    assert row["2nd diff"] == -11 and row["MAD"] and row["Fixed"]


def test_method_b_matches_manual_formula():
    h = km.load_history(ROOT / "data" / "CO_dflies_sept_futures_curve_edited.xlsx")
    cur = pd.Series(np.linspace(-0.03, 0.05, h.shape[1]), index=h.columns)
    z, info = km.normalized_kink(h, cur)
    n = h.sub(h.mean(axis=1), axis=0).div(h.std(axis=1, ddof=1), axis=0)
    exp = ((cur - cur.mean()) / cur.std(ddof=1) - n.mean()) / n.std(ddof=1)
    assert info["ok"] and np.allclose(z.Kink_Z, exp)


if __name__ == "__main__":
    for f in (test_strip_and_weights, test_method_a_known_values, test_method_b_matches_manual_formula):
        f(); print("ok", f.__name__)
