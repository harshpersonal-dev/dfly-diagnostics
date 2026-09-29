"""Builds the live 1MDF strip from outright contract codes.

A 1-month double fly named after its FIRST leg, e.g. COH27 dfly, is
    P[H27] - 3*P[J27] + 3*P[K27] - P[M27]
(verified against the CurveTerminal workbook: '1MDF close' M1 equals
[1,-3,3,-1] over Curve close M1..M4 to floating-point precision).
Legs are assembled from consecutive outright codes; nothing is
auto-matched to QH products or roll-adjusted - the leg list is shown
in the UI so it can be confirmed by eye.
"""
from __future__ import annotations

import re

import numpy as np

MONTHS = "FGHJKMNQUVXZ"
WEIGHTS = (1.0, -3.0, 3.0, -1.0)
_CODE = re.compile(r"^([A-Z]{2})([FGHJKMNQUVXZ])(\d{2})$")


def parse_code(code: str) -> tuple[str, int, int]:
    m = _CODE.match(code.strip().upper())
    if not m:
        raise ValueError(f"'{code}' is not a valid contract code like COH27")
    return m.group(1), MONTHS.index(m.group(2)), int(m.group(3))


def consecutive_codes(start: str, n: int) -> list[str]:
    prefix, mi, yy = parse_code(start)
    out = []
    for _ in range(n):
        out.append(f"{prefix}{MONTHS[mi]}{yy:02d}")
        mi += 1
        if mi == 12:
            mi, yy = 0, (yy + 1) % 100
    return out


def build_strip(start: str, n_dflies: int) -> tuple[list[str], list[str]]:
    """Returns (dfly_names, leg_codes). dfly i uses leg_codes[i:i+4]."""
    legs = consecutive_codes(start, n_dflies + 3)
    return legs[:n_dflies], legs


def dfly_values(prices: dict[str, float | None], legs: list[str]) -> np.ndarray:
    """Dfly value per strip position; NaN if any of its 4 legs is missing/stale."""
    out = np.full(len(legs) - 3, np.nan)
    for i in range(len(out)):
        p = [prices.get(c) for c in legs[i:i + 4]]
        if all(x is not None and np.isfinite(x) for x in p):
            out[i] = sum(w * x for w, x in zip(WEIGHTS, p))
    return out


def legs_of(legs: list[str], i: int) -> list[str]:
    return legs[i:i + 4]
