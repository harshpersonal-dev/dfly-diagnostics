"""
Specific-contract lifecycle reconstruction (View 3 of the new
dashboard). Builds a constructed structure (e.g. a 1-month double fly)
from its underlying OHLC legs -- either looked up automatically from the
workbook's Contracts mapping, or uploaded/pasted by the user.

Uploaded data is a pure in-memory transformation: callers are
responsible for storing/reading it via st.session_state (never written
to disk), per the "no permanent storage of uploads" requirement.
"""

import io
import json
import re

import pandas as pd

REQUIRED_OHLC_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]

LEG_RE = re.compile(r"([FGHJKMNQUVXZ])(\d{2})")


def parse_legs(instrument_code: str) -> list[tuple[str, int]]:
    """Parses a workbook instrument code such as "COH24-J24-K24" (a fly)
    or "COH17-J17" (a spread) into [(month_letter, 2-digit_year), ...]."""
    body = instrument_code[2:] if instrument_code.startswith("CO") else instrument_code
    return [(m.group(1), int(m.group(2))) for m in LEG_RE.finditer(body)]


def shift_instrument_year(instrument_code: str, delta_years: int) -> str:
    """Shifts every leg of an instrument code by `delta_years`, keeping
    the same month letters -- e.g. "COH24-J24-K24" shifted by -1 becomes
    "COH23-J23-K23". Used to look up the "same seasonal contract" in a
    prior/future year for multi-year overlay."""
    def _shift(match: re.Match) -> str:
        letter, yy = match.group(1), int(match.group(2))
        return f"{letter}{(yy + delta_years) % 100:02d}"
    prefix = "CO" if instrument_code.startswith("CO") else ""
    body = instrument_code[2:] if instrument_code.startswith("CO") else instrument_code
    return prefix + LEG_RE.sub(_shift, body)


def parse_uploaded_ohlc(raw_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parses an uploaded OHLC file, either JSON (list of records, in the
    style already used elsewhere in this project) or CSV with columns
    Date, Open, High, Low, Close, Volume (case-insensitive, extra columns
    ignored). Raises ValueError with a user-facing message on bad input."""
    text = raw_bytes.decode("utf-8", errors="replace")

    if filename.lower().endswith(".json"):
        try:
            records = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON: {e}")
        df = pd.DataFrame(records)
    else:
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise ValueError(f"Could not parse CSV: {e}")

    # Normalize column names case-insensitively.
    rename = {c: c.strip().title() for c in df.columns}
    df = df.rename(columns=rename)

    missing = [c for c in REQUIRED_OHLC_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Uploaded file is missing required column(s): {', '.join(missing)}. "
            f"Expected: {', '.join(REQUIRED_OHLC_COLS)}."
        )

    df["Date"] = pd.to_datetime(df["Date"])
    for c in ("Open", "High", "Low", "Close", "Volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.sort_values("Date").reset_index(drop=True)


def contracts_for_generic(contracts_row: pd.Series, generic_position: int) -> str | None:
    """Looks up which physical contract sat at curve position
    `generic_position` on the date `contracts_row` was taken from
    (a single row of curves.data_loader.load_contracts())."""
    col = f"M{generic_position}"
    if col not in contracts_row.index:
        return None
    val = contracts_row[col]
    return None if pd.isna(val) else str(val)


def construct_dfly(fly_a: pd.Series, fly_b: pd.Series) -> pd.Series:
    """Constructs a double-fly from two 1-month flies on their common
    date intersection:  Dfly = Fly_A - Fly_B."""
    aligned_a, aligned_b = fly_a.align(fly_b, join="inner")
    return aligned_a - aligned_b


def construct_from_ohlc(ohlc_a: pd.DataFrame, ohlc_b: pd.DataFrame, field: str = "Close") -> pd.Series:
    """Builds a constructed-structure series from two uploaded/looked-up
    OHLC legs, using `field` (default Close), aligned on the date
    intersection of both legs."""
    a = ohlc_a.set_index("Date")[field]
    b = ohlc_b.set_index("Date")[field]
    aligned_a, aligned_b = a.align(b, join="inner")
    return aligned_a - aligned_b


def align_multi_year(series_by_year: dict[int, pd.Series], reference_day_offset: bool = True) -> pd.DataFrame:
    """Overlays multiple years of the same seasonal structure on a
    common x-axis. If `reference_day_offset` is True, the x-axis becomes
    "trading days from the start of each year's series" so, e.g., the
    Jan-2024, Jan-2025, and Jan-2026 realisations of the same calendar
    fly can be compared on the same chart even though their calendar
    dates differ by exactly one year (roughly)."""
    frames = {}
    for year, series in series_by_year.items():
        s = series.dropna().reset_index(drop=True)
        s.index.name = "trading_day_offset"
        frames[str(year)] = s
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames)
