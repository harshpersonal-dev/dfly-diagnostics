"""
Read-access layer over the prepared parquet files (see
prepare_curve_data.py). Every public function here is cached with
@st.cache_data so repeated widget interactions in the new pages don't
re-read parquet from disk.

Future-proofing: get_curve(date) is the single seam the live QH API will
eventually plug into. When that happens, only this function's body needs
to change -- everything downstream (kink detection, seasonality,
lifecycle) already consumes its DataFrame shape and is untouched.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

PREPARED_DIR = Path(__file__).resolve().parent.parent / "data" / "prepared"

STRUCTURE_LABELS = {
    "1MF": "1-Month Fly",
    "2MF": "2-Month Fly",
    "3MF": "3-Month Fly",
    "6MF": "6-Month Fly",
    "1MDF": "1-Month Double Fly",
    "2MDF": "2-Month Double Fly",
    "CND": "Condor",
    "1MS": "1-Month Spread",
    "2MS": "2-Month Spread",
    "3MS": "3-Month Spread",
    "6MS": "6-Month Spread",
    "12MS": "12-Month Spread",
}


def _structure_path(structure: str, field: str = "close") -> Path:
    fname = f"{structure.lower()}_{field}.parquet"
    return PREPARED_DIR / "structures" / fname


@st.cache_data
def data_available() -> bool:
    return (PREPARED_DIR / "curve_close.parquet").exists()


@st.cache_data
def load_curve_close() -> pd.DataFrame:
    """Wide (Date, M1..M24) outright close curve, indexed by curve position."""
    df = pd.read_parquet(PREPARED_DIR / "curve_close.parquet")
    return df.sort_values("Date").set_index("Date")


@st.cache_data
def load_contracts() -> pd.DataFrame:
    """Wide (Date, M1..M24) contract code occupying each curve position."""
    df = pd.read_parquet(PREPARED_DIR / "contracts.parquet")
    return df.sort_values("Date").set_index("Date")


@st.cache_data
def load_front_months() -> pd.DataFrame:
    return pd.read_parquet(PREPARED_DIR / "front_months.parquet")


@st.cache_data
def load_structure(structure: str, field: str = "close") -> pd.DataFrame:
    """Wide (Date, M1..MN) sheet for a given structure (1MDF, 1MF, 2MF, ...)."""
    path = _structure_path(structure, field)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df.sort_values("Date").set_index("Date")


@st.cache_data
def available_structures() -> list[str]:
    struct_dir = PREPARED_DIR / "structures"
    if not struct_dir.exists():
        return []
    names = sorted({p.stem.rsplit("_", 1)[0] for p in struct_dir.glob("*_close.parquet")})
    # Preserve a sensible/known display order where possible.
    order = ["1mf", "2mf", "3mf", "6mf", "1mdf", "2mdf", "cnd", "1ms", "2ms", "3ms", "6ms", "12ms"]
    names = [n for n in order if n in names] + [n for n in names if n not in order]
    return [n.upper() for n in names]


@st.cache_data
def available_instruments(structure_type: str | None = None) -> pd.DataFrame:
    """Distinct (Instrument, Type) pairs available in the OHLC sheet,
    optionally filtered to one Type ("outright", "spread", "fly")."""
    path = PREPARED_DIR / "ohlc.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["Instrument", "Type"])
    filters = [("Type", "==", structure_type)] if structure_type else None
    ohlc = pd.read_parquet(path, columns=["Instrument", "Type"], filters=filters)
    return ohlc.drop_duplicates().sort_values("Instrument").reset_index(drop=True)


@st.cache_data
def load_ohlc_for_instrument(instrument: str) -> pd.DataFrame:
    """Filters the (large) long-format OHLC parquet down to one instrument
    code on demand, rather than holding the whole 120k-row table in every
    page's working set."""
    path = PREPARED_DIR / "ohlc.parquet"
    if not path.exists():
        return pd.DataFrame()
    ohlc = pd.read_parquet(path, filters=[("Instrument", "==", instrument)])
    return ohlc.sort_values("Date").reset_index(drop=True)


@st.cache_data
def available_dates() -> pd.DatetimeIndex:
    return load_curve_close().index


def get_curve(date: pd.Timestamp) -> pd.DataFrame:
    """Returns the full outright curve on `date` as a tidy DataFrame with
    columns [curve_position, contract, close], sorted by curve_position,
    dropping positions with no data on that date.

    This is the seam described in the spec: later, swapping this
    function's body for a live QH API call is all that's needed for the
    whole kink -> seasonality -> lifecycle pipeline to run on live data.
    """
    close = load_curve_close()
    contracts = load_contracts()
    if date not in close.index:
        return pd.DataFrame(columns=["curve_position", "contract", "close"])

    close_row = close.loc[date]
    contract_row = contracts.loc[date] if date in contracts.index else pd.Series(dtype=object)

    rows = []
    for col in close.columns:
        price = close_row[col]
        if pd.isna(price):
            continue
        position = int(col[1:])  # "M7" -> 7
        rows.append({
            "curve_position": position,
            "contract": contract_row.get(col),
            "close": float(price),
        })
    out = pd.DataFrame(rows).sort_values("curve_position").reset_index(drop=True)
    return out


def nearest_date(target: pd.Timestamp, dates: pd.DatetimeIndex) -> pd.Timestamp:
    """Snaps to the closest available business date (handles weekends/holidays)."""
    idx = dates.get_indexer([target], method="nearest")[0]
    return dates[idx]


def shift_date(current: pd.Timestamp, dates: pd.DatetimeIndex, steps: int) -> pd.Timestamp:
    """Moves `steps` trading days forward (+) or backward (-) from `current`,
    clamped to the available date range."""
    pos = dates.get_indexer([current])[0]
    if pos == -1:
        pos = dates.get_indexer([current], method="nearest")[0]
    new_pos = min(max(pos + steps, 0), len(dates) - 1)
    return dates[new_pos]
