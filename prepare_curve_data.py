"""
One-time (or "whenever the workbook is refreshed") conversion of the
CurveTerminal Excel workbook into small, typed parquet files that the
new Curve Structure & Kink Analysis Dashboard reads at runtime.

This script is READ-ONLY with respect to the classic diagnostics app:
it does not touch dflies_data.csv, app.py, or diagnostics.py, and the
classic page never imports anything produced here.

Run once (or whenever you get a fresh workbook):
    python prepare_curve_data.py
or, with a custom source/output location:
    python prepare_curve_data.py --source path/to/workbook.xlsx --out-dir data/prepared

Output layout (all under --out-dir, default "data/prepared"):
    curve_close.parquet        Daily.* replacement: wide M1..M24 outright close by date
    contracts.parquet          Wide M1..M24 contract code occupying each curve position
    front_months.parquet       Roll calendar (Contract, Delivery, First/Last session as M1, Expiry)
    structures/<name>.parquet  One file per wide structure sheet (1MDF close, 1MF close, 2MF close,
                                3MF close, 6MF close, 2MDF close, CND close, 1MS..12MS close/open, etc.)
    ohlc.parquet               Long-format per-instrument OHLC (~120k rows) -- kept as a single
                                parquet file since it is already small/fast to filter with pandas
                                at read time (dashboard filters to the requested instrument on demand
                                rather than re-reading Excel).

Wide "structure" sheets (1MDF, 1MF, 2MF, ...) already index columns by
CURVE POSITION (M1, M2, ...), not by a fixed physical contract. That is
exactly the roll-adjusted numbering the dashboard needs: "the 7th generic
1-month-double-fly on date t" is column M7 of the "1MDF close" sheet on
that date, regardless of which physical contracts happen to sit there.
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = "data/CurveTerminal_CO_2014-07-11_2026-09-25-094c62.xlsx"
DEFAULT_OUT_DIR = "data/prepared"

# Wide (Date, M1, M2, ...) sheets that back the structure-seasonality view.
STRUCTURE_SHEETS = [
    "1MDF close", "1MDF open",
    "2MDF close", "2MDF open",
    "1MF close", "1MF open", "1MF high", "1MF low",
    "2MF close", "2MF open",
    "3MF close", "3MF open",
    "6MF close", "6MF open",
    "CND close", "CND open",
    "1MS close", "1MS open", "1MS high", "1MS low",
    "2MS close", "2MS open",
    "3MS close", "3MS open",
    "6MS close", "6MS open",
    "12MS close", "12MS open",
]


def _read_wide_sheet(xls: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xls, sheet_name=sheet)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def convert(source: str, out_dir: str) -> None:
    out = Path(out_dir)
    (out / "structures").mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(source)

    # --- Outright curve (close), by curve position, wide M1..M24 ---
    curve_close = _read_wide_sheet(xls, "Curve close")
    curve_close.to_parquet(out / "curve_close.parquet", index=False)
    print(f"Wrote curve_close.parquet: {curve_close.shape}")

    # --- Contracts occupying each curve position, wide M1..M24 ---
    contracts = _read_wide_sheet(xls, "Contracts")
    contracts.to_parquet(out / "contracts.parquet", index=False)
    print(f"Wrote contracts.parquet: {contracts.shape}")

    # --- Roll calendar ---
    front_months = pd.read_excel(xls, sheet_name="Front months")
    for col in ("First session as M1", "Last session as M1", "Expiry"):
        if col in front_months.columns:
            front_months[col] = pd.to_datetime(front_months[col])
    front_months.to_parquet(out / "front_months.parquet", index=False)
    print(f"Wrote front_months.parquet: {front_months.shape}")

    # --- Structure sheets (flies, spreads, condors) ---
    available = set(xls.sheet_names)
    for sheet in STRUCTURE_SHEETS:
        if sheet not in available:
            continue
        df = _read_wide_sheet(xls, sheet)
        fname = sheet.replace(" ", "_").lower() + ".parquet"
        df.to_parquet(out / "structures" / fname, index=False)
        print(f"Wrote structures/{fname}: {df.shape}")

    # --- Long-format OHLC (per instrument), kept whole -- dashboard filters on demand ---
    if "OHLC" in available:
        ohlc = pd.read_excel(xls, sheet_name="OHLC")
        ohlc["Date"] = pd.to_datetime(ohlc["Date"])
        ohlc.to_parquet(out / "ohlc.parquet", index=False)
        print(f"Wrote ohlc.parquet: {ohlc.shape}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Path to the source .xlsx workbook")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Directory to write prepared parquet files")
    args = parser.parse_args()
    convert(args.source, args.out_dir)


if __name__ == "__main__":
    main()
