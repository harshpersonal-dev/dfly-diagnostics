"""
Converts the pipe-delimited markdown table (Excel-serial dates) into a
clean CSV that the Streamlit app loads directly.

Run once (or whenever you get fresh source data):
    python convert_data.py
"""

import pandas as pd

SOURCE_MD = "convertcsv.md"
OUTPUT_CSV = "dflies_data.csv"


def parse_pipe_md(path):
    with open(path) as f:
        lines = f.readlines()

    # Line 0 = spreadsheet column letters (A, B, C...), line 1 = --- separator,
    # line 2 = the real header row (Date, CO1-2-3-4, ...)
    header_line = lines[2]
    header = [c.strip() for c in header_line.strip().strip("|").split("|")]

    rows = []
    for line in lines[3:]:
        line = line.strip()
        if not line or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(cells)

    df = pd.DataFrame(rows, columns=header)
    # Dates arrive as Excel serial numbers (e.g. 43102 = 2018-01-02)
    df["Date"] = pd.to_datetime(df["Date"].astype(float), unit="D", origin="1899-12-30")
    for c in df.columns[1:]:
        df[c] = df[c].astype(float)

    return df.sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    df = parse_pipe_md(SOURCE_MD)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}: {len(df)} rows, {len(df.columns) - 1} Dfly columns")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
