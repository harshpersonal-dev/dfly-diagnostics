# Dfly Diagnostics App

A small local web app: pick a generic Dfly number and a date range, and it
runs the ADF stationarity test, half-life, quality score, median/max heat,
stop-out rate, and reward-to-risk on exactly that slice of history.

Your uploaded data (`convertcsv.md`) has already been converted into
`dflies_data.csv` and is included here — you don't need to re-run the
converter unless you get updated source data.

## Files

- `app.py` — the Streamlit app (what you run)
- `diagnostics.py` — the analysis functions (ADF, half-life, quality score,
  heat) — same math as the earlier scripts in this conversation, just
  packaged as reusable functions
- `convert_data.py` — parses the pipe-delimited markdown table (with Excel
  serial dates) into a clean CSV. Only needed if you get a new source file.
- `dflies_data.csv` — your data, already converted and ready to use

## How to run

1. Install dependencies (one time):

   ```
   pip install streamlit pandas numpy statsmodels
   ```

2. From inside this folder, launch the app:

   ```
   streamlit run app.py
   ```

3. It will open automatically in your browser (usually
   `http://localhost:8501`). If it doesn't, open that URL manually.

## Using the app

- **Sidebar → Generic Dfly number**: pick which Dfly (CO1-2-3-4,
  CO2-3-4-5, etc.) to analyze.
- **Sidebar → Duration**: choose a preset (20 days, 1 month, 3/6 months,
  1 year, full history) or "Custom range" to pick any start/end date
  directly from your data.
- **Sidebar → Trade parameters**: set your tick size, dollar value per
  tick, and stop distance in ticks — this converts your dollar risk into
  the price-unit stop distance used for the stop-out rate and
  reward-to-risk calculations. Defaults are pre-filled for a $10/tick,
  2-tick ($20) stop.
- **Sidebar → Entry z-score / Rolling lookback / ADF regression type**:
  advanced knobs if you want to test sensitivity to these choices.

The main panel shows the selected series as a chart, then four metric
panels: ADF stat/p-value/verdict, half-life/quality score, median/max
heat, and stop-out rate/reward-to-risk. Expand "Full raw output" for every
number in one place, and "ADF critical values" to see the exact table the
p-value was compared against.

## Updating with new data

If you get a fresh version of the source markdown file, drop it in this
folder as `convertcsv.md` and re-run:

```
python convert_data.py
```

This regenerates `dflies_data.csv`. Refresh the running app (or restart
it) to pick up the new data — `st.cache_data` will otherwise keep serving
the old file from cache within a single running session.

## Notes on the calculations

See the parameter reference document from earlier in this conversation
(`dfly_parameters_explained.md`) for the exact formulas and worked
examples behind ADF, half-life, quality score, and heat. `diagnostics.py`
implements those same formulas exactly.
