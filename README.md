# Dfly Desk (dfly-diagnostics)

Streamlit app with three top-level tabs:

| Tab | What it is | Logic status |
|---|---|---|
| Diagnostics | ADF, half-life, quality score, heat, correlations | `diagnostics.py` **unchanged**; `views/page_diagnostics.py` is the old `app.py` (only `set_page_config` removed, data path fixed, `use_container_width` -> `width="stretch"`) |
| Seasonality | Period-wise percentile seasonality | `curves/seasonality.py` and `curves/data_loader.py` **unchanged**; page rebuilt as UI only |
| Kink MAD | Live 1MDF strip from the QH Fairvalue API, two kink methods combined | new (`kink/`) |

Removed: old Curve Kink Scanner, Contract Lifecycle (`kink_detection.py`, `lifecycle.py`, old pages).

## Setup (VS Code)
```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```
Copy into `data/`: your CurveTerminal workbook (`CurveTerminal_CO_2014-07-11_....xlsx`), then once:
```
python prepare_curve_data.py --source data/<your workbook>.xlsx    # writes data/prepared/ (needed by Seasonality)
streamlit run app.py
```
Bundled in `data/`: `dflies_data.csv`, `CO-1MDF.xlsx` (manual snapshot), `CO_dflies_sept_futures_curve_edited.xlsx` (Method B history).

## Kink MAD tab
1. Connect to the VPN, pick **Live (QH API)**, paste the **raw** token (no `Bearer`), press **Test connection**.
2. Set the first strip contract (default `COH27`) and number of dflies (default 16). Dfly *i* = `P[i] - 3P[i+1] + 3P[i+2] - P[i+3]`, named after its first leg.
3. Polling: one batched request every 10 s (never faster than 8 s). 429 -> 2/4/8 s backoff, 3 retries. After 3 consecutive failed polls a leg is STALE and every dfly using it is blanked.
4. Method A (MAD / fixed ticks / turning points) and Method B (normalised kink z vs history) are shown per contract; **CONFIRMED** = both flag it.
5. **Manual snapshot** mode analyses an uploaded `CO-1MDF.xlsx` with no API.

The token is held only in `st.session_state`; it is never written to disk. API calls run in Python, so browser CORS does not apply. If TLS errors appear on the corporate network, set `REQUESTS_CA_BUNDLE` to your company CA `.pem`.

Tests: `python tests/test_kink_mad.py`
