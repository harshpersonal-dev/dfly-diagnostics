"""Kink MAD - live 1MDF strip from the QH Fairvalue API, analysed two ways:
  Method A: 2nd-difference kinks (MAD-robust and/or fixed tick threshold) + turning points
  Method B: normalised kink z-score vs history + pooled quadratic residual
Both are shown per contract with a confluence grade.  The token lives only in
st.session_state (server memory of this session): never written to disk."""
import io
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import theme
from kink import kink_mad as km
from kink import qh_client as qh
from kink.strip import build_strip, dfly_values, legs_of

ROOT = Path(__file__).resolve().parent.parent
HIST_DEFAULT = ROOT / "data" / "CO_dflies_sept_futures_curve_edited.xlsx"
MANUAL_DEFAULT = ROOT / "data" / "CO-1MDF.xlsx"
POLL_S = 10          # live polling period
MIN_GAP_S = 8        # never poll more often than this, even if the page reruns
STALE_AFTER = 3      # consecutive failed polls before a leg is STALE

S = st.session_state
S.setdefault("qh_quotes", {})                       # leg code -> {"price","ts","fails"}
S.setdefault("qh_codes", None)                      # cached products=* code set
S.setdefault("qh_test", None)                       # (kind, message) of last Test Connection
S.setdefault("qh_poll", {"err": None, "at": 0.0})   # last poll outcome

st.title("Kink MAD")

# ------------------------------------------------------------------ sidebar
sb = st.sidebar
sb.header("Curve source")
mode = sb.radio("Source", ["Live (QH API)", "Manual snapshot"], key="km_mode", label_visibility="collapsed")
is_live = mode.startswith("Live")
tick = sb.number_input("Tick size (price units)", value=0.01, format="%.4f", key="km_tick")

access = ""
live_on = False
if is_live:
    sb.text_input("Access token — paste the raw token only", type="password", key="access_input",
                  help="No 'Bearer' prefix needed; the app adds it. Kept in memory for this session only.")
    access = S.get("access_input", "")
    start = sb.text_input("First strip contract", value="COH27", key="km_start").strip().upper()
    n_dfly = int(sb.number_input("Number of dflies", 4, 30, 16, 1, key="km_n"))
    live_on = sb.toggle(f"Live polling (every {POLL_S} s)", value=True, key="km_live")
    b1, b2 = sb.columns(2)
    test_clicked = b1.button("Test connection", key="km_test", width="stretch")
    if b2.button("Refresh now", key="km_refresh_btn", width="stretch"):
        S["km_force"] = True
    if test_clicked:
        if not access:
            S["qh_test"] = ("auth", "Paste a token first.")
        else:
            with st.spinner("Calling products=* ..."):
                try:
                    S["qh_codes"] = qh.fetch_all_codes(access)
                    S["qh_test"] = ("ok", f"Connected — {len(S['qh_codes'])} products cached.")
                except qh.QHError as e:
                    S["qh_test"] = (e.kind, str(e))
    if S["qh_test"]:
        (sb.success if S["qh_test"][0] == "ok" else sb.error)(S["qh_test"][1])
    try:
        dfly_codes, legs = build_strip(start, n_dfly)
    except ValueError as e:
        sb.error(str(e))
        st.stop()
else:
    manual_file = sb.file_uploader("CO-1MDF.xlsx (row 1 codes, row 2 dfly in ticks)", type=["xlsx"], key="km_manual_file")

with sb.expander("Method A — MAD / fixed threshold", expanded=False):
    tradable_from = int(st.number_input("Tradable from position", 1, 30, 1, 1, key="ka_trad"))
    fixed_thresh = st.number_input("Fixed threshold (ticks)", 0.0, 50.0, 5.0, 0.5, key="ka_fixed")
    borderline_at = st.number_input("Borderline at (ticks)", 0.0, 50.0, 5.0, 0.5, key="ka_border")
    mad_k = st.number_input("MAD multiplier k", 0.5, 10.0, 3.0, 0.5, key="ka_k")
    mad_on = st.selectbox("MAD computed on", ["full", "tradable"], key="ka_madon")
    min_turn = st.number_input("Noise turn amplitude ≤ (ticks)", 0.0, 20.0, 2.0, 0.5, key="ka_turn")
    count_zeros = st.checkbox("Count zero plateaus as turns", False, key="ka_zeros")
with sb.expander("Method B — normalised z-score", expanded=False):
    hist_file = st.file_uploader("Historical dfly panel (.xlsx)", type=["xlsx"], key="kb_hist",
                                 help="Date + one column per contract, e.g. 'CO H27 Dfly'. Default: data/CO_dflies_sept_futures_curve_edited.xlsx")
    z_thr = st.slider("Kink z threshold |z| ≥", 1.0, 4.0, 2.0, 0.1, key="kb_z")
    show_flagged = st.checkbox("Table: flagged rows only", False, key="k_flagged")

mad_params = km.MadParams(tradable_from, fixed_thresh, borderline_at, mad_k, mad_on, min_turn, count_zeros)


@st.cache_data(show_spinner=False)
def _load_hist(name: str, data: bytes | None):
    return km.load_history(io.BytesIO(data) if data else str(HIST_DEFAULT))


def _read_manual(src):
    raw = pd.read_excel(src, header=None)
    if raw.shape[0] > raw.shape[1]:
        raw = raw.T
    codes = [str(c).strip().upper() for c in raw.iloc[0]]
    return codes, pd.to_numeric(raw.iloc[1], errors="coerce").to_numpy(float)


# ------------------------------------------------------------------ analysis + render
def analyse_and_render(codes, dfly_px, extra=None):
    """codes: ['COH27', ...]; dfly_px: dfly in price units (NaN where unavailable);
    extra: optional DataFrame(Contract-short, Legs, Status) for live mode."""
    if not np.isfinite(dfly_px).any():
        st.warning("No usable dfly values yet.")
        return
    ticks = np.round(dfly_px / tick, 1)
    tbl, info, turns = km.mad_analysis(codes, ticks, mad_params)

    try:
        hist = _load_hist(hist_file.name if hist_file else "default", hist_file.getvalue() if hist_file else None)
        zt, zinfo = km.normalized_kink(hist, pd.Series(dfly_px, index=codes))
    except Exception as e:   # unreadable / missing history should not kill Method A
        zt, zinfo = pd.DataFrame(), {"ok": False, "reason": f"history could not be loaded: {e}", "used": []}
    comb = km.combine(tbl, zt, z_thr, tick)
    if extra is not None:
        comb = comb.merge(extra, on="Contract", how="left")

    n_conf = int((comb.Confluence == "CONFIRMED").sum())
    n_a = int((comb.Confluence == "A only").sum())
    n_b = int((comb.Confluence == "B only").sum())
    d2abs = np.abs(comb["2nd diff"].to_numpy(float))
    big = comb.loc[int(np.nanargmax(d2abs)), "Contract"] if np.isfinite(d2abs).any() else "—"

    k = st.columns(5)
    k[0].metric("Confirmed (A+B)", n_conf)
    k[1].metric("A only", n_a)
    k[2].metric("B only", n_b)
    k[3].metric("Biggest 2nd diff", big)
    k[4].metric("MAD cutoff (ticks)", f"{info['cutoff']:.1f}" if info["mad_usable"] else "off (MAD=0)")
    if not zinfo.get("ok"):
        st.warning(f"Method B unavailable: {zinfo.get('reason', 'unknown')}. Showing Method A only.")

    # ---- chart
    pos = comb["Pos"].to_numpy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("1MDF strip (ticks)", "2nd difference (ticks)", "Kink z-score"))
    fig.add_trace(go.Scatter(x=pos, y=comb["Dfly"], mode="lines+markers", line=dict(color="#58A6FF", width=2),
                             marker=dict(size=6), name="Dfly", text=comb["Contract"],
                             hovertemplate="%{text}: %{y:.1f}<extra></extra>"), row=1, col=1)
    for label, color in (("CONFIRMED", theme.BAD), ("A only", theme.WARN), ("B only", "#A371F7")):
        m = comb[comb.Confluence == label]
        if len(m):
            fig.add_trace(go.Scatter(x=m["Pos"], y=m["Dfly"], mode="markers", name=label,
                                     marker=dict(size=13, color=color, line=dict(width=1, color="white")),
                                     text=m["Contract"], hovertemplate="%{text}<extra>" + label + "</extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=pos, y=comb["2nd diff"], marker_color=np.where(comb["Signal A"], theme.WARN, "#3B4452"),
                         name="2nd diff", showlegend=False), row=2, col=1)
    if info["mad_usable"]:
        for s in (1, -1):
            fig.add_hline(y=s * info["cutoff"], line_dash="dot", line_color=theme.WARN, row=2, col=1)
    for s in (1, -1):
        fig.add_hline(y=s * fixed_thresh, line_dash="dash", line_color=theme.MUTED, row=2, col=1)
    fig.add_trace(go.Bar(x=pos, y=comb["Kink_Z"], marker_color=np.where(comb["Signal B"], "#A371F7", "#3B4452"),
                         name="Kink z", showlegend=False), row=3, col=1)
    for s in (1, -1):
        fig.add_hline(y=s * z_thr, line_dash="dot", line_color="#A371F7", row=3, col=1)
    if tradable_from > 1:
        fig.add_vrect(x0=0.5, x1=tradable_from - 0.5, fillcolor="#FFFFFF", opacity=0.05, line_width=0,
                      annotation_text="outside mandate", annotation_position="top left")
    fig.update_xaxes(tickmode="array", tickvals=pos, ticktext=comb["Contract"], tickangle=-45)
    st.plotly_chart(theme.style_fig(fig, 720), width="stretch")

    # ---- table
    cols = ["Pos", "Contract", "Dfly", "2nd diff", "Flag", "Kink_Z", "Quad_Resid", "Confluence",
            "A says", "B says", "Directions agree", "Tradable"]
    if extra is not None:
        cols += ["Status", "Legs"]
    view = comb[cols]
    if show_flagged:
        view = view[(view.Confluence != "") | (view.Flag != "")]

    def _row_style(row):
        if row.get("Status") in ("STALE", "no data"):
            return ["background-color:#2a2a2a;color:#888"] * len(row)
        bg = {"CONFIRMED": "#4a1d1d", "A only": "#3d2e0e", "B only": "#2b1d4a"}.get(row["Confluence"], "")
        return [f"background-color:{bg}" if bg else ""] * len(row)

    st.dataframe(view.style.apply(_row_style, axis=1).format(
        {"Dfly": "{:.1f}", "2nd diff": "{:+.1f}", "Kink_Z": "{:+.2f}", "Quad_Resid": "{:+.1f}"}, na_rep="—"),
        hide_index=True, width="stretch", height=min(60 + 35 * len(view), 640))
    st.download_button("Download table (CSV)", comb.to_csv(index=False).encode(),
                       file_name=f"kink_mad_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv",
                       key=f"dl_{time.time_ns()}")

    with st.expander("Sign-change turning points (Method A)"):
        st.dataframe(turns, hide_index=True, width="stretch")
    with st.expander("How to read this / diagnostics"):
        st.markdown(
            f"- **Method A** MAD(full) = {info['mad_full']:.2f}, MAD(tradable) = {info['mad_trad']:.2f}; "
            f"cutoff = {mad_k:g} × MAD({mad_on}); fixed rule |2nd diff| > {fixed_thresh:g} ticks.\n"
            "- **Method B** each day's strip is z-normalised across contracts; kink z = (today's normalised value − "
            "historical normalised mean) / historical normalised SD.\n"
            "- **Direction:** 2nd diff > 0 → dfly is *below* its neighbours (cheap / dip). Kink z > 0 → dfly is "
            "*above* its historical shape (rich). *Directions agree* = yes when both point the same way.\n"
            "- **CONFIRMED** = flagged by A and B. Treat single-method flags as watch-list items.")
        if zinfo.get("ok"):
            st.write(f"Method B used {zinfo['n_contracts']} contracts × {zinfo['n_rows']} historical days "
                     f"(only contracts present in both the live strip and the history file).")
            unused = [c for c in codes if c not in zinfo["used"]]
            if unused:
                st.caption("Not in history / NaN, excluded from Method B: " + ", ".join(unused))


# ------------------------------------------------------------------ live polling
def _poll(placeholder):
    def on_retry(n, wait):
        placeholder.warning(f"Rate limited (429) — retrying {n}/3 in {wait}s…")

    err, quotes = None, {}
    try:
        quotes = qh.fetch(access, legs, on_retry=on_retry)
    except qh.QHError as e:
        err = e
    placeholder.empty()
    for c in legs:
        rec = S["qh_quotes"].setdefault(c, {"price": None, "ts": None, "fails": 0})
        if c in quotes:
            rec.update(price=quotes[c].price, ts=quotes[c].ts, fails=0)
        else:
            rec["fails"] += 1
    S["qh_poll"] = {"err": err, "at": time.time()}


live_active = is_live and bool(access) and live_on


@st.fragment(run_every=POLL_S if live_active else None)
def live_panel():
    if not access:
        st.info("Paste your access token in the sidebar to start.")
        return
    force = S.pop("km_force", False)
    ph = st.empty()
    if (live_active or force) and (force or time.time() - S["qh_poll"]["at"] >= MIN_GAP_S):
        _poll(ph)

    poll = S["qh_poll"]
    quotes = S["qh_quotes"]
    if not poll["at"]:
        st.info("No data yet — turn on live polling or press Refresh now.")
        return

    prices = {c: (quotes[c]["price"] if c in quotes and quotes[c]["fails"] < STALE_AFTER else None) for c in legs}
    stale = [c for c in legs if c in quotes and quotes[c]["fails"] >= STALE_AFTER]
    nodata = [c for c in legs if c not in quotes or quotes[c]["price"] is None]

    badges = [theme.badge("LIVE" if live_active else "PAUSED", "ok" if live_active else "mute"),
              theme.badge(f"last poll {datetime.fromtimestamp(poll['at']):%H:%M:%S}", "mute")]
    if poll["err"]:
        badges.append(theme.badge("last poll FAILED", "bad"))
    if stale:
        badges.append(theme.badge(f"{len(stale)} STALE leg(s)", "bad"))
    st.markdown(" ".join(badges), unsafe_allow_html=True)
    if poll["err"]:
        st.error(str(poll["err"]))
    if stale:
        st.error("STALE (3+ failed polls, value withheld): " + ", ".join(stale) +
                 " — any dfly using these legs is blanked.")
    if S["qh_codes"] is not None:
        missing = [c for c in legs if c not in S["qh_codes"]]
        if missing:
            st.warning("Not found in the QH products list — confirm manually: " + ", ".join(missing))

    px = dfly_values(prices, legs)
    now = time.time()
    status = []
    for i in range(len(px)):
        ls = legs_of(legs, i)
        status.append("STALE" if any(c in stale for c in ls) else
                      "no data" if any(c in nodata for c in ls) else "ok")
    extra = pd.DataFrame({"Contract": [km.short_code(c) for c in dfly_codes], "Status": status,
                          "Legs": [" / ".join(km.short_code(c) for c in legs_of(legs, i)) for i in range(len(px))]})
    analyse_and_render(dfly_codes, px, extra)

    with st.expander("Leg prices (outrights)"):
        st.dataframe(pd.DataFrame([{
            "Leg": c, "Price": quotes.get(c, {}).get("price"),
            "Quote age (s)": (now - quotes[c]["ts"] / 1000) if c in quotes and quotes[c]["ts"] else None,
            "Failed polls": quotes.get(c, {}).get("fails", 0),
            "State": "STALE" if c in stale else "no data" if c in nodata else "ok"} for c in legs]),
            hide_index=True, width="stretch")
        st.caption("Legs are consecutive outrights from the first strip contract, weights [1, −3, 3, −1] per dfly. "
                   "They must be confirmed by you as the correct QH products — not auto-matched or roll-adjusted.")


if is_live:
    live_panel()
else:
    src = manual_file if manual_file else (str(MANUAL_DEFAULT) if MANUAL_DEFAULT.exists() else None)
    if src is None:
        st.info("Upload a CO-1MDF.xlsx snapshot in the sidebar.")
    else:
        m_codes, m_ticks = _read_manual(src)
        st.markdown(theme.badge("MANUAL SNAPSHOT", "warn") + theme.badge(f"{len(m_codes)} dflies", "mute"),
                    unsafe_allow_html=True)
        analyse_and_render(m_codes, m_ticks * tick)
