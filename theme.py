"""Global look & feel: CSS injected once per rerun + a shared Plotly layout."""
import streamlit as st

ACCENT, WARN, BAD, MUTED = "#3DDC97", "#F5A623", "#FF5C5C", "#8B949E"

_CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1500px;}
h1, h2, h3 {letter-spacing: -0.01em;}
[data-testid="stMetric"] {background:#161B22; border:1px solid #262C36; border-radius:10px; padding:12px 16px;}
[data-testid="stMetricLabel"] p {color:#8B949E; font-size:0.78rem; text-transform:uppercase; letter-spacing:.05em;}
[data-testid="stMetricValue"] {font-size:1.5rem;}
[data-testid="stSidebar"] {border-right:1px solid #262C36;}
[data-testid="stDataFrame"] {border:1px solid #262C36; border-radius:10px;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600; margin-right:6px;}
.badge.ok {background:#12382A; color:#3DDC97;} .badge.warn {background:#3D2E0E; color:#F5A623;}
.badge.bad {background:#3D1414; color:#FF5C5C;} .badge.mute {background:#21262D; color:#8B949E;}
.small {color:#8B949E; font-size:.82rem;}
</style>
"""


def apply():
    st.markdown(_CSS, unsafe_allow_html=True)


def badge(text: str, kind: str = "mute") -> str:
    return f'<span class="badge {kind}">{text}</span>'


def style_fig(fig, height=420):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified",
        font=dict(family="Inter, Segoe UI, sans-serif", size=12))
    fig.update_xaxes(gridcolor="#1F252E", zeroline=False)
    fig.update_yaxes(gridcolor="#1F252E", zeroline=False)
    return fig
