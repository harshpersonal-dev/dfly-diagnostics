"""Dfly Desk - entry point.  Run:  streamlit run app.py

Top navigation, one script per page (each page owns its sidebar):
  Diagnostics  - original ADF / half-life / heat / correlation app (logic unchanged)
  Seasonality  - period-wise percentile seasonality (curves/seasonality.py unchanged)
  Kink MAD     - live QH curve -> Method A (MAD/fixed) + Method B (normalised z) kinks
"""
import streamlit as st

import theme

st.set_page_config(page_title="Dfly Desk", page_icon="📈", layout="wide")
theme.apply()

pages = [
    st.Page("views/page_diagnostics.py", title="Diagnostics", icon=":material/monitoring:", default=True),
    st.Page("views/page_seasonality.py", title="Seasonality", icon=":material/calendar_month:"),
    st.Page("views/page_kink_mad.py", title="Kink MAD", icon=":material/bolt:"),
]
st.navigation(pages, position="top").run()
