"""
app.py  --                                 granary monitor (Streamlit). Entry point only: every tab lives in dashboard/.

    pip install -r requirements.txt
    python pipeline/prepare_app_data.py ...      # see README.md; writes the CSVs this app reads into data/
    streamlit run app.py

Four components, one shared 'selected granary' (sidebar, or click a granary on any map):
    1 Overview                 dashboard/tab_overview.py   (also holds the reliability notes)
    2 Dynamic maps             dashboard/tab_map.py
    3 Yield: past + outlook    dashboard/tab_yield.py
    4 Advisory + intervention  dashboard/tab_advisory.py
"""
import streamlit as st

from dashboard import tab_advisory, tab_map, tab_modelcheck, tab_overview, tab_tracker, tab_transfer, tab_yield
from dashboard.data import load

APP_TITLE = "🌾 Granary monitor"

st.set_page_config(page_title=APP_TITLE, layout="wide")
D = load()

# a map click is applied here, before the sidebar selectbox is drawn (Streamlit cannot change a drawn widget)
if "_pending_granary" in st.session_state:
    st.session_state["granary"] = st.session_state.pop("_pending_granary")
if st.session_state.get("granary") not in D.ALL:
    st.session_state["granary"] = D.ALL[0]

st.title(APP_TITLE)
st.caption("Decision support for KPKM and granary administrators, from satellite (Sentinel-1/2) and weather data. "
           f"Satellite data ends {D.AS_OF}; reported yield ends {D.LAST_YEAR}.")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest year with yield", D.LAST_YEAR)
k2.metric("Average reported yield", f"{D.yh[D.yh.Year == D.LAST_YEAR].Yield.mean():,.0f} kg/ha",
          help="Simple average over granaries. Planted area is not available, so it is not area-weighted.")
k3.metric("Granaries with yield records", len(D.ALL))
k4.metric("Granaries with satellite forecasts", len(D.SAT))

with st.sidebar:
    st.markdown("### Granary")
    st.selectbox("Selected granary", D.ALL, key="granary",
                 help="Used by every tab. You can also click a granary on a map.")
    st.caption("For KPKM and granary administrators. Each granary is shown as the administrative district(s) it covers.")

tabs = st.tabs(["1 · Overview", "2 · Dynamic maps", "3 · Yield and outlook", "4 · Advisory and intervention", "5 · Decision tracker",
                "6 · Pre-Transfer Assessment", "7 · Transfer Learning", "8 · Model check"])
with tabs[0]:
    tab_overview.render(D)
with tabs[1]:
    tab_map.render(D)
with tabs[2]:
    tab_yield.render(D)
with tabs[3]:
    tab_advisory.render(D)
with tabs[4]:
    tab_tracker.render(D)
with tabs[5]:
    tab_transfer.render_assessment(D)
with tabs[6]:
    tab_transfer.render_transfer(D)
with tabs[7]:
    tab_modelcheck.render(D)
