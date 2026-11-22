"""
app.py  --  PaddyTwin AI (PACE V10). Entry point only: every page lives in dashboard/.

    pip install -r requirements.txt
    streamlit run app.py

Navigation (sidebar cards): an Overview hub, the KPKM Programme Manager workspace (4 pages) and the Model Validator & Data Steward workflow (4 steps, with Next / Back buttons).
One shared Selected Granary (sidebar, or click a granary polygon on a map). The validator writes results/governance_registry.json (Step 4); the KPKM workspace only reads it and
never runs or configures a model.

  Overview                      dashboard/tab_overview.render_overview_hub
  KPKM Programme Manager        1 Granary Field Condition Map   tab_map.render_leaflet_map(role="kpkm")
                                2 Historical Replay & Yield Fit tab_yield.render_yield_view("historical")  -> tab_governance.render_replay
                                3 In-Season Nowcast & Outlook   tab_yield.render_yield_view("outlook")     -> tab_governance.render_outlook
                                4 Programme Manager Decision Brief  tab_advisory.render_decision_brief
  Model Validator & Data Steward 1 Spatial & Optical Quality Map tab_map.render_leaflet_map(role="validator")
                                2 Data & Domain Divergence Audit tab_transfer.render_domain_audit
                                3 Transfer Testing & Train/Test Audit tab_transfer.render_benchmark_audit
                                4 Validator Sign-Off & Assurance Gate tab_transfer.render_validator_signoff
The order of the validator's steps is enforced by acts_transfer/workflow.py: changing anything upstream clears and recalculates what depends on it.
"""
import streamlit as st
import streamlit.components.v1 as components

from dashboard import tab_advisory, tab_map, tab_overview, tab_transfer, tab_yield
from dashboard.data import load
from dashboard.tab_governance import ROLE_KPKM, ROLE_VAL

st.set_page_config(page_title="PaddyTwinAI · PACE V10", layout="wide")


def apply_botanical_theme():
    """Warm ivory and botanical olive-lime palette with the PaddyTwin sidebar: dark-green sidebar, bordered navigation cards, a lime outline on the active one, and a role card at the bottom"""
    st.markdown(
        """
        <style>
        .stApp { background-color: #F7F8F3; color: #0B2117; }
        section[data-testid="stSidebar"] { background-color: #0B2117; border-right: 1px solid #1E3F30; }
        .sb-tagline { color: #F7F8F3; font-size: 15px; font-weight: 600; line-height: 1.4; margin-bottom: 1.2rem; }
        .sb-section { color: #A3B8AE; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin: 1rem 0 0.4rem 0; }

        /* bordered sidebar navigation cards */
        section[data-testid="stSidebar"] div[role="radiogroup"] > label { background-color: #122E21; border: 1px solid #264B39; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; width: 100%; }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { background-color: #1A3D2C !important; border: 1.5px solid #9FCE34 !important; }
        section[data-testid="stSidebar"] div[role="radiogroup"] label p { color: #F7F8F3 !important; font-weight: 600 !important; font-size: 13.5px !important; }
        /* the select box keeps its own light surface, so its text stays dark */
        section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #0B2117 !important; }

        /* role card at the bottom of the sidebar */
        .sb-role-box { background: #122E21; border: 1px solid #264B39; border-radius: 12px; padding: 14px; margin-top: 1.4rem; }
        .sb-role-title { color: #9FCE34; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px; }
        .sb-role-desc { color: #DCE3D3; font-size: 12px; line-height: 1.45; margin-bottom: 10px; }
        .sb-role-footer { color: #8CA398; font-size: 10px; letter-spacing: 0.06em; text-transform: uppercase; }

        /* sage mist KPI cards, chartreuse lime action buttons, white content cards */
        div[data-testid="stMetric"] { background-color: #E7ECE0; border: 1px solid #DCE3D3; border-radius: 14px; padding: 16px 20px; }
        div[data-testid="stMetricLabel"] { color: #0B2117 !important; font-weight: 600; }
        div[data-testid="stMetricValue"] { color: #769D18 !important; font-weight: 700; }
        div.stButton > button { background-color: #9FCE34; color: #0B2117 !important; font-weight: 700; border: none; border-radius: 10px; padding: 0.65rem 1.2rem; transition: 0.2s ease; }
        div.stButton > button:hover { background-color: #769D18; color: #FFFFFF !important; }
        div.stButton > button:disabled { background-color: #DCE3D3; color: #6B7A70 !important; }
        .pace-card { background: #FFFFFF; border: 1px solid #DCE3D3; border-radius: 16px; padding: 20px; margin-bottom: 16px; box-shadow: 0 2px 10px rgba(11, 33, 23, 0.04); }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_botanical_theme()
D = load()

# session state
if st.session_state.get("granary") not in D.ALL:
    st.session_state["granary"] = "IADA Barat Laut Selangor" if "IADA Barat Laut Selangor" in D.ALL else D.ALL[0]
st.session_state.setdefault("nav_mode", "Overview")
st.session_state.setdefault("validator_step", 1)
st.session_state.setdefault("scroll_top", False)
if "_pending_granary" in st.session_state:                      # a click on the pydeck map (the fallback map) is applied before the sidebar is drawn
    st.session_state["granary"] = st.session_state.pop("_pending_granary")

if st.session_state["scroll_top"]:
    components.html("<script>const d=window.parent.document;const e=d.querySelector('section[data-testid=\"stMain\"]')||d.querySelector('section.main');if(e){e.scrollTo(0,0);}</script>", height=0)
    st.session_state["scroll_top"] = False

KPKM_PAGES = ["Granary Field Condition Map", "Historical Replay & Yield Fit", "In-Season Nowcast & 2025–2028 Outlook", "Programme Manager Decision Brief"]
VALIDATOR_STEPS = ["Step 1 · Spatial & Optical Map", "Step 2 · Domain Divergence Audit", "Step 3 · Transfer Testing & Audit", "Step 4 · Validator Sign-Off Gate"]
STEP_TITLES = {2: "Step 2 · Data & Domain Divergence Audit", 3: "Step 3 · Transfer Testing & Train/Test Audit", 4: "Step 4 · Validator Sign-Off & Assurance Gate"}

# sidebar navigation
with st.sidebar:
    st.markdown('<div class="sb-tagline">A clearer view.<br>Better field decisions.</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-section">YOUR WORKSPACE</div>', unsafe_allow_html=True)
    workspaces = ["Overview", "KPKM Programme Manager", "Model Validator & Data Steward"]
    st.session_state["nav_mode"] = st.radio("Workspace", workspaces, index=workspaces.index(st.session_state["nav_mode"]), label_visibility="collapsed")
    mode = st.session_state["nav_mode"]
    if mode == "KPKM Programme Manager":
        st.markdown('<div class="sb-section">PROGRAMME MANAGER PAGES</div>', unsafe_allow_html=True)
        page = st.radio("Pages", KPKM_PAGES, label_visibility="collapsed")
        role_title, role_desc = "PROGRAMME MANAGER", "Monitor paddy fields, coordinate field teams and review programme evidence."
    elif mode == "Model Validator & Data Steward":
        st.markdown('<div class="sb-section">VALIDATOR AUDIT STEPS</div>', unsafe_allow_html=True)
        chosen = st.radio("Steps", VALIDATOR_STEPS, index=st.session_state["validator_step"] - 1, label_visibility="collapsed")
        st.session_state["validator_step"] = VALIDATOR_STEPS.index(chosen) + 1
        role_title, role_desc = "MODEL VALIDATOR & STEWARD", "Audit optical quality, test Detrended Residual Transfer, and record governance sign-off."
    else:
        role_title, role_desc = "NATIONAL OVERVIEW", f"Cross-role executive summary across all {len(D.ALL)} Malaysian paddy granary schemes."
    st.markdown('<div class="sb-section">SELECTED GRANARY</div>', unsafe_allow_html=True)
    st.session_state["granary"] = st.selectbox("Selected Granary", D.ALL, index=D.ALL.index(st.session_state["granary"]), label_visibility="collapsed",
                                               help="Used by every page. You can also click a granary on a map.")
    st.markdown(f'<div class="sb-role-box"><div class="sb-role-title">{role_title}</div><div class="sb-role-desc">{role_desc}</div>'
                '<div class="sb-role-footer">MALAYSIA &nbsp;/&nbsp; PADDY SYSTEMS</div></div>', unsafe_allow_html=True)

st.session_state["workspace_role"] = ROLE_VAL if mode == "Model Validator & Data Steward" else ROLE_KPKM      # read by the governance views and the map


def _go(step=None, mode_=None):
    """a Next / Back button: move to another validator step (or workspace) and scroll to the top"""
    if step is not None:
        st.session_state["validator_step"] = step
    if mode_ is not None:
        st.session_state["nav_mode"] = mode_
    st.session_state["scroll_top"] = True
    st.rerun()


# route views
if mode == "Overview":
    tab_overview.render_overview_hub(D)

elif mode == "KPKM Programme Manager":
    if page == KPKM_PAGES[0]:
        tab_map.render_leaflet_map(D, role="kpkm")
    elif page == KPKM_PAGES[1]:
        tab_yield.render_yield_view("historical", D)
    elif page == KPKM_PAGES[2]:
        tab_yield.render_yield_view("outlook", D)
    else:
        tab_advisory.render_decision_brief(D)

else:
    step = st.session_state["validator_step"]
    if step in STEP_TITLES:
        st.markdown(f"## {STEP_TITLES[step]}")
    if step == 1:
        tab_map.render_leaflet_map(D, role="validator", title="Step 1 · Spatial & Optical Quality Map")
        st.divider()
        if st.button("Next Step: Data & Domain Divergence Audit →", key="nx_1", width="stretch"):
            _go(2)
    elif step == 2:
        tab_transfer.render_domain_audit(D)
        st.divider()
        c1, c2 = st.columns([1, 2])
        if c1.button("← Back to Spatial Map", key="bk_2", width="stretch"):
            _go(1)
        if c2.button("Next Step: Transfer Testing & Train/Test Audit →", key="nx_2", width="stretch"):
            _go(3)
    elif step == 3:
        tab_transfer.render_benchmark_audit(D)
        st.divider()
        c1, c2 = st.columns([1, 2])
        if c1.button("← Back to Domain Audit", key="bk_3", width="stretch"):
            _go(2)
        if c2.button("Transfer Testing Complete — Proceed to Step 4: Validator Sign-Off →", key="nx_3", width="stretch"):
            _go(4)
    else:
        tab_transfer.render_validator_signoff(D)
        st.divider()
        c1, c2 = st.columns([1, 2])
        if c1.button("← Back to Transfer Testing", key="bk_4", width="stretch"):
            _go(3)
        if c2.button("Finish & Return to Overview Summary →", key="nx_4", width="stretch"):
            _go(mode_="Overview")
