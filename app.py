"""
app.py  --  PaddyTwin AI (PACE V10). Entry point only: every page lives in dashboard/.

    pip install -r requirements.txt
    streamlit run app.py
"""
import streamlit as st
import streamlit.components.v1 as components

from acts_transfer.automl_runner import run_full_automl_for_granary
from dashboard import tab_advisory, tab_map, tab_overview, tab_transfer, tab_yield
from dashboard.data import load
from dashboard.tab_governance import ROLE_KPKM, ROLE_VAL
from dashboard.ui import inject_gemini_sidebar_css, render_gemini_nav_item

st.set_page_config(page_title="PaddyTwinAI · National Paddy Intelligence", layout="wide")

_GRANARY_STATE_LOOKUP = {
    "MADA": "Kedah & Perlis",
    "IADA Pulau Pinang": "Pulau Pinang",
    "IADA Kerian": "Perak",
    "IADA Seberang Perak": "Perak",
    "IADA Barat Laut Selangor": "Selangor",
    "KADA": "Kelantan",
    "IADA Kemasin Semerak": "Kelantan",
    "KETARA": "Terengganu",
    "IADA KETARA": "Terengganu",
    "IADA Pekan": "Pahang",
    "IADA Rompin": "Pahang",
}


def apply_botanical_theme():
    """Warm ivory and botanical olive-lime palette with distinct #E7ECE0 KPI cards and equal-height row frames."""
    st.markdown(
        """
        <style>
        .stApp { background-color: #F7F8F3; color: #0B2117; }
        section[data-testid="stSidebar"] { background-color: #0B2117; border-right: 1px solid #1E3F30; }
        .sb-tagline { color: #F7F8F3; font-size: 14px; font-weight: 600; line-height: 1.4; margin-bottom: 0.75rem; }
        .sb-section { color: #94A89E; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; margin: 0.5rem 0 0.25rem 0; }

        /* Select boxes keep light surface with dark text */
        section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #0B2117 !important; font-size: 13px !important; }

        /* Role summary footer card */
        .sb-role-box { background: #122E21; border: 1px solid #264B39; border-radius: 10px; padding: 14px; margin-top: 0.75rem; }
        .sb-role-title { color: #9FCE34; font-size: 10.5px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 5px; }
        .sb-role-desc { color: #DCE3D3; font-size: 12px; line-height: 1.45; margin-bottom: 8px; }
        .sb-role-footer { color: #8CA398; font-size: 9.5px; letter-spacing: 0.06em; text-transform: uppercase; }

        /* Force every horizontal column block and child wrapper to stretch equally */
        div[data-testid="stHorizontalBlock"] {
            align-items: stretch !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            display: flex !important;
            flex-direction: column !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] > div {
            flex: 1 1 auto !important;
            display: flex !important;
            flex-direction: column !important;
        }

        /* Restored previous sage-tinted KPI card design (#E7ECE0) with uniform height */
        div[data-testid="stMetric"] {
            background-color: #E7ECE0 !important;
            border: 1px solid #C8D4BC !important;
            border-radius: 14px !important;
            padding: 16px 18px !important;
            min-height: 124px !important;
            height: 100% !important;
            box-sizing: border-box !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            box-shadow: 0 2px 8px rgba(11, 33, 23, 0.04) !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #0B2117 !important;
            font-size: 13px !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #769D18 !important;
            font-size: 25px !important;
            font-weight: 800 !important;
            line-height: 1.2 !important;
        }

        /* Primary action buttons */
        section[data-testid="stMain"] div.stButton > button {
            background-color: #9FCE34;
            color: #0B2117 !important;
            font-weight: 700;
            border: none;
            border-radius: 10px;
            padding: 0.65rem 1.2rem;
            transition: 0.2s ease;
        }
        section[data-testid="stMain"] div.stButton > button:hover {
            background-color: #769D18;
            color: #FFFFFF !important;
        }
        section[data-testid="stMain"] div.stButton > button:disabled {
            background-color: #DCE3D3;
            color: #6B7A70 !important;
        }

        /* Bottom workspace cards: fixed equal height so left and right frames match identically */
        .pace-card {
            background: #FFFFFF;
            border: 1px solid #C8D4BC;
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 12px;
            min-height: 195px;
            height: 195px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            box-shadow: 0 2px 10px rgba(11, 33, 23, 0.04);
        }
        .pace-card p {
            margin: 0 0 10px 0;
            line-height: 1.45;
        }
        .pace-card ul {
            margin: 0;
            padding-left: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    inject_gemini_sidebar_css()


def _granary_state(D, granary):
    g_state = _GRANARY_STATE_LOOKUP.get(granary)
    if g_state is None and hasattr(D, "districts") and "State" in D.districts.columns:
        m = D.districts[D.districts["District"] == granary]
        if len(m):
            g_state = str(m.iloc[0]["State"])
    return g_state


def _sync_sidebar_granary_to_map(D):
    cur = st.session_state.get("granary")
    prev = st.session_state.get("_last_synced_granary")
    if cur and cur != prev:
        for k in ["tl_bundle", "tl_assessment", "tl_eval_results", "tl_choice", "tl_all_donor_ranking", "automl_summary"]:
            st.session_state.pop(k, None)
        g_state = _granary_state(D, cur)
        if g_state:
            st.session_state["map_state_filter"] = g_state
        st.session_state["_last_synced_granary"] = cur


apply_botanical_theme()
D = load()

if "_pending_granary" in st.session_state:
    st.session_state["granary"] = st.session_state.pop("_pending_granary")

if st.session_state.get("granary") not in D.ALL:
    st.session_state["granary"] = "IADA Barat Laut Selangor" if "IADA Barat Laut Selangor" in D.ALL else D.ALL[0]

OVERVIEW_MODE = "Overview · National Yield & Health"
WORKSPACE_ROLES = [
    "KPKM Programme Manager",
    "Model Validator & Data Steward",
]
VALID_MODES = {OVERVIEW_MODE, *WORKSPACE_ROLES}

st.session_state.setdefault("nav_mode", OVERVIEW_MODE)
if st.session_state["nav_mode"] in ("Overview", "Overview · Summary & Model Status") or st.session_state["nav_mode"] not in VALID_MODES:
    st.session_state["nav_mode"] = OVERVIEW_MODE

st.session_state.setdefault("kpkm_page_idx", 0)
st.session_state.setdefault("validator_step", 1)
st.session_state.setdefault("map_state_filter", _granary_state(D, st.session_state["granary"]) or "All States")
st.session_state.setdefault("_last_synced_granary", st.session_state["granary"])
st.session_state.setdefault("scroll_top", False)

if st.session_state["scroll_top"]:
    components.html("<script>const d=window.parent.document;const e=d.querySelector('section[data-testid=\"stMain\"]')||d.querySelector('section.main');if(e){e.scrollTo(0,0);}</script>", height=0)
    st.session_state["scroll_top"] = False

KPKM_PAGES = [
    "Page 1 · Granary Field Map",
    "Page 2 · Yield & Model Accuracy",
    "Page 3 · Forecast & What-If Simulator",
    "Page 4 · Prescriptive Action Plan",
]
VALIDATOR_STEPS = [
    "Step 1 · Data Quality & Map",
    "Step 2 · Comparable Regions & Drivers",
    "Step 3 · Forecast Accuracy Test",
    "Step 4 · Safety Check & Publish",
]
STEP_TITLES = {
    2: "Step 2 · Find Comparable Regions & Key Yield Drivers",
    3: "Step 3 · Test Forecast Accuracy on Past Seasons",
    4: "Step 4 · Safety Check & Publish Forecast",
}

# ----------------------------------------------------------------------
# Sidebar Navigation
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sb-tagline">A clearer view.<br>Better field decisions.</div>', unsafe_allow_html=True)

    if render_gemini_nav_item(
        OVERVIEW_MODE,
        key="nav_ov",
        active=(st.session_state["nav_mode"] == OVERVIEW_MODE),
    ):
        st.session_state["nav_mode"] = OVERVIEW_MODE
        st.session_state["scroll_top"] = True
        st.rerun()

    st.session_state["granary"] = st.selectbox(
        "Selected Granary",
        D.ALL,
        index=D.ALL.index(st.session_state["granary"]),
    )
    _sync_sidebar_granary_to_map(D)

    cur_mode = st.session_state["nav_mode"]
    role_idx = WORKSPACE_ROLES.index(cur_mode) if cur_mode in WORKSPACE_ROLES else None
    chosen_role = st.selectbox(
        "Workspace",
        WORKSPACE_ROLES,
        index=role_idx,
        placeholder="Select a role workspace...",
    )
    if chosen_role is not None and chosen_role != cur_mode:
        st.session_state["nav_mode"] = chosen_role
        st.session_state["scroll_top"] = True
        st.rerun()

    mode = st.session_state["nav_mode"]

    if mode == "KPKM Programme Manager":
        st.markdown('<div class="sb-section">PAGES</div>', unsafe_allow_html=True)
        for idx, label in enumerate(KPKM_PAGES):
            if render_gemini_nav_item(
                label,
                key=f"nav_kpkm_{idx}",
                active=(st.session_state["kpkm_page_idx"] == idx),
            ):
                st.session_state["kpkm_page_idx"] = idx
                st.session_state["scroll_top"] = True
                st.rerun()
        role_title, role_desc = "PROGRAMME MANAGER", "Monitor paddy fields, coordinate field teams and review programme evidence."

    elif mode == "Model Validator & Data Steward":
        st.markdown('<div class="sb-section">STEPS</div>', unsafe_allow_html=True)
        for s_idx, label in enumerate(VALIDATOR_STEPS, start=1):
            if render_gemini_nav_item(
                label,
                key=f"nav_val_{s_idx}",
                active=(st.session_state["validator_step"] == s_idx),
            ):
                st.session_state["validator_step"] = s_idx
                st.session_state["scroll_top"] = True
                st.rerun()
        role_title, role_desc = "MODEL VALIDATOR & STEWARD", "Verify data quality, test forecast accuracy, and publish approved models."

    else:
        role_title, role_desc = "NATIONAL OVERVIEW", f"Big-picture yield, weather drivers, and model readiness across all {len(D.ALL)} granaries."

    st.markdown(
        f'<div class="sb-role-box"><div class="sb-role-title">{role_title}</div><div class="sb-role-desc">{role_desc}</div>'
        '<div class="sb-role-footer">MALAYSIA / PADDY SYSTEMS</div></div>',
        unsafe_allow_html=True,
    )

st.session_state["workspace_role"] = ROLE_VAL if mode == "Model Validator & Data Steward" else ROLE_KPKM


def _short_winner(label: str, max_len: int = 20) -> str:
    """Keep top KPI card subtitles concise so all cards stay uniform."""
    s = str(label or "n/a")
    return s if len(s) <= max_len else s[: max_len - 1].rstrip() + "…"


def _render_automl_score_strip(D):
    """Run AutoML automatically from Step 1 and display balanced 0–100% stage scorecards."""
    ds = D.DATA_DIR / "transfer_stage_dataset.csv"
    dy = D.DATA_DIR / "transfer_yield_dataset.csv"
    mtime = (ds if ds.exists() else dy).stat().st_mtime if (ds.exists() or dy.exists()) else 0.0
    auto = run_full_automl_for_granary(st.session_state["granary"], str(D.DATA_DIR), mtime, allow_stale=True)

    st.session_state["tl_bundle"] = auto["bundle"]
    st.session_state["tl_assessment"] = auto["assessment"]
    st.session_state["tl_eval_results"] = auto["eval_results"]
    st.session_state["tl_choice"] = auto["choice"]
    st.session_state["tl_all_donor_ranking"] = auto["donor_table"]
    st.session_state["automl_summary"] = auto

    tab_transfer.render_refresh_controls(st.session_state["granary"])
    if auto.get("data_fixes"):
        st.warning(
            f"Data check: {auto['data_fixes']} out-of-range vegetation value(s) in the stage table were treated as missing before modelling."
        )
    sc = auto["scores"]
    gate_ok = bool(auto["choice"]["safeguard"]["gate_passed"])
    st.markdown(f"#### Quality & Forecast Readiness Scorecard — {st.session_state['granary']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "1. Data Coverage Score",
        f"{sc['step1_data']:.1f}%",
        "Cloud-free optical + yield",
        help="Combined satellite clarity and historical yield completeness.",
    )
    c2.metric(
        "2. Region Similarity",
        f"{sc['step2_domain']:.1f}%",
        _short_winner(sc["donor_winner"][0]),
        help=f"Best comparable donor pool: {sc['donor_winner'][0]}",
    )
    c3.metric(
        "3. Forecast Skill Score",
        f"{sc['step3_automl']:.1f}%",
        _short_winner(sc["model_winner"][0]),
        help=f"Top-scoring model architecture: {sc['model_winner'][0]}",
    )
    c4.metric(
        "4. Safety Gate Status",
        "PASS" if gate_ok else "LOCAL SAFE",
        f"Readiness: {sc['step4_gate']:.1f}%",
        help="PASS = Transfer learning improves accuracy; LOCAL SAFE = Uses local baseline to prevent negative transfer.",
    )

    with st.expander("View Detailed Model & Donor Competition Breakdown (0–100%)", expanded=False):
        w1, w2, w3, w4 = st.columns(4)
        w1.info(f"**Best Donor Pool ({sc['donor_winner'][1]}%)**\n\n{sc['donor_winner'][0]}")
        w2.info(f"**Best Model ({sc['model_winner'][1]}%)**\n\n{sc['model_winner'][0]}")
        w3.info(f"**Best Feature Set ({sc['feature_winner'][1]}%)**\n\n{sc['feature_winner'][0]}")
        w4.info(f"**Best Scaling ({sc['scaling_winner'][1]}%)**\n\n{sc['scaling_winner'][0]}")

        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Comparable Region Similarity Ranking (%)**")
            st.dataframe(auto["donor_table"], hide_index=True, width="stretch")
        with t2:
            st.markdown("**Forecast Model Leaderboard (%)**")
            st.dataframe(auto["model_table"], hide_index=True, width="stretch")
    st.divider()
    return auto


def _go(step=None, mode_=None):
    if step is not None:
        st.session_state["validator_step"] = step
    if mode_ is not None:
        st.session_state["nav_mode"] = mode_
    st.session_state["scroll_top"] = True
    st.rerun()


# ----------------------------------------------------------------------
# Route views
# ----------------------------------------------------------------------
if mode == OVERVIEW_MODE:
    tab_overview.render_overview_hub(D)

elif mode == "KPKM Programme Manager":
    page_idx = st.session_state["kpkm_page_idx"]
    if page_idx == 0:
        tab_map.render_leaflet_map(D, role="kpkm")
    elif page_idx == 1:
        tab_yield.render_yield_view("historical", D)
    elif page_idx == 2:
        tab_yield.render_yield_view("outlook", D)
    else:
        tab_advisory.render_decision_brief(D)

else:
    # Locked CSV ingestion & continuous retraining panel (set allow_upload=True to unlock)
    tab_transfer.render_steward_ingestion_panel(D, allow_upload=False)

    step = st.session_state["validator_step"]
    if step in STEP_TITLES:
        st.markdown(f"## {STEP_TITLES[step]}")

    _render_automl_score_strip(D)

    if step == 1:
        tab_map.render_leaflet_map(D, role="validator", title="Step 1 · Check Satellite & Harvest Data Coverage")
        st.divider()
        if st.button("Next Step: Find Comparable Regions & Key Yield Drivers →", key="nx_1", width="stretch"):
            _go(2)
    elif step == 2:
        tab_transfer.render_domain_audit(D)
        st.divider()
        c1, c2 = st.columns([1, 2])
        if c1.button("← Back to Spatial Map", key="bk_2", width="stretch"):
            _go(1)
        if c2.button("Next Step: Test Forecast Accuracy on Past Seasons →", key="nx_2", width="stretch"):
            _go(3)
    elif step == 3:
        tab_transfer.render_benchmark_audit(D)
        st.divider()
        c1, c2 = st.columns([1, 2])
        if c1.button("← Back to Comparable Regions", key="bk_3", width="stretch"):
            _go(2)
        if c2.button("Testing Complete — Proceed to Step 4: Safety Check & Publish →", key="nx_3", width="stretch"):
            _go(4)
    else:
        tab_transfer.render_validator_signoff(D)
        st.divider()
        c1, c2 = st.columns([1, 2])
        if c1.button("← Back to Accuracy Test", key="bk_4", width="stretch"):
            _go(3)
        if c2.button("Finish & Return to National Overview →", key="nx_4", width="stretch"):
            _go(mode_=OVERVIEW_MODE)