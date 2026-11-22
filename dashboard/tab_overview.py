"""Component 1 · Overview: what the dashboard is, the questions it answers, how it works, and what to keep in mind.

The wording lives here; the names and credits live in about_config.py. Numbers on the page come from the loaded data, so they stay true
when the data is refreshed.
"""
import pandas as pd
import streamlit as st

from . import store, tab_reliability
from .about_config import APP_NAME, CLOSING, PACE_ROWS, RESEARCH, RESEARCH_PROBLEM, SCOPE_ROWS, TAGLINE


def _card(col, icon, title, body, where):
    with col.container(border=True):
        st.markdown(f"#### {icon} {title}")
        st.write(body)
        st.caption(where)


def render(D):
    n_g = len(D.ALL)
    y0, y1 = int(D.yh.Year.min()), int(D.yh.Year.max())
    st.title(f"About {APP_NAME}")
    st.subheader(TAGLINE)
    st.markdown(
        f"{APP_NAME} helps KPKM and the granary authorities understand paddy yield, follow crop condition, look ahead, and record what to do. "
        f"It brings the reported yield, satellite vegetation data, weather and soil information of {n_g} granaries together in one interactive "
        "dashboard.")

    st.markdown("""
    <style>
    /* Make all four columns equal in width */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }

    /* Give every card the same height and layout */
    .info-card {
        height: 230px;
        box-sizing: border-box;
        padding: 20px;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        background-color: white;

        display: flex;
        flex-direction: column;
    }

    /* Icon */
    .info-card-icon {
        font-size: 26px;
        margin-bottom: 10px;
    }

    /* Card title */
    .info-card h4 {
        font-size: 17px;
        line-height: 1.4;
        margin: 0 0 10px 0;
        min-height: 48px;
    }

    /* Description */
    .info-card p {
        font-size: 14px;
        line-height: 1.5;
        margin: 0 0 12px 0;
        flex-grow: 1;
    }

    /* Tab label stays at the bottom */
    .info-card-tab {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        margin-top: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    c = st.columns(4)
    _card(c[0], "🕘", "What happened?", f"Explore the yield each granary reported, year by year ({y0} to {y1}).", "Tab 6 · Yield Prediction")
    _card(c[1], "🔮", "What happen next?", "See an outlook for the next 1 to 4 years, with a range around it.", "Tab 7 · Future Yield Outlook")
    _card(c[2], "🛰️", "What is happening?", "Examine crop condition, the vegetation forecast, resilience, weather and soil.", "Tab 2 · Dynamic maps")
    _card(c[3], "💡", "What should do?", "Review draft interventions: who should act, by when and why.", "Tab 8 · Intervention Advisory")

    st.markdown("### Research problem and scope")
    st.write(RESEARCH_PROBLEM)
    st.dataframe(pd.DataFrame(SCOPE_ROWS, columns=["Item", "Scope"]), hide_index=True, width="stretch")
    with st.expander("How the project is organised (PACE: Plan, Analyze, Construct, Execute)"):
        st.dataframe(pd.DataFrame(PACE_ROWS, columns=["Stage", "Dashboard tab(s)", "What happens"]), hide_index=True, width="stretch")

    st.markdown("### What you can do")
    a, b = st.columns(2)
    with a:
        st.markdown("**📊 Explore reported yield**  \nCompare granaries and years on maps, rankings and trend charts.")
        st.markdown("**🌱 Understand crop condition**  \nSee whether each crop is behind, on track or ahead, its estimated stage, the 10 and 30 day "
                    "vegetation forecast, and how resilient the granary is.")
        st.markdown("**📈 See the yield outlook**  \nA range for the next 1 to 4 years, with the typical error measured in back-tests.")
    with b:
        st.markdown("**💡 Get advice and a report**  \nDraft interventions with owners and timing, an optional AI-written note, and an editable "
                    "Word report.")
        st.markdown("**✅ Record decisions and follow up**  \nAccept, modify, reject or defer each intervention, then track status and "
                    "overdue items.")
        st.markdown("**🔬 Assess, then transfer**  \nTabs 3 to 5: compare a source and a target granary, harmonise the data, see which transfer method suits "
                    "the evidence, run it, and compare it with baselines.")
        st.markdown("**🧪 Check the models**  \nSee the results without and with transfer learning, and the measured errors.")

    st.markdown("### How it works")
    st.info("**Yield, satellite, weather and soil data  →  Granary records  →  Analytics  →  Prediction  →  Advisory rules  →  "
            "AI wording (optional)  →  KPKM decision and follow-up**")
    st.caption("The granary records hold one entry per granary and 10-day period (satellite) or per granary and year (yield). The dashboard "
               "supports decisions; it is not a real-time crop simulator.")

    st.markdown("### Who can benefit?")
    st.write("KPKM programme managers and granary authorities, who decide where to look first and what to do. Researchers and students can "
             "use it to see how the forecasts and the yield outlook are built and tested.")

    with st.expander("🔬 Data, technology and research"):
        st.markdown(
            "**Data**\n"
            f"- Reported yield per granary and year, {y0} to {y1} (file supplied for the project).\n"
            "- Optical satellite: Sentinel-2 surface reflectance (NDVI, EVI, LSWI, MNDWI), cropland pixels only (ESA WorldCover 2021).\n"
            "- Radar satellite: Sentinel-1 (VV backscatter), which works through cloud.\n"
            "- Weather: ERA5-Land (temperature, rainfall, solar radiation).\n"
            "- Soil: OpenLandMap topsoil pH.\n"
            "- Boundaries: Malaysian district and state boundaries; each granary is drawn as the districts it covers.\n\n"
            "**Methods**\n"
            "- Vegetation forecast (10 and 30 days): ridge regression and gradient boosting, checked year by year on years the model had not seen.\n"
            "- Crop status and resilience: measured against the range that forecast expects.\n"
            "- Yield outlook: transfer learning, which learns how much one year's yield carries into the next from the other granaries.\n"
            "- Advice: editable rules, each tied to a published source, and an optional AI model that only rewrites the result in plain language.\n"
            "- Decisions are stored in a database; the Word report is generated from the same records.")

    st.markdown("### ⚠️ Important notes")
    notes = []
    if D.AS_OF:
        notes.append(f"The satellite data ends **{D.AS_OF}**. Crop status, forecasts and advice describe that date, not today, until the data is refreshed.")
    notes += [
        f"Reported yield has only {y1 - y0 + 1} yearly values per granary, and its definition may differ from official statistics: check it with the data owner.",
        "The yield outlook was about as accurate as 'same as last year' in back-tests (about 8% error at 1 year, 11% at 2, 14% at 3). Treat it as a planning range.",
        "Crop status and resilience are signals that something differs from expected. They cannot say why, so each intervention is something to check, ask or prepare.",
        "Interventions are drafts from rules that an agronomist has not yet reviewed. The estimated growth stage comes from the planting calendar, not field observation.",
        "On the maps each granary is the district(s) it covers; the satellite values come only from sampled paddy pixels inside them.",
    ]
    for n in notes:
        st.markdown(f"- {n}")

    st.markdown("### Research and development")
    for line in RESEARCH:
        st.markdown(line)

    with st.expander("How reliable is this? (model track record and known limits)"):
        tab_reliability.render(D)

    st.markdown(f"##### {CLOSING}")
    n_s = len(D.series) if D.series is not None else 0
    st.caption(f"Loaded {len(D.yh)} granary-year yield records ({y0} to {y1}) from yield_history.csv"
               + (f" and {n_s:,} ten-day satellite records from ndvi_district_decade.csv" if n_s else "")
               + f". Decisions you record are saved to: {store.describe(D.DATA_DIR)}.")


# ------------------------------------------------------------------------------------------------ PACE V9 overview hub
def render_overview_hub(D):
    """Executive overview and governance status: the entry page. KPIs, one summary card per workspace with a jump button, and the national snapshot with each granary's registry state."""
    import html

    from acts_transfer import governance as gov

    from .tab_map import _map_table
    st.markdown("## Executive Overview & Governance Status")
    st.caption("Unified cross-role summary across KPKM Programme operations and Model Validator governance.")
    ds, dy = D.DATA_DIR / "transfer_stage_dataset.csv", D.DATA_DIR / "transfer_yield_dataset.csv"
    if not (ds.exists() or dy.exists()):
        st.warning("The granary-year table is missing, so the overview has no yield figures.")
        return
    df = _map_table(str(ds), str(dy), (ds if ds.exists() else dy).stat().st_mtime)
    latest_year = int(df.Year.max())
    latest = df[df.Year == latest_year]
    reg = gov.load()
    n_app, n_loc, n_pend = gov.counts(D.ALL, reg)
    total = len(D.ALL)
    if reg.get("error"):
        st.warning(reg["error"])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Latest Reference Year", f"{latest_year}")
    k2.metric("Mean Granary Yield", f"{latest.Yield_kg_ha.mean():,.0f} kg/ha", help="Simple average over granaries. Planted area is not available, so it is not area-weighted.")
    k3.metric("Monitored Granaries", f"{total} Schemes")
    k4.metric("Mean Optical Coverage", f"{latest.Optical_Valid_Fraction.mean() * 100:.1f}%", help="Share of ten-day optical observations that were really observed, averaged over granaries.")
    k5.metric("Verified Transfer Models", f"{n_app} / {total}", help="Granaries whose Detrended Residual Transfer forecast a Model Validator has signed off.")

    st.markdown("### Workspace Summaries")
    left, right = st.columns(2)
    top = latest.loc[latest.Yield_kg_ha.idxmax()]
    with left:
        st.markdown(
            f"""<div class="pace-card"><h4 style="color:#0B2117;margin-top:0;">🌾 KPKM Programme Manager Summary</h4>
            <p style="color:#43544B;">Monitor spatial field conditions, heading-stage (70 days after sowing) nowcasts and 2025–2028 climate scenario pathways across the Peninsular Malaysia granaries.</p>
            <ul><li><b>Active Focus Granary:</b> {html.escape(str(st.session_state.get('granary', '')))}</li>
            <li><b>Highest {latest_year} Yield:</b> {html.escape(str(top.District))} ({top.Yield_kg_ha:,.0f} kg/ha)</li></ul></div>""", unsafe_allow_html=True)
        if st.button("Enter KPKM Programme Manager Workspace →", key="jump_kpkm"):
            st.session_state["nav_mode"] = "KPKM Programme Manager"
            st.session_state["scroll_top"] = True
            st.rerun()
    with right:
        ovf = latest.Optical_Valid_Fraction.mean() * 100
        st.markdown(
            f"""<div class="pace-card"><h4 style="color:#0B2117;margin-top:0;">🛡️ Model Validator & Data Steward Summary</h4>
            <p style="color:#43544B;">Audit multi-sensor cloud coverage, screen East Coast monsoon outliers (KETARA), and approve Detrended Residual Transfer models.</p>
            <ul><li><b>Mean {latest_year} Optical Coverage:</b> {ovf:.1f}%</li>
            <li><b>Safeguard Gate Status:</b> {n_app} Approved | {n_loc} Local Baseline ({n_pend} pending review)</li></ul></div>""", unsafe_allow_html=True)
        if st.button("Enter Model Validator Workflow →", key="jump_validator"):
            st.session_state["nav_mode"] = "Model Validator & Data Steward"
            st.session_state["validator_step"] = 1
            st.session_state["scroll_top"] = True
            st.rerun()

    st.markdown("### National Granary Governance & Yield Snapshot")
    snap = latest[["District", "Year", "Yield_kg_ha", "Optical_Valid_Fraction"]].copy()
    snap["Yield (t/ha)"] = (snap.Yield_kg_ha / 1000).round(2)
    snap["Optical Coverage (%)"] = (snap.Optical_Valid_Fraction * 100).round(1)
    snap["Governance Status"] = snap.District.map(lambda d: gov.effective(d, reg)["status"])
    st.dataframe(snap[["District", "Year", "Yield (t/ha)", "Yield_kg_ha", "Optical Coverage (%)", "Governance Status"]], width="stretch", hide_index=True)
    with st.expander("About the project, the research problem and how the work is organised (PACE)"):
        render(D)
