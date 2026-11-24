"""Component 1 · Overview: what the dashboard is, the questions it answers, how it works, and what to keep in mind.

The wording lives here; the names and credits live in about_config.py. Numbers on the page come from the loaded data, so they stay true
when the data is refreshed.
"""
from pathlib import Path
import pandas as pd
import streamlit as st

from . import store, tab_reliability
from .about_config import APP_NAME, CLOSING, PACE_ROWS, RESEARCH, RESEARCH_PROBLEM, SCOPE_ROWS, TAGLINE


def _card(col, icon, title, body, where):
    with col.container(border=True):
        st.markdown(f"#### {icon} {title}")
        st.write(body)
        st.caption(where)


def render_freshness_closed_loop(D):
    """Section 5 · Closed-loop freshness warning, KPKM meeting coordination drawer, and Data Steward ingestion & retraining drawer."""
    from acts_transfer.freshness_loop import (
        append_coordination_request,
        check_data_freshness,
        load_update_log,
        merge_new_yield_csv,
        run_retraining_pipeline,
    )

    y_df = getattr(D, "yh", pd.DataFrame())
    status = check_data_freshness(y_df)
    if not status["is_stale"]:
        return

    missing_str = ", ".join(map(str, status["missing_years"]))
    log = load_update_log()
    pending = [r for r in log.get("requests", []) if r.get("status") == "PENDING_STEWARD"]

    st.markdown("### 🔄 Data Freshness & Closed-Loop Retraining")
    c_val, c_kpkm = st.columns(2)

    with c_val:
        st.warning(
            f"**Model Validator & Data Steward — Freshness & Drift Warning:** "
            f"Latest dataset year is **{status['max_data_year']}** (system year **{status['current_year']}**, "
            f"lag = **{status['lag_years']} year(s)**; missing: **{missing_str}**). "
            f"Pending Programme Manager requests: **{len(pending)}**."
        )
        with st.expander("Data Steward Ingestion & Pipeline Retraining Drawer (Step 1–4 Reset)", expanded=bool(pending)):
            if pending:
                st.markdown("**Pending Programme Manager Requests (`results/data_update_log.json`):**")
                for req in pending:
                    st.info(
                        f"**[{req['id']} · {req['granary']}]** Items: {', '.join(req.get('requested_items', []))} — "
                        f"{req.get('meeting_notes', '')}"
                    )
            up_yield = st.file_uploader(
                "1. New annual harvest yield CSV (merges/deduplicates on ['District', 'Year'])",
                type=["csv"],
                key="ov_loop_yield_up",
            )
            up_daily = st.file_uploader(
                "2. Multi-file polygon_daily/granary_daily_*.csv uploader",
                type=["csv"],
                accept_multiple_files=True,
                key="ov_loop_daily_up",
            )
            if st.button("Save Dataset, Run Pipeline & Restart Step 1–4 Loop", type="primary", width="stretch", key="ov_run_loop_btn"):
                if up_yield is not None:
                    merge_new_yield_csv(pd.read_csv(up_yield), data_dir=D.DATA_DIR)
                if up_daily:
                    pdir = Path("polygon_daily")
                    pdir.mkdir(exist_ok=True)
                    for f in up_daily:
                        (pdir / f.name).write_bytes(f.getvalue())
                ok, step_logs = run_retraining_pipeline(automl_note="AutoML default config applied")
                if ok:
                    st.cache_data.clear()
                    st.session_state["validator_step"] = 1
                    st.success("Dataset saved, pipeline executed, and Validator Step 1–4 loop reset.")
                    st.rerun()
                else:
                    st.error("Pipeline execution encountered an error:\n" + "\n".join(step_logs))

    with c_kpkm:
        st.info(
            f"**KPKM Programme Manager — Operational Notice:** "
            f"Harvest records currently run through **{status['max_data_year']}** (missing **{missing_str}**). "
            "Pair baseline recommendations with ground verification and log regional IADA updates below."
        )
        with st.expander("Programme Manager Meeting Coordination Drawer", expanded=False):
            scope = st.radio(
                "Request scope",
                ["Selected Granary", "All 10 Granaries"],
                horizontal=True,
                key="ov_kpkm_req_scope",
            )
            items = st.multiselect(
                "Available regional IADA items",
                ["Official Yield kg/ha", "Planting/Canal Calendar", "Field Notes"],
                default=["Official Yield kg/ha"],
                key="ov_kpkm_req_items",
            )
            notes = st.text_area("Meeting notes & agreed field measures", key="ov_kpkm_req_notes")
            if st.button("Save Meeting Notes & Request to Data Steward Log", key="ov_kpkm_req_save", width="stretch"):
                entry = append_coordination_request(
                    scope=scope,
                    granary=st.session_state.get("granary", "MADA"),
                    items=items,
                    meeting_notes=notes,
                )
                st.success(f"Saved request **{entry['id']}** to `results/data_update_log.json` for the Data Steward.")


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
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }
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
    .info-card-icon {
        font-size: 26px;
        margin-bottom: 10px;
    }
    .info-card h4 {
        font-size: 17px;
        line-height: 1.4;
        margin: 0 0 10px 0;
        min-height: 48px;
    }
    .info-card p {
        font-size: 14px;
        line-height: 1.5;
        margin: 0 0 12px 0;
        flex-grow: 1;
    }
    .info-card-tab {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        margin-top: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    c = st.columns(4)
    _card(c[0], "🕘", "What happened?", f"Explore the yield each granary reported, year by year ({y0} to {y1}).", "Page 2 · Historical Yield & Model Accuracy")
    _card(c[1], "🔮", "What happen next?", "See an outlook for the next 1 to 4 years, with a range around it.", "Page 3 · Yield Forecast & Weather What-If Simulator")
    _card(c[2], "🛰️", "What is happening?", "Examine crop condition, the vegetation forecast, resilience, weather and soil.", "Page 1 · Granary Field Map")
    _card(c[3], "💡", "What should do?", "Review draft interventions: who should act, by when and why.", "Page 4 · Prescriptive Action Plan")

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
        st.markdown("**🔬 Assess, then transfer**  \nSteps 1 to 4: compare source and target granaries, harmonise the data, run AutoML transfer evaluation, and sign off.")
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


def _render_yield_charts(df):
    """Two charts for the overview: (1) reported yield across the years for every granary, (2) a ranking of the granaries, so the best, the worst
    and the granaries closest to the selected one can be read at a glance. Uses the same granary-year table as the rest of the hub."""
    import plotly.graph_objects as go

    d = df[["District", "Year", "Yield_kg_ha"]].dropna()
    if d.empty:
        return
    sel = st.session_state.get("granary")
    years = sorted((int(y) for y in d.Year.unique()), reverse=True)

    st.markdown("### Yield Across the Years, by Granary")
    st.caption("One line per granary (reported yield, kg/ha). The granary selected in the sidebar is drawn thick; click a name in the legend to hide or show a line.")
    fig = go.Figure()
    for g in sorted(d.District.unique()):
        s = d[d.District == g].sort_values("Year")
        on = g == sel
        fig.add_scatter(x=s.Year, y=s.Yield_kg_ha, mode="lines+markers", name=g, opacity=1.0 if on else 0.55,
                        line=dict(width=4 if on else 1.5), marker=dict(size=9 if on else 5),
                        hovertemplate="%{x}: %{y:,.0f} kg/ha<extra>" + g + "</extra>")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="Yield (kg/ha)", xaxis=dict(dtick=1), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch", key="ov_yield_lines")

    st.markdown("### Granary Ranking by Yield")
    rank_options = ["Average of all years"] + [f"Year {y}" for y in years]
    basis = st.selectbox(
        "Rank by",
        rank_options,
        index=0,
        key="ov_rank_basis",
        help="Choose 'Average of all years' or select any individual year (e.g., 2025, 2024, ..., 2017) to compare granary rankings.",
    )
    if basis == "Average of all years":
        r = d.groupby("District").Yield_kg_ha.mean().dropna().sort_values(ascending=False)
        period_label = "all-year average"
    else:
        chosen_year = int(basis.replace("Year ", ""))
        r = d[d.Year == chosen_year].set_index("District").Yield_kg_ha.dropna().sort_values(ascending=False)
        period_label = f"year {chosen_year}"

    n = len(r)
    best, worst = r.index[0], r.index[-1]
    colour = ["#2e7d32" if g == best else "#c62828" if g == worst else "#1f6fb2" if g == sel else "#9aa5a0" for g in r.index]
    bar = go.Figure(go.Bar(x=r.values, y=[f"#{i + 1}  {g}" for i, g in enumerate(r.index)], orientation="h", marker_color=colour,
                           marker_line_color=["#000000" if g == sel else "rgba(0,0,0,0)" for g in r.index], marker_line_width=[3 if g == sel else 0 for g in r.index],
                           text=[f"{v:,.0f}" for v in r.values], textposition="outside", hovertemplate="%{y}: %{x:,.0f} kg/ha<extra></extra>"))
    bar.update_layout(height=max(320, 46 * n), margin=dict(l=10, r=40, t=10, b=10), xaxis_title=f"Yield (kg/ha) — {period_label}", yaxis=dict(autorange="reversed"), showlegend=False)
    bar.update_xaxes(range=[0, float(r.max()) * 1.12])
    st.plotly_chart(bar, width="stretch", key="ov_yield_rank")
    st.caption(f"Ranking for **{period_label}**. Green = best, red = worst, blue = the selected granary (outlined). Best: **{best}** ({r.iloc[0]:,.0f} kg/ha). Worst: **{worst}** ({r.iloc[-1]:,.0f} kg/ha). "
               f"Gap between them: {r.iloc[0] - r.iloc[-1]:,.0f} kg/ha.")
    if sel in r.index:
        i, v = list(r.index).index(sel), float(r[sel])
        up = r.index[i - 1] if i > 0 else None
        dn = r.index[i + 1] if i < n - 1 else None
        msg = [f"**{sel}** ranks **{i + 1} of {n}** ({period_label}) at {v:,.0f} kg/ha."]
        msg.append(f"Next above: **{up}** ({r[up] - v:+,.0f} kg/ha)." if up else "It is the top granary, so nothing is above it.")
        msg.append(f"Next below: **{dn}** ({r[dn] - v:+,.0f} kg/ha)." if dn else "It is the lowest granary, so nothing is below it.")
        near = [g for g in (up, dn) if g]
        if near:
            c = min(near, key=lambda g: abs(float(r[g]) - v))
            msg.append(f"Closest overall: **{c}**, {abs(float(r[c]) - v):,.0f} kg/ha away.")
        st.info(" ".join(msg))

# ------------------------------------------------------------------------------------------------ PACE V10 overview hub
def render_overview_hub(D):
    """Executive overview and governance status: the entry page."""
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

    # Section 5 · Closed-loop freshness & retraining drawers
    render_freshness_closed_loop(D)

    _render_yield_charts(df)

    st.markdown("### Workspace Summaries")
    left, right = st.columns(2)
    top = latest.loc[latest.Yield_kg_ha.idxmax()]
    with left:
        st.markdown(
            f"""<div class="pace-card"><h4 style="color:#0B2117;margin-top:0;">🌾 KPKM Programme Manager Summary</h4>
            <p style="color:#43544B;">Monitor spatial field conditions, heading-stage (70 days after sowing) nowcasts and 2025–2028 climate scenario pathways across the Peninsular Malaysia granaries.</p>
            <ul><li><b>Active Focus Granary:</b> {html.escape(str(st.session_state.get('granary', '')))}</li>
            <li><b>Highest {latest_year} Yield:</b> {html.escape(str(top.District))} ({top.Yield_kg_ha:,.0f} kg/ha)</li></ul></div>""",
            unsafe_allow_html=True,
        )
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
            <li><b>Safeguard Gate Status:</b> {n_app} Approved | {n_loc} Local Baseline ({n_pend} pending review)</li></ul></div>""",
            unsafe_allow_html=True,
        )
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