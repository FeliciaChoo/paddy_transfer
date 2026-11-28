"""Component 1 · Overview: Big-to-Small-Picture National Paddy Yield & Crop Health Overview.

Layout progression (Macro -> Meso -> Micro):
  1. Macro: 5 equal-height national KPIs (#E7ECE0 sage cards) + Key Takeaway banner + compact freshness strip.
  2. Meso (Drivers): Why Did Yield Change? Standardised (SD) weather & crop greenness trends across all 9 granaries.
  3. Meso (Comparison): Interactive Granary Spotlight selector, Multi-Year Yield Trajectories,
     Year-Selectable Granary Ranking (All-Year Avg or 2017..latest), and Visual Snapshot Table.
  4. Micro: Equal-height Workspace Launch Cards for KPKM Programme Manager and Model Validator & Data Steward.
"""
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import store, tab_reliability
from .about_config import APP_NAME, CLOSING, PACE_ROWS, RESEARCH, RESEARCH_PROBLEM, SCOPE_ROWS, TAGLINE


def _card(col, icon, title, body, where):
    with col.container(border=True):
        st.markdown(f"#### {icon} {title}")
        st.write(body)
        st.caption(where)


def _load_automl_summary() -> dict:
    """Load results/automl_granary_summary.json if available so Overview displays live ΔNT & winning regimes."""
    p = Path("results/automl_granary_summary.json")
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw.get("granaries", raw)
    except Exception:
        pass
    return {}


def _render_national_anomaly_chart(df: pd.DataFrame):
    """Tier 2 (Meso Drivers): Standardised (Z-score SD) national climate & canopy anomaly chart."""
    st.markdown("### Why Did Yield Change? Weather vs. Crop Greenness Trends")
    st.caption(
        "Each line shows how far above or below normal (in Standard Deviation units, SD) national rainfall, temperature, "
        "satellite greenness, and yield moved each year. Zero (0 SD) is the 2017–latest average."
    )

    panel_path = Path("malaysia_granary_annual_by_granary.csv")
    src = pd.read_csv(panel_path) if panel_path.exists() else df.copy()
    if "Year" not in src.columns:
        return

    candidate_cols = [
        ("Yield_kg_ha", "Reported Yield (kg/ha)", "#0B2117", 3.5),
        ("Rain_mm", "Seasonal Rainfall", "#1f6fb2", 2.0),
        ("Temp_C", "Mean Temperature", "#c62828", 2.0),
        ("EVI_mean", "Crop Greenness (EVI)", "#2e7d32", 2.2),
        ("NDVI_mean", "Crop Greenness (NDVI)", "#2e7d32", 2.2),
        ("LSWI_mean", "Surface Water (LSWI)", "#00838f", 1.8),
        ("Soil_Moisture", "Soil Moisture", "#6a1b9a", 1.8),
    ]

    lower_map = {c.lower(): c for c in src.columns}
    series_list = []
    seen_labels = set()
    for col_hint, label, color, width in candidate_cols:
        matched = lower_map.get(col_hint.lower())
        if matched is None:
            for c in src.columns:
                if col_hint.split("_")[0].lower() in c.lower() and pd.api.types.is_numeric_dtype(src[c]):
                    matched = c
                    break
        if matched and label not in seen_labels:
            yr_mean = src.groupby("Year")[matched].mean().dropna()
            if len(yr_mean) >= 3 and float(yr_mean.std(ddof=0)) > 1e-9:
                z = (yr_mean - yr_mean.mean()) / yr_mean.std(ddof=0)
                series_list.append((label, z, color, width))
                seen_labels.add(label)

    if len(series_list) < 2:
        st.info("Annual weather and satellite anomaly trends will display automatically once multi-year climate columns are loaded.")
        return

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", line_width=1.2, annotation_text="Historical Average (0 SD)")
    for label, z, color, width in series_list:
        fig.add_scatter(
            x=z.index,
            y=z.values,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=width),
            marker=dict(size=8 if width > 2.5 else 5),
            hovertemplate="%{x}: %{y:+.2f} SD<extra>" + label + "</extra>",
        )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=10, t=15, b=10),
        yaxis_title="Deviation from Normal (SD)",
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", y=-0.25),
        plot_bgcolor="#FFFFFF",
    )
    st.plotly_chart(fig, width="stretch", key="ov_national_sd_chart")


def _render_yield_charts(df: pd.DataFrame):
    """Tier 3 (Meso Comparison): Interactive Granary Spotlight + Multi-year lines + Year-selectable Ranking."""
    d = df[["District", "Year", "Yield_kg_ha"]].dropna()
    if d.empty:
        return

    granaries = sorted(d.District.unique())
    cur_sel = st.session_state.get("granary")
    if cur_sel not in granaries:
        cur_sel = granaries[0]

    years = sorted(int(y) for y in d.Year.unique())
    latest = years[-1]

    st.markdown("### Compare Granary Performance Across Seasons")
    picked = st.selectbox(
        "Click to spotlight a granary across all charts below (synced with sidebar)",
        granaries,
        index=granaries.index(cur_sel),
        key="ov_inline_granary_picker",
    )
    if picked != cur_sel:
        st.session_state["_pending_granary"] = picked
        st.rerun()
    sel = picked

    col_line, col_rank = st.columns([1.15, 1.0], gap="large")

    with col_line:
        st.markdown("#### Multi-Year Yield Trajectory (kg/ha)")
        st.caption(f"**{sel}** is highlighted in bold green; click legend items to filter schemes.")
        fig = go.Figure()
        for g in granaries:
            s = d[d.District == g].sort_values("Year")
            on = g == sel
            fig.add_scatter(
                x=s.Year,
                y=s.Yield_kg_ha,
                mode="lines+markers",
                name=g,
                opacity=1.0 if on else 0.42,
                line=dict(width=4 if on else 1.5, color="#5C8210" if on else None),
                marker=dict(size=9 if on else 5),
                hovertemplate="%{x}: %{y:,.0f} kg/ha<extra>" + g + "</extra>",
            )
        fig.update_layout(
            height=410,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Yield (kg/ha)",
            xaxis=dict(dtick=1),
            legend=dict(orientation="h", y=-0.28),
            plot_bgcolor="#FFFFFF",
        )
        st.plotly_chart(fig, width="stretch", key="ov_yield_lines")

    with col_rank:
        st.markdown("#### Granary Ranking by Yield")
        rank_options = [f"Latest year ({latest})", "Average of all years"] + [str(y) for y in reversed(years[:-1])]
        basis = st.selectbox(
            "Rank granaries by",
            rank_options,
            index=0,
            key="ov_rank_basis",
        )

        if basis.startswith("Average"):
            r = d.groupby("District").Yield_kg_ha.mean().dropna().sort_values(ascending=False)
        else:
            chosen_yr = int(basis.split("(")[-1].rstrip(")")) if "(" in basis else int(basis)
            r = d[d.Year == chosen_yr].set_index("District").Yield_kg_ha.dropna().sort_values(ascending=False)

        if not r.empty:
            n = len(r)
            best, worst = r.index[0], r.index[-1]
            colour = [
                "#2e7d32" if g == best else "#c62828" if g == worst else "#1f6fb2" if g == sel else "#9aa5a0"
                for g in r.index
            ]
            bar = go.Figure(
                go.Bar(
                    x=r.values,
                    y=[f"#{i + 1}  {g}" for i, g in enumerate(r.index)],
                    orientation="h",
                    marker_color=colour,
                    marker_line_color=["#0B2117" if g == sel else "rgba(0,0,0,0)" for g in r.index],
                    marker_line_width=[2.5 if g == sel else 0 for g in r.index],
                    text=[f"{v:,.0f}" for v in r.values],
                    textposition="outside",
                    hovertemplate="%{y}: %{x:,.0f} kg/ha<extra></extra>",
                )
            )
            bar.update_layout(
                height=345,
                margin=dict(l=10, r=48, t=5, b=10),
                xaxis_title="Yield (kg/ha)",
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                plot_bgcolor="#FFFFFF",
            )
            bar.update_xaxes(range=[0, float(r.max()) * 1.15])
            st.plotly_chart(bar, width="stretch", key="ov_yield_rank")

    if not r.empty and sel in r.index:
        i, v = list(r.index).index(sel), float(r[sel])
        up = r.index[i - 1] if i > 0 else None
        dn = r.index[i + 1] if i < n - 1 else None
        msg = [f"**{sel}** ranks **#{i + 1} of {n}** at **{v:,.0f} kg/ha** ({basis})."]
        msg.append(f"Next above: **{up}** ({r[up] - v:+,.0f} kg/ha)." if up else "Currently #1 nationally.")
        msg.append(f"Next below: **{dn}** ({r[dn] - v:+,.0f} kg/ha)." if dn else "Currently lowest nationally.")
        st.info(" ".join(msg))


def render(D):
    """Detailed PACE methodology & data documentation (inside bottom expander)."""
    n_g = len(D.ALL)
    y0, y1 = int(D.yh.Year.min()), int(D.yh.Year.max())
    st.markdown(f"#### About {APP_NAME} — {TAGLINE}")
    st.markdown(
        f"{APP_NAME} brings reported yield, satellite vegetation data, weather and soil information of {n_g} Malaysian granaries "
        "together in one leak-free cross-granary AutoML transfer learning and governance dashboard."
    )
    c = st.columns(4)
    _card(c[0], "🕘", "What happened?", f"Explore reported yield year by year ({y0} to {y1}).", "Historical Yield & Model Accuracy")
    _card(c[1], "🔮", "What happens next?", "See 1–4 year outlooks and climate scenarios.", "Yield Forecast & Simulator")
    _card(c[2], "🛰️", "What is happening?", "Inspect crop condition, heading-stage nowcasts & resilience.", "Granary Field Map")
    _card(c[3], "💡", "What should we do?", "Review prescriptive advisories & log field decisions.", "Prescriptive Action Plan")

    st.markdown("##### Research problem and scope")
    st.write(RESEARCH_PROBLEM)
    st.dataframe(pd.DataFrame(SCOPE_ROWS, columns=["Item", "Scope"]), hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame(PACE_ROWS, columns=["Stage", "Dashboard tab(s)", "What happens"]), hide_index=True, width="stretch")
    for line in RESEARCH:
        st.markdown(line)
    with st.expander("Model track record and known limits"):
        tab_reliability.render(D)
    st.caption(f"{CLOSING} · Decisions saved to: {store.describe(D.DATA_DIR)}.")


def render_overview_hub(D):
    """National Paddy Yield & Crop Health Overview: Macro -> Meso -> Micro."""
    from acts_transfer import governance as gov
    from acts_transfer.freshness_loop import check_data_freshness, load_update_log
    from .tab_map import _map_table

    st.markdown("## National Paddy Yield & Crop Health Overview")
    st.caption("Big-picture yield performance, weather drivers, and AI forecast readiness across Peninsular Malaysia's granary schemes.")

    ds, dy = D.DATA_DIR / "transfer_stage_dataset.csv", D.DATA_DIR / "transfer_yield_dataset.csv"
    if not (ds.exists() or dy.exists()):
        st.warning("The granary-year table is missing, so the overview has no yield figures.")
        return

    df = _map_table(str(ds), str(dy), (ds if ds.exists() else dy).stat().st_mtime)
    years = sorted(int(y) for y in df.Year.unique())
    latest_year = years[-1]
    prev_year = years[-2] if len(years) >= 2 else latest_year
    latest = df[df.Year == latest_year]
    prev = df[df.Year == prev_year]

    reg = gov.load()
    n_app, n_loc, n_pend = gov.counts(D.ALL, reg)
    total = len(D.ALL)
    automl_summary = _load_automl_summary()
    n_pass = sum(
        1 for g in D.ALL if str(automl_summary.get(g, {}).get("gate_status", "")).upper() == "PASS"
    ) or n_app

    # ------------------------------------------------------------------ TIER 1: MACRO (5 Balanced #E7ECE0 KPIs)
    yoy_mean_delta = float(latest.Yield_kg_ha.mean() - prev.Yield_kg_ha.mean()) if not prev.empty else 0.0
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Harvest Year",
        f"{latest_year}",
        f"{years[0]}–{latest_year} panel",
        help="Latest official annual harvest season loaded in the dashboard.",
    )
    k2.metric(
        "National Avg Yield",
        f"{latest.Yield_kg_ha.mean():,.0f} kg/ha",
        f"{yoy_mean_delta:+,.0f} vs {prev_year}" if prev_year != latest_year else "Baseline year",
        help="Simple average yield across all monitored granaries.",
    )
    k3.metric(
        "Granary Schemes",
        f"{total} Active",
        "Peninsular Malaysia",
        help="Major irrigated paddy granaries tracked nationwide.",
    )
    k4.metric(
        "Satellite Clarity",
        f"{latest.Optical_Valid_Fraction.mean() * 100:.1f}%",
        "Cloud-free optical",
        help="Average share of 10-day Sentinel-2 optical windows free of heavy cloud cover.",
    )
    k5.metric(
        "Verified AI Models",
        f"{n_pass} / {total}",
        f"{n_app} signed off",
        help="Granaries passing the Negative-Transfer Safeguard Gate (ΔNT < 0).",
    )

    # Key Takeaway Callout Banner
    best_row = latest.loc[latest["Yield_kg_ha"].idxmax()]
    worst_row = latest.loc[latest["Yield_kg_ha"].idxmin()]
    direction = "rose" if yoy_mean_delta >= 0 else "dipped"
    st.markdown(
        f"""
        <div style="background:#FFFFFF; border-left:5px solid #769D18; border-radius:12px;
                    padding:16px 20px; margin:14px 0 18px 0; box-shadow:0 2px 8px rgba(11,33,23,0.05);">
            <div style="font-size:11.5px; font-weight:700; letter-spacing:0.06em; color:#769D18; text-transform:uppercase;">
                Key Takeaway · {latest_year} Season Snapshot
            </div>
            <div style="font-size:15px; color:#0B2117; margin-top:4px; line-height:1.5;">
                National yield averaged <b>{latest.Yield_kg_ha.mean():,.0f} kg/ha</b> ({direction} <b>{abs(yoy_mean_delta):,.0f} kg/ha</b> vs. {prev_year}).
                <b>{html.escape(str(best_row.District))}</b> led the country at <b>{best_row.Yield_kg_ha:,.0f} kg/ha</b>, while
                <b>{html.escape(str(worst_row.District))}</b> trailed at <b>{worst_row.Yield_kg_ha:,.0f} kg/ha</b>
                — a <b>{best_row.Yield_kg_ha - worst_row.Yield_kg_ha:,.0f} kg/ha</b> yield gap across schemes.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compact 1-line data freshness status strip
    y_df = getattr(D, "yh", pd.DataFrame())
    fresh = check_data_freshness(y_df)
    log = load_update_log()
    pending_reqs = sum(1 for r in log.get("requests", []) if r.get("status") == "PENDING_STEWARD")
    if fresh.get("is_stale"):
        missing_str = ", ".join(map(str, fresh["missing_years"]))
        st.info(
            f"**Data Freshness Notice:** Official harvest records run through **{fresh['max_data_year']}** "
            f"(missing **{missing_str}** vs. system year {fresh['current_year']}) · "
            f"**Pending Data Requests:** {pending_reqs}."
        )

    # ------------------------------------------------------------------ TIER 2: MESO DRIVERS
    _render_national_anomaly_chart(df)

    # ------------------------------------------------------------------ TIER 3: MESO COMPARISON
    _render_yield_charts(df)

    st.markdown("### Granary Scorecard: Yield, Weather Drivers & AI Safety Status")
    st.caption(
        f"Side-by-side comparison of {prev_year} vs. {latest_year} yields, satellite visibility, and each granary's verified forecast model."
    )
    prev_map = prev.set_index("District")["Yield_kg_ha"].to_dict() if not prev.empty else {}
    rows = []
    for _, r in latest.sort_values("District").iterrows():
        g = r["District"]
        y_curr = float(r["Yield_kg_ha"])
        y_prev = prev_map.get(g)
        yoy = (y_curr - float(y_prev)) if y_prev is not None and pd.notna(y_prev) else np.nan
        info = automl_summary.get(g, {})
        gate = info.get("gate_status", "PASS" if gov.effective(g, reg)["status"] == "Approved" else "FALLBACK")
        d_nt = info.get("delta_nt")
        gate_str = f"{gate} ({d_nt:+.1f} kg/ha)" if isinstance(d_nt, (int, float)) else str(gate)
        rows.append(
            {
                "Granary": g,
                f"{prev_year} Yield": round(float(y_prev), 0) if y_prev is not None and pd.notna(y_prev) else None,
                f"{latest_year} Yield": round(y_curr, 0),
                "YoY Change": round(yoy, 0) if pd.notna(yoy) else None,
                "Satellite Clarity (%)": round(float(r["Optical_Valid_Fraction"]) * 100, 1),
                "Key Signal Regime": info.get("best_feature_set", "Meteo Anomalies (Z-Score)"),
                "AI Safety Gate": gate_str,
                "Sign-Off Status": gov.effective(g, reg)["status"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            f"{prev_year} Yield": st.column_config.NumberColumn(format="%d kg/ha"),
            f"{latest_year} Yield": st.column_config.NumberColumn(format="%d kg/ha"),
            "YoY Change": st.column_config.NumberColumn(format="%+d kg/ha"),
            "Satellite Clarity (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        },
    )

    # ------------------------------------------------------------------ TIER 4: MICRO (Equal-height Role Workspace Launch Cards)
    st.markdown("### Take Action in a Role Workspace")
    left, right = st.columns(2, gap="large")
    top = latest.loc[latest.Yield_kg_ha.idxmax()]
    sel_g = str(st.session_state.get("granary", ""))
    with left:
        st.markdown(
            f"""<div class="pace-card">
                <div>
                    <h4 style="color:#0B2117;margin:0 0 8px 0;">🌾 KPKM Programme Manager</h4>
                    <p style="color:#43544B;">Inspect field maps, check 70-day heading crop status, simulate 2025–2028 weather scenarios, and log field advisories.</p>
                </div>
                <ul>
                    <li><b>Focused Granary:</b> {html.escape(sel_g)}</li>
                    <li><b>Top {latest_year} Performer:</b> {html.escape(str(top.District))} ({top.Yield_kg_ha:,.0f} kg/ha)</li>
                </ul>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Open KPKM Programme Manager Workspace →", key="jump_kpkm", width="stretch"):
            st.session_state["nav_mode"] = "KPKM Programme Manager"
            st.session_state["scroll_top"] = True
            st.rerun()
    with right:
        ovf = latest.Optical_Valid_Fraction.mean() * 100
        st.markdown(
            f"""<div class="pace-card">
                <div>
                    <h4 style="color:#0B2117;margin:0 0 8px 0;">🛡️ Model Validator & Data Steward</h4>
                    <p style="color:#43544B;">Audit cloud coverage, compare donor regions, inspect forecast accuracy tests, and sign off safe AI models.</p>
                </div>
                <ul>
                    <li><b>National Satellite Clarity:</b> {ovf:.1f}% · <b>Pending Requests:</b> {pending_reqs}</li>
                    <li><b>Model Registry:</b> {n_app} Approved | {n_loc} Local Baseline ({n_pend} pending)</li>
                </ul>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Open Model Validator & Data Steward Workspace →", key="jump_validator", width="stretch"):
            st.session_state["nav_mode"] = "Model Validator & Data Steward"
            st.session_state["validator_step"] = 1
            st.session_state["scroll_top"] = True
            st.rerun()

    with st.expander("About the project, research methodology, and data sources (PACE)"):
        render(D)