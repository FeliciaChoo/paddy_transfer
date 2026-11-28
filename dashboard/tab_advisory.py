"""Component 4 · Advisory and intervention, for KPKM / programme administrators.

The administrator selects a granary and clicks 'Get insight' to view a clean, 3-tier executive brief
combining current crop status, yield outlook, official irrigation schedules, and stage-matched DOA/IADA
references attached ONLY when relevant.
"""
import html
import json
from pathlib import Path
import pandas as pd
import streamlit as st

from . import bridge, report, tracker
from .llm import generate_manager_insight, narrate_counterfactual_cards
from .ui import selected_granary

LEVEL_ICON = {"Urgent": "🔴", "Soon": "🟠", "Plan ahead": "🟢", "Routine": "⚪"}

STATUS_THEME = {
    "Action Needed": {"bg": "#FEE2E2", "fg": "#991B1B", "border": "#DC2626"},
    "Review Recommended": {"bg": "#FEF3C7", "fg": "#92400E", "border": "#D97706"},
    "On Track": {"bg": "#DCFCE7", "fg": "#166534", "border": "#16A34A"},
}


def _clean_stage(case: dict) -> str:
    """Return a clean stage name, or empty string if stage is unknown/missing."""
    raw = str(case.get("stage", "")).split(" / ")[0].strip()
    if not raw or "unknown" in raw.lower() or raw.lower() in ("none", "nan"):
        return ""
    return raw


def _load_granary_schedule(D, granary: str) -> tuple[str, pd.DataFrame]:
    """Pull official staggered block/phase irrigation windows for the selected granary."""
    for base in [D.DATA_DIR, D.DATA_DIR.parent / "external"]:
        p = base / "granary_irrigation_schedules.csv"
        if p.exists():
            df = pd.read_csv(p)
            col = next((c for c in df.columns if "granary" in c.lower()), None)
            if col:
                sub = df[df[col].astype(str).str.lower().str.contains(granary.lower().split()[-1], na=False)]
                if not sub.empty:
                    summary_parts = []
                    for row in sub.head(4).to_dict(orient="records"):
                        vals = [f"{k}: {v}" for k, v in row.items() if pd.notna(v) and k != col]
                        summary_parts.append(" | ".join(vals[:4]))
                    return "; ".join(summary_parts), sub
    return "", pd.DataFrame()


def _match_relevant_references(D, case: dict, acts: list) -> list[dict]:
    """Attach ONLY official reference rows that directly match the active growth stage or suggested actions."""
    p = D.DATA_DIR / "advisory_reference_practice.csv"
    if not p.exists():
        return []
    ref = pd.read_csv(p)

    context_blob = " ".join([
        _clean_stage(case),
        str(case.get("status", "")),
        " ".join(str(a[0]) + " " + str(a[3]) for a in acts),
    ]).lower()

    topic_keywords = {
        "water": ["water", "irrigation", "drain", "moisture", "flood", "drought", "tillering", "panicle"],
        "fertil": ["fertil", "baja", "urea", "nutrient", "npk", "leaf", "lcc", "tillering"],
        "pest": ["pest", "planthopper", "pianggang", "blast", "disease", "scout", "heading", "borer"],
    }

    matched = []
    for r in ref.to_dict(orient="records"):
        row_text = f"{r.get('What', '')} {r.get('Window', '')}".lower()
        for kws in topic_keywords.values():
            if any(k in context_blob for k in kws) and any(k in row_text for k in kws):
                matched.append(r)
                break
    return matched


def _fallback_insight(g: str, case: dict, acts: list, schedule_info: str) -> dict:
    """Deterministic plain-English executive brief when Gemini API key is unset or offline."""
    stage = _clean_stage(case)
    stage_phrase = f"the {stage.lower()} stage" if stage else "its current seasonal window"
    status = case.get("status", "on track")

    if status == "behind":
        label = "Action Needed"
        headline = f"Verify slower crop growth and water supply in {g}"
        situation = (
            f"Crop growth in {g} is tracking behind normal expectations for {stage_phrase}. "
            "Field officers should check canal water levels and block planting dates before authorizing input changes."
        )
    elif status == "ahead":
        label = "Review Recommended"
        headline = f"Confirm early planting dates and block progress in {g}"
        situation = (
            f"Canopy growth in {g} appears ahead of {stage_phrase}. "
            "Confirm whether blocks were sown earlier than scheduled before treating this as a yield gain."
        )
    else:
        label = "On Track" if case.get("has_sat") else "Review Recommended"
        headline = f"Confirm block planting dates and routine water delivery in {g}"
        sched_note = " Official staggered block schedules apply across irrigation phases." if schedule_info else ""
        situation = (
            f"{g} is currently in {stage_phrase} based on regional seasonal timing.{sched_note} "
            "Verify actual sowing dates across blocks during this week's round."
        )

    # Filter out CLI / technical data-refresh rules so Programme Managers see field actions first
    field_acts = [
        a for a in acts
        if "--end-year" not in str(a[0]) and "satellite extraction" not in str(a[0]).lower()
    ]
    if field_acts:
        top = field_acts[0]
        action_step = f"{top[0].rstrip('.')}."
        lead_officer = top[1]
        when = top[2]
        purpose = top[3]
    else:
        action_step = "Confirm actual block sowing dates and check inlet/outlet water delivery across staggered blocks."
        lead_officer = "Block Coordinator & Extension Agronomist"
        when = "During this week's field round"
        purpose = "Verify local crop stage and irrigation delivery before approving input changes"

    return {
        "status_label": label,
        "headline": headline,
        "situation": situation,
        "action_step": action_step,
        "lead_officer": lead_officer,
        "when": when,
        "purpose": purpose,
    }


def _render_executive_brief(
    g: str,
    case: dict,
    eff: dict,
    rows: list | None,
    insight: dict,
    schedule_df: pd.DataFrame,
    matched_refs: list[dict],
    bad_nums: list[str],
):
    """Render the 3-tier user-friendly Programme Manager Executive Brief."""
    # ---- TIER 1: At-a-Glance Status Strip ----
    c1, c2, c3 = st.columns(3)
    stage_display = _clean_stage(case) or "Seasonal Schedule"
    cond_map = {"behind": "Behind Normal", "ahead": "Ahead of Schedule", "on track": "On Track"}
    cond_text = cond_map.get(case.get("status", ""), "Pending Check") if case.get("has_sat") else "Field Check Needed"
    c1.metric("Crop Condition", cond_text, f"Window: {stage_display}", delta_color="off")

    if rows:
        r0 = rows[0]
        trust_note = "Verified transfer model" if eff.get("status") == "approved" else "Local baseline outlook"
        c2.metric("Expected Harvest", f"{r0['predicted']:,.0f} kg/ha", trust_note, delta_color="off")
    else:
        c2.metric("Expected Harvest", "Baseline Only", "Awaiting model sign-off", delta_color="off")

    res_label = case.get("res_class", "Standard").title() if "res_class" in case else "Standard"
    res_sub = f"Recovery score: {case['res_index']:.0f}/100" if "res_index" in case else "Follow local block schedule"
    c3.metric("Drought / Flood Resilience", res_label, res_sub, delta_color="off")

    # ---- TIER 2: Primary Manager Action Card (Two Separate Side-by-Side Grey Boxes) ----
    status_label = insight.get("status_label", "Review Recommended")
    theme = STATUS_THEME.get(status_label, STATUS_THEME["Review Recommended"])

    badge = html.escape(status_label)
    headline = html.escape(insight.get("headline", f"Review field status in {g}"))
    situation = html.escape(insight.get("situation", ""))
    action_step = html.escape(insight.get("action_step", ""))
    lead_officer = html.escape(insight.get("lead_officer", "Extension Agronomist"))
    when = html.escape(insight.get("when", "This week's field round"))
    purpose = html.escape(insight.get("purpose", "Verify crop stage and water delivery"))

    source_tags = []
    if not schedule_df.empty:
        source_tags.append("MYSA MakGeoPadi / KPKM Irrigation Schedule")
    for r in matched_refs:
        if r.get("Source") and r["Source"] not in source_tags:
            source_tags.append(r["Source"])

    source_footer = ""
    if source_tags:
        joined_sources = html.escape("  ·  ".join(source_tags))
        source_footer = (
            f'<div style="margin-top:16px; padding-top:12px; border-top:1px dashed #CBD5E1; '
            f'font-size:12.5px; color:#475569;">'
            f'<strong>Validated sources used:</strong> {joined_sources}</div>'
        )

    card_html = (
        f'<div style="background:#FFFFFF; border:1px solid #E2E8F0; border-left:5px solid {theme["border"]}; '
        f'border-radius:12px; padding:22px 24px; margin:12px 0 16px 0; box-shadow:0 1px 3px rgba(15,23,42,0.04);">'
        f'<span style="background:{theme["bg"]}; color:{theme["fg"]}; font-weight:700; font-size:12px; '
        f'padding:4px 12px; border-radius:999px; letter-spacing:0.2px;">{badge}</span>'
        f'<h4 style="color:#0F172A; margin:12px 0 8px 0; font-size:18px; font-weight:700;">{headline}</h4>'
        f'<p style="color:#334155; font-size:15px; line-height:1.6; margin-bottom:18px;">{situation}</p>'
        f'<div style="display:grid; grid-template-columns: 1fr 1fr; gap:14px;">'
        f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px 16px;">'
        f'<div style="color:#64748B; font-size:11px; text-transform:uppercase; font-weight:700; letter-spacing:0.4px;">'
        f'Recommended Action &amp; Lead Role</div>'
        f'<div style="color:#0F172A; font-size:14.5px; font-weight:600; margin-top:6px;">{action_step}</div>'
        f'<div style="color:#475569; font-size:13px; margin-top:6px;">Lead: <strong>{lead_officer}</strong></div>'
        f'</div>'
        f'<div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px 16px;">'
        f'<div style="color:#64748B; font-size:11px; text-transform:uppercase; font-weight:700; letter-spacing:0.4px;">'
        f'When &amp; Operational Goal</div>'
        f'<div style="color:#0F172A; font-size:14.5px; font-weight:700; margin-top:6px;">{when}</div>'
        f'<div style="color:#475569; font-size:13px; margin-top:6px;">{purpose}</div>'
        f'</div>'
        f'</div>'
        f'{source_footer}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    if bad_nums:
        st.caption("Verify ungrounded figures against source data: " + ", ".join(bad_nums))

    # ---- TIER 3: Progressive Disclosure Drawer ----
    if not schedule_df.empty or matched_refs:
        with st.expander("View matched official irrigation schedule & crop-stage guide"):
            if not schedule_df.empty:
                st.markdown(f"**Official Water-Release & Planting Windows ({g})**")
                st.dataframe(schedule_df, hide_index=True, use_container_width=True)
            if matched_refs:
                st.markdown("**Relevant Agronomic Practice for Current Stage / Action**")
                for r in matched_refs:
                    st.markdown(
                        f'<div class="pace-card"><b>👤 Responsible:</b> {html.escape(r["Who"])}<br>'
                        f'<b>📋 Guidance:</b> {html.escape(r["What"])}<br>'
                        f'<b>🗓️ Stage Window:</b> {html.escape(r["Window"])}<br><br>'
                        f'<span style="background:#E7ECE0;padding:4px 10px;border-radius:8px;font-size:12px;">'
                        f'<b>📚 Source:</b> {html.escape(r["Source"])}</span></div>',
                        unsafe_allow_html=True,
                    )


def field_checklist(case, acts):
    stage = _clean_stage(case)
    items = [(
        "Crop stage",
        f"Confirm the stage in the field matches the estimate ({stage}), because stage drives every other reading."
        if (case.get("has_sat") and stage)
        else "Confirm the crop stage in the field.",
    )]
    if case.get("has_sat") and case["status"] == "behind":
        items += [
            ("Where it is low", "Walk the lowest-greenness blocks first and compare with a block that looks normal."),
            ("Water", "Check water level and drainage in those blocks (flooded or dry fields change satellite greenness)."),
            ("Causes", "Look for pests, disease, nutrient shortage or late planting before deciding on any input."),
        ]
    elif case.get("has_sat") and case["status"] == "ahead":
        items += [
            ("Planting date", "Confirm whether planting was earlier than the calendar before reading the high greenness as good news.")
        ]
    else:
        items += [("Routine", "Keep the normal monitoring round; nothing unusual is flagged.")]
    for a in acts[:3]:
        items.append((f"Action: {a[0]}", f"Responsible: {a[1]}. By: {a[2]}."))
    items.append(("Record", "Write what you found in the decision log below, so the next review can compare it with this advice."))
    return pd.DataFrame(items, columns=["Check", "What to verify"]).assign(Done="☐")


def _render_counterfactual_cards(granary: str, registry_path: str = "results/governance_registry.json"):
    p = Path(registry_path)
    if not p.exists():
        return
    try:
        reg = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    entry = reg.get(granary)
    if not entry or not entry.get("prescriptions"):
        return

    presc = entry["prescriptions"]
    st.markdown("### Causal counterfactual prescriptions (Field Officer action cards)")
    st.caption(
        f"Approved transfer model: **{entry.get('approved_method', 'HGB + RULE')}** "
        f"(ΔNT = `{entry.get('delta_nt')}`, updated `{entry.get('updated_at', 'N/A')}`). "
        "Levers are bounded by published JPS/IADA irrigation schedules."
    )
    df_cf = pd.DataFrame([
        {
            "Lever": item.get("lever"),
            "Target Window": item.get("window"),
            "Expected Gain (kg/ha)": f"+{item.get('delta_y_kg_ha')}",
            "80% CI (kg/ha)": f"[{item.get('ci_low')}, {item.get('ci_high')}]",
            "SOP Rule": item.get("rule_id"),
            "Schedule Bound": item.get("schedule_bound"),
        }
        for item in presc
    ])
    st.dataframe(df_cf, hide_index=True, width="stretch")

    card_text, bad_nums, _ = narrate_counterfactual_cards(granary, presc)
    if card_text:
        with st.container(border=True):
            st.markdown(card_text)
        if bad_nums:
            st.caption("Verify these ungrounded numbers against the table above: " + ", ".join(bad_nums))


def render(D):
    from . import tab_governance as G

    st.subheader("Programme Manager Decision Brief")
    g = selected_granary()
    eff = G.gov.effective(g)
    has_pred = bool(eff["entry"])
    G._badge(eff)
    st.caption(
        "What this advice rests on:  ✅ crop condition from satellite  ·  "
        + (
            f"✅ validator-reviewed yield forecast ({eff['status']})"
            if has_pred
            else "🔒 no validator-reviewed yield forecast for this granary, so the advice below uses crop condition and the local baseline only"
        )
    )
    st.markdown(f"Selected granary: **{g}**. Change it in the sidebar or by clicking a granary on a map, then press the button.")

    # Button gate: nothing renders until pressed
    if st.button(f"Get insight for {g}", type="primary", key="get_advisory"):
        st.session_state["advisory_for"] = g
    if st.session_state.get("advisory_for") != g:
        st.caption("Nothing is shown until you press the button.")
        return

    case = bridge.build_case(D, g)
    rows, label, _ = G._served(eff, "Normal") if eff["published"] else (None, None, False)
    if rows:
        r0 = rows[0]
        case["pred_line"] = f"Yield outlook {r0['season']}: {r0['predicted']:,.0f} kg/ha ({r0['low']:,.0f} to {r0['high']:,.0f}) from {label} [{eff['badge']}]"
    acts = bridge.interventions(case, D.irules)

    if case["has_sat"] and case["stale_days"] > 45:
        st.warning(
            f"The satellite data ends {case['as_of']}, {case['stale_days']} days ago, so this advice describes that date and not "
            "today. Refresh the data before acting on it."
        )

    # Pull granary irrigation schedule + stage-matched official practices
    schedule_info, schedule_df = _load_granary_schedule(D, g)
    matched_refs = _match_relevant_references(D, case, acts)

    # Generate plain-English executive brief via Gemini (with deterministic fallback)
    insight, bad_nums, _ = generate_manager_insight(g, case, acts, schedule_info, matched_refs)
    if not insight:
        insight = _fallback_insight(g, case, acts, schedule_info)

    st.markdown("### Executive Insight & Next Steps")
    _render_executive_brief(g, case, eff, rows, insight, schedule_df, matched_refs, bad_nums)

    # Secondary tabs for full action table, KPKM decision tracker, and Word report
    t_plan, t_details = st.tabs(["Action plan & decision log", "Detailed metrics & report"])

    with t_plan:
        _render_counterfactual_cards(g)

        st.markdown("### Priority actions: Who · What · When · Why")
        if acts:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Priority": f"{LEVEL_ICON[bridge.priority_label(a[7])]} {bridge.priority_label(a[7])}",
                        "What": a[0],
                        "Who": a[1],
                        "When": a[2],
                        "Why": a[3],
                    }
                    for a in acts
                ]),
                hide_index=True,
                width="stretch",
            )
            plan = pd.DataFrame([
                {"Priority": bridge.priority_label(a[7]), "Who": a[1], "What": a[0], "When": a[2], "Why": a[3]}
                for a in acts
            ])
            st.download_button(
                "Download the actions (CSV)",
                plan.to_csv(index=False).encode("utf-8"),
                file_name=f"priority_actions_{g.replace(' ', '_')}.csv",
                mime="text/csv",
                key="dl_actions",
            )
        else:
            st.info("No intervention is suggested for this granary's current situation.")

        st.markdown("### Field verification checklist")
        st.caption("Verify on the ground before spending. Tick off in the decision log below.")
        st.dataframe(field_checklist(case, acts), hide_index=True, width="stretch")

        cur = tracker.current(tracker.load(D.DATA_DIR))
        st.markdown("### KPKM decision and next step")
        st.caption(
            "After discussing the advice, record what KPKM decides for each intervention: Accept (do it as proposed), Modify (do something "
            "different), Reject (do not do it) or Defer (not now)."
        )
        ver = st.session_state.get("dec_ver", 0)
        if acts:
            frame = tracker.editor_frame(case, acts, cur)
            edited = st.data_editor(
                frame,
                key=f"dec_{g}_{case.get('as_of', '')}_{ver}",
                hide_index=True,
                width="stretch",
                disabled=["Rule", "Priority", "Proposed intervention"],
                column_config={
                    "Rule": None,
                    "Priority": st.column_config.TextColumn(width="small"),
                    "Proposed intervention": st.column_config.TextColumn(width="large"),
                    "Decision": st.column_config.SelectboxColumn(options=tracker.DECISIONS, required=True, width="small"),
                    "What KPKM will do (next step)": st.column_config.TextColumn(
                        width="large", help="Required when the decision is Modify"
                    ),
                    "Responsible officer": st.column_config.TextColumn(width="medium"),
                    "Target date": st.column_config.DateColumn(format="DD MMM YYYY", width="small"),
                    "Status": st.column_config.SelectboxColumn(options=tracker.STATUSES, required=True, width="small"),
                    "Reason / remarks": st.column_config.TextColumn(width="medium", help="Required when the decision is Reject"),
                },
            )
        else:
            edited = None

        c1, c2 = st.columns([1, 2])
        by = c1.text_input("Decided by (name or unit)", key=f"by_{ver}")
        overall = c2.text_area(
            "Overall next step or conclusion for this granary (optional)",
            value=tracker.overall_text(case, cur),
            key=f"overall_{g}_{case.get('as_of', '')}_{ver}",
            height=90,
        )
        if st.button("Save decisions", key="save_decisions"):
            problems = tracker.validate(edited) if edited is not None else []
            if problems:
                st.error("Not saved yet:\n\n" + "\n".join(f"- {p}" for p in problems))
            else:
                n = tracker.save(D.DATA_DIR, case, acts, edited if edited is not None else pd.DataFrame(), by, overall)
                st.session_state["dec_ver"] = ver + 1
                st.session_state["dec_saved"] = n
                st.rerun()
        if st.session_state.pop("dec_saved", None) is not None:
            st.success("Saved. You can follow up on it in the Decision tracker tab, and it is included in the report.")

        with st.expander("Decision tracker: follow up earlier decisions"):
            from . import tab_tracker
            tab_tracker.render(D)

    with t_details:
        stage_detail = _clean_stage(case) or "Seasonal Schedule"
        with st.container(border=True):
            st.markdown(f"### Detailed metrics: {g}")
            if case["admin"]:
                st.caption(f"Administrative area: {case['admin']}")
            if case["has_sat"]:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Crop status", case["status"].upper(), f"{case['below']} of {case['checked']} recent periods below range", delta_color="off")
                c2.metric("Stage in 30 days", stage_detail, "estimated", delta_color="off")
                c3.metric("NDVI trend", case["trend"], f"{case['last_ndvi']:.2f} to {case['f30']:.2f} in 30 days", delta_color="off")
                c4.metric("Confidence", case["confidence"].upper(), help="; ".join(case["conf_reasons"]))
                if "res_class" in case:
                    c5.metric("Resilience", case["res_class"].upper(), f"index {case['res_index']:.0f} of 100", delta_color="off")
            for s in bridge.summary_text(D, case):
                st.markdown(s)

        cur = tracker.current(tracker.load(D.DATA_DIR))
        recorded = tracker.recorded_for(cur, g, case.get("as_of", ""))
        memo_text = f"{insight['headline']}\n\n{insight['situation']}\n\nRecommended action: {insight['action_step']}"

        st.markdown("### Export Word Report")
        c1, c2 = st.columns(2)
        stamp = pd.Timestamp.today().strftime("%Y%m%d")
        safe = g.replace(" ", "_")
        c1.download_button(
            f"Download report: {g}",
            report.build_single(D, case, acts, memo_text, recorded),
            file_name=f"advisory_{safe}_{stamp}.docx",
            key="dl_single",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        cases = {k: bridge.build_case(D, k) for k in D.ALL}
        c2.download_button(
            "Download report: all granaries",
            report.build_all(D, cases, cur),
            file_name=f"advisory_all_granaries_{stamp}.docx",
            key="dl_all",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )


def render_decision_brief(D=None):
    """PACE V9 page: the Programme Manager Decision Brief."""
    from .data import load
    render(D or load())