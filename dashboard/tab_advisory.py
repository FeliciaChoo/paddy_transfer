"""Component 4 · Advisory and intervention, for KPKM / programme administrators.

The page is deliberately plain. The administrator picks a granary, presses "Get advisory", and sees what is happening to the paddy
there and what to do about it. A Word report (summary, interventions, a decision and planning table, a space for the final
conclusion, and the sources) can then be downloaded. The workings (evidence, rules, references) live in the bridge and in the report
appendix, not on this page.
"""
import pandas as pd
import streamlit as st

from . import bridge, report, tracker
from .llm import memo
from .ui import selected_granary

BRIEF = {"behind": "Recent greenness is below the range expected for this stage. Verify on the ground before spending.",
         "ahead": "Recent greenness is above the expected range. Confirm the cause (for example earlier planting) before reading it as good news.",
         "on track": "Recent greenness is inside the range expected for this stage."}
LEVEL_ICON = {"Urgent": "🔴", "Soon": "🟠", "Plan ahead": "🟢", "Routine": "⚪"}


def _reference_practice(D):
    """Reference agronomic practice with its resource citation, by growth stage (PACE V10). Editable: data/advisory_reference_practice.csv"""
    import html
    from .about_config import ADVISORY_KNOWLEDGE_SOURCES
    p = D.DATA_DIR / "advisory_reference_practice.csv"
    if not p.exists():
        return
    ref = pd.read_csv(p)
    st.markdown("### Reference practice by growth stage (with resource citations)")
    st.caption("General agronomic guidance, not computed from this granary's data. It was supplied with the V10 specification and has not been checked by this project: confirm the figures with your "
               "local DOA / IADA extension officer before acting on them. Days are HST, days after sowing.")
    for r in ref.itertuples():
        st.markdown(f"""<div class="pace-card"><b>👤 Who:</b> {html.escape(r.Who)}<br><b>📋 Action:</b> {html.escape(r.What)}<br>
                    <b>🗓️ When:</b> {html.escape(r.Window)}<br><br><span style="background:#E7ECE0;padding:5px 10px;border-radius:8px;font-size:12px;"><b>📚 Source Resource:</b> {html.escape(r.Source)}</span></div>""",
                    unsafe_allow_html=True)
    with st.expander("Knowledge sources behind this brief"):
        for k in ADVISORY_KNOWLEDGE_SOURCES:
            st.markdown(f"- {k}")


def field_checklist(case, acts):
    """a field verification checklist tied to the case: what to check on the ground before spending, in the order of the priority actions. Rule-based wording; nothing is invented."""
    items = [("Crop stage", f"Confirm the stage in the field matches the estimate ({case['stage'].split(' / ')[0]}), because stage drives every other reading." if case.get("has_sat") else "Confirm the crop stage in the field.")]
    if case.get("has_sat") and case["status"] == "behind":
        items += [("Where it is low", "Walk the lowest-greenness blocks first and compare with a block that looks normal."),
                  ("Water", "Check water level and drainage in those blocks (flooded or dry fields change satellite greenness)."),
                  ("Causes", "Look for pests, disease, nutrient shortage or late planting before deciding on any input.")]
    elif case.get("has_sat") and case["status"] == "ahead":
        items += [("Planting date", "Confirm whether planting was earlier than the calendar before reading the high greenness as good news.")]
    else:
        items += [("Routine", "Keep the normal monitoring round; nothing unusual is flagged.")]
    for a in acts[:3]:
        items.append((f"Action: {a[0]}", f"Responsible: {a[1]}. By: {a[2]}."))
    items.append(("Record", "Write what you found in the decision log below, so the next review can compare it with this advice."))
    return pd.DataFrame(items, columns=["Check", "What to verify"]).assign(Done="☐")


def render(D):
    from . import tab_governance as G
    st.subheader("Programme Manager Decision Brief")
    g = selected_granary()
    eff = G.gov.effective(g)
    has_pred = bool(eff["entry"])
    G._badge(eff)
    st.caption("What this advice rests on:  ✅ crop condition from satellite  ·  " + (f"✅ validator-reviewed yield forecast ({eff['status']})" if has_pred else
               "🔒 no validator-reviewed yield forecast for this granary, so the advice below uses crop condition and the local baseline only"))
    st.markdown(f"Selected granary: **{g}**. Change it in the sidebar or by clicking a granary on a map, then press the button.")
    if st.button(f"Get advisory for {g}", type="primary", key="get_advisory"):
        st.session_state["advisory_for"] = g
    if st.session_state.get("advisory_for") != g:
        st.caption("Nothing is shown until you press the button.")
        return

    case = bridge.build_case(D, g)
    rows, label, is_t = G._served(eff, "Normal") if eff["published"] else (None, None, False)
    if rows:
        r0 = rows[0]
        case["pred_line"] = f"Yield outlook {r0['season']}: {r0['predicted']:,.0f} kg/ha ({r0['low']:,.0f} to {r0['high']:,.0f}) from {label} [{eff['badge']}]"
    acts = bridge.interventions(case, D.irules)
    t_und, t_plan = st.tabs(["Understand the results", "Plan next steps"])
    if case["has_sat"] and case["stale_days"] > 45:
        st.warning(f"The satellite data ends {case['as_of']}, {case['stale_days']} days ago, so this advice describes that date and not "
                   "today. Refresh the data before acting on it.")

    with t_und:
        # ---- what is happening
        with st.container(border=True):
            st.markdown(f"### Summary: {g}")
            if case["admin"]:
                st.caption(f"Administrative area: {case['admin']}")
            if case["has_sat"]:
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Crop status", case["status"].upper(), f"{case['below']} of {case['checked']} recent periods below range", delta_color="off")
                c2.metric("Stage in 30 days", case["stage"].split(" / ")[0], "estimated", delta_color="off")
                c3.metric("NDVI trend", case["trend"], f"{case['last_ndvi']:.2f} to {case['f30']:.2f} in 30 days", delta_color="off")
                c4.metric("Confidence", case["confidence"].upper(), help="; ".join(case["conf_reasons"]))
                if "res_class" in case:
                    c5.metric("Resilience", case["res_class"].upper(), f"index {case['res_index']:.0f} of 100", delta_color="off",
                              help="How often and how deeply greenness falls below the expected range and how fast it recovers, relative to the "
                                   "other granaries. Not shown to predict yield.")
            for s in bridge.summary_text(D, case):
                st.markdown(s)


        if rows:
            st.markdown("### Yield forecast behind this brief")
            st.markdown(case["pred_line"])
            st.caption("The Normal-scenario forecast. Other scenarios and the in-season nowcast are in tab 4.")
    with t_plan:
        # ---- what to do
        st.markdown("### Priority actions: Who · What · When · Why")
        if acts:
            st.dataframe(pd.DataFrame([{"Priority": f"{LEVEL_ICON[bridge.priority_label(a[7])]} {bridge.priority_label(a[7])}",
                                        "What": a[0], "Who": a[1], "When": a[2], "Why": a[3]} for a in acts]),
                         hide_index=True, width="stretch")
            plan = pd.DataFrame([{"Priority": bridge.priority_label(a[7]), "Who": a[1], "What": a[0], "When": a[2], "Why": a[3]} for a in acts])
            st.download_button("Download the actions (CSV)", plan.to_csv(index=False).encode("utf-8"), file_name=f"priority_actions_{g.replace(' ', '_')}.csv", mime="text/csv", key="dl_actions")
        else:
            st.info("No intervention is suggested for this granary's current situation.")
        _reference_practice(D)
        st.markdown("### Field verification checklist")
        st.caption("Verify on the ground before spending. Tick off in the decision log below.")
        st.dataframe(field_checklist(case, acts), hide_index=True, width="stretch")

        # ---- AI written summary: only when the server has a key; silent otherwise
        text, bad, _ = memo(bridge.memo_task(case, acts), bridge.evidence(D, case))
        if text:
            st.markdown("### Advisory note")
            with st.container(border=True):
                st.markdown(text)
            if bad:
                st.caption("Check these figures, they are not in the data behind this advice: " + ", ".join(bad))

        # ---- KPKM's decision and next step, kept so follow-up can be tracked
        cur = tracker.current(tracker.load(D.DATA_DIR))
        st.markdown("### KPKM decision and next step")
        st.caption("After discussing the advice, record what KPKM decides for each intervention: Accept (do it as proposed), Modify (do something "
                   "different), Reject (do not do it) or Defer (not now). Decisions are kept, so follow-up can be tracked in the Decision tracker "
                   "tab and the choices can be compared with the advice.")
        ver = st.session_state.get("dec_ver", 0)
        if acts:
            frame = tracker.editor_frame(case, acts, cur)
            edited = st.data_editor(
                frame, key=f"dec_{g}_{case.get('as_of', '')}_{ver}", hide_index=True, width="stretch",
                disabled=["Rule", "Priority", "Proposed intervention"],
                column_config={
                    "Rule": None,
                    "Priority": st.column_config.TextColumn(width="small"),
                    "Proposed intervention": st.column_config.TextColumn(width="large"),
                    "Decision": st.column_config.SelectboxColumn(options=tracker.DECISIONS, required=True, width="small"),
                    "What KPKM will do (next step)": st.column_config.TextColumn(width="large",
                                                                               help="Required when the decision is Modify"),
                    "Responsible officer": st.column_config.TextColumn(width="medium"),
                    "Target date": st.column_config.DateColumn(format="DD MMM YYYY", width="small"),
                    "Status": st.column_config.SelectboxColumn(options=tracker.STATUSES, required=True, width="small"),
                    "Reason / remarks": st.column_config.TextColumn(width="medium", help="Required when the decision is Reject")})
        else:
            edited = None
        c1, c2 = st.columns([1, 2])
        by = c1.text_input("Decided by (name or unit)", key=f"by_{ver}")
        overall = c2.text_area("Overall next step or conclusion for this granary (optional)", value=tracker.overall_text(case, cur),
                               key=f"overall_{g}_{case.get('as_of', '')}_{ver}", height=90)
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
            st.success("Saved. You can follow up on it in the Decision tracker tab, and it is included in the report below.")
        cur = tracker.current(tracker.load(D.DATA_DIR))                   # re-read so the report below shows what was just saved
        recorded = tracker.recorded_for(cur, g, case.get("as_of", ""))

        # ---- the report
        st.markdown("### Report")
        st.caption("A Word report you can edit: summary, interventions, the decisions recorded above (or a blank table to fill in), the "
                   "final conclusion, and the sources.")
        c1, c2 = st.columns(2)
        stamp = pd.Timestamp.today().strftime("%Y%m%d")
        safe = g.replace(" ", "_")
        c1.download_button(f"Download report: {g}", report.build_single(D, case, acts, text, recorded),
                           file_name=f"advisory_{safe}_{stamp}.docx", key="dl_single",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        cases = {k: bridge.build_case(D, k) for k in D.ALL}
        c2.download_button("Download report: all granaries", report.build_all(D, cases, cur),
                           file_name=f"advisory_all_granaries_{stamp}.docx", key="dl_all",
                           mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        with st.expander("Decision tracker: follow up earlier decisions"):
            from . import tab_tracker
            tab_tracker.render(D)


def render_decision_brief(D=None):
    """PACE V9 page: the Programme Manager Decision Brief ('Understand the results' | 'Plan next steps')"""
    from .data import load
    render(D or load())
