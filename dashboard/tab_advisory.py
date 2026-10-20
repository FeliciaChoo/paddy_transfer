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


def render(D):
    st.subheader("Advisory and intervention")
    g = selected_granary()
    st.markdown(f"Selected granary: **{g}**. Change it in the sidebar or by clicking a granary on a map, then press the button.")
    if st.button(f"Get advisory for {g}", type="primary", key="get_advisory"):
        st.session_state["advisory_for"] = g
    if st.session_state.get("advisory_for") != g:
        st.caption("Nothing is shown until you press the button.")
        return

    case = bridge.build_case(D, g)
    acts = bridge.interventions(case, D.irules)
    if case["has_sat"] and case["stale_days"] > 45:
        st.warning(f"The satellite data ends {case['as_of']}, {case['stale_days']} days ago, so this advice describes that date and not "
                   "today. Refresh the data before acting on it.")

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

    # ---- what to do
    st.markdown("### Recommended interventions")
    if acts:
        st.dataframe(pd.DataFrame([{"Priority": f"{LEVEL_ICON[bridge.priority_label(a[7])]} {bridge.priority_label(a[7])}",
                                    "Intervention": a[0], "Responsible": a[1], "By when": a[2], "Why": a[3]} for a in acts]),
                     hide_index=True, width="stretch")
    else:
        st.info("No intervention is suggested for this granary's current situation.")

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
