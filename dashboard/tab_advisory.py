"""Component 4 · Advisory and intervention, for KPKM / programme administrators.

The page is a chain with no gaps, so every recommendation can be traced:
    model results -> evidence (E1..E8) -> confidence -> intervention rules (who, when, why) -> AI memo -> the administrator's decision
The chain lives in dashboard/bridge.py. The AI memo is written on the server and may only use the evidence list.
"""
import pandas as pd
import streamlit as st

from . import bridge
from .llm import memo
from .ui import info_panel, pct, selected_granary

CAUTION = ("Draft decision support from satellite data and a statistical forecast. Satellite greenness cannot diagnose a cause, so "
           "every intervention here is something to CHECK, ASK or PREPARE, not a confirmed problem. Rule text is a draft until "
           "an agronomist has reviewed it.")
BRIEF = {"behind": "Recent greenness has been below the range expected for this stage. Verify on the ground before spending.",
         "ahead": "Recent greenness is above the expected range. Confirm the cause (for example earlier planting) before "
                  "reading it as good news.",
         "on track": "Recent greenness is inside the range expected for this stage."}


def all_cases(D):
    return {g: bridge.build_case(D, g) for g in D.ALL}


def priority_table(D, cases):
    """every granary ranked for administrator attention: recent below-range decades first, then low confidence"""
    rows = []
    for g, c in cases.items():
        acts = bridge.interventions(c, D.irules)
        first = acts[0] if acts else ("No rule matches", "-", "-", "", "-", "")
        score = (c["below"] * 10 + c["below_share"]) if c["has_sat"] else -1
        rows.append(dict(score=score, Granary=g, **{"Administrative area": c["admin"] or "-"},
                         **{"Recent status": c.get("status", "no satellite")},
                         **{"Recent decades below range": f"{c['below']} of {c['checked']}" if c["has_sat"] else "-"},
                         Confidence=c["confidence"], **{"Stage in 30 days": c.get("stage", "-")},
                         **{"First intervention": first[0], "Owner": first[1], "When": first[2]},
                         **{"Data issues": ", ".join(f.replace("_", " ") for f in c["flags"] if f != "low_confidence") or "-"}))
    t = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    t.insert(0, "Priority", range(1, len(t) + 1))
    return t.drop(columns="score")


def render(D):
    st.subheader("Advisory and intervention")
    info_panel(
        "Turns the model results into a ranked intervention list for KPKM and the granary authorities. For each granary: the "
        "evidence, how confident we are, who should do what and by when, a plain-language memo written by AI from that evidence "
        "only, and a place to record the decision taken. Interventions come from editable rules (data/intervention_rules.csv).",
        "",
        "See which granaries need attention first, what is happening to the paddy in each, who should act and by when, and keep a "
        "record of what was decided so the alerts can be checked against real events later.",
        "draft advice for review by an agronomist. Satellite greenness cannot diagnose a cause, so actions are things to check, "
        "ask or prepare. The satellite data may be months old: look at the confidence level.")

    cases = all_cases(D)
    g = selected_granary()
    case = cases[g]

    # ---- 1. all granaries, ranked
    st.markdown("#### Intervention priority list: all granaries")
    st.caption("Ranked by the number of recent decades below the forecast range, then by the share of decades below range in "
               f"{D.FINAL_TEST_YEAR}. A prompt to check the field, not a confirmed problem. Pick a granary in the sidebar (or "
               "click it on a map) to see its detail below.")
    st.dataframe(priority_table(D, cases), hide_index=True, width="stretch")

    # ---- 2. the selected granary: what is happening, how sure, what to do
    acts = bridge.interventions(case, D.irules)
    ev = bridge.evidence(D, case)
    with st.container(border=True):
        st.markdown(f"### What is happening to the paddy: {g}")
        if case["admin"]:
            st.caption(f"Administrative area: {case['admin']}")
        if case["has_sat"]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Crop status", case["status"].upper(), f"{case['below']} of {case['checked']} recent decades below range",
                      delta_color="off")
            c2.metric("Stage in 30 days", case["stage"].split(" / ")[0], "estimated, not observed", delta_color="off")
            c3.metric("NDVI trend", case["trend"], f"{case['last_ndvi']:.2f} to {case['f30']:.2f} in 30 days", delta_color="off")
            c4.metric("Confidence", case["confidence"].upper(), help="; ".join(case["conf_reasons"]))
            st.markdown(BRIEF[case["status"]])
        else:
            st.info("No satellite forecast exists for this granary; only yield history is available.")
        st.markdown("**Why this confidence:** " + "; ".join(case["conf_reasons"]) + ".")

        st.markdown("### Recommended interventions")
        if acts:
            st.dataframe(pd.DataFrame([dict(No=i, Intervention=a[0], Owner=a[1], When=a[2], Why=a[3], Rule=a[4], Review=a[5])
                                       for i, a in enumerate(acts, 1)]), hide_index=True, width="stretch")
        else:
            st.markdown("No rule matches this granary's current situation. Add or edit rules in `data/intervention_rules.csv`.")
        with st.expander("Evidence behind this (the AI memo may use only these items)"):
            for i, t in ev:
                st.markdown(f"- **[{i}]** {t}")
        st.markdown("### Caution")
        st.markdown(CAUTION)

    # ---- 3. AI memo: written on the server from the evidence and the approved interventions
    st.markdown("#### AI memo")
    with st.spinner("Writing the memo from the evidence..."):
        text, bad, problem = memo(bridge.memo_task(case, acts), ev)
    if text:
        with st.container(border=True):
            st.markdown(text)
        st.caption("Written by Gemini AI from the evidence list and the approved interventions only. Check before use.")
        if bad:
            st.warning("Numbers in this memo that are not in the evidence list: " + ", ".join(bad) + ". Verify them.")
    elif problem == "not_configured":
        st.caption("The AI memo is not switched on for this server. The interventions above are complete without it; a server "
                   "administrator can enable it by setting GEMINI_API_KEY on the server (see .streamlit/secrets.toml.example).")
    else:
        st.warning(f"The AI memo is unavailable right now ({problem}); the rule-based interventions above still apply.")

    # ---- 4. the human decision goes back into a record
    st.markdown("#### Record the decision")
    with st.form(f"decision_{g}"):
        opts = [f"{a[4]} | {a[0][:110]}" for a in acts] or ["(no intervention proposed)"]
        pick = st.selectbox("Intervention", opts)
        c1, c2 = st.columns([2, 1])
        dec = c1.radio("Decision", ["Accept", "Modify", "Reject", "Defer"], horizontal=True)
        by = c2.text_input("Decided by (name or unit)")
        note = st.text_area("Note: what was decided, changed, or why not", height=80)
        saved = st.form_submit_button("Save decision")
    if saved:
        i = opts.index(pick)
        rule_id, action = (acts[i][4], acts[i][0]) if acts else ("-", "(none)")
        bridge.log_decision(D.DATA_DIR, case, action, rule_id, dec, note, by)
        st.success("Decision saved to data/intervention_log.csv.")
    log = bridge.read_log(D.DATA_DIR)
    mine = log[log.granary == g]
    if len(mine):
        st.dataframe(mine.tail(8).iloc[::-1], hide_index=True, width="stretch")
    st.caption("The log is a file on the server running this app; on Streamlit Community Cloud it is cleared when the app "
               "restarts, so download or back it up if you need it kept.")
    with st.expander("Edit the interventions: data/intervention_rules.csv"):
        st.caption("Columns status / trend / stage_contains / flag / needs_confidence decide when a rule applies ('any' matches "
                   "everything; use | to list several values). Placeholders such as {granary} and {stage} are filled in.")
        st.dataframe(D.irules, hide_index=True, width="stretch")
    if len(log):
        with st.expander("All recorded decisions"):
            st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")
            st.download_button("Download the decision log (CSV)", log.to_csv(index=False).encode("utf-8"),
                               "intervention_log.csv", "text/csv")
