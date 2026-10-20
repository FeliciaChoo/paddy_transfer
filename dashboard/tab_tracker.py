"""Component 5 · Decision tracker: what KPKM decided about the proposed interventions, and the follow-up.

Shows every recorded decision (the advice next to what KPKM decided), lets the follow-up fields be updated as work progresses,
compares KPKM's choices with the advice, and keeps the full history.
"""
import pandas as pd
import streamlit as st

from . import store, tracker
from .ui import info_panel

SHOW = {"granary": "Granary", "priority": "Priority", "proposed": "Proposed intervention", "decision": "Decision",
        "kpkm_action": "What KPKM will do (next step)", "responsible": "Responsible officer", "target_date": "Target date",
        "status": "Status", "reason": "Reason / remarks", "decided_by": "Decided by", "data_as_of": "Advice based on data of",
        "saved_at": "Last updated"}


def render(D):
    st.subheader("Decision tracker")
    info_panel(
        "Every decision KPKM records on the Advisory tab is kept here: the proposed intervention, what KPKM decided (accept, modify, reject or "
        "defer), the next step, who is responsible, the target date and the status. Updates are added as new versions, so the history is kept.",
        "",
        "Follow up what was decided, see which items are open or overdue, and see how often KPKM changes or rejects the advice. Where a "
        "rule is often changed or rejected, review that rule.",
        "decisions are saved in the database named below. A local SQLite file lasts as long as that file does; on Streamlit Community Cloud "
        "it is cleared when the app restarts, so use an external database there (see the README) or download the history.")
    hist = tracker.load(D.DATA_DIR)
    cur = tracker.current(hist)
    items = cur[cur.rule_id != tracker.OVERALL] if not cur.empty else cur
    if items.empty:
        st.info("No decisions recorded yet. Open the Advisory tab, press Get advisory for a granary, and record KPKM's decisions there.")
        return
    s = tracker.summary(cur)
    k = st.columns(6)
    k[0].metric("Decisions recorded", s["recorded"])
    k[1].metric("Accepted", s["Accept"])
    k[2].metric("Modified", s["Modify"])
    k[3].metric("Rejected", s["Reject"])
    k[4].metric("Open follow-ups", s["open"], help="Accepted, modified or deferred items that are not yet Done.")
    k[5].metric("Overdue", s["overdue"], help="Target date passed and the item is not Done or Dropped.")
    n = max(s["recorded"], 1)
    st.caption(f"KPKM accepted {s['Accept'] / n:.0%} of the proposed interventions as proposed, changed {s['Modify'] / n:.0%}, rejected "
               f"{s['Reject'] / n:.0%} and deferred {s['Defer'] / n:.0%}.")

    # ---- filters
    c1, c2, c3 = st.columns(3)
    gs = c1.multiselect("Granary", sorted(items.granary.unique()), key="trk_g")
    ds = c2.multiselect("Decision", tracker.DECISIONS[1:], key="trk_d")
    ss = c3.multiselect("Status", tracker.STATUSES, key="trk_s")
    keep = pd.Series(True, index=items.index)
    for col, chosen in (("granary", gs), ("decision", ds), ("status", ss)):
        if chosen:
            keep &= items[col].isin(chosen)
    view = items[keep & (items.decision != "Pending")].sort_values(["granary", "priority"]).reset_index(drop=True)

    st.markdown("#### Advice and what KPKM decided")
    st.caption("The proposed intervention is on the left; KPKM's decision and next step are to its right. Update the status, responsible "
               "officer, date or remarks as work progresses, then press Save changes.")
    show = view[list(SHOW)].rename(columns=SHOW)
    show["Target date"] = [tracker._as_date(x) for x in view.target_date]
    edited = st.data_editor(
        show, key=f"trk_edit_{st.session_state.get('trk_ver', 0)}", hide_index=True, width="stretch",
        disabled=["Granary", "Priority", "Proposed intervention", "Decision", "Decided by", "Advice based on data of", "Last updated"],
        column_config={
            "Status": st.column_config.SelectboxColumn(options=tracker.STATUSES, required=True),
            "Target date": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Proposed intervention": st.column_config.TextColumn(width="large"),
            "What KPKM will do (next step)": st.column_config.TextColumn(width="large")})
    by = st.text_input("Updated by (name or unit)", key="trk_by")
    if st.button("Save changes", key="trk_save"):
        b = edited.rename(columns={v: k for k, v in SHOW.items()})
        n_saved = tracker.save_edits(D.DATA_DIR, view[tracker.COLS], b, by)
        st.session_state["trk_ver"] = st.session_state.get("trk_ver", 0) + 1
        st.session_state["trk_saved"] = n_saved
        st.rerun()
    if st.session_state.pop("trk_saved", None) is not None:
        st.success("Saved.")

    # ---- how KPKM's choices differ from the advice
    cmp_ = tracker.by_rule(cur, D.irules)
    if not cmp_.empty:
        st.markdown("#### How KPKM's decisions differ from the advice")
        st.caption("One row per intervention type (rule). A rule that is often changed or rejected is a candidate for review.")
        st.dataframe(cmp_, hide_index=True, width="stretch")
    od = tracker.overdue(cur)
    if len(od):
        st.markdown("#### Overdue")
        st.dataframe(od[["granary", "priority", "proposed", "decision", "responsible", "target_date", "status"]].rename(columns=SHOW),
                     hide_index=True, width="stretch")
    ov = cur[cur.rule_id == tracker.OVERALL]
    if len(ov):
        st.markdown("#### Overall next steps recorded per granary")
        st.dataframe(ov[["granary", "kpkm_action", "decided_by", "data_as_of", "saved_at"]].rename(
            columns={"granary": "Granary", "kpkm_action": "Overall next step / conclusion", "decided_by": "Decided by",
                     "data_as_of": "Advice based on data of", "saved_at": "Saved"}), hide_index=True, width="stretch")

    with st.expander("Full history (every saved version)"):
        st.dataframe(hist.sort_values("saved_at", ascending=False).rename(columns=SHOW), hide_index=True, width="stretch")
    st.caption(f"Stored in: {store.describe(D.DATA_DIR)}")
    st.download_button("Download the decision history (CSV)", hist.to_csv(index=False).encode("utf-8"),
                       "decision_tracker.csv", "text/csv", key="trk_dl")
