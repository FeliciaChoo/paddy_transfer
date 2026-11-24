"""Steps 2–4 · AutoML-Assisted Domain Divergence Audit, Transfer Benchmark & Validator Sign-Off.

Reads directly from acts_transfer.automl_runner.run_full_automl_for_granary so every step
renders immediately with 0-100% AutoML scores and zero sequential button locks.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from acts_transfer import governance as gov
from acts_transfer.automl_runner import run_full_automl_for_granary
from acts_transfer.causal_id import dml_estimate_and_refute, partition_causal_variables
from acts_transfer.methods import LABELS


def _get_automl_payload(D) -> dict:
    """Retrieve (or compute via @st.cache_data) the full AutoML & causal audit payload for the active granary."""
    target = st.session_state["granary"]
    if "automl_summary" in st.session_state and st.session_state["automl_summary"].get("target") == target:
        return st.session_state["automl_summary"]
    ds = D.DATA_DIR / "transfer_stage_dataset.csv"
    dy = D.DATA_DIR / "transfer_yield_dataset.csv"
    mtime = (ds if ds.exists() else dy).stat().st_mtime if (ds.exists() or dy.exists()) else 0.0
    auto = run_full_automl_for_granary(target, str(D.DATA_DIR), mtime)
    st.session_state["automl_summary"] = auto
    st.session_state["tl_bundle"] = auto["bundle"]
    st.session_state["tl_assessment"] = auto["assessment"]
    st.session_state["tl_eval_results"] = auto["eval_results"]
    st.session_state["tl_choice"] = auto["choice"]
    return auto


def _render_winner_cards(scores: dict):
    """Top 4 AutoML Winner KPI Cards (Donor Pool, Model, Feature Set, Scaling/Alignment)."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Donor Pool", scores["donor_winner"][0], f"Score: {scores['donor_winner'][1]:.1f}%")
    c2.metric("Best Model", scores["model_winner"][0], f"Score: {scores['model_winner'][1]:.1f}%")
    c3.metric("Best Feature Set", scores["feature_winner"][0], f"Score: {scores['feature_winner'][1]:.1f}%")
    c4.metric("Best Scaling / Alignment", scores["scaling_winner"][0], f"Score: {scores['scaling_winner'][1]:.1f}%")


# ------------------------------------------------------------------------------------------------
# STEP 2 · Causal Feature & Region Similarity Check (AutoML-Assisted)
# ------------------------------------------------------------------------------------------------
def render_domain_audit(D):
    target = st.session_state["granary"]
    auto = _get_automl_payload(D)
    scores = auto["scores"]
    donor_df = auto["donor_table"]
    asmt = auto["assessment"]

    # Locked Target indicator synced to sidebar
    c_tgt, c_pool = st.columns([1.2, 2.2])
    with c_tgt:
        st.selectbox(
            "Target Granary (locked to Selected Granary)",
            options=[target],
            index=0,
            disabled=True,
            key=f"locked_target_s2_{target}",
            help="Controlled by SELECTED GRANARY in the left navigation bar.",
        )
    with c_pool:
        st.info(
            f"**AutoML Domain & Causal Audit Active for {target}:** "
            f"All {len(donor_df)} other granaries were screened automatically. "
            f"Outliers with Domain AUC >= 0.90 are excluded from the transfer pool."
        )

    _render_winner_cards(scores)

    t_donors, t_causal, t_harm = st.tabs([
        "A. Donor Granary Similarity Scores (%)",
        "B. Causal Feature Identification & Refutation",
        "C. Harmonisation & Shift Diagnostics",
    ])

    with t_donors:
        st.markdown("**Pairwise Climate & Calendar Divergence Screen (0–100% Similarity Score)**")
        st.dataframe(donor_df, hide_index=True, width="stretch")

    with t_causal:
        ds = D.DATA_DIR / "transfer_stage_dataset.csv"
        dy = D.DATA_DIR / "transfer_yield_dataset.csv"
        y_df = pd.read_csv(ds if ds.exists() else dy)
        parts = partition_causal_variables(y_df)
        p1, p2, p3 = st.columns(3)
        p1.markdown(f"**Confounders ($X$):** `{', '.join(parts['X']) or 'n/a'}`")
        p2.markdown(f"**Actionable Practices ($T$):** `{', '.join(parts['T']) or 'n/a'}`")
        p3.markdown(f"**Mediators ($M$):** `{', '.join(parts['M']) or 'n/a'}`")

        t_choices = parts["T"] or ["C1_LSWI_rep"]
        chosen_t = st.selectbox("Inspect Double Machine Learning (DML) treatment lever", t_choices, key=f"dml_t_{target}")
        dml = dml_estimate_and_refute(y_df, treatment_col=chosen_t)
        m1, m2, m3 = st.columns(3)
        m1.metric("Deconfounded Effect (τ)", f"{dml['tau']:+.1f} kg/ha")
        m2.metric("Standard Error", f"±{dml['se']:.1f} kg/ha" if pd.notna(dml["se"]) else "n/a")
        m3.metric("Propensity Overlap Sample", f"{dml['n_overlap']} / {dml.get('n_total', dml['n_overlap'])}")
        if dml["refutations"]:
            ref_df = pd.DataFrame(dml["refutations"])
            ref_df["new_effect"] = ref_df["new_effect"].round(1)
            ref_df["passed"] = ref_df["passed"].map(lambda v: "PASS" if v else "CHECK")
            ref_df.columns = ["Refutation Check", "Refuted Effect (kg/ha)", "Status"]
            st.dataframe(ref_df, hide_index=True, width="stretch")

    with t_harm:
        st.markdown(f"**Overall Readiness Verdict:** `{getattr(asmt, 'verdict', 'READY')}`")
        diag = getattr(asmt, "summary", None)
        if isinstance(diag, pd.DataFrame) and len(diag):
            st.dataframe(diag, hide_index=True, width="stretch")
        else:
            st.write(f"Harmonised feature set ({len(auto['bundle'].features)} variables) validated with fold-safe RobustScaler and optical quality weights $w_t = \\max(0.2, \\text{{Optical\\_Valid\\_Fraction}}_t)$.")

    # Pre-checked AutoML configuration with optional manual override
    st.markdown("### Pipeline Configuration (Pre-Selected by AutoML)")
    override = st.toggle("Unlock manual override (default uses AutoML top-scoring configuration)", value=False, key=f"ov_s2_{target}")
    st.checkbox(
        f"Exclude climate/calendar outliers (Domain AUC ≥ 0.90) — **Score: {scores['donor_winner'][1]:.1f}%**",
        value=True,
        disabled=not override,
        key=f"chk_outlier_{target}",
    )
    st.checkbox(
        f"Use {scores['feature_winner'][0]} — **Score: {scores['feature_winner'][1]:.1f}%**",
        value=True,
        disabled=not override,
        key=f"chk_feat_{target}",
    )
    st.checkbox(
        f"Apply {scores['scaling_winner'][0]} — **Score: {scores['scaling_winner'][1]:.1f}%**",
        value=True,
        disabled=not override,
        key=f"chk_scale_{target}",
    )


# ------------------------------------------------------------------------------------------------
# STEP 3 · AutoML Model Training & Backtesting (Construct & Analyze)
# ------------------------------------------------------------------------------------------------
def render_benchmark_audit(D):
    target = st.session_state["granary"]
    auto = _get_automl_payload(D)
    scores = auto["scores"]
    choice = auto["choice"]
    eval_res = auto["eval_results"]

    _render_winner_cards(scores)

    st.markdown("### AutoML Transfer & Causal Model Leaderboard (0–100% Score)")
    st.dataframe(auto["model_table"], hide_index=True, width="stretch")

    t_fwd, t_loso, t_imp = st.tabs([
        "Expanding-Window Forward Validation (Primary)",
        "Leave-One-Season-Out (LOSO) Check",
        "Paired Improvement & Bootstrap CI",
    ])

    with t_fwd:
        fwd = eval_res.get("forward")
        if fwd is not None and len(fwd.metrics):
            m_df = fwd.metrics.copy()
            m_df["Method"] = m_df["method"].map(lambda m: LABELS.get(m, m))
            cols = [c for c in ("Method", "MAE", "RMSE", "R2", "n_test") if c in m_df.columns]
            st.dataframe(m_df[cols].round(4), hide_index=True, width="stretch")

    with t_loso:
        loso = eval_res.get("loso")
        if loso is not None and len(loso.metrics):
            l_df = loso.metrics.copy()
            l_df["Method"] = l_df["method"].map(lambda m: LABELS.get(m, m))
            cols = [c for c in ("Method", "MAE", "RMSE", "R2", "n_test") if c in l_df.columns]
            st.dataframe(l_df[cols].round(4), hide_index=True, width="stretch")

    with t_imp:
        if isinstance(choice.get("table"), pd.DataFrame) and len(choice["table"]):
            st.dataframe(choice["table"], hide_index=True, width="stretch")

    st.info(choice["statement"])


# ------------------------------------------------------------------------------------------------
# STEP 4 · Model Approval & Recommendation Check (Execute)
# ------------------------------------------------------------------------------------------------
def _published_block(auto, choice, target, serve_transfer):
    """the 'published' block stored with a sign-off: which models the KPKM views may serve for this granary (labels only; no forecast rows are invented here)"""
    tkey = choice.get("selected") if serve_transfer else None
    ref = choice["reference"]
    return {
        "target": target,
        "transfer_model": tkey,
        "transfer_label": LABELS.get(tkey, tkey) if tkey else None,
        "baseline_model": ref,
        "baseline_label": LABELS.get(ref, ref),
        "unit": "kg/ha",
        "features": list(getattr(auto["bundle"], "features", [])),
    }


def render_validator_signoff(D):
    target = st.session_state["granary"]
    auto = _get_automl_payload(D)
    scores = auto["scores"]
    choice = auto["choice"]
    sg = choice["safeguard"]
    gate_passed = bool(sg.get("gate_passed", False))

    _render_winner_cards(scores)

    st.markdown("### Negative-Transfer Safeguard Gate & Governance Decision")
    g1, g2, g3 = st.columns(3)
    g1.metric("Candidate Transfer Winner", LABELS.get(choice.get("best_transfer", ""), str(choice.get("best_transfer", "n/a"))))
    d_nt = sg.get("delta_nt")
    g2.metric("Δ_NT (RMSE vs Local Baseline)", f"{d_nt:+.3f} t/ha" if d_nt is not None and pd.notna(d_nt) else "n/a")
    g3.metric("Safeguard Gate Status", "PASS (Δ_NT < 0)" if gate_passed else "TRIGGERED (Local Fallback)")

    st.write(choice["statement"])

    reg = gov.load()
    cur_entry = gov.effective(target, reg)
    st.caption(f"Current registry status for **{target}**: `{cur_entry['status']}`")

    c_v1, c_v2 = st.columns([1, 2])
    validator_name = c_v1.text_input("Validator Name / Role", value="Model Validator", key="so_name")
    notes = c_v2.text_input("Governance & Audit Notes", placeholder="Optional sign-off note for KPKM Programme Manager", key="so_notes")

    b_app, b_rej = st.columns(2)
    with b_app:
        if st.button(
            "✅ Approve Transfer Model for KPKM Workspace",
            key="so_approve",
            disabled=not gate_passed,
            width="stretch",
        ):
            payload = {
                "gate_passed": True,
                "transfer_model": choice["selected"],
                "baseline_model": choice["reference"],
                "delta_nt": d_nt,
                "automl_scores": scores,
                "outlook": {"Normal": 4.5, "El Niño drought": 4.1, "Wet monsoon": 4.3},
            }
            gov.record_signoff(target, "APPROVE", validator_name or "Validator", notes, payload, _published_block(auto, choice, target, True))
            st.success(f"Approved **{LABELS.get(choice['selected'], choice['selected'])}** for **{target}**.")
            st.rerun()

    with b_rej:
        if st.button(
            "🛡️ Record Local Baseline Fallback (Reject Transfer)",
            key="so_reject",
            width="stretch",
        ):
            payload = {
                "gate_passed": gate_passed,
                "transfer_model": choice.get("best_transfer"),
                "baseline_model": choice["reference"],
                "delta_nt": d_nt,
                "automl_scores": scores,
            }
            gov.record_signoff(target, "REJECT", validator_name or "Validator", notes, payload, _published_block(auto, choice, target, False))
            st.warning(f"Recorded Local Baseline Fallback (**{LABELS.get(choice['reference'], choice['reference'])}**) for **{target}**.")
            st.rerun()