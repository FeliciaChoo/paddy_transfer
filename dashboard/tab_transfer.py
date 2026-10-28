"""Tabs 6 and 7: the Pre-Transfer Assessment (steps 1 to 4) and Transfer Learning (steps 5 to 7).

The interface only DISPLAYS and TRIGGERS. The order of the steps is enforced by acts_transfer.workflow.TransferWorkflow in the backend: a locked
button here is a convenience, and calling the backend directly cannot skip a step either. Every number shown is computed by acts_transfer on the
project's real data (data/transfer_yield_dataset.csv and data/transfer_decade_features.csv, built by pipeline/build_transfer_dataset.py).
"""
import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from acts_transfer import Config, Domain, TransferWorkflow
from acts_transfer import harmonise as hm
from acts_transfer.assessment import METRIC_GUIDE
from acts_transfer.errors import WorkflowError
from acts_transfer.methods import LABELS, available_bases
from acts_transfer.report import workflow_markdown

FEATS = ["NDVI_mean", "NDVI_max", "NDVI_amplitude", "EVI_mean", "SAR_VV_dB_mean", "Mean_Temp_C_mean", "Annual_Rain_mm", "Solar_Radiation_MJ_mean"]
FINE = ["NDVI", "EVI", "SAR_VV_dB", "Mean_Temp_C", "Rain_decade_mm", "Solar_Radiation_MJ"]
UNITS = {"NDVI_mean": "unitless", "NDVI_max": "unitless", "NDVI_amplitude": "unitless", "EVI_mean": "unitless", "SAR_VV_dB_mean": "dB",
         "Mean_Temp_C_mean": "deg C", "Annual_Rain_mm": "mm per year", "Solar_Radiation_MJ_mean": "MJ/m2", "Yield_kg_ha": "kg/ha"}
ICON = {"complete": "✅", "current": "▶️", "locked": "🔒", "blocked": "⛔"}
READY = {"ready": ("success", "READY"), "conditional": ("warning", "CONDITIONALLY READY"), "not_ready": ("error", "NOT READY")}
SUIT = {"recommended": ("success", "TRANSFER RECOMMENDED"), "conditionally_recommended": ("warning", "TRANSFER CONDITIONALLY RECOMMENDED"),
        "not_recommended": ("error", "TRANSFER NOT RECOMMENDED")}
BASELINE_KEYS = ("source_only", "target_only", "target_mean", "season_trend", "pooled")


# ------------------------------------------------------------------------------------------------ plumbing
@st.cache_data(show_spinner=False)
def _load(path_y, path_f, mtime):
    return pd.read_csv(path_y), pd.read_csv(path_f)


def _data(D):
    y, f = D.DATA_DIR / "transfer_yield_dataset.csv", D.DATA_DIR / "transfer_decade_features.csv"
    if not (y.exists() and f.exists()):
        return None
    return _load(str(y), str(f), y.stat().st_mtime)


def _domain(Y, F, g):
    return Domain(g, Y[Y.District == g].sort_values("Year").reset_index(drop=True), list(FEATS), "Yield_kg_ha", "Year", dict(UNITS), "Optical_observed_pct",
                  F[F.District == g].reset_index(drop=True), list(FINE))


def _wf():
    if "acts_wf" not in st.session_state:
        st.session_state["acts_wf"] = TransferWorkflow(Config())
    return st.session_state["acts_wf"]


def _progress(wf):
    rows = wf.status()
    done = sum(r["status"] == "complete" for r in rows)
    st.progress(done / len(rows), text=f"Workflow progress: {done} of {len(rows)} steps complete")
    cols = st.columns(len(rows))
    for c, r in zip(cols, rows):
        c.markdown(f"{ICON[r['status']]}<br><span style='font-size:0.8em'>{r['step']}</span>", unsafe_allow_html=True)
    blocked = [r for r in rows if r["status"] == "blocked"]
    if blocked:
        st.error(f"{blocked[0]['step']} is blocked: {blocked[0]['detail']}")


def _badge(kind, text):
    getattr(st, kind)(f"**{text}**")


def _step_status(wf, key):
    return {r["key"]: r["status"] for r in wf.status()}[key]


def _ecdf(a, b, name_a, name_b, feature):
    fig = go.Figure()
    for v, nm, c in ((a, name_a, "#1f77b4"), (b, name_b, "#d62728")):
        v = np.sort(pd.to_numeric(pd.Series(v), errors="coerce").dropna().values)
        fig.add_trace(go.Scatter(x=v, y=np.arange(1, len(v) + 1) / max(len(v), 1), mode="lines", line=dict(shape="hv", color=c), name=f"{nm} (n={len(v)})"))
    fig.update_layout(height=300, xaxis_title=feature, yaxis_title="cumulative share", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3))
    return fig


# ------------------------------------------------------------------------------------------------ tab 6
def render_assessment(D):
    st.subheader("Pre-Transfer Assessment")
    wf = _wf()
    data = _data(D)
    if data is None:
        st.error("The transfer datasets are missing. Run `python pipeline/build_transfer_dataset.py --clean eda/DailyGranary_clean.csv --out data` first.")
        return
    Y, F = data
    with st.expander("Why this comes first, and where you are in the workflow", expanded=False):
        st.markdown(
            "Transfer learning can make predictions **worse** when the source and target differ in their data, their units, or how inputs relate to yield. "
            "So nothing is transferred until the data have been **assessed** (Step 2) and **harmonised** (Step 3), and the result has been judged "
            "(Step 4). The order is enforced by the backend, not only by this page: Transfer Learning (tab 7) stays locked until Steps 1 to 4 are complete.\n\n"
            "**Step 1** select domains → **2** assess data → **3** harmonise data → **4** evaluate suitability → **5** select strategy → **6** run transfer → "
            "**7** compare results (Steps 5 to 7 are in tab 7).")
    _progress(wf)

    # ---------------------------------------------------------------- Step 1
    st.markdown("### Step 1 · Select source and target domains")
    names = sorted(Y.District.unique())
    c1, c2, c3 = st.columns([2, 2, 1])
    src = c1.selectbox("Source (knowledge comes from here)", names, index=names.index("IADA Barat Laut Selangor") if "IADA Barat Laut Selangor" in names else 0, key="tr_src",
                       help="IADA Barat Laut Selangor covers the Sekinchan area.")
    tgt = c2.selectbox("Target (predictions are made here)", names, index=names.index("MADA") if "MADA" in names else 1, key="tr_tgt")
    base = c3.selectbox("Base model", available_bases(), key="tr_base", help="Parameter transfer needs Ridge. XGBoost appears only if installed.")
    n_seasons = int(Y[Y.District == tgt].Year.nunique())
    c4, c5 = st.columns(2)
    scarce = c4.slider("Labelled target seasons a model may train on (scarcity)", 2, max(n_seasons - 1, 2), max(n_seasons - 1, 2), key="tr_scarce",
                       help="Both domains have the same number of yearly yields here, so 'data-scarce' is simulated by limiting how many target seasons are used for training.")
    strat = c5.radio("Primary validation", ["Leave-one-season-out", "Forward (earlier seasons only)"], horizontal=True, key="tr_strat",
                     help="Both are always run; this chooses which one is shown first.")
    with st.expander("Thresholds (configurable; defaults and their basis are in acts_transfer/config.py)"):
        t1, t2, t3, t4 = st.columns(4)
        alpha = t1.number_input("Significance level (after BH correction)", 0.001, 0.2, 0.05, 0.005, key="tr_alpha")
        smd_m = t2.number_input("SMD 'medium' from", 0.1, 1.5, 0.5, 0.1, key="tr_smdm")
        smd_l = t3.number_input("SMD 'large' from", 0.2, 2.5, 0.8, 0.1, key="tr_smdl")
        perms = t4.number_input("Permutations (MMD, CORAL)", 100, 2000, 300, 100, key="tr_perm")
        t5, t6, t7 = st.columns(3)
        mpf = t5.number_input("Labelled target rows wanted per feature", 1.0, 10.0, 3.0, 0.5, key="tr_mpf")
        qmin = t6.number_input("Flag data-quality column below (%)", 0.0, 100.0, 20.0, 5.0, key="tr_qmin")
        mwarn = t7.number_input("Missing-value warning above (%)", 1.0, 50.0, 10.0, 1.0, key="tr_mwarn")
    if src == tgt:
        st.warning("Source and target must be different.")
    if st.button("Confirm domains and load data", key="tr_confirm", disabled=src == tgt):
        cfg = Config(alpha=alpha, smd_medium=smd_m, smd_large=smd_l, mmd_permutations=int(perms), min_obs_per_feature=mpf, quality_min_pct=qmin,
                     missing_warn_pct=mwarn, max_target_train_groups=None if scarce >= n_seasons - 1 else int(scarce),
                     cv_strategy="loso" if strat.startswith("Leave") else "forward")
        wf.select_domains(_domain(Y, F, src), _domain(Y, F, tgt), base, cfg)
        st.session_state.pop("tr_override", None)
    if wf.source is not None:
        st.success(f"Selected: **{wf.source.name}** (source, {len(wf.source.df)} yearly rows) → **{wf.target.name}** (target, {len(wf.target.df)} yearly rows); base model {wf.base}.")
        if wf.source.name != src or wf.target.name != tgt:
            st.info("The selection above differs from the confirmed one: press the button to apply it (this clears the later steps).")

    # ---------------------------------------------------------------- Step 2
    st.markdown("### Step 2 · Assess data")
    s2 = _step_status(wf, "assess")
    if wf.source is None:
        st.info("🔒 Locked: complete Step 1 first.")
    else:
        if st.button("Run data assessment" if wf.report is None else "Re-run data assessment", key="tr_assess", type="primary"):
            with st.spinner("Assessing data quality, feature compatibility and domain differences..."):
                try:
                    wf.assess()
                except Exception as e:
                    st.error(f"The assessment failed: {e}")
        if wf.report is not None:
            _show_assessment(wf)

    # ---------------------------------------------------------------- Step 3
    st.markdown("### Step 3 · Harmonise data")
    if wf.report is None:
        st.info("🔒 Locked: complete Step 2 first.")
    elif wf.report.readiness == "not_ready":
        st.error("⛔ Blocked: data readiness is NOT READY. Resolve the blocking problems listed in Step 2 (they come from the data, not from this page).")
    else:
        spec = hm.default_spec(wf.report, wf.source, wf.target)
        st.markdown("Static, label-free clean-up only. Imputation and standardisation are **fitted inside each training fold later**, so no held-out value shapes them. "
                    "The original datasets are never modified.")
        conv = st.checkbox("Convert yield from kg/ha to t/ha", value=True, key="tr_conv")
        if not conv:
            spec.conversions = []
        drop = st.multiselect("Drop features (optional)", wf.source.features, key="tr_drop")
        spec.drop_features = {f: "dropped by the user" for f in drop}
        if st.button("Apply data harmonisation", key="tr_harm", type="primary"):
            try:
                with st.spinner("Harmonising..."):
                    wf.harmonise(spec)
            except WorkflowError as e:
                st.error(f"Harmonisation refused: {e}")
        if wf.bundle is not None:
            b = wf.bundle
            c1, c2 = st.columns(2)
            c1.markdown("**Transformations applied**")
            c1.dataframe(pd.DataFrame([{"step": l["step"], "domain": l["domain"], "detail": str(l["detail"]), "rows affected": l["rows_affected"]} for l in b.log]),
                         hide_index=True, width="stretch")
            c2.markdown("**Features: retained or removed**")
            c2.dataframe(b.feature_table, hide_index=True, width="stretch")
            st.markdown("**Validation checks**")
            v = pd.DataFrame(b.validation)
            v["passed"] = np.where(v.passed, "✅ passed", "❌ FAILED")
            st.dataframe(v, hide_index=True, width="stretch")
            with st.expander("Harmonised feature table (first rows; label column y is in t/ha after conversion)"):
                st.dataframe(pd.concat([b.source.head(4).assign(domain=wf.source.name), b.target.head(4).assign(domain=wf.target.name)]).round(3), hide_index=True,
                             width="stretch")
            st.success("Harmonisation validated.") if b.ready else st.error("Harmonisation validation FAILED.")

    # ---------------------------------------------------------------- Step 4
    st.markdown("### Step 4 · Evaluate transfer suitability")
    if wf.bundle is None:
        st.info("🔒 Locked: complete Step 3 first.")
    else:
        if st.button("Evaluate transfer suitability", key="tr_suit", type="primary"):
            try:
                with st.spinner("Evaluating scenarios A to F..."):
                    wf.evaluate_suitability()
            except WorkflowError as e:
                st.error(f"Refused: {e}")
        if wf.suit is not None:
            s = wf.suit
            _badge(*SUIT[s["suitability"]])
            for r in s["reasons"]:
                st.markdown(f"- {r}")
            st.caption("This verdict comes from distribution evidence and data quantity. It is not a prediction error: errors are measured only in Step 7, on held-out seasons.")
            for sc in s["scenarios"]:
                with st.expander(f"{'🟢' if sc['met'] else '⚪'} Scenario {sc['id']}: {sc['name']}  ({'conditions met' if sc['met'] else 'not met'})"):
                    st.markdown(f"**Observed conditions:** {sc['observed']}")
                    st.markdown(f"**Evidence:** {sc['evidence']}")
                    st.markdown(f"**Recommended method(s):** {', '.join(LABELS.get(m, m) for m in sc['methods']) or 'none'}")
                    st.markdown(f"**Why suitable:** {sc['why']}")
                    st.markdown(f"**Assumptions:** {sc['assumptions']}")
                    st.markdown(f"**Benefits and risks:** {sc['benefits_risks']}")
                    st.markdown(f"**Alternatives:** {sc['alternatives']}")
                    st.markdown(f"**Can this system run it?** " + (("yes: " + ", ".join(sc["executable"])) if sc["executable"] else "no eligible method")
                                + (f"  (not eligible: {sc['not_executable']})" if sc["not_executable"] else ""))
            st.success("Steps 1 to 4 are complete. Continue in tab **7 · Transfer Learning**.")


def _show_assessment(wf):
    r = wf.report
    _badge(*READY[r.readiness])
    for x in r.readiness_reasons:
        st.markdown(f"- {x}")
    q1, q2 = st.columns(2)
    for col, nm, q in ((q1, "Source", r.source_quality), (q2, "Target", r.target_quality)):
        s = q["summary"]
        col.markdown(f"**{nm}: {s['name']}**")
        col.caption(f"{s['n_obs']} rows · seasons {s['groups'][0]}–{s['groups'][-1]} · {s['n_labelled']} labelled · {s['features']} features · {s['duplicates']} duplicates")
        col.dataframe(q["features"][["feature", "unit", "missing_pct", "outliers", "invalid", "min", "max", "mean"]].round(3), hide_index=True, width="stretch")
        for sev, m in q["issues"]:
            col.markdown(f"- {'⛔' if sev == 'block' else ('⚠️' if sev == 'warn' else 'ℹ️')} {m}")
    st.markdown("**Feature compatibility** (features that are only proposed are shown as such and are never used or filled in)")
    st.dataframe(r.compat, hide_index=True, width="stretch")
    st.markdown("**Domain differences between source and target**")
    lvl = st.radio("Level of comparison", ["Modelling level (yearly rows, the model's input)", "Ten-day level (many more observations of the same drivers)"], key="tr_lvl", horizontal=True)
    dd = r.diff_model if lvl.startswith("Modelling") else r.diff_fine
    if not dd:
        st.info("Not available at this level.")
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MMD²", f"{dd.get('mmd2', float('nan')):.4f}", f"permutation p = {dd.get('mmd_p', float('nan')):.3f}", delta_color="off",
                  help="Joint difference over all shared features. The smallest possible p is 1/(permutations+1).")
        m2.metric("CORAL distance", f"{dd.get('coral', float('nan')):.4f}", f"p = {dd.get('coral_p', float('nan')):.3f}" + ("" if dd.get("coral_reliable", True) else " (not interpretable)"),
                  delta_color="off", help="How features vary together. Not interpretable when there are fewer rows than features.")
        m3.metric("Rows used (source / target)", f"{dd['n_complete_source']} / {dd['n_complete_target']}")
        m4.metric("Features differing (adj. KS)", f"{int(dd['table'].KS_significant.sum())} / {len(dd['table'])}" if "KS_significant" in dd["table"] else "n/a")
        st.dataframe(dd["table"][["feature", "mean_source", "mean_target", "SMD", "SMD_size", "KS", "KS_p_adj", "Wasserstein", "Wasserstein_per_SD"]].round(3),
                     hide_index=True, width="stretch")
        fd = getattr(r, "fine_diag", None)
        if fd and not lvl.startswith("Modelling"):
            a1, a2, a3 = st.columns(3)
            a1.metric("Climate AUC", "n/a" if fd.get("auc_climate") is None else f"{fd['auc_climate']:.2f}",
                      help="Can a classifier tell the two climates apart? 0.5 = no, 1.0 = perfectly. At or above the configured level the domain is called far.")
            a2.metric("Vegetation and radar AUC", "n/a" if fd.get("auc_vegetation") is None else f"{fd['auc_vegetation']:.2f}")
            a3.metric("Calendar offset", f"{fd['calendar_offset_days']} days" if "calendar_offset_days" in fd else "n/a",
                      help="Informational: how far the mean seasonal NDVI curves are shifted. Annual-aggregate features are much less sensitive to it.")
        feat = st.selectbox("Compare the distribution of", dd["table"].feature.tolist(), key="tr_featsel")
        sdf = wf.source.df if lvl.startswith("Modelling") else wf.source.fine_df
        tdf = wf.target.df if lvl.startswith("Modelling") else wf.target.fine_df
        st.plotly_chart(_ecdf(sdf[feat], tdf[feat], wf.source.name, wf.target.name, feat), width="stretch", key="tr_ecdf")
        row = dd["table"][dd["table"].feature == feat].iloc[0]
        st.caption(f"{feat}: mean {row.mean_source:.3g} vs {row.mean_target:.3g}; |SMD| = {abs(row.SMD):.2f} ({row.SMD_size}); KS = {row.KS:.2f}, adjusted p = {row.KS_p_adj:.3f}. "
                   "A small SMD with a significant KS means the shape or spread differs even though the means agree.")
    if r.label:
        st.markdown("**Yield (label) distribution:** " + f"means {r.label['mean_source']:.0f} vs {r.label['mean_target']:.0f} kg/ha; SMD {r.label['SMD']:+.2f}; KS p = {r.label['KS_p']:.3f}")
    st.markdown("**Negative-transfer risks**")
    for x in r.risks or ["none detected by the tests (they have low power with few rows)"]:
        st.markdown(f"- {x}")
    st.markdown("**Recommended next steps**")
    for x in r.next_steps:
        st.markdown(f"- {x}")
    with st.expander("What each metric measures, and when to use it"):
        for k, v in METRIC_GUIDE.items():
            st.markdown(f"- **{k}**: {v}")
        st.caption("These are measures of how similar two distributions are. They are not prediction errors, and a small distance does not guarantee a small error.")
    st.download_button("Download the assessment report (Markdown)", r.to_markdown().encode("utf-8"), "pre_transfer_assessment.md", "text/markdown", key="tr_dl_assess")


# ------------------------------------------------------------------------------------------------ tab 7
def render_transfer(D):
    st.subheader("Transfer Learning")
    wf = _wf()
    _progress(wf)
    ok, why = wf.can_run()
    if not ok:
        st.warning(f"🔒 Transfer learning is locked. {why}")
        st.markdown("Complete Steps 1 to 4 in tab **6 · Pre-Transfer Assessment** first. The backend refuses to run a transfer method without a valid assessment, "
                    "validated harmonisation and a suitability result built from them, even if this page is bypassed.")
        return
    s = wf.suit
    _badge(*SUIT[s["suitability"]])
    for r in s["reasons"]:
        st.markdown(f"- {r}")

    # ---------------------------------------------------------------- Step 5
    st.markdown("### Step 5 · Select transfer strategy")
    tab = s["methods"].copy()
    st.dataframe(tab[["method", "family", "eligible", "recommended_by", "reason"]].rename(columns={"recommended_by": "recommended for scenario"}), hide_index=True, width="stretch")
    recommended = [m for sc in s["scenarios"] if sc["met"] for m in sc["executable"] if m not in BASELINE_KEYS]
    eligible = [k for k in tab[tab.eligible].key if k not in BASELINE_KEYS]
    default = list(dict.fromkeys(recommended)) or eligible[:1]
    chosen = st.multiselect("Transfer methods to run (the baselines always run)", eligible, default=[m for m in default if m in eligible], format_func=lambda k: LABELS.get(k, k), key="tr_methods")
    for k in chosen:
        n = tab[tab.key == k].iloc[0]
        with st.expander(f"How {LABELS.get(k, k)} uses distance and features"):
            st.markdown(f"- **Features compared:** {n.features_compared}\n- **Scaling:** {n.scaling}\n- **Metric:** {n.metric}\n- **Why suitable:** {n.why_suitable}\n"
                        f"- **How distance acts:** {n.distance_influence}\n- **Limitations:** {n.limitations}")
    st.caption("A smaller feature distance does not guarantee a smaller yield error: distribution similarity and prediction error are different questions.")
    override = None
    if s["suitability"] == "not_recommended":
        st.error("Transfer is NOT RECOMMENDED. You may still run it for research, but you must state why; the reason is recorded in the report.")
        override = st.text_input("Override reason (required)", key="tr_override")
    if st.button("Confirm strategy", key="tr_strategy"):
        try:
            wf.select_strategy(chosen)
            st.success(f"Strategy confirmed: {', '.join(LABELS.get(k, k) for k in chosen) or 'baselines only'}")
        except WorkflowError as e:
            st.error(f"Refused: {e}")

    # ---------------------------------------------------------------- Step 6
    st.markdown("### Step 6 · Run transfer learning")
    if wf.strategy is None:
        st.info("🔒 Locked: confirm a strategy in Step 5 first.")
    else:
        if st.button("Run transfer learning and baselines", key="tr_run", type="primary"):
            try:
                with st.spinner("Running leave-one-season-out and forward validation (all fitting inside the training folds)..."):
                    wf.run(override_reason=override)
            except WorkflowError as e:
                st.error(f"Refused: {e}")
            except Exception as e:
                st.error(f"The run failed: {type(e).__name__}: {e}")

    # ---------------------------------------------------------------- Step 7
    st.markdown("### Step 7 · Compare results")
    if wf.result is None:
        st.info("🔒 Locked: run Step 6 first.")
        return
    _show_results(wf)


def _show_results(wf):
    results = wf.results
    first = "loso" if wf.cfg.cv_strategy == "loso" else "forward"
    pick = st.radio("Validation shown", ["Leave-one-season-out", "Forward (earlier seasons only)"], index=0 if first == "loso" else 1, horizontal=True, key="tr_valshow")
    r = results["loso" if pick.startswith("Leave") else "forward"]
    if r.predictions.empty:
        st.warning("This validation produced no predictions (too few seasons).")
        return
    for n in r.notes:
        st.warning(n)
    st.caption(f"Yield in t/ha after harmonisation · {r.n_folds} held-out seasons · every method is scored on the same held-out rows · base model {wf.base}.")
    m = r.metrics.copy()
    m["kind"] = np.where(m.method.isin(BASELINE_KEYS), "baseline", "transfer")
    m["method"] = m.method.map(lambda k: LABELS.get(k, k))
    st.dataframe(m[["method", "kind", "n_test", "MAE", "RMSE", "R2", "R2_note"]].round(3), hide_index=True, width="stretch")
    fig = go.Figure(go.Bar(x=m.method, y=m.MAE, marker_color=["#9aa3ad" if k == "baseline" else "#2f855a" for k in m.kind], text=m.MAE.round(3), textposition="outside"))
    fig.update_layout(height=320, yaxis_title="MAE (t/ha), lower is better", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch", key="tr_bar")
    st.markdown("**Does the answer depend on how it is validated?**")
    sens = pd.DataFrame({("Leave-one-season-out" if k == "loso" else "Forward"): v.metrics.set_index("method").MAE for k, v in results.items()}).round(3)
    sens.index = [LABELS.get(i, i) for i in sens.index]
    st.dataframe(sens.sort_values(sens.columns[-1]), width="stretch")

    st.markdown("**Improvement or deterioration against a baseline** (negative change = lower error)")
    bases = [b for b in BASELINE_KEYS if b in set(r.improvement.baseline)]
    bsel = st.selectbox("Baseline", bases, index=bases.index("target_only") if "target_only" in bases else 0, format_func=lambda k: LABELS.get(k, k), key="tr_base_sel")
    imp = r.improvement[r.improvement.baseline == bsel].copy()
    imp["method"] = imp.method.map(lambda k: LABELS.get(k, k))
    st.dataframe(imp[["method", "n_seasons", "mean_abs_err_change", "relative_change_pct", "ci95_low", "ci95_high", "better_in", "verdict"]].round(3), hide_index=True, width="stretch")
    trans = r.improvement[(r.improvement.baseline == bsel) & (~r.improvement.method.isin(BASELINE_KEYS))]
    if len(trans):
        better = trans[trans.verdict == "improved"].method.tolist()
        st.info(f"Against {LABELS.get(bsel, bsel)}: " + (f"clearly improved: {', '.join(LABELS.get(k, k) for k in better)}." if better else
                "no transfer method was clearly better. A lower average is not enough when only a few seasons are tested: look at the interval and the verdict."))

    st.markdown("**Predicted versus actual, and where the errors are**")
    meths = st.multiselect("Methods to plot", r.metrics.method.tolist(), default=r.metrics.method.tolist()[:4], format_func=lambda k: LABELS.get(k, k), key="tr_plot")
    p = r.predictions[r.predictions.method.isin(meths)]
    c1, c2 = st.columns(2)
    f1 = go.Figure()
    lo, hi = p[["y_true", "y_pred"]].min().min(), p[["y_true", "y_pred"]].max().max()
    f1.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dot", color="#999"), name="perfect"))
    for k, d in p.groupby("method"):
        f1.add_trace(go.Scatter(x=d.y_true, y=d.y_pred, mode="markers", name=LABELS.get(k, k), text=d.group, hovertemplate="season %{text}<br>actual %{x:.2f}<br>predicted %{y:.2f}<extra></extra>"))
    f1.update_layout(height=340, xaxis_title="actual yield (t/ha)", yaxis_title="predicted (t/ha)", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3))
    c1.plotly_chart(f1, width="stretch", key="tr_scatter")
    f2 = go.Figure()
    for k, d in p.groupby("method"):
        f2.add_trace(go.Box(y=d.abs_err, name=LABELS.get(k, k), boxpoints="all", jitter=0.4))
    f2.update_layout(height=340, yaxis_title="absolute error (t/ha)", margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    c2.plotly_chart(f2, width="stretch", key="tr_box")
    f3 = go.Figure()
    for k, d in p.groupby("method"):
        f3.add_trace(go.Bar(x=d.group.astype(str), y=d.abs_err, name=LABELS.get(k, k)))
    f3.update_layout(height=320, barmode="group", xaxis_title="held-out season", yaxis_title="absolute error (t/ha)", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(f3, width="stretch", key="tr_season")

    st.markdown("**Why did the error go up or down?**")
    st.caption("Evidence only. 'Verified finding' = a measured condition that is true in this run (it does not prove it caused the error). 'Possible explanation' = consistent "
               "with the data but unproven. 'Not supported' = the measurements point against it.")
    cand = [k for k in r.metrics.method if k not in BASELINE_KEYS]
    if cand:
        d1, d2 = st.columns(2)
        dm = d1.selectbox("Method", cand, format_func=lambda k: LABELS.get(k, k), key="tr_diag_m")
        db = d2.selectbox("Compared with", [b for b in BASELINE_KEYS if b in set(r.metrics.method)], format_func=lambda k: LABELS.get(k, k), key="tr_diag_b",
                          index=0)
        diag_wf = wf
        saved = wf.result
        wf.result = r
        try:
            d, summ = diag_wf.diagnose(dm, db)
        finally:
            wf.result = saved
        if len(d):
            st.markdown(summ)
            st.dataframe(d, hide_index=True, width="stretch")
    fails = wf.result.failures
    if fails:
        st.warning("Some method runs failed: " + "; ".join(sorted({f"{LABELS.get(f['method'], f['method'])}: {f['reason']}" for f in fails})))
    d1, d2 = st.columns(2)
    d1.download_button("Download the full report (Markdown)", workflow_markdown(wf).encode("utf-8"), "transfer_assessment_and_evaluation.md", "text/markdown", key="tr_dl_full")
    d2.download_button("Download held-out predictions (CSV)", r.predictions.to_csv(index=False).encode("utf-8"), "transfer_predictions.csv", "text/csv", key="tr_dl_pred")
