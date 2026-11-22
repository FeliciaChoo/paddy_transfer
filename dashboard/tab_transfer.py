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
from acts_transfer import predict as predmod
from acts_transfer.assessment import METRIC_GUIDE
from acts_transfer.errors import WorkflowError
from acts_transfer.methods import LABELS, available_bases
from acts_transfer.report import workflow_markdown

from . import tab_modelcheck, tab_yield

TARGET_DEFAULT, SOURCE_DEFAULT, EXCLUDED_DEFAULT = "IADA Barat Laut Selangor", "MADA", "KETARA"
# legacy yearly summaries, used only when the growth-stage table has not been built yet
FEATS_LEGACY = ["NDVI_mean", "NDVI_max", "NDVI_amplitude", "EVI_mean", "SAR_VV_dB_mean", "Mean_Temp_C_mean", "Annual_Rain_mm", "Solar_Radiation_MJ_mean"]
FEATS = list(FEATS_LEGACY)                      # replaced by the C1_*/C2_* growth-stage features when data/transfer_stage_dataset.csv exists
FINE = ["EVI", "LSWI", "MNDWI", "SAR_VV_dB", "Mean_Temp_C", "Tmax_C", "Rain_decade_mm", "Solar_Radiation_MJ"]
_UNIT_OF = {"EVI": "unitless", "LSWI": "unitless", "MNDWI": "unitless", "SAR_VV_dB": "dB", "Rain_mm": "mm per stage", "Tmax_C": "deg C", "Solar_MJ": "MJ/m2"}
UNITS = {"NDVI_mean": "unitless", "NDVI_max": "unitless", "NDVI_amplitude": "unitless", "EVI_mean": "unitless", "SAR_VV_dB_mean": "dB",
         "Mean_Temp_C_mean": "deg C", "Annual_Rain_mm": "mm per year", "Solar_Radiation_MJ_mean": "MJ/m2", "Yield_kg_ha": "kg/ha", "Soil_pH": "pH", "Optical_Valid_Fraction": "fraction"}
ICON = {"complete": "✅", "current": "▶️", "locked": "🔒", "blocked": "⛔"}
READY = {"ready": ("success", "READY"), "conditional": ("warning", "CONDITIONALLY READY"), "not_ready": ("error", "NOT READY")}
SUIT = {"recommended": ("success", "TRANSFER RECOMMENDED"), "conditionally_recommended": ("warning", "TRANSFER CONDITIONALLY RECOMMENDED"),
        "not_recommended": ("error", "TRANSFER NOT RECOMMENDED")}
BASELINE_KEYS = ("source_only", "target_only", "target_mean", "season_trend", "persistence", "pooled")


# ------------------------------------------------------------------------------------------------ plumbing
@st.cache_data(show_spinner=False)
def _load(path_y, path_f, mtime):
    return pd.read_csv(path_y), pd.read_csv(path_f)


def stage_feature_names(Y):
    """the C1_* (Cycle 1, main season) and C2_* (Cycle 2, off-season) growth-stage features in the table, in a stable order, plus Soil_pH when present"""
    f = [c for c in Y.columns if c.startswith(("C1_", "C2_")) and not c.endswith(("Greenup_Decade", "Peak_Decade"))]
    return f + [c for c in ("Soil_pH", "Optical_Valid_Fraction") if f and c in Y.columns]


def split_cycles(feats):
    """(main-season features + soil pH, off-season C2_ features). The off-season cycle exists in only some granary-years, so it is opt-in"""
    c2 = [f for f in feats if f.startswith("C2_")]
    return [f for f in feats if not f.startswith("C2_")], c2


def _units_for(feats):
    import re
    u = dict(UNITS)
    for f in feats:
        m = re.match(r"^C[12]_(.+?)_(veg|rep|rip)$", f)
        if m:
            u[f] = _UNIT_OF.get(m.group(1), "unitless")
    return u


def _data(D):
    """(yearly table, ten-day table). The growth-stage table is preferred: one row per granary-year with C1_*/C2_* features (pipeline/build_stage_features.py)."""
    global FEATS
    ys, y, f = D.DATA_DIR / "transfer_stage_dataset.csv", D.DATA_DIR / "transfer_yield_dataset.csv", D.DATA_DIR / "transfer_decade_features.csv"
    if ys.exists() and f.exists():
        Y, F = _load(str(ys), str(f), ys.stat().st_mtime)
        FEATS = stage_feature_names(Y) or list(FEATS_LEGACY)
        return Y, F
    if not (y.exists() and f.exists()):
        return None
    FEATS = list(FEATS_LEGACY)
    return _load(str(y), str(f), y.stat().st_mtime)


def _domain(Y, F, g, feats=None):
    """one granary as a domain. g may be a single granary name or a list: a list is a POOL, with the granary kept in the 'District' column as the unit"""
    names = [g] if isinstance(g, str) else list(g)
    df = Y[Y.District.isin(names)].sort_values(["District", "Year"]).reset_index(drop=True)
    fine = F[F.District.isin(names)].reset_index(drop=True)
    nm = names[0] if len(names) == 1 else f"Pool of {len(names)} granaries"
    return Domain(nm, df, list(feats or split_cycles(FEATS)[0]), "Yield_kg_ha", "Year", _units_for(FEATS), "Optical_observed_pct", fine, list(FINE), unit="District")


@st.cache_data(show_spinner=False)
def _screen_cached(F, target):
    from acts_transfer.screening import screen_granaries
    return screen_granaries(F, target)


def _screen(F, target):
    try:
        return _screen_cached(F, target)
    except Exception:
        return None


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
    st.subheader("Data Assessment (Pre-Transfer Assessment)")
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
            "(Step 4). The order is enforced by the backend, not only by this page: Transfer Strategy (tab 4) stays locked until Steps 1 to 4 are complete.\n\n"
            "**Step 1** select domains → **2** assess data → **3** harmonise data → **4** evaluate suitability → **5** select strategy → **6** run transfer → "
            "**7** compare results (Step 5 is tab 4 · Transfer Strategy; Steps 6 and 7 are tab 5 · Train and Evaluate Model; yield prediction, the outlook and the advisory follow in tabs 6 to 8).")
    flow_bar(wf, "assessment")
    st.caption("Steps inside this tab:")
    _progress(wf)

    # ---------------------------------------------------------------- Step 1
    st.markdown("### Step 1 · Select source and target domains")
    names = sorted(Y.District.unique())
    c1, c2, c3 = st.columns([2, 2, 1])
    tgt = c2.selectbox("Target (predictions are made here)", names, index=names.index(TARGET_DEFAULT) if TARGET_DEFAULT in names else 0, key="tr_tgt",
                       help="IADA Barat Laut Selangor covers the Sekinchan area: the data-scarce target (7 yearly records).")
    src = c1.selectbox("Primary source (knowledge comes from here)", [n for n in names if n != tgt],
                       index=[n for n in names if n != tgt].index(SOURCE_DEFAULT) if SOURCE_DEFAULT in names and SOURCE_DEFAULT != tgt else 0, key="tr_src",
                       help="MADA (Kedah) is the primary benchmark source.")
    base = c3.selectbox("Base model", available_bases(), key="tr_base", help="Parameter transfer needs Ridge. XGBoost appears only if installed.")
    pool_mode = st.radio("Source pool", ["Primary source only", "Compatible pool (screened: all granaries except the target and those flagged)", "All other granaries, flagged ones included"],
                         horizontal=False, key="tr_pool",
                         help="A pooled source gives the granary-intercept model several granaries to learn from. The screen (Step 2) flags a granary whose climate is almost "
                              "perfectly separable from the target (domain AUC >= 0.90), which is KETARA here. Run the compatible pool and the all-granary pool to compare 8 and 9 sources.")
    screen = _screen(F, tgt)
    flagged = screen[screen.flagged].granary.tolist() if screen is not None and len(screen) else [EXCLUDED_DEFAULT]
    if pool_mode.startswith("Primary"):
        src_names = [src]
    elif pool_mode.startswith("Compatible"):
        src_names = [n for n in names if n != tgt and n not in flagged]
    else:
        src_names = [n for n in names if n != tgt]
    st.caption(f"Source granaries used ({len(src_names)}): " + ", ".join(src_names) + (f". Excluded by the screen: {', '.join(flagged)}." if not pool_mode.startswith("All") else "."))
    main_f, c2_f = split_cycles(FEATS)
    use_c2 = bool(c2_f) and st.checkbox("Also use Cycle 2 (off-season) stage features", value=False, key="tr_c2",
                                        help="Cycle 2 is detected in only some granary-years (never in most MADA years in this extraction), so these features are mostly missing and the "
                                             "readiness check will block them above 40% missing. Leave off unless your source and target both have an off-season crop.")
    use_rip = st.checkbox("Also use the Ripening-stage features (70 to 100 days)", value=False, key="tr_rip",
                          help="Off by default: ripening starts after the heading-stage nowcast (70 days), so these features are not available when the in-season forecast is made.")
    feats_sel = [f for f in main_f if use_rip or not f.endswith("_rip")] + (c2_f if use_c2 else [])
    n_seasons = int(Y[Y.District == tgt].Year.nunique())
    c4, c5 = st.columns(2)
    scarce = c4.slider("Labelled target seasons a model may train on (scarcity)", 2, max(n_seasons - 1, 2), max(n_seasons - 1, 2), key="tr_scarce",
                       help="Both domains have the same number of yearly yields here, so 'data-scarce' is simulated by limiting how many target seasons are used for training.")
    strat = c5.radio("Primary validation", ["Leave-one-year-out (LOYO)", "Forward expanding-window (earlier years only)"], horizontal=True, key="tr_strat",
                     help="Both are always run; this chooses which one is shown first.")
    with st.expander("Thresholds (configurable; defaults and their basis are in acts_transfer/config.py)"):
        t1, t2, t3, t4 = st.columns(4)
        alpha = t1.number_input("Significance level (after BH correction)", 0.001, 0.2, 0.05, 0.005, key="tr_alpha")
        smd_m = t2.number_input("SMD 'medium' from", 0.1, 1.5, 0.5, 0.1, key="tr_smdm")
        smd_l = t3.number_input("SMD 'large' from", 0.2, 2.5, 0.8, 0.1, key="tr_smdl")
        perms = t4.number_input("Permutations (MMD, CORAL)", 100, 2000, 300, 100, key="tr_perm")
        gate_label = st.radio("Negative-transfer safeguard (deployment rule)", ["Delta_NT < 0 (methodology E2)", "Delta_NT < 0 AND statistical support (stricter)"], horizontal=True, key="tr_gate",
                              help="Delta_NT = RMSE of the transfer model minus the smaller of the target-only Ridge and persistence RMSE. The stricter rule also needs the bootstrap interval "
                                   "to exclude zero and the result to hold under leave-one-year-out.")
        nt_gate = "delta_nt" if gate_label.startswith("Delta_NT < 0 (") else "delta_nt+stats"
        t5, t6, t7 = st.columns(3)
        mpf = t5.number_input("Labelled target rows wanted per feature", 1.0, 10.0, 3.0, 0.5, key="tr_mpf")
        qmin = t6.number_input("Flag data-quality column below (%)", 0.0, 100.0, 20.0, 5.0, key="tr_qmin")
        mwarn = t7.number_input("Missing-value warning above (%)", 1.0, 50.0, 10.0, 1.0, key="tr_mwarn")
    if st.button("Confirm domains and load data", key="tr_confirm", disabled=tgt in src_names):
        cfg = Config(deployable=DEPLOYABLE, alpha=alpha, smd_medium=smd_m, smd_large=smd_l, mmd_permutations=int(perms), min_obs_per_feature=mpf, quality_min_pct=qmin,
                     missing_warn_pct=mwarn, max_target_train_groups=None if scarce >= n_seasons - 1 else int(scarce),
                     cv_strategy="loso" if strat.startswith("Leave") else "forward", nt_gate=nt_gate)
        wf.select_domains(_domain(Y, F, src_names, feats_sel), _domain(Y, F, tgt, feats_sel), base, cfg)
        st.session_state.pop("tr_override", None)
    if wf.source is not None:
        st.success(f"Selected: **{wf.source.name}** (source, {len(wf.source.df)} yearly rows) → **{wf.target.name}** (target, {len(wf.target.df)} yearly rows); base model {wf.base}.")
        if wf.target.name != tgt or sorted(wf.source.df.District.unique()) != sorted(src_names):
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
        _show_screening(F, tgt, flagged)
        _show_stewardship(Y, tgt)

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
        st.markdown("**Fold-safe harmonisation (fitted strictly inside each training fold)**")
        h1, h2, h3 = st.columns(3)
        spec.scaler = "robust" if h1.radio("Scaler", ["RobustScaler (median/IQR)", "StandardScaler"], key="tr_scaler", horizontal=False).startswith("Robust") else "standard"
        spec.select_features = h2.checkbox("VIF / correlation feature selection", value=True, key="tr_sel", help="Drops collinear features (VIF above 10, or |r| above 0.95), then the least label-related, until at most p remain.")
        spec.max_features = int(h3.number_input("Maximum features p", 2, 12, 6, 1, key="tr_pmax"))
        st.caption("Also applied inside the folds by the methods that use them: median imputation, within-granary Z-score (anomalies; Z-Score Anomaly Transfer, Granary-Intercept offsets) and CORAL "
                   "covariance alignment. Their effect on the source-target distance is shown below after harmonisation (a label-free diagnostic, never used to fit).")
        _stage_inspector(wf)
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
            sp = hm.shift_preview(b, wf.cfg)
            if len(sp):
                st.markdown("**Distance between source and target under each harmonisation** (label-free; smaller is closer)")
                st.dataframe(sp.round(4), hide_index=True, width="stretch")
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
                with st.spinner("Evaluating whether transfer learning is suitable..."):
                    wf.evaluate_suitability()
            except WorkflowError as e:
                st.error(f"Refused: {e}")
        if wf.suit is not None:
            _badge(*SUIT[wf.suit["suitability"]])
            for r in wf.suit["reasons"]:
                st.markdown(f"- {r}")
            st.caption("This verdict comes from distribution evidence and data quantity. It is not a prediction error: errors are measured only when the model is "
                       "trained and evaluated (tab 5).")
            st.success("Data assessment complete. Continue in tab **4 · Transfer Strategy**, where this evidence is turned into a recommended method.")


def _stage_inspector(wf):
    """the C1_*/C2_* growth-stage features of the selected domains, with how many values are missing in each (nothing is filled before the folds)"""
    feats = wf.source.features
    stage = [f for f in feats if f.startswith(("C1_", "C2_"))]
    if not stage:
        return
    with st.expander(f"Growth-stage features (C1 = main season, C2 = off-season): {len(stage)} candidates", expanded=False):
        st.markdown("Built from the ten-day EVI series by Savitzky–Golay smoothing (window 5, order 2): the green-up of each cycle fixes the windows "
                    "**Vegetative 0–40 d**, **Reproductive/Heading 40–70 d** and **Ripening 70–100 d**, so granaries with different planting calendars are compared at the same crop stage. "
                    "A cycle that does not exist in a granary-year is left missing; the harmonisation drops a feature that is missing too often and imputes the rest inside each fold.")
        frames = [wf.source.df.assign(role="source"), wf.target.df.assign(role="target")]
        d = pd.concat(frames)
        miss = d.groupby("role")[stage].apply(lambda g: 100 * g.isna().mean()).T.round(0).rename(columns=lambda c: f"{c} % missing")
        st.dataframe(miss, width="stretch")


def _show_stewardship(Y, tgt):
    """Data Steward checks (A1, A2): one row per granary-year, the spatial descriptors kept, and the optical-quality weights"""
    with st.expander("Data stewardship: consolidation and optical-quality weights (A1, A2)", expanded=False):
        n_g, n_y = Y.District.nunique(), Y.Year.nunique()
        dup = int(Y.duplicated(["District", "Year"]).sum())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Granary-year rows", len(Y), help=f"{n_g} granaries × {n_y} years. Pixels are collapsed, never used as rows.")
        c2.metric("Duplicate (District, Year)", dup, help="Must be 0: one analytical row per granary-year prevents pseudo-replication.")
        if "n_pixels" in Y:
            c3.metric("Pixels per granary", f"{int(Y.groupby('District').n_pixels.max().min())} to {int(Y.groupby('District').n_pixels.max().max())}")
        if "Optical_complete_window" in Y:
            c4.metric("Optical-complete rows (2019 to 2024)", int(Y.Optical_complete_window.sum()))
        have = [c for c in ("EVI_pix_mean", "EVI_pix_std", "EVI_pix_p10", "EVI_pix_p90") if c in Y]
        st.caption("Spatial descriptors kept per granary-year (pixel mean, std, 10th and 90th percentile of EVI): " + (", ".join(have) if have else "not in this table") + ".")
        if "Optical_Valid_Fraction" in Y:
            w = Y[["District", "Year", "Optical_Valid_Fraction"]].assign(weight=lambda d: np.maximum(0.2, d.Optical_Valid_Fraction.fillna(0.2)))
            low = w[w.Optical_Valid_Fraction < 0.4].sort_values("Optical_Valid_Fraction")
            st.markdown("**Optical sample weights** w = max(0.2, Optical_Valid_Fraction): rows with under 40% real optical observations (the rest is gap-filled)")
            st.dataframe(low.rename(columns={"Optical_Valid_Fraction": "valid optical fraction"}).round(3), hide_index=True, width="stretch")
            t = w[w.District == tgt]
            if len(t):
                st.caption(f"{tgt}: weights {', '.join(f'{int(r.Year)}: {r.weight:.2f}' for r in t.itertuples())}. The sensitivity check in Tab 4 re-runs the benchmark without the weakest year and without weights.")


def _show_screening(F, tgt, flagged):
    """methodology A3: every candidate granary against the target, before anything is trained"""
    with st.expander("Domain-divergence screen of every candidate source granary (A3)", expanded=False):
        sc = _screen(F, tgt)
        if sc is None or sc.empty:
            st.info("The ten-day table is needed for the screen.")
            return
        st.markdown(f"Each granary is compared with **{tgt}** on the ten-day drivers: a domain classifier (climate AUC), MMD², CORAL distance and the largest KS statistic. "
                    "A granary with climate AUC at or above 0.90 is **flagged** and left out of the compatible pool; the pool can still be run with it (Step 1) to compare "
                    "the 8-source and 9-source pools. The calendar offset is shown for information only: it is circular and unstable for single-season granaries, so it does not flag.")
        show = sc.rename(columns={"AUC_climate": "climate AUC", "shift_days": "calendar offset (days)", "KS_max": "max KS"}).copy()
        show["flagged"] = np.where(show.flagged, "⚠️ flagged", "ok")
        st.dataframe(show.round(3), hide_index=True, width="stretch")
        if flagged:
            st.caption("Flagged: " + ", ".join(flagged) + ". Values come from this project's extracted drivers (climate = temperature, rain, solar); a different feature set gives a somewhat different AUC.")


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


# ------------------------------------------------------------------------------------------------ the six-stage flow and the downstream tabs
ICON2 = {"complete": "✅", "ready": "▶️", "locked": "🔒", "blocked": "⛔", "partial": "🟡"}
TAB_OF = {"assessment": 3, "strategy": 4, "training": 5, "prediction": 6, "outlook": 7, "advisory": 8}
BASE_SET = BASELINE_KEYS
PRIMARY_METHODS = ("detrended_residual_transfer", "granary_intercept_ridge", "kmm_ridge", "two_stage_tradaboost_r2", "s_tradaboost_r2")
DEPLOYABLE = ("detrended_residual_transfer",)      # PACE V8: only the Detrended Residual Transfer hybrid can be approved; the others are benchmarks


def flow_bar(wf, current):
    """the same status strip on tabs 3 to 8: where you are, what is done, what is locked and why, and what a change upstream cleared"""
    stages = wf.stages()
    done = sum(s["status"] == "complete" for s in stages)
    st.progress(done / len(stages), text=f"Workflow: {done} of {len(stages)} stages complete · Data Assessment → Transfer Strategy → Train and Evaluate → "
                                         "Yield Prediction → Future Yield Outlook → Intervention Advisory")
    for c, s in zip(st.columns(len(stages)), stages):
        b = "**" if s["key"] == current else ""
        c.markdown(f"{ICON2[s['status']]} {b}Tab {TAB_OF[s['key']]}: {s['label']}{b}<br><span style='font-size:0.78em'>{s['detail']}</span>", unsafe_allow_html=True)
    key = f"seen_notices_{current}"
    for n in wf.notices[st.session_state.get(key, 0):]:
        st.info("🔄 " + n)
    st.session_state[key] = len(wf.notices)


def _locked(msg, hint):
    st.warning(f"🔒 Locked. {msg}")
    st.markdown(hint)


def _lbl(k):
    return LABELS.get(k, k)


def _scenario_cards(s):
    for sc in s["scenarios"]:
        with st.expander(f"{'🟢' if sc['met'] else '⚪'} Scenario {sc['id']}: {sc['name']}  ({'conditions met' if sc['met'] else 'not met'})"):
            st.markdown(f"**Observed conditions:** {sc['observed']}")
            st.markdown(f"**Evidence from the assessment:** {sc['evidence']}")
            st.markdown(f"**Recommended method(s):** {', '.join(_lbl(m) for m in sc['methods']) or 'none'}")
            st.markdown(f"**Why suitable:** {sc['why']}")
            st.markdown(f"**Assumptions:** {sc['assumptions']}")
            st.markdown(f"**Benefits and risks:** {sc['benefits_risks']}")
            st.markdown(f"**Alternatives:** {sc['alternatives']}")
            st.markdown("**Can this system run it?** " + (("yes: " + ", ".join(sc["executable"])) if sc["executable"] else "no eligible method")
                        + (f"  (not eligible: {sc['not_executable']})" if sc["not_executable"] else ""))


# ------------------------------------------------------------------------------------------------ tab 4 · Transfer Strategy
def render_strategy(D):
    st.subheader("Transfer Strategy")
    wf = _wf()
    flow_bar(wf, "strategy")
    ok, why = wf.can_run()
    if not ok:
        _locked(why, "Complete the **Data Assessment** (tab 3) first. The strategy is chosen from its evidence, and the backend refuses to continue without it.")
        return
    s = wf.suit
    _badge(*SUIT[s["suitability"]])
    for r in s["reasons"]:
        st.markdown(f"- {r}")
    st.markdown("### How the assessment points to a method")
    st.caption("Each scenario lists its conditions, the evidence from tab 3, the method it recommends and why. Several can apply at once.")
    _scenario_cards(s)
    st.markdown("### Methods this system can run")
    tab = s["methods"]
    st.dataframe(tab[["method", "family", "eligible", "recommended_by", "reason"]].rename(columns={"recommended_by": "recommended for scenario"}), hide_index=True, width="stretch")
    rec = list(s["defaults"])
    if rec:
        st.success("Recommended by the assessment: " + ", ".join(_lbl(m) for m in rec))
    else:
        st.info("The assessment does not recommend a transfer method for this data. **Direct model training** on the target data (target-only) and the baselines are the "
                "recommended route. Transfer methods can still be tried below as a labelled comparison.")
    st.markdown("**Always run for comparison:** persistence (last year's yield), the target-only Ridge, source-only, the naive pooled model, the target mean and the season "
                "trend. A transfer method is judged against these, never on its own.")
    elig = [k for k in tab[tab.eligible].key if k not in BASE_SET]
    prim = [k for k in PRIMARY_METHODS if k in elig]
    with st.container(border=True):
        st.markdown("**Methods of the methodology**")
        st.markdown("- **Detrended Residual Transfer (Hybrid Season Trend + S-TrAdaBoost.R2)**: the primary proposed architecture, and the only method that can be approved for deployment. "
                    "A regularised linear trend per granary carries the yield drift; S-TrAdaBoost.R2, weighted by optical quality, transfers only the residual (stress) anomaly from MADA.\n"
                    "- **Granary-Intercept Ridge** (hierarchical): one shared slope, one partially pooled intercept per granary, so the yield-level gap between source granaries and the target does not distort the slopes.\n"
                    "- **KMM + Weighted Ridge**: Kernel Mean Matching re-weights source granary-years by how well they match the target's climate and canopy (features only, no labels), then a weighted Ridge.\n"
                    "- **Two-Stage TrAdaBoost.R2**: stage 1 can only shrink source weights (chosen on target error), stage 2 boosts target rows only, so boosting cannot overfit 7 target rows.\n"
                    "- **S-TrAdaBoost.R2**: importance-sampled, balanced-weight boosting so a large source cannot swamp a small target.")
        st.caption("Z-Score Anomaly Transfer, CORAL, TCA and the original TrAdaBoost.R2 are also available when eligible." +
                   ("" if len(prim) == len(PRIMARY_METHODS) else f" Not eligible for this data: {', '.join(_lbl(k) for k in PRIMARY_METHODS if k not in prim)} (see the table above for why)."))
        st.button("Add the methodology methods", key="tr_add_primary", disabled=not prim,
                  on_click=lambda: st.session_state.__setitem__("tr_methods", sorted(set(st.session_state.get("tr_methods", [])) | set(prim))))
    chosen = st.multiselect("Transfer methods to try", elig, default=[m for m in rec if m in elig], format_func=_lbl, key="tr_methods",
                            help="Pre-selected only if the assessment recommends them. Nothing is pre-selected otherwise.")
    can_ref = bool(s["reference"])
    ref = st.checkbox("Also run TrAdaBoost.R2 (the existing approach) as a reference for comparison", value=False, disabled=not can_ref, key="tr_ref",
                      help="It is NOT recommended or selected by default: it is only included when you ask, so that it can be compared with the other methods.")
    if not can_ref:
        st.caption("TrAdaBoost.R2 is not available here: " + tab.set_index("key").loc["tradaboost_r2", "reason"])
    for k in chosen + (["tradaboost_r2"] if ref and "tradaboost_r2" not in chosen else []):
        n = tab[tab.key == k].iloc[0]
        with st.expander(f"How {_lbl(k)} uses distance and features"):
            st.markdown(f"- **Features compared:** {n.features_compared}\n- **Scaling:** {n.scaling}\n- **Metric:** {n.metric}\n- **Why suitable:** {n.why_suitable}\n"
                        f"- **How distance acts:** {n.distance_influence}\n- **Limitations:** {n.limitations}")
    st.caption("A smaller feature distance does not guarantee a smaller yield error: distribution similarity and prediction error are different questions.")
    if s["suitability"] == "not_recommended" and (chosen or ref):
        st.error("Transfer is NOT RECOMMENDED for these domains. You may still try a transfer method for research, but you must say why; it is recorded in the report, "
                 "and such a run can never become the model used for predictions.")
        st.text_input("Override reason (required to run a transfer method)", key="tr_override")
    st.markdown("### Review and confirm")
    final = chosen + (["tradaboost_r2"] if ref and "tradaboost_r2" not in chosen else [])
    plan = final or ["target_only"]
    st.markdown("**Plan:** " + ", ".join(_lbl(m) for m in plan) + (" (direct model training; no transfer method)" if not final else "") + " plus the baselines.")
    if st.button("Confirm strategy", key="tr_strategy", type="primary"):
        try:
            wf.select_strategy(plan)
            st.success("Strategy confirmed. Continue in tab **5 · Train and Evaluate Model**.")
        except WorkflowError as e:
            st.error(f"Refused: {e}")
    if wf.strategy is not None:
        st.success(f"Confirmed strategy: {', '.join(_lbl(m) for m in wf.strategy)}. Changing it clears the evaluation, predictions and outlook that depend on it.")


# ------------------------------------------------------------------------------------------------ tab 5 · Train and Evaluate Model
def safeguard_badge(wf, compact=False):
    """the negative-transfer safeguard (methodology E2) as a status: transfer model deployed, or automatic fallback to a local baseline"""
    c = wf.choice
    sg = c.get("safeguard") or {}
    d = sg.get("delta_nt")
    if c["transfer_supported"]:
        txt = f"🟢 Transfer model deployed: {_lbl(c['selected'])}"
    else:
        txt = f"🛡️ Safeguard fallback: {_lbl(c['selected'])}"
    if compact:
        return txt
    a, b, e = st.columns([2, 1, 2])
    a.markdown(f"**{txt}**")
    if d is not None and not np.isnan(d):
        kg = float(getattr(wf.bundle, "y_to_kg", 1.0))
        b.metric("Delta_NT (RMSE)", f"{d * kg:+.0f} kg/ha", help="RMSE of the best transfer method minus the smaller of target-only Ridge and persistence. Below zero: deploy. Zero or above: block transfer.",
                 delta=("transfer better" if d < 0 else "transfer not better"), delta_color="normal" if d < 0 else "inverse")
        e.caption(f"Best transfer method: {_lbl(sg.get('delta_nt_method'))} against {_lbl(sg.get('delta_nt_baseline'))} · rule: "
                  + ("Delta_NT < 0" if sg.get("gate") == "delta_nt" else "Delta_NT < 0 and statistical support"))


def _verdict_banner(wf):
    c = wf.choice
    safeguard_badge(wf)
    if c["transfer_supported"]:
        st.success("✅ **Transfer learning improves performance on this data.** " + c["statement"])
    else:
        st.warning("⚠️ **The evaluation does not show that transfer learning improves performance.** " + c["statement"])
    neg = []
    for strat, r in wf.results.items():
        imp = r.improvement
        for row in imp[(imp.verdict == "worse") & (~imp.method.isin(BASE_SET)) & (imp.baseline.isin(["target_only", c["reference"]]))].itertuples():
            neg.append(f"{_lbl(row.method)} was clearly worse than {_lbl(row.baseline)} ({strat}, {row.relative_change_pct:+.0f}%)")
    if neg:
        st.error("**Negative transfer detected:** " + "; ".join(dict.fromkeys(neg)) + ". See 'Why did the error go up or down?' below for the evidence.")


def render_training(D):
    """Tab 5 has two sub-tabs. Evidence-first transfer qualification: Discover (the 10-granary benchmark sweep) -> Verify (this target) -> Deploy (tabs 6 to 8)."""
    st.subheader("Train and Evaluate Model")
    t_verify, t_discover, t_audit = st.tabs(["Sub-tab 1 · Verify: qualify the target", "Sub-tab 2 · Discover: 10-granary benchmark sweep", "Sub-tab 3 · Train vs test audit"])
    with t_verify:
        _render_verify(D)
    with t_discover:
        _render_discover(D)
    with t_audit:
        _render_audit(D)


def _render_audit(D):
    """PACE V8 E1: the in-sample TRAINING error next to the out-of-fold TEST error of every method, to expose small-sample overfitting, plus two sensitivity checks on optical quality"""
    wf = _wf()
    st.markdown("### Train vs test: how much does each method overfit?")
    if wf.result is None or wf.choice is None:
        st.info("Train and evaluate in sub-tab 1 first.")
        return
    st.markdown("**Train** is the error on the target years a model was fitted on (in-sample). **Test** is the error on the year it never saw (out-of-fold). "
                "A big jump from train to test is overfitting: Target-Only fits its 7 rows almost perfectly and then misses by an order of magnitude more.")
    pick = st.radio("Validation", ["Forward expanding-window (primary)", "Leave-one-year-out (strict past-only persistence)"], horizontal=True, key="au_pick")
    k = "forward" if pick.startswith("Forward") else "loso"
    m = wf.results[k].metrics.copy()
    kg = float(getattr(wf.bundle, "y_to_kg", 1.0))
    tbl = wf.choice["table"].set_index("method")
    m["Delta_NT (kg/ha)"] = m.method.map(lambda x: tbl.loc[x, "Delta_NT (RMSE)"] * kg if x in tbl.index else np.nan)
    m["Delta_NT vs target-only (kg/ha)"] = m.method.map(lambda x: tbl.loc[x, "Delta_NT vs target-only (RMSE)"] * kg if x in tbl.index else np.nan)
    m["Delta_DS vs source-only (kg/ha)"] = m.method.map(lambda x: tbl.loc[x, "Delta_DS vs source-only (RMSE)"] * kg if x in tbl.index else np.nan)
    m["Model"] = m.method.map(_lbl)
    cols = ["Model", "n_test", "train_RMSE_kg_ha", "RMSE_kg_ha", "gap_RMSE_kg_ha", "MAE_kg_ha", "rRMSE_pct", "MAPE_pct", "win_rate_vs_persistence_pct", "Delta_NT (kg/ha)", "Delta_NT vs target-only (kg/ha)",
            "Delta_DS vs source-only (kg/ha)"]
    out = m[[c for c in cols if c in m.columns]].rename(columns={"n_test": "test seasons", "train_RMSE_kg_ha": "Train RMSE (kg/ha)", "RMSE_kg_ha": "Test RMSE (kg/ha)", "gap_RMSE_kg_ha": "Test − train",
                                                                 "MAE_kg_ha": "Test MAE (kg/ha)", "rRMSE_pct": "rRMSE (%)", "MAPE_pct": "MAPE (%)", "win_rate_vs_persistence_pct": "Win rate vs persistence (%)"})
    st.dataframe(out.sort_values("Test RMSE (kg/ha)").round(0), hide_index=True, width="stretch")
    st.caption("Methods that use no fitted model (season trend, persistence) have no training error. Delta_NT is the test-RMSE gap to the hardest local baseline over the seasons they share; Delta_DS is the gap to Source-Only.")
    st.markdown("### Optical-quality sensitivity (A1)")
    st.markdown("Rows with few cloud-free optical observations are down-weighted by **w = max(0.2, Optical_Valid_Fraction)** inside the transfer model. Check that the conclusion survives removing the weakest year, and removing the weights.")
    yrs = sorted(wf.target.df.Year.unique()) if "Year" in wf.target.df else []
    c1, c2 = st.columns([1, 3])
    drop = c1.selectbox("Exclude year", yrs, index=0, key="au_drop") if yrs else None
    if c2.button("Run the sensitivity check", key="au_run") and drop is not None:
        st.session_state["au_res"] = _sensitivity(wf, int(drop))
    res = st.session_state.get("au_res")
    if res is not None:
        st.dataframe(res.round(0), hide_index=True, width="stretch")


def _sensitivity(wf, drop_year):
    import dataclasses
    from acts_transfer import evaluation
    from acts_transfer.methods import REGISTRY
    b = wf.bundle
    meths = [x for x in ("detrended_residual_transfer", "s_tradaboost_r2", "target_only") if x in wf.result.methods and x in REGISTRY]
    cfg = dataclasses.replace(wf.run_cfg, cv_strategy="forward", bootstrap_n=100)
    variants = {"as run (weighted)": b,
                f"excluding {drop_year}": dataclasses.replace(b, source=b.source[b.source.group != drop_year], target=b.target[b.target.group != drop_year]),
                "unweighted (w = 1)": dataclasses.replace(b, source=b.source.assign(w=1.0), target=b.target.assign(w=1.0))}
    rows = []
    for name, bb in variants.items():
        try:
            met = evaluation.evaluate(bb, meths, wf.base, cfg).metrics.set_index("method")
        except Exception as e:
            rows.append({"Variant": name, "Model": f"failed: {e}"})
            continue
        for mk in meths + ["season_trend", "persistence"]:
            if mk in met.index:
                rows.append({"Variant": name, "Model": _lbl(mk), "test seasons": int(met.loc[mk, "n_test"]), "Test RMSE (kg/ha)": met.loc[mk, "RMSE_kg_ha"], "Test MAE (kg/ha)": met.loc[mk, "MAE_kg_ha"]})
    return pd.DataFrame(rows)


def _sweep_load(D):
    """the benchmark-sweep tables written by pipeline/run_benchmark_sweep.py (or the button in the Discover sub-tab), or None"""
    d = D.DATA_DIR
    names = {k: d / f"benchmark_sweep_{k}.csv" for k in ("pairs", "pooled", "granary", "suitability")}
    if not all(p.exists() for p in names.values()):
        return None
    return {k: _read_csv(str(p), p.stat().st_mtime) for k, p in names.items()}


@st.cache_data(show_spinner=False)
def _read_csv(path, mtime):
    return pd.read_csv(path)


SWEEP_SOURCE = "MADA"


def _domain_relevance(D, wf):
    """PACE V8 criterion 1: the source pool must hold no granary the divergence screen flags (an incompatible agro-climatic calendar, KETARA)"""
    got = _data(D)
    if got is None or "District" not in wf.source.df:
        return dict(ok=True, reason="")
    flagged = _screen(got[1], wf.target.name)
    bad = sorted(set(wf.source.df.District.unique()) & set(flagged[flagged.flagged].granary)) if flagged is not None and len(flagged) else []
    return dict(ok=not bad, reason=("the source pool contains " + ", ".join(bad) + ", flagged by the domain-divergence screen (domain AUC at least 0.90)") if bad else "")


def _qualification(wf, sweep, required):
    """the sweep requirement handed to the model choice: only when the source is exactly the sweep source (MADA) and a sweep exists"""
    src = sorted(wf.source.df.District.unique()) if "District" in wf.source.df else []
    if not required or src != [SWEEP_SOURCE]:
        return None
    if sweep is None:                                               # no evidence yet: nothing can be qualified
        return dict(required=True, passed={"loso": {}, "forward": {}}, source=SWEEP_SOURCE)
    p = sweep["pooled"]
    passed = {st_: {r.method: bool(r.passed) for r in p[(p.strategy == st_) & (p.comparison == "Delta_NT")].itertuples()} for st_ in ("loso", "forward")}
    return dict(required=True, passed=passed, source=SWEEP_SOURCE)


def _render_discover(D):
    st.markdown("### Discover: does transfer from MADA help across the granaries, or only here?")
    st.markdown("MADA is **not assumed** to be a good source. The same source is transferred to every other granary as a pseudo-target under identical out-of-fold splits, and the "
                "**paired error differences** are pooled (yield levels in kg/ha are never averaged across granaries). MADA cannot be its own target, so there are 9 target granaries "
                "(about 63 granary-years under leave-one-year-out).")
    st.markdown("- **Delta_NT** = error of the transfer method − error of Target-Only (below 0: positive transfer)\n"
                "- **Delta_DS** = error of the transfer method − error of Source-Only (below 0: adaptation recovers accuracy lost to domain shift)\n"
                "- tests: paired bootstrap 95% interval of the mean difference, and the Wilcoxon signed-rank test. A method **passes the sweep** when the pooled Delta_NT interval is entirely below 0 "
                "and the one-sided Wilcoxon p is below 0.05.")
    sweep = _sweep_load(D)
    c1, c2 = st.columns([1, 3])
    if c1.button("Re-run the sweep", key="sw_run", help="About one minute. Leave-one-year-out and forward validation for 6 transfer methods and 5 baselines on 9 targets."):
        from acts_transfer import Config
        from acts_transfer.sweep import run_sweep
        got = _data(D)
        if got is None:
            c2.error("The granary-year and ten-day tables are needed.")
        else:
            Y, F = got
            with st.spinner("Running the benchmark sweep..."):
                res = run_sweep(Y, F, SWEEP_SOURCE, Config())
            for k in ("pairs", "pooled", "granary", "suitability"):
                res[k].round(4).to_csv(D.DATA_DIR / f"benchmark_sweep_{k}.csv", index=False)
            _read_csv.clear()
            sweep = _sweep_load(D)
            c2.success("Sweep finished and saved.")
    if sweep is None:
        st.info("No sweep results yet. Press 'Re-run the sweep', or run `python pipeline/run_benchmark_sweep.py`.")
        return
    strat = st.radio("Validation", ["Leave-one-year-out (primary for the sweep)", "Forward expanding-window"], horizontal=True, key="sw_strat")
    sk = "loso" if strat.startswith("Leave") else "forward"
    P = sweep["pooled"]
    P = P[P.strategy == sk]
    st.markdown("#### Pooled results across the sweep granaries")
    nt = P[P.comparison == "Delta_NT"].copy()
    nt["method"] = nt.method.map(_lbl)
    nt["pooled Delta_NT (kg/ha)"] = nt.mean_diff_kg_ha
    nt["95% CI (pairs)"] = nt.apply(lambda r: f"{r.ci95_low:+.0f} to {r.ci95_high:+.0f}", axis=1)
    nt["95% CI (granaries)"] = nt.apply(lambda r: f"{r.cluster_ci95_low:+.0f} to {r.cluster_ci95_high:+.0f}", axis=1)
    nt["passes the sweep"] = np.where(nt.passed, "✅ yes", "no")
    st.dataframe(nt[["method", "n_pairs", "n_granaries", "pooled Delta_NT (kg/ha)", "95% CI (pairs)", "95% CI (granaries)", "wilcoxon_p_less", "pct_pairs_transfer_better", "passes the sweep"]]
                 .rename(columns={"wilcoxon_p_less": "Wilcoxon p (one-sided)", "pct_pairs_transfer_better": "% pairs transfer better"}).round(3), hide_index=True, width="stretch")
    st.caption("'Granaries' resamples whole granaries, which respects that the years of one granary are not independent; it is the more cautious interval.")
    cmp_ = st.radio("Also compare with", ["Source-Only (Delta_DS)", "Persistence"], horizontal=True, key="sw_cmp")
    tag = "Delta_DS" if cmp_.startswith("Source") else "vs persistence"
    o = P[P.comparison == tag].copy()
    o["method"] = o.method.map(_lbl)
    st.dataframe(o[["method", "n_pairs", "mean_diff_kg_ha", "ci95_low", "ci95_high", "wilcoxon_p_less", "pct_pairs_transfer_better"]].rename(
        columns={"mean_diff_kg_ha": f"pooled difference (kg/ha)", "ci95_low": "CI low", "ci95_high": "CI high", "wilcoxon_p_less": "Wilcoxon p (one-sided)"}).round(3), hide_index=True, width="stretch")
    G = sweep["granary"]
    G = G[G.strategy == sk]
    methods = sorted(G.method.unique())
    m = st.selectbox("Method for the per-granary view", methods, format_func=_lbl, key="sw_method")
    g = G[G.method == m].sort_values("Delta_NT_vs_target_only")
    st.markdown("#### Per granary (Delta_NT in kg/ha; errors are RMSE over the held-out years)")
    show = g[["granary", "n_years", "RMSE_transfer", "RMSE_target_only", "RMSE_persistence", "RMSE_source_only", "Delta_NT_vs_target_only", "Delta_NT_gate", "Delta_DS", "AUC", "yield_CV_pct"]].rename(columns={
        "Delta_NT_vs_target_only": "Delta_NT vs target-only", "Delta_NT_gate": "Delta_NT (gate: vs min(target-only, persistence))", "yield_CV_pct": "yield CV %"})
    st.dataframe(show.round(1), hide_index=True, width="stretch")
    try:
        import plotly.express as px
        fig = px.bar(g, x="granary", y="Delta_NT_vs_target_only", color="AUC", color_continuous_scale="Viridis", labels={"Delta_NT_vs_target_only": "Delta_NT vs target-only (kg/ha)"})
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch", key="sw_bar")
    except Exception:
        pass
    st.markdown("#### Suitability mapping")
    st.markdown("Spearman rank correlation of each granary's relative Delta_NT with the descriptors of how different MADA is from it. With 9 granaries this describes where positive and negative "
                "transfer occurred; it is **not** a fitted law, and a correlation near the p = 0.05 line should not be read as proof.")
    S = sweep["suitability"]
    S = S[S.method == m]
    if len(S):
        st.dataframe(S.rename(columns={"spearman_rho": "Spearman ρ", "p_value": "p"}).round(2), hide_index=True, width="stretch")
    st.caption("Source: " + SWEEP_SOURCE + " · features: C1_* main-season stage features, Soil_pH, Optical_Valid_Fraction (those with at most 40% missing) · Ridge base model.")


def _qualification_panel(wf, sweep):
    """Verify: the two parts of the evidence-first rule for the best transfer method on THIS target"""
    c = wf.choice
    m = c.get("best_transfer")
    if not m:
        return
    from acts_transfer.sweep import qualification
    key = c.get("primary_key", "forward")
    q = qualification(sweep, wf.target.name, m, key) if sweep is not None else None
    row = c["table"][c["table"].method == m].iloc[0]
    with st.container(border=True):
        st.markdown(f"**Verify: is the source qualified for {wf.target.name}?** (best transfer method: {_lbl(m)})")
        a, b = st.columns(2)
        sweep_ok = bool(row["sweep qualified"]) if c.get("sweep_required") else (q["sweep_passed"] if q else None)
        a.markdown(f"**① Benchmark sweep (Discover):** " + ("✅ passed" if sweep_ok else ("not required for this source" if not c.get("sweep_required") and q is None else "❌ not passed")))
        a.caption(q["sweep_detail"] if q else "No sweep evidence for this source.")
        b.markdown("**② Safeguard gate:** " + ("✅ Delta_NT < 0" if row["0 Delta_NT < 0"] else "❌ Delta_NT ≥ 0"))
        kg = float(getattr(wf.bundle, "y_to_kg", 1.0))
        b.caption(f"Delta_NT = {row['Delta_NT (RMSE)'] * kg:+.0f} kg/ha vs {_lbl(row['Delta_NT baseline'])}; vs target-only {row['Delta_NT vs target-only (RMSE)'] * kg:+.0f} kg/ha; "
                  f"Delta_DS vs source-only {row['Delta_DS vs source-only (RMSE)'] * kg:+.0f} kg/ha (RMSE, {primary_label(c)}).")
        if q and q.get("position"):
            st.caption(f"Position in the sweep (leave-one-year-out, {_lbl(m)}): rank {q['position'][0]} of {q['position'][1]} granaries by Delta_NT vs target-only (1 = most helped by transfer).")
        st.markdown("**Verdict:** " + ("🟢 qualified, the transfer forecast is unlocked for tabs 6 to 8." if bool(row["supported"]) else "🛡️ not qualified, tabs 6 to 8 use the local fallback baseline."))


def primary_label(c):
    return "forward validation" if c.get("primary_key", "forward") == "forward" else "leave-one-year-out"


def _render_verify(D):
    wf = _wf()
    flow_bar(wf, "training")
    if wf.strategy is None:
        _locked("Confirm a strategy in tab 4 first." if wf.suit is not None else "Complete the data assessment (tab 3) and confirm a strategy (tab 4) first.",
                "Training needs a confirmed strategy and a validated, harmonised dataset. The backend refuses to train without them.")
        return
    has_transfer = any(m not in BASE_SET for m in wf.strategy)
    st.markdown("**Strategy to train:** " + ", ".join(_lbl(m) for m in wf.strategy) + " · baselines are trained alongside, on the same held-out seasons.")
    override = None
    if wf.suit["suitability"] == "not_recommended" and has_transfer:
        st.error("Transfer is NOT RECOMMENDED for these domains. Training a transfer method needs the override reason from tab 4; the run is research only.")
        override = st.session_state.get("tr_override")
    sweep = _sweep_load(D)
    wf.sweep_qualification = None
    wf.domain_relevance = _domain_relevance(D, wf)
    if has_transfer:
        only_sweep_src = sorted(wf.source.df.District.unique()) == [SWEEP_SOURCE] if "District" in wf.source.df else False
        if only_sweep_src:
            req = st.checkbox("Also require the 10-granary benchmark sweep (Discover) to pass (optional extra evidence)", value=False, key="tr_req_sweep",
                              help="Evidence-first qualification: MADA is a qualified source for this target only if the pooled sweep passes for the method AND the safeguard (Delta_NT < 0) passes here.")
            wf.sweep_qualification = _qualification(wf, sweep, req)
            if req and sweep is None:
                st.warning("The benchmark sweep has not been run, so no transfer model can be qualified. Run it in the Discover sub-tab.")
        else:
            st.caption("The benchmark sweep covers MADA as the source. For another source or a pooled source there is no sweep evidence, so only the safeguard (Delta_NT < 0) decides.")
    if st.button("Train and evaluate", key="tr_run", type="primary"):
        try:
            with st.spinner("Training every model with leave-one-year-out (LOYO) and forward expanding-window validation (all fitting inside the training folds)..."):
                wf.run(override_reason=override)
        except WorkflowError as e:
            st.error(f"Refused: {e}")
        except Exception as e:
            st.error(f"The run failed: {type(e).__name__}: {e}")
    if wf.result is None:
        st.info("Press **Train and evaluate**. Results appear here, and the yield prediction and outlook tabs unlock once a model has been evaluated.")
        return
    _verdict_banner(wf)
    _qualification_panel(wf, sweep)
    c = wf.choice
    st.markdown(f"**Model that will generate the predictions in tabs 6 to 8:** {_lbl(c['selected'])} "
                f"({'transfer learning' if c['selected'] not in BASE_SET else 'a baseline: transfer learning is not used'}).")
    with st.expander("Why this model was selected (the criteria a transfer method must meet)"):
        st.markdown("Rule E2: **Delta_NT = RMSE(transfer) − min(RMSE target-only Ridge, RMSE persistence)**. Below zero the transfer model is deployed; zero or above, transfer is blocked "
                    "and the best local baseline is used. Criteria 1 to 4 (lower MAE than the best baseline, a bootstrap interval that excludes zero, no worse under leave-one-year-out, "
                    "enough held-out years) are always computed and shown as the **strength of the evidence**; with the stricter rule chosen in tab 3 they must also hold.")
        if len(c["table"]):
            t = c["table"].copy()
            for col in t.columns:
                if t[col].dtype == bool:
                    t[col] = t[col].map({True: "✅", False: "❌"})
            t["method"] = t.method.map(_lbl)
            st.dataframe(t, hide_index=True, width="stretch")
    _show_results(wf)
    with st.expander("Earlier tests: transfer learning on the yield-outlook method, and stored TrAdaBoost results"):
        tab_modelcheck.render(D)


# ------------------------------------------------------------------------------------------------ tab 6 · Yield Prediction
def _need_model(wf, label):
    if wf.choice is None:
        _locked("Train and evaluate a model in tab 5 first.", f"{label} uses the model selected after evaluation. The backend refuses to generate it without an evaluated model.")
        return False
    return True


def _model_banner(wf):
    c = wf.choice
    uses = "ignores the satellite and weather inputs and predicts from the season trend alone" if c["selected"] in ("season_trend", "target_mean", "persistence") else "uses the satellite and weather inputs"
    safeguard_badge(wf)
    (st.success if c["transfer_supported"] else st.info)(
        f"**Model generating these predictions: {_lbl(c['selected'])}.** It {uses}. "
        + ("Transfer learning was supported by the evaluation." if c["transfer_supported"] else
           "Transfer learning was not supported by the evaluation, so any transfer prediction shown is for comparison only and is not claimed to be more accurate."))


def _inputs_widget(wf, key, n_rows, first_season):
    last = int(wf.bundle.target.dropna(subset=["y"]).group.max())
    feats = wf.bundle.features
    seasons = [first_season + i for i in range(n_rows)]
    scn = st.radio("Input assumption", ["Typical season (mean of the observed seasons)", "Same as the last observed season", "Custom values"] + (["Upload future features (CSV)"] if key == "out" else []),
                   key=f"{key}_scn", horizontal=True,
                   help="There is no weather forecast in this project: a prediction from features rests on the assumption chosen here, and the result says which.")
    if scn.startswith("Typical"):
        return predmod.make_inputs(wf.bundle, "typical", seasons), "typical season (mean of the observed seasons)"
    if scn.startswith("Same"):
        return predmod.make_inputs(wf.bundle, "last", seasons), "same as the last observed season"
    if scn.startswith("Upload"):
        up = st.file_uploader("CSV with a 'season' column and these features: " + ", ".join(feats), type="csv", key=f"{key}_upload")
        if up is None:
            st.info("No file uploaded yet: showing the typical-season assumption.")
            return predmod.make_inputs(wf.bundle, "typical", seasons), "typical season (no file uploaded)"
        try:
            return predmod.make_inputs(wf.bundle, "upload", seasons, pd.read_csv(up)), "uploaded future features"
        except WorkflowError as e:
            st.error(f"The file cannot be used: {e}. Showing the typical-season assumption.")
            return predmod.make_inputs(wf.bundle, "typical", seasons), "typical season (the uploaded file was rejected)"
    base = predmod.make_inputs(wf.bundle, "typical", seasons)
    ed = st.data_editor(base.round(4), key=f"{key}_custom", hide_index=True, width="stretch", disabled=["season"])
    try:
        return predmod.make_inputs(wf.bundle, "custom", seasons, ed), "custom values entered by the user"
    except WorkflowError as e:
        st.error(str(e))
        return base, "typical season (custom values rejected)"


def render_prediction(D):
    st.subheader("Yield Prediction")
    wf = _wf()
    flow_bar(wf, "prediction")
    if not _need_model(wf, "The yield prediction"):
        return
    _model_banner(wf)
    c = wf.choice
    prim = "forward" if c["primary"] == "forward" else "loso"
    r = wf.results[prim]
    st.markdown("### Validated predictions: what each model predicted for seasons it had not seen")
    st.caption(f"{'Forward validation (trained on earlier seasons only)' if prim == 'forward' else 'Leave-one-season-out'}. These are out-of-sample predictions, so they can be compared with what really happened.")
    keys = list(dict.fromkeys([c["selected"], c["reference"], "target_only"] + [m for m in wf.result.methods if m not in BASE_SET]))
    keys = [k for k in keys if k in set(r.predictions.method)]
    pv = r.predictions[r.predictions.method.isin(keys)].copy()
    pv["model"] = pv.method.map(_lbl)
    pv["role"] = pv.method.map(lambda k: predmod.role_of(k, c))
    wide = pv.pivot_table(index="group", columns="model", values="y_pred")
    wide.insert(0, "actual yield", pv.drop_duplicates("group").set_index("group").y_true)
    st.dataframe(wide.round(2).reset_index().rename(columns={"group": "season"}), hide_index=True, width="stretch")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wide.index, y=wide["actual yield"], mode="lines+markers", name="actual", line=dict(color="black", width=3)))
    for m in [x for x in wide.columns if x != "actual yield"]:
        fig.add_trace(go.Scatter(x=wide.index, y=wide[m], mode="lines+markers", name=m, line=dict(dash="dot")))
    fig.update_layout(height=330, xaxis_title="season", yaxis_title="yield (t/ha)", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch", key="pr_val")
    st.markdown("### Predict yield for chosen inputs")
    last = int(wf.bundle.target.dropna(subset=["y"]).group.max())
    season = st.number_input("Season to predict", last + 1, last + 10, last + 1, 1, key="pr_season")
    inputs, label = _inputs_widget(wf, "pr", 1, int(season))
    tab = wf.predict_for(inputs, label)
    st.caption(f"Input assumption: **{label}**. Every row names the model that made it. The ± band is the typical (80th percentile) error that model made on held-out seasons.")
    show = tab[["model", "role", "season", "predicted", "band", "low", "high", "band_source"]].rename(columns={"predicted": "predicted yield (t/ha)", "band": "± typical error"})
    st.dataframe(show.round(2), hide_index=True, width="stretch")
    colors = {"SELECTED": "#2f855a", "best baseline": "#2b6cb0", "baseline": "#9aa3ad", "transfer": "#c05621"}
    bar = go.Figure(go.Bar(x=tab.model, y=tab.predicted, error_y=dict(type="data", array=tab.band.fillna(0)),
                           marker_color=[next((v for k, v in colors.items() if k in ro), "#999") for ro in tab.role], text=tab.role, hovertemplate="%{x}: %{y:.2f} t/ha<extra></extra>"))
    bar.update_layout(height=320, yaxis_title="predicted yield (t/ha)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bar, width="stretch", key="pr_bar")
    with st.expander("Reported yield records and map (all granaries)"):
        tab_yield.render_yield(D)


# ------------------------------------------------------------------------------------------------ tab 7 · Future Yield Outlook
def render_outlook(D):
    st.subheader("Future Yield Outlook")
    wf = _wf()
    flow_bar(wf, "outlook")
    if not _need_model(wf, "The future yield outlook"):
        return
    _model_banner(wf)
    last = int(wf.bundle.target.dropna(subset=["y"]).group.max())
    n = st.slider("Years ahead", 1, 4, 3, key="out_n")
    inputs, label = _inputs_widget(wf, "out", n, last + 1)
    tab = wf.outlook_for(inputs, label)
    base = D.outlook[D.outlook.District == wf.target.name].sort_values("Forecast_year")
    base = pd.DataFrame({"season": base.Forecast_year, "baseline": base.Central / 1000, "b_low": base.Low / 1000, "b_high": base.High / 1000})
    wide = tab.pivot_table(index="season", columns="model", values="predicted")
    out = base[base.season.isin(wide.index)].set_index("season").join(wide, how="right")
    out["selected - baseline"] = out[_lbl(wf.choice["selected"])] - out["baseline"]
    st.caption(f"Input assumption: **{label}**. 'Baseline outlook' is the dashboard's carry-over method (tab 7 below), shown for comparison.")
    shown = out.rename(columns={"baseline": "baseline outlook (carry-over)", "b_low": "baseline low", "b_high": "baseline high"})
    st.dataframe(shown.round(2).reset_index(), hide_index=True, width="stretch")
    sel = tab[tab.model_key == wf.choice["selected"]].sort_values("season")
    hist = wf.bundle.target.dropna(subset=["y"]).sort_values("group")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.group, y=hist.y, mode="lines+markers", name="observed", line=dict(color="black", width=3)))
    if len(base):
        fig.add_trace(go.Scatter(x=base.season, y=base.baseline, mode="lines+markers", name="baseline outlook (carry-over)", line=dict(color="#2b6cb0")))
        fig.add_trace(go.Scatter(x=list(base.season) + list(base.season)[::-1], y=list(base.b_high) + list(base.b_low)[::-1], fill="toself", fillcolor="rgba(43,108,176,0.15)",
                                 line=dict(width=0), name="baseline range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sel.season, y=sel.predicted, mode="lines+markers", name=f"selected: {_lbl(wf.choice['selected'])}", line=dict(color="#2f855a", width=3)))
    if sel.band.notna().any():
        fig.add_trace(go.Scatter(x=list(sel.season) + list(sel.season)[::-1], y=list(sel.high) + list(sel.low)[::-1], fill="toself", fillcolor="rgba(47,133,90,0.18)",
                                 line=dict(width=0), name="selected range", hoverinfo="skip"))
    for k, d in tab[(tab.model_key != wf.choice["selected"]) & (~tab.model_key.isin(["target_only"]))].groupby("model_key"):
        if k != wf.choice["reference"]:
            fig.add_trace(go.Scatter(x=d.season, y=d.predicted, mode="lines+markers", name=f"{_lbl(k)} (not selected)", line=dict(dash="dot")))
    fig.update_layout(height=380, xaxis_title="season", yaxis_title="yield (t/ha)", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch", key="out_fig")
    c = wf.choice
    r = wf.results["forward" if c["primary"] == "forward" else "loso"]
    mae = c["mae"].get(c["selected"], float("nan"))
    st.markdown("### Confidence and limitations")
    lines = [f"**Measured accuracy of the selected model** ({_lbl(c['selected'])}): mean absolute error {mae:.2f} t/ha on {r.n_folds} held-out seasons ({c['primary']} validation).",
             f"**Input assumption:** {label}. There is no weather forecast in this project, so a feature-based outlook is only as good as this assumption.",
             "**Range:** the ± band is the typical error measured on held-out seasons" + (", measured at the same number of seasons ahead where that could be tested (see 'band source'); with only 2 to 4 tests per horizon it is indicative, not precise."
                                                                                            if c["selected"] in ("season_trend", "target_mean") else " with KNOWN inputs, so with assumed inputs the true uncertainty is larger.")]
    if c["selected"] == "season_trend":
        lines.append("**Extrapolation:** the selected model extends the straight-line trend of the target's seasons; a trend that held for a few years does not have to continue, and the farther ahead, the less reliable it is.")
    if c["selected"] in ("season_trend", "target_mean"):
        lines.append("**This model does not use the satellite or weather inputs**, so changing the inputs does not change it. The evaluation did not find that the features improve on it.")
    lines.append("Transfer learning " + ("was supported by the evaluation." if c["transfer_supported"] else "was NOT supported by the evaluation, so it is not used for this outlook."))
    lines.append("This outlook is recalculated automatically whenever the strategy, the training or the data change; the notice at the top says what was recalculated.")
    for x in lines:
        st.markdown(f"- {x}")
    with st.expander("Baseline outlook for all granaries (carry-over method)"):
        tab_yield.render_outlook(D)


def advisory_evidence(g):
    """the validated yield prediction for granary g, if this workflow's model is for g; used by the advisory so it never relies on an unvalidated model"""
    wf = st.session_state.get("acts_wf")
    if wf is None or wf.choice is None or wf.target is None or wf.target.name != g:
        return None
    last = int(wf.bundle.target.dropna(subset=["y"]).group.max())
    art = wf.outlook or wf.prediction
    if art is None:
        wf.predict_for(predmod.make_inputs(wf.bundle, "last", [last + 1]), "same as the last observed season")
        art = wf.prediction
    t = art["table"]
    row = t[(t.model_key == wf.choice["selected"])].sort_values("season").iloc[0]
    yl = float(wf.bundle.target.dropna(subset=["y"]).sort_values("group").y.iloc[-1])
    band = f" (typical error ±{row.band:.2f})" if not np.isnan(row.band) else ""
    line = (f"The validated yield model for {g} is {row.model} ({'transfer learning' if row.model_key not in BASE_SET else 'a baseline'}); it predicts {row.predicted:.2f} t/ha for "
            f"{int(row.season)}{band} against {yl:.2f} t/ha last observed. "
            + ("Transfer learning was supported by the evaluation. " if wf.choice["transfer_supported"] else "Transfer learning was not supported by the evaluation, so it is not used. ")
            + "A predicted change is not a realised improvement: actual yield changes only through what is done in the field.")
    return dict(line=line, model=row.model, supported=wf.choice["transfer_supported"], predicted=float(row.predicted), season=int(row.season))


def _show_results(wf):
    results = wf.results
    first = "loso" if wf.cfg.cv_strategy == "loso" else "forward"
    pick = st.radio("Validation shown", ["Leave-one-year-out (LOYO)", "Forward expanding-window (earlier years only)"], index=0 if first == "loso" else 1, horizontal=True, key="tr_valshow")
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
    cols = ["method", "kind", "n_test", "MAE_kg_ha", "RMSE_kg_ha", "rRMSE_pct", "MAPE_pct", "R2", "win_rate_vs_persistence_pct", "RMSE_ci_low", "RMSE_ci_high", "R2_note"]
    show = m[[c for c in cols if c in m.columns]].copy()
    kg = float(getattr(wf.bundle, "y_to_kg", 1.0))
    for ci in ("RMSE_ci_low", "RMSE_ci_high"):
        if ci in show:
            show[ci] = show[ci] * kg
    show = show.rename(columns={"MAE_kg_ha": "MAE (kg/ha)", "RMSE_kg_ha": "RMSE (kg/ha)", "rRMSE_pct": "rRMSE (%)", "MAPE_pct": "MAPE (%)", "win_rate_vs_persistence_pct": "win rate vs persistence (%)",
                                "RMSE_ci_low": "RMSE 95% CI low (kg/ha)", "RMSE_ci_high": "RMSE 95% CI high (kg/ha)"})
    st.dataframe(show.round(3), hide_index=True, width="stretch")
    st.caption("Bootstrap 95% intervals resample the held-out seasons, so with this few seasons they are wide: read them as the honest size of the uncertainty. "
               "R² is shown only with at least 10 held-out points (an indicative value is in the note).")
    fig = go.Figure(go.Bar(x=m.method, y=m.MAE, marker_color=["#9aa3ad" if k == "baseline" else "#2f855a" for k in m.kind], text=m.MAE.round(3), textposition="outside"))
    fig.update_layout(height=320, yaxis_title="MAE (t/ha), lower is better", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, width="stretch", key="tr_bar")
    st.markdown("**Does the answer depend on how it is validated?**")
    sens = pd.DataFrame({("Leave-one-year-out (LOYO)" if k == "loso" else "Forward expanding-window"): v.metrics.set_index("method").MAE for k, v in results.items()}).round(3)
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


# ------------------------------------------------------------------------------------------------ PACE V9 validator pages
def _D(D):
    from .data import load
    return D or load()


def render_domain_audit(D=None):
    """Validator Step 2 · Data & Domain Divergence Audit: select the domains, assess (consolidation, optical weights, MMD2, CORAL, KS, domain AUC, KETARA screen), harmonise, suitability"""
    render_assessment(_D(D))


def render_benchmark_audit(D=None):
    """Validator Step 3 · Transfer Testing & Train/Test Audit: strategy, train and evaluate (Verify, Discover, train-vs-test audit), forecast preview before sign-off"""
    D = _D(D)
    s_strat, s_train, s_prev = st.tabs(["Transfer strategy", "Benchmark, train/test audit and sweep", "Forecast preview (before sign-off)"])
    with s_strat:
        render_strategy(D)
    with s_train:
        render_training(D)
    with s_prev:
        render_prediction(D)
        render_outlook(D)


def render_validator_signoff(D=None):
    """Validator Step 4 · Validator Sign-Off & Assurance Gate"""
    from . import tab_governance
    tab_governance.render_signoff(_D(D))
