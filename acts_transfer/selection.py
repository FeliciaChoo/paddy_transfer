"""Stage 4/5 · Is transfer suitable, and which method fits the observed conditions?

The assessment gives evidence; this module turns it into scenarios A to F with explicit, configurable conditions. A scenario is reported with
its observed conditions, the evidence, the recommended methods, why, assumptions, expected benefits and risks, alternatives and whether this
system can run it. Nothing here evaluates prediction error: that happens only in evaluation.py, on held-out seasons.

Evidence used (all from assessment.py, never from held-out labels):
  shift          joint MMD permutation p < alpha at the modelling level (or, as supporting evidence, at the finer ten-day level)
  feature shift  features with an adjusted KS p < alpha, and features with |SMD| >= smd_large
  covariance     CORAL permutation p < alpha, only when the covariance is estimable (more rows than features)
  label shift    yield KS p < alpha, only when both domains have labels
  labels         labelled target rows available per training fold (the scarcity setting)
  source skill   leave-one-out error of a Ridge model on the SOURCE data against the source mean (uses source labels only)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .assessment import AssessmentReport
from .config import Config
from .harmonise import HarmonisedBundle
from .methods import LABELS, NOT_IMPLEMENTED, REGISTRY, eligibility

# Spec section 7: for every method, what is compared, how it is scaled, which metric, why, how distance acts, limits.
METHOD_NOTES = {
    "source_only": dict(compares="none", scaling="source mean and SD", metric="none", why="a baseline: shows what the source knowledge gives without adaptation",
                        influence="none", limits="assumes the target shares the source's feature distribution and feature-yield relationship"),
    "target_only": dict(compares="none", scaling="target training-row mean and SD", metric="none", why="a baseline: shows what the target data alone can do",
                        influence="none", limits="unstable when the target has few labelled rows"),
    "target_mean": dict(compares="none", scaling="none", metric="none", why="the no-transfer, no-feature baseline: if a model cannot beat the mean of the target's own yields, its features add nothing",
                        influence="none", limits="ignores every input, so it cannot respond to a poor or good season"),
    "pooled": dict(compares="none", scaling="pooled mean and SD of source and target training rows", metric="none",
                   why="a naive transfer baseline: any method that cannot beat it adds nothing", influence="none",
                   limits="treats every source row as equally relevant"),
    "tradaboost_r2": dict(compares="none directly: relevance is judged by prediction error, not feature distance",
                          scaling="pooled mean and SD of source and target training rows",
                          metric="normalised absolute error of the weak learner on each row (uses labels)",
                          why="suits a moderate shift when several labelled target rows exist: it down-weights source rows that fit the target poorly",
                          influence="each of 30 rounds multiplies a source row's weight by beta^loss; the final weights are reported",
                          limits="needs enough labelled target rows to judge the source rows; with few the weights are unstable"),
    "kmm": dict(compares="all retained shared features", scaling="pooled mean and SD of source rows and target training rows",
                metric="MMD with an RBF kernel (median-heuristic bandwidth); KMM minimises that discrepancy",
                why="suits covariate shift with few or no target labels: the weights use features only",
                influence="weights in [0, 10] make the weighted source feature mean match the target's; the effective sample size is reported",
                limits="assumes the feature-yield relationship is shared; weights are unstable where the domains barely overlap"),
    "coral": dict(compares="all retained shared features", scaling="pooled mean and SD, then covariance alignment",
                  metric="covariance difference (CORAL distance) and mean", why="suits differences in how features vary together",
                  influence="no weights: the source features are re-coloured to the target covariance and mean",
                  limits="second-order only; the target covariance is noisy with few rows (a regulariser is added)"),
    "tca": dict(compares="all retained shared features", scaling="pooled mean and SD", metric="MMD in a linear kernel space",
                why="suits a shift that a shared low-dimensional space can remove", influence="learns 4 components that keep variance and cut the MMD",
                limits="needs enough rows; ignores labels when choosing the components"),
    "ridge_prior": dict(compares="none (corrects a source model with target labels)", scaling="source mean and SD", metric="none; penalty by leave-one-out",
                        why="suits a useful source model plus a few target labels: a strongly regularised correction keeps the source coefficients",
                        influence="the penalty chosen by leave-one-out sets how far the correction may move from the source model",
                        limits="Ridge only; assumes the source coefficients are a sensible starting point"),
}


def source_skill(bundle: HarmonisedBundle):
    """leave-one-out MAE of Ridge on the source against predicting the source mean; ratio < 1 means the source model has skill (source labels only)"""
    d = bundle.source.dropna(subset=["y"])
    if len(d) < 4:
        return None
    X = d[bundle.features].fillna(d[bundle.features].median()).values
    y = d.y.values
    e_m, e_b = [], []
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        sc = StandardScaler().fit(X[k])
        m = Ridge(alpha=1.0).fit(sc.transform(X[k]), y[k])
        e_m.append(abs(m.predict(sc.transform(X[i:i + 1]))[0] - y[i]))
        e_b.append(abs(y[k].mean() - y[i]))
    return float(np.mean(e_m) / np.mean(e_b)), len(y)


def n_train_target(bundle, cfg):
    n_groups = bundle.target.dropna(subset=["y"]).group.nunique()
    n = n_groups - 1 if cfg.cv_strategy == "loso" else max(n_groups - 2, 0)
    return min(n, cfg.max_target_train_groups) if cfg.max_target_train_groups else n


def select(report: AssessmentReport, bundle: HarmonisedBundle, cfg: Config, base="ridge"):
    """returns dict(suitability, reasons, scenarios, methods, facts)"""
    dm, df_ = report.diff_model, report.diff_fine
    tab = dm["table"] if dm else pd.DataFrame()
    p = len(bundle.features)
    n_t = n_train_target(bundle, cfg)
    n_s = int(bundle.source.y.notna().sum())
    mmd_sig = bool(dm and dm.get("mmd_p", 1) < cfg.alpha)
    mmd_fine = bool(df_ and df_.get("mmd_p", 1) < cfg.alpha)
    n_ks = int(tab.KS_significant.sum()) if "KS_significant" in tab else 0
    n_large = int((tab.SMD.abs() >= cfg.smd_large).sum()) if "SMD" in tab else 0
    coral_ok = bool(dm and "coral" in dm and dm.get("coral_reliable", False))
    coral_sig = bool(coral_ok and dm["coral_p"] < cfg.alpha)
    coral_fine_sig = bool(df_ and "coral" in df_ and df_.get("coral_reliable", False) and df_["coral_p"] < cfg.alpha)
    lab = report.label
    label_sig = bool(lab and lab["KS_p"] < cfg.alpha)
    label_large = bool(lab and abs(lab["SMD"]) >= cfg.smd_large)
    sk = source_skill(bundle)
    units_ok = all(v["passed"] for v in bundle.validation if v["check"].startswith("units"))
    facts = dict(auc_climate=(report.fine_diag or {}).get("auc_climate"), n_target_train=n_t, n_source=n_s, n_features=p, mmd_sig=mmd_sig, mmd_fine_sig=mmd_fine, n_ks_features=n_ks, n_large_smd=n_large,
                 coral_reliable=coral_ok, coral_sig=coral_sig, label_sig=label_sig, source_skill_ratio=sk[0] if sk else None)
    ev_shift = (f"modelling-level MMD p = {dm['mmd_p']:.3f}" if dm and "mmd_p" in dm else "MMD not computed") + \
               (f"; ten-day-level MMD p = {df_['mmd_p']:.3f}" if df_ and "mmd_p" in df_ else "") + \
               f"; {n_ks}/{len(tab)} features differ (adjusted KS); {n_large} with |SMD| >= {cfg.smd_large:g}"
    ev_cov = (f"modelling-level CORAL p = {dm['coral_p']:.3f}" if coral_ok else "modelling-level covariance not interpretable (fewer rows than features)") + \
             (f"; ten-day-level CORAL p = {df_['coral_p']:.3f}" if df_ and "coral_p" in df_ else "")
    exe = lambda keys: [k for k in keys if eligibility(k, n_s, n_t, p, base, cfg)[0]]  # noqa: E731

    S = []
    A_met = (not mmd_sig) and n_ks == 0 and n_large == 0 and not label_sig and not mmd_fine
    S.append(dict(id="A", name="Similar source and target distributions", met=A_met,
                  observed="No joint, per-feature or label difference was detected." if A_met else "A difference was detected, so this scenario does not apply.",
                  evidence=ev_shift, methods=["source_only", "pooled", "target_only"],
                  why="If the domains look alike, transfer may be unnecessary: compare source-only and pooled models with a target-only model before adding complexity.",
                  assumptions="Absence of a detected difference is not proof of similarity when there are few rows.",
                  benefits_risks="Simplest option; risk is hidden differences that low power cannot detect.", alternatives="TrAdaBoost.R2 or KMM as a check"))
    B_met = (mmd_sig or n_ks > 0 or mmd_fine) and units_ok and not coral_sig and not coral_fine_sig and not label_sig
    S.append(dict(id="B", name="Moderate distribution shift with compatible features", met=B_met,
                  observed="Features differ, units agree, and no covariance or label difference was detected." if B_met else "Not met (covariance or label shift is present, or no shift was found).",
                  evidence=ev_shift + "; " + ev_cov, methods=["kmm", "tradaboost_r2"],
                  why="Instance weighting: KMM weights source rows so their feature mean matches the target (no labels needed); TrAdaBoost.R2 lowers the weight of "
                      "source rows that the model fits badly on target rows (needs labelled target rows).",
                  assumptions="The feature-yield relationship is shared (KMM); enough target labels to judge source rows (TrAdaBoost).",
                  benefits_risks="Keeps useful source rows and drops misleading ones; unstable weights with few rows.", alternatives="CORAL, TCA"))
    C_met = coral_sig or coral_fine_sig or n_ks >= max(2, int(np.ceil(len(tab) / 2))) or n_large >= 2
    S.append(dict(id="C", name="Significant feature-distribution differences", met=bool(C_met),
                  observed="Several features, or the feature covariance, differ clearly." if C_met else "Not met.", evidence=ev_shift + "; " + ev_cov,
                  methods=["coral", "tca"],
                  why="Feature-based: CORAL aligns the source's mean and covariance to the target; TCA finds a shared space that cuts the domain gap.",
                  assumptions="The relationship between the aligned features and yield is shared.",
                  benefits_risks="Can remove a systematic offset; can also remove real signal, and the covariance is noisy with few rows.",
                  alternatives="KMM, TrAdaBoost.R2"))
    D_met = sk is not None and sk[0] < 1.0
    S.append(dict(id="D", name="A useful source model exists", met=bool(D_met),
                  observed=(f"A Ridge model on the source has leave-one-out error {sk[0]:.2f} x that of the source mean ({sk[1]} labelled rows)." if sk else
                            "Too few source labels to judge a source model."),
                  evidence=f"source skill ratio = {sk[0]:.2f}" if sk else "n/a", methods=["ridge_prior"],
                  why="Parameter-based: keep the source Ridge coefficients and learn a strongly regularised correction from the target rows.",
                  assumptions="Ridge base model; the source coefficients are a useful prior. Tree models cannot be fine-tuned this way.",
                  benefits_risks="Needs very few target rows; inherits any source bias.", alternatives="TrAdaBoost.R2"))
    E_met = mmd_sig or mmd_fine
    kind = "supervised" if n_t >= 3 else ("semi-supervised (1 to 2 target labels plus target features)" if n_t >= 1 else "unsupervised (no target labels)")
    S.append(dict(id="E", name="Domain shift requiring adaptation", met=bool(E_met),
                  observed=f"A joint feature difference was detected; target labels available per training fold: {n_t} -> {kind}." if E_met else "No joint shift was detected.",
                  evidence=ev_shift, methods=(["tradaboost_r2", "ridge_prior", "coral"] if n_t >= 3 else ["kmm", "coral", "tca"]),
                  why=f"Adaptation type follows the labels actually available: {kind}.",
                  assumptions="Adaptation helps only if the shift is in the features and not in the feature-yield relationship.",
                  benefits_risks="May reduce the source-target gap; a feature gap does not by itself mean a worse prediction.", alternatives="source-only and target-only baselines"))
    F_conds = []
    if report.readiness == "not_ready":
        F_conds.append("data readiness is NOT READY")
    if label_sig and (mmd_sig or mmd_fine):
        F_conds.append("both the features and the yield distributions differ significantly")
    fd = report.fine_diag or {}
    if fd.get("auc_climate") is not None and fd["auc_climate"] >= cfg.far_auc:
        F_conds.append(f"the climates are very different (domain-classifier AUC {fd['auc_climate']:.2f} >= {cfg.far_auc:g})")
    if n_t < 2:
        F_conds.append(f"only {n_t} labelled target row(s) per training fold")
    if label_large and (mmd_sig or mmd_fine) and n_t < cfg.min_obs_per_feature * p:
        F_conds.append("a large yield shift together with a feature shift and few target labels")
    F_met = bool(F_conds)
    S.append(dict(id="F", name="Severe mismatch or negative-transfer risk", met=F_met,
                  observed=("; ".join(F_conds) if F_met else "None of the severe-mismatch conditions holds."), evidence=ev_shift +
                  (f"; label KS p = {lab['KS_p']:.3f}" if lab else ""), methods=["target_only"] if F_met else [],
                  why="Do not force transfer: collect more target data, improve harmonisation, try a different source, or keep a target-only baseline.",
                  assumptions="-", benefits_risks="Avoids a model that is worse than not transferring.", alternatives="a different source domain"))

    for s in S:
        s["executable"] = exe(s["methods"]) if s["methods"] else []
        s["not_executable"] = {k: eligibility(k, n_s, n_t, p, base, cfg)[1] for k in s["methods"] if k not in s["executable"]}

    # overall suitability
    reasons = []
    if F_met or report.readiness == "not_ready":
        suit = "not_recommended"
        reasons += F_conds or ["data not ready"]
    else:
        enough = n_t >= cfg.min_obs_per_feature * p
        shift_scn = [s["id"] for s in S if s["met"] and s["id"] in "BCE"]
        if shift_scn and report.readiness == "ready" and enough:
            suit = "recommended"
            reasons.append(f"a shift is present (scenario {', '.join(shift_scn)}) and there are enough labelled target rows ({n_t} for {p} features)")
        else:
            suit = "conditionally_recommended"
            if not shift_scn:
                reasons.append("no shift needing adaptation was found: transfer may be unnecessary, so compare with baselines")
            if report.readiness != "ready":
                reasons.append("the data readiness status carries warnings (see the assessment)")
            if not enough:
                reasons.append(f"only {n_t} labelled target rows per training fold for {p} features (rule of thumb: {cfg.min_obs_per_feature:g} per feature), "
                               "so any difference found will be uncertain")
    # method table
    rows = []
    for k in list(REGISTRY) + list(NOT_IMPLEMENTED):
        ok, why = eligibility(k, n_s, n_t, p, base, cfg)
        by = [s["id"] for s in S if k in s["methods"] and s["met"]]
        note = METHOD_NOTES.get(k, {})
        rows.append(dict(key=k, method=LABELS.get(k, k), family=(REGISTRY[k].family if k in REGISTRY else "not implemented"), eligible=ok,
                         reason=why, recommended_by=", ".join(by) if by else ("-" if k in REGISTRY else "not offered"),
                         features_compared=note.get("compares", "-"), scaling=note.get("scaling", "-"), metric=note.get("metric", "-"),
                         why_suitable=note.get("why", "-"), distance_influence=note.get("influence", "-"), limitations=note.get("limits", "-")))
    return dict(suitability=suit, reasons=reasons, scenarios=S, methods=pd.DataFrame(rows), facts=facts, bundle_token=bundle.token, base=base)
