"""Stage 7 · Why did a method do worse (or better) than its baseline? Evidence, not stories.

Every row separates three things:
  Measured          a number computed from this run or from the assessment
  Status            Verified finding      the measured condition is true in this run (it does NOT prove it caused the error)
                    Possible explanation  a mechanism consistent with the measurements that the data cannot prove
                    Not supported         the measurements point against this explanation
                    Not applicable        the method has no such mechanism
  Limitation        why the evidence is weaker than it looks
Distribution differences found in the assessment are NOT prediction errors, and a decline in performance is never attributed to domain shift
automatically: the shift is one hypothesis among several.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .assessment import AssessmentReport
from .config import Config
from .evaluation import EvaluationResult
from .harmonise import HarmonisedBundle

VERIFIED, POSSIBLE, NOT_SUPPORTED, NA = "Verified finding", "Possible explanation", "Not supported", "Not applicable"


def _coef(df, feats):
    d = df.dropna(subset=["y"])
    X = d[feats].fillna(d[feats].median()).values
    sc = StandardScaler().fit(X)
    return Ridge(alpha=1.0).fit(sc.transform(X), d.y.values).coef_


def diagnose(bundle: HarmonisedBundle, report: AssessmentReport, res: EvaluationResult, method: str, baseline: str, cfg: Config):
    pred = res.predictions
    wide = pred.pivot_table(index="group", columns="method", values="abs_err")
    if method not in wide or baseline not in wide:
        return pd.DataFrame(), f"No predictions for {method} or {baseline}."
    d = (wide[method] - wide[baseline]).dropna()
    worse = d.mean() > 0
    fi = pd.DataFrame([r for r in res.fold_info if r["method"] == method])
    p = len(bundle.features)
    dm, df_ = report.diff_model, report.diff_fine
    rows = []

    def add(h, measured, status, limit, action):
        rows.append({"Hypothesis": h, "Measured": measured, "Status": status, "Limitation": limit, "Recommended action": action})

    # 1. distribution difference
    tab = dm["table"] if dm else pd.DataFrame()
    n_ks = int(tab.KS_significant.sum()) if "KS_significant" in tab else 0
    mmd_p = dm.get("mmd_p") if dm else None
    sig = bool((mmd_p is not None and mmd_p < cfg.alpha) or n_ks)
    add("Large source-target distribution difference",
        f"modelling-level MMD p = {mmd_p:.3f}; {n_ks}/{len(tab)} features differ (adjusted KS)" + (f"; ten-day-level MMD p = {df_['mmd_p']:.3f}" if df_ and 'mmd_p' in df_ else ""),
        (POSSIBLE if sig else NOT_SUPPORTED),
        "A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power.",
        "Align or weight the features (KMM, CORAL) and compare with the target-only baseline." if sig else "Look at the other explanations first.")
    # 2. weights
    if method in ("kmm", "tradaboost_r2") and "ess_ratio" in fi:
        ess = float(fi.ess_ratio.mean())
        extra = f"; source weight share {fi.source_weight_share.mean():.2f}" if "source_weight_share" in fi else ""
        add("Irrelevant or incorrectly weighted source rows", f"effective sample size = {ess:.2f} of the source rows{extra}",
            (VERIFIED if ess < 0.5 else NOT_SUPPORTED),
            "The weights show where the method put its trust, not whether that trust was right.",
            "Cap the weights, lower the boosting rounds, or fall back to the pooled model." if ess < 0.5 else "The weights are spread, so concentration is not the problem.")
    else:
        add("Irrelevant or incorrectly weighted source rows", "this method does not weight source rows", NA, "-", "-")
    # 3. target observations
    n_t = float(fi.n_target_train.mean()) if len(fi) else np.nan
    low = n_t < cfg.min_obs_per_feature * p
    add("Too few target observations", f"{n_t:.1f} labelled target rows per training fold for {p} features (configured: {cfg.min_obs_per_feature:g} per feature)",
        VERIFIED if low else NOT_SUPPORTED, "That the condition holds does not prove it caused the larger error.",
        "Collect more labelled target seasons or reduce the number of features." if low else "-")
    # 4. different relationship
    try:
        cs, ct = _coef(bundle.source, bundle.features), _coef(bundle.target, bundle.features)
        cos = float(cs @ ct / (np.linalg.norm(cs) * np.linalg.norm(ct)))
        agree = float(np.mean(np.sign(cs) == np.sign(ct)))
        add("Different feature-yield relationship", f"cosine of standardised Ridge coefficients (source vs target) = {cos:+.2f}; same sign for {agree:.0%} of features",
            POSSIBLE if cos < 0.3 else NOT_SUPPORTED,
            f"The target coefficients come from only {int(bundle.target.y.notna().sum())} rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method.",
            "Prefer parameter-based correction or target-only learning if the relationships differ." if cos < 0.3 else "-")
    except Exception as e:
        add("Different feature-yield relationship", f"could not be computed ({type(e).__name__})", NA, "-", "-")
    # 5. mismatch / units
    mism = int((report.compat.status == "shared, units differ").sum())
    nonshared = int(report.compat.status.isin(["source only", "target only"]).sum())
    uok = all(v["passed"] for v in bundle.validation if v["check"].startswith("units"))
    add("Feature mismatch or inconsistent units", f"{mism} shared feature(s) with differing units; {nonshared} not shared; unit check after harmonisation: {'passed' if uok else 'FAILED'}",
        VERIFIED if (mism or nonshared or not uok) else NOT_SUPPORTED, "Declared units can still hide a different definition (for example a different aggregation window).",
        "Resolve the mismatch in the Data Harmonisation stage." if (mism or not uok) else "-")
    # 6. seasonal variability
    pos = d.clip(lower=0)
    share = float(pos.max() / pos.sum()) if pos.sum() > 0 else 0.0
    top = d.idxmax()
    add("Seasonal variability: one season drives the difference", f"worst season {top} accounts for {share:.0%} of the extra error; per-season change range {d.min():+.3f} to {d.max():+.3f}",
        POSSIBLE if (share > 0.5 and len(d) >= 4 and worse) else NOT_SUPPORTED,
        "With few seasons one unusual season dominates any average.", "Check season " + str(top) + " (weather, planting date, data quality)." if share > 0.5 else "-")
    # 7. overfitting
    if method != "source_only" and "train_mae_target" in fi:
        tr = float(fi.train_mae_target.mean())
        te = float(pred[pred.method == method].abs_err.mean())
        ratio = tr / te if te else np.nan
        add("Overfitting", f"mean absolute error on the target training rows = {tr:.3f} vs {te:.3f} on held-out seasons (ratio {ratio:.2f})",
            POSSIBLE if ratio < 0.5 else NOT_SUPPORTED, "A gap is normal for any model; a large one signals memorising few rows.",
            "Increase regularisation or use fewer features." if ratio < 0.5 else "-")
    else:
        add("Overfitting", "the source-only model uses no target training rows", NA, "-", "-")
    # 8. unstable evaluation
    n_f = len(d)
    ci = res.improvement[(res.improvement.method == method) & (res.improvement.baseline == baseline)]
    ci_txt = (f"; 95% range of the change [{ci.ci95_low.iloc[0]:+.3f}, {ci.ci95_high.iloc[0]:+.3f}]" if len(ci) and not np.isnan(ci.ci95_low.iloc[0]) else "; no interval (too few seasons)")
    add("Unstable evaluation from a very small target sample", f"{n_f} held-out seasons{ci_txt}", VERIFIED if n_f < cfg.r2_min_test_n else NOT_SUPPORTED,
        "With this few held-out points a difference of the size seen can arise by chance.", "Treat the comparison as indicative only; add seasons before drawing conclusions." if n_f < cfg.r2_min_test_n else "-")
    # 8b. time trend
    T = bundle.target.dropna(subset=["y"])
    from scipy.stats import spearmanr
    try:
        rho = float(spearmanr(T.group, T.y)[0])
        st_ = res.metrics.set_index("method").MAE.get("season_trend", np.nan)
        me = res.metrics.set_index("method").MAE.get(method, np.nan)
        strong = abs(rho) >= cfg.trend_flag_rho
        add("The features may only proxy a time trend", f"Spearman(season, target yield) = {rho:+.2f}; MAE of this method {me:.3f} vs the pure season-trend baseline {st_:.3f}",
            POSSIBLE if (strong and not np.isnan(st_) and me >= 0.8 * st_) else (VERIFIED if strong else NOT_SUPPORTED),
            "A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not.",
            "Judge the method by forward validation and against the season-trend baseline." if strong else "-")
    except Exception:
        pass
    # 9. method assumptions
    lab_sig = bool(report.label and report.label["KS_p"] < cfg.alpha)
    if method == "kmm":
        add("Inappropriate model assumption", f"KMM assumes only the features shift; yield distribution KS p = {report.label['KS_p']:.3f}" if report.label else "no label comparison",
            VERIFIED if lab_sig else NOT_SUPPORTED, "A non-significant test does not prove the label relationship is shared.", "Use target labels (TrAdaBoost.R2, parameter transfer)." if lab_sig else "-")
    elif method == "tradaboost_r2":
        add("Inappropriate model assumption", f"needs about {cfg.min_target_tradaboost}+ target rows to judge source rows; has {n_t:.1f} per fold",
            VERIFIED if n_t < cfg.min_target_tradaboost + 2 else NOT_SUPPORTED, "The rule is a minimum for the algorithm to run, not a guarantee of stability.", "Collect more target rows." if n_t < cfg.min_target_tradaboost + 2 else "-")
    elif method == "coral":
        add("Inappropriate model assumption", f"covariance estimated from {n_t:.1f} target rows for {p} features", VERIFIED if n_t < p + 1 else NOT_SUPPORTED,
            "The regulariser keeps it defined but pulls the alignment toward the identity.", "Use fewer features or stronger regularisation." if n_t < p + 1 else "-")
    else:
        add("Inappropriate model assumption", "no specific assumption was violated that the data can check", NA, "-", "-")
    out = pd.DataFrame(rows)
    verified = out[out.Status == VERIFIED]["Hypothesis"].tolist()
    possible = out[out.Status == POSSIBLE]["Hypothesis"].tolist()
    head = "worse than" if worse else "better than"
    summary = (f"{method} was {head} {baseline} on average ({d.mean():+.3f}, {int((d < 0).sum())} of {n_f} seasons better). "
               f"Verified findings: {', '.join(verified) or 'none'}. Possible explanations (not proven): {', '.join(possible) or 'none'}.")
    return out, summary
