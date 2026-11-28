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
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .assessment import AssessmentReport
from .config import Config
from .evaluation import EvaluationResult, plausible_label_bounds
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


# ================================================================================================================================================
# PRE-APPROVAL DIAGNOSTICS (shown above the Approve / Reject buttons in the Validator sign-off)
#
# Three pure functions, no Streamlit and no files, so the dashboard and pytest call exactly the same code:
#   compute_weather_ablation          how many kg/ha does the real weather / satellite signal save against zeroed anomalies?
#   compute_few_shot_scarcity         local-only vs transfer error when the target granary has only n = 2, 3, 4 seasons of history
#   compute_weather_driver_attribution  which weather / satellite feature moves yield most (kg/ha per +1 SD)
# All three use one transparent learner, AnomalyRidge: within-granary anomalies (each granary's own mean removed), one Ridge slope shared by the
# donors and the target, and the TARGET's own level added back. A feature anomaly of zero therefore means "a normal season for this granary".
# Fold safety: means and standard deviations are computed from the training rows of each fold only; the donor rows are limited to seasons strictly
# BEFORE the season being predicted; the held-out season's label is never read. The attribution is descriptive (it uses every labelled row) and is
# not a forecast.
# ================================================================================================================================================
METEO_KEYWORDS = ("Rain", "Tmax", "Tmin", "Temp", "Solar")      # name fragments of the meteorological stage features (C1_Rain_mm_rep, C1_Tmax_C_rep, C1_Mean_Temp_C_rep, ...); optical and SAR features are not in it
STATIC_COVARIATES = ("Soil_pH", "Optical_Valid_Fraction")      # season-invariant descriptors, not weather or satellite drivers


def weather_columns(bundle, weather_cols=None):
    """the weather / satellite drivers of a bundle: every retained feature except the static covariates (or the explicit list given, restricted to the bundle)"""
    if weather_cols is not None:
        return [c for c in weather_cols if c in bundle.features]
    return [c for c in bundle.features if c not in STATIC_COVARIATES]


def _nanmean0(a):
    """column means that ignore gaps; a column with no value at all gets 0 (its anomaly is then 0)"""
    a = np.asarray(a, float)
    with np.errstate(all="ignore"):
        m = np.nanmean(a, axis=0) if a.ndim > 1 else np.nanmean(a)
    return np.nan_to_num(m, nan=0.0)


class AnomalyRidge:
    """Within-granary-anomaly Ridge. fit(Xs, ys, us, Xt, yt): raw training arrays only (donor rows Xs/ys with their granary ids us, labelled target rows Xt/yt).
    Every granary is centred on its own mean (a missing feature value becomes an anomaly of 0), the pooled anomalies are scaled by their standard deviation,
    and one Ridge slope vector is fitted on all of them. predict() adds the target's own mean yield to the slope times the target's anomaly of the new row."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, Xs, ys, us, Xt, yt, meta=None):
        Xt, yt = np.asarray(Xt, float), np.asarray(yt, float)
        if len(yt) < 1:
            raise ValueError("AnomalyRidge needs at least one labelled target row")
        blocks_x, blocks_y, n_src = [], [], 0
        Xs, ys, us = np.asarray(Xs, float).reshape(-1, Xt.shape[1]), np.asarray(ys, float), np.asarray(us)
        for g in dict.fromkeys(us.tolist()):                      # donor granaries: centre each on its own training mean
            k = (us == g) & ~np.isnan(ys)
            if k.sum() >= 2:
                blocks_x.append(np.nan_to_num(Xs[k] - _nanmean0(Xs[k]), nan=0.0))
                blocks_y.append(ys[k] - ys[k].mean())
                n_src += int(k.sum())
        self.mu_x_, self.mu_y_ = _nanmean0(Xt), float(np.mean(yt))     # the target's level comes from ITS training rows only
        if len(yt) >= 2:
            blocks_x.append(np.nan_to_num(Xt - self.mu_x_, nan=0.0))
            blocks_y.append(yt - self.mu_y_)
        if blocks_x:
            A, B = np.vstack(blocks_x), np.concatenate(blocks_y)
            sd = A.std(axis=0)
            self.sd_ = np.where(sd > 1e-9, sd, 1.0)
            self.coef_ = Ridge(alpha=self.alpha, fit_intercept=False).fit(A / self.sd_, B).coef_
        else:                                                     # a single target row and no usable donor: no slope can be learned
            self.sd_, self.coef_ = np.ones(Xt.shape[1]), np.zeros(Xt.shape[1])
        self.n_train_target_, self.n_train_source_ = len(yt), n_src
        self.train_Xt_, self.train_yt_, self.meta_ = Xt.copy(), yt.copy(), dict(meta or {})
        return self

    def z(self, X):
        return np.nan_to_num((np.asarray(X, float) - self.mu_x_) / self.sd_, nan=0.0)

    def predict(self, X, zero_cols=()):
        """zero_cols: indices of features whose anomaly is set to 0 (ablation = the weather of an average season)"""
        z = self.z(X)
        if len(zero_cols):
            z[:, list(zero_cols)] = 0.0
        return self.mu_y_ + z @ self.coef_


class LocalRidge:
    """Local-only baseline for the few-shot table: standardised Ridge (median imputation, mean/SD) fitted on the target's own n training rows."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, Xt, yt, meta=None):
        Xt, yt = np.asarray(Xt, float), np.asarray(yt, float)
        self.med_ = np.nan_to_num(np.nanmedian(Xt, axis=0) if len(Xt) else np.zeros(Xt.shape[1]), nan=0.0) if Xt.size else np.zeros(Xt.shape[1])
        Xi = np.where(np.isnan(Xt), self.med_, Xt)
        self.sc_ = StandardScaler().fit(Xi)                                   # mean and SD from the n training rows only
        self.m_ = Ridge(alpha=self.alpha).fit(self.sc_.transform(Xi), yt)
        self.mean_, self.scale_ = self.sc_.mean_.copy(), self.sc_.scale_.copy()
        self.train_Xt_, self.train_yt_, self.meta_ = Xt.copy(), yt.copy(), dict(meta or {})
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return self.m_.predict(self.sc_.transform(np.where(np.isnan(X), self.med_, X)))


def _label_tables(bundle):
    """target (sorted by season, labelled rows only) and source tables of a bundle"""
    T = bundle.target.dropna(subset=["y"]).sort_values("group").reset_index(drop=True)
    S = bundle.source.dropna(subset=["y"]).reset_index(drop=True)
    return T, S


def _units(S):
    return S["unit"].astype(str).values if "unit" in S else np.array(["source"] * len(S))


def _bounded(pred, bundle):
    b = plausible_label_bounds(bundle)
    return np.clip(pred, *b) if b is not None else pred


def compute_weather_ablation(bundle, kg, weather_cols=None, min_train=2, alpha=1.0):
    """How much of the forecast accuracy comes from the real weather / satellite signal?
    Expanding-window forward validation on the target granary: for every season g with at least `min_train` earlier target seasons, AnomalyRidge is fitted on
    the earlier seasons only (donor rows strictly before g) and g is predicted twice: with its real weather / satellite features, and with the anomalies of
    those features zeroed (an average season for this granary). Returns kg/ha: err_real and err_ablated (RMSE over the held-out seasons) and
    weather_gain = err_ablated - err_real (positive = the real weather lowers the error). The learner is the same in both runs; only the held-out season's
    weather inputs differ, so no label and no future row is involved."""
    feats = list(bundle.features)
    wc = weather_columns(bundle, weather_cols)
    zero_idx = [feats.index(c) for c in wc]
    T, S = _label_tables(bundle)
    out = dict(err_real=float("nan"), err_ablated=float("nan"), weather_gain=float("nan"), n_folds=0, weather_features=wc, per_season=pd.DataFrame())
    if not wc or len(T) <= min_train:
        return out
    Xs_all, ys_all, us_all, gs_all = S[feats].values, S["y"].values, _units(S), S["group"].values
    rows = []
    for i in range(min_train, len(T)):
        g = T.group.iloc[i]
        tr, te = T.iloc[:i], T.iloc[[i]]
        k = gs_all < g                                                # donor seasons strictly before the season being predicted
        m = AnomalyRidge(alpha).fit(Xs_all[k], ys_all[k], us_all[k], tr[feats].values, tr["y"].values,
                                    meta=dict(test_season=g, max_source_group=(gs_all[k].max() if k.any() else None), max_target_group=tr.group.max()))
        y = float(te["y"].iloc[0])
        p_real = float(_bounded(m.predict(te[feats].values), bundle)[0])
        p_abl = float(_bounded(m.predict(te[feats].values, zero_cols=zero_idx), bundle)[0])
        rows.append(dict(season=int(g), actual=y * kg, real=p_real * kg, ablated=p_abl * kg, err_real=abs(y - p_real) * kg, err_ablated=abs(y - p_abl) * kg))
    per = pd.DataFrame(rows)
    out.update(err_real=float(np.sqrt((per.err_real ** 2).mean())), err_ablated=float(np.sqrt((per.err_ablated ** 2).mean())), n_folds=len(per), per_season=per)
    out["weather_gain"] = out["err_ablated"] - out["err_real"]
    return out


def compute_few_shot_scarcity(bundle, kg, shots=(2, 3, 4), alpha=1.0):
    """Does transfer help when the target granary is young? For every n in `shots`, forecast each held-out season using only the n MOST RECENT earlier target
    seasons (and, for transfer, the donor seasons strictly before it). All n are scored on the SAME held-out seasons (those with at least max(shots) earlier
    seasons), so the rows are comparable. Local-Only = standardised Ridge on the n target rows; Transfer = AnomalyRidge (donor slopes, target level).
    Errors are RMSE in kg/ha; Transfer Advantage = Local-Only - Transfer (positive = transfer is better, negative = it hurts)."""
    cols = ["Target History", "Local-Only Error (kg/ha)", "Transfer Error (kg/ha)", "Transfer Advantage", "Test Seasons"]
    feats = list(bundle.features)
    T, S = _label_tables(bundle)
    ok = [int(n) for n in shots if 1 <= int(n) < len(T)]
    if not ok:
        return pd.DataFrame(columns=cols)
    Xs_all, ys_all, us_all, gs_all = S[feats].values, S["y"].values, _units(S), S["group"].values
    first = max(ok)                                                   # the first season that has `first` earlier seasons
    res = []
    for n in ok:
        e_loc, e_tr = [], []
        for i in range(first, len(T)):
            g = T.group.iloc[i]
            tr, te = T.iloc[i - n:i], T.iloc[[i]]                    # the n most recent seasons before g; g itself is never in the training slice
            meta = dict(test_season=g, n=n, max_target_group=tr.group.max())
            loc = LocalRidge(alpha).fit(tr[feats].values, tr["y"].values, meta=meta)
            k = gs_all < g
            tra = AnomalyRidge(alpha).fit(Xs_all[k], ys_all[k], us_all[k], tr[feats].values, tr["y"].values,
                                          meta=dict(meta, max_source_group=(gs_all[k].max() if k.any() else None)))
            y = float(te["y"].iloc[0])
            e_loc.append(abs(y - float(_bounded(loc.predict(te[feats].values), bundle)[0])) * kg)
            e_tr.append(abs(y - float(_bounded(tra.predict(te[feats].values), bundle)[0])) * kg)
        rl, rt = float(np.sqrt(np.mean(np.square(e_loc)))), float(np.sqrt(np.mean(np.square(e_tr))))
        res.append({"Target History": f"n = {n} seasons", "Local-Only Error (kg/ha)": rl, "Transfer Error (kg/ha)": rt, "Transfer Advantage": rl - rt, "Test Seasons": len(e_loc)})
    return pd.DataFrame(res, columns=cols)


def compute_weather_driver_attribution(bundle, kg, top_k=6, weather_cols=None, alpha=1.0):
    """Which weather / satellite drivers move yield most? One AnomalyRidge is fitted on every labelled row (donors and target) and each driver's standardised
    slope is reported as the yield change for a +1 SD shift of that driver, in kg/ha (positive = more yield). Sorted by the absolute impact, at most top_k rows.
    Descriptive, not causal: correlated drivers share their effect and the slope comes from a handful of seasons."""
    cols = ["Driver", "Impact (kg/ha per +1 SD)", "Abs Impact", "Direction"]
    feats = list(bundle.features)
    wc = weather_columns(bundle, weather_cols)
    T, S = _label_tables(bundle)
    if not wc or len(T) < 2:
        return pd.DataFrame(columns=cols)
    m = AnomalyRidge(alpha).fit(S[feats].values, S["y"].values, _units(S), T[feats].values, T["y"].values)
    imp = {c: float(m.coef_[feats.index(c)] * kg) for c in wc}
    df = pd.DataFrame({"Driver": list(imp), "Impact (kg/ha per +1 SD)": list(imp.values())})
    df["Abs Impact"] = df["Impact (kg/ha per +1 SD)"].abs()
    df["Direction"] = np.where(df["Impact (kg/ha per +1 SD)"] >= 0, "raises yield", "lowers yield")
    return df.sort_values("Abs Impact", ascending=False, kind="mergesort").head(int(top_k)).reset_index(drop=True)[cols]


# ================================================================================================================================================
# GATE EVIDENCE (shown on the Step 4 sign-off card and read by pipeline/gate_evidence_audit.py)
# The default gate is criterion 0: Delta_NT < 0 (a point estimate on the common held-out seasons). Criteria 1 to 4 are computed for every method but only
# decide deployment when cfg.nt_gate = "delta_nt+stats". This function reads them from the finished AutoML payload and says, in words, how strong the evidence is.
# It changes no decision and writes nothing.
# ================================================================================================================================================
def _col(table, prefix):
    return next((c for c in table.columns if str(c).startswith(prefix)), None)


def gate_evidence(auto: dict, method: str | None = None) -> dict:
    """Evidence behind the safeguard gate for `method` (default: the selected transfer method, else the best transfer candidate), in kg/ha.
    Criterion 1 is judged against the best of ALL baselines (source-only and pooled included, not only the local ones); criterion 2 needs at least
    `min_seasons` paired held-out seasons, otherwise it is not testable; criterion 3 is the leave-one-season-out comparison. strength is one of
    'supported', 'not_testable', 'not_distinguishable', 'worse' or 'mixed'; reason explains it."""
    ch = auto["choice"]
    t = ch["table"]
    kg = float(getattr(auto["bundle"], "y_to_kg", 1.0)) or 1.0
    if method is None:
        sel = ch.get("selected")
        method = sel if ch.get("transfer_supported") and sel in set(t["method"]) else ch.get("best_transfer")
    row = t[t["method"] == method]
    if not len(row):
        return {"available": False}
    r = row.iloc[0]
    ref_col = next((c for c in t.columns if str(c).startswith("best baseline (")), None)
    ref = re.search(r"\((.+)\)", ref_col).group(1) if ref_col else None
    c4col = _col(t, "4 at least")
    min_seasons = int(re.search(r"at least (\d+)", c4col).group(1)) if c4col else 5
    imp = getattr(auto.get("eval_results", {}).get("forward"), "improvement", None)
    pair = imp[(imp["method"] == method) & (imp["baseline"] == ref)] if imp is not None and len(imp) else None
    n_pairs = int(pair["n_seasons"].iloc[0]) if pair is not None and len(pair) else 0
    verdict = str(pair["verdict"].iloc[0]) if pair is not None and len(pair) else "n/a"
    chg = lo = hi = float("nan")
    if pair is not None and len(pair):
        chg, lo, hi = (float(pair[c].iloc[0]) * kg for c in ("mean_abs_err_change", "ci95_low", "ci95_high"))
    ev = dict(available=True, method=method, best_baseline=ref, delta_nt_kg_ha=float(r["Delta_NT (RMSE)"]) * kg, delta_nt_baseline=str(r["Delta_NT baseline"]),
              c0_delta_nt_negative=bool(r["0 Delta_NT < 0"]), gate_passed=bool(r["gate passed"]), domain_ok=bool(r["Criterion 1 domain relevance"]),
              c1_lower_mae=bool(r[_col(t, "1 lower MAE")]), c2_interval_excludes_zero=bool(r[_col(t, "2 interval")]), c3_loso_win=bool(r[_col(t, "3 also")]),
              c4_enough_seasons=bool(r[c4col]) if c4col else False, held_out_seasons=n_pairs, common_seasons=int(r["seasons compared"]), min_seasons=min_seasons,
              mae_change_kg_ha=chg, ci95_low_kg_ha=lo, ci95_high_kg_ha=hi, verdict=verdict, stats_ok=bool(r["statistically supported"]))
    sg = ch.get("safeguard", {})                                           # the nested selection check (method chosen on earlier seasons, judged on later ones)
    nd = sg.get("nested_delta_nt")
    ev.update(nested_available=nd is not None, nested_delta_nt_kg_ha=float(nd) * kg if nd is not None else float("nan"), nested_method=sg.get("nested_method"),
              nested_ok=bool(sg.get("nested_ok", False)), nested_select_seasons=int(sg.get("nested_select_seasons", 0)), nested_hold_seasons=int(sg.get("nested_hold_seasons", 0)),
              nested_gate=bool(sg.get("nested_gate", False)))
    failing = [n for n, ok in (("lower MAE than the best baseline", ev["c1_lower_mae"]), ("leave-one-season-out win", ev["c3_loso_win"])) if not ok]
    if ev["stats_ok"]:
        ev["strength"], ev["reason"] = "supported", "the improvement holds on MAE and leave-one-season-out, and its 95% interval excludes zero."
    elif n_pairs < min_seasons:
        ev["strength"], ev["reason"] = "not_testable", f"only {n_pairs} paired held-out seasons; at least {min_seasons} are needed before the interval can be tested."
    elif verdict == "worse":
        ev["strength"], ev["reason"] = "worse", "the interval shows it is worse than the best baseline."
    elif not ev["c2_interval_excludes_zero"]:
        ev["strength"], ev["reason"] = "not_distinguishable", f"the 95% interval ({lo:+,.0f} to {hi:+,.0f} kg/ha) includes zero, so it cannot be told apart from chance."
    else:
        ev["strength"], ev["reason"] = "mixed", "it fails: " + ", ".join(failing) + "."
    return ev
