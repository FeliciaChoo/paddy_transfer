"""Stage 2 · Assess data (before any transfer happens). Uses real data only; every number is computed here.

Metric guide (what each metric is for; they are NOT interchangeable and are never summed into one score):
  SMD   standardised mean difference (Cohen's d, pooled SD). One feature at a time. Unitless, so features can be compared. Sensitive to the
        mean only; says nothing about shape or spread. With few rows its sampling error is large.
  KS    Kolmogorov-Smirnov statistic and p-value. One feature at a time, any shape difference (location, spread, shape). Needs enough rows to
        have power; p-values are Benjamini-Hochberg adjusted across features.
  W1    Wasserstein-1 distance. One feature at a time, in the feature's own units (and divided by the pooled SD for a unitless version): the
        average amount a value must move to turn one distribution into the other.
  MMD   maximum mean discrepancy (unbiased, RBF kernel, median-heuristic bandwidth) on all shared features together after pooled
        standardisation. Detects joint differences that single-feature tests miss. Its p-value comes from a permutation test of the domain labels.
  CORAL distance  squared Frobenius norm of the covariance difference, divided by 4 d^2 (Sun et al. 2016), on pooled-standardised features. It
        compares how features move TOGETHER, not their means. Permutation p-value as for MMD.
  Label shift  the same SMD / KS / W1 on the yield itself, only where both domains have labels. It is a difference in the label distribution,
        which transfer methods that assume a shared label relationship are sensitive to.
A small distance does not guarantee a small prediction error, and a large one does not guarantee a large one: these measure similarity of
distributions, the evaluation stage measures prediction error.
"""
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

from ._util import md_table
from .errors import WorkflowError
from .config import Config, Domain

METRIC_GUIDE = {
    "SMD": "Mean difference in pooled-SD units for ONE feature. Unitless. Mean only; unstable with few rows.",
    "KS": "Largest gap between the two cumulative distributions for ONE feature; detects shape and spread. Needs rows to have power; p adjusted (BH).",
    "Wasserstein": "Average movement needed to turn one distribution into the other for ONE feature, in its own units (and per pooled SD).",
    "MMD": "Joint difference over ALL shared features (RBF kernel, median bandwidth, standardised). Permutation p-value.",
    "CORAL": "Difference in covariance (how features move together), standardised. Permutation p-value.",
    "Label shift": "SMD / KS / Wasserstein on the yield itself (needs labels in both domains).",
    "Domain AUC": "Cross-validated AUC of a classifier that tries to tell source rows from target rows (climate features; vegetation/radar features). 0.5 = indistinguishable, "
                  "near 1 = very different. An effect size, so unlike a permutation p-value it does not saturate when n is large. Needs many rows.",
    "Calendar offset": "Shift in days that best aligns the two mean seasonal NDVI curves (circular). Informational: models that use annual aggregates or phase features are "
                       "much less sensitive to it than models that use calendar dates.",
}


class AssessmentError(WorkflowError):
    pass


def token_of(source: Domain, target: Domain, cfg: Config):
    """a fingerprint of the exact data and configuration an assessment was computed on"""
    h = hashlib.sha256()
    for d in (source, target):
        h.update(d.name.encode())
        h.update(pd.util.hash_pandas_object(d.df[[d.group, d.target] + list(d.features)], index=False).values.tobytes())
        h.update(json.dumps([d.features, d.target, d.group, d.units], sort_keys=True, default=str).encode())
    h.update(json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ A. data quality
def quality(d: Domain, cfg: Config):
    df = d.df
    issues = []
    n = len(df)
    groups = sorted(df[d.group].dropna().unique().tolist())
    if n < cfg.min_obs_block:
        issues.append(("block", f"only {n} observations (at least {cfg.min_obs_block} are needed to cross-validate)"))
    rows = []
    for f in d.features:
        s = pd.to_numeric(df[f], errors="coerce")
        miss = float(s.isna().mean() * 100)
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        out = int(((s < q1 - cfg.outlier_iqr_k * iqr) | (s > q3 + cfg.outlier_iqr_k * iqr)).sum()) if iqr > 0 else 0
        rng = cfg.range_for(f)
        inv = int(((s < rng[0]) | (s > rng[1])).sum()) if rng else None
        rows.append(dict(feature=f, dtype=str(df[f].dtype), unit=d.units.get(f, "not declared"), missing_pct=round(miss, 1), outliers=out,
                         invalid=inv if inv is not None else "no range", min=s.min(), max=s.max(), mean=s.mean(), sd=s.std()))
        if miss > cfg.missing_block_pct:
            issues.append(("block", f"{f}: {miss:.0f}% missing (blocking above {cfg.missing_block_pct:.0f}%)"))
        elif miss > cfg.missing_warn_pct:
            issues.append(("warn", f"{f}: {miss:.0f}% missing (warning above {cfg.missing_warn_pct:.0f}%)"))
        if inv:
            issues.append(("warn", f"{f}: {inv} value(s) outside the plausible range {rng}; they will be set to missing, not kept"))
        if out:
            issues.append(("info", f"{f}: {out} extreme outlier(s) beyond {cfg.outlier_iqr_k} x IQR (kept; check them)"))
    y = pd.to_numeric(df[d.target], errors="coerce")
    n_lab = int(y.notna().sum())
    if n_lab < cfg.min_obs_block:
        issues.append(("block", f"only {n_lab} labelled rows in '{d.target}'"))
    ry = cfg.range_for(d.target)
    if ry and ((y < ry[0]) | (y > ry[1])).any():
        issues.append(("warn", f"'{d.target}': values outside the plausible range {ry}"))
    dup = int(df.duplicated().sum())
    dup_g = int(df.duplicated(subset=[d.group]).sum())
    if dup:
        issues.append(("warn", f"{dup} duplicated row(s)"))
    if dup_g:
        issues.append(("warn", f"{dup_g} repeated value(s) of '{d.group}' (expected one row per {d.group})"))
    gaps = []
    if groups and all(isinstance(g, (int, np.integer)) for g in groups):
        gaps = sorted(set(range(min(groups), max(groups) + 1)) - set(groups))
        if gaps:
            issues.append(("warn", f"missing {d.group}(s): {gaps}"))
    if d.quality_col and d.quality_col in df:
        low = df[df[d.quality_col] < cfg.quality_min_pct][d.group].tolist()
        if low:
            issues.append(("warn", f"{d.quality_col} below {cfg.quality_min_pct:.0f}% in {d.group} {low}: those rows rest mostly on gap-filled inputs"))
    summary = dict(name=d.name, n_obs=n, n_groups=len(groups), groups=groups, n_labelled=n_lab, features=len(d.features), duplicates=dup,
                   target=d.target, target_unit=d.units.get(d.target, "not declared"))
    return dict(summary=summary, features=pd.DataFrame(rows), issues=issues)


# ------------------------------------------------------------------------------------------------ B. feature compatibility
def compatibility(s: Domain, t: Domain, cfg: Config):
    rows = []
    for f in sorted(set(s.features) | set(t.features)):
        in_s, in_t = f in s.features, f in t.features
        us, ut = s.units.get(f), t.units.get(f)
        status = "shared" if in_s and in_t else ("source only" if in_s else "target only")
        note = ""
        if in_s and in_t:
            if us and ut and us != ut:
                status, note = "shared, units differ", f"source: {us}; target: {ut}"
            elif not (us and ut):
                note = "units not declared in both domains; assumed equal"
            elif str(s.df[f].dtype) != str(t.df[f].dtype):
                note = f"dtype differs ({s.df[f].dtype} vs {t.df[f].dtype})"
        rows.append(dict(feature=f, status=status, available_in_source=in_s, available_in_target=in_t,
                         unit_source=us or "-", unit_target=ut or "-", note=note))
    for p in cfg.proposed_features:
        rows.append(dict(feature=p, status="PROPOSED, not available", available_in_source=False, available_in_target=False,
                         unit_source="-", unit_target="-", note="not in either dataset; it is not used and is never filled in"))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ C. domain differences
def _bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def smd_of(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((b.mean() - a.mean()) / sp) if sp > 0 else (0.0 if a.mean() == b.mean() else np.inf)


def _rbf(A, B, sigma):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * sigma ** 2))


def mmd2_unbiased(Zs, Zt, sigma=None):
    pooled = np.vstack([Zs, Zt])
    if sigma is None:
        d = np.sqrt(((pooled[:, None, :] - pooled[None, :, :]) ** 2).sum(-1))
        sigma = float(np.median(d[np.triu_indices_from(d, 1)])) or 1.0
    n, m = len(Zs), len(Zt)
    Kss, Ktt, Kst = _rbf(Zs, Zs, sigma), _rbf(Zt, Zt, sigma), _rbf(Zs, Zt, sigma)
    return float((Kss.sum() - np.trace(Kss)) / (n * (n - 1)) + (Ktt.sum() - np.trace(Ktt)) / (m * (m - 1)) - 2 * Kst.mean()), sigma


def coral_distance(Zs, Zt):
    d = Zs.shape[1]
    Cs, Ct = np.cov(Zs, rowvar=False), np.cov(Zt, rowvar=False)
    return float(((Cs - Ct) ** 2).sum() / (4 * d * d))


def _perm_p(stat_fn, Zs, Zt, observed, n_perm, seed):
    rng = np.random.RandomState(seed)
    pooled = np.vstack([Zs, Zt])
    ns = len(Zs)
    hits = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        hits += stat_fn(pooled[idx[:ns]], pooled[idx[ns:]]) >= observed
    return (hits + 1) / (n_perm + 1)


def domain_differences(Xs: pd.DataFrame, Xt: pd.DataFrame, cfg: Config):
    """per-feature SMD / KS / Wasserstein, and joint MMD and CORAL on shared features (complete cases, pooled standardisation)"""
    feats = [c for c in Xs.columns if c in Xt.columns]
    rows = []
    for f in feats:
        a, b = pd.to_numeric(Xs[f], errors="coerce").dropna().values, pd.to_numeric(Xt[f], errors="coerce").dropna().values
        if len(a) < 2 or len(b) < 2:
            rows.append(dict(feature=f, n_source=len(a), n_target=len(b)))
            continue
        ks = stats.ks_2samp(a, b)
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        w = float(stats.wasserstein_distance(a, b))
        rows.append(dict(feature=f, n_source=len(a), n_target=len(b), mean_source=a.mean(), mean_target=b.mean(), sd_source=a.std(ddof=1),
                         sd_target=b.std(ddof=1), SMD=smd_of(a, b), KS=float(ks.statistic), KS_p=float(ks.pvalue), Wasserstein=w,
                         Wasserstein_per_SD=w / sp if sp > 0 else np.nan))
    tab = pd.DataFrame(rows)
    if "KS_p" in tab:
        ok = tab.KS_p.notna()
        tab.loc[ok, "KS_p_adj"] = _bh(tab.loc[ok, "KS_p"].values)
        a_s = tab.SMD.abs()
        tab["SMD_size"] = np.select([a_s >= cfg.smd_large, a_s >= cfg.smd_medium, a_s >= cfg.smd_small],
                                    ["large", "medium", "small"], default="negligible")
        tab["KS_significant"] = tab.KS_p_adj < cfg.alpha
    Ss, St = Xs[feats].apply(pd.to_numeric, errors="coerce").dropna(), Xt[feats].apply(pd.to_numeric, errors="coerce").dropna()
    glob = dict(features_used=feats, n_complete_source=len(Ss), n_complete_target=len(St))
    if len(feats) >= 1 and len(Ss) >= 3 and len(St) >= 3:
        pooled = pd.concat([Ss, St])
        mu, sd = pooled.mean(), pooled.std(ddof=0).replace(0, 1.0)
        Zs, Zt = ((Ss - mu) / sd).values, ((St - mu) / sd).values
        mmd, sigma = mmd2_unbiased(Zs, Zt)
        glob.update(mmd2=mmd, mmd_bandwidth=sigma,
                    mmd_p=_perm_p(lambda a, b: mmd2_unbiased(a, b, sigma)[0], Zs, Zt, mmd, cfg.mmd_permutations, cfg.random_state))
        if len(feats) >= 2:
            cd = coral_distance(Zs, Zt)
            glob.update(coral=cd, coral_p=_perm_p(coral_distance, Zs, Zt, cd, cfg.mmd_permutations, cfg.random_state + 1),
                        coral_reliable=bool(min(len(Ss), len(St)) > len(feats)))
    else:
        glob["note"] = "too few complete rows for the joint metrics (MMD, CORAL)"
    return dict(table=tab, **glob)


def fine_diagnostics(source: Domain, target: Domain, cfg: Config):
    """domain-classifier AUC (climate; vegetation and radar) and the crop-calendar offset, from the finer ten-day observations"""
    if source.fine_df is None or target.fine_df is None:
        return None
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fa, fb = source.fine_df, target.fine_df
    shared = [f for f in (source.fine_features or []) if f in (target.fine_features or [])]
    out = {}

    def auc(cols):
        a, b = fa[cols].dropna(), fb[cols].dropna()
        if len(a) < cfg.auc_min_rows or len(b) < cfg.auc_min_rows:
            return None
        X, y = pd.concat([a, b]).values, np.r_[np.zeros(len(a)), np.ones(len(b))]
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced"))
        return float(cross_val_score(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=cfg.random_state), scoring="roc_auc").mean())
    clim = [f for f in shared if any(k in f for k in ("Temp", "Rain", "Solar"))]
    veg = [f for f in shared if any(k in f for k in ("NDVI", "EVI", "SAR"))]
    out["auc_climate"], out["auc_vegetation"] = (auc(clim) if clim else None), (auc(veg) if veg else None)
    out["climate_features"], out["vegetation_features"] = clim, veg
    if "Decade_N" in fa and "Decade_N" in fb and "NDVI" in fa and "NDVI" in fb:
        ca, cb = fa.groupby("Decade_N").NDVI.mean().sort_index().values, fb.groupby("Decade_N").NDVI.mean().sort_index().values
        if len(ca) == len(cb) and len(ca) >= 12:
            half = len(ca) // 2
            cors = {k: float(np.corrcoef(ca, np.roll(cb, k))[0, 1]) for k in range(-half, half)}
            k = max(cors, key=cors.get)
            out.update(calendar_offset_days=abs(k) * 10, corr_at_zero=cors[0], corr_best=cors[k])
    return out


def label_shift(ys, yt):
    a, b = pd.to_numeric(ys, errors="coerce").dropna().values, pd.to_numeric(yt, errors="coerce").dropna().values
    if len(a) < 2 or len(b) < 2:
        return None
    ks = stats.ks_2samp(a, b)
    return dict(n_source=len(a), n_target=len(b), mean_source=a.mean(), mean_target=b.mean(), SMD=smd_of(a, b), KS=float(ks.statistic),
                KS_p=float(ks.pvalue), Wasserstein=float(stats.wasserstein_distance(a, b)))


# ------------------------------------------------------------------------------------------------ D. the report object
class AssessmentReport:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def to_markdown(self):
        L = [f"# Pre-transfer assessment: {self.source_name} to {self.target_name}", "",
             f"Computed {self.created}; fingerprint `{self.token}`.", "", f"**Data readiness: {self.readiness.upper().replace('_', ' ')}**", ""]
        L += ["## Readiness reasons"] + [f"- {r}" for r in self.readiness_reasons] + [""]
        for nm, q in (("Source", self.source_quality), ("Target", self.target_quality)):
            s = q["summary"]
            L += [f"## {nm} data: {s['name']}", f"- {s['n_obs']} observations, {s['n_groups']} groups ({s['groups']}), {s['n_labelled']} labelled, {s['features']} features",
                  *[f"- [{sev}] {msg}" for sev, msg in q["issues"]], ""]
        L += ["## Feature compatibility", md_table(self.compat), ""]
        for nm, dd in (("Modelling level", self.diff_model), ("Ten-day level (finer observations)", self.diff_fine)):
            if dd is None:
                continue
            L += [f"## Domain differences: {nm}", md_table(dd["table"]),
                  f"MMD^2 = {dd.get('mmd2', float('nan')):.4f} (permutation p = {dd.get('mmd_p', float('nan')):.3f}; the smallest possible p is "
                  f"1/(permutations+1)); CORAL = {dd.get('coral', float('nan')):.4f} (p = {dd.get('coral_p', float('nan')):.3f}"
                  f"{'' if dd.get('coral_reliable', True) else '; NOT interpretable: fewer rows than features'})", ""]
        if getattr(self, "fine_diag", None):
            fd = self.fine_diag
            L += ["## Domain classifier and crop calendar (ten-day level)",
                  f"- Climate AUC {fd['auc_climate']:.2f}" if fd.get("auc_climate") is not None else "- Climate AUC: too few rows",
                  f"- Vegetation and radar AUC {fd['auc_vegetation']:.2f}" if fd.get("auc_vegetation") is not None else "- Vegetation and radar AUC: too few rows",
                  (f"- Calendar offset {fd['calendar_offset_days']} days (curve correlation {fd['corr_at_zero']:+.2f} unshifted, {fd['corr_best']:+.2f} aligned); informational"
                   if "calendar_offset_days" in fd else "- Calendar offset: not computed"), ""]
        if self.label:
            L += ["## Label (yield) shift", md_table(pd.DataFrame([self.label])), ""]
        L += ["## Negative-transfer risks"] + [f"- {r}" for r in self.risks] + ["", "## Recommended next steps"] + [f"- {r}" for r in self.next_steps]
        L += ["", "## What each metric means"] + [f"- **{k}**: {v}" for k, v in METRIC_GUIDE.items()]
        return "\n".join(L)


def assess(source: Domain, target: Domain, cfg: Config):
    qs, qt = quality(source, cfg), quality(target, cfg)
    comp = compatibility(source, target, cfg)
    shared = comp[comp.status.isin(["shared", "shared, units differ"])].feature.tolist()
    Xs, Xt = source.df[shared], target.df[shared]
    diff_model = domain_differences(Xs, Xt, cfg) if shared else None
    diff_fine = None
    if source.fine_df is not None and target.fine_df is not None:
        fs = [f for f in (source.fine_features or []) if f in (target.fine_features or [])]
        if fs:
            diff_fine = domain_differences(source.fine_df[fs], target.fine_df[fs], cfg)
    lab = label_shift(source.df[source.target], target.df[target.target])
    fdiag = fine_diagnostics(source, target, cfg)

    block, warn, risks, steps = [], [], [], []
    for nm, q in (("source", qs), ("target", qt)):
        block += [f"{nm}: {m}" for sev, m in q["issues"] if sev == "block"]
        warn += [f"{nm}: {m}" for sev, m in q["issues"] if sev == "warn"]
    if not shared:
        block.append("no shared features between source and target")
    mism = comp[comp.status == "shared, units differ"]
    if len(mism):
        warn.append(f"{len(mism)} shared feature(s) with different declared units: {mism.feature.tolist()} (harmonise before comparing)")
    n_t_lab = qt["summary"]["n_labelled"]
    if shared and n_t_lab < cfg.min_obs_per_feature * len(shared):
        warn.append(f"{n_t_lab} labelled target rows for {len(shared)} features (the configured rule of thumb is {cfg.min_obs_per_feature:g} rows per feature)")
        risks.append(f"Too few labelled target rows for a flexible model ({n_t_lab} rows, {len(shared)} features): estimates will be unstable and the "
                     "evaluation noisy.")
    for nm, dd in (("modelling level", diff_model), ("ten-day level", diff_fine)):
        if not dd:
            continue
        t = dd["table"]
        if "KS_significant" in t:
            ns = int(t.KS_significant.sum())
            nl = int((t.SMD.abs() >= cfg.smd_large).sum())
            if ns:
                risks.append(f"{ns} of {len(t)} features differ in distribution at the {nm} (adjusted KS p < {cfg.alpha:g}): "
                             f"{t[t.KS_significant].feature.tolist()}.")
            if nl:
                risks.append(f"{nl} feature(s) have a large standardised mean difference (|SMD| >= {cfg.smd_large:g}) at the {nm}.")
        if dd.get("mmd_p", 1) < cfg.alpha:
            risks.append(f"The joint feature distributions differ at the {nm} (MMD permutation p = {dd['mmd_p']:.3f}).")
        if "coral" in dd and not dd.get("coral_reliable", True):
            warn.append(f"CORAL at the {nm} is not interpretable: the covariance of {len(dd['features_used'])} features is estimated from "
                        f"{min(dd['n_complete_source'], dd['n_complete_target'])} rows (fewer rows than features).")
        elif dd.get("coral_p", 1) < cfg.alpha:
            risks.append(f"The feature covariance differs at the {nm} (CORAL permutation p = {dd['coral_p']:.3f}).")
    if fdiag and fdiag.get("auc_climate") is not None and fdiag["auc_climate"] >= cfg.far_auc:
        risks.append(f"FAR DOMAIN: a classifier tells the two climates apart with AUC = {fdiag['auc_climate']:.2f} (>= {cfg.far_auc:g}; 0.5 is chance). "
                     "Transfer between very different climates often hurts.")
    if lab and lab["KS_p"] < cfg.alpha:
        risks.append(f"The yield distributions differ (KS p = {lab['KS_p']:.3f}, SMD {lab['SMD']:+.2f}): methods that assume a shared label relationship are at risk.")
    elif lab and abs(lab["SMD"]) >= cfg.smd_large:
        risks.append(f"The yield means differ by {lab['SMD']:+.2f} pooled SD (KS not significant, but few rows).")
    readiness = "not_ready" if block else ("conditional" if warn else "ready")
    reasons = block + warn or ["no blocking problem and no warning was found"]
    steps += (["Resolve the blocking problems above before continuing."] if block else [])
    steps += (["Harmonise names and units (Step 3); invalid values will be set to missing, never kept."] if not block else [])
    if warn and not block:
        steps.append("Review each warning; the workflow can continue, and the warnings stay in the report.")
    if risks:
        steps.append("Keep a target-only baseline and a source-only baseline in every comparison, because the risks above mean transfer may not help.")
    token = token_of(source, target, cfg)
    return AssessmentReport(source_name=source.name, target_name=target.name, created=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), token=token,
                            source_quality=qs, target_quality=qt, compat=comp, shared_features=shared, diff_model=diff_model, diff_fine=diff_fine,
                            label=lab, fine_diag=fdiag, readiness=readiness, readiness_reasons=reasons, blockers=block, warnings=warn, risks=risks, next_steps=steps,
                            n_source=len(source.df), n_target=len(target.df), n_target_labelled=n_t_lab, config=cfg.to_dict())
