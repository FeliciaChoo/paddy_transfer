"""Stage 3 · Harmonise data. Static, label-free clean-up only; everything that is FITTED is deferred to the training folds.

Done here (once, recorded in the log, originals untouched):
  rename            source and target columns to one canonical name
  convert units     explicit factor, for example kg/ha -> t/ha (never inferred)
  duplicates        exact duplicate rows removed
  invalid values    values outside the configured plausible range are set to MISSING (never deleted silently, never kept)
  features          only features present in BOTH domains are kept; constant or too-incomplete ones are dropped with the reason
Deferred to each training fold (see evaluation.py and methods.Prep), so no held-out label or feature ever shapes them:
  imputation of missing values (median), scaling (robust median/IQR, or mean/SD), VIF / correlation feature selection (at most max_features, default 6),
  and, inside the methods that use them, within-granary Z-scores (anomalies) and CORAL covariance alignment.
shift_preview() below shows what Z-scoring and CORAL do to the source-target distance. It is a label-free diagnostic on all rows and is never used to fit a model.
Nothing is invented: a feature that does not exist in a domain is not created.
"""
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .assessment import AssessmentError, AssessmentReport, token_of
from .config import Config, Domain


class StaleAssessment(AssessmentError):
    pass


class AssessmentBlocked(AssessmentError):
    pass


@dataclass
class HarmonisationSpec:
    rename: dict = field(default_factory=lambda: {"source": {}, "target": {}})
    conversions: list = field(default_factory=list)      # {'column','factor','new_name','new_unit'}
    drop_features: dict = field(default_factory=dict)    # feature -> reason (user choice)
    drop_duplicates: bool = True
    invalid_to_nan: bool = True
    missing_strategy: str = "median (fitted inside each training fold)"
    scaler: str = "robust"                               # 'robust' = RobustScaler (median/IQR), 'standard' = mean/SD; fitted inside each training fold
    select_features: bool = True                         # VIF / correlation selection inside each training fold
    max_features: int = 6                                # p <= 6
    scaling: str = "robust scaling (median/IQR), within-granary Z-score and CORAL alignment, all fitted inside each training fold"


def default_spec(report: AssessmentReport, source: Domain, target: Domain):
    """a proposal built from the assessment: kg/ha -> t/ha for yield, nothing else renamed (no mismatch was found)"""
    sp = HarmonisationSpec()
    for col in {source.target, target.target}:
        if col.lower().endswith("_kg_ha"):
            sp.conversions.append(dict(column=col, factor=0.001, new_name=col[: -len("_kg_ha")] + "_t_ha", new_unit="t/ha"))
    return sp


@dataclass
class HarmonisedBundle:
    source: pd.DataFrame
    target: pd.DataFrame
    features: list
    target_col: str
    group: str
    log: list
    feature_table: pd.DataFrame
    validation: list
    ready: bool
    token: str
    assessment_token: str
    source_name: str
    target_name: str
    spec: dict
    fine_source: object = None
    fine_target: object = None
    fine_features: object = None
    y_to_kg: float = 1.0                    # multiply the harmonised label by this to get kg/ha (1000 after the t/ha conversion)


def _apply(df, d: Domain, side: str, spec: HarmonisationSpec, cfg: Config, log):
    out = df.copy()
    ren = spec.rename.get(side, {})
    if ren:
        out = out.rename(columns=ren)
        log.append(dict(step="rename", domain=side, detail=ren, rows_affected=0))
    target, group = ren.get(d.target, d.target), ren.get(d.group, d.group)
    units = {ren.get(k, k): v for k, v in d.units.items()}
    for cv in spec.conversions:
        col = ren.get(cv["column"], cv["column"])
        if col in out.columns and (col == target or col in d.features or col in units):
            out[cv["new_name"]] = pd.to_numeric(out[col], errors="coerce") * cv["factor"]
            out = out.drop(columns=col)
            log.append(dict(step="convert units", domain=side, detail=f"{col} x {cv['factor']} -> {cv['new_name']} ({cv['new_unit']})",
                            rows_affected=int(out[cv["new_name"]].notna().sum())))
            if col == target:
                target = cv["new_name"]
            units[cv["new_name"]] = cv["new_unit"]
    if spec.drop_duplicates:
        n0 = len(out)
        out = out.drop_duplicates().reset_index(drop=True)
        log.append(dict(step="remove duplicates", domain=side, detail="exact duplicate rows", rows_affected=n0 - len(out)))
    return out, target, group, units


def harmonise(report: AssessmentReport, source: Domain, target: Domain, spec: HarmonisationSpec, cfg: Config):
    if getattr(report, "token", None) != token_of(source, target, cfg):
        raise StaleAssessment("the assessment was computed on different data or settings; run the assessment again before harmonising")
    if report.readiness == "not_ready":
        raise AssessmentBlocked("data readiness is NOT READY: " + "; ".join(report.blockers))
    log = []
    s_df, s_tgt, s_grp, s_units = _apply(source.df, source, "source", spec, cfg, log)
    t_df, t_tgt, t_grp, t_units = _apply(target.df, target, "target", spec, cfg, log)
    ren_s, ren_t = spec.rename.get("source", {}), spec.rename.get("target", {})
    fs = [ren_s.get(f, f) for f in source.features]
    ft = [ren_t.get(f, f) for f in target.features]
    shared = [f for f in fs if f in ft]
    ftab = []
    for f in sorted(set(fs) | set(ft)):
        keep, why = True, "shared by both domains"
        if f not in shared:
            keep, why = False, "not present in both domains"
        elif f in spec.drop_features:
            keep, why = False, spec.drop_features[f]
        else:
            for nm, df in (("source", s_df), ("target", t_df)):
                col = pd.to_numeric(df[f], errors="coerce")
                if col.nunique(dropna=True) <= 1:
                    keep, why = False, f"constant in the {nm} data (no information)"
                elif col.isna().mean() * 100 > cfg.missing_block_pct:
                    keep, why = False, f"{col.isna().mean() * 100:.0f}% missing in the {nm} data"
        ftab.append(dict(feature=f, retained=keep, reason=why, unit=s_units.get(f, t_units.get(f, "not declared"))))
    ftab = pd.DataFrame(ftab)
    feats = ftab[ftab.retained].feature.tolist()
    removed = ftab[~ftab.retained]
    if len(removed):
        log.append(dict(step="drop features", domain="both", detail=dict(zip(removed.feature, removed.reason)), rows_affected=0))
    if spec.invalid_to_nan:
        for nm, df in (("source", s_df), ("target", t_df)):
            n_inv = 0
            for c in feats + [s_tgt if nm == "source" else t_tgt]:
                r = cfg.range_for(c)
                if r:
                    v = pd.to_numeric(df[c], errors="coerce")
                    bad = (v < r[0]) | (v > r[1])
                    n_inv += int(bad.sum())
                    df.loc[bad, c] = np.nan
            log.append(dict(step="invalid values to missing", domain=nm, detail="values outside the plausible range", rows_affected=n_inv))
    s_unit = ren_s.get(source.unit, source.unit) if source.unit else None
    t_unit = ren_t.get(target.unit, target.unit) if target.unit else None
    keep_s = [s_grp, s_tgt] + feats + ([s_unit] if s_unit else [])
    keep_t = [t_grp, t_tgt] + feats + ([t_unit] if t_unit else [])
    hs = s_df[keep_s].rename(columns={s_grp: "group", s_tgt: "y", **({s_unit: "unit"} if s_unit else {})}).copy()
    ht = t_df[keep_t].rename(columns={t_grp: "group", t_tgt: "y", **({t_unit: "unit"} if t_unit else {})}).copy()
    if "unit" not in hs:
        hs["unit"] = source.name
    if "unit" not in ht:
        ht["unit"] = target.name
    def _w(dom, df, out):
        """optical-quality sample weight w = max(0.2, Optical_Valid_Fraction); 1.0 where the domain has no quality column"""
        q = dom.quality_col
        if q and q in df:
            frac = pd.to_numeric(df.loc[out.index, q], errors="coerce")
            frac = frac / 100.0 if frac.max() > 1.5 else frac                     # a percentage or a fraction
            return np.maximum(0.2, frac.fillna(0.2).values)
        return np.ones(len(out))
    hs = hs[["group", "unit", "y"] + feats].copy()
    ht = ht[["group", "unit", "y"] + feats].copy()
    hs["w"], ht["w"] = _w(source, s_df, hs), _w(target, t_df, ht)
    for df in (hs, ht):
        for c in feats + ["y"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    log.append(dict(step="deferred to training folds", domain="both", detail=f"{spec.missing_strategy}; {spec.scaling}", rows_affected=0))
    # validation checks
    val = []
    val.append(dict(check="at least one shared feature retained", passed=len(feats) > 0, detail=f"{len(feats)} retained"))
    val.append(dict(check="same feature names in both domains", passed=list(hs.columns) == list(ht.columns), detail="identical columns"))
    pooled = hs["unit"].nunique() > 1 or ht["unit"].nunique() > 1
    val.append(dict(check="all retained features numeric", passed=all(pd.api.types.is_numeric_dtype(hs[c]) and pd.api.types.is_numeric_dtype(ht[c]) for c in feats),
                    detail="numeric dtype"))
    val.append(dict(check="no infinite values", passed=bool(not np.isinf(hs[feats + ['y']].values).any() and not np.isinf(ht[feats + ['y']].values).any()),
                    detail="checked after conversion"))
    val.append(dict(check="labels present", passed=bool(hs.y.notna().sum() >= cfg.min_obs_block and ht.y.notna().sum() >= cfg.min_obs_block),
                    detail=f"source {int(hs.y.notna().sum())}, target {int(ht.y.notna().sum())} labelled rows"))
    val.append(dict(check="no duplicate rows left", passed=bool(not hs.duplicated().any() and not ht.duplicated().any()), detail="after de-duplication"))
    val.append(dict(check="one row per group", passed=bool(not hs.duplicated(["unit", "group"]).any() and not ht.duplicated(["unit", "group"]).any()),
                    detail="one row per granary and season/year" if pooled else "group = season/year"))
    unit_ok = True
    for f in feats:
        us, ut = s_units.get(f), t_units.get(f)
        unit_ok &= (us == ut) or not (us and ut)
    val.append(dict(check="units agree for every retained feature", passed=bool(unit_ok), detail="declared units compared after conversion"))
    ready = all(v["passed"] for v in val)
    fine = (None, None, None)
    if source.fine_df is not None and target.fine_df is not None:
        ff = [f for f in (source.fine_features or []) if f in (target.fine_features or [])]
        fine = (source.fine_df, target.fine_df, ff)
    tok = hashlib.sha256((report.token + json.dumps(spec.__dict__, sort_keys=True, default=str) +
                          str(pd.util.hash_pandas_object(hs, index=False).sum()) + str(pd.util.hash_pandas_object(ht, index=False).sum())).encode()).hexdigest()[:16]
    y_to_kg = next((1.0 / cv["factor"] for cv in spec.conversions if cv["column"] in (target.target, ren_t.get(target.target, target.target))
                    and cv["new_unit"] == "t/ha" and cv["factor"]), 1.0)
    return HarmonisedBundle(hs, ht, feats, "y", "group", log, ftab, val, ready, tok, report.token, source.name, target.name,
                            spec.__dict__, *fine, y_to_kg=float(y_to_kg))


def fold_settings(bundle: HarmonisedBundle, cfg: Config):
    """the in-fold harmonisation chosen in Step 3 (scaler, feature selection, p <= max_features) applied to the run configuration"""
    import dataclasses
    sp = bundle.spec or {}
    return dataclasses.replace(cfg, scaler=sp.get("scaler", cfg.scaler), select_features=bool(sp.get("select_features", cfg.select_features)),
                               max_features=int(sp.get("max_features", cfg.max_features)))


def shift_preview(bundle: HarmonisedBundle, cfg: Config):
    """Label-free diagnostic of how much each harmonisation reduces the source-target distance on the retained features. Rows: raw (pooled standardisation),
    within-granary Z-score (anomalies), and CORAL-aligned source. Uses all rows, so it is for DISPLAY only and never fits a model."""
    from .assessment import coral_distance, mmd2_unbiased
    from .methods import coral_align
    feats = bundle.features
    S, T = bundle.source.dropna(subset=feats), bundle.target.dropna(subset=feats)
    if len(feats) < 2 or len(S) < 3 or len(T) < 3:
        return pd.DataFrame()
    pool = pd.concat([S[feats], T[feats]])
    mu, sd = pool.mean(), pool.std(ddof=0).replace(0, 1.0)
    Zs, Zt = ((S[feats] - mu) / sd).values, ((T[feats] - mu) / sd).values

    def z_by_unit(df):
        out = df[feats].copy()
        for _, idx in df.groupby("unit").groups.items():
            blk = df.loc[idx, feats]
            out.loc[idx] = (blk - blk.mean()) / blk.std(ddof=1).replace(0, 1.0).fillna(1.0)
        return out.fillna(0.0).values
    As, At = z_by_unit(S), z_by_unit(T)
    Ca = coral_align(Zs, Zt, cfg.coral_eps)
    rows = []
    for nm, a, b in (("raw (pooled standardisation)", Zs, Zt), ("within-granary Z-score (anomalies)", As, At), ("CORAL-aligned source", Ca, Zt)):
        rows.append(dict(harmonisation=nm, CORAL_distance=float(coral_distance(a, b)), MMD2=float(mmd2_unbiased(a, b)[0]), rows_source=len(a), rows_target=len(b)))
    return pd.DataFrame(rows)
