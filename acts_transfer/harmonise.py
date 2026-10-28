"""Stage 3 · Harmonise data. Static, label-free clean-up only; everything that is FITTED is deferred to the training folds.

Done here (once, recorded in the log, originals untouched):
  rename            source and target columns to one canonical name
  convert units     explicit factor, for example kg/ha -> t/ha (never inferred)
  duplicates        exact duplicate rows removed
  invalid values    values outside the configured plausible range are set to MISSING (never deleted silently, never kept)
  features          only features present in BOTH domains are kept; constant or too-incomplete ones are dropped with the reason
Deferred to each training fold (see evaluation.py), so no held-out label or feature ever shapes them:
  imputation of missing values (median), standardisation (mean/SD)
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
    scaling: str = "standardise (fitted inside each training fold)"


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
    keep_s = [s_grp, s_tgt] + feats
    keep_t = [t_grp, t_tgt] + feats
    hs = s_df[keep_s].rename(columns={s_grp: "group", s_tgt: "y"}).copy()
    ht = t_df[keep_t].rename(columns={t_grp: "group", t_tgt: "y"}).copy()
    for df in (hs, ht):
        for c in feats + ["y"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    log.append(dict(step="deferred to training folds", domain="both", detail=f"{spec.missing_strategy}; {spec.scaling}", rows_affected=0))
    # validation checks
    val = []
    val.append(dict(check="at least one shared feature retained", passed=len(feats) > 0, detail=f"{len(feats)} retained"))
    val.append(dict(check="same feature names in both domains", passed=list(hs.columns) == list(ht.columns), detail="identical columns"))
    val.append(dict(check="all retained features numeric", passed=all(pd.api.types.is_numeric_dtype(hs[c]) and pd.api.types.is_numeric_dtype(ht[c]) for c in feats),
                    detail="numeric dtype"))
    val.append(dict(check="no infinite values", passed=bool(not np.isinf(hs[feats + ['y']].values).any() and not np.isinf(ht[feats + ['y']].values).any()),
                    detail="checked after conversion"))
    val.append(dict(check="labels present", passed=bool(hs.y.notna().sum() >= cfg.min_obs_block and ht.y.notna().sum() >= cfg.min_obs_block),
                    detail=f"source {int(hs.y.notna().sum())}, target {int(ht.y.notna().sum())} labelled rows"))
    val.append(dict(check="no duplicate rows left", passed=bool(not hs.duplicated().any() and not ht.duplicated().any()), detail="after de-duplication"))
    val.append(dict(check="one row per group", passed=bool(not hs.group.duplicated().any() and not ht.group.duplicated().any()), detail="group = season/year"))
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
    return HarmonisedBundle(hs, ht, feats, "y", "group", log, ftab, val, ready, tok, report.token, source.name, target.name,
                            spec.__dict__, *fine)
