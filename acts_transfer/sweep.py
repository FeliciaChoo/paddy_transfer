"""Evidence-first transfer qualification (PACE V4): Discover -> Verify -> Deploy.

Step 1  DISCOVER  Regional benchmark sweep. The same source (MADA) is transferred to EACH other granary as a pseudo-target, with identical out-of-fold splits
                  (leave-one-year-out is primary; forward expanding-window is run as well). Per held-out granary-year the absolute errors of every transfer method
                  and of the baselines are recorded in kg/ha, and the paired difference is POOLED across granaries. Yield levels (kg/ha) are never averaged
                  across granaries; only error DIFFERENCES are pooled.
                    Delta_NT = |err| of the transfer method - |err| of Target-Only   (< 0 positive transfer, > 0 negative transfer)
                    Delta_DS = |err| of the transfer method - |err| of Source-Only   (< 0: adaptation recovers accuracy lost to domain shift)
                  Tests on the pooled differences: paired non-parametric bootstrap 95% CI of the mean (over pairs, and, as a sensitivity check that respects the
                  dependence of years inside a granary, over granaries) and the Wilcoxon signed-rank test.
                  MADA cannot be its own target, so the sweep has 9 target granaries (about 63 granary-year pairs), not 10.
        Suitability mapping: the per-granary Delta_NT is set against domain AUC, MMD2, CORAL, calendar offset, historical yield CV and the number of target
                  years (Spearman rank correlation), to see where positive and negative transfer occur. With 9 granaries this is descriptive, not a fitted law.
Step 2  VERIFY    For the granary of interest the out-of-fold Delta_NT (5-7 held-out years) and its position among the sweep granaries are inspected. A transfer
                  method is QUALIFIED for that granary only if (a) the pooled sweep passes for that method (bootstrap CI of the pooled Delta_NT entirely below zero and
                  Wilcoxon one-sided p < 0.05) and (b) the Negative-Transfer Safeguard Gate passes for the granary (Delta_NT = RMSE_transfer - min(RMSE_target-only, RMSE_persistence) < 0).
                  MADA is therefore qualified empirically, never assumed.
Nothing here sees the held-out year when fitting: every fit goes through evaluation.evaluate().
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from .config import Config
from .evaluation import evaluate
from .harmonise import HarmonisedBundle

SWEEP_METHODS = ["detrended_residual_transfer", "granary_intercept_ridge", "kmm_ridge", "two_stage_tradaboost_r2", "s_tradaboost_r2", "z_anomaly_transfer", "tradaboost_r2"]
SWEEP_BASELINES = ["target_only", "source_only", "pooled", "target_mean", "season_trend"]
MAX_MISSING = 0.40
KG = 1000.0


def stage_features(Y):
    """C1_* (main season) growth-stage features plus the static covariates Soil_pH and Optical_Valid_Fraction (Cycle 2 is too often absent to sweep)"""
    f = [c for c in Y.columns if c.startswith("C1_") and not c.endswith(("Greenup_Decade", "Peak_Decade", "_rip"))]      # heading-stage features only: ripening is after the 70-day nowcast
    return f + [c for c in ("Soil_pH", "Optical_Valid_Fraction") if c in Y.columns]


def build_bundle(Y, source_names, target, feats):
    """a harmonised bundle for one source -> target pair, without the interactive assessment: yield in t/ha, one row per granary-year, features that are
    missing in more than 40% of either domain or constant are left out (everything else is imputed and scaled inside each fold)"""
    S = Y[Y.District.isin(source_names)].sort_values(["District", "Year"])
    T = Y[Y.District == target].sort_values("Year")
    keep = [f for f in feats if S[f].isna().mean() <= MAX_MISSING and T[f].isna().mean() <= MAX_MISSING and S[f].nunique() > 1 and T[f].nunique() > 1]

    def tab(df):
        w = np.maximum(0.2, df.Optical_Valid_Fraction.fillna(0.2).values) if "Optical_Valid_Fraction" in df else np.ones(len(df))
        o = pd.DataFrame({"group": df.Year.values, "unit": df.District.values, "y": df.Yield_kg_ha.values / KG})
        for f in keep:
            o[f] = df[f].values
        o["w"] = w
        return o
    return HarmonisedBundle(tab(S), tab(T), keep, "y", "group", [], pd.DataFrame(), [], True, "sweep", "sweep", "+".join(source_names), target, {}, y_to_kg=KG), keep


def _boot_mean(d, n, seed, cluster=None):
    r = np.random.RandomState(seed)
    d = np.asarray(d, float)
    if cluster is None:
        v = [d[r.randint(0, len(d), len(d))].mean() for _ in range(n)]
    else:
        groups = [d[cluster == c] for c in np.unique(cluster)]
        v = [np.concatenate([groups[i] for i in r.randint(0, len(groups), len(groups))]).mean() for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


def _wilcoxon(d):
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) < 6:
        return np.nan, np.nan
    two = wilcoxon(d, alternative="two-sided").pvalue
    less = wilcoxon(d, alternative="less").pvalue
    return float(two), float(less)


def run_sweep(Y, F, source="MADA", cfg: Config = None, methods=None, progress=None):
    """-> dict(pairs=DataFrame, pooled=DataFrame, granary=DataFrame, suitability=DataFrame, meta=dict). Y: granary-year table, F: ten-day table (for the domain screen)"""
    from .screening import screen_granaries
    cfg = cfg or Config()
    methods = methods or SWEEP_METHODS
    feats = stage_features(Y)
    targets = [g for g in sorted(Y.District.unique()) if g != source]
    pairs, grows = [], []
    for i, tgt in enumerate(targets):
        if progress:
            progress(i, len(targets), tgt)
        b, used = build_bundle(Y, [source], tgt, feats)
        if not used:
            continue
        screen = screen_granaries(F, tgt, cfg)
        sr = screen[screen.granary == source]
        sr = sr.iloc[0].to_dict() if len(sr) else {}
        res = {}
        for strat in ("loso", "forward"):
            c = Config(**{**cfg.__dict__, "cv_strategy": strat})
            r = evaluate(b, list(methods) + SWEEP_BASELINES, "ridge", c)
            res[strat] = r
            p = r.predictions.copy()
            wide = p.pivot_table(index="group", columns="method", values="abs_err") * KG
            ytrue = p.groupby("group").y_true.first() * KG
            for m in methods:
                if m not in wide.columns:
                    continue
                for g, row in wide.iterrows():
                    if np.isnan(row.get(m, np.nan)):
                        continue
                    pairs.append(dict(strategy=strat, granary=tgt, group=int(g), method=m, y_true=float(ytrue[g]), err_transfer=float(row[m]),
                                      err_target_only=float(row.get("target_only", np.nan)), err_source_only=float(row.get("source_only", np.nan)),
                                      err_persistence=float(row.get("persistence", np.nan))))
            met = r.metrics.set_index("method")
            for m in methods:
                if m not in met.index:
                    continue
                rm_t, rm_p = met.RMSE.get("target_only", np.nan) * KG, met.RMSE.get("persistence", np.nan) * KG
                base = np.nanmin([rm_t, rm_p])
                grows.append(dict(strategy=strat, granary=tgt, method=m, n_years=int(met.loc[m, "n_test"]), RMSE_transfer=met.RMSE[m] * KG, RMSE_target_only=rm_t,
                                  RMSE_persistence=rm_p, RMSE_source_only=met.RMSE.get("source_only", np.nan) * KG,
                                  Delta_NT_vs_target_only=met.RMSE[m] * KG - rm_t, Delta_NT_gate=met.RMSE[m] * KG - base,
                                  Delta_DS=met.RMSE[m] * KG - met.RMSE.get("source_only", np.nan) * KG,
                                  AUC=sr.get("AUC_climate", np.nan), MMD2=sr.get("MMD2", np.nan), CORAL=sr.get("CORAL", np.nan), calendar_offset_days=sr.get("shift_days", np.nan),
                                  yield_CV_pct=100 * Y[Y.District == tgt].Yield_kg_ha.std() / Y[Y.District == tgt].Yield_kg_ha.mean(), n_target_years=int(Y[Y.District == tgt].Year.nunique()),
                                  mean_yield_kg_ha=float(Y[Y.District == tgt].Yield_kg_ha.mean())))
    pairs, gran = pd.DataFrame(pairs), pd.DataFrame(grows)
    pooled = pool_pairs(pairs, cfg)
    suit = suitability(gran)
    meta = dict(source=source, targets=targets, n_targets=len(targets), features=feats, bootstrap_n=cfg.bootstrap_n, note="MADA cannot be its own target: the sweep has one fewer target than granaries.")
    return dict(pairs=pairs, pooled=pooled, granary=gran, suitability=suit, meta=meta)


def pool_pairs(pairs: pd.DataFrame, cfg: Config = None):
    """pooled paired error differences per method and strategy, with bootstrap CI, Wilcoxon and the pass flag"""
    cfg = cfg or Config()
    rows = []
    for (strat, m), d in pairs.groupby(["strategy", "method"]):
        for tag, col in (("Delta_NT", "err_target_only"), ("Delta_DS", "err_source_only"), ("vs persistence", "err_persistence")):
            dd = (d.err_transfer - d[col]).dropna()
            if len(dd) < 6:
                continue
            idx = dd.index
            lo, hi = _boot_mean(dd.values, cfg.bootstrap_n, cfg.random_state)
            clo, chi = _boot_mean(dd.values, cfg.bootstrap_n, cfg.random_state, cluster=d.loc[idx, "granary"].values)
            p2, p1 = _wilcoxon(dd.values)
            rel = 100 * dd.mean() / d.loc[idx, col].mean() if d.loc[idx, col].mean() else np.nan
            rows.append(dict(strategy=strat, method=m, comparison=tag, n_pairs=len(dd), n_granaries=int(d.loc[idx, "granary"].nunique()), mean_diff_kg_ha=float(dd.mean()),
                             relative_pct=float(rel), ci95_low=lo, ci95_high=hi, cluster_ci95_low=clo, cluster_ci95_high=chi, wilcoxon_p_two_sided=p2, wilcoxon_p_less=p1,
                             pct_pairs_transfer_better=float(100 * (dd < 0).mean()),
                             passed=bool(tag == "Delta_NT" and hi < 0 and (p1 == p1) and p1 < 0.05)))
    return pd.DataFrame(rows)


def suitability(gran: pd.DataFrame, strategy="loso"):
    """Spearman rank correlation of the per-granary Delta_NT (relative to the target-only RMSE) with each descriptor, per method"""
    rows = []
    g = gran[gran.strategy == strategy]
    for m, d in g.groupby("method"):
        d = d.copy()
        d["rel"] = 100 * d.Delta_NT_vs_target_only / d.RMSE_target_only
        d["abs_offset"] = d.calendar_offset_days.abs()
        for col, lab in (("AUC", "domain AUC"), ("MMD2", "MMD2"), ("CORAL", "CORAL distance"), ("abs_offset", "calendar offset (|days|)"),
                         ("yield_CV_pct", "historical yield CV (%)"), ("n_target_years", "target years nt")):
            x = d[[col, "rel"]].dropna()
            if len(x) >= 5 and x[col].nunique() > 1:
                rho, p = spearmanr(x[col], x["rel"])
                rows.append(dict(method=m, descriptor=lab, n_granaries=len(x), spearman_rho=float(rho), p_value=float(p)))
    return pd.DataFrame(rows)


def qualification(sweep: dict, target: str, method: str, strategy="loso"):
    """the Verify step: is `method` qualified as a MADA-style transfer source for `target`? Returns the two parts of the rule and the evidence."""
    out = dict(method=method, target=target, sweep_passed=False, sweep_detail="the benchmark sweep has not been run for this method", granary_row=None, position=None)
    if not sweep or sweep.get("pooled") is None or sweep["pooled"].empty:
        return out
    p = sweep["pooled"]
    r = p[(p.strategy == strategy) & (p.method == method) & (p.comparison == "Delta_NT")]
    if len(r):
        r = r.iloc[0]
        out["sweep_passed"] = bool(r.passed)
        out["sweep_detail"] = (f"pooled Delta_NT {r.mean_diff_kg_ha:+.0f} kg/ha over {int(r.n_pairs)} granary-years in {int(r.n_granaries)} granaries "
                               f"(bootstrap 95% CI {r.ci95_low:+.0f} to {r.ci95_high:+.0f}; Wilcoxon one-sided p = {r.wilcoxon_p_less:.3f})")
    g = sweep["granary"]
    g = g[(g.strategy == strategy) & (g.method == method)]
    if len(g):
        g = g.sort_values("Delta_NT_vs_target_only").reset_index(drop=True)
        row = g[g.granary == target]
        if len(row):
            out["granary_row"] = row.iloc[0].to_dict()
            out["position"] = (int(row.index[0]) + 1, len(g))
    return out
