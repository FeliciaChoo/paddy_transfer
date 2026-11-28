"""Offline concordance replay + pooled cross-granary evaluation of the transferred recommendations.

    python pipeline/validate_concordance_uplift.py                 # full run (2,000 cluster-bootstrap draws, 500 placebo permutations)
    python pipeline/validate_concordance_uplift.py --quick         # 200 draws, 100 permutations, three fast methods (what the tests use)
    python pipeline/validate_concordance_uplift.py --automl        # also score the model the AutoML selected per granary (optimistic: it was chosen on the same seasons)

Writes results/concordance_uplift_report.json and results/concordance_uplift_summary.csv.

WHAT THIS IS, AND IS NOT
  It replays history: for every held-out granary-season it asks what the recommendation would have been using ONLY earlier seasons, whether what the farmers actually did matched
  it ("concordant"), and whether concordant seasons yielded more than discordant ones once weather is taken out. That is an ASSOCIATION in observational data, not a field trial:
  management is not randomised, so concordant seasons may differ in ways nothing here measures. With 7 held-out seasons per granary the intervals are wide. The four acceptance
  assertions of the protocol are REPORTED as findings (holds / statistically supported), never enforced, so the run can disprove the claim it tests.

DECISIONS THAT DIFFER FROM THE WRITTEN PROTOCOL (each one checked against the code and data, see the "deviations" block of the report)
  * causal_adapt.optimize_target_prescriptions is not called: given a real HarmonisedBundle it returns an empty list (it reads bundle.X_target / y_target / feature_names, which
    the bundle does not have), and its "schedule bounds" read columns the schedule file does not contain (start_decade; the file has Water_Start_Decade). Its levers and steps
    (-1 decade of planting slip, +0.05 LSWI_rep) are reused here; the decision to apply a step comes from the training-fold treatment-effect estimate.
  * The gazetted water window cannot be a hard bound on the green-up decade: for most granaries the satellite green-up falls well before the window opens. Steps are bounded by what
    the granary's own earlier seasons support (training range +/- one step) and the report states how many seasons fall inside the gazetted window.
  * Compatibility is decided by the domain screen (median AUC of the other granaries against the target >= 0.90), not by naming KETARA.
  * The existing refutation flags are lenient (the placebo "passes" if |effect| < max(|tau|, 150 kg/ha)), so a permutation placebo test is run here.
  * Weather adjustment includes rainfall (C1_Rain_mm_rep); the legacy confounder list used by causal_id has none.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):      # before numpy is imported: thousands of tiny fits run faster on one BLAS thread
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from acts_transfer.config import Config  # noqa: E402
from acts_transfer.sweep import clean_physical_ranges  # noqa: E402

OUTCOME = "Yield_kg_ha"
LEVERS = {
    "planting": dict(col="Schedule_Slip_Decades", step=-1.0, tol=1.0, label="planting timing (decades from the gazetted water start)"),
    "moisture": dict(col="C1_LSWI_rep", step=0.05, tol=0.04, label="reproductive-stage moisture (LSWI)"),
}
WEATHER_X = ["C1_Rain_mm_rep", "C1_Tmax_C_rep", "C1_Mean_Temp_C_rep", "C1_Solar_MJ_rip", "Soil_pH"]
HETERO = ["C1_Rain_mm_rep", "C1_Tmax_C_rep"]          # the treatment effect may vary with these two only (more would be fitted to a handful of rows)
AUC_EXCLUDE = 0.90
STRESS_TAIL = 0.20                                    # the most extreme 20% of seasons (by |z| within the granary) in rainfall or in temperature
LAM0 = 0.0                                            # ridge penalty on the AVERAGE effect (standardised units). The main analysis is the plain R-learner (0); a shrunk version
                                                      # (5) is run as a sensitivity analysis, because the target-only estimate fitted on 6-7 seasons is unstable
TARGET_BOOST = 2.0                                    # weight of the target's own earlier seasons against one donor row (fixed in advance, not tuned)
TRANSFER_METHODS = ["detrended_residual_transfer", "granary_intercept_ridge", "kmm_ridge", "coral", "two_stage_tradaboost_r2", "delta_two_stage_tradaboost_r2", "z_anomaly_transfer"]
QUICK_METHODS = ["detrended_residual_transfer", "granary_intercept_ridge", "kmm_ridge"]
SIMPLE_BASELINES = ["persistence", "target_only", "season_trend"]
REFERENCES = [
    "Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108(2), 299-319.",
    "Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. Ecography, 40(8), 913-929.",
    "Pardoe, D., & Stone, P. (2010). Boosting for regression transfer. ICML, 863-870.",
    "Kakimoto, S., Mieno, T., Tanaka, T. S. T., & Bullock, D. S. (2022). Causal forest approach for site-specific input management via on-farm precision experimentation. "
    "Computers and Electronics in Agriculture, 199, 107164. (nitrogen-rate field experiments: relevant as method background, not as evidence for this panel)",
    "Ma, Y., Chen, S., Ermon, S., & Lobell, D. B. (2024). Transfer learning in environmental remote sensing. Remote Sensing of Environment, 301, 113924.",
    "Ben-David, S., et al. (2010). A theory of learning from different domains. Machine Learning, 79(1-2), 151-175. (not re-checked in this run)",
]
REFERENCES_DROPPED = ["Thorburn, P. J., et al. (2024). 'Causal machine learning for actionable agronomic recommendations', Agricultural Systems: no such publication could be found, not cited."]


# ------------------------------------------------------------------------------------------------------------------------------ data and folds
def load_panel(root: Path = ROOT) -> pd.DataFrame:
    p = root / "data" / "transfer_stage_dataset.csv"
    Y, _ = clean_physical_ranges(pd.read_csv(p))
    return Y.dropna(subset=[OUTCOME]).sort_values(["District", "Year"]).reset_index(drop=True)


def domain_compatibility(root: Path = ROOT, cfg: Config | None = None) -> pd.DataFrame:
    """per granary taken as the target: median / max domain AUC against the others and the number of donors the screen flags. Compatible when the median AUC < AUC_EXCLUDE."""
    from acts_transfer.screening import screen_granaries
    cfg = cfg or Config()
    G = pd.read_csv(root / "data" / "transfer_decade_features.csv")
    rows = []
    for t in sorted(G["District"].unique()):
        s = screen_granaries(G, t, cfg)
        rows.append(dict(District=t, median_auc=float(s["AUC_climate"].median()), max_auc=float(s["AUC_climate"].max()),
                         donors_flagged=int(s["flagged"].sum()), n_donors=int(len(s))))
    d = pd.DataFrame(rows)
    d["compatible"] = d["median_auc"] < AUC_EXCLUDE
    return d


def make_folds(Y: pd.DataFrame, granaries: list[str], cfg: Config) -> list[dict]:
    """Forward expanding-window folds, one per held-out granary-season (the same folds evaluation.evaluate uses). Every training row of every granary is from a STRICTLY EARLIER
    season than the held-out one (source_time_safe): same-season rows of other granaries share that season's weather and are not used."""
    from acts_transfer.evaluation import _folds
    folds = []
    for g in granaries:
        years = sorted(Y.loc[Y.District == g, "Year"].unique().tolist())
        for y, train_years in _folds(years, cfg):
            tr = Y[(Y.District.isin(granaries)) & (Y.Year < y)]
            te = Y[(Y.District == g) & (Y.Year == y)]
            folds.append(dict(District=g, Year=int(y), train_target_years=[int(t) for t in train_years], train_index=tr.index.to_numpy(), test_index=te.index.to_numpy()))
    return folds


def audit_leakage(folds: list[dict], Y: pd.DataFrame) -> dict:
    """no training row (target or donor) may come from the held-out season or later, and the held-out row may not be in its own training set"""
    bad = []
    for f in folds:
        tr = Y.loc[f["train_index"]]
        if len(tr) and int(tr.Year.max()) >= f["Year"]:
            bad.append(dict(District=f["District"], Year=f["Year"], reason="training rows from the held-out season or later"))
        if set(f["test_index"]) & set(f["train_index"]):
            bad.append(dict(District=f["District"], Year=f["Year"], reason="held-out row inside its own training set"))
        if f["train_target_years"] and max(f["train_target_years"]) >= f["Year"]:
            bad.append(dict(District=f["District"], Year=f["Year"], reason="target training season not earlier than the held-out season"))
    return dict(n_folds=len(folds), source_time_safe=not bad, violations=bad[:20])


# ------------------------------------------------------------------------------------------------------------------------------ the estimator
class PartialLinear:
    """Within-granary partially linear model  Y = tau(H) * T + g(X) + e  (Robinson 1988 residualisation; R-learner of Nie & Wager 2021 with a CATE linear in H).
    Nuisances m(X) = E[Y|X] and e(X) = E[T|X] are ridge regressions on within-granary-centred, standardised X, cross-fitted for the residuals. The granary mean is the intercept,
    so the comparison is between seasons of the same granary. Rows can carry weights (the transfer-weighted fit)."""

    def __init__(self, tcol, xcols=None, hcols=None, ycol=OUTCOME, alpha=5.0, lam_h=20.0, lam0=LAM0, k=4, seed=0):
        self.tcol, self.xcols, self.hcols, self.ycol = tcol, list(xcols or WEATHER_X), list(hcols or HETERO), ycol
        self.alpha, self.lam_h, self.lam0, self.k, self.seed = alpha, lam_h, lam0, k, seed
        self.ok = False

    def _xmat(self, rows: pd.DataFrame) -> np.ndarray:
        return rows[self.xcols].fillna(self.med).fillna(0.0).to_numpy(float)

    def _centre(self, rows: pd.DataFrame, X: np.ndarray) -> np.ndarray:
        mu = np.vstack([self.mu_x.get(g, self.mu_x_all) for g in rows["District"]])
        return (X - mu) / self.sx

    def fit(self, df: pd.DataFrame, w: np.ndarray | None = None):
        d = df.dropna(subset=[self.tcol, self.ycol]).copy()
        w = np.ones(len(df)) if w is None else np.asarray(w, float)
        if w.shape[0] == len(df):
            w = pd.Series(w, index=df.index).loc[d.index].to_numpy()
        self.med = d[self.xcols].median()
        self.mu_x_all = None
        if len(d) < 6:
            return self
        X = self._xmat(d)
        g = d["District"].to_numpy()
        T, Yv = d[self.tcol].to_numpy(float), d[self.ycol].to_numpy(float)
        self.mu_x = {k: X[g == k].mean(0) for k in np.unique(g)}
        self.mu_t = {k: T[g == k].mean() for k in np.unique(g)}
        self.mu_y = {k: Yv[g == k].mean() for k in np.unique(g)}
        self.mu_x_all, self.mu_t_all, self.mu_y_all = X.mean(0), float(T.mean()), float(Yv.mean())
        Xd = X - np.vstack([self.mu_x[k] for k in g])
        self.sx = Xd.std(0) + 1e-9
        Xs = Xd / self.sx
        Td = T - np.array([self.mu_t[k] for k in g])
        Yd = Yv - np.array([self.mu_y[k] for k in g])
        ry, rt = np.zeros(len(d)), np.zeros(len(d))
        for tr, va in KFold(min(self.k, max(2, len(d) // 3)), shuffle=True, random_state=self.seed).split(Xs):
            ry[va] = Yd[va] - Ridge(alpha=self.alpha).fit(Xs[tr], Yd[tr], sample_weight=w[tr]).predict(Xs[va])
            rt[va] = Td[va] - Ridge(alpha=self.alpha).fit(Xs[tr], Td[tr], sample_weight=w[tr]).predict(Xs[va])
        self.by = Ridge(alpha=self.alpha).fit(Xs, Yd, sample_weight=w)
        self.bt = Ridge(alpha=self.alpha).fit(Xs, Td, sample_weight=w)
        self.hidx = [self.xcols.index(h) for h in self.hcols if h in self.xcols]
        self.sd_t = float(rt.std()) + 1e-12
        Z = np.column_stack([rt / self.sd_t] + [(rt / self.sd_t) * Xs[:, j] for j in self.hidx])
        A = Z.T @ (w[:, None] * Z) + np.diag([self.lam0] + [self.lam_h] * len(self.hidx))
        self.theta = np.linalg.solve(A, Z.T @ (w * ry))
        self.ok = bool(np.sum(w * rt ** 2) > 1e-10 and np.all(np.isfinite(self.theta)))
        return self

    def y_hat(self, rows: pd.DataFrame) -> np.ndarray:
        Xs = self._centre(rows, self._xmat(rows))
        return np.array([self.mu_y.get(g, self.mu_y_all) for g in rows["District"]]) + self.by.predict(Xs)

    def t_hat(self, rows: pd.DataFrame) -> np.ndarray:
        Xs = self._centre(rows, self._xmat(rows))
        return np.array([self.mu_t.get(g, self.mu_t_all) for g in rows["District"]]) + self.bt.predict(Xs)

    def tau(self, rows: pd.DataFrame) -> np.ndarray:
        """treatment effect in yield units per one unit of the treatment, for each row (0 when the model could not be fitted)"""
        if not self.ok:
            return np.zeros(len(rows))
        Xs = self._centre(rows, self._xmat(rows))
        return (self.theta[0] + Xs[:, self.hidx] @ self.theta[1:]) / self.sd_t

    def tau0(self) -> float:
        return float(self.theta[0] / self.sd_t) if self.ok else 0.0


def domain_weights(tr: pd.DataFrame, target: str, xcols=None) -> np.ndarray:
    """importance weights of the training rows: the target's own rows count TARGET_BOOST; a donor row counts the odds that it looks like the target (domain classifier on the
    weather columns, clipped to 0.1-1). With fewer than 4 target rows the classifier would be fitted to almost nothing, so every donor row counts 0.5 instead."""
    xcols = list(xcols or WEATHER_X)
    is_t = (tr["District"] == target).to_numpy()
    w = np.where(is_t, TARGET_BOOST, 0.5)
    if is_t.sum() < 4 or (~is_t).sum() < 4:
        return w
    X = tr[xcols].fillna(tr[xcols].median()).fillna(0.0).to_numpy(float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    p = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500).fit(X, is_t.astype(int)).predict_proba(X)[:, 1]
    w[~is_t] = np.clip(p[~is_t] / np.maximum(1 - p[~is_t], 1e-6), 0.1, 1.0)
    return w


# ------------------------------------------------------------------------------------------------------------------------------ replay
def recommend_target(hist: pd.Series, step: float, tau0: float, fitted: bool, tbase: str = "last") -> tuple[float, float, bool]:
    """(T_base, T_rec, step applied) from the granary's EARLIER values only. T_base is the last observed value (or their median). The project's step is recommended only when the
    treatment-effect estimate says it raises yield (tau0 * step > 0), and the target is kept inside what the granary's earlier seasons support, plus one step."""
    t_base = float(hist.iloc[-1]) if tbase == "last" else float(hist.median())
    applied = bool(fitted and tau0 * step > 0)
    lo = float(hist.min()) + (step if step < 0 else 0.0)
    hi = float(hist.max()) + (step if step > 0 else 0.0)
    t_rec = float(np.clip(t_base + (step if applied else 0.0), min(lo, t_base), max(hi, t_base)))
    return t_base, t_rec, applied


def is_concordant(t_obs, t_rec, tol: float) -> float:
    """1 when the management actually observed lies within the tolerance of the recommended target, 0 when not, NaN when it was not observed"""
    if t_obs is None or pd.isna(t_obs):
        return float("nan")
    return float(abs(float(t_obs) - float(t_rec)) <= tol + 1e-12)


def replay(Y: pd.DataFrame, folds: list[dict], tbase: str = "last", lam0: float = LAM0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One pass over the folds. Returns (seasons, rloss): `seasons` has one row per held-out granary-season with the recommendation, concordance and the three outcome scales;
    `rloss` has one row per season, lever and regime with the Robinson R-loss. Everything is fitted on the training rows of that fold only."""
    seasons, losses = [], []
    for f in folds:
        tr = Y.loc[f["train_index"]]
        te = Y.loc[f["test_index"]]
        if te.empty:
            continue
        g, y = f["District"], f["Year"]
        tg = tr[tr.District == g]
        row = te.iloc[0]
        base = dict(District=g, Year=y, yield_kg_ha=float(row[OUTCOME]), n_target_train=len(tg), n_train=len(tr))
        w = domain_weights(tr, g)
        models = {}
        for lev, spec in LEVERS.items():
            c = spec["col"]
            m_tr = PartialLinear(c, lam0=lam0, seed=y).fit(tr, w)                                          # transfer-weighted
            m_t = PartialLinear(c, lam0=lam0, seed=y).fit(tg)                                              # target only
            m_s = PartialLinear(c, lam0=lam0, seed=y).fit(tr[tr.District != g])                            # donors only
            models[lev] = (m_tr, m_t, m_s)
        # weather-adjusted outcome: expected yield given this season's weather and the granary's history (from the transfer-weighted nuisance), treatment not included
        ref = models["planting"][0]
        if ref.ok and len(tg):
            base["y_expected"] = float(ref.y_hat(te)[0])
            base["y_res"] = base["yield_kg_ha"] - base["y_expected"]
        else:
            base["y_expected"] = base["y_res"] = np.nan
        base["y_gdm"] = base["yield_kg_ha"] - float(tg[OUTCOME].mean()) if len(tg) else np.nan
        for lev, spec in LEVERS.items():
            c, step, tol = spec["col"], spec["step"], spec["tol"]
            m_tr, m_t, m_s = models[lev]
            t_obs = row[c]
            hist = tg[c].dropna()
            if len(hist) == 0:
                base.update({f"{lev}_obs": t_obs, f"{lev}_base": np.nan, f"{lev}_rec": np.nan, f"{lev}_step_applied": np.nan, f"{lev}_concordant": np.nan})
                continue
            t_base, t_rec, applied = recommend_target(hist, step, m_tr.tau0(), m_tr.ok, tbase)
            tau0 = m_tr.tau0()
            base.update({f"{lev}_obs": float(t_obs) if pd.notna(t_obs) else np.nan, f"{lev}_base": t_base, f"{lev}_rec": t_rec, f"{lev}_tau0": tau0,
                         f"{lev}_step_applied": float(step if applied else 0.0),
                         f"{lev}_concordant": is_concordant(t_obs, t_rec, tol)})
            # R-loss on the held-out season; the nuisances are common (the transfer-weighted ones) so only the effect estimate differs between regimes
            if m_tr.ok and pd.notna(t_obs) and len(tg):
                r_y = float(row[OUTCOME] - m_tr.y_hat(te)[0])
                r_t = float(t_obs - m_tr.t_hat(te)[0])
                for name, mdl in (("transfer_weighted", m_tr), ("target_only", m_t), ("source_only", m_s)):
                    tau = float(mdl.tau(te)[0])
                    losses.append(dict(District=g, Year=y, lever=lev, regime=name, loss=(r_y - tau * r_t) ** 2, fitted=bool(mdl.ok)))
                losses.append(dict(District=g, Year=y, lever=lev, regime="no_effect (tau=0)", loss=r_y ** 2, fitted=True))
        seasons.append(base)
    S = pd.DataFrame(seasons)
    if len(S):
        S["concordant_both"] = np.where(S[["planting_concordant", "moisture_concordant"]].isna().any(axis=1), np.nan, S["planting_concordant"] * S["moisture_concordant"])
    return S, pd.DataFrame(losses)


# ------------------------------------------------------------------------------------------------------------------------------ cluster bootstrap (by granary)
class ClusterBoot:
    """Resamples whole granaries with replacement. Each statistic is a ratio of per-granary sums, so all draws are computed at once from a draws x granaries multiplicity matrix."""

    def __init__(self, clusters: np.ndarray, B: int = 2000, seed: int = 0):
        self.names = sorted(pd.unique(clusters))
        self.idx = {c: np.where(clusters == c)[0] for c in self.names}
        G = len(self.names)
        rng = np.random.default_rng(seed)
        self.M = np.vstack([np.ones(G), rng.multinomial(G, np.full(G, 1.0 / G), size=B)])               # row 0 = the observed sample

    def sums(self, v: np.ndarray, mask: np.ndarray | None = None):
        v = np.asarray(v, float)
        ok = np.isfinite(v) if mask is None else (np.isfinite(v) & mask)
        s = np.array([np.nansum(np.where(ok[self.idx[c]], v[self.idx[c]], 0.0)) for c in self.names])
        n = np.array([ok[self.idx[c]].sum() for c in self.names], float)
        return s, n

    @staticmethod
    def _ci(x: np.ndarray) -> dict:
        d = x[1:][np.isfinite(x[1:])]
        est = float(x[0]) if np.isfinite(x[0]) else float("nan")
        if len(d) < 20:
            return dict(estimate=est, ci_low=float("nan"), ci_high=float("nan"), n_valid_draws=int(len(d)))
        lo, hi = np.percentile(d, [2.5, 97.5])
        return dict(estimate=est, ci_low=float(lo), ci_high=float(hi), n_valid_draws=int(len(d)))

    def mean(self, v, mask=None):
        s, n = self.sums(v, mask)
        with np.errstate(invalid="ignore", divide="ignore"):
            return self._ci((self.M @ s) / (self.M @ n))

    def mean_diff(self, v, group, mask=None):
        """mean of v where group==1 minus mean where group==0"""
        g = np.asarray(group, float)
        s1, n1 = self.sums(v, (g == 1) if mask is None else (g == 1) & mask)
        s0, n0 = self.sums(v, (g == 0) if mask is None else (g == 0) & mask)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = self._ci((self.M @ s1) / (self.M @ n1) - (self.M @ s0) / (self.M @ n0))
        out.update(n_group1=int(n1.sum()), n_group0=int(n0.sum()))
        return out

    def rmse_gain(self, e_base, e_new, mask=None):
        """RMSE(base) - RMSE(new) for paired error vectors: positive means the new method is better"""
        ok = np.isfinite(e_base) & np.isfinite(e_new) & (True if mask is None else mask)
        sb, n = self.sums(np.asarray(e_base, float) ** 2, ok)
        sn, _ = self.sums(np.asarray(e_new, float) ** 2, ok)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = self._ci(np.sqrt((self.M @ sb) / (self.M @ n)) - np.sqrt((self.M @ sn) / (self.M @ n)))
        out["n"] = int(n.sum())
        return out

    def mae_gain(self, e_base, e_new, mask=None):
        ok = np.isfinite(e_base) & np.isfinite(e_new) & (True if mask is None else mask)
        sb, n = self.sums(np.abs(e_base), ok)
        sn, _ = self.sums(np.abs(e_new), ok)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = self._ci((self.M @ sb) / (self.M @ n) - (self.M @ sn) / (self.M @ n))
        out["n"] = int(n.sum())
        return out

    def win_rate(self, e_base, e_new, mask=None):
        ok = np.isfinite(e_base) & np.isfinite(e_new) & (True if mask is None else mask)
        win = (np.abs(e_new) < np.abs(e_base)).astype(float)
        s, n = self.sums(win, ok)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = self._ci(100.0 * (self.M @ s) / (self.M @ n))
        out["n"] = int(n.sum())
        return out


# ------------------------------------------------------------------------------------------------------------------------------ refutation
def _dml_tau(df: pd.DataFrame, tcol: str, xcols: list[str], seed: int = 0, T_override: np.ndarray | None = None, extra_x: np.ndarray | None = None) -> float:
    """cross-fitted within-granary DML slope of yield on the treatment (kg/ha per unit); ridge nuisances, so it is fast enough for hundreds of permutations"""
    d = df.dropna(subset=[tcol, OUTCOME])
    X = d[xcols].fillna(d[xcols].median()).fillna(0.0).to_numpy(float)
    if extra_x is not None:
        X = np.column_stack([X, extra_x])
    g = d["District"].to_numpy()
    T = d[tcol].to_numpy(float) if T_override is None else T_override
    Yv = d[OUTCOME].to_numpy(float)
    cen = lambda a: a - np.vstack([a[g == k].mean(0) for k in g]) if a.ndim == 2 else a - np.array([a[g == k].mean() for k in g])      # noqa: E731
    Xd, Td, Yd = cen(X), cen(T), cen(Yv)
    Xs = Xd / (Xd.std(0) + 1e-9)
    ry, rt = np.zeros(len(d)), np.zeros(len(d))
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(Xs):
        ry[va] = Yd[va] - Ridge(alpha=5.0).fit(Xs[tr], Yd[tr]).predict(Xs[va])
        rt[va] = Td[va] - Ridge(alpha=5.0).fit(Xs[tr], Td[tr]).predict(Xs[va])
    den = float(rt @ rt)
    return float(rt @ ry / den) if den > 1e-12 else 0.0


def refute(Y: pd.DataFrame, lever: str, n_perm: int, seed: int = 0) -> dict:
    spec = LEVERS[lever]
    tcol = spec["col"]
    xcols = [c for c in WEATHER_X if c in Y.columns]
    d = Y.dropna(subset=[tcol, OUTCOME]).reset_index(drop=True)
    tau = _dml_tau(d, tcol, xcols, seed)
    rng = np.random.default_rng(seed + 11)
    g = d["District"].to_numpy()
    T = d[tcol].to_numpy(float)
    perm = []
    for _ in range(n_perm):                                                  # placebo: the treatment shuffled WITHIN each granary, which breaks its link to that granary's yields
        Tp = T.copy()
        for k in np.unique(g):
            ii = np.where(g == k)[0]
            Tp[ii] = rng.permutation(T[ii])
        perm.append(_dml_tau(d, tcol, xcols, seed, T_override=Tp))
    perm = np.array(perm)
    rcc = np.array([_dml_tau(d, tcol, xcols, seed, extra_x=rng.normal(size=len(d))) for _ in range(max(20, n_perm // 3))])
    sub = []
    gs = np.unique(g)
    for _ in range(max(20, n_perm // 3)):                                    # 80% of the granaries
        keep = rng.choice(gs, size=max(2, int(round(0.8 * len(gs)))), replace=False)
        s = d[d.District.isin(keep)].reset_index(drop=True)
        sub.append(_dml_tau(s, tcol, xcols, seed))
    sub = np.array(sub)
    plo, phi = np.percentile(perm, [2.5, 97.5])
    return dict(lever=lever, treatment=tcol, n=len(d), tau_per_unit=tau,
                placebo=dict(n_perm=int(n_perm), mean=float(perm.mean()), sd=float(perm.std()), ci95_low=float(plo), ci95_high=float(phi), interval_contains_zero=bool(plo <= 0 <= phi),
                             p_value_two_sided=float((1 + np.sum(np.abs(perm) >= abs(tau))) / (1 + len(perm))),
                             note="the placebo is centred on zero by construction; the informative number is whether the real estimate stands out from it (p_value)"),
                random_common_cause=dict(mean=float(rcc.mean()), sd=float(rcc.std()), max_abs_shift=float(np.max(np.abs(rcc - tau))), stable=bool(np.max(np.abs(rcc - tau)) <= 0.25 * abs(tau) + 1e-9)),
                subset_80pct_granaries=dict(mean=float(sub.mean()), min=float(sub.min()), max=float(sub.max()), same_sign=bool(np.all(np.sign(sub) == np.sign(tau)))))


def legacy_dml(Y: pd.DataFrame) -> list[dict]:
    """the project's own GBR double-ML estimate and its four refutation flags, for comparison (a single run each; its placebo flag is lenient by construction)"""
    from acts_transfer.causal_id import dml_estimate_and_refute
    out = []
    for lev, spec in LEVERS.items():
        try:
            r = dml_estimate_and_refute(Y, treatment_col=spec["col"])
            out.append(dict(lever=lev, tau=float(r["tau"]), se=None if r.get("se") is None or not np.isfinite(r.get("se", np.nan)) else float(r["se"]), n_overlap=int(r.get("n_overlap", 0)),
                            confounders=r.get("confounders", []), flags={x["test"]: bool(x["passed"]) for x in r.get("refutations", [])},
                            note="confounders exclude rainfall; the placebo flag passes if |placebo effect| < max(|tau|, 150 kg/ha)"))
        except Exception as e:
            out.append(dict(lever=lev, error=f"{type(e).__name__}: {e}"))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ pooled forecast evaluation
def stress_flags(Y: pd.DataFrame) -> pd.Series:
    """climate stress/anomaly season: among the most extreme 20% by |z| (z within the granary) in rainfall or in temperature. Descriptive: uses every season of the panel."""
    z = pd.DataFrame(index=Y.index)
    for c in ("C1_Rain_mm_rep", "C1_Tmax_C_rep"):
        if c in Y.columns:
            g = Y.groupby("District")[c]
            z[c] = ((Y[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).abs()
    if z.empty:
        return pd.Series(False, index=Y.index)
    cut = z.quantile(1 - STRESS_TAIL)
    return (z.ge(cut)).any(axis=1)


def forecast_predictions(Y: pd.DataFrame, granaries: list[str], methods: list[str], cfg: Config) -> pd.DataFrame:
    """Pre-specified evaluation, no search: every other compatible granary is the donor pool, all stage features, ridge base. Forward folds (train strictly earlier)."""
    from acts_transfer.evaluation import BASELINES, evaluate
    from acts_transfer.sweep import build_bundle, stage_features
    feats = stage_features(Y)
    frames = []
    for g in granaries:
        donors = [d for d in granaries if d != g]
        bundle, _ = build_bundle(Y, source_names=donors, target=g, feats=feats)
        res = evaluate(bundle, list(methods) + list(BASELINES), "ridge", cfg)
        p = res.predictions.copy()
        p["District"], p["kg"] = g, float(getattr(bundle, "y_to_kg", 1.0))
        frames.append(p)
    P = pd.concat(frames, ignore_index=True)
    P["err"] = (P["y_pred"] - P["y_true"]) * P["kg"]
    return P.rename(columns={"group": "Year"})


def pooled_forecast_gains(P: pd.DataFrame, methods: list[str], stress: pd.DataFrame, B: int, seed: int) -> dict:
    out = {}
    wide = P.pivot_table(index=["District", "Year"], columns="method", values="err").reset_index()
    wide = wide.merge(stress, on=["District", "Year"], how="left")
    wide["stress"] = wide["stress"].fillna(False).astype(bool)
    cb = ClusterBoot(wide["District"].to_numpy(), B, seed)
    for m in methods:
        if m not in wide:
            continue
        out[m] = {}
        for b in SIMPLE_BASELINES + ["best_simple_per_granary"]:
            if b == "best_simple_per_granary":
                base = np.full(len(wide), np.nan)
                for g, idx in wide.groupby("District").groups.items():
                    sub = wide.loc[idx]
                    ok = sub[[m] + SIMPLE_BASELINES].dropna().index
                    if len(ok) < 3:
                        continue
                    best = min(SIMPLE_BASELINES, key=lambda k: float(np.sqrt((wide.loc[ok, k] ** 2).mean())))
                    base[ok] = wide.loc[ok, best].to_numpy()
                eb = base
            elif b in wide:
                eb = wide[b].to_numpy(float)
            else:
                continue
            en = wide[m].to_numpy(float)
            res = {"all": dict(rmse_gain_kg_ha=cb.rmse_gain(eb, en), mae_gain_kg_ha=cb.mae_gain(eb, en), season_win_rate_pct=cb.win_rate(eb, en))}
            for name, mask in (("normal_years", ~wide["stress"].to_numpy()), ("climate_stress_years", wide["stress"].to_numpy())):
                res[name] = dict(rmse_gain_kg_ha=cb.rmse_gain(eb, en, mask), mae_gain_kg_ha=cb.mae_gain(eb, en, mask), season_win_rate_pct=cb.win_rate(eb, en, mask))
            out[m][b] = res
    return out


# ------------------------------------------------------------------------------------------------------------------------------ run
def _num(x):
    if isinstance(x, (np.floating, np.integer)):
        return None if (isinstance(x, np.floating) and not np.isfinite(x)) else x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_clean(v) for v in o.tolist()]
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    return _num(o)


def excluded_by_request(included: list[str]) -> list[str]:
    """granaries the written protocol names for exclusion (KETARA); run as a sensitivity analysis when the domain screen keeps them"""
    return [g for g in ("KETARA",) if g in included]


def _core(Yi: pd.DataFrame, included: list[str], cfg: Config, B: int, seed: int, tbase: str, lam0: float, light: bool = False) -> dict:
    """folds -> leakage audit -> replay -> concordance uplift and R-loss, for one set of granaries"""
    folds = make_folds(Yi, included, cfg)
    audit = audit_leakage(folds, Yi)
    S, R = replay(Yi, folds, tbase, lam0)
    S2 = replay(Yi, folds, "median", lam0)[0] if (tbase == "last" and not light) else None
    S = S.merge(stress_flags(Yi).rename("stress").to_frame().join(Yi[["District", "Year"]]), on=["District", "Year"], how="left")
    S["stress"] = S["stress"].fillna(False).astype(bool)
    cb = ClusterBoot(S["District"].to_numpy(), B, seed)
    conc = dict(n_seasons=int(len(S)), n_granaries=int(S.District.nunique()))
    for key, col in (("both_levers", "concordant_both"), ("planting_only", "planting_concordant"), ("moisture_only", "moisture_concordant")):
        c = S[col].to_numpy(float)
        n1, n0 = int(np.nansum(c == 1)), int(np.nansum(c == 0))
        entry = dict(n_concordant=n1, n_discordant=n0, share_concordant=float(n1 / max(n1 + n0, 1)))
        for oname, ocol in (("raw_yield_kg_ha", "yield_kg_ha"), ("granary_adjusted_kg_ha", "y_gdm"), ("weather_adjusted_kg_ha", "y_res")):
            entry[oname] = cb.mean_diff(S[ocol].to_numpy(float), c)
        for name, mask in (("normal_years", ~S["stress"].to_numpy()), ("climate_stress_years", S["stress"].to_numpy())):
            entry[f"weather_adjusted_{name}_kg_ha"] = cb.mean_diff(S["y_res"].to_numpy(float), c, mask)
        conc[key] = entry
    if S2 is not None and len(S2):
        c2 = S2["concordant_both"].to_numpy(float)
        cb2 = ClusterBoot(S2["District"].to_numpy(), B, seed)
        conc["sensitivity_baseline_is_training_median"] = dict(n_concordant=int(np.nansum(c2 == 1)), n_discordant=int(np.nansum(c2 == 0)),
                                                              weather_adjusted_kg_ha=cb2.mean_diff(S2["y_res"].to_numpy(float), c2))
    conc["recommendation_applied_share"] = {lev: float(np.nanmean((S[f"{lev}_step_applied"] != 0).astype(float))) for lev in LEVERS}
    conc["interpretation"] = ("Association in observational data, not a causal effect: management was not randomised. The recommendation is a small step from the last observed value, "
                              "so 'concordant' mostly means 'management stayed near its recent level'.")
    rl = {}
    if len(R):
        for lev in LEVERS:
            sub = R[R.lever == lev]
            piv = sub.pivot_table(index=["District", "Year"], columns="regime", values="loss").reset_index().dropna()
            if piv.empty:
                continue
            cbl = ClusterBoot(piv["District"].to_numpy(), B, seed)
            e = {"n_seasons": int(len(piv))}
            for reg in ("transfer_weighted", "target_only", "source_only", "no_effect (tau=0)"):
                e[f"mean_rloss_{reg}"] = cbl.mean(piv[reg].to_numpy(float))
                e[f"median_rloss_{reg}"] = float(piv[reg].median())
            for other in ("target_only", "source_only", "no_effect (tau=0)"):
                e[f"transfer_minus_{other}"] = cbl.mean(piv["transfer_weighted"].to_numpy(float) - piv[other].to_numpy(float))
                e[f"median_paired_difference_vs_{other}"] = float(np.median(piv["transfer_weighted"].to_numpy(float) - piv[other].to_numpy(float)))
                e[f"seasons_where_transfer_has_lower_loss_than_{other}_pct"] = cbl.win_rate(np.sqrt(piv[other].to_numpy(float)), np.sqrt(piv["transfer_weighted"].to_numpy(float)))
            e["share_target_only_fitted"] = float(sub[sub.regime == "target_only"]["fitted"].mean())
            rl[lev] = e
        rl["note"] = ("The three effect estimators are scored with the same nuisance models, so only the effect estimate differs. The target-only estimator is fitted on 2-7 seasons and is "
                      "often unfittable (then it equals the no-effect benchmark) or extrapolates wildly, so mean differences against it are driven by a few seasons: read the medians and win rates, "
                      "and treat the no-effect benchmark as the meaningful comparison. Negative difference = transfer is better.")
    return dict(folds=folds, audit=audit, S=S, S2=S2, conc=conc, rl=rl)


def run(root: Path = ROOT, out_dir: Path | None = None, quick: bool = False, boot: int | None = None, perm: int | None = None, seed: int = 0, methods: list[str] | None = None,
        tbase: str = "last", forecast: bool = True, automl: bool = False, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    B = boot or (200 if quick else 2000)
    n_perm = perm or (100 if quick else 500)
    methods = methods or (QUICK_METHODS if quick else TRANSFER_METHODS)
    log = (lambda m: print(f"[{time.perf_counter() - t0:6.1f}s] {m}", flush=True)) if verbose else (lambda m: None)
    cfg = Config(cv_strategy="forward")

    Y = load_panel(root)
    comp = domain_compatibility(root, cfg)
    included = sorted(comp.loc[comp.compatible, "District"].tolist())
    excluded = sorted(comp.loc[~comp.compatible, "District"].tolist())
    Yi = Y[Y.District.isin(included)].reset_index(drop=True)
    log(f"panel {len(Y)} rows, {Y.District.nunique()} granaries, {int(Y.Year.min())}-{int(Y.Year.max())}; compatible {len(included)}, excluded {excluded}")

    core = _core(Yi, included, cfg, B, seed, tbase, LAM0)
    folds, audit, S, S2, conc, rl = core["folds"], core["audit"], core["S"], core["S2"], core["conc"], core["rl"]
    log(f"{len(folds)} forward folds; leakage audit {'clean' if audit['source_time_safe'] else 'VIOLATIONS'}; replayed {len(S)} held-out granary-seasons; R-loss done")
    sensitivity = {}
    if not quick:
        for name, kw in (("effect_estimate_shrunk_lam0_5", dict(lam0=5.0)),):
            c2 = _core(Yi, included, cfg, B, seed, tbase, **kw, light=True)
            sensitivity[name] = dict(concordance_weather_adjusted_kg_ha=c2["conc"]["both_levers"]["weather_adjusted_kg_ha"], r_loss=c2["rl"],
                                     note="same analysis with a ridge penalty on the average treatment effect; a shrinkage change did not remove the target-only outliers (they come from "
                                          "held-out treatment values far outside the few training seasons), so the means below stay outlier-driven: read the medians and win rates")
        for g in excluded_by_request(included):
            keep = [x for x in included if x != g]
            c3 = _core(Y[Y.District.isin(keep)].reset_index(drop=True), keep, cfg, B, seed, tbase, LAM0, light=True)
            sensitivity[f"excluding_{g}"] = dict(held_out_seasons=int(len(c3["S"])), concordance=c3["conc"]["both_levers"], r_loss=c3["rl"],
                                                 note="the protocol excludes this granary as domain-incompatible; the screen does not flag it, so the main analysis keeps it")
        log("sensitivity analyses done")

    refs = {lev: refute(Yi, lev, n_perm, seed) for lev in LEVERS}
    log(f"refutation done ({n_perm} permutations)")
    legacy = legacy_dml(Yi)

    # gazetted window diagnostic
    window = {}
    sch = root / "external" / "granary_irrigation_schedules.csv"
    if sch.exists():
        c1 = pd.read_csv(sch)
        c1 = c1[c1.Cycle == "C1"].set_index("District")
        for g in included:
            if g in c1.index:
                s, e = int(c1.loc[g, "Water_Start_Decade"]), int(c1.loc[g, "Water_End_Decade"])
                gd = Yi.loc[Yi.District == g, "C1_Greenup_Decade"]
                window[g] = dict(window=[s, e], median_greenup_decade=float(gd.median()), share_of_seasons_inside_window=float(((gd >= s) & (gd <= e)).mean()))

    pooled, per_granary = {}, {}
    if forecast:
        log(f"pre-specified forecast evaluation: {methods}")
        P = forecast_predictions(Yi, included, methods, cfg)
        st = stress_flags(Yi).rename("stress").to_frame().join(Yi[["District", "Year"]])
        pooled = pooled_forecast_gains(P, methods, st, B, seed)
        for g, d in P[P.method.isin(methods + SIMPLE_BASELINES)].groupby("District"):
            w = d.pivot_table(index="Year", columns="method", values="err")
            per_granary[g] = {m: float(np.sqrt((w[m].dropna() ** 2).mean())) for m in w.columns}
        if automl:
            from acts_transfer.automl_runner import run_full_automl_for_granary
            mt = (root / "data" / "transfer_stage_dataset.csv").stat().st_mtime
            sel = []
            for g in included:
                a = run_full_automl_for_granary(g, str(root / "data"), mt, True)
                ch = a["choice"]
                m = ch["selected"] if ch.get("transfer_supported") else ch.get("best_transfer")
                pr = a["eval_results"]["forward"].predictions
                kg = float(a["bundle"].y_to_kg)
                w = pr.assign(err=(pr.y_pred - pr.y_true) * kg).rename(columns={"group": "Year"})
                w["District"] = g
                w["picked"] = m
                sel.append(w[w.method.isin([m] + SIMPLE_BASELINES)])
            if sel:
                Q = pd.concat(sel)
                wq = Q.pivot_table(index=["District", "Year"], columns="method", values="err")
                pick = Q.drop_duplicates("District").set_index("District")["picked"]
                sel_err = np.array([wq.loc[(g, y), pick[g]] if pick[g] in wq.columns else np.nan for g, y in wq.index])
                cbq = ClusterBoot(np.array([g for g, _ in wq.index]), B, seed)
                pooled["automl_selected_per_granary"] = {b: dict(rmse_gain_kg_ha=cbq.rmse_gain(wq[b].to_numpy(float), sel_err), mae_gain_kg_ha=cbq.mae_gain(wq[b].to_numpy(float), sel_err),
                                                                 season_win_rate_pct=cbq.win_rate(wq[b].to_numpy(float), sel_err)) for b in SIMPLE_BASELINES if b in wq}
                pooled["automl_selected_per_granary"]["note"] = "optimistic: the pool, regime and method were chosen on these same seasons"
        log("pooled evaluation done")

    # acceptance assertions: reported, never enforced
    acc = {}
    acc["a1_no_future_or_same_year_source_rows"] = dict(holds=bool(audit["source_time_safe"]), n_folds=audit["n_folds"])
    u = conc["both_levers"]["weather_adjusted_kg_ha"]
    acc["a2_concordant_seasons_have_higher_weather_adjusted_yield"] = dict(estimate_kg_ha=u["estimate"], ci95=[u["ci_low"], u["ci_high"]], n_concordant=conc["both_levers"]["n_concordant"],
                                                                         n_discordant=conc["both_levers"]["n_discordant"], holds=bool(u["estimate"] > 0) if u["estimate"] == u["estimate"] else False,
                                                                         statistically_supported=bool(u["ci_low"] > 0) if u["ci_low"] == u["ci_low"] else False)
    a3 = {}
    for lev in LEVERS:
        if lev in rl:
            d1, d2, d0 = rl[lev]["transfer_minus_target_only"], rl[lev]["transfer_minus_source_only"], rl[lev]["transfer_minus_no_effect (tau=0)"]
            a3[lev] = dict(vs_target_only=d1, vs_source_only=d2, vs_no_effect=d0, holds_vs_target_and_source=bool(d1["estimate"] < 0 and d2["estimate"] < 0),
                           statistically_supported=bool(d1["ci_high"] < 0 and d2["ci_high"] < 0), beats_no_effect_benchmark=bool(d0["estimate"] < 0))
    acc["a3_transfer_cate_has_lower_rloss_than_target_only_and_source_only"] = a3
    acc["a4_placebo_effect_indistinguishable_from_zero"] = {lev: dict(interval_contains_zero=refs[lev]["placebo"]["interval_contains_zero"], placebo_mean=refs[lev]["placebo"]["mean"],
                                                                      real_effect_p_value=refs[lev]["placebo"]["p_value_two_sided"]) for lev in LEVERS}
    acc["status"] = "a1 (integrity) is enforced by the tests; a2-a4 are reported as findings and are not enforced by the script or by the tests"

    report = dict(
        meta=dict(created=datetime.now().isoformat(timespec="seconds"), seconds=round(time.perf_counter() - t0, 1), quick=quick, bootstrap_draws=B, permutations=n_perm, seed=seed, tbase=tbase,
                  methods=methods, cv_strategy="forward", source_time_safe=True, tolerances={k: v["tol"] for k, v in LEVERS.items()}, steps={k: v["step"] for k, v in LEVERS.items()},
                  stress_definition=f"most extreme {int(STRESS_TAIL * 100)}% of seasons by |z| (within granary) in rainfall or temperature"),
        data=dict(rows=int(len(Y)), granaries=int(Y.District.nunique()), years=[int(Y.Year.min()), int(Y.Year.max())], included=included, excluded=excluded,
                  held_out_seasons=int(len(S)), protocol_claimed=dict(panel_rows=70, held_out_seasons=59), note="the protocol's N = 70 / 59 do not match the project data; the real counts are above"),
        domain_screen=comp.to_dict(orient="records"), leakage_audit=audit, concordance=conc, r_loss=rl, refutation=refs, legacy_dml=legacy, gazetted_window=window,
        pooled_forecast_gains=pooled, per_granary_rmse_kg_ha=per_granary, sensitivity=sensitivity, acceptance=acc,
        deviations=["optimize_target_prescriptions not called: returns [] for a real HarmonisedBundle and does not apply its schedule bounds (reads start_decade; the file has Water_Start_Decade)",
                    "gazetted water window not used as a hard bound: see gazetted_window (satellite green-up falls outside it for most granaries); steps bounded by the granary's own training range",
                    "compatibility decided by the domain screen, not by name", "permutation placebo added because the legacy placebo flag is lenient",
                    "weather adjustment includes rainfall; the legacy confounder list has none", "pooled gains use a pre-specified pool (all other compatible granaries) and feature set, not the AutoML selection"],
        limitations=["observational association, not a causal effect", "about 7 held-out seasons per granary: wide intervals", "the recommendation is a fixed small step from the last observed value",
                     "the target-only effect estimator is starved of data", "stress strata use all seasons of the panel (descriptive)"],
        references=REFERENCES, references_dropped=REFERENCES_DROPPED)
    report = _clean(report)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "concordance_uplift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary_rows(report).to_csv(out_dir / "concordance_uplift_summary.csv", index=False)
        S.to_csv(out_dir / "concordance_uplift_seasons.csv", index=False)
        log(f"saved to {out_dir}")
    report["_seasons"] = S
    return report


def summary_rows(rep: dict) -> pd.DataFrame:
    rows = []

    def add(section, metric, d, note=""):
        if isinstance(d, dict) and "estimate" in d:
            rows.append(dict(section=section, metric=metric, estimate=d["estimate"], ci_low=d.get("ci_low"), ci_high=d.get("ci_high"), n=d.get("n", d.get("n_group1")), note=note))

    c = rep["concordance"]
    for key in ("both_levers", "planting_only", "moisture_only"):
        e = c[key]
        for o in ("raw_yield_kg_ha", "granary_adjusted_kg_ha", "weather_adjusted_kg_ha"):
            add("concordance_uplift", f"{key}: {o} (concordant - discordant)", e[o], f"{e['n_concordant']} concordant / {e['n_discordant']} discordant")
    for lev, e in rep["r_loss"].items():
        if lev == "note":
            continue
        for k in ("transfer_minus_target_only", "transfer_minus_source_only", "transfer_minus_no_effect (tau=0)"):
            add("r_loss", f"{lev}: {k}", e[k], "negative = transfer better")
    for lev, r in rep["refutation"].items():
        rows.append(dict(section="placebo", metric=f"{lev}: real effect per unit", estimate=r["tau_per_unit"], ci_low=r["placebo"]["ci95_low"], ci_high=r["placebo"]["ci95_high"], n=r["n"],
                         note=f"ci = placebo 95% interval; permutation p = {r['placebo']['p_value_two_sided']:.3f}"))
    for m, bs in rep["pooled_forecast_gains"].items():
        if not isinstance(bs, dict):
            continue
        for b, res in bs.items():
            if not isinstance(res, dict) or "all" not in res:
                continue
            for stratum in ("all", "normal_years", "climate_stress_years"):
                add(f"pooled_{stratum}", f"{m} vs {b}: RMSE gain kg/ha", res[stratum]["rmse_gain_kg_ha"], "positive = transfer better")
                add(f"pooled_{stratum}", f"{m} vs {b}: season win rate %", res[stratum]["season_win_rate_pct"])
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--boot", type=int, default=None)
    ap.add_argument("--perm", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--tbase", choices=["last", "median"], default="last")
    ap.add_argument("--automl", action="store_true")
    ap.add_argument("--no-forecast", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    rep = run(ROOT, Path(a.out), a.quick, a.boot, a.perm, a.seed, a.methods, a.tbase, not a.no_forecast, a.automl)
    print(json.dumps(rep["acceptance"], indent=2, default=str))


if __name__ == "__main__":
    main()
