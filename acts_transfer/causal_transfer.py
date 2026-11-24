"""Step 3 · Feature Engineering & Cross-Region Causal Transfer Model Training.

Aligns crop stages relative to green-up (HST), excludes climate-incompatible donors
(Domain AUC >= 0.90), applies optical quality weights w_t = max(0.2, Optical_Valid_Fraction_t),
and fits Transfer-Weighted Causal Forest / X-Learner & Detrended Residual Transfer estimators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler


def compute_optical_quality_weights(df: pd.DataFrame, floor: float = 0.2) -> np.ndarray:
    """Compute sample weights w_t = max(0.2, Optical_Valid_Fraction_t)."""
    if "Optical_Valid_Fraction" in df.columns:
        frac = pd.to_numeric(df["Optical_Valid_Fraction"], errors="coerce").fillna(0.5).to_numpy(dtype=float)
    elif "Optical_observed_pct" in df.columns:
        frac = pd.to_numeric(df["Optical_observed_pct"], errors="coerce").fillna(50.0).to_numpy(dtype=float) / 100.0
    else:
        frac = np.ones(len(df), dtype=float)
    return np.maximum(floor, np.clip(frac, 0.0, 1.0))


def filter_compatible_donors(
    df: pd.DataFrame,
    target_granary: str,
    climate_cols: list[str] | None = None,
    auc_threshold: float = 0.90,
) -> tuple[list[str], dict[str, float]]:
    """Compute pairwise Domain Classifier AUC between target and each donor; exclude donors with AUC >= 0.90."""
    if climate_cols is None:
        climate_cols = [c for c in ("Rain_mm", "Tmax_C", "Mean_Temp_C", "Solar_MJ", "Soil_pH", "C1_Rain_mm_veg", "C1_Tmax_C_rep") if c in df.columns]
    tgt = df[df["District"] == target_granary][climate_cols].dropna()
    donors = sorted(d for d in df["District"].unique() if d != target_granary)
    auc_scores: dict[str, float] = {}
    kept: list[str] = []
    for d in donors:
        src = df[df["District"] == d][climate_cols].dropna()
        if len(tgt) < 4 or len(src) < 4 or not climate_cols:
            auc_scores[d] = 0.5
            kept.append(d)
            continue
        X = np.vstack([tgt.to_numpy(dtype=float), src.to_numpy(dtype=float)])
        y = np.array([1] * len(tgt) + [0] * len(src))
        X_sc = RobustScaler().fit_transform(X)
        n_splits = min(3, len(tgt), len(src))
        if n_splits < 2:
            auc = 0.5
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            probs = np.zeros(len(y))
            for tr, va in skf.split(X_sc, y):
                clf = GradientBoostingRegressor(max_depth=2, n_estimators=30, random_state=42)
                clf.fit(X_sc[tr], y[tr])
                probs[va] = clf.predict(X_sc[va])
            auc = float(roc_auc_score(y, probs))
        auc_scores[d] = auc
        if auc < auc_threshold:
            kept.append(d)
    return kept, auc_scores


class TransferXLearner:
    """Two-stage X-Learner / Causal Forest estimator with optical quality sample weights."""

    def __init__(self, treatment_idx: int = 0, random_state: int = 42):
        self.treatment_idx = treatment_idx
        self.random_state = random_state
        self.mu0 = ExtraTreesRegressor(n_estimators=80, max_depth=4, random_state=random_state)
        self.mu1 = ExtraTreesRegressor(n_estimators=80, max_depth=4, random_state=random_state + 1)
        self.tau_model = RandomForestRegressor(n_estimators=80, max_depth=4, random_state=random_state + 2)
        self.t_median = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        t_col = X[:, self.treatment_idx]
        self.t_median = float(np.nanmedian(t_col))
        treated = t_col > self.t_median
        control = ~treated
        if treated.sum() < 3 or control.sum() < 3:
            self.mu0.fit(X, y, sample_weight=w)
            self.mu1.fit(X, y, sample_weight=w)
            self.tau_model.fit(X, np.zeros(len(y)), sample_weight=w)
            return self
        self.mu0.fit(X[control], y[control], sample_weight=w[control])
        self.mu1.fit(X[treated], y[treated], sample_weight=w[treated])
        d1 = y[treated] - self.mu0.predict(X[treated])
        d0 = self.mu1.predict(X[control]) - y[control]
        d_all = np.concatenate([d1, d0])
        X_all = np.vstack([X[treated], X[control]])
        w_all = np.concatenate([w[treated], w[control]])
        self.tau_model.fit(X_all, d_all, sample_weight=w_all)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        t_col = X[:, self.treatment_idx]
        base = np.where(t_col > self.t_median, self.mu1.predict(X), self.mu0.predict(X))
        cate = self.tau_model.predict(X)
        return base + 0.25 * np.where(t_col > self.t_median, cate, -cate)