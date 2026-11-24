"""Step 2 · Causal Feature Analysis (Identification & Refutation).

Partitions stage/yield variables into Confounders (X), Actionable Practices (T),
Mediators (M), and Outcome (Y), applies propensity overlap filtering (0.2 < e(x) < 0.8),
estimates Double Machine Learning (DML) causal effects, and executes 4 refutation checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

CONFOUNDERS_X = [
    "Rain_mm", "Tmax_C", "Mean_Temp_C", "Solar_MJ", "Soil_pH",
    "C1_Rain_mm_veg", "C1_Tmax_C_rep", "C1_Solar_MJ_rep",
]
PRACTICES_T = [
    "C1_Greenup_Decade", "C1_MNDWI_veg", "C1_SAR_VV_dB_veg", "C1_LSWI_rep",
]
MEDIATORS_M = [
    "C1_EVI_veg", "C1_EVI_rep",
]
OUTCOME_Y = "Yield_kg_ha"


def partition_causal_variables(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return available columns partitioned into X (confounders), T (practices), M (mediators), Y (outcome)."""
    cols = set(df.columns)
    return {
        "X": [c for c in CONFOUNDERS_X if c in cols],
        "T": [c for c in PRACTICES_T if c in cols],
        "M": [c for c in MEDIATORS_M if c in cols],
        "Y": [OUTCOME_Y] if OUTCOME_Y in cols else (["y"] if "y" in cols else []),
    }


def propensity_overlap_filter(
    X: np.ndarray,
    t_continuous: np.ndarray,
    low: float = 0.2,
    high: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Binarize practice above median to estimate propensity e(x) = P(T > med | X) and filter to (low, high)."""
    t_bin = (t_continuous > np.nanmedian(t_continuous)).astype(int)
    if len(np.unique(t_bin)) < 2 or len(X) < 12:
        e_hat = np.full(len(t_continuous), 0.5)
        return np.ones(len(t_continuous), dtype=bool), e_hat
    clf = GradientBoostingClassifier(max_depth=2, n_estimators=40, random_state=42)
    clf.fit(X, t_bin)
    e_hat = clf.predict_proba(X)[:, 1]
    mask = (e_hat > low) & (e_hat < high)
    if mask.sum() < 8:
        mask = np.ones(len(t_continuous), dtype=bool)
    return mask, e_hat


def dml_estimate_and_refute(
    df: pd.DataFrame,
    treatment_col: str = "C1_LSWI_rep",
    outcome_col: str = OUTCOME_Y,
    seed: int = 42,
) -> dict:
    """Estimate deconfounded causal effect tau via cross-fitted DML and run 4 refutation checks."""
    parts = partition_causal_variables(df)
    x_cols = [c for c in parts["X"] if c != treatment_col]
    y_col = outcome_col if outcome_col in df.columns else (parts["Y"][0] if parts["Y"] else None)
    if y_col is None or treatment_col not in df.columns:
        return {"treatment": treatment_col, "tau": 0.0, "se": np.nan, "n_overlap": 0, "refutations": []}

    sub = df[x_cols + [treatment_col, y_col]].dropna().copy()
    if len(sub) < 10:
        return {"treatment": treatment_col, "tau": 0.0, "se": np.nan, "n_overlap": len(sub), "refutations": []}

    X = sub[x_cols].to_numpy(dtype=float) if x_cols else np.zeros((len(sub), 1))
    T = sub[treatment_col].to_numpy(dtype=float)
    Y = sub[y_col].to_numpy(dtype=float)

    mask, e_hat = propensity_overlap_filter(X, T, low=0.2, high=0.8)
    X_ov, T_ov, Y_ov = X[mask], T[mask], Y[mask]

    def _crossfit_tau(X_arr: np.ndarray, T_arr: np.ndarray, Y_arr: np.ndarray, rs: int) -> tuple[float, float]:
        n = len(Y_arr)
        kf = KFold(n_splits=min(5, max(2, n // 4)), shuffle=True, random_state=rs)
        y_res = np.zeros(n)
        t_res = np.zeros(n)
        for tr, va in kf.split(X_arr):
            my = GradientBoostingRegressor(max_depth=2, n_estimators=50, random_state=rs)
            mt = GradientBoostingRegressor(max_depth=2, n_estimators=50, random_state=rs)
            my.fit(X_arr[tr], Y_arr[tr])
            mt.fit(X_arr[tr], T_arr[tr])
            y_res[va] = Y_arr[va] - my.predict(X_arr[va])
            t_res[va] = T_arr[va] - mt.predict(X_arr[va])
        denom = float(np.dot(t_res, t_res))
        if denom <= 1e-12:
            return 0.0, np.nan
        tau = float(np.dot(t_res, y_res) / denom)
        eps = y_res - tau * t_res
        se = float(np.sqrt(np.sum((t_res * eps) ** 2) / (denom ** 2)))
        return tau, se

    tau, se = _crossfit_tau(X_ov, T_ov, Y_ov, seed)
    rng = np.random.RandomState(seed)

    # 4 DoWhy-aligned refutation checks
    # 1. Random Common Cause
    X_rcc = np.column_stack([X_ov, rng.normal(0, 1, size=len(Y_ov))])
    tau_rcc, _ = _crossfit_tau(X_rcc, T_ov, Y_ov, seed + 1)

    # 2. Placebo Treatment (permuted T should collapse effect toward 0)
    T_placebo = rng.permutation(T_ov)
    tau_placebo, _ = _crossfit_tau(X_ov, T_placebo, Y_ov, seed + 2)

    # 3. Data Subset Refuter (80% subsample stability)
    idx_sub = rng.choice(len(Y_ov), size=max(8, int(0.8 * len(Y_ov))), replace=False)
    tau_sub, _ = _crossfit_tau(X_ov[idx_sub], T_ov[idx_sub], Y_ov[idx_sub], seed + 3)

    # 4. Unobserved Confounder Sensitivity (simulated gamma = 0.25 SD)
    u_sim = rng.normal(0, 1, size=len(Y_ov))
    tau_unobs, _ = _crossfit_tau(X_ov, T_ov + 0.1 * u_sim, Y_ov - 0.25 * np.std(Y_ov) * u_sim, seed + 4)

    refutations = [
        {"test": "Random Common Cause", "new_effect": tau_rcc, "passed": abs(tau_rcc - tau) <= max(1.5 * (se or 1.0), 0.25 * abs(tau) + 50.0)},
        {"test": "Placebo Treatment", "new_effect": tau_placebo, "passed": abs(tau_placebo) < max(abs(tau), 150.0)},
        {"test": "Data Subset (80%)", "new_effect": tau_sub, "passed": abs(tau_sub - tau) <= max(2.0 * (se or 1.0), 0.35 * abs(tau) + 75.0)},
        {"test": "Unobserved Confounder", "new_effect": tau_unobs, "passed": np.sign(tau_unobs) == np.sign(tau) or abs(tau) < 50.0},
    ]
    return {
        "treatment": treatment_col,
        "confounders": x_cols,
        "mediators": parts["M"],
        "outcome": y_col,
        "tau": tau,
        "se": se,
        "n_total": len(sub),
        "n_overlap": int(mask.sum()),
        "propensity_mean": float(np.mean(e_hat[mask])),
        "refutations": refutations,
    }