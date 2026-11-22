"""Domain-divergence screening of EVERY candidate source granary against the target (methodology step A3), before any model is trained.

For each candidate granary g and the target granary, on the ten-day observations of the shared drivers:
  AUC        domain-classifier ROC-AUC on the climate drivers (temperature, rain, solar): 0.5 = indistinguishable, 1.0 = trivially separable
  MMD2       unbiased squared maximum mean discrepancy (RBF kernel, median-heuristic bandwidth) on standardised drivers
  CORAL      second-order (covariance) distance on the same drivers
  KS_max     the largest two-sample Kolmogorov-Smirnov statistic over the drivers
  shift_days the crop-calendar offset: the lag (in days) that best aligns the two mean seasonal EVI curves (NDVI where EVI is absent)
A granary is FLAGGED when its climate AUC is at least 0.90. The calendar offset is reported for information only and does not flag: a lag between two
seasonal curves is circular (a 180-day lag and a 180-day lead are the same), and a granary with a single broad season gives an unstable lag.
Flagged granaries are left out of the 'compatible pool'; the dashboard also lets the pool be run WITH them, so the 8-source and 9-source pools can be compared.
Nothing here is fitted to yield, and nothing here changes a result: it is a screen that is shown and acted on by the user.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .assessment import coral_distance, mmd2_unbiased
from .config import Config, Domain
from .assessment import fine_diagnostics

DRIVERS = ["EVI", "LSWI", "MNDWI", "SAR_VV_dB", "Mean_Temp_C", "Tmax_C", "Rain_decade_mm", "Solar_Radiation_MJ"]
SCREEN_AUC = 0.90


def _fine(F, g, feats):
    d = F[F.District == g].reset_index(drop=True)
    return Domain(g, d, [], "Yield_kg_ha", "Year", {}, None, d, feats)


def _signed_shift(fa, fb, col):
    ca, cb = fa.groupby("Decade_N")[col].mean().sort_index().values, fb.groupby("Decade_N")[col].mean().sort_index().values
    ok = ~(np.isnan(ca) | np.isnan(cb))
    if len(ca) != len(cb) or ok.sum() < 12:
        return np.nan
    ca, cb = ca[ok], cb[ok]
    half = len(ca) // 2
    cors = {k: float(np.corrcoef(ca, np.roll(cb, k))[0, 1]) for k in range(-half, half)}
    return float(max(cors, key=cors.get) * 10)


def screen_granaries(F: pd.DataFrame, target: str, cfg: Config = None, auc_cut=SCREEN_AUC):
    """one row per candidate granary, with the flag and the reason; sorted by AUC (most different first)"""
    cfg = cfg or Config()
    feats = [f for f in DRIVERS if f in F.columns]
    tf = F[F.District == target]
    rows = []
    for g in sorted(F.District.unique()):
        if g == target:
            continue
        fa = F[F.District == g]
        fd = fine_diagnostics(_fine(F, g, feats), _fine(F, target, feats), cfg) or {}
        cols = [f for f in feats if fa[f].notna().sum() > 20 and tf[f].notna().sum() > 20]
        A, B = fa[cols].dropna(), tf[cols].dropna()
        mmd = cor = ks = np.nan
        if len(A) > 30 and len(B) > 30 and cols:
            mu, sd = pd.concat([A, B]).mean(), pd.concat([A, B]).std(ddof=0).replace(0, 1.0)
            Za, Zb = ((A - mu) / sd).values, ((B - mu) / sd).values
            rs = np.random.RandomState(cfg.random_state)
            ia, ib = rs.choice(len(Za), min(len(Za), 400), replace=False), rs.choice(len(Zb), min(len(Zb), 400), replace=False)   # subsample: MMD is O(n^2)
            mmd, cor = mmd2_unbiased(Za[ia], Zb[ib])[0], coral_distance(Za, Zb)
            ks = max(ks_2samp(A[c], B[c]).statistic for c in cols)
        sh = _signed_shift(fa, tf, "EVI" if fa.EVI.notna().sum() > 50 and tf.EVI.notna().sum() > 50 else "NDVI")
        auc = fd.get("auc_climate")
        reasons = []
        if auc is not None and auc >= auc_cut:
            reasons.append(f"climate domain AUC {auc:.2f} >= {auc_cut:.2f}")
        rows.append(dict(granary=g, AUC_climate=auc, MMD2=mmd, CORAL=cor, KS_max=ks, shift_days=sh, flagged=bool(reasons), reason="; ".join(reasons)))
    return pd.DataFrame(rows).sort_values("AUC_climate", ascending=False, na_position="last").reset_index(drop=True)


def compatible_pool(screen: pd.DataFrame):
    """the granaries that pass the screen"""
    return [] if screen is None or screen.empty else screen.loc[~screen.flagged, "granary"].tolist()
