"""
pre_transfer_assessment.py  --  is it SAFE to transfer between these granaries, and from which ones?

    python pipeline/pre_transfer_assessment.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv \
           --yield-hist data/yield_history.csv --out data

Runs BEFORE any transfer model. Uses covariates only (never yield), so it is allowed for a granary that has no labels yet.

Measures
  distance   standardised Wasserstein distance between two granaries, averaged over NDVI, EVI, SAR, temperature, rain, solar
             (decade-level values). Small = similar. Soil pH is left out: it is one constant per granary, not a distribution.
  domain AUC a classifier tries to tell "this granary" from "all the others" using (a) climate, (b) NDVI/EVI/SAR.
             0.5 = indistinguishable (good for transfer), near 1.0 = very different (negative-transfer risk).
  calendar   day-of-year of the first NDVI peak, and the offset (days) to the nearest source granary.
  quality    share of rows with a real optical observation / interpolated NDVI (a hidden source of domain shift).
  yield      mean, coefficient of variation, large year-to-year jumps.

Writes   pre_transfer_distance.csv   (granary x granary matrix)
         pre_transfer_profile.csv    (one row per granary: nearest sources, risk flags)

Importable: load_decade_table(), distance_matrix(g, max_year), nearest(D, target, k, exclude)
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

warnings.filterwarnings("ignore")

DIST_FEATS = ["NDVI_clean", "EVI_clean", "SAR_clean", "Mean_Temp_C", "Daily_Rain_mm", "Solar_Radiation_MJ"]
CLIMATE = ["Mean_Temp_C", "Daily_Rain_mm", "Solar_Radiation_MJ"]
OPTICAL = ["NDVI_clean", "EVI_clean", "SAR_clean"]
COLS = ["Year", "District", "Decade_N"] + DIST_FEATS + ["Optical_Observed", "flag_interpolated"]


def load_decade_table(clean_csv):
    """one row per (granary, year, decade): the mean over pixels"""
    d = pd.read_csv(clean_csv, usecols=COLS)
    return (d.groupby(["District", "Year", "Decade_N"])[DIST_FEATS + ["Optical_Observed", "flag_interpolated"]]
              .mean().reset_index())


def distance_matrix(g, max_year=None):
    """standardised Wasserstein distance matrix. max_year: use only years <= max_year (no peeking at the test period)."""
    if max_year is not None:
        g = g[g.Year <= max_year]
    names = sorted(g.District.unique())
    sd = g[DIST_FEATS].std()
    vals = {n: {f: g.loc[g.District == n, f].dropna().values for f in DIST_FEATS} for n in names}
    M = pd.DataFrame(0.0, index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ds = [wasserstein_distance(vals[a][f], vals[b][f]) / sd[f]
                  for f in DIST_FEATS if len(vals[a][f]) and len(vals[b][f])]
            M.loc[a, b] = M.loc[b, a] = float(np.mean(ds))
    return M


def nearest(D, target, k, exclude=()):
    """names of the k granaries closest to target (never the target itself or anything in exclude)"""
    s = D.loc[target].drop(labels=[target] + [e for e in exclude if e in D.index], errors="ignore").sort_values()
    return list(s.index[:k])


def domain_auc(g, fs):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    out = {}
    for n in sorted(g.District.unique()):
        X = g[fs]
        y = (g.District == n).astype(int)
        m = X.notna().all(axis=1)
        out[n] = float(cross_val_score(HistGradientBoostingClassifier(max_iter=60), X[m], y[m], cv=3,
                                       scoring="roc_auc").mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--pheno", default="eda/eda_phenology.csv")
    ap.add_argument("--yield-hist", default="data/yield_history.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--far-auc", type=float, default=0.85, help="climate AUC above this = far domain")
    ap.add_argument("--calendar-days", type=int, default=30, help="peak offset above this = different crop calendar")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    g = load_decade_table(a.clean)
    D = distance_matrix(g)
    D.round(4).to_csv(os.path.join(a.out, "pre_transfer_distance.csv"))

    prof = pd.DataFrame(index=D.index)
    prof["mean_distance"] = D.sum(axis=1) / (len(D) - 1)
    prof["nearest_1"] = [nearest(D, n, 1)[0] for n in D.index]
    prof["nearest_1_distance"] = [D.loc[n, nearest(D, n, 1)[0]] for n in D.index]
    prof["nearest_3"] = [", ".join(nearest(D, n, 3)) for n in D.index]
    prof["auc_climate"] = pd.Series(domain_auc(g, CLIMATE))
    prof["auc_optical_sar"] = pd.Series(domain_auc(g, OPTICAL))

    ph = pd.read_csv(a.pheno).set_index("District")
    doy = pd.to_datetime(ph["peak_A_date"] + " 2021", format="%d %b %Y").dt.dayofyear
    prof["first_peak_doy"] = doy.reindex(prof.index)

    def offset(n):
        o = (doy[n] - doy.drop(n)).abs()
        o = np.minimum(o, 365 - o)
        return float(o[nearest(D, n, 1)[0]])
    prof["peak_offset_to_nearest_days"] = [offset(n) if n in doy.index else np.nan for n in prof.index]

    q = g.groupby("District")[["Optical_Observed", "flag_interpolated"]].mean()
    prof["optical_observed_pct"] = (100 * q["Optical_Observed"]).round(1)
    prof["interpolated_pct"] = (100 * q["flag_interpolated"]).round(1)

    if os.path.exists(a.yield_hist):
        y = pd.read_csv(a.yield_hist)
        s = y.groupby("District").Yield.agg(["mean", "std", "count"])
        prof["yield_mean"] = s["mean"].round(0)
        prof["yield_cv"] = (s["std"] / s["mean"]).round(3)
        prof["yield_years"] = s["count"]
        prof["yield_jumps"] = y.groupby("District").Flag_large_jump.sum() if "Flag_large_jump" in y else np.nan

    far = prof["auc_climate"] > a.far_auc
    cal = prof["peak_offset_to_nearest_days"] > a.calendar_days
    prof["risk"] = np.select([far, cal], ["far domain: do not pool blindly", "calendar differs: align stage first"],
                             default="ok to transfer from nearest sources")
    prof = prof.round(3)
    prof.to_csv(os.path.join(a.out, "pre_transfer_profile.csv"), index_label="District")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("Pre-transfer profile (distance: lower = more similar; AUC: 0.5 = same, 1.0 = very different):")
    print(prof.drop(columns=["nearest_3"]).to_string())
    print("\nWrote pre_transfer_distance.csv and pre_transfer_profile.csv to", a.out)


if __name__ == "__main__":
    main()
