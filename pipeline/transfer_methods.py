"""
transfer_methods.py  --  transfer learning methods OTHER than instance re-weighting, on the same NDVI task as transfer_benchmark.py

    python pipeline/transfer_methods.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out data

Same task, same features (ndvi_transfer.FEATS), same strictly temporal split and the same leave-one-granary-out design as
transfer_benchmark.py, so the numbers can be compared line by line with transfer_summary.csv (pooled_ridge, tradaboost).

Methods (one family each)
  references      persistence, target_only, pooled_ridge       (normal training; pooled_ridge is what final_model.py uses)
  FEATURE-BASED   std_pool          every granary's features are z-scored with its OWN mean/std, then pooled
                                    (removes level differences such as temperature or NDVI baseline between granaries)
                  coral             CORAL: source features are re-coloured to the target's covariance and mean, then pooled
  PARAMETER-BASED residual          fit on the sources, then a small Ridge on the target's k years corrects the source model
                  granary_intercept pooled Ridge with one intercept per granary (the target gets its own offset)
  SOURCE SELECTION nearest3         pool only the 3 granaries closest to the target (distance from pre_transfer_assessment.py,
                                    computed on TRAINING years only, covariates only)
                  dist_weight       all sources, but each source row is weighted exp(-distance / median distance)
                                    (weights are per GRANARY by distance; this is not TrAdaBoost-style per-row re-weighting)

Outputs (in --out)  transfer_methods_results.csv   one row per (target, horizon, k, model)
                    transfer_methods_summary.csv   mean MAE over granaries
                    transfer_methods_paired.csv    each method minus pooled_ridge and minus persistence, bootstrap CI over granaries
                    transfer_methods_vs_distance.csv  does a granary's distance predict how much a method helps it?
The dashboard files transfer_summary.csv / transfer_verdict.csv / transfer_paired.csv are NOT touched.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ndvi_transfer as nt                      # noqa: E402
import pre_transfer_assessment as pta           # noqa: E402

warnings.filterwarnings("ignore")
FEATS = nt.FEATS
METHODS = ["persistence", "target_only", "pooled_ridge", "std_pool", "coral", "residual",
           "granary_intercept", "nearest3", "dist_weight"]
TRANSFER = METHODS[3:]


def ridge(X, y, w=None, alpha=1.0):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, y, ridge__sample_weight=w)


def _mat_pow(C, p):
    vals, vecs = np.linalg.eigh(C)
    vals = np.clip(vals, 1e-6, None)
    return (vecs * vals ** p) @ vecs.T


def coral_align(Xs, Xt, eps=1.0):
    """re-colour source features so their mean and covariance match the target (CORAL, Sun et al. 2016)"""
    mu_s, mu_t = Xs.mean(0), Xt.mean(0)
    Cs = np.cov(Xs, rowvar=False) + eps * np.eye(Xs.shape[1])
    Ct = np.cov(Xt, rowvar=False) + eps * np.eye(Xt.shape[1])
    return (Xs - mu_s) @ _mat_pow(Cs, -0.5) @ _mat_pow(Ct, 0.5) + mu_t


def zscore_by(df, mu=None, sd=None):
    X = df[FEATS].values
    mu = X.mean(0) if mu is None else mu
    sd = X.std(0) if sd is None else sd
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (X - mu) / sd, mu, sd


def run_cell(rows, y0, target, sources, test_years, k, min_year, D):
    t0 = min(test_years)
    cut = (t0 - y0) * 36
    test = rows[(rows.District == target) & rows.year.isin(test_years)]
    train = rows[(rows.year >= min_year) & (rows.year < t0) & (rows.t_target < cut)]
    src = train[train.District.isin(sources)]
    tgt = train[(train.District == target) & (train.year >= t0 - k)]
    if len(test) < 20 or len(src) < 50 or len(tgt) < 10:
        return None
    Xte, yte = test[FEATS].values, test["delta"].values
    Xs, ys, Xt, yt = src[FEATS].values, src["delta"].values, tgt[FEATS].values, tgt["delta"].values
    Xp, yp = np.vstack([Xs, Xt]), np.r_[ys, yt]
    mae = {"persistence": np.abs(yte).mean(),
           "target_only": np.abs(ridge(Xt, yt).predict(Xte) - yte).mean()}
    m_pool = ridge(Xp, yp)
    mae["pooled_ridge"] = np.abs(m_pool.predict(Xte) - yte).mean()

    # --- FEATURE-BASED: per-granary standardisation
    parts, ys_all = [], []
    for g, d in src.groupby("District"):
        z, _, _ = zscore_by(d)
        parts.append(z)
        ys_all.append(d["delta"].values)
    zt, mu_t, sd_t = zscore_by(tgt)
    zte, _, _ = zscore_by(test, mu_t, sd_t)
    m = Ridge(alpha=1.0).fit(np.vstack(parts + [zt]), np.r_[np.concatenate(ys_all), yt])
    mae["std_pool"] = np.abs(m.predict(zte) - yte).mean()

    # --- FEATURE-BASED: CORAL (globally standardised, then source re-coloured to the target)
    sc = StandardScaler().fit(Xp)
    Zs, Zt, Zte = sc.transform(Xs), sc.transform(Xt), sc.transform(Xte)
    Zs_al = coral_align(Zs, Zt)
    m = Ridge(alpha=1.0).fit(np.vstack([Zs_al, Zt]), np.r_[ys, yt])
    mae["coral"] = np.abs(m.predict(Zte) - yte).mean()

    # --- PARAMETER-BASED: source model + target residual correction
    ms = ridge(Xs, ys)
    mr = ridge(Xt, yt - ms.predict(Xt), alpha=10.0)
    mae["residual"] = np.abs(ms.predict(Xte) + mr.predict(Xte) - yte).mean()

    # --- PARAMETER-BASED: one intercept per granary
    names = sorted(set(sources) | {target})
    def dummies(ids):
        return np.array([[1.0 if i == n else 0.0 for n in names] for i in ids])
    sc2 = StandardScaler().fit(Xp)
    A = np.hstack([sc2.transform(Xp), dummies(np.r_[src.District.values, tgt.District.values])])
    Ate = np.hstack([sc2.transform(Xte), dummies(test.District.values)])
    m = Ridge(alpha=1.0).fit(A, yp)
    mae["granary_intercept"] = np.abs(m.predict(Ate) - yte).mean()

    # --- SOURCE SELECTION: nearest 3 granaries
    near = pta.nearest(D, target, 3, exclude=[s for s in D.index if s not in sources])
    sn = src[src.District.isin(near)]
    Xn, yn = np.vstack([sn[FEATS].values, Xt]), np.r_[sn["delta"].values, yt]
    mae["nearest3"] = np.abs(ridge(Xn, yn).predict(Xte) - yte).mean() if len(sn) >= 50 else np.nan

    # --- SOURCE SELECTION: weight by granary distance
    dist = D.loc[target, sources].astype(float)
    tau = max(float(dist.median()), 1e-6)
    wmap = np.exp(-dist / tau)
    w = np.r_[src.District.map(wmap).values, np.ones(len(tgt))]
    mae["dist_weight"] = np.abs(ridge(Xp, yp, w).predict(Xte) - yte).mean()

    return [dict(target=target, k_years=k, MAE=v, model=mm, n_test=len(yte), n_target_rows=len(tgt),
                 n_source_rows=len(src)) for mm, v in mae.items()]


def boot_ci(x, n=4000, seed=0):
    r = np.random.RandomState(seed)
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    m = [x[r.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--pheno", default="eda/eda_phenology.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--test-years", nargs="+", type=int, default=[2022, 2023, 2024])
    ap.add_argument("--min-year", type=int, default=2019)
    ap.add_argument("--min-pix", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ts = nt.load_series(a.clean, a.min_pix)
    ph = pd.read_csv(a.pheno).set_index("District")
    granaries = sorted(set(ts.District) & set(ph.index))
    print("distance matrix (covariates only, training years only) ...")
    D = pta.distance_matrix(pta.load_decade_table(a.clean), max_year=min(a.test_years) - 1)
    print(f"{len(granaries)} granaries as targets (each uses the other {len(granaries) - 1} as sources)")

    res = []
    for h in a.horizons:
        rows, y0 = nt.make_rows(ts, ph, h)
        for tg in granaries:
            for k in a.ks:
                r = run_cell(rows, y0, tg, [g for g in granaries if g != tg], a.test_years, k, a.min_year, D)
                if r:
                    res += [dict(x, horizon=h) for x in r]
        print(f"  horizon {h}: done")
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(a.out, "transfer_methods_results.csv"), index=False)
    W = df.pivot_table(index=["target", "horizon", "k_years"], columns="model", values="MAE").reset_index()

    summ = df.groupby(["horizon", "k_years", "model"]).MAE.mean().unstack("model")[METHODS].round(4).reset_index()
    summ.to_csv(os.path.join(a.out, "transfer_methods_summary.csv"), index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nMean NDVI-forecast MAE over granaries (lower is better):")
    print(summ.to_string(index=False))

    prow = []
    for h, k in W[["horizon", "k_years"]].drop_duplicates().itertuples(index=False):
        g = W[(W.horizon == h) & (W.k_years == k)]
        for m in TRANSFER:
            for ref in ["pooled_ridge", "persistence"]:
                d = (g[m] - g[ref]).values                    # negative = the method is better
                lo, hi = boot_ci(d)
                prow.append(dict(horizon=h, k_years=k, method=m, versus=ref, n_granaries=int(np.isfinite(d).sum()),
                                 mean_MAE_diff=np.nanmean(d), ci95_low=lo, ci95_high=hi, method_wins=int((d < 0).sum()),
                                 relative_change_pct=100 * np.nanmean(d) / g[ref].mean()))
    paired = pd.DataFrame(prow).round(4)
    paired.to_csv(os.path.join(a.out, "transfer_methods_paired.csv"), index=False)
    print("\nMethod minus pooled_ridge (negative = better than normal training; CI = bootstrap over granaries):")
    print(paired[paired.versus == "pooled_ridge"].drop(columns="versus").to_string(index=False))

    # does a granary's distance from the rest predict how much a method helps it? (cold start = lowest k)
    k0 = min(a.ks)
    md = (D.sum(axis=1) / (len(D) - 1)).rename("mean_distance")
    vrows = []
    for h in a.horizons:
        g = W[(W.horizon == h) & (W.k_years == k0)].set_index("target").join(md)
        for m in TRANSFER:
            gain = 100 * (g["pooled_ridge"] - g[m]) / g["pooled_ridge"]
            ok = gain.notna() & g.mean_distance.notna()
            rho, p = spearmanr(g.mean_distance[ok], gain[ok]) if ok.sum() > 3 else (np.nan, np.nan)
            vrows.append(dict(horizon=h, k_years=k0, method=m, spearman_distance_vs_gain=rho, p_value=p,
                              granaries_improved=int((gain > 0).sum()), n=int(ok.sum())))
    vd = pd.DataFrame(vrows).round(3)
    vd.to_csv(os.path.join(a.out, "transfer_methods_vs_distance.csv"), index=False)
    print(f"\nDoes distance predict the gain over pooled_ridge (k={k0})? negative rho = far granaries gain less:")
    print(vd.to_string(index=False))


if __name__ == "__main__":
    main()
