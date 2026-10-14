"""
transfer_benchmark.py  --  is transfer learning (TrAdaBoost.R2) worth it, compared with NORMAL training?

    python pipeline/transfer_benchmark.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out transfer_benchmark

Question: a granary has only k years of its own satellite history (a newly added granary). The other granaries have more.
Does TrAdaBoost.R2 (source = the other granaries, target = this granary's k years) forecast NDVI better than normal training?

Task      NDVI forecast h decades (10 days each) ahead, same features and same strictly temporal split as ndvi_transfer.py
          (train years 2019..2021, test years 2022-2024, no row whose target date falls in the test period).
Targets   every granary in turn; sources = all the other granaries (leave-one-granary-out).
Models    persistence      NDVI stays at its last value (the bar every model must beat)
          clim_delta       seasonal baseline
          target_only      Ridge on the target's k years only
          pooled_ridge     NORMAL training: Ridge on sources + target k years, equal weights  (what final_model.py does)
          pooled_hgb       NORMAL training: gradient boosting on the same pooled rows
          pooled_ada       CONTROL: AdaBoost.R2 with the SAME weak learner / rounds as TrAdaBoost, on the same pooled rows,
                           WITHOUT the transfer re-weighting. If TrAdaBoost only beats weaker models but not this one, the gain is
                           'boosting', not 'transfer'.
          tradaboost       TrAdaBoost.R2 (Pardoe & Stone 2010), stochastic -> averaged over --seeds seeds
Evidence  paired per (granary, horizon, k): wins/losses vs each normal model, and a bootstrap over granaries for the mean MAE
          difference. Writes transfer_results.csv, transfer_summary.csv, transfer_paired.csv and transfer_verdict.csv.
          Run it with  --out data  and the dashboard picks them up automatically (the bridge reads them).
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ndvi_transfer as nt                      # noqa: E402  (same features, rows and baselines as the rest of the pipeline)
from tradaboost_r2 import TrAdaBoostR2          # noqa: E402

warnings.filterwarnings("ignore")
FEATS = nt.FEATS
N_ROUNDS, DEPTH = 30, 3
MODELS = ["persistence", "clim_delta", "target_only", "pooled_ridge", "pooled_hgb", "pooled_ada", "tradaboost"]
NORMAL = ["pooled_ridge", "pooled_hgb", "pooled_ada"]


def ridge(X, y):
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)


def tree():
    return DecisionTreeRegressor(max_depth=DEPTH)


def run_cell(rows, y0, target, sources, test_years, k, min_year, seeds):
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
           "clim_delta": np.abs(nt.clim_delta(tgt, test) - yte).mean(),
           "target_only": np.abs(ridge(Xt, yt).predict(Xte) - yte).mean(),
           "pooled_ridge": np.abs(ridge(Xp, yp).predict(Xte) - yte).mean(),
           "pooled_hgb": np.abs(nt.fit_model("hgb", Xp, yp).predict(Xte) - yte).mean()}
    ada, tra = [], []
    for s in range(seeds):
        a = AdaBoostRegressor(estimator=tree(), n_estimators=N_ROUNDS, random_state=s).fit(Xp, yp)
        ada.append(np.abs(a.predict(Xte) - yte).mean())
        t = TrAdaBoostR2(base_estimator=tree(), n_estimators=N_ROUNDS, random_state=s).fit(Xs, ys, Xt, yt)
        tra.append(np.abs(t.predict(Xte) - yte).mean())
    mae["pooled_ada"], mae["tradaboost"] = float(np.mean(ada)), float(np.mean(tra))
    return [dict(target=target, k_years=k, MAE=v, model=m, n_test=len(yte), n_target_rows=len(tgt), n_source_rows=len(src))
            for m, v in mae.items()]


def boot_ci(x, n=4000, seed=0):
    """bootstrap over granaries (the independent units) for the mean of x"""
    r = np.random.RandomState(seed)
    x = np.asarray(x, float)
    m = [x[r.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--pheno", default="eda/eda_phenology.csv")
    ap.add_argument("--out", default="transfer_benchmark")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--test-years", nargs="+", type=int, default=[2022, 2023, 2024])
    ap.add_argument("--min-year", type=int, default=2019)
    ap.add_argument("--min-pix", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ts = nt.load_series(a.clean, a.min_pix)
    ph = pd.read_csv(a.pheno).set_index("District")
    granaries = sorted(set(ts.District) & set(ph.index))
    print(f"{len(granaries)} granaries as targets (each uses the other {len(granaries) - 1} as sources)")
    res = []
    for h in a.horizons:
        rows, y0 = nt.make_rows(ts, ph, h)
        for tg in granaries:
            for k in a.ks:
                r = run_cell(rows, y0, tg, [g for g in granaries if g != tg], a.test_years, k, a.min_year, a.seeds)
                if r:
                    res += [dict(x, horizon=h) for x in r]
        print(f"  horizon {h}: done")
    df = pd.DataFrame(res)
    df.to_csv(os.path.join(a.out, "transfer_results.csv"), index=False)
    W = df.pivot_table(index=["target", "horizon", "k_years"], columns="model", values="MAE").reset_index()

    # ---- summary: mean MAE over granaries
    summ = (df.groupby(["horizon", "k_years", "model"]).MAE.mean().unstack("model")[MODELS].round(4).reset_index())
    summ.to_csv(os.path.join(a.out, "transfer_summary.csv"), index=False)
    pd.set_option("display.width", 220)
    print("\nMean NDVI-forecast MAE over granaries (lower is better):")
    print(summ.to_string(index=False))

    # ---- paired comparison: tradaboost vs each normal model
    prow = []
    for h, k in W[["horizon", "k_years"]].drop_duplicates().itertuples(index=False):
        g = W[(W.horizon == h) & (W.k_years == k)]
        for nm in NORMAL + ["persistence"]:
            d = (g["tradaboost"] - g[nm]).values          # negative = TrAdaBoost better
            lo, hi = boot_ci(d)
            prow.append(dict(horizon=h, k_years=k, versus=nm, n_granaries=len(d), mean_MAE_diff=d.mean(),
                             ci95_low=lo, ci95_high=hi, tradaboost_wins=int((d < 0).sum()),
                             relative_change_pct=100 * d.mean() / g[nm].mean()))
    paired = pd.DataFrame(prow).round(4)
    paired.to_csv(os.path.join(a.out, "transfer_paired.csv"), index=False)
    print("\nTrAdaBoost minus other model (MAE; negative = TrAdaBoost better; CI = bootstrap over granaries):")
    print(paired.to_string(index=False))

    # ---- verdict per granary and horizon, at the LOWEST k (the cold-start case transfer is meant for)
    k0 = min(a.ks)
    ver = []
    for (tg, h), g in W[W.k_years == k0].groupby(["target", "horizon"]):
        r = g.iloc[0]
        best_n = min(NORMAL, key=lambda m: r[m])
        gain = (r[best_n] - r["tradaboost"]) / r[best_n] * 100
        ver.append(dict(District=tg, horizon=int(h), k_years=k0, MAE_normal_best=r[best_n], best_normal_model=best_n,
                        MAE_tradaboost=r["tradaboost"], MAE_persistence=r["persistence"],
                        gain_pct=gain, verdict=("transfer better" if gain > 5 else "transfer worse" if gain < -5 else "no clear difference")))
    ver = pd.DataFrame(ver).round(4)
    ver.to_csv(os.path.join(a.out, "transfer_verdict.csv"), index=False)
    print(f"\nPer granary, cold start (k={k0} year of own data), gain of TrAdaBoost over the best normal model (+ = better):")
    print(ver[["District", "horizon", "MAE_normal_best", "best_normal_model", "MAE_tradaboost", "gain_pct", "verdict"]].to_string(index=False))
    print("\nverdict counts:", ver.groupby(["horizon", "verdict"]).size().to_dict())


if __name__ == "__main__":
    main()
