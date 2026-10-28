"""
yield_dynamics_methods.py -- two more ways to transfer yield DYNAMICS (no instance weighting)

    python pipeline/yield_dynamics_methods.py --yield-hist data/yield_history.csv --distance data/pre_transfer_distance.csv --out data

Same back-test as yield_dynamics_transfer.py (predict year Y from years <= Y-h, test years 2021..2024, >= 3 years of history),
so the numbers sit next to its ar1_transfer result.

Methods   last_year, ewma_own      references (no transfer / normal training)
          ar1_transfer             carry-over rate learned on ALL other granaries            (existing method, for comparison)
          ar1_nearest              carry-over rate learned only on the k granaries NEAREST to the target
                                   (distance from pre_transfer_assessment.py; covariates only, no yield used)   SOURCE SELECTION
          ar1_shrunk               the target's own carry-over rate, pulled toward the pooled rate when it has few years
                                   (rate = (n*own + n0*pooled) / (n + n0))                                        PARAMETER SHRINKAGE
Outputs   yield_methods_results.csv, yield_methods_summary.csv (mean % error by horizon and method, paired CI vs last_year)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yield_dynamics_transfer import ape, boot, ewma_pred, fit_alpha, fit_phi   # noqa: E402

ORDER = ["last_year", "ewma_own", "ar1_transfer", "ar1_nearest", "ar1_shrunk"]


def run(yh, D, horizons, test_years, k_near, n0):
    S = {g: d.sort_values("Year") for g, d in yh.groupby("District")}
    rows = []
    for h in horizons:
        for Y in test_years:
            for g, d in S.items():
                hist, act = d[d.Year <= Y - h], d[d.Year == Y]
                if len(hist) < 3 or act.empty:
                    continue
                y, truth = hist.Yield.values, act.Yield.iloc[0]
                m = y.mean()
                a_own = fit_alpha([y], h) if len(y) >= h + 3 else 0.5
                others = {o: S[o][S[o].Year <= Y - h].Yield.values for o in S if o != g}
                phi_all = fit_phi(list(others.values()), h)
                near = [o for o in D.loc[g].drop(labels=[g]).sort_values().index if o in others][:k_near]
                phi_near = fit_phi([others[o] for o in near], h)
                n_pairs = max(len(y) - h, 0)
                phi_own = fit_phi([y], h) if len(y) >= h + 3 else phi_all
                phi_shr = (n_pairs * phi_own + n0 * phi_all) / (n_pairs + n0)
                preds = {"last_year": y[-1], "ewma_own": ewma_pred(y, a_own),
                         "ar1_transfer": m + phi_all * (y[-1] - m),
                         "ar1_nearest": m + phi_near * (y[-1] - m),
                         "ar1_shrunk": m + phi_shr * (y[-1] - m)}
                for k, p in preds.items():
                    rows.append(dict(horizon=h, test_year=Y, District=g, method=k, actual=truth, predicted=p,
                                     ape=ape(p, truth), phi_all=phi_all, phi_near=phi_near, phi_shrunk=phi_shr,
                                     nearest=", ".join(near)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--yield-hist", default="data/yield_history.csv")
    ap.add_argument("--distance", default="data/pre_transfer_distance.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--test-years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--k-near", type=int, default=3)
    ap.add_argument("--n0", type=float, default=4.0, help="how many 'pseudo-years' the pooled rate is worth in the shrinkage")
    a = ap.parse_args()
    if not os.path.exists(a.distance):
        raise SystemExit(f"{a.distance} not found: run pipeline/pre_transfer_assessment.py first")
    yh = pd.read_csv(a.yield_hist)[["District", "Year", "Yield"]].dropna()
    D = pd.read_csv(a.distance, index_col=0)
    r = run(yh, D, a.horizons, a.test_years, a.k_near, a.n0)
    r.to_csv(os.path.join(a.out, "yield_methods_results.csv"), index=False)
    w = r.pivot_table(index=["horizon", "test_year", "District"], columns="method", values="ape")[ORDER]
    summ = pd.concat([w.groupby(level=0).size().rename("n"), w.groupby(level=0).mean().round(2)], axis=1)
    summ.to_csv(os.path.join(a.out, "yield_methods_summary.csv"))
    pd.set_option("display.width", 200)
    print("Mean absolute % error of yield prediction h years ahead (lower is better):")
    print(summ.to_string())
    print("\nEach transfer method minus last_year (pp; negative = better), 95% bootstrap CI over granaries:")
    for h, g in w.groupby(level=0):
        for m in ["ar1_transfer", "ar1_nearest", "ar1_shrunk"]:
            d = (g[m] - g["last_year"]).groupby(level=2).mean()
            lo, hi = boot(d.values)
            print(f"  h={h}  {m:13s} {d.mean():+.2f} pp [{lo:+.2f}, {hi:+.2f}]   better in {(g[m] < g['last_year']).mean() * 100:.0f}% of cases")


if __name__ == "__main__":
    main()
