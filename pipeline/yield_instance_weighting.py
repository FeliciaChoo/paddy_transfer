"""
yield_instance_weighting.py -- instance-weighting transfer learning applied to the YIELD DYNAMICS problem (not to satellite inputs)

    python pipeline/yield_instance_weighting.py --yield-hist data/yield_history.csv --out data

Why this test: instance weighting (TrAdaBoost and relatives) is meant for 'little target data, plenty of related source data'. Here the
instances are year-to-year transitions: x = last year's deviation from the granary's mean, y = this year's deviation (as fractions of the mean).
The target granary has only 1-5 such transitions; the other granaries have many. Methods (all use only years before the test year):
  last_year            reference, no learning
  ewma_own             normal training: smoothing weight learned on the granary alone
  ar1_transfer         parameter transfer: ONE carry-over rate learned on all other granaries equally (what the dashboard uses)
  ar1_similarity       INSTANCE WEIGHTING: the carry-over rate is learned on the other granaries weighted by how similar they are to the
                       target (yield level and volatility; Gaussian kernel, bandwidth 1 SD)
  pool_upweight        INSTANCE WEIGHTING: pooled ridge on all transitions, the target's own transitions weighted x5
  tradaboost_ridge     TrAdaBoost.R2 on the transitions, ridge as the weak learner  (source transitions are re-weighted by boosting)
  tradaboost_stump     the same with depth-1 trees
Cases where the granary has no transition of its own yet (for example 3 years ahead) are left out for every method, so all columns are
compared on the same cases.
Output: mean absolute % error per horizon, and the paired difference against last_year and against ar1_transfer (bootstrap over granaries).
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tradaboost_r2 import TrAdaBoostR2
from yield_dynamics_transfer import ape, boot, ewma_pred, fit_alpha, fit_phi, GRID
warnings.filterwarnings("ignore")


def transitions(s, h):
    """(x, y) pairs: deviation of year t-h from the mean known then, and of year t, both as fractions of that mean"""
    xs, ys = [], []
    for t in range(h + 1, len(s)):
        m = s[:t - h + 1].mean()
        xs.append((s[t - h] - m) / m)
        ys.append((s[t] - m) / m)
    return np.array(xs).reshape(-1, 1), np.array(ys)


def fit_phi_weighted(series, weights, h):
    best, bp = np.inf, 0.5
    for p in GRID:
        e, w = [], []
        for s, wt in zip(series, weights):
            for t in range(h + 1, len(s)):
                m = s[:t - h + 1].mean()
                e.append(ape(m + p * (s[t - h] - m), s[t]))
                w.append(wt)
        if e and np.average(e, weights=w) < best:
            best, bp = np.average(e, weights=w), p
    return bp


def similarity_weights(target, others, bw=1.0):
    f = lambda s: np.array([np.log(s.mean()), s.std() / s.mean() if len(s) > 1 else 0.0])
    F = np.array([f(target)] + [f(o) for o in others])
    sd = F.std(axis=0) + 1e-9
    d2 = (((F[1:] - F[0]) / sd) ** 2).sum(axis=1)
    return np.exp(-d2 / (2 * bw ** 2))


def run(yh, horizons, test_years, seeds):
    S = {g: d.sort_values("Year") for g, d in yh.groupby("District")}
    rows = []
    for h in horizons:
        for Y in test_years:
            for g, d in S.items():
                hist, act = d[d.Year <= Y - h], d[d.Year == Y]
                if len(hist) < 3 or act.empty:
                    continue
                y, truth = hist.Yield.values, act.Yield.iloc[0]
                others = [o[o.Year <= Y - h].Yield.values for k, o in S.items() if k != g]
                others = [o for o in others if len(o) >= 3]
                m = y.mean()
                a_own = fit_alpha([y], h) if len(y) >= h + 3 else 0.5
                p_eq = fit_phi(others, h)
                p_sim = fit_phi_weighted(others, similarity_weights(y, others), h)
                Xs = np.vstack([transitions(o, h)[0] for o in others]); ys = np.concatenate([transitions(o, h)[1] for o in others])
                Xt, yt = transitions(y, h)
                x_new = np.array([[(y[-1] - m) / m]])
                preds = {"last_year": y[-1], "ewma_own": ewma_pred(y, a_own), "ar1_transfer": m + p_eq * (y[-1] - m),
                         "ar1_similarity": m + p_sim * (y[-1] - m)}
                if len(yt):
                    X = np.vstack([Xs, Xt]); Y_ = np.r_[ys, yt]; w = np.r_[np.ones(len(ys)), 5.0 * np.ones(len(yt))]
                    preds["pool_upweight"] = m * (1 + Ridge(alpha=0.01).fit(X, Y_, sample_weight=w).predict(x_new)[0])
                    for name, mk in (("tradaboost_ridge", lambda: Ridge(alpha=0.01)), ("tradaboost_stump", lambda: DecisionTreeRegressor(max_depth=1))):
                        p = [TrAdaBoostR2(base_estimator=mk(), n_estimators=20, random_state=s).fit(Xs, ys, Xt, yt).predict(x_new)[0] for s in range(seeds)]
                        preds[name] = m * (1 + float(np.mean(p)))
                # no own transitions yet: instance weighting has nothing to weight against, so the method is NOT scored for this case
                for k, p in preds.items():
                    rows.append(dict(horizon=h, test_year=Y, District=g, method=k, ape=ape(p, truth)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--yield-hist", default="data/yield_history.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--test-years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    yh = pd.read_csv(a.yield_hist)[["District", "Year", "Yield"]].dropna()
    r = run(yh, a.horizons, a.test_years, a.seeds)
    os.makedirs(a.out, exist_ok=True)
    order = ["last_year", "ewma_own", "ar1_transfer", "ar1_similarity", "pool_upweight", "tradaboost_ridge", "tradaboost_stump"]
    w = r.pivot_table(index=["horizon", "test_year", "District"], columns="method", values="ape")[order]
    w = w.dropna()                                    # compare every method on exactly the same cases (those where instance weighting can run)
    summ = w.groupby(level=0).mean().round(2)
    summ.insert(0, "cases", w.groupby(level=0).size())
    summ.to_csv(os.path.join(a.out, "yield_instance_weighting_summary.csv"))
    pd.set_option("display.width", 220)
    print("Mean absolute % error of the yield prediction (lower is better):"); print(summ.to_string())
    for ref in ("last_year", "ar1_transfer"):
        print(f"\nEach method minus {ref} (points; negative = better), 95% bootstrap range over granaries; share of cases better:")
        for h, g in w.groupby(level=0):
            for m in order:
                if m == ref:
                    continue
                d = (g[m] - g[ref]).groupby(level=2).mean(); lo, hi = boot(d.values)
                print(f"  h={h} {m:17s} {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]  {(g[m] < g[ref]).mean():.0%}")


if __name__ == "__main__":
    main()
