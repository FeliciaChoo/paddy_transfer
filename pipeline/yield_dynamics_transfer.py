"""
yield_dynamics_transfer.py -- transfer of yield DYNAMICS between granaries (the form of transfer learning that suits scarce yield data)

    python pipeline/yield_dynamics_transfer.py --yield-hist data/yield_history.csv --resil data/resilience_indicators.csv --out data

Idea      One granary has only a few yearly yields, so a model fitted to it alone is unstable. The other granaries all show how
          yield behaves from one year to the next (how much last year's level carries forward, how fast it reverts to the
          granary's usual level). That behaviour is learned on the SOURCE granaries and applied to the target granary.
          Nothing from the target's own future is used; the sources only use years before the test year.
Methods   last_year, own_mean, avg_last3     no learning (the current Yield-tab baselines)
          ewma_own                           exponential smoothing, weight learned from the target's own few years  (NORMAL training)
          ewma_transfer                      the same smoother, weight learned from the OTHER granaries              (TRANSFER)
          ar1_transfer                       last year pulled toward the granary's own mean by a rate learned from the others (TRANSFER)
          + resilience / forecast-error signals as an extra input (see --use-resil) are tested on top of ar1_transfer.
Test      predict yield of each year 2021..2024 for each granary from earlier years only (>= 3 years of history).
Measure   mean absolute % error, win rate against last_year, and a bootstrap over granaries for the mean difference.
"""
import argparse, os
import numpy as np, pandas as pd

GRID = np.round(np.linspace(0.0, 1.0, 21), 2)


def ewma_pred(y, a):
    lvl = y[0]
    for v in y[1:]:
        lvl = a * v + (1 - a) * lvl
    return lvl


def ape(p, y):
    return abs(p - y) / y * 100


def fit_alpha(series_list, h=1):
    """smoothing weight that minimises the h-step-ahead % error over a set of yield series"""
    best, besta = np.inf, 0.5
    for a in GRID:
        e = [ape(ewma_pred(s[:t - h + 1], a), s[t]) for s in series_list for t in range(h + 1, len(s))]
        if e and np.mean(e) < best:
            best, besta = np.mean(e), a
    return besta


def fit_phi(series_list, h=1):
    """carry-over rate of last year's deviation from the granary's own mean, learned on other granaries for h steps ahead"""
    best, bestp = np.inf, 0.5
    for p in GRID:
        e = []
        for s in series_list:
            for t in range(h + 1, len(s)):
                m = np.mean(s[:t - h + 1])
                e.append(ape(m + p * (s[t - h] - m), s[t]))
        if e and np.mean(e) < best:
            best, bestp = np.mean(e), p
    return bestp


def boot(x, n=4000, seed=0):
    r = np.random.RandomState(seed); x = np.asarray(x, float)
    m = [x[r.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def run_backtest(yh, horizons=(1, 2, 3), test_years=(2021, 2022, 2023, 2024), transfer=True):
    """Predict each granary's yield h years ahead from earlier years only, for every test year.

    transfer=False runs only the methods that do NOT borrow from other granaries (last_year, own_mean, avg_last3, ewma_own).
    transfer=True adds the two transfer methods (ewma_transfer, ar1_transfer): their rates are learned on the OTHER granaries.
    yh needs the columns District, Year, Yield. Returns one row per (horizon, test year, granary, method)."""
    yh = yh[["District", "Year", "Yield"]].dropna()
    S = {g: d.sort_values("Year") for g, d in yh.groupby("District")}
    rows = []
    for h in horizons:
        for Y in test_years:
            for g, d in S.items():
                hist, act = d[d.Year <= Y - h], d[d.Year == Y]       # only years up to Y-h are known when forecasting year Y
                if len(hist) < 3 or act.empty:
                    continue
                y, truth = hist.Yield.values, act.Yield.iloc[0]
                a_own = fit_alpha([y], h) if len(y) >= h + 3 else 0.5
                m = y.mean()
                preds = {"last_year": y[-1], "own_mean": m, "avg_last3": y[-3:].mean(), "ewma_own": ewma_pred(y, a_own)}
                a_tr = p_tr = np.nan
                if transfer:
                    others = [o[o.Year <= Y - h].Yield.values for k, o in S.items() if k != g]
                    a_tr, p_tr = fit_alpha(others, h), fit_phi(others, h)
                    preds["ewma_transfer"] = ewma_pred(y, a_tr)
                    preds["ar1_transfer"] = m + p_tr * (y[-1] - m)
                for k, p in preds.items():
                    rows.append(dict(horizon=h, test_year=Y, District=g, method=k, actual=truth, predicted=p, ape=ape(p, truth),
                                     alpha_transfer=a_tr, phi_transfer=p_tr))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--yield-hist", default="data/yield_history.csv")
    ap.add_argument("--resil", default="data/resilience_indicators.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--test-years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3])
    a = ap.parse_args()
    yh = pd.read_csv(a.yield_hist)[["District", "Year", "Yield"]].dropna()
    rs = pd.read_csv(a.resil) if os.path.exists(a.resil) else None
    r = run_backtest(yh, a.horizons, a.test_years)
    r.to_csv(os.path.join(a.out, "yield_dynamics_results.csv"), index=False)
    order = ["last_year", "own_mean", "avg_last3", "ewma_own", "ewma_transfer", "ar1_transfer"]
    w = r.pivot_table(index=["horizon", "test_year", "District"], columns="method", values="ape")[order]
    cnt = w.groupby(level=0).size().rename("n")
    par = r.groupby("horizon")[["alpha_transfer", "phi_transfer"]].mean().round(2)
    print("Mean absolute % error of yield prediction h years ahead (lower is better):")
    print(pd.concat([cnt, w.groupby(level=0).mean().round(2), par], axis=1).to_string())
    print("\nTRANSFER vs NORMAL training (ar1_transfer minus ewma_own; negative = transfer better), 95% bootstrap CI over granaries:")
    for h, g in w.groupby(level=0):
        for m in ["ewma_transfer", "ar1_transfer"]:
            d = (g[m] - g["ewma_own"]).groupby(level=2).mean(); lo, hi = boot(d.values)
            print(f"  h={h}  {m:14s} {d.mean():+.2f} pp [{lo:+.2f}, {hi:+.2f}]   better in {(g[m] < g['ewma_own']).mean()*100:.0f}% of cases")
    print("\nTRANSFER vs the rule the dashboard used before (last year for years 1-2, 3-year average from year 3):")
    for h, g in w.groupby(level=0):
        cur = g["last_year"] if h <= 2 else g["avg_last3"]
        for m in ["ar1_transfer"]:
            d = (g[m] - cur).groupby(level=2).mean(); lo, hi = boot(d.values)
            print(f"  h={h}  {m:14s} {d.mean():+.2f} pp [{lo:+.2f}, {hi:+.2f}]   better in {(g[m] < cur).mean()*100:.0f}% of cases")
    w.groupby(level=0).mean().round(3).to_csv(os.path.join(a.out, "yield_dynamics_summary.csv"))

    # ---- do resilience / forecast-error signals add anything on top? residual of the transfer model vs prior-year signals
    if rs is not None:
        res = r[(r.method == "ar1_transfer") & (r.horizon == 1)].assign(resid_pct=lambda x: (x.actual - x.predicted) / x.actual * 100)
        cols = ["Stress_share", "Model_below_share", "Anomaly", "Amplitude", "Peak_NDVI"]
        rs2 = rs.rename(columns={"Year": "test_year"})[["District", "test_year"] + cols]
        # use the signal of the SAME year as the yield being predicted (a nowcast: satellite for that season is known by harvest)
        j = res.merge(rs2, on=["District", "test_year"]).dropna(subset=cols)
        from scipy import stats
        print(f"\nDoes this season's satellite resilience signal explain what the transfer model missed? (n={len(j)})")
        for c in cols:
            rho, p = stats.spearmanr(j[c], j.resid_pct)
            print(f"  Spearman({c:18s}, yield surprise %) = {rho:+.2f}  p={p:.2f}")


if __name__ == "__main__":
    main()
