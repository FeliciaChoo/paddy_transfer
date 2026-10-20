"""
yield_source_transfer.py -- transfer learning from a LONG related yield series (the transfer-learning idea in its natural form)

    python pipeline/yield_source_transfer.py --yield-hist data/yield_history.csv --source external/dosm_paddy_yield_1980_2019.csv --out data

Source domain  DOSM average paddy yield 1980-2019: the ten granary areas combined, and Malaysia (40 years each).
Target         the yield of each granary in 2021-2024 (7 yearly values per granary at best).
Only relative behaviour is borrowed from the source (carry-over rate, growth rate), never its level: the project's own yields sit about
17% below DOSM's granary-area figures, so the two are defined differently.
No leakage: the source ends in 2019, before every test year.

Methods (predict yield h years ahead from earlier years only)
  last_year            reference, no learning
  ar1_transfer         carry-over rate learned on the other granaries (7 years each)          <- what the dashboard uses
  src_ar1              carry-over rate learned on the 40-year granary-area series
  src_ar1_nat          the same, learned on the 40-year national series
  src_ar1_recent       learned on the last 15 years of the granary-area series only           (instance SELECTION by time)
  src_ar1_recency      learned on all 40 years, recent years weighted more (half-life 8 yrs)  (instance WEIGHTING by time)
  src_drift            last year x (1 + average h-year growth in the granary-area series)
  src_drift_recent     the same with the last 15 years only
"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yield_dynamics_transfer import GRID, ape, boot, fit_phi

HALF_LIFE, RECENT = 8.0, 15


def fit_phi_recency(s, years, h, half_life):
    best, bp = np.inf, 0.5
    w = 0.5 ** ((years[-1] - years[h + 1:]) / half_life)
    for p in GRID:
        e = [ape(s[:t - h + 1].mean() + p * (s[t - h] - s[:t - h + 1].mean()), s[t]) for t in range(h + 1, len(s))]
        if np.average(e, weights=w) < best:
            best, bp = np.average(e, weights=w), p
    return bp


def growth(s, h):
    return float(np.mean(s[h:] / s[:-h] - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--yield-hist", default="data/yield_history.csv")
    ap.add_argument("--source", default="external/dosm_paddy_yield_1980_2019.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--test-years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    a = ap.parse_args()
    yh = pd.read_csv(a.yield_hist)[["District", "Year", "Yield"]].dropna()
    S = {g: d.sort_values("Year") for g, d in yh.groupby("District")}
    src = pd.read_csv(a.source)
    G, N, yrs = src.Granary_areas_kg_ha.values.astype(float), src.Malaysia_kg_ha.values.astype(float), src.Year.values
    rows = []
    for h in a.horizons:
        phi_g, phi_n = fit_phi([G], h), fit_phi([N], h)
        phi_r = fit_phi([G[-RECENT:]], h)
        phi_w = fit_phi_recency(G, yrs, h, HALF_LIFE)
        g_all, g_rec = growth(G, h), growth(G[-RECENT:], h)
        for Y in a.test_years:
            for g, d in S.items():
                hist, act = d[d.Year <= Y - h], d[d.Year == Y]
                if len(hist) < 3 or act.empty:
                    continue
                y, truth = hist.Yield.values, act.Yield.iloc[0]
                m, last = y.mean(), y[-1]
                others = [o[o.Year <= Y - h].Yield.values for k, o in S.items() if k != g]
                preds = {"last_year": last, "ar1_transfer": m + fit_phi(others, h) * (last - m),
                         "src_ar1": m + phi_g * (last - m), "src_ar1_nat": m + phi_n * (last - m),
                         "src_ar1_recent": m + phi_r * (last - m), "src_ar1_recency": m + phi_w * (last - m),
                         "src_drift": last * (1 + g_all), "src_drift_recent": last * (1 + g_rec)}
                for k, p in preds.items():
                    rows.append(dict(horizon=h, test_year=Y, District=g, method=k, ape=ape(p, truth)))
        print(f"h={h}: learned on the source -> carry-over rate: granary areas {phi_g:.2f}, national {phi_n:.2f}, last 15 yrs {phi_r:.2f}, "
              f"recency-weighted {phi_w:.2f}; average {h}-year growth: all {g_all:+.1%}, last 15 yrs {g_rec:+.1%}")
    r = pd.DataFrame(rows)
    order = ["last_year", "ar1_transfer", "src_ar1", "src_ar1_nat", "src_ar1_recent", "src_ar1_recency", "src_drift", "src_drift_recent"]
    w = r.pivot_table(index=["horizon", "test_year", "District"], columns="method", values="ape")[order]
    summ = w.groupby(level=0).mean().round(2); summ.insert(0, "cases", w.groupby(level=0).size())
    summ.to_csv(os.path.join(a.out, "yield_source_transfer_summary.csv"))
    pd.set_option("display.width", 220)
    print("\nMean absolute % error (lower is better):"); print(summ.to_string())
    for ref in ("last_year", "ar1_transfer"):
        print(f"\nEach method minus {ref} (points; negative = better), 95% bootstrap range over granaries; share of cases better:")
        for h, g in w.groupby(level=0):
            for m in order:
                if m == ref:
                    continue
                dd = (g[m] - g[ref]).groupby(level=2).mean(); lo, hi = boot(dd.values)
                print(f"  h={h} {m:17s} {dd.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]  {(g[m] < g[ref]).mean():.0%}")


if __name__ == "__main__":
    main()
