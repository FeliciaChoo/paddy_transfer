"""
yield_outlook_transfer.py -- the yield outlook built with TRANSFER LEARNING of yield dynamics (replaces the baseline outlook)

    python pipeline/yield_dynamics_transfer.py --out data        # 1. evidence: back-test (also gives the uncertainty bands)
    python pipeline/yield_outlook_transfer.py  --out data        # 2. the outlook the dashboard reads

How it works  A granary has too few yearly yields to learn from alone. The OTHER granaries show how yield carries over from one
              year to the next. For each horizon h (1, 2, 3 years ahead) the carry-over rate phi_h is learned on the other granaries
              (leave-one-granary-out) and applied to the target:
                    outlook = granary mean + phi_h x (last yield - granary mean)
              phi near 1 = 'next year looks like last year'; phi near 0.5 = 'half way back to the usual level'.
Band          the 80th percentile of the absolute % error this method made in the back-test at that horizon (so the band is earned,
              not assumed). Year 4 has no back-test (too little history), so it uses the year-3 rate and band and says so.
Writes        yield_outlook.csv (same columns as before), yield_backtest.csv (adds the transfer rows), yield_outlook_baseline.csv (a copy of the old outlook).
"""
import argparse, os, shutil, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yield_dynamics_transfer import fit_phi, fit_alpha, GRID  # noqa: E402

LABEL = {"ar1_transfer": "transfer-learned carry-over (rate learned on the other granaries)",
         "ewma_own": "normal training (smoothing learned on the granary alone)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--yield-hist", default="data/yield_history.csv")
    ap.add_argument("--results", default="data/yield_dynamics_results.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 4])
    a = ap.parse_args()
    if not os.path.exists(a.results):
        raise SystemExit("Run pipeline/yield_dynamics_transfer.py first: it provides the back-test the bands come from.")
    yh = pd.read_csv(a.yield_hist)[["District", "Year", "Yield"]].dropna()
    S = {g: d.sort_values("Year") for g, d in yh.groupby("District")}
    bt = pd.read_csv(a.results)
    tl = bt[bt.method == "ar1_transfer"]
    band = tl.groupby("horizon").ape.apply(lambda s: np.percentile(s, 80))
    mape = tl.groupby("horizon").ape.mean()
    top = int(band.index.max())

    out = []
    for g, d in S.items():
        y, last_year = d.Yield.values, int(d.Year.iloc[-1])
        others = [o.Yield.values for k, o in S.items() if k != g]
        m = y.mean()
        for h in a.horizons:
            hh = min(h, top)                                   # no back-test beyond `top` years: reuse its rate and band
            phi = fit_phi(others, hh)
            c = m + phi * (y[-1] - m)
            b = float(band[hh])
            note = "" if h <= top else f"; no back-test at {h} years, so the {top}-year rate and band are reused"
            out.append(dict(District=g, Last_year=last_year, Last_yield=y[-1], Forecast_year=last_year + h, Horizon_years=h,
                            Central=c, Band_pct=b, Low=c * (1 - b / 100), High=c * (1 + b / 100),
                            Method=f"{LABEL['ar1_transfer']}, rate {phi:.2f}{note}"))
    new = pd.DataFrame(out).round(1)

    p = os.path.join(a.out, "yield_outlook.csv")
    base = os.path.join(a.out, "yield_outlook_baseline.csv")
    if os.path.exists(p) and not os.path.exists(base):
        shutil.copy(p, base)                                    # keep the old baseline once
    new.to_csv(p, index=False)

    # ---- add the transfer rows to the back-test table the dashboard shows
    pb = os.path.join(a.out, "yield_backtest.csv")
    old = pd.read_csv(pb) if os.path.exists(pb) else pd.DataFrame(columns=["Horizon_years", "Method", "n", "MAPE_pct", "P80_abs_error_pct"])
    old = old[~old.Method.isin(LABEL.values())]
    add = []
    for key, lab in LABEL.items():
        for h, g in bt[bt.method == key].groupby("horizon"):
            add.append(dict(Horizon_years=int(h), Method=lab, n=len(g), MAPE_pct=g.ape.mean(), P80_abs_error_pct=np.percentile(g.ape, 80)))
    pd.concat([old, pd.DataFrame(add)], ignore_index=True).round(2).sort_values(["Horizon_years", "MAPE_pct"]).to_csv(pb, index=False)

    pd.set_option("display.width", 200)
    print("phi learned per horizon (average over granaries):", {h: round(float(np.mean([fit_phi([o.Yield.values for k, o in S.items() if k != g], min(h, top)) for g in S])), 2) for h in a.horizons})
    print("band (P80 error %):", {int(h): round(float(v), 1) for h, v in band.items()}, "| back-test MAPE %:", {int(h): round(float(v), 1) for h, v in mape.items()})
    print(new[new.District == list(S)[0]].to_string(index=False))


if __name__ == "__main__":
    main()
