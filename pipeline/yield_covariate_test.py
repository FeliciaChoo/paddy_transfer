"""
yield_covariate_test.py -- does a climate driver (ENSO, the Oceanic Nino Index) explain yield swings, tested on the 40-year DOSM series?

    python pipeline/yield_covariate_test.py --source external/dosm_paddy_yield_1980_2019.csv --oni external/oni_ersstv3b_1980_2015.csv

Series  average paddy yield of the ten granary areas combined, and of Malaysia (DOSM / Department of Agriculture), 1980-2015.
Step 1  remove the long-run trend (quadratic in time, on log yield) -> the yearly swing in %.
Step 2  correlate the swing with ONI in several windows (Spearman, n = 35). Six windows are tried, so a result needs p < 0.0083 (Bonferroni)
        before it counts as more than chance.
Step 3  out-of-sample: would 'last year x (1 + growth + b x ONI)' have beaten 'last year' as a ONE-YEAR-AHEAD forecast? The growth rate and b are
        learned only on years before the forecast year (rolling origin). A CONTROL with growth alone (no ONI) separates the effect of the trend
        from the effect of ENSO. Windows fixed in advance (SON of the previous year, known at planting; DJF of the year).
"""
import argparse
import numpy as np, pandas as pd
from scipy import stats


def windows(oni):
    o = oni.set_index("Year")
    w = pd.DataFrame(index=o.index[1:])
    prev = o.shift(1)
    w["SON(prev yr)"] = prev["SON"]
    w["DJF"] = o["DJF"]
    w["MAM"] = o["MAM"]
    w["JJA"] = o["JJA"]
    w["year mean"] = o.mean(axis=1)
    w["SON(prev)..MAM"] = (prev["SON"] + o["DJF"] + o["MAM"]) / 3
    return w.dropna()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", default="external/dosm_paddy_yield_1980_2019.csv")
    ap.add_argument("--oni", default="external/oni_ersstv3b_1980_2015.csv")
    a = ap.parse_args()
    src, oni = pd.read_csv(a.source).set_index("Year"), pd.read_csv(a.oni)
    W = windows(oni)
    for name, col in (("Granary areas combined", "Granary_areas_kg_ha"), ("Malaysia", "Malaysia_kg_ha")):
        y = src.loc[W.index, col].astype(float)
        ly = np.log(src[col].astype(float)); t = np.arange(len(ly))
        trend = np.polyval(np.polyfit(t, ly.values, 2), t)
        swing = pd.Series((ly.values - trend) * 100, index=ly.index).loc[W.index]
        print(f"\n== {name}: yearly swing around the trend, mean absolute {swing.abs().mean():.1f}%  (n = {len(W)} years, 1981-2015)")
        for c in W.columns:
            rho, p = stats.spearmanr(W[c], swing)
            print(f"   Spearman(ONI {c:15s}, yield swing) = {rho:+.2f}  p = {p:.3f}" + ("   <- passes Bonferroni" if p < 0.05 / 6 else ""))
        # out-of-sample 1-year-ahead forecast with a pre-chosen window
        ly_all = np.log(src[col].astype(float))
        for win in ("SON(prev yr)", "DJF"):
            err0, err1, errd = [], [], []
            for yr in W.index[12:]:
                past = [k for k in W.index if k < yr]
                d = np.array([ly_all[k] - ly_all[k - 1] for k in past]); x = W.loc[past, win].values
                b = np.polyfit(x, d, 1)
                pred0 = ly_all[yr - 1]; pred1 = ly_all[yr - 1] + np.polyval(b, W.loc[yr, win])
                predd = ly_all[yr - 1] + d.mean()                    # CONTROL: the average growth rate alone, no ONI
                errd.append(abs(np.exp(predd) - src.loc[yr, col]) / src.loc[yr, col] * 100)
                err0.append(abs(np.exp(pred0) - src.loc[yr, col]) / src.loc[yr, col] * 100)
                err1.append(abs(np.exp(pred1) - src.loc[yr, col]) / src.loc[yr, col] * 100)
            d = np.array(err1) - np.array(err0)
            print(f"   forecast with ONI {win:13s}: last year {np.mean(err0):.2f}% | last year + average growth only {np.mean(errd):.2f}% | "
                  f"growth + ONI {np.mean(err1):.2f}%   (n = {len(err0)}; ONI adds {np.mean(err1) - np.mean(errd):+.2f} points over growth alone)")


if __name__ == "__main__":
    main()
