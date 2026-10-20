"""
yield_transfer_benchmark.py -- can transfer learning improve YIELD prediction when a granary has little yield history?

    python pipeline/yield_transfer_benchmark.py --yield-csv malaysia_granary_annual_by_granary.csv --out data

Setting   Predict the yield of year Y (2022..2024) for one granary using only years < Y. The granary's own history is limited to
          its last k years (k=2 is the 'new granary' case); the other granaries' years < Y are the source data.
Inputs    that year's satellite + weather summary: peak NDVI, peak EVI, annual rain, mean temperature (granary averages).
          Everything is centred on each granary's own mean, so the models learn 'better or worse than usual', not 'which granary'.
Methods   last_year, own_mean, avg_last3         no model at all (what the Yield tab uses for the outlook)
          target_only (ridge)                    k years of the granary only
          pooled_ridge  NORMAL training          sources + the granary's k years
          tradaboost    TRANSFER LEARNING        TrAdaBoost.R2, sources + the granary's k years
Measure   mean absolute % error over granaries and test years, plus how often each model beats own_mean.
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tradaboost_r2 import TrAdaBoostR2
warnings.filterwarnings("ignore")
X_COLS = ["Peak_NDVI", "Peak_EVI", "Annual_Rain_mm", "Mean_Temp_C"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--yield-csv", default="malaysia_granary_annual_by_granary.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--test-years", nargs="+", type=int, default=[2022, 2023, 2024])
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 3])
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    g = (pd.read_csv(a.yield_csv).groupby(["District", "Year"], as_index=False)
         .agg(Yield=("Target_Yield", "first"), **{c: (c, "mean") for c in X_COLS}).dropna())
    G = sorted(g.District.unique())
    rows = []
    for k in a.ks:
        for Y in a.test_years:
            for tg in G:
                own = g[(g.District == tg) & (g.Year < Y)].sort_values("Year").tail(k)
                te = g[(g.District == tg) & (g.Year == Y)]
                src = g[(g.District != tg) & (g.Year < Y)]
                if len(own) < k or te.empty:
                    continue
                y = te.Yield.iloc[0]
                def centre(df, ref):                     # subtract the granary's own mean (from years it is allowed to know)
                    return df[X_COLS].values - ref[X_COLS].mean().values
                Xt, yt = centre(own, own), own.Yield.values - own.Yield.mean()
                Xs = np.vstack([centre(d, d.sort_values("Year").tail(k)) for _, d in src.groupby("District")])
                ys = np.concatenate([d.Yield.values - d.sort_values("Year").tail(k).Yield.mean() for _, d in src.groupby("District")])
                Xe = centre(te, own)
                mean = own.Yield.mean()
                pred = {"last_year": own.Yield.iloc[-1], "own_mean": mean,
                        "avg_last3": g[(g.District == tg) & (g.Year < Y)].sort_values("Year").Yield.tail(3).mean()}
                rd = lambda X, Yv: make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, Yv)
                pred["target_only"] = mean + rd(Xt, yt).predict(Xe)[0]
                pred["pooled_ridge"] = mean + rd(np.vstack([Xs, Xt]), np.r_[ys, yt]).predict(Xe)[0]
                tra = [TrAdaBoostR2(base_estimator=DecisionTreeRegressor(max_depth=2), n_estimators=20, random_state=s)
                       .fit(Xs, ys, Xt, yt).predict(Xe)[0] for s in range(a.seeds)]
                pred["tradaboost"] = mean + float(np.mean(tra))
                for m, p in pred.items():
                    rows.append(dict(k_years=k, test_year=Y, District=tg, method=m, actual=y, predicted=p, ape=abs(p - y) / y * 100))
    r = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    r.to_csv(os.path.join(a.out, "yield_transfer_results.csv"), index=False)
    order = ["last_year", "own_mean", "avg_last3", "target_only", "pooled_ridge", "tradaboost"]
    summ = r.pivot_table(index="k_years", columns="method", values="ape", aggfunc="mean")[order].round(2)
    pd.set_option("display.width", 200)
    print("Mean absolute % error of yield prediction (lower is better), over granaries x test years:")
    print(summ.to_string())
    w = r.pivot_table(index=["k_years", "test_year", "District"], columns="method", values="ape")
    print("\nShare of cases where the method beats the granary's own mean (no model):")
    print((w[order[3:]].lt(w["own_mean"], axis=0).groupby(level=0).mean() * 100).round(0).astype(int).to_string())
    print(f"\n{len(w)//len(a.ks)} granary-years per k; yield labels used: {len(g)}")
    summ.reset_index().to_csv(os.path.join(a.out, "yield_transfer_summary.csv"), index=False)


if __name__ == "__main__":
    main()
