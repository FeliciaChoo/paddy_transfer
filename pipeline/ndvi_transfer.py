"""
ndvi_transfer.py  --  transfer learning on a task that HAS enough data: NDVI forecasting

Why: annual yield has ~6 labels per district, so nothing can beat "same as last year".
The satellite time series, in contrast, has hundreds of decades per district. So we forecast
the district NDVI h decades ahead, train on the SOURCE districts, and ask:

    how many years of the TARGET district's own data are needed, and does transfer from the
    source districts reduce that need?

    python ndvi_transfer.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out ndvi_transfer
    python ndvi_transfer.py ... --targets "Perak Tengah" "Pasir Puteh" --horizons 1 3
    python ndvi_transfer.py ... --source "Kuala Selangor" "Sabak Bernam"      # near-source only
    python ndvi_transfer.py ... --model hgb                                   # gradient boosting instead of Ridge

Protocol (strictly temporal, no future information)
  * test years  = --test-years (default 2022 2023 2024); the target district is evaluated there
  * train years = --min-year (default 2019, because 2018 optical is almost empty) .. first test year - 1,
    and the forecast TARGET time of every training row is also before the first test decade
  * k = number of target years just before the test period that the target district may use
    (k=0 -> zero-shot, k=1 -> 2021, k=2 -> 2020-21, k=3 -> 2019-21)
  * NDVI labels are NOT yield labels: districts that share a yield series (Kuala Selangor / Sabak Bernam,
    Pasir Puteh / Bachok, ...) are still independent satellite series, so they ARE all valid sources.

Scenarios
  persistence   NDVI stays at the last observed value
  clim_delta    seasonal baseline: change = climatology(phase+h) - climatology(phase), phase = decades since
                the last planting trough; climatology from the target's k years (k>=1) or from the sources (k=0)
  source_only   model trained on the source districts only (zero-shot), same for every k
  target_only   model trained on the target's k years only
  pooled        sources + target k years, equal weights
  weighted      same, target rows weighted x --target-weight
  residual      source model + a small Ridge correction fitted on the target k years

Outputs in --out: ndvi_transfer_results.csv and ndvi_transfer_<target>_h<h>.png
Needs: pandas numpy scikit-learn (matplotlib optional)
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

FFILL_LIMIT = 4          # last observed NDVI is carried forward at most 4 decades (causal, no future values)
FEATS = ["ndvi0", "ndvi1", "ndvi2", "ndvi3", "age", "sar0", "sar1", "rain3", "tmax", "sol",
         "phase", "ph_sin", "ph_cos"]
SCEN = ["persistence", "clim_delta", "source_only", "target_only", "pooled", "weighted", "residual"]


# ----------------------------------------------------------------------
# data
# ----------------------------------------------------------------------
def load_series(clean_csv, min_pix):
    cols = ["District", "Year", "Decade_N", "NDVI_obs_clean", "SAR_clean", "Daily_Rain_mm",
            "Mean_Max_Temp_C", "Solar_Radiation_MJ"]
    d = pd.read_csv(clean_csv, usecols=cols)
    ts = d.groupby(["District", "Year", "Decade_N"]).agg(
        ndvi=("NDVI_obs_clean", "mean"), npix=("NDVI_obs_clean", "count"), sar=("SAR_clean", "mean"),
        rain=("Daily_Rain_mm", "mean"), tmax=("Mean_Max_Temp_C", "mean"),
        sol=("Solar_Radiation_MJ", "mean")).reset_index()
    ts.loc[ts.npix < min_pix, "ndvi"] = np.nan          # need >= min_pix really observed pixels in the decade
    return ts


def make_rows(ts, ph, h):
    """One row per (district, forecast origin i): features known at i, target = NDVI at i+h."""
    y0, y1 = int(ts.Year.min()), int(ts.Year.max())
    n = (y1 - y0 + 1) * 36
    out = []
    for dist, g in ts.groupby("District"):
        if dist not in ph.index:
            print(f"  (no phenology row for {dist}; skipped)")
            continue
        t = ((g.Year - y0) * 36 + g.Decade_N - 1).astype(int).values
        full = {}
        for c in ["ndvi", "sar", "rain", "tmax", "sol"]:
            s = pd.Series(np.nan, index=np.arange(n))
            s.loc[t] = g[c].values
            full[c] = s
        nd = full["ndvi"].values
        ff = full["ndvi"].ffill(limit=FFILL_LIMIT).values
        last = pd.Series(np.where(~np.isnan(nd), np.arange(n), np.nan)).ffill().values
        age = np.arange(n) - last
        sar = full["sar"].ffill(limit=3).values
        rain3 = full["rain"].rolling(3, min_periods=3).sum().values
        dec = np.arange(n) % 36 + 1
        a, b = int(ph.loc[dist, "trough_A"]), int(ph.loc[dist, "trough_B"])
        phase = np.minimum((dec - a) % 36, (dec - b) % 36).astype(float)   # decades since last planting trough
        i = np.arange(3, n - h)
        out.append(pd.DataFrame({
            "District": dist, "year": y0 + i // 36, "t_target": i + h,
            "ndvi0": ff[i], "ndvi1": ff[i - 1], "ndvi2": ff[i - 2], "ndvi3": ff[i - 3], "age": age[i],
            "sar0": sar[i], "sar1": sar[i - 1], "rain3": rain3[i], "tmax": full["tmax"].values[i],
            "sol": full["sol"].values[i], "phase": phase[i],
            "ph_sin": np.sin(2 * np.pi * phase[i] / 18), "ph_cos": np.cos(2 * np.pi * phase[i] / 18),
            "phase_f": phase[i + h], "y_abs": nd[i + h], "delta": nd[i + h] - ff[i]}))
    rows = pd.concat(out, ignore_index=True)
    return rows.dropna(subset=FEATS + ["delta"]).reset_index(drop=True), y0


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------
def fit_model(kind, X, y, w=None):
    if kind == "hgb":
        m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.1, max_iter=150, random_state=0)
        return m.fit(X, y, sample_weight=w)
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return m.fit(X, y, ridge__sample_weight=w)


def clim_delta(train, test):
    clim = train.groupby("phase_f")["y_abs"].mean()
    now, fut = test["phase"].map(clim), test["phase_f"].map(clim)
    return (fut - now).fillna(0.0).values


def evaluate(rows, y0, target, sources, test_years, ks, kind, tw, min_year, h):
    t0 = min(test_years)
    cut = (t0 - y0) * 36                                  # first decade index of the test period
    test = rows[(rows.District == target) & rows.year.isin(test_years)]
    train = rows[(rows.year >= min_year) & (rows.year < t0) & (rows.t_target < cut)]
    src, tgt_all = train[train.District.isin(sources)], train[train.District == target]
    if test.empty or src.empty:
        print(f"  {target}: no test rows or no source rows (check years / districts)")
        return []
    Xte, yte = test[FEATS], test["delta"].values
    ms = fit_model(kind, src[FEATS], src["delta"])
    pred_src = ms.predict(Xte)
    res = []
    for k in ks:
        tgt = tgt_all[tgt_all.year >= t0 - k] if k > 0 else tgt_all.iloc[0:0]
        preds = {"persistence": np.zeros(len(test)),
                 "clim_delta": clim_delta(tgt if len(tgt) else src, test),
                 "source_only": pred_src}
        if len(tgt) >= 10:
            preds["target_only"] = fit_model(kind, tgt[FEATS], tgt["delta"]).predict(Xte)
            pool = pd.concat([src, tgt])
            preds["pooled"] = fit_model(kind, pool[FEATS], pool["delta"]).predict(Xte)
            w = np.r_[np.ones(len(src)), np.full(len(tgt), float(tw))]
            preds["weighted"] = fit_model(kind, pool[FEATS], pool["delta"], w).predict(Xte)
            r = tgt["delta"].values - ms.predict(tgt[FEATS])
            mr = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(tgt[FEATS], r)
            preds["residual"] = pred_src + mr.predict(Xte)
        for sc, p in preds.items():
            err = np.abs(p - yte)
            res.append(dict(target=target, horizon=h, k_years=k, scenario=sc, MAE=err.mean(),
                            RMSE=np.sqrt((err ** 2).mean()), n=len(err)))
    return res


def add_skill(df):
    key = ["target", "horizon", "k_years"]
    for name, sc in [("skill_vs_persistence", "persistence"), ("skill_vs_clim", "clim_delta")]:
        base = df[df.scenario == sc].set_index(key)["MAE"]
        df[name] = 1 - df["MAE"] / df.set_index(key).index.map(base).values
    return df.round(4)


def plot(df, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("plots skipped:", e)
        return
    for (tg, h), g in df.groupby(["target", "horizon"]):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for sc in SCEN:
            s = g[g.scenario == sc].sort_values("k_years")
            if sc in ("persistence", "clim_delta") or s.k_years.nunique() == 1:
                ax.plot(s.k_years, s.MAE, "--" if sc != "source_only" else "-", marker="o", label=sc)
            else:
                ax.plot(s.k_years, s.MAE, marker="o", label=sc)
        ax.set_xlabel("years of target data used (k)")
        ax.set_ylabel("MAE of NDVI forecast")
        ax.set_title(f"{tg}, horizon {h} decades", fontsize=9)
        ax.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"ndvi_transfer_{tg.replace(' ', '_')}_h{h}.png"), dpi=130)
        plt.close()


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--pheno", default="eda/eda_phenology.csv")
    ap.add_argument("--out", default="ndvi_transfer")
    ap.add_argument("--targets", nargs="+", default=["Perak Tengah", "Pasir Puteh"])
    ap.add_argument("--source", nargs="+", default=None, help="default: every other district in the data")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 3], help="forecast horizon in decades (10 days)")
    ap.add_argument("--test-years", nargs="+", type=int, default=[2022, 2023, 2024])
    ap.add_argument("--min-year", type=int, default=2019)
    ap.add_argument("--max-k", type=int, default=3)
    ap.add_argument("--min-pix", type=int, default=5, help="min observed pixels for a district-decade NDVI")
    ap.add_argument("--target-weight", type=float, default=5.0)
    ap.add_argument("--model", choices=["ridge", "hgb"], default="ridge")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ts = load_series(a.clean, a.min_pix)
    ph = pd.read_csv(a.pheno).set_index("District")
    avail = sorted(set(ts.District) & set(ph.index))
    miss = [t for t in a.targets if t not in avail]
    if miss:
        raise SystemExit(f"Target(s) not found (need data AND a phenology row): {miss}. Available: {avail}")
    print("districts:", avail)
    use = ts.groupby("District")["ndvi"].apply(lambda s: int(s.notna().sum()))
    print("decades with a usable NDVI (>= --min-pix observed pixels):", use.to_dict())
    if use.max() < 36:
        raise SystemExit("Almost no usable NDVI decades. Lower --min-pix (districts with few pixels need a smaller value).")

    allres = []
    for h in a.horizons:
        rows, y0 = make_rows(ts, ph, h)
        for tg in a.targets:
            sources = [s for s in (a.source or avail) if s in avail and s not in a.targets]
            print(f"\nhorizon {h} decades | target {tg} | {len(sources)} sources: {sources}")
            allres += evaluate(rows, y0, tg, sources, a.test_years, list(range(0, a.max_k + 1)),
                               a.model, a.target_weight, a.min_year, h)
    if not allres:
        raise SystemExit("No results (check --test-years / --min-year against the years in your data).")
    df = add_skill(pd.DataFrame(allres))
    df.to_csv(os.path.join(a.out, "ndvi_transfer_results.csv"), index=False)
    plot(df, a.out)

    pd.set_option("display.width", 200)
    for (tg, h), g in df.groupby(["target", "horizon"]):
        print(f"\n=== {tg} | horizon {h} decades | MAE of NDVI (lower = better), n = {int(g.n.iloc[0])} test decades ===")
        print(g.pivot(index="scenario", columns="k_years", values="MAE").reindex(SCEN).round(4).to_string())
        print("skill vs clim_delta (>0 beats the seasonal baseline):")
        print(g.pivot(index="scenario", columns="k_years", values="skill_vs_clim").reindex(SCEN[2:]).round(2).to_string())
    print("\nRead it as: does source_only / residual / weighted at k=0..1 reach the MAE that target_only needs "
          "k=2..3 years to reach? That gap is the benefit of transfer. Anything that does not beat "
          "persistence AND clim_delta is not a real gain.")


if __name__ == "__main__":
    main()