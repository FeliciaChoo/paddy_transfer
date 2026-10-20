"""
resilience_from_forecast.py -- crop resilience measured THROUGH the NDVI forecast, plus the recent weather / water / soil context.

    python pipeline/resilience_from_forecast.py --clean eda/DailyGranary_clean.csv --out data

Why the forecast: the NDVI model says what greenness to EXPECT (with an 80% range). Resilience research on satellite time series
splits resilience into RESISTANCE (how little the vegetation is knocked down by a shock) and RECOVERY (how fast it comes back).
Here a 'shock' is a decade in which the observed NDVI falls below the forecast range, measured on the held-out forecasts only
(chosen_predictions.csv, 10-day horizon), so nothing is fitted to the data it is scored on.

Per granary
  Shock_share            share of decades below the forecast range                    (resistance, lower is better)
  Shock_depth            average NDVI shortfall below the range when it happens       (resistance, lower is better)
  Recovery_decades       average length of a run of below-range decades               (recovery, shorter is better)
  Peak_NDVI_CV           year-to-year variation of the seasonal peak NDVI             (stability, lower is better)
  Resilience_index       0-100, the average of the four ranks (100 = most resilient OF THESE GRANARIES; it is relative)
  Resilience_class       higher / middle / lower third
  Typical_recovery_days  Recovery_decades x 10, used by the dashboard's 'recovery watch'
Per granary, latest context (needs --clean)  -> context_signals.csv
  recent 3-decade mean of temperature, rain, solar radiation, MNDWI, LSWI, SAR VV, soil pH, each with a z-score against the
  granary's own history for the same time of year.
"""
import argparse, os
import numpy as np, pandas as pd


def runs(flags, dates, max_gap=12):
    """lengths of runs of consecutive True decades (a gap longer than max_gap days breaks a run)"""
    out, cur, prev = [], 0, None
    for f, d in zip(flags, dates):
        if prev is not None and (d - prev).days > max_gap:
            if cur:
                out.append(cur)
            cur = 0
        if f:
            cur += 1
        elif cur:
            out.append(cur)
            cur = 0
        prev = d
    if cur:
        out.append(cur)
    return out


def pct_rank(s, higher_is_better):
    r = s.rank(pct=True, method="average")
    return (r if higher_is_better else 1 - r + 1 / len(s)) * 100


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--pred", default="data/chosen_predictions.csv")
    ap.add_argument("--resil", default="data/resilience_indicators.csv")
    ap.add_argument("--yield-hist", default="data/yield_history.csv")
    ap.add_argument("--clean", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--horizon", type=int, default=1)
    a = ap.parse_args()

    p = pd.read_csv(a.pred, parse_dates=["target_date"])
    p = p[p.horizon == a.horizon].dropna(subset=["observed", "lower"]).sort_values(["District", "target_date"])
    rows = []
    for g, d in p.groupby("District"):
        below = (d.observed < d.lower).values
        short = np.where(below, d.lower.values - d.observed.values, 0.0)
        rn = runs(below, list(d.target_date))
        rows.append(dict(District=g, n_decades=len(d), Shock_share=below.mean(),
                         Shock_depth=short[below].mean() if below.any() else 0.0,
                         Recovery_decades=float(np.mean(rn)) if rn else 0.0, Longest_run=max(rn) if rn else 0))
    r = pd.DataFrame(rows).set_index("District")
    rs = pd.read_csv(a.resil)
    cv = rs.dropna(subset=["Peak_NDVI"]).groupby("District").Peak_NDVI.agg(lambda s: s.std() / s.mean())
    r["Peak_NDVI_CV"] = cv
    comp = pd.concat([pct_rank(r.Shock_share, False), pct_rank(r.Shock_depth, False),
                      pct_rank(r.Recovery_decades, False), pct_rank(r.Peak_NDVI_CV, False)], axis=1)
    r["Resilience_index"] = comp.mean(axis=1)
    q = r.Resilience_index.rank(pct=True)
    r["Resilience_class"] = np.where(q > 2 / 3, "higher", np.where(q > 1 / 3, "middle", "lower"))
    r["Typical_recovery_days"] = (r.Recovery_decades.clip(lower=1) * 10).round(0)
    r = r.reset_index().round(4)
    r.to_csv(os.path.join(a.out, "resilience_forecast.csv"), index=False)
    pd.set_option("display.width", 220)
    print(r.sort_values("Resilience_index", ascending=False).to_string(index=False))

    # ---- is the index related to how STABLE the reported yield is? (10 granaries: a plausibility check, not a proof)
    from scipy import stats
    yh = pd.read_csv(a.yield_hist)
    ycv = yh.groupby("District").Yield.agg(lambda s: s.std() / s.mean()).rename("Yield_CV")
    yd = yh.sort_values(["District", "Year"]).groupby("District").Yield.apply(lambda s: s.pct_change().abs().mean() * 100).rename("Mean_abs_change_pct")
    j = r.set_index("District").join([ycv, yd])
    print("\nPlausibility check vs yield stability (n=10 granaries; lower yield variation = more stable):")
    for c in ["Resilience_index", "Shock_share", "Recovery_decades"]:
        for t in ["Yield_CV", "Mean_abs_change_pct"]:
            rho, pv = stats.spearmanr(j[c], j[t])
            print(f"  Spearman({c:17s}, {t:20s}) = {rho:+.2f}  p={pv:.2f}")

    # ---- recent weather / water / soil context against each granary's own normal for that time of year
    if os.path.exists(a.clean):
        c = pd.read_csv(a.clean, usecols=lambda x: x in ["District", "Year", "SubSeasonal_Period", "Start_Date", "Decade_N", "Mean_Max_Temp_C",
                        "Mean_Min_Temp_C", "Decade_Rain_mm", "Daily_Rain_mm", "Solar_Radiation_MJ", "MNDWI", "LSWI", "SAR_VV_dB",
                        "NDVI", "Soil_pH", "Optical_Observed"])
        rain = "Decade_Rain_mm" if "Decade_Rain_mm" in c.columns else "Daily_Rain_mm"
        feats = ["Mean_Max_Temp_C", "Mean_Min_Temp_C", rain, "Solar_Radiation_MJ", "MNDWI", "LSWI", "SAR_VV_dB"]
        c["dec"] = c["Decade_N"] if "Decade_N" in c.columns else c["SubSeasonal_Period"]
        gd = c.groupby(["District", "Year", "dec"], as_index=False)[feats + ["Soil_pH"]].mean()
        out = []
        for g, d in gd.groupby("District"):
            d = d.sort_values(["Year", "dec"])
            last = d.iloc[-3:]
            row = dict(District=g, Latest_year=int(last.Year.iloc[-1]), Latest_decade=int(last.dec.iloc[-1]), Soil_pH=d.Soil_pH.mean())
            for f in feats:
                clim = d[d.dec.isin(last.dec)].groupby("dec")[f].agg(["mean", "std"])
                z = [(r_[f] - clim.loc[r_.dec, "mean"]) / clim.loc[r_.dec, "std"] if clim.loc[r_.dec, "std"] > 0 else 0.0
                     for _, r_ in last.iterrows()]
                row[f] = last[f].mean()
                row[f + "_z"] = float(np.mean(z))
            out.append(row)
        ctx = pd.DataFrame(out).rename(columns={rain: "Rain_mm", rain + "_z": "Rain_mm_z"}).round(3)
        ctx.to_csv(os.path.join(a.out, "context_signals.csv"), index=False)
        print("\ncontext_signals.csv:", len(ctx), "granaries"); print(ctx[["District", "Mean_Max_Temp_C", "Mean_Max_Temp_C_z", "Mean_Min_Temp_C_z",
              "Rain_mm_z", "Solar_Radiation_MJ_z", "MNDWI_z", "LSWI_z", "Soil_pH"]].to_string(index=False))
    else:
        print("no --clean file: context_signals.csv not written")


if __name__ == "__main__":
    main()
