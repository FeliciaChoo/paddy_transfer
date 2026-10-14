"""
prepare_app_data.py  --  build the data files for the Streamlit app from your existing CSVs

    python prepare_app_data.py --yield-csv malaysia_granary_annual_ml_ready.csv ^
        --pheno eda/eda_phenology.csv --final-dir final_model ^
        --clean eda/DailyGranary_clean.csv --out data

Inputs
    --yield-csv   annual yield file (District, Year, Longitude, Latitude, Target_Yield, Peak_NDVI ...)
    --pheno       eda/eda_phenology.csv
    --final-dir   folder written by final_model.py (latest_forecast, alerts_final_year, chosen_predictions,
                  fold_results, chosen_config)
    --clean       eda/DailyGranary_clean.csv (optional, needed for the NDVI resilience indicators)
Outputs in --out (the app reads these)
    districts.csv, granary_footprints.csv, yield_history.csv, yield_outlook.csv, yield_backtest.csv,
    phenology.csv, latest_forecast.csv, alerts_final_year.csv, chosen_predictions.csv, model_skill.csv,
    condition_summary.csv, ndvi_district_decade.csv, resilience_indicators.csv, ndvi_yield_relation.csv,
    advisory_rules.csv (written once; EDIT it by hand. Re-created only if it has the old format)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr

JUMP_PCT = 20.0              # a year-to-year yield change of this size or more is flagged for checking
TREND_DELTA = 0.05           # 30-day NDVI change beyond this = rising / falling
RECENT_DECADES = 4           # recent-status window (latest decades of the final year)
HORIZONS = [1, 2, 3, 4]      # yield outlook years ahead
MIN_DECADES = 12             # a district-year needs this many observed decades to get NDVI indicators

# rule_id, audience, focus, stage_contains ('|' = any of), status, trend, signal, priority, message, why
FARMER_RULES = [
    ("F01", "farmer", "checks", "any", "behind", "any", "any", 1,
     "Inspect the fields soon: check water level and drainage, signs of nutrient deficiency, pests or disease, and whether planting was late or uneven.",
     "Recent NDVI is below the range expected from earlier years [E3]. Low greenness has several possible causes and satellite data cannot tell which."),
    ("F02", "farmer", "checks", "heading|grain", "behind", "any", "any", 1,
     "Check first for disease, pest damage and water stress, and speak to the extension officer.",
     "Low greenness around heading or grain filling can reduce grain yield [E1, E3]."),
    ("F03", "farmer", "actions", "any", "ahead", "any", "any", 2,
     "Greener than expected is usually favourable; confirm it is not caused by an earlier planting date or by weeds.",
     "NDVI above the forecast range can also come from causes that are not crop health [E3]."),
    ("F04", "farmer", "planning", "maturity|grain", "on track|ahead", "falling", "any", 2,
     "Start planning harvest, drying and transport for the coming weeks.",
     "NDVI is expected to fall as the crop matures [E1, E2]."),
    ("F05", "farmer", "actions", "germination|seedling|tillering", "on track|ahead", "rising", "any", 2,
     "Keep to the water and fertiliser schedule recommended by your extension officer.",
     "The canopy is developing as expected [E2, E3]."),
    ("F06", "farmer", "planning", "fallow|turnaround", "any", "any", "any", 2,
     "Use this period for land preparation, drainage maintenance and planning water supply for the next planting.",
     "Fields are probably between crops [E1]."),
    ("F07", "farmer", "checks", "panicle|booting|max tillering", "on track", "falling", "any", 3,
     "Check water supply and nutrient status, and scout for pests.",
     "NDVI is expected to fall during an important growth stage [E1, E2]."),
    ("F08", "farmer", "actions", "heading|panicle|booting|flowering", "any", "any", "any", 2,
     "Keep the water supply steady and scout the fields for pests and disease.",
     "This is a sensitive stage for yield [E1]."),
    ("F09", "farmer", "any", "any", "any", "any", "weak", 3,
     "Treat the growth stage as approximate for this granary.",
     "Planting dates are mixed here, so the stage estimate is less reliable [E7]."),
    ("F99", "farmer", "any", "any", "any", "any", "any", 9,
     "Confirm with the local extension officer before acting.",
     "This is decision support from satellite data and a statistical forecast, not a field diagnosis."),
]


def footprint(pts, pad=0.02):
    """outline of a granary's sampled paddy pixels (convex hull pushed outward a little); NOT an official boundary"""
    pts = np.unique(np.round(pts, 4), axis=0)
    c = pts.mean(axis=0)
    poly = None
    if len(pts) >= 3:
        try:
            poly = pts[ConvexHull(pts).vertices]
        except Exception:
            poly = None
    if poly is None or len(poly) < 3:
        lo, hi = pts.min(0) - pad, pts.max(0) + pad
        poly = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    else:
        v = poly - c
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1
        poly = poly + pad * v / n
    return json.dumps(np.round(poly, 4).tolist())


def yield_backtest(Yr, years):
    """expanding back-test of simple yield baselines at 1-4 years ahead (one district per distinct series)"""
    meth = {"last year": lambda h_: h_.iloc[-1], "3-year average": lambda h_: h_.iloc[-3:].mean(),
            "average of all past years": lambda h_: h_.mean()}
    rows = []
    for h in HORIZONS:
        for m in list(meth) + ["trend (last 5 years)"]:
            errs = []
            for d in Yr.index:
                for o in range(2020, 2024):
                    t = o + h
                    if t > max(years):
                        continue
                    hist = Yr.loc[d, [y for y in years if y <= o]].dropna()
                    if len(hist) < 3:
                        continue
                    if m.startswith("trend"):
                        s = hist.iloc[-5:]
                        pred = np.polyval(np.polyfit(np.arange(len(s)), s.values, 1), len(s) - 1 + h)
                    else:
                        pred = meth[m](hist)
                    errs.append(abs(pred - Yr.loc[d, t]) / Yr.loc[d, t] * 100)
            rows.append(dict(Horizon_years=h, Method=m, n=len(errs), MAPE_pct=np.mean(errs),
                             P80_abs_error_pct=np.percentile(errs, 80)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--yield-csv", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--final-dir", required=True)
    ap.add_argument("--clean", default=None)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    W = lambda df, name: df.to_csv(os.path.join(a.out, name), index=False)  # noqa: E731

    # ---------- districts, footprints, yield history ----------
    cols = ["District", "Year", "Longitude", "Latitude", "Target_Yield", "Peak_NDVI", "Peak_EVI", "Annual_Rain_mm",
            "Mean_Temp_C"]
    y = pd.read_csv(a.yield_csv, usecols=lambda c: c in cols)
    dist = y.groupby("District").agg(Latitude=("Latitude", "mean"), Longitude=("Longitude", "mean"),
                                     n_pixels=("Year", "size")).reset_index()
    W(dist.round(5), "districts.csv")
    fp = [dict(District=d, polygon=footprint(g[["Longitude", "Latitude"]].drop_duplicates().values))
          for d, g in y.groupby("District")]
    W(pd.DataFrame(fp), "granary_footprints.csv")

    Y = y.groupby(["District", "Year"])["Target_Yield"].first().unstack()
    years = [int(c) for c in Y.columns]
    key = Y.apply(lambda r: tuple(r.round(1)), axis=1)
    groups = {}
    for d, k in key.items():
        groups.setdefault(k, []).append(d)
    shared = {d: ", ".join(x for x in v if x != d) for v in groups.values() for d in v}
    rows = []
    for d in Y.index:
        for yr in Y.columns:
            v, prev = Y.loc[d, yr], (Y.loc[d, yr - 1] if (yr - 1) in Y.columns else np.nan)
            ch = (v / prev - 1) * 100 if pd.notna(prev) else np.nan
            rows.append(dict(District=d, Year=int(yr), Yield=v, Change_pct=ch,
                             Flag_large_jump=bool(pd.notna(ch) and abs(ch) >= JUMP_PCT),
                             Shares_series_with=shared[d]))
    yh = pd.DataFrame(rows)
    yh["Yield_z"] = yh.groupby("District")["Yield"].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    W(yh.round(3), "yield_history.csv")

    # ---------- yield outlook (baseline, with a back-tested uncertainty band) ----------
    rep = [v[0] for v in groups.values()]
    bt = yield_backtest(Y.loc[rep], years)
    W(bt.round(2), "yield_backtest.csv")
    ly = bt[bt.Method == "last year"].set_index("Horizon_years")["P80_abs_error_pct"]
    band = {h: float(ly.loc[:h].max()) for h in HORIZONS}          # never narrower than a shorter horizon
    out = []
    for d in Y.index:
        s = Y.loc[d].dropna()
        last_year, last = int(s.index[-1]), float(s.iloc[-1])
        for h in HORIZONS:
            central = last if h <= 2 else float(s.iloc[-3:].mean())  # choice supported by the back-test
            out.append(dict(District=d, Last_year=last_year, Last_yield=last, Forecast_year=last_year + h,
                            Horizon_years=h, Central=central, Band_pct=band[h], Low=central * (1 - band[h] / 100),
                            High=central * (1 + band[h] / 100),
                            Method="same as last year" if h <= 2 else "average of the last 3 years"))
    W(pd.DataFrame(out).round(1), "yield_outlook.csv")

    # ---------- does annual NDVI / weather track annual yield? (descriptive, one district per series) ----------
    D = y.groupby(["District", "Year"]).mean(numeric_only=True).reset_index()
    D = D[D.District.isin(rep)]
    rel = []
    for c, label in [("Peak_NDVI", "Peak NDVI"), ("Peak_EVI", "Peak EVI"), ("Annual_Rain_mm", "Annual rainfall"),
                     ("Mean_Temp_C", "Mean temperature")]:
        if c not in D:
            continue
        Z = D[["District", "Year", c, "Target_Yield"]].copy()
        for k in (c, "Target_Yield"):
            Z[k + "_z"] = Z.groupby("District")[k].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
        ok = Z.dropna(subset=[c + "_z", "Target_Yield_z"])
        rho, p = spearmanr(ok[c + "_z"], ok["Target_Yield_z"])
        per = [spearmanr(g[c], g["Target_Yield"])[0] for _, g in Z.groupby("District") if g[c].nunique() > 2]
        rel.append(dict(Indicator=label, Pooled_rho=rho, p_value=p, n_granary_years=len(ok),
                        Median_rho_within_granary=float(np.nanmedian(per)),
                        Granaries_positive=f"{sum(v > 0 for v in per)}/{len(per)}"))
    W(pd.DataFrame(rel).round(3), "ndvi_yield_relation.csv")

    # ---------- phenology, forecasts, alerts, predictions ----------
    ph = pd.read_csv(a.pheno)
    W(ph, "phenology.csv")
    lf = pd.read_csv(os.path.join(a.final_dir, "latest_forecast.csv"))
    W(lf, "latest_forecast.csv")
    al = pd.read_csv(os.path.join(a.final_dir, "alerts_final_year.csv"))
    W(al[["horizon", "District", "origin_date", "target_date", "forecast", "lower", "upper", "observed",
          "stage_at_target", "status"]], "alerts_final_year.csv")
    cp = pd.read_csv(os.path.join(a.final_dir, "chosen_predictions.csv"))
    W(cp[["horizon", "test_year", "District", "origin_date", "target_date", "ndvi_last", "forecast", "lower", "upper",
          "observed", "inside_interval"]], "chosen_predictions.csv")
    fr = pd.read_csv(os.path.join(a.final_dir, "fold_results.csv"))
    cc = pd.read_csv(os.path.join(a.final_dir, "chosen_config.csv"))
    sk = []
    for _, c in cc.iterrows():
        r = fr[(fr.horizon == c.horizon) & (fr.model == c.model)]
        for yr, g in r.groupby("test_year"):
            m, p = g[g.config == c.config].MAE.mean(), g[g.config == "persist"].MAE.mean()
            sk.append(dict(horizon=int(c.horizon), test_year=int(yr), model=f"{c.model} ({c.config})",
                           MAE_model=m, MAE_persistence=p, skill_vs_persistence=1 - m / p))
    W(pd.DataFrame(sk).round(4), "model_skill.csv")

    # ---------- one row per district: current condition ----------
    cs = []
    for d in sorted(lf.District.unique()):
        l1, l3 = lf[(lf.District == d) & (lf.horizon_decades == 1)], lf[(lf.District == d) & (lf.horizon_decades == 3)]
        if l1.empty or l3.empty:
            continue
        l1, l3 = l1.iloc[0], l3.iloc[0]
        delta = l3.forecast_ndvi - l1.last_ndvi
        recent = al[(al.District == d) & (al.horizon == 1)].copy()
        recent["origin_date"] = pd.to_datetime(recent["origin_date"])
        recent = recent.sort_values("origin_date").tail(RECENT_DECADES)
        below = int(recent.status.str.startswith("BELOW").sum())
        above = int((recent.status == "above forecast range").sum())
        full = al[al.District == d]
        conf, amp = ph.loc[ph.District == d, "confidence"], ph.loc[ph.District == d, "NDVI_amplitude"]
        cs.append(dict(District=d, As_of=l1.from_date, Last_NDVI=l1.last_ndvi, Forecast_10d=l1.forecast_ndvi,
                       Forecast_30d=l3.forecast_ndvi, Lower_30d=l3.lower, Upper_30d=l3.upper,
                       Expected_change_30d=delta,
                       Trend="rising" if delta > TREND_DELTA else ("falling" if delta < -TREND_DELTA else "flat"),
                       Stage_10d=l1.stage_at_target, Stage_30d=l3.stage_at_target,
                       Recent_status="behind" if below >= 2 else ("ahead" if above >= 2 else "on track"),
                       Recent_below=below, Recent_above=above, Recent_checked=len(recent),
                       Below_share_10d=(full[full.horizon == 1].status.str.startswith("BELOW")).mean(),
                       Below_share_30d=(full[full.horizon == 3].status.str.startswith("BELOW")).mean(),
                       NDVI_amplitude=float(amp.iloc[0]) if len(amp) else np.nan,
                       Signal="weak" if (len(conf) and str(conf.iloc[0]).upper().startswith("WEAK")) else "clear"))
    W(pd.DataFrame(cs).round(3), "condition_summary.csv")

    # ---------- NDVI series and resilience indicators (need --clean) ----------
    if a.clean and os.path.exists(a.clean):
        c = pd.read_csv(a.clean, usecols=["District", "Year", "Decade_N", "NDVI_obs_clean"])
        g = c.dropna(subset=["NDVI_obs_clean"]).groupby(["District", "Year", "Decade_N"]).agg(
            NDVI=("NDVI_obs_clean", "mean"), n_pixels=("NDVI_obs_clean", "size")).reset_index()
        g["Date"] = [(pd.Timestamp(int(yr), 1, 1) + pd.Timedelta(days=10 * (int(dn) - 1))).date()
                     for yr, dn in zip(g.Year, g.Decade_N)]
        W(g.round(4), "ndvi_district_decade.csv")
        nrm = g.groupby(["District", "Decade_N"])["NDVI"].agg(norm="mean", sd="std", k="count").reset_index()
        g = g.merge(nrm, on=["District", "Decade_N"], how="left")
        g["dev"] = g["NDVI"] - g["norm"]
        g["z"] = np.where((g.k >= 4) & (g.sd > 0), g.dev / g.sd, np.nan)
        res = g.groupby(["District", "Year"]).agg(
            n_obs_decades=("NDVI", "size"), Peak_NDVI=("NDVI", "max"), Mean_NDVI=("NDVI", "mean"),
            Amplitude=("NDVI", lambda s: s.max() - s.min()), Green_share=("NDVI", lambda s: (s >= 0.5).mean()),
            Anomaly=("dev", "mean"),
            Stress_share=("z", lambda s: (s.dropna() <= -1).mean() if s.notna().any() else np.nan)).reset_index()
        ind = ["Peak_NDVI", "Mean_NDVI", "Amplitude", "Green_share", "Anomaly", "Stress_share"]
        res.loc[res.n_obs_decades < MIN_DECADES, ind] = np.nan           # too few clear-sky decades to trust
        mb = cp[cp.horizon == 1].dropna(subset=["lower"]).assign(b=lambda d_: d_.observed < d_.lower)
        mb = mb.groupby(["District", "test_year"])["b"].mean().rename("Model_below_share").reset_index().rename(
            columns={"test_year": "Year"})
        res = res.merge(mb, on=["District", "Year"], how="left").merge(
            yh[["District", "Year", "Yield"]], on=["District", "Year"], how="left")
        W(res.round(4), "resilience_indicators.csv")
        print(f"resilience_indicators.csv: {res.District.nunique()} districts with NDVI series")
    else:
        print("no --clean file: NDVI history and resilience indicators will be hidden in the app")

    # ---------- advisory rules: written once so your edits are never overwritten ----------
    rp = os.path.join(a.out, "advisory_rules.csv")
    old = os.path.exists(rp) and "audience" not in pd.read_csv(rp, nrows=1).columns
    if old or not os.path.exists(rp):
        r = pd.DataFrame(FARMER_RULES, columns=["rule_id", "audience", "focus", "stage_contains", "status", "trend",
                                                "signal", "priority", "message", "why"])
        r["review_status"] = "DRAFT: needs review by an agronomist"
        r.to_csv(rp, index=False)
        print("advisory_rules.csv written (edit it to change the farmer / field-officer advice)")
    print(f"done: {len(dist)} granaries with yield, {len(cs)} with satellite forecasts -> {a.out}/")


if __name__ == "__main__":
    main()