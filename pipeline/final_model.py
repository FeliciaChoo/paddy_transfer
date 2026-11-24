"""
final_model.py  --  year-by-year training, test on the final year, forecasts + alerts + weather scenarios

Keep this file next to ndvi_transfer.py.

    python final_model.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out final_model
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ndvi_transfer as nt  # noqa: E402

warnings.filterwarnings("ignore")

LAGS = ["ndvi0", "ndvi1", "ndvi2", "ndvi3", "age"]
PHASE = ["phase", "ph_sin", "ph_cos"]
SAR = ["sar0", "sar1"]
CLIMATE = ["rain3_anom", "tmax_anom", "sol_anom", "lswi0", "mndwi0", "repro_heat_stress", "repro_water_stress"]
FEATS_BY = {
    "LPSC": LAGS + PHASE + SAR + CLIMATE,
    "LPS": LAGS + PHASE + SAR,
    "LAGS": LAGS,
}
CONFIGS = ["LPSC", "LPS", "LAGS", "RULE"]
KINDS = ["ridge", "hgb"]
STAGE_BINS = [(10, "germination / establishment"), (20, "seedling"), (40, "tillering"),
              (55, "max tillering / stem elongation"), (75, "panicle initiation / booting"),
              (90, "heading / flowering"), (105, "grain filling"), (120, "maturity / harvest")]


def make_rows_ext(ts, ph, h):
    y0, y1 = int(ts.Year.min()), int(ts.Year.max())
    n = (y1 - y0 + 1) * 36
    out = []
    for dist, g in ts.groupby("District"):
        if dist not in ph.index:
            continue
        t = ((g.Year - y0) * 36 + g.Decade_N - 1).astype(int).values
        full = {}
        for c in ["ndvi", "sar", "rain", "tmax", "sol", "lswi", "mndwi"]:
            s = pd.Series(np.nan, index=np.arange(n))
            if c in g.columns:
                s.loc[t] = g[c].values
            full[c] = s
        nd = full["ndvi"].values
        ff = full["ndvi"].ffill(limit=nt.FFILL_LIMIT).values
        last = pd.Series(np.where(~np.isnan(nd), np.arange(n), np.nan)).ffill().values
        age = np.arange(n) - last
        sar = full["sar"].ffill(limit=3).values
        lswi = full["lswi"].ffill(limit=nt.FFILL_LIMIT).fillna(0.0).values
        mndwi = full["mndwi"].ffill(limit=nt.FFILL_LIMIT).fillna(0.0).values
        rain3 = full["rain"].rolling(3, min_periods=1).sum().fillna(0.0).values
        tmax_v = full["tmax"].ffill(limit=3).bfill().values
        sol_v = full["sol"].ffill(limit=3).bfill().values

        dec = np.arange(n + h) % 36 + 1
        a, b = int(ph.loc[dist, "trough_A"]), int(ph.loc[dist, "trough_B"])
        phase = np.minimum((dec - a) % 36, (dec - b) % 36).astype(float)

        # Decade-of-year climatological anomalies & 20-85 DAP reproductive stress interactions
        dec_i = dec[:n]
        rain3_clim = pd.Series(rain3).groupby(dec_i).transform("mean").values
        tmax_clim = pd.Series(tmax_v).groupby(dec_i).transform("mean").values
        sol_clim = pd.Series(sol_v).groupby(dec_i).transform("mean").values

        rain3_anom = rain3 - rain3_clim
        tmax_anom = tmax_v - tmax_clim
        sol_anom = sol_v - sol_clim

        dap = phase[:n] * 10.0 + 5.0
        in_repro = ((dap >= 20.0) & (dap <= 85.0)).astype(float)
        repro_heat_stress = in_repro * np.maximum(0.0, tmax_anom)
        repro_water_stress = in_repro * np.minimum(0.0, rain3_anom)

        nd_ext = np.r_[nd, np.full(h, np.nan)]
        i = np.arange(3, n)
        out.append(pd.DataFrame({
            "District": dist, "year": y0 + i // 36, "Decade_N": dec[i], "t_target": i + h,
            "ndvi0": ff[i], "ndvi1": ff[i - 1], "ndvi2": ff[i - 2], "ndvi3": ff[i - 3], "age": age[i],
            "sar0": sar[i], "sar1": sar[i - 1],
            "lswi0": lswi[i], "mndwi0": mndwi[i],
            "rain3": rain3[i], "tmax": tmax_v[i], "sol": sol_v[i],
            "rain3_anom": rain3_anom[i], "tmax_anom": tmax_anom[i], "sol_anom": sol_anom[i],
            "repro_heat_stress": repro_heat_stress[i], "repro_water_stress": repro_water_stress[i],
            "phase": phase[i],
            "ph_sin": np.sin(2 * np.pi * phase[i] / 18), "ph_cos": np.cos(2 * np.pi * phase[i] / 18),
            "phase_f": phase[i + h], "Decade_f": dec[i + h], "y_abs": nd_ext[i + h],
            "delta": nd_ext[i + h] - ff[i], "is_last": i == n - 1}))
    return pd.concat(out, ignore_index=True), y0, n


def date_of(y0, t_index):
    t_index = int(t_index)
    return (pd.Timestamp(int(y0) + t_index // 36, 1, 1) + pd.Timedelta(days=10 * (t_index % 36))).date()


def stage_of(phase_decades):
    dap = phase_decades * 10 + 5
    for hi, name in STAGE_BINS:
        if dap <= hi:
            return name
    return "fallow / turnaround"


def fold_predictions(rows, y0, Y, kind, weak, min_year, min_train):
    need = FEATS_BY["LPSC"] + ["delta"]
    cut = (Y - y0) * 36
    tr = rows[(rows.year >= min_year) & (rows.year < Y) & (rows.t_target < cut)].dropna(subset=need)
    te = rows[(rows.year == Y)].dropna(subset=need)
    if len(tr) < min_train or te.empty:
        return None, len(tr)
    te = te.copy()
    for name, feats in FEATS_BY.items():
        te["pred_" + name] = nt.fit_model(kind, tr[feats], tr["delta"]).predict(te[feats])
    te["pred_RULE"] = np.where(te.District.isin(weak), te["pred_LAGS"], te["pred_LPSC"])
    te["pred_persist"] = 0.0
    te["pred_clim"] = nt.clim_delta(tr, te)
    return te, len(tr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--pheno", default="eda/eda_phenology.csv")
    ap.add_argument("--out", default="final_model")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--first-test-year", type=int, default=2019)
    ap.add_argument("--final-year", type=int, default=2025)
    ap.add_argument("--select-years", nargs="+", type=int, default=[2021, 2022, 2023, 2024])
    ap.add_argument("--min-year", type=int, default=2017)
    ap.add_argument("--min-train", type=int, default=150)
    ap.add_argument("--min-pix", type=int, default=1)
    ap.add_argument("--level", type=float, default=0.80)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.final_year in a.select_years:
        raise SystemExit("--select-years must not include the final year")

    ts = nt.load_series(a.clean, a.min_pix)
    ph = pd.read_csv(a.pheno).set_index("District")
    weak = set(ph.index[ph["confidence"].astype(str).str.upper().str.startswith("WEAK")])

    years = list(range(a.first_test_year, a.final_year + 1))
    preds, ntrain, recs, ROWS = {}, {}, [], {}
    for h in a.horizons:
        rows, y0, n_all = make_rows_ext(ts, ph, h)
        for kind in KINDS:
            for Y in years:
                te, ntr = fold_predictions(rows, y0, Y, kind, weak, a.min_year, a.min_train)
                ntrain[(h, kind, Y)] = ntr
                preds[(h, kind, Y)] = te
                if te is None:
                    continue
                for cfg in CONFIGS + ["persist", "clim"]:
                    e = (te["delta"] - te["pred_" + cfg]).abs()
                    for d, v in e.groupby(te["District"]):
                        recs.append(dict(horizon=h, model=kind, config=cfg, test_year=Y,
                                         train_years=f"{a.min_year}-{Y - 1}" if Y - 1 > a.min_year else str(a.min_year),
                                         n_train=ntr, district=d, MAE=v.mean(), n_test=len(v)))
        ROWS[h] = (rows, y0, n_all)
    res = pd.DataFrame(recs)
    res.to_csv(os.path.join(a.out, "fold_results.csv"), index=False)

    chosen = {}
    for h in a.horizons:
        r = res[res.horizon == h]
        sel = (r[(r.test_year.isin(a.select_years)) & (r.config.isin(CONFIGS))]
               .groupby(["model", "config"])["MAE"].mean().sort_values())
        if not sel.empty:
            chosen[h] = sel.index[0]
    pd.DataFrame([dict(horizon=h, model=m, config=c) for h, (m, c) in chosen.items()]).to_csv(
        os.path.join(a.out, "chosen_config.csv"), index=False)

    all_pred, alerts, latest = [], [], []
    scenarios = {
        "Normal": {"rain3_anom": 0.0, "tmax_anom": 0.0, "sol_anom": 0.0},
        "El Niño drought": {"rain3_anom": -45.0, "tmax_anom": 1.5, "sol_anom": 2.0},
        "Wet monsoon": {"rain3_anom": 60.0, "tmax_anom": -0.8, "sol_anom": -2.5},
    }
    for h, (kind, cfg) in chosen.items():
        rows, y0, n_all = ROWS[h]
        q_by_year = {}
        for Y in years:
            prev = preds.get((h, kind, Y - 1))
            if prev is not None and len(prev) >= 50:
                q_by_year[Y] = float(np.quantile((prev["delta"] - prev["pred_" + cfg]).abs(), a.level))
        for Y in years:
            te = preds.get((h, kind, Y))
            if te is None:
                continue
            q = q_by_year.get(Y, np.nan)
            p = te["pred_" + cfg].values
            o = pd.DataFrame({"horizon": h, "model": kind, "config": cfg, "test_year": Y, "District": te.District.values,
                              "origin_date": [date_of(y0, t - h) for t in te.t_target.values],
                              "target_date": [date_of(y0, t) for t in te.t_target.values],
                              "origin_decade": te.Decade_N.values, "ndvi_last": te.ndvi0.values,
                              "forecast": te.ndvi0.values + p, "lower": te.ndvi0.values + p - q,
                              "upper": te.ndvi0.values + p + q, "observed": te.ndvi0.values + te.delta.values,
                              "persistence": te.ndvi0.values, "interval_half_width": q})
            o["inside_interval"] = (o.observed >= o.lower) & (o.observed <= o.upper)
            all_pred.append(o)
            if Y == a.final_year:
                f = o.copy()
                f["target_decade"] = te.Decade_f.values
                f["stage_at_target"] = [stage_of(x) for x in te.phase_f.values]
                f["status"] = np.where(f.observed < f.lower, "BELOW forecast range (behind schedule?)",
                              np.where(f.observed > f.upper, "above forecast range", "within range"))
                alerts.append(f)

        need = FEATS_BY["LPSC"] + ["delta"]
        tr = rows[(rows.year >= a.min_year)].dropna(subset=need)
        last = rows[rows.is_last].copy()
        models = {nm: nt.fit_model(kind, tr[ft], tr["delta"]) for nm, ft in FEATS_BY.items()}
        q_prod = q_by_year.get(a.final_year, np.nan)
        for _, r_ in last.iterrows():
            nm = "LAGS" if (cfg == "LAGS" or (cfg == "RULE" and r_.District in weak)) else ("LPSC" if cfg in ("LPSC", "RULE") else "LPS")
            ft = FEATS_BY[nm]
            if r_[ft].isna().any():
                latest.append(dict(District=r_.District, horizon_decades=h, status="no forecast: recent NDVI not observed"))
                continue
            d = float(models[nm].predict(pd.DataFrame([r_[ft]]))[0])
            row_out = dict(District=r_.District, from_date=date_of(y0, n_all - 1),
                           target_date=date_of(y0, n_all - 1 + h), horizon_decades=h,
                           horizon_days=h * 10, last_ndvi=round(r_.ndvi0, 3),
                           forecast_ndvi=round(r_.ndvi0 + d, 3), lower=round(r_.ndvi0 + d - q_prod, 3),
                           upper=round(r_.ndvi0 + d + q_prod, 3), stage_at_target=stage_of(r_.phase_f),
                           model=kind, config=nm)
            # Compute Normal / El Nino drought / Wet monsoon scenario projections via LPSC
            for sc_name, shifts in scenarios.items():
                r_sc = r_.copy()
                r_sc["rain3_anom"] = r_["rain3_anom"] + shifts["rain3_anom"]
                r_sc["tmax_anom"] = r_["tmax_anom"] + shifts["tmax_anom"]
                r_sc["sol_anom"] = r_["sol_anom"] + shifts["sol_anom"]
                dap = r_["phase"] * 10.0 + 5.0
                in_rep = 1.0 if (20.0 <= dap <= 85.0) else 0.0
                r_sc["repro_heat_stress"] = in_rep * max(0.0, r_sc["tmax_anom"])
                r_sc["repro_water_stress"] = in_rep * min(0.0, r_sc["rain3_anom"])
                d_sc = float(models["LPSC"].predict(pd.DataFrame([r_sc[FEATS_BY["LPSC"]]]))[0])
                col_key = sc_name.lower().replace(" ", "_").replace("ñ", "n")
                row_out[f"forecast_ndvi_{col_key}"] = round(r_.ndvi0 + d_sc, 3)
            latest.append(row_out)

    if all_pred:
        pd.concat(all_pred, ignore_index=True).round(4).to_csv(os.path.join(a.out, "chosen_predictions.csv"), index=False)
    if alerts:
        al = pd.concat(alerts, ignore_index=True).dropna(subset=["lower"])
        al.sort_values(["status", "District", "origin_decade"], ascending=[False, True, True]).round(4).to_csv(
            os.path.join(a.out, "alerts_final_year.csv"), index=False)
    if latest:
        pd.DataFrame(latest).to_csv(os.path.join(a.out, "latest_forecast.csv"), index=False)


if __name__ == "__main__":
    main()