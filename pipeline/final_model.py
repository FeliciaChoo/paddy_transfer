"""
final_model.py  --  year-by-year training, test on the final year, forecasts + alerts

Keep this file next to ndvi_transfer.py.

    python final_model.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out final_model
    python final_model.py ... --first-test-year 2019 --final-year 2024 --horizons 1 3

EXPANDING WINDOW (the scheme you asked for), all districts pooled in every fold:
    train 2018            -> test 2019
    train 2018-2019       -> test 2020
    train 2018-2020       -> test 2021
    ...
    train 2018-2023       -> test 2024   (the FINAL, held-out year)
A training row's forecast TARGET also has to lie before the test year, so nothing from the test year leaks in.

What is compared (per horizon, per fold)
    baselines   persistence (NDVI stays at last value), clim_delta (seasonal norm by planting phase)
    models      Ridge and HistGradientBoosting
    feature     LPS (own NDVI history + planting phase + SAR) | LAGS (own NDVI history only) |
    rules       RULE = LPS for districts with a clear planting signal, LAGS for WEAK ones (eda_phenology.csv)
MODEL CHOICE uses only the folds in --select-years (default 2021-2023). The final year is never used to choose.

Outputs in --out
    fold_results.csv        every fold x horizon x model x config x district (MAE and skill)
    chosen_config.csv       the config chosen from the selection folds, per horizon
    chosen_predictions.csv  predictions of the chosen config in every fold, with 80% intervals
    alerts_final_year.csv   behind-schedule check in the final year: observed NDVI vs forecast interval
    latest_forecast.csv     10-day and 30-day forecasts from the last available decade, with intervals and stage
Needs: pandas numpy scikit-learn
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
FEATS_BY = {"LPS": LAGS + PHASE + SAR, "LAGS": LAGS}
CONFIGS = ["LPS", "LAGS", "RULE"]
KINDS = ["ridge", "hgb"]
STAGE_BINS = [(10, "germination / establishment"), (20, "seedling"), (40, "tillering"),
              (55, "max tillering / stem elongation"), (75, "panicle initiation / booting"),
              (90, "heading / flowering"), (105, "grain filling"), (120, "maturity / harvest")]


# ----------------------------------------------------------------------
# rows: like nt.make_rows, but also keeps the LAST decade (no target yet) so we can forecast beyond the data
# ----------------------------------------------------------------------
def make_rows_ext(ts, ph, h):
    y0, y1 = int(ts.Year.min()), int(ts.Year.max())
    n = (y1 - y0 + 1) * 36
    out = []
    for dist, g in ts.groupby("District"):
        if dist not in ph.index:
            continue
        t = ((g.Year - y0) * 36 + g.Decade_N - 1).astype(int).values
        full = {}
        for c in ["ndvi", "sar", "rain", "tmax", "sol"]:
            s = pd.Series(np.nan, index=np.arange(n))
            s.loc[t] = g[c].values
            full[c] = s
        nd = full["ndvi"].values
        ff = full["ndvi"].ffill(limit=nt.FFILL_LIMIT).values
        last = pd.Series(np.where(~np.isnan(nd), np.arange(n), np.nan)).ffill().values
        age = np.arange(n) - last
        sar = full["sar"].ffill(limit=3).values
        rain3 = full["rain"].rolling(3, min_periods=3).sum().values
        dec = np.arange(n + h) % 36 + 1
        a, b = int(ph.loc[dist, "trough_A"]), int(ph.loc[dist, "trough_B"])
        phase = np.minimum((dec - a) % 36, (dec - b) % 36).astype(float)
        nd_ext = np.r_[nd, np.full(h, np.nan)]
        i = np.arange(3, n)
        out.append(pd.DataFrame({
            "District": dist, "year": y0 + i // 36, "Decade_N": dec[i], "t_target": i + h,
            "ndvi0": ff[i], "ndvi1": ff[i - 1], "ndvi2": ff[i - 2], "ndvi3": ff[i - 3], "age": age[i],
            "sar0": sar[i], "sar1": sar[i - 1], "rain3": rain3[i], "tmax": full["tmax"].values[i],
            "sol": full["sol"].values[i], "phase": phase[i],
            "ph_sin": np.sin(2 * np.pi * phase[i] / 18), "ph_cos": np.cos(2 * np.pi * phase[i] / 18),
            "phase_f": phase[i + h], "Decade_f": dec[i + h], "y_abs": nd_ext[i + h],
            "delta": nd_ext[i + h] - ff[i], "is_last": i == n - 1}))
    return pd.concat(out, ignore_index=True), y0, n


def date_of(y0, t_index):
    """start date of decade number t_index (0 = first decade of year y0); 36 blocks per year from 1 Jan"""
    t_index = int(t_index)
    return (pd.Timestamp(int(y0) + t_index // 36, 1, 1) + pd.Timedelta(days=10 * (t_index % 36))).date()


def stage_of(phase_decades):
    dap = phase_decades * 10 + 5
    for hi, name in STAGE_BINS:
        if dap <= hi:
            return name
    return "fallow / turnaround"


# ----------------------------------------------------------------------
# one fold
# ----------------------------------------------------------------------
def fold_predictions(rows, y0, Y, kind, weak, min_year, min_train):
    """train on years < Y (targets before Y), predict year Y. Returns test rows with pred columns, or None."""
    need = FEATS_BY["LPS"] + ["delta"]
    cut = (Y - y0) * 36
    tr = rows[(rows.year >= min_year) & (rows.year < Y) & (rows.t_target < cut)].dropna(subset=need)
    te = rows[(rows.year == Y)].dropna(subset=need)
    if len(tr) < min_train or te.empty:
        return None, len(tr)
    te = te.copy()
    for name, feats in FEATS_BY.items():
        te["pred_" + name] = nt.fit_model(kind, tr[feats], tr["delta"]).predict(te[feats])
    te["pred_RULE"] = np.where(te.District.isin(weak), te["pred_LAGS"], te["pred_LPS"])
    te["pred_persist"] = 0.0
    te["pred_clim"] = nt.clim_delta(tr, te)
    return te, len(tr)


def mae_table(te, config):
    e = (te["delta"] - te["pred_" + config]).abs()
    return e.groupby(te["District"]).mean()


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--pheno", default="eda/eda_phenology.csv")
    ap.add_argument("--out", default="final_model")
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 3])
    ap.add_argument("--first-test-year", type=int, default=2019)
    ap.add_argument("--final-year", type=int, default=2024)
    ap.add_argument("--select-years", nargs="+", type=int, default=[2021, 2022, 2023],
                    help="folds used to choose the model (the final year is never used)")
    ap.add_argument("--min-year", type=int, default=2018, help="first training year")
    ap.add_argument("--min-train", type=int, default=150, help="fewer training rows than this = fold skipped")
    ap.add_argument("--min-pix", type=int, default=5)
    ap.add_argument("--level", type=float, default=0.80, help="prediction interval coverage")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.final_year in a.select_years:
        raise SystemExit("--select-years must not include the final year")

    ts = nt.load_series(a.clean, a.min_pix)
    ph = pd.read_csv(a.pheno).set_index("District")
    weak = set(ph.index[ph["confidence"].astype(str).str.upper().str.startswith("WEAK")])
    districts = sorted(set(ts.District) & set(ph.index))
    print(f"districts ({len(districts)}): {districts}")
    print(f"weak planting signal (RULE uses lags only there): {sorted(weak) or 'none'}")

    years = list(range(a.first_test_year, a.final_year + 1))
    preds, ntrain, recs, ROWS = {}, {}, [], {}
    for h in a.horizons:
        rows, y0, n_all = make_rows_ext(ts, ph, h)
        rows_h = rows
        for kind in KINDS:
            for Y in years:
                te, ntr = fold_predictions(rows_h, y0, Y, kind, weak, a.min_year, a.min_train)
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

    # ---------- what happened in each fold ----------
    print("\n=== FOLDS (training rows available; folds with too few rows are skipped) ===")
    show = pd.DataFrame([dict(test_year=Y, train=f"{a.min_year}-{Y - 1}" if Y - 1 > a.min_year else str(a.min_year),
                              **{f"h{h}_ridge_n_train": ntrain[(h, 'ridge', Y)] for h in a.horizons},
                              status="ok" if all(preds[(h, 'ridge', Y)] is not None for h in a.horizons) else "SKIPPED (too little data)")
                         for Y in years])
    print(show.to_string(index=False))

    # ---------- per-year table, pooled over districts ----------
    pd.set_option("display.width", 220)
    chosen = {}
    for h in a.horizons:
        r = res[res.horizon == h]
        print(f"\n=== horizon {h} decade(s) ({h * 10} days): MAE by test year (mean over districts) ===")
        for kind in KINDS:
            t = (r[(r.model == kind)].groupby(["config", "test_year"])["MAE"].mean().unstack("test_year")
                 .reindex(CONFIGS + ["persist", "clim"]))
            t["mean_select_folds"] = t[[y for y in a.select_years if y in t.columns]].mean(axis=1)
            print(f"-- {kind}"); print(t.round(4).to_string())
        sel = (r[(r.test_year.isin(a.select_years)) & (r.config.isin(CONFIGS))]
               .groupby(["model", "config"])["MAE"].mean().sort_values())
        if sel.empty:
            print("  no selection folds available; check --select-years / data")
            continue
        chosen[h] = sel.index[0]
        persist = r[(r.test_year.isin(a.select_years)) & (r.config == "persist")]["MAE"].mean()
        print(f"  CHOSEN from folds {a.select_years} only: model={chosen[h][0]}, config={chosen[h][1]} "
              f"(selection MAE {sel.iloc[0]:.4f} vs persistence {persist:.4f})")
    pd.DataFrame([dict(horizon=h, model=m, config=c) for h, (m, c) in chosen.items()]).to_csv(
        os.path.join(a.out, "chosen_config.csv"), index=False)

    # ---------- chosen config: intervals, final year, alerts ----------
    all_pred, alerts, latest = [], [], []
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
        # production forecast: train on everything with a known target, forecast from the last decade
        need = FEATS_BY["LPS"] + ["delta"]
        tr = rows[(rows.year >= a.min_year)].dropna(subset=need)
        last = rows[rows.is_last].copy()
        models = {nm: nt.fit_model(kind, tr[ft], tr["delta"]) for nm, ft in FEATS_BY.items()}
        q_prod = q_by_year.get(a.final_year, np.nan)
        for _, r_ in last.iterrows():
            nm = "LAGS" if (cfg == "LAGS" or (cfg == "RULE" and r_.District in weak)) else "LPS"
            ft = FEATS_BY[nm]
            if r_[ft].isna().any():
                latest.append(dict(District=r_.District, horizon_decades=h, status="no forecast: recent NDVI not observed"))
                continue
            d = float(models[nm].predict(pd.DataFrame([r_[ft]]))[0])
            latest.append(dict(District=r_.District, from_date=date_of(y0, n_all - 1),
                               target_date=date_of(y0, n_all - 1 + h), horizon_decades=h,
                               horizon_days=h * 10, last_ndvi=round(r_.ndvi0, 3),
                               forecast_ndvi=round(r_.ndvi0 + d, 3), lower=round(r_.ndvi0 + d - q_prod, 3),
                               upper=round(r_.ndvi0 + d + q_prod, 3), stage_at_target=stage_of(r_.phase_f),
                               model=kind, config=nm))
    if all_pred:
        pp = pd.concat(all_pred, ignore_index=True)
        pp.round(4).to_csv(os.path.join(a.out, "chosen_predictions.csv"), index=False)
        print(f"\n=== INTERVAL CHECK for the chosen configs (nominal coverage {a.level:.0%}) ===")
        c = pp.dropna(subset=["lower"]).groupby(["horizon", "test_year"]).agg(
            coverage=("inside_interval", "mean"), half_width=("interval_half_width", "mean"), n=("observed", "size"))
        print(c.round(3).to_string())
    if alerts:
        al = pd.concat(alerts, ignore_index=True).dropna(subset=["lower"])
        al = al.sort_values(["status", "District", "origin_decade"], ascending=[False, True, True])
        al.round(4).to_csv(os.path.join(a.out, "alerts_final_year.csv"), index=False)
        print(f"\n=== FINAL YEAR {a.final_year}: observed vs forecast interval (held out; chosen config) ===")
        summ = al.groupby(["horizon", "District"]).agg(
            n=("observed", "size"), below=("status", lambda s: (s.str.startswith("BELOW")).mean()),
            above=("status", lambda s: (s == "above forecast range").mean()),
            MAE=("observed", lambda s: np.nan)).drop(columns="MAE")
        mae = al.assign(e=(al.observed - al.forecast).abs()).groupby(["horizon", "District"])["e"].mean()
        pers = al.assign(e=(al.observed - al.persistence).abs()).groupby(["horizon", "District"])["e"].mean()
        summ["MAE_model"], summ["MAE_persistence"] = mae, pers
        summ["skill_vs_persistence"] = 1 - mae / pers
        print(summ.round(3).to_string())
        print("(about 10% below and 10% above is what a well-calibrated 80% interval gives by chance; "
              "a district far above that share of BELOW decades deserves a field check)")
    if latest:
        lf = pd.DataFrame(latest)
        lf.to_csv(os.path.join(a.out, "latest_forecast.csv"), index=False)
        print("\n=== LATEST FORECAST from the last available decade ===")
        print(lf.to_string(index=False))
    print(f"\nSaved to ./{a.out}/. Folds are expanding-window; the final year was never used to choose the model.")


if __name__ == "__main__":
    main()