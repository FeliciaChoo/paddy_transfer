"""paddy_robustness.py  --  does restricting the satellite features to real paddy change the yield result?

    python pipeline/paddy_robustness.py --clean eda/DailyGranary_clean.csv --tags data/paddy_pixel_tags.csv --out data

Rebuilds the granary-year satellite features from four pixel sets and repeats three tests on each:

    all_pixels             every sampled pixel (what the project used so far)
    on_paddy               pixel centre inside a cleaned paddy polygon
    inside_granary_scheme  inside a polygon that is in a granary ('Dalam')
    own_scheme_only        inside the scheme its own granary is mapped to (geo/granary_scheme_map.csv)

    Test 1  year-to-year signal : Spearman(feature, yield) pooled and WITHIN granary (the part a forecast can use)
    Test 2  forecast            : leave-one-year-out; yield = last year x (1 + ridge on the CHANGE in satellite features),
                                  against plain "same as last year"
    Test 3  new granary         : leave-one-granary-out; satellite ridge against the mean of the other granaries

Same gate as the rest of the project (acts_transfer/modelchoice.py): the mask is reported as an ACCURACY gain only if a
bootstrap interval for (model error - baseline error) lies entirely below zero. Otherwise it is a QUALITY-CONTROL step.

Outputs in --out:  paddy_mask_correlation.csv, paddy_mask_growth_test.csv, paddy_mask_cold_start.csv, paddy_mask_verdict.md
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

SETS = {"all_pixels": None, "on_paddy": "In_Paddy", "inside_granary_scheme": "In_Granary_Scheme", "own_scheme_only": "In_Own_Scheme"}
FEATS = ["peak_ndvi", "mean_ndvi", "peak_evi", "sar", "rain", "temp"]
GROWTH_FEATS = ["peak_ndvi", "mean_ndvi", "peak_evi", "sar"]
COLD_FEATS = ["peak_ndvi", "peak_evi", "sar", "temp", "rain"]
MIN_PIXELS = 5            # a granary-year needs this many pixels in the set, otherwise it is left out of that set
N_BOOT = 4000


def pixel_year_features(clean_path):
    cols = ["Pixel_ID", "Year", "NDVI_obs_clean", "NDVI_clean", "EVI_clean", "SAR_clean", "Daily_Rain_mm", "Mean_Temp_C"]
    d = pd.read_csv(clean_path, usecols=cols)
    f = d.groupby(["Pixel_ID", "Year"]).agg(
        peak_ndvi=("NDVI_obs_clean", "max"),          # real optical observations only
        mean_ndvi=("NDVI_clean", "mean"), peak_evi=("EVI_clean", "max"), sar=("SAR_clean", "mean"),
        rain=("Daily_Rain_mm", "sum"), temp=("Mean_Temp_C", "mean")).reset_index()
    return f


def granary_year(f, tags, yields, mask_col, min_px=MIN_PIXELS):
    t = tags[["Pixel_ID", "Granary"] + [c for c in SETS.values() if c]].copy()
    g = f.merge(t, on="Pixel_ID", how="inner")
    if mask_col:
        g = g[g[mask_col].astype("boolean").fillna(False).astype(bool)]
    a = g.groupby(["Granary", "Year"]).agg(**{k: (k, "median") for k in FEATS}, npx=("Pixel_ID", "nunique")).reset_index()
    a = a[a.npx >= min_px].rename(columns={"Granary": "District"})
    return a.merge(yields, on=["District", "Year"])


def boot_ci(diff, seed=0):
    rng = np.random.default_rng(seed)
    bs = [rng.choice(diff, len(diff)).mean() for _ in range(N_BOOT)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def correlation_table(sets):
    rows = []
    for name, a in sets.items():
        for ft in GROWTH_FEATS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r, p = spearmanr(a[ft], a.Yield, nan_policy="omit")
                w = a.copy()
                w["fa"] = w[ft] - w.groupby("District")[ft].transform("mean")
                w["ya"] = w.Yield / w.groupby("District").Yield.transform("mean") - 1
                r2, p2 = spearmanr(w.fa, w.ya, nan_policy="omit")
            rows.append({"pixel_set": name, "feature": ft, "n_granary_years": len(a), "pooled_rho": round(r, 3), "pooled_p": round(p, 3),
                         "within_granary_rho": round(r2, 3), "within_granary_p": round(p2, 3)})
    return pd.DataFrame(rows)


def growth_test(a, use=GROWTH_FEATS, alpha=10.0):
    a = a.sort_values(["District", "Year"]).copy()
    for ft in use:
        a["d_" + ft] = a.groupby("District")[ft].diff()
    a["g"] = a.groupby("District").Yield.pct_change()
    a["prev"] = a.groupby("District").Yield.shift(1)
    a = a.dropna(subset=["g"] + ["d_" + u for u in use])
    X = a[["d_" + u for u in use]].values
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    err_m, err_b = [], []
    for yr in sorted(a.Year.unique()):
        te = (a.Year == yr).values
        if te.all() or not te.any():
            continue
        pr = Ridge(alpha=alpha).fit(X[~te], a.g.values[~te]).predict(X[te])
        yt, pv = a.Yield.values[te], a.prev.values[te]
        err_m += list(np.abs(pv * (1 + pr) - yt) / yt * 100)
        err_b += list(np.abs(pv - yt) / yt * 100)
    err_m, err_b = np.array(err_m), np.array(err_b)
    lo, hi = boot_ci(err_m - err_b)
    return {"n": len(err_b), "baseline_MAPE_last_year": round(err_b.mean(), 2), "model_MAPE": round(err_m.mean(), 2),
            "diff_model_minus_baseline": round((err_m - err_b).mean(), 2), "ci_low": round(lo, 2), "ci_high": round(hi, 2)}


def cold_start_test(a, use=COLD_FEATS, alpha=5.0):
    a = a.dropna(subset=use + ["Yield"]).reset_index(drop=True)
    X = a[use].values
    err_b, err_m = [], []
    for gname in a.District.unique():
        te = (a.District == gname).values
        tr = ~te
        if tr.sum() < 3:
            continue
        mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-9
        pr = Ridge(alpha=alpha).fit((X[tr] - mx) / sx, a.Yield.values[tr]).predict((X[te] - mx) / sx)
        yt = a.Yield.values[te]
        err_b += list(np.abs(a.Yield.values[tr].mean() - yt) / yt * 100)
        err_m += list(np.abs(pr - yt) / yt * 100)
    err_b, err_m = np.array(err_b), np.array(err_m)
    lo, hi = boot_ci(err_m - err_b, seed=1)
    return {"n": len(err_b), "granaries": int(a.District.nunique()), "baseline_MAPE_mean_of_others": round(err_b.mean(), 2),
            "model_MAPE": round(err_m.mean(), 2), "diff_model_minus_baseline": round((err_m - err_b).mean(), 2),
            "ci_low": round(lo, 2), "ci_high": round(hi, 2)}


def verdict(growth, cold):
    """ACCURACY_GAIN only if some MASKED set beats its baseline with an interval entirely below zero in test 2 or test 3."""
    hits = []
    for label, df in (("forecast (test 2)", growth), ("new granary (test 3)", cold)):
        for _, r in df[df.pixel_set != "all_pixels"].iterrows():
            if r.ci_high < 0:
                hits.append(f"{r.pixel_set} in {label}: {r.diff_model_minus_baseline:+.2f} points (95% CI {r.ci_low:+.2f} to {r.ci_high:+.2f})")
    if hits:
        return "ACCURACY_GAIN", hits
    return "QUALITY_CONTROL_ONLY", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--tags", default="data/paddy_pixel_tags.csv")
    ap.add_argument("--yield-history", default="data/yield_history.csv")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tags = pd.read_csv(a.tags)
    yields = pd.read_csv(a.yield_history)[["District", "Year", "Yield"]]
    f = pixel_year_features(a.clean)
    missing = set(f.Pixel_ID) - set(tags.Pixel_ID)
    if missing:
        raise SystemExit(f"{len(missing)} pixels in the satellite file are missing from the tag file: re-run paddy_mask.py on the same data.")

    sets = {name: granary_year(f, tags, yields, col) for name, col in SETS.items()}
    print("granary-years per pixel set:", {k: len(v) for k, v in sets.items()})
    corr = correlation_table(sets)
    corr.to_csv(os.path.join(a.out, "paddy_mask_correlation.csv"), index=False)
    growth = pd.DataFrame([{"pixel_set": k, **growth_test(v)} for k, v in sets.items()])
    cold = pd.DataFrame([{"pixel_set": k, **cold_start_test(v)} for k, v in sets.items()])
    growth.to_csv(os.path.join(a.out, "paddy_mask_growth_test.csv"), index=False)
    cold.to_csv(os.path.join(a.out, "paddy_mask_cold_start.csv"), index=False)

    print("\nTest 1  within-granary Spearman (peak_ndvi):")
    print(corr[corr.feature == "peak_ndvi"][["pixel_set", "n_granary_years", "pooled_rho", "pooled_p", "within_granary_rho", "within_granary_p"]].to_string(index=False))
    print("\nTest 2  forecast vs 'same as last year' (MAPE %):")
    print(growth.to_string(index=False))
    print("\nTest 3  new granary vs mean of the others (MAPE %):")
    print(cold.to_string(index=False))

    label, hits = verdict(growth, cold)
    w = corr[corr.feature == "peak_ndvi"].within_granary_rho
    txt = [f"# Paddy mask: verdict\n", f"**{label}**\n"]
    if label == "ACCURACY_GAIN":
        txt.append("A masked pixel set beat its baseline with a 95% interval entirely below zero:\n")
        txt += [f"- {h}" for h in hits]
        txt.append("\nRe-check with another random seed and with the data owner's scheme boundaries before relying on it.")
    else:
        txt.append("No masked pixel set beat its baseline with a 95% interval entirely below zero. Use the paddy mask as a "
                   "documented quality-control and sensitivity step (it shows where the labels and the pixels disagree), "
                   "not as an accuracy improvement.\n")
        txt.append(f"Within-granary Spearman of peak NDVI with yield across the four pixel sets: {w.min():+.2f} to {w.max():+.2f} "
                   f"(a usable forecast signal would need a clearly positive, significant value).")
    with open(os.path.join(a.out, "paddy_mask_verdict.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(txt) + "\n")
    print(f"\nVERDICT: {label}")


if __name__ == "__main__":
    main()
