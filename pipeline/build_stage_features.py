"""
build_stage_features.py -- PACE steps A2 (consolidate to one row per granary-year) and C1 (phenology-aligned growth-stage features).

    python pipeline/build_stage_features.py --clean eda/DailyGranary_clean.csv --out data

Reads the pixel-decade table written by eda_clean.py (about 189,000 rows: 10 granaries x 63-134 pixels x 7 years x 36 ten-day decades) and writes

  transfer_stage_dataset.csv   ONE ROW PER (District, Year): N = 70 = 10 granaries x 7 years (60 rows for 2019-2024). Yield_kg_ha, the growth-stage
                               features C1_* (Cycle 1, main season) and C2_* (Cycle 2, off-season), Soil_pH, Optical_Valid_Fraction, and the spatial
                               dispersion of the pixels (mean is the feature; std, p10, p90 are kept as descriptors).
  phenology_greenup.csv        the detected green-up decade of each cycle, for every granary-year (so the alignment can be inspected)
  transfer_decade_features.csv (refreshed by build_transfer_dataset.py; this script does not touch it)

Method
  1. A2  pixels -> granary-decade by the mean (pixel std / p10 / p90 of the seasonal EVI are saved as spatial descriptors). Yield is reported once per
         granary-year, so the pixel rows are collapsed with groupby(['District','Year']): no pseudo-replication.
  2. C1  the 36 ten-day EVI values of a granary-year are smoothed with a Savitzky-Golay filter (window 5, polynomial order 2). Peaks are found on the
         smoothed curve; the first peak is Cycle 1 (main season) and the second Cycle 2 (off-season). The green-up of a cycle is the first decade, after
         the preceding trough, at which the smoothed EVI has climbed 20% of the way from trough to peak.
  3. Stage windows are counted from green-up, in ten-day decades: Vegetative 0-40 d (4 decades), Reproductive/Heading 40-70 d (3), Ripening 70-100 d (3).
           Vegetative    EVI (smoothed), MNDWI, SAR VV
           Reproductive  EVI (smoothed), LSWI, SAR VV, Rain (total mm), Tmax
           Ripening      EVI (smoothed), Solar radiation
  A cycle that does not exist in a granary-year (a single-peak year) or whose window runs past the end of the year is left MISSING, never filled here: median
  imputation happens later, inside each training fold. A feature that is missing in more than the configured share of rows is dropped by the harmonisation step.
Nothing is invented. Every value is an aggregate of the extracted satellite and weather series.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

DRV = {"EVI_clean": "EVI", "LSWI_clean": "LSWI", "MNDWI_clean": "MNDWI", "SAR_clean": "SAR_VV_dB", "Mean_Max_Temp_C": "Tmax_C", "Mean_Temp_C": "Mean_Temp_C",
       "Daily_Rain_mm": "Rain_mm", "Solar_Radiation_MJ": "Solar_MJ"}
STAGES = {"veg": (0, 4), "rep": (4, 7), "rip": (7, 10)}           # decades after green-up: 0-40 d, 40-70 d, 70-100 d
STAGE_VARS = {"veg": ["EVI", "MNDWI", "SAR_VV_dB"], "rep": ["EVI", "LSWI", "SAR_VV_dB", "Rain_mm", "Tmax_C", "Mean_Temp_C"], "rip": ["EVI", "Solar_MJ"]}
SG_WINDOW, SG_ORDER, RISE = 5, 2, 0.20


def interp(v):
    s = pd.Series(v, dtype=float)
    return s.interpolate(limit_direction="both").values if s.notna().sum() >= 2 else np.full(len(s), np.nan)


def greenups(evi):
    """Savitzky-Golay smooth the 36 decades, find up to two peaks, return (smoothed curve, [(green-up index, peak index), ...])"""
    x = interp(evi)
    if np.isnan(x).any():
        return x, []
    s = savgol_filter(x, SG_WINDOW, SG_ORDER)
    amp = s.max() - s.min()
    if amp < 0.05:                                              # a flat curve has no cycle
        return s, []
    pk, pr = find_peaks(s, distance=8, prominence=0.15 * amp)
    if len(pk) == 0:
        pk = np.array([int(np.argmax(s))])                      # a single broad season: its maximum is the one cycle
    if len(pk) > 2:
        pk = np.sort(pk[np.argsort(pr["prominences"])[-2:]])    # the two most prominent, in time order
    out, prev = [], 0
    for p in pk:
        seg = s[prev:p + 1]
        t = prev + int(np.argmin(seg))
        thr = s[t] + RISE * (s[p] - s[t])
        g = next((i for i in range(t, p + 1) if s[i] >= thr), p)
        out.append((g, int(p)))
        prev = int(p)
    return s, out


def stage_features(dec, evi_s, ups):
    """dec: the 36-row granary-decade table of one granary-year (index = decade 1..36); evi_s: smoothed EVI; ups: green-ups"""
    row = {}
    for c in (1, 2):
        g = ups[c - 1][0] if len(ups) >= c else None
        row[f"C{c}_Greenup_Decade"] = np.nan if g is None else g + 1
        row[f"C{c}_Peak_Decade"] = np.nan if g is None else ups[c - 1][1] + 1
        for st, (a, b) in STAGES.items():
            for v in STAGE_VARS[st]:
                name = f"C{c}_{v}_{st}"
                if g is None or g + a >= len(dec):
                    row[name] = np.nan
                    continue
                lo, hi = g + a, min(g + b, len(dec))
                if v == "EVI":
                    seg = evi_s[lo:hi]
                else:
                    seg = dec[v].values[lo:hi]
                seg = seg[~np.isnan(seg)]
                if len(seg) < max(1, (b - a) // 2):               # fewer than half of the window's decades: missing, not a guess
                    row[name] = np.nan
                else:
                    row[name] = float(seg.sum() if v == "Rain_mm" else seg.mean())
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    need = ["District", "Year", "Decade_N", "Pixel_ID", "Target_Yield", "Soil_pH", "Optical_Observed"] + list(DRV)
    c = pd.read_csv(a.clean, usecols=need)
    # ---- A2: one row per granary-year-decade (mean over pixels), plus pixel dispersion of the year's mean EVI
    dec_all = c.groupby(["District", "Year", "Decade_N"]).agg(**{v: (k, "mean") for k, v in DRV.items()}, Optical_Observed=("Optical_Observed", "mean")).reset_index()
    pix = c.groupby(["District", "Year", "Pixel_ID"]).EVI_clean.mean().reset_index()
    disp = pix.groupby(["District", "Year"]).EVI_clean.agg(EVI_pix_mean="mean", EVI_pix_std="std", EVI_pix_p10=lambda v: v.quantile(.1),
                                                             EVI_pix_p90=lambda v: v.quantile(.9), n_pixels="size").reset_index()
    base = c.groupby(["District", "Year"]).agg(Yield_kg_ha=("Target_Yield", "mean"), Soil_pH=("Soil_pH", "mean"),
                                               Optical_Valid_Fraction=("Optical_Observed", "mean")).reset_index()
    rows, ph = [], []
    for (g, y), d in dec_all.groupby(["District", "Year"]):
        d = d.sort_values("Decade_N").reset_index(drop=True)
        if len(d) != 36:
            continue
        s, ups = greenups(d["EVI"].values)
        f = stage_features(d, s, ups)
        rows.append(dict(District=g, Year=int(y), **f))
        ph.append(dict(District=g, Year=int(y), n_cycles=len(ups), **{k: v for k, v in f.items() if k.endswith(("Greenup_Decade", "Peak_Decade"))}))
    out = base.merge(disp, on=["District", "Year"]).merge(pd.DataFrame(rows), on=["District", "Year"])
    out["Optical_observed_pct"] = 100 * out.Optical_Valid_Fraction
    out["Optical_complete_window"] = out.Year.between(2019, int(out["Year"].max()))# the 2019-2024 optical-complete subset (N = 60)
    out["Yield_kg_ha"] = out.Yield_kg_ha.round(1)

    sched_path = "external/granary_irrigation_schedules.csv"
    if os.path.exists(sched_path):
        sched = pd.read_csv(sched_path)
        c1_sched = sched[sched["Cycle"] == "C1"][
            ["District", "Water_Start_Decade", "Water_End_Decade", "Block_Stagger_Decades"]
        ]
        out = out.merge(c1_sched, on="District", how="left")
        out["Schedule_Slip_Decades"] = out["C1_Greenup_Decade"] - out["Water_Start_Decade"]
        # Approximate diurnal range from Tmax - Mean_Temp (half-range >= 4.1C corresponds to Tmax - Tmin >= 8.2C)
        out["StemBorer_Weather_Risk"] = (
                ((out["C1_Tmax_C_rep"] - out["C1_Mean_Temp_C_rep"]) >= 4.1)
                & (out["C1_Rain_mm_rep"] < 50.0)
        ).astype(int)

    os.makedirs(a.out, exist_ok=True)
    out.round(4).to_csv(os.path.join(a.out, "transfer_stage_dataset.csv"), index=False)
    pd.DataFrame(ph).to_csv(os.path.join(a.out, "phenology_greenup.csv"), index=False)
    feats = [x for x in out.columns if x.startswith(("C1_", "C2_")) and not x.endswith(("Greenup_Decade", "Peak_Decade"))]
    print(f"transfer_stage_dataset.csv: {len(out)} rows ({out.District.nunique()} granaries x {out.Year.nunique()} years); {out.Optical_complete_window.sum()} in 2019-2024")
    print(f"stage features: {len(feats)}; share missing per cycle:")
    for cyc in ("C1_", "C2_"):
        cols = [x for x in feats if x.startswith(cyc)]
        print(f"  {cyc}: {100 * out[cols].isna().mean().mean():.1f}% missing on average; cycles detected in {int((pd.DataFrame(ph)['n_cycles'] >= int(cyc[1])).sum())} of {len(ph)} granary-years")


if __name__ == "__main__":
    main()
