"""
eda_clean.py  --  EDA + NON-DESTRUCTIVE cleaning of DailyGranary.csv

    python eda_clean.py --data DailyGranary_all.csv --out eda

NO ROW IS EVER DELETED. The script asserts that the output has exactly the same rows.
Problems are handled by
    * flag_* columns   (True = this row has that issue)
    * *_clean columns  (the original value with flagged values set to NaN; originals untouched)

Outputs in --out:
    DailyGranary_clean.csv      same rows + flags + clean columns
    eda_missingness.csv, eda_range_checks.csv, eda_phenology.csv, cleaning_log.csv
    eda_phenology.png, eda_coverage.png
"""
import argparse
import os

import numpy as np
import pandas as pd

SPIKE_THRESHOLD = 0.25      # |observed NDVI - rolling median of 5 observations| above this = spike
RAIN_EXTREME_MM = 400       # per decade (10 days)
MAX_GAP_DECADES = 3         # interpolation limit for the *_clean filled series
DATE0 = pd.Timestamp(2021, 1, 1)


def decade_date(n):
    return (DATE0 + pd.Timedelta(days=10 * (int(n) - 1))).strftime("%d %b")


# ----------------------------------------------------------------------
# flags + clean columns
# ----------------------------------------------------------------------
def add_flags(d):
    d = d.sort_values(["Pixel_ID", "Year", "Decade_N"]).reset_index(drop=True)
    g = d.groupby("Pixel_ID")

    obs = d["NDVI"].where(d["Optical_Observed"])            # real observations only (not interpolated)
    med = obs.groupby(d["Pixel_ID"]).transform(
        lambda s: s.dropna().rolling(5, center=True, min_periods=3).median().reindex(s.index))
    d["flag_no_optical"] = ~d["Optical_Observed"]
    d["flag_interpolated"] = d["NDVI"].notna() & ~d["Optical_Observed"]
    d["flag_ndvi_spike"] = ((obs - med).abs() > SPIKE_THRESHOLD).fillna(False)   # cloud/shadow suspects
    d["flag_ndvi_negative"] = d["NDVI"] < 0                  # flooded/bare field: legitimate, NOT an error
    d["flag_evi_clamped"] = d["EVI"].abs() >= 1.0
    d["flag_sar_positive"] = d["SAR_VV_dB"] > 0              # unusual for paddy VV, check not delete
    d["flag_rain_extreme"] = d["Daily_Rain_mm"] > RAIN_EXTREME_MM
    d["flag_stage_mismatch"] = d["Stage_Consistent"].eq(False)
    shared = d.groupby(["Year", "Target_Yield"])["District"].transform("nunique") > 1
    d["flag_label_shared"] = shared                          # identical yield used by several districts

    # clean columns: original with flagged values -> NaN. Originals are untouched.
    obs_clean = obs.where(~d["flag_ndvi_spike"])
    d["NDVI_obs_clean"] = obs_clean                          # observed-only (use this for forecasting!)
    d["NDVI_clean"] = obs_clean.groupby(d["Pixel_ID"]).transform(
        lambda s: s.interpolate(limit=MAX_GAP_DECADES, limit_area="inside"))
    for c in ["EVI", "LSWI", "MNDWI"]:
        d[f"{c}_clean"] = d[c].where(~d["flag_ndvi_spike"])
    d["SAR_clean"] = d["SAR_VV_dB"].where(~d["flag_sar_positive"])
    return d


# ----------------------------------------------------------------------
# phenology: when does each district actually plant / peak?
# ----------------------------------------------------------------------
def phenology(d):
    rows, profiles = [], {}
    for dist, g in d[(d["Year"] >= 2019) & d["Optical_Observed"]].groupby("District"):
        prof = g.groupby("Decade_N")["NDVI_obs_clean"].mean().reindex(range(1, 37)).interpolate(limit_direction="both")
        ext = pd.concat([prof.iloc[-2:], prof, prof.iloc[:2]])
        sm = ext.rolling(3, center=True).mean().iloc[2:-2]    # circular smoothing
        sm.index = range(1, 37)
        profiles[dist] = (prof, sm)
        t1 = int(sm.idxmin())
        cand = [k for k in range(1, 37) if 12 <= min((k - t1) % 36, (t1 - k) % 36) <= 18]
        t2 = int(sm.loc[cand].idxmin())
        a, b = sorted([t1, t2])
        p1 = int(sm.loc[a + 1:b - 1].idxmax())
        outer = pd.concat([sm.loc[b + 1:36], sm.loc[1:a - 1]])
        p2 = int(outer.idxmax())
        amp = ((sm[p1] - sm[a]) + (sm[p2] - sm[b])) / 2
        rows.append(dict(District=dist,
                         trough_A=a, trough_A_date=decade_date(a), peak_A=p1, peak_A_date=decade_date(p1),
                         trough_B=b, trough_B_date=decade_date(b), peak_B=p2, peak_B_date=decade_date(p2),
                         trough_to_peak_decades=round(np.mean([(p1 - a) % 36, (p2 - b) % 36]), 1),
                         NDVI_amplitude=round(amp, 2),
                         confidence="clear" if amp >= 0.20 else "WEAK (mixed planting dates?)"))
    return pd.DataFrame(rows), profiles


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="DailyGranary.csv")
    ap.add_argument("--out", default="eda")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    raw = pd.read_csv(a.data)
    n0, keys0 = len(raw), raw[["Pixel_ID", "Year", "Decade_N"]].copy()
    print(f"Loaded {n0:,} rows, {raw['Pixel_ID'].nunique()} pixels, {raw['District'].nunique()} districts")

    # ---------- structure ----------
    print("\n1) STRUCTURE")
    print("   duplicate (pixel, year, decade) keys:", int(raw.duplicated(["Pixel_ID", "Year", "Decade_N"]).sum()))
    per = raw.groupby("Pixel_ID").size()
    print(f"   rows per pixel: min {per.min()}, max {per.max()} (expected 252 = 36 decades x 7 years)")
    print("   rows / pixels by district:")
    print(raw.groupby("District").agg(rows=("Year", "size"), pixels=("Pixel_ID", "nunique")).to_string())

    # ---------- missingness ----------
    print("\n2) MISSINGNESS (optical is the weak link)")
    raw["_nd_nan"] = raw["NDVI"].isna()
    miss = raw.groupby(["District", "Year"]).agg(
        optical_observed_pct=("Optical_Observed", lambda s: round(s.mean() * 100, 1)),
        ndvi_still_missing_pct=("_nd_nan", lambda s: round(s.mean() * 100, 1)),
        sar_missing_pct=("SAR_VV_dB", lambda s: round(s.isna().mean() * 100, 1))).reset_index()
    miss.to_csv(os.path.join(a.out, "eda_missingness.csv"), index=False)
    print(miss.pivot(index="District", columns="Year", values="optical_observed_pct").to_string())
    print("   (% of pixel-decades with a REAL optical observation; 2018 is almost empty)")

    # ---------- range checks ----------
    print("\n3) RANGE / PLAUSIBILITY CHECKS")
    checks = {
        "NDVI outside [-1,1]": ((raw.NDVI < -1) | (raw.NDVI > 1)).sum(),
        "NDVI < 0 (flooded/bare, legitimate)": (raw.NDVI < 0).sum(),
        "EVI clamped at +-1": (raw.EVI.abs() >= 1).sum(),
        "SAR VV > 0 dB": (raw.SAR_VV_dB > 0).sum(),
        "SAR VV < -25 dB": (raw.SAR_VV_dB < -25).sum(),
        f"Rain > {RAIN_EXTREME_MM} mm / decade": (raw.Daily_Rain_mm > RAIN_EXTREME_MM).sum(),
        "Rain < 0": (raw.Daily_Rain_mm < 0).sum(),
        "Tmax < Tmin": (raw.Mean_Max_Temp_C < raw.Mean_Min_Temp_C).sum(),
        "Solar <= 0": (raw.Solar_Radiation_MJ <= 0).sum(),
    }
    rc = pd.Series(checks, name="rows").astype(int)
    rc.to_csv(os.path.join(a.out, "eda_range_checks.csv"))
    print(rc.to_string())

    # ---------- flags ----------
    print("\n4) FLAGS ADDED (rows kept)")
    d = add_flags(raw.drop(columns="_nd_nan"))
    flag_cols = [c for c in d.columns if c.startswith("flag_")]
    log = pd.DataFrame({"rows_flagged": d[flag_cols].sum(), "pct": (d[flag_cols].mean() * 100).round(1)})
    log.to_csv(os.path.join(a.out, "cleaning_log.csv"))
    print(log.to_string())

    # ---------- labels ----------
    print("\n5) LABEL (Target_Yield) STRUCTURE")
    yt = d.groupby(["District", "Year"])["Target_Yield"].first().unstack()
    print(yt.round(0).to_string())
    print(f"   {yt.shape[0]} districts but only {yt.drop_duplicates().shape[0]} distinct yield series "
          f"-> districts sharing a series are NOT independent labels.")
    print("   CV of yield per district (std/mean):", (yt.std(axis=1) / yt.mean(axis=1)).round(3).to_dict())

    # ---------- phenology ----------
    print("\n6) PHENOLOGY: when does NDVI actually bottom out (planting) and peak?")
    ph, profiles = phenology(d)
    ph.to_csv(os.path.join(a.out, "eda_phenology.csv"), index=False)
    print(ph.to_string(index=False))
    print("   Compare with the assumed planting dates in the table (1 Jan and 1 Jul).")
    print(f"   Rows whose stage disagrees with NDVI/trend under that assumption: "
          f"{d['flag_stage_mismatch'].mean() * 100:.0f}%")

    # ---------- plots ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(profiles)
        cols = min(3, max(1, n))
        rows_n = -(-n // cols)                                  # ceil: grid fits any number of districts
        fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 2.9 * rows_n), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, (dist, (prof, sm)) in zip(axes, profiles.items()):
            ax.plot(prof.index, prof.values, ".", color="grey", label="mean observed NDVI")
            ax.plot(sm.index, sm.values, "-", color="tab:green", label="smoothed")
            r = ph[ph.District == dist].iloc[0]
            for k in (r.trough_A, r.trough_B):
                ax.axvline(k, color="tab:red", ls="--", alpha=.6)
            for k in (r.peak_A, r.peak_B):
                ax.axvline(k, color="tab:blue", ls=":", alpha=.6)
            ax.set_title(f"{dist} (red=trough/planting, blue=peak)", fontsize=8)
        for ax in axes[n:]:
            ax.set_visible(False)                               # hide unused panels
        for ax in axes[max(0, n - cols):n]:
            ax.set_xlabel("decade of year (1-36)")
        axes[0].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "eda_phenology.png"), dpi=130)
        plt.close()
        fig, ax = plt.subplots(figsize=(max(8, 0.9 * miss["District"].nunique() + 4), 3.5))
        miss.pivot(index="Year", columns="District", values="optical_observed_pct").plot.bar(ax=ax)
        ax.set_ylabel("% pixel-decades with real optical obs")
        ax.legend(fontsize=6)
        plt.tight_layout()
        plt.savefig(os.path.join(a.out, "eda_coverage.png"), dpi=130)
    except Exception as e:
        print("plots skipped:", e)

    # ---------- save, with the no-deletion guarantee ----------
    assert len(d) == n0, "row count changed!"
    k1 = d[["Pixel_ID", "Year", "Decade_N"]].sort_values(["Pixel_ID", "Year", "Decade_N"]).reset_index(drop=True)
    k0 = keys0.sort_values(["Pixel_ID", "Year", "Decade_N"]).reset_index(drop=True)
    assert k1.equals(k0), "row keys changed!"
    path = os.path.join(a.out, "DailyGranary_clean.csv")
    d.to_csv(path, index=False)
    print(f"\nSaved {path}: {len(d):,} rows (== input), {d.shape[1]} columns "
          f"({d.shape[1] - raw.shape[1] + 1} new). No rows removed.")


if __name__ == "__main__":
    main()