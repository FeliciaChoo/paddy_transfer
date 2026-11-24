import argparse
import glob
import os
import numpy as np
import pandas as pd

NAME_MAP = {
    "IADA BARAT LAUT SELANGOR": "IADA Barat Laut Selangor",
    "IADA KEMASIN SEMARAK": "IADA Kemasin Semerak",
    "IADA KERIAN": "IADA Kerian",
    "IADA KETARA": "KETARA",
    "KETARA": "KETARA",
    "IADA PEKAN": "IADA Pekan",
    "IADA PULAU PINANG": "IADA Pulau Pinang",
    "IADA ROMPIN": "IADA Rompin",
    "IADA SEBERANG PERAK": "IADA Seberang Perak",
    "KADA": "KADA",
    "MADA": "MADA",
}

def fuse_optical(df: pd.DataFrame, idx: str) -> np.ndarray:
    s2_c, ls_c = f"s2_{idx}_mean", f"landsat_{idx}_mean"
    s2_w = df["s2_ndvi_count"].where(df[s2_c].notna(), 0.0)
    ls_w = df["landsat_ndvi_count"].where(df[ls_c].notna(), 0.0)
    num = df[s2_c].fillna(0.0) * s2_w + df[ls_c].fillna(0.0) * ls_w
    den = s2_w + ls_w
    return np.where(den > 0, num / den, np.nan)

def assign_stage(dap: int) -> tuple[str, str]:
    if dap <= 10: return "Germination", "Vegetative"
    if dap <= 20: return "Seedling", "Vegetative"
    if dap <= 40: return "Tillering", "Vegetative"
    if dap <= 55: return "Max Tillering", "Vegetative"
    if dap <= 75: return "Panicle Initiation", "Reproductive"
    if dap <= 90: return "Heading", "Reproductive"
    if dap <= 105: return "Grain Filling", "Ripening"
    if dap <= 120: return "Maturity", "Ripening"
    return "Fallow", "Turnaround"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="polygon_daily")
    ap.add_argument("--pattern", default="granary_daily_*.csv")
    ap.add_argument("--yield-source", default="KPKM_Paddy_Granary_Panel_2017_2025.csv")
    ap.add_argument("--meta-csv", default="data/granary_meta.csv")
    ap.add_argument("--ref-daily", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--out", default="DailyGranary_by_granary.csv")
    ap.add_argument("--annual-out", default="malaysia_granary_annual_by_granary.csv")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.input_dir, a.pattern)))
    if not files:
        raise FileNotFoundError(f"No files found matching {os.path.join(a.input_dir, a.pattern)}")

    ph_map, sar_ref = None, None
    if os.path.exists(a.ref_daily):
        ref = pd.read_csv(a.ref_daily, usecols=lambda c: c in ["District", "Year", "Decade_N", "Soil_pH", "SAR_VV_dB"])
        if "Soil_pH" in ref.columns:
            ph_map = ref.groupby("District")["Soil_pH"].mean()
        if "SAR_VV_dB" in ref.columns:
            sar_ref = ref.groupby(["District", "Year", "Decade_N"], as_index=False)["SAR_VV_dB"].mean()

    raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    raw = raw.drop_duplicates(subset=["granary_id", "date"], keep="last")
    raw["date"] = pd.to_datetime(raw["date"])
    raw["Year"] = raw["date"].dt.year
    raw["Decade_N"] = np.minimum(36, (raw["date"].dt.dayofyear - 1) // 10 + 1)
    raw["District"] = raw["granary_name"].str.strip().str.upper().map(NAME_MAP).fillna(raw["granary_name"])

    for idx in ["ndvi", "evi", "lswi", "mndwi"]:
        raw[idx.upper()] = fuse_optical(raw, idx)
    raw["obs_cnt"] = raw["s2_ndvi_count"].fillna(0) + raw["landsat_ndvi_count"].fillna(0)

    agg = {
        "NDVI": ("NDVI", "mean"),
        "EVI": ("EVI", "mean"),
        "MNDWI": ("MNDWI", "mean"),
        "LSWI": ("LSWI", "mean"),
        "Mean_Max_Temp_C": ("tmax_c", "mean"),
        "Mean_Min_Temp_C": ("tmin_c", "mean"),
        "Mean_Temp_C": ("tmean_c", "mean"),
        "Daily_Rain_mm": ("rainfall_mm", "sum"),
        "Solar_Radiation_MJ": ("solar_mj_m2", "mean"),
        "N_Optical_Obs": ("obs_cnt", "sum"),
    }
    if "sar_vv_db" in raw.columns:
        agg["SAR_VV_dB"] = ("sar_vv_db", "mean")

    dec = raw.groupby(["District", "granary_id", "Year", "Decade_N"], as_index=False).agg(**agg)

    grid = pd.MultiIndex.from_product(
        [sorted(dec["District"].unique()), sorted(dec["Year"].unique()), range(1, 37)],
        names=["District", "Year", "Decade_N"],
    ).to_frame(index=False)
    id_map = dec[["District", "granary_id"]].drop_duplicates().set_index("District")["granary_id"]
    grid["granary_id"] = grid["District"].map(id_map)
    dec = grid.merge(dec, on=["District", "granary_id", "Year", "Decade_N"], how="left")

    if os.path.exists(a.meta_csv):
        meta = pd.read_csv(a.meta_csv)[["District", "Marker_Lat", "Marker_Lon"]].rename(
            columns={"Marker_Lat": "Latitude", "Marker_Lon": "Longitude"}
        )
        dec = dec.merge(meta, on="District", how="left")
    else:
        dec["Latitude"], dec["Longitude"] = np.nan, np.nan

    if os.path.exists(a.yield_source):
        yld_raw = pd.read_csv(a.yield_source)
        yld_raw["District"] = (
            yld_raw["granary_name"]
            .astype(str)
            .str.strip()
            .replace({"IADA KETARA": "KETARA"})
        )
        yld_raw["District"] = (
            yld_raw["District"].str.upper().map(NAME_MAP).fillna(yld_raw["District"])
        )
        yld = (
            yld_raw.rename(columns={"year": "Year", "avg_paddy_yield_kg_ha": "Target_Yield"})
            [["District", "Year", "Target_Yield"]]
            .dropna()
        )
        dec = dec.merge(yld, on=["District", "Year"], how="left")
    else:
        dec["Target_Yield"] = np.nan

    dec["Soil_pH"] = dec["District"].map(ph_map).fillna(5.5) if ph_map is not None else 5.5
    if "SAR_VV_dB" not in dec.columns:
        if sar_ref is not None:
            dec = dec.merge(sar_ref, on=["District", "Year", "Decade_N"], how="left")
        else:
            dec["SAR_VV_dB"] = np.nan

    dec["Source_District"] = dec["District"]
    dec["Granary"] = dec["District"]
    dec["Pixel_ID"] = dec["granary_id"]
    dec["Start_Date"] = [
        (pd.Timestamp(int(y), 1, 1) + pd.Timedelta(days=10 * (int(d) - 1))).strftime("%Y-%m-%d")
        for y, d in zip(dec["Year"], dec["Decade_N"])
    ]
    dec["Month"] = pd.to_datetime(dec["Start_Date"]).dt.month
    dec["Optical_Observed"] = (dec["N_Optical_Obs"].fillna(0) > 0) & dec["NDVI"].notna()
    dec["Cropping_Season"] = np.where(dec["Decade_N"] <= 18, "Main_Season", "Off_Season")
    dec["SubSeasonal_Period"] = dec["Cropping_Season"] + "_D" + (((dec["Decade_N"] - 1) % 18) + 1).astype(str)
    dec["DAP"] = ((dec["Decade_N"] - 1) % 18) * 10 + 5
    st = [assign_stage(x) for x in dec["DAP"]]
    dec["Stage"] = [x[0] for x in st]
    dec["Stage_Group"] = [x[1] for x in st]

    dec = dec.sort_values(["Pixel_ID", "Year", "Decade_N"]).reset_index(drop=True)
    dec["NDVI_Growth_Velocity"] = dec.groupby("Pixel_ID")["NDVI"].diff()
    dec["Rolling_Rain_3_Decades"] = dec.groupby("Pixel_ID")["Daily_Rain_mm"].transform(
        lambda s: s.rolling(3, min_periods=1).sum()
    )
    dec["Trend"] = np.where(
        dec["NDVI_Growth_Velocity"] > 0.02, "Rising",
        np.where(dec["NDVI_Growth_Velocity"] < -0.02, "Falling", "Stable"),
    )
    dec["Stage_Consistent"] = True

    dec.to_csv(a.out, index=False)
    ann = dec.groupby(["District", "Year", "Longitude", "Latitude"], as_index=False).agg(
        Target_Yield=("Target_Yield", "first"),
        Peak_NDVI=("NDVI", "max"),
        Peak_EVI=("EVI", "max"),
        Annual_Rain_mm=("Daily_Rain_mm", "sum"),
        Mean_Temp_C=("Mean_Temp_C", "mean"),
    )
    ann.to_csv(a.annual_out, index=False)
    print(f"Saved {a.out} ({len(dec):,} rows) and {a.annual_out} ({len(ann):,} rows)")

if __name__ == "__main__":
    main()