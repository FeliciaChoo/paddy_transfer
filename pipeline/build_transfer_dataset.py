"""
build_transfer_dataset.py -- the two tables the Pre-Transfer Assessment and Transfer Learning tabs read.

    python pipeline/build_transfer_dataset.py --clean eda/DailyGranary_clean.csv --out data

Writes
  transfer_yield_dataset.csv     one row per granary and YEAR (the model's unit: yield is reported once a year, so there is no season split)
                                 raw pipeline names and units; harmonisation (acts_transfer.harmonise) renames and converts them.
  transfer_decade_features.csv   one row per granary and TEN-DAY period: finer observations of the same drivers, used to compare the
                                 distributions of source and target with far more observations than seven yearly rows.
Nothing is invented: every value is an aggregate of the extracted satellite and weather series, or the reported yield.
Units as extracted: NDVI/EVI unitless; SAR VV in dB; temperature in deg C; 'Daily_Rain_mm' is a ten-day TOTAL in mm (the name is misleading);
solar radiation in MJ/m2; yield in kg/ha.
"""
import argparse, os
import numpy as np, pandas as pd

COLS = ["District", "Year", "Decade_N", "NDVI_clean", "EVI_clean", "SAR_clean", "Mean_Temp_C", "Daily_Rain_mm", "Solar_Radiation_MJ",
        "Optical_Observed", "Target_Yield"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    c = pd.read_csv(a.clean, usecols=COLS)
    drv = ["NDVI_clean", "EVI_clean", "SAR_clean", "Mean_Temp_C", "Daily_Rain_mm", "Solar_Radiation_MJ"]
    dec = c.groupby(["District", "Year", "Decade_N"])[drv + ["Optical_Observed"]].mean().reset_index()
    fine = dec.rename(columns={"NDVI_clean": "NDVI", "EVI_clean": "EVI", "SAR_clean": "SAR_VV_dB", "Daily_Rain_mm": "Rain_decade_mm",
                               "Solar_Radiation_MJ": "Solar_Radiation_MJ"}).drop(columns="Optical_Observed")
    rows = []
    for (g, y), d in dec.groupby(["District", "Year"]):
        rows.append(dict(
            District=g, Year=int(y), NDVI_mean=d.NDVI_clean.mean(), NDVI_max=d.NDVI_clean.max(),
            NDVI_amplitude=d.NDVI_clean.max() - d.NDVI_clean.min(), EVI_mean=d.EVI_clean.mean(), SAR_VV_dB_mean=d.SAR_clean.mean(),
            Mean_Temp_C_mean=d.Mean_Temp_C.mean(), Annual_Rain_mm=d.Daily_Rain_mm.sum(), Solar_Radiation_MJ_mean=d.Solar_Radiation_MJ.mean(),
            Optical_observed_pct=100 * d.Optical_Observed.mean(),
            Yield_kg_ha=c[(c.District == g) & (c.Year == y)].Target_Yield.mean()))
    yearly = pd.DataFrame(rows).round(4)
    os.makedirs(a.out, exist_ok=True)
    yearly.to_csv(os.path.join(a.out, "transfer_yield_dataset.csv"), index=False)
    fine.round(4).to_csv(os.path.join(a.out, "transfer_decade_features.csv"), index=False)
    print(f"transfer_yield_dataset.csv: {len(yearly)} rows ({yearly.District.nunique()} granaries x {yearly.Year.nunique()} years); "
          f"transfer_decade_features.csv: {len(fine)} rows")
    print("yield missing:", int(yearly.Yield_kg_ha.isna().sum()))


if __name__ == "__main__":
    main()
