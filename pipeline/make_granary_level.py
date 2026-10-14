"""make_granary_level.py -- turn the district-level DailyGranary.csv into a GRANARY-level one.

    python pipeline/make_granary_level.py --daily DailyGranary.csv

Writes (both in the project root, next to DailyGranary.csv):
    DailyGranary_by_granary.csv           same rows, but the 'District' column now holds the GRANARY name
                                          (the original district is kept in 'Source_District').
    malaysia_granary_annual_by_granary.csv  one row per sampled pixel and year: District(=granary), Year, Longitude, Latitude,
                                          Target_Yield, Peak_NDVI, Peak_EVI, Annual_Rain_mm, Mean_Temp_C
                                          (derived from the daily table, because the original annual yield file is not needed)

Why relabel instead of changing the pipeline: eda_clean.py, final_model.py, prepare_app_data.py and the dashboard all treat
the 'District' column as their unit. Putting the granary name there makes every one of them work per granary with no other
change. Districts of the same granary (Kuala Selangor + Sabak Bernam) are pooled into one series.
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--daily", default="DailyGranary.csv")
ap.add_argument("--out", default="DailyGranary_by_granary.csv")
ap.add_argument("--annual-out", default="malaysia_granary_annual_by_granary.csv")
a = ap.parse_args()

d = pd.read_csv(a.daily)
if "Granary" not in d.columns:
    raise SystemExit("No 'Granary' column. Run pipeline/add_granary_column.py on the file first.")
if d["Granary"].isna().any():
    raise SystemExit(f"Rows without a granary: {sorted(d.loc[d.Granary.isna(), 'District'].unique())}. Fix GRANARY_OF.")

# ---- yield must be one series per granary-year (pooled districts of a granary report the same yield)
ys = d.groupby(["Granary", "Year"])["Target_Yield"].agg(["min", "max"])
bad = ys[(ys["max"] - ys["min"]).abs() > 1e-6]
if len(bad):
    print("WARNING: districts inside one granary have DIFFERENT yields in these granary-years; the first value is used:")
    print(bad.to_string())

g = d.rename(columns={"District": "Source_District"}).rename(columns={"Granary": "District"})
g.insert(list(g.columns).index("District") + 1, "Granary", g["District"])
# a pooled granary has several 'Cropping_Season' / phenology columns already per pixel, so nothing else needs merging
g.to_csv(a.out, index=False)
print(f"{a.out}: {len(g):,} rows, {g.District.nunique()} granaries, {g.Pixel_ID.nunique()} pixels")

# ---- annual per-pixel table (the format prepare_app_data.py reads)
rain = "Daily_Rain_mm" if "Daily_Rain_mm" in g.columns else "Decade_Rain_mm"   # this column is a decade TOTAL
ann = (g.groupby(["District", "Year", "Pixel_ID"], as_index=False)
         .agg(Longitude=("Longitude", "first"), Latitude=("Latitude", "first"),
              Target_Yield=("Target_Yield", "first"), Peak_NDVI=("NDVI", "max"), Peak_EVI=("EVI", "max"),
              Annual_Rain_mm=(rain, "sum"), Mean_Temp_C=("Mean_Temp_C", "mean")))
ann = ann.drop(columns="Pixel_ID")
ann.to_csv(a.annual_out, index=False)
print(f"{a.annual_out}: {len(ann):,} rows")
print(ann.groupby("District").agg(pixels_years=("Year", "size"), first=("Year", "min"), last=("Year", "max")).to_string())
