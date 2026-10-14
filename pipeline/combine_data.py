"""combine_data.py -- join the old and the newly extracted DailyGranary files into one.

    python combine_data.py DailyGranary.csv DailyGranary_new.csv --out DailyGranary_all.csv
(any number of input files; duplicated rows for the same pixel/year/decade are reported and dropped)
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", default="DailyGranary_all.csv")
a = ap.parse_args()

df = pd.concat([pd.read_csv(f) for f in a.files], ignore_index=True)
dup = df.duplicated(["Pixel_ID", "Year", "Decade_N"]).sum()
if dup:
    print(f"dropping {dup:,} duplicated pixel/year/decade rows")
    df = df.drop_duplicates(["Pixel_ID", "Year", "Decade_N"])
df.to_csv(a.out, index=False)
print(f"{a.out}: {len(df):,} rows, {df['District'].nunique()} districts -> {sorted(df['District'].unique())}")