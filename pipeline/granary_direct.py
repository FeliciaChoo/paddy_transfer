"""
granary_direct.py  --  Earth Engine  ->  DailyGranary CSV in ONE step (no Drive, no downloads)

    python granary_direct.py                                  # all 4 districts, 80 pixels each
    python granary_direct.py --pixels 60 --districts perak_tengah
    python granary_direct.py --yield-csv malaysia_granary_annual_ml_ready.csv --out DailyGranary.csv

How it avoids Drive: instead of exporting every scene for every grid point (huge), Earth
Engine computes the 10-day composites itself (median NDVI/EVI/MNDWI/LSWI, mean SAR VV) and
returns only the small result table straight into Python (getInfo, max 5000 pixels per call;
each request stacks several decades into one image so Earth Engine runs a single sampling job). Everything after that (weather units, gap handling,
DAP season/stage, yield merge) is the same code as granary_pipeline.py.

TRADE-OFF: this works on a SAMPLE of cropland pixels per district (--pixels, default 80,
similar to the 29-84 pixels in your spreadsheet), not all ~30,000 grid points. For every
pixel use granary_pipeline.py (Drive exports).

RESUMABLE: each piece is cached in ./gee_cache. If it stops or times out, run the same
command again; finished pieces are not re-fetched. Delete gee_cache to start over.

Keep this file in the same folder as granary_pipeline.py (it reuses its functions).
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import granary_pipeline as gp

ee = gp.ee

N_PIXELS = 80
DECADES_PER_REQUEST = 12         # decades stacked into ONE sampling call (36 = whole year)
MAX_PIXELS = 4500                # getInfo() returns at most 5000 features
CACHE_DIR = "gee_cache"
SEED = 42
UNMASK = -9999.0                 # stand-in for "no valid observation" inside Earth Engine
OPT = ["NDVI", "EVI", "MNDWI", "LSWI"]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def decade_bounds(year, n):
    """Same 36 blocks as the table: block n starts 1 Jan + 10*(n-1) days; block 36 runs to 31 Dec."""
    start = pd.Timestamp(year, 1, 1) + pd.Timedelta(days=10 * (n - 1))
    end = pd.Timestamp(year + 1, 1, 1) if n == 36 else start + pd.Timedelta(days=10)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def decade_chunks(n_pixels=None):
    """Split the 36 decades into requests of DECADES_PER_REQUEST decades each."""
    k = max(1, min(36, DECADES_PER_REQUEST))
    return [(lo, min(lo + k - 1, 36)) for lo in range(1, 37, k)]


def wide_to_long(df, base_cols, lo, hi):
    """One row per point with '<band>_<n>' columns -> one row per point per decade."""
    cols_out = ["Longitude", "Latitude", "Decade_N"] + base_cols
    if df.empty:
        return pd.DataFrame(columns=cols_out)
    out = []
    for n in range(lo, hi + 1):
        ren = {f"{b}_{n}": b for b in base_cols}
        sub = df[["Longitude", "Latitude"] + list(ren)].rename(columns=ren)
        sub["Decade_N"] = n
        out.append(sub)
    res = pd.concat(out, ignore_index=True)[cols_out]
    res[base_cols] = res[base_cols].replace(UNMASK, np.nan)
    return res


def roi_for(prefix):
    lon0, lon1, lat0, lat1 = gp.ROIS[prefix]
    b = gp.BUFFER_DEG
    return ee.Geometry.Rectangle([lon0 - b, lat0 - b, lon1 + b, lat1 + b])


def fetch(fc, what, retries=3):
    """fc.getInfo() with retries -> DataFrame of feature properties."""
    for i in range(retries):
        try:
            feats = fc.getInfo()["features"]
            return pd.DataFrame([f["properties"] for f in feats])
        except Exception as e:
            wait = 15 * (i + 1)
            print(f"      {what}: {str(e)[:140]}\n      retrying in {wait}s ({i + 1}/{retries})")
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after {retries} tries. Try fewer --pixels and re-run "
                       f"(finished pieces are cached).")


def cached(name, builder, columns):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    t0 = time.time()
    df = builder()
    if df.empty:
        df = pd.DataFrame(columns=columns)
    df.to_csv(path, index=False)
    print(f"      {name}: {len(df):,} rows ({time.time() - t0:.0f}s)")
    return df


def points_fc(pts):
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([r.Longitude, r.Latitude]),
                   {"Longitude": r.Longitude, "Latitude": r.Latitude})
        for r in pts.itertuples()])


# ----------------------------------------------------------------------
# Earth Engine pieces
# ----------------------------------------------------------------------
def client_grid(prefix):
    """The ~250 m sampling grid over the district, built locally (no Earth Engine call)."""
    lon0, lon1, lat0, lat1 = gp.ROIS[prefix]
    dlat = gp.GRID_SPACING_M / 111320.0
    dlon = gp.GRID_SPACING_M / (111320.0 * np.cos(np.radians((lat0 + lat1) / 2)))
    lon, lat = np.meshgrid(np.arange(lon0 + dlon / 2, lon1, dlon),
                           np.arange(lat0 + dlat / 2, lat1, dlat))
    return pd.DataFrame({"Longitude": np.round(lon.ravel(), 5), "Latitude": np.round(lat.ravel(), 5)})


def classify_cropland(cands):
    """Keep only the candidate points that sit on WorldCover cropland (class 40).
    A light point lookup, done in small batches so it never times out."""
    crop = ee.Image("ESA/WorldCover/v200/2021").select("Map").eq(40).rename("crop")
    res = fetch(crop.sampleRegions(collection=points_fc(cands), scale=10, tileScale=4),
                "cropland check")
    if res.empty or "crop" not in res:
        return pd.DataFrame(columns=["Longitude", "Latitude"])
    return res.loc[res["crop"] == 1, ["Longitude", "Latitude"]]


def get_points(prefix, roi, n, batch=1000):
    """n random cropland grid points. Cached, so every run uses the SAME pixels.
    The grid is shuffled locally; Earth Engine only classifies `batch` points at a time
    until n cropland points are found."""
    def build():
        grid = client_grid(prefix).sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"      {prefix}: local grid has {len(grid):,} points")
        found, total = [], 0
        for i in range(0, len(grid), batch):
            ok = classify_cropland(grid.iloc[i:i + batch])
            found.append(ok)
            total += len(ok)
            print(f"      {prefix}: checked {min(i + batch, len(grid)):,} candidates -> {total}/{n} cropland")
            if total >= n:
                break
        df = pd.concat(found, ignore_index=True).head(n)
        if len(df) < n:
            print(f"      WARNING: only {len(df)}/{n} cropland points exist in {prefix}; using those.")
        return df

    return cached(f"{prefix}_points_{n}.csv", build, ["Longitude", "Latitude"])


def _placeholder(bands):
    """Fully-masked image so an empty decade still yields the right bands instead of an error."""
    return (ee.Image.constant([0] * len(bands)).rename(bands).toFloat()
            .updateMask(ee.Image.constant(0)))


def fetch_optical(prefix, roi, pts, year, lo, hi):
    names = OPT + ["N_Optical_Obs"]

    def build():
        pfc, ph = points_fc(pts), _placeholder(OPT)
        imgs = []
        for n in range(lo, hi + 1):
            s, e = decade_bounds(year, n)
            col = (gp.build_s2_collection(roi, s, e).select(OPT).map(lambda im: im.toFloat())
                   .merge(ee.ImageCollection([ph])))
            comp = col.median().addBands(col.select("NDVI").count().rename("N_Optical_Obs"))
            imgs.append(comp.rename([f"{b}_{n}" for b in names]))
        # ONE image with every decade as bands -> ONE sampling call (no concurrent aggregations)
        stack = ee.Image.cat(imgs).toFloat().unmask(UNMASK)
        wide = fetch(stack.sampleRegions(collection=pfc, scale=gp.GRID_SPACING_M, tileScale=4),
                     f"{prefix} optical {year} d{lo}-{hi}")
        long = wide_to_long(wide, names, lo, hi)
        long = long[long["NDVI"].notna()].copy()
        long["Year"] = year
        return long
    return cached(f"{prefix}_{len(pts)}px_optical_{year}_d{lo}-{hi}.csv", build,
                  ["Longitude", "Latitude", "Decade_N"] + names + ["Year"])


def fetch_sar(prefix, roi, pts, year, lo, hi):
    names = ["SAR_VV_dB"]

    def build():
        pfc, ph = points_fc(pts), _placeholder(names)
        imgs = []
        for n in range(lo, hi + 1):
            s, e = decade_bounds(year, n)
            col = (gp.get_sar_collection(roi, s, e).map(lambda im: im.toFloat())
                   .merge(ee.ImageCollection([ph])))
            imgs.append(col.mean().rename([f"SAR_VV_dB_{n}"]))
        stack = ee.Image.cat(imgs).toFloat().unmask(UNMASK)
        wide = fetch(stack.sampleRegions(collection=pfc, scale=gp.GRID_SPACING_M, tileScale=4),
                     f"{prefix} sar {year} d{lo}-{hi}")
        long = wide_to_long(wide, names, lo, hi)
        long = long[long["SAR_VV_dB"].notna()].copy()
        long["Year"] = year
        return long
    return cached(f"{prefix}_{len(pts)}px_sar_{year}_d{lo}-{hi}.csv", build,
                  ["Longitude", "Latitude", "Decade_N", "SAR_VV_dB", "Year"])


def fetch_weather(prefix, roi, year):
    def build():
        fc = gp.get_era5_weather_collection(roi, f"{year}-01-01", f"{year + 1}-01-01")
        return fetch(fc, f"{prefix} weather {year}")
    return cached(f"{prefix}_weather_{year}.csv", build,
                  ["date", "Mean_Max_Temp_K", "Mean_Min_Temp_K", "Mean_Temp_K", "Rain_m", "Solar_J_m2"])


def fetch_soil(prefix, pts):
    def build():
        fc = gp.get_soil_ph_image().sampleRegions(collection=points_fc(pts),
                                                  scale=gp.GRID_SPACING_M, tileScale=4)
        return fetch(fc, f"{prefix} soil")
    return cached(f"{prefix}_{len(pts)}px_soil.csv", build, ["Longitude", "Latitude", "Soil_pH"])


# ----------------------------------------------------------------------
# one district
# ----------------------------------------------------------------------
def with_pixel_id(df):
    df = df.copy()
    df["Pixel_ID"] = gp._pixel_id(df)
    return df


def run_district(prefix, n_pixels, yields):
    district = gp.DISTRICTS[prefix]
    print(f"\n{'=' * 60}\n[{district}]\n{'=' * 60}")
    roi = roi_for(prefix)
    if n_pixels > MAX_PIXELS:
        raise SystemExit(f"--pixels max is {MAX_PIXELS} (getInfo returns at most 5000 rows)")
    pts = get_points(prefix, roi, n_pixels)
    print(f"  {len(pts)} cropland pixels")
    chunks = decade_chunks(len(pts))
    years = range(gp.START_YEAR, gp.END_YEAR + 1)

    opt, sar, wx = [], [], []
    for y in years:
        print(f"  {y}")
        for lo, hi in chunks:
            opt.append(fetch_optical(prefix, roi, pts, y, lo, hi))
            sar.append(fetch_sar(prefix, roi, pts, y, lo, hi))
        wx.append(fetch_weather(prefix, roi, y))
    soil = fetch_soil(prefix, pts)

    def stack(frames, num_cols):
        """Concatenate cached chunks; empty chunks read back as text columns, so force numeric."""
        frames = [f for f in frames if len(f)]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["Longitude", "Latitude", "Year", "Decade_N"] + num_cols)
        for c in ["Longitude", "Latitude", "Year", "Decade_N"] + num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return with_pixel_id(df)

    opt = stack(opt, OPT + ["N_Optical_Obs"])[["Pixel_ID", "Year", "Decade_N"] + OPT + ["N_Optical_Obs"]]
    sar = stack(sar, ["SAR_VV_dB"])[["Pixel_ID", "Year", "Decade_N", "SAR_VV_dB"]]
    soil["Soil_pH"] = pd.to_numeric(soil["Soil_pH"], errors="coerce")
    soil = with_pixel_id(soil)[["Pixel_ID", "Soil_pH"]]
    wx = pd.concat(wx, ignore_index=True)
    for c in ["Mean_Max_Temp_K", "Mean_Min_Temp_K", "Mean_Temp_K", "Rain_m", "Solar_J_m2"]:
        wx[c] = pd.to_numeric(wx[c], errors="coerce")
    wx = gp.weather_to_decades(wx)
    return gp.assemble_district(opt, sar, soil, wx, district, yields)


# ----------------------------------------------------------------------
def main():
    global CACHE_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--districts", nargs="*", default=list(gp.DISTRICTS))
    ap.add_argument("--pixels", type=int, default=N_PIXELS, help="cropland pixels per district")
    ap.add_argument("--yield-csv", default="malaysia_granary_annual_ml_ready.csv")
    ap.add_argument("--out", default="DailyGranary.csv")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--end-year", type=int, default=None,
                    help=f"last year to extract (default {gp.END_YEAR}). Finished years stay cached, so raising it only fetches the new year(s)")
    a = ap.parse_args()

    CACHE_DIR = a.cache_dir
    if a.end_year:
        gp.END_YEAR = a.end_year
    gp.init_ee()
    yields = gp.load_yield(a.yield_csv)
    print(f"Districts requested: {', '.join(a.districts)}")
    frames, failed = [], {}
    base = os.path.splitext(a.out)[0]
    for p in a.districts:
        try:
            df = run_district(p, a.pixels, yields)
        except Exception as e:                      # one bad district must not lose the others
            failed[p] = f"{type(e).__name__}: {e}"
            print(f"\n!!! {p} FAILED -> {failed[p]}\n    Cached pieces are kept; fix and re-run to retry it.")
            continue
        gp.write_table(df, f"{base}_{p}.csv", quiet=True)    # safe per-district copy
        print(f"  saved {base}_{p}.csv")
        frames.append(df)

    if not frames:
        raise SystemExit("No district finished, nothing written.")
    gp.write_table(pd.concat(frames, ignore_index=True), a.out)
    if failed:
        print(f"\n!!! {a.out} is INCOMPLETE. Missing districts: {', '.join(failed)}")
        for k, v in failed.items():
            print(f"    {k}: {v}")
    else:
        print(f"\nAll {len(frames)} districts included in {a.out}.")


if __name__ == "__main__":
    main()