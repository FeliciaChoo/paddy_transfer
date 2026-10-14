"""
granary_pipeline.py  --  GEE extraction + DailyGranary table in ONE script

    python granary_pipeline.py extract      # build + submit GEE exports (resumable)
    python granary_pipeline.py process      # build the decade table from downloaded CSVs
    python granary_pipeline.py all          # extract, wait for GEE, then process

Flow:
    GEE (Sentinel-2, Sentinel-1, ERA5, soil)  --export-->  Google Drive folder
    Drive folder  --(Drive for Desktop sync, or download)-->  EXPORT_DIR on your machine
    EXPORT_DIR + yield CSV  -->  decade table (SubSeasonal_Period from planting, DAP-based Stage)

Earth Engine can only export to Drive / Cloud Storage / Assets, so the CSVs must reach
EXPORT_DIR before 'process' can read them. 'all' stops with a list of missing files if they
are not there yet (or waits, see --wait-minutes).

'process' needs only pandas + numpy. 'extract' also needs earthengine-api.

FIXES vs the old extraction.py
  * No double baseline correction (S2_SR_HARMONIZED already removes the post-2022-01-25 offset).
  * Cloud masking uses Cloud Score+ instead of QA60. QA60 is not reliably populated for
    scenes from 25 Jan 2022 onward, so clouds slip through unmasked in that period.
  * End dates are exclusive in Earth Engine, so ranges now end on 1 Jan of the next year.
  * Optical exports carry the suffix OPTICAL_SUFFIX so old (buggy) tasks/files are never reused.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import ee
except ImportError:           # lets 'process' run without earthengine-api installed
    ee = None

# ======================================================================
# CONFIG
# ======================================================================
EE_PROJECT = "paddy-499912"
DRIVE_FOLDER = "paddy_transfer_exports"

START_YEAR, END_YEAR = 2018, 2024
BUFFER_DEG = 0.01
GRID_SPACING_M = 250
MAX_CONCURRENT = 2
POLL_INTERVAL_SEC = 30

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CSPLUS_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CS_THRESHOLD = 0.60          # cs_cdf >= this = clear (Google suggests 0.5-0.65)
MAX_SCENE_CLOUD = 70         # scene-level CLOUDY_PIXEL_PERCENTAGE pre-filter
OPTICAL_SUFFIX = "_v2"

# lon_min, lon_max, lat_min, lat_max
ROIS = {
    "kuala_selangor": (101.1683, 101.4917, 3.1800, 3.5753),
    "sabak_bernam":   (100.8808, 101.3479, 3.5394, 3.8268),
    "pasir_puteh":    (102.2822, 102.5337, 5.7672, 5.9109),
    "perak_tengah":   (100.7730, 101.0964, 3.9706, 4.5455),
    "kota_setar":           (100.27, 100.6293, 5.9828, 6.2343),    # same yield series as Kubang Pasu, Yan
    "kerian":               (100.3777, 100.7011, 4.8689, 5.1204),
    "seberang_perai_utara": (100.3777, 100.5215, 5.4079, 5.5516),
    "kota_bharu":           (102.1744, 102.3181, 5.9109, 6.1265),  # same yield series as Pasir Mas, Tumpat
    "besut":                (102.39, 102.6415, 5.3719, 5.8031),
    # --- added: IADA Pekan / IADA Rompin (Pahang). APPROXIMATE boxes, not measured from your yield file.
    # Pekan paddy sits along the Pahang river (Mambang, Pulau Jawa, Pulau Tambun, Ganchong, Pahang Tua, Serandu);
    # Rompin paddy is the coastal "paya" blocks (Paya Setajam, Paya Laka, Paya Sepayang, Paya Tebu Hitam).
    # Only WorldCover cropland pixels are kept, so a slightly large box is harmless; a box that MISSES the paddy
    # shows up as "only X/80 cropland points" when granary_direct.py runs. Tighten to the min/max Longitude/Latitude
    # of that district in malaysia_granary_annual_ml_ready.csv if you want it exact.
    "pekan":                (103.30, 103.50, 3.40, 3.65),
    "rompin":               (103.40, 103.60, 2.65, 2.90),
}
DISTRICTS = {   # export prefix -> District name used in the yield CSV
    "kuala_selangor": "Kuala Selangor",
    "sabak_bernam":   "Sabak Bernam",
    "pasir_puteh":    "Pasir Puteh",
    "perak_tengah":   "Perak Tengah",
    "kota_setar":           "Kota Setar",
    "kerian":               "Kerian",
    "seberang_perai_utara": "Seberang Perai Utara",
    "kota_bharu":           "Kota Bharu",
    "besut":                "Besut",
    "pekan":                "Pekan",
    "rompin":               "Rompin",
}

# District (as named in DISTRICTS / the yield CSV) -> granary scheme. Written to the "Granary" column of DailyGranary.csv.
# Districts that share one yield series are listed in the comment; only the extracted district is a row in the table.
# Edit here if a mapping is wrong (e.g. Perak Tengah -> IADA Seberang Perak is a guess).
GRANARY_OF = {
    "Kota Setar":           "MADA",                      # also Kubang Pasu, Yan (same yield series)
    "Kota Bharu":           "KADA",                      # also Pasir Mas, Tumpat (same yield series)
    "Kerian":               "IADA Kerian",
    "Kuala Selangor":       "IADA Barat Laut Selangor",
    "Sabak Bernam":         "IADA Barat Laut Selangor",
    "Perak Tengah":         "IADA Seberang Perak",       # guess
    "Seberang Perai Utara": "IADA Pulau Pinang",
    "Besut":                "KETARA",
    "Pasir Puteh":          "IADA Kemasin Semerak",      # also Bachok
    "Pekan":                "IADA Pekan",
    "Rompin":               "IADA Rompin",
}

# ---- processing settings ----
# ASSUMPTION: season starts (planting) as (month, day). Replace with the real local calendar.
SEASON_STARTS = {"Season_1_Off": (1, 1), "Season_2_Main": (7, 1)}
CROP_DURATION_DAYS = 120
MAX_GAP_DECADES = 3          # interpolate optical gaps up to 30 days; longer stays NaN
NDVI_TOLERANCE = 0.10
TREND_THRESHOLD = 0.03
MAX_PIXELS_PER_DISTRICT = None   # e.g. 80 to subsample; None = all cropland pixels
RANDOM_SEED = 42
LEGACY_COLUMN_NAMES = True       # old sheet calls the decade rain TOTAL "Daily_Rain_mm"

STAGES = [  # (name, DAP_lo, DAP_hi, NDVI_lo, NDVI_hi, NDVI_rep, coarse group)
    ("Germination_Establishment",      0,  10, 0.10, 0.25, 0.18, "Nursery_LandPrep"),
    ("Seedling",                      11,  20, 0.20, 0.40, 0.30, "Nursery_LandPrep"),
    ("Tillering",                     21,  40, 0.35, 0.65, 0.50, "Vegetative"),
    ("Max_Tillering_Stem_Elongation", 41,  55, 0.55, 0.75, 0.65, "Vegetative"),
    ("Panicle_Initiation_Booting",    56,  75, 0.65, 0.85, 0.75, "Reproductive"),
    ("Heading_Flowering",             76,  90, 0.65, 0.80, 0.72, "Reproductive"),
    ("Grain_Filling",                 91, 105, 0.45, 0.70, 0.58, "Ripening_Harvest"),
    ("Maturity_Harvest",             106, 120, 0.20, 0.50, 0.35, "Ripening_Harvest"),
]
OPTICAL_COLS = ["NDVI", "EVI", "MNDWI", "LSWI"]
MANIFEST = "extract_manifest.json"


# ======================================================================
# PART 1 -- GEE EXTRACTION
# ======================================================================
def init_ee():
    if ee is None:
        sys.exit("earthengine-api is not installed: pip install earthengine-api")
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=EE_PROJECT)


def get_era5_weather_collection(roi_geom, start_date, end_date):
    # one value per date for the whole district (ERA5 is ~9 km, so per-pixel is pointless)
    weather_point = roi_geom.centroid().buffer(10000)
    era5 = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .filterBounds(weather_point).filterDate(start_date, end_date))

    def extract_weather(image):
        stats = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=weather_point, scale=9000)
        rain = ee.Algorithms.If(stats.contains("total_precipitation_sum"),
                                stats.get("total_precipitation_sum"), stats.get("total_precipitation"))
        solar = ee.Algorithms.If(stats.contains("surface_solar_radiation_downwards_sum"),
                                 stats.get("surface_solar_radiation_downwards_sum"),
                                 stats.get("surface_solar_radiation_downwards"))
        return ee.Feature(None, {
            "date": image.date().format("YYYY-MM-dd"),
            "Mean_Max_Temp_K": stats.get("temperature_2m_max"),
            "Mean_Min_Temp_K": stats.get("temperature_2m_min"),
            "Mean_Temp_K": stats.get("temperature_2m"),
            "Rain_m": rain, "Solar_J_m2": solar})

    return era5.map(extract_weather)


def mask_s2_clouds(image):
    """Cloud Score+ mask. Expects the 'cs_cdf' band linked in by build_s2_collection()."""
    clear = image.select("cs_cdf").gte(CS_THRESHOLD)
    return (image.select("B.*").updateMask(clear).divide(10000)
            .copyProperties(image, ["system:time_start"]))


def mask_to_cropland(image):
    return image.updateMask(ee.Image("ESA/WorldCover/v200/2021").select("Map").eq(40))


def add_optical_indices(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nir, red, blue = image.select("B8"), image.select("B4"), image.select("B2")
    evi = (nir.subtract(red).multiply(2.5)
           .divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1))
           .rename("EVI").clamp(-1, 1))
    mndwi = image.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    lswi = image.normalizedDifference(["B8", "B11"]).rename("LSWI")
    return image.addBands([ndvi, evi, mndwi, lswi]).copyProperties(image, ["system:time_start"])


def build_s2_collection(roi_geom, start, end):
    col = (ee.ImageCollection(S2_COLLECTION).filterBounds(roi_geom).filterDate(start, end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_SCENE_CLOUD))
           .linkCollection(ee.ImageCollection(CSPLUS_COLLECTION), ["cs_cdf"]))
    return col.map(mask_s2_clouds).map(add_optical_indices).map(mask_to_cropland)


def get_sar_collection(roi_geom, start_date, end_date):
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(roi_geom)
          .filterDate(start_date, end_date)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .select("VV"))
    return s1.map(lambda im: im.focal_mean(radius=30, units="meters")
                  .rename("SAR_VV_dB").copyProperties(im, ["system:time_start"]))


def get_soil_ph_image():
    return (ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02")
            .select("b0").divide(10).rename("Soil_pH"))


def generate_grid_points(roi_geom, spacing_m=250):
    grid = roi_geom.coveringGrid(roi_geom.projection().atScale(spacing_m), spacing_m)
    points = grid.map(lambda f: ee.Feature(f.geometry().centroid(1), {}))
    return points.filterBounds(roi_geom)


def sample_image(image, band_names, points, scale, tile_scale=4):
    samples = image.select(band_names).sampleRegions(
        collection=points, scale=scale, geometries=True, tileScale=tile_scale)

    def add_metadata(f):
        c = f.geometry().coordinates()
        return (f.set("Longitude", c.get(0)).set("Latitude", c.get(1))
                .set("date", image.date().format("YYYY-MM-dd")))
    return samples.map(add_metadata)


def sample_soil(soil_image, points, scale, tile_scale=4):
    samples = soil_image.sampleRegions(collection=points, scale=scale,
                                       geometries=True, tileScale=tile_scale)
    return samples.map(lambda f: f.set("Longitude", f.geometry().coordinates().get(0))
                       .set("Latitude", f.geometry().coordinates().get(1)))


def _to_drive(collection, label):
    return ee.batch.Export.table.toDrive(collection=collection, description=label,
                                         folder=DRIVE_FOLDER, fileNamePrefix=label,
                                         fileFormat="CSV")


def build_export_tasks(name, roi_geom, start_date, end_date, grid_spacing_m=GRID_SPACING_M,
                       tile_scale=4, optical_split_threshold=15000):
    """Returns [(label, UNSTARTED task)]. end_date is exclusive (pass 1 Jan of the next year)."""
    print(f"[{name}] Building sample grid ({grid_spacing_m} m)...")
    points = generate_grid_points(roi_geom, grid_spacing_m)
    n_points = points.size().getInfo()
    split = n_points > optical_split_threshold
    print(f"[{name}] {n_points} grid points -> optical {'per-year' if split else 'single task'}")

    tasks = []

    def optical_task(s, e, label):
        fc = build_s2_collection(roi_geom, s, e).map(
            lambda im: sample_image(im, OPTICAL_COLS, points, grid_spacing_m, tile_scale)).flatten()
        return (label, _to_drive(fc, label))

    if split:
        for year in range(int(start_date[:4]), int(end_date[:4])):
            tasks.append(optical_task(f"{year}-01-01", f"{year + 1}-01-01",
                                      f"{name}_optical_{year}{OPTICAL_SUFFIX}"))
    else:
        tasks.append(optical_task(start_date, end_date, f"{name}_optical{OPTICAL_SUFFIX}"))

    s1 = get_sar_collection(roi_geom, start_date, end_date).map(
        lambda im: sample_image(im, ["SAR_VV_dB"], points, grid_spacing_m, tile_scale)).flatten()
    tasks.append((f"{name}_sar", _to_drive(s1, f"{name}_sar")))
    tasks.append((f"{name}_soil", _to_drive(
        sample_soil(get_soil_ph_image(), points, grid_spacing_m, tile_scale), f"{name}_soil")))
    tasks.append((f"{name}_weather", _to_drive(
        get_era5_weather_collection(roi_geom, start_date, end_date), f"{name}_weather")))
    return tasks


def run_extraction():
    """Resumable: reads the real Earth Engine task list; COMPLETED tasks are skipped,
    RUNNING/READY ones are re-attached to, FAILED/CANCELLED/new ones are (re)submitted."""
    init_ee()
    print("Checking existing task states on Earth Engine...")
    existing = {}
    for t in ee.batch.Task.list():
        try:
            st = t.status()
            if st.get("description"):
                existing[st["description"]] = (st.get("state"), t)
        except Exception:
            continue

    pending, running, finished, manifest = [], {}, {}, {}
    for name, (lon0, lon1, lat0, lat1) in ROIS.items():
        roi = ee.Geometry.Rectangle([lon0 - BUFFER_DEG, lat0 - BUFFER_DEG,
                                     lon1 + BUFFER_DEG, lat1 + BUFFER_DEG])
        print(f"\n{'=' * 60}\n[{name}]\n{'=' * 60}")
        try:
            district_tasks = build_export_tasks(name, roi, f"{START_YEAR}-01-01",
                                                f"{END_YEAR + 1}-01-01")
        except Exception as e:
            print(f"[{name}] FAILED TO BUILD TASKS: {e}")
            continue
        manifest[name] = [lbl for lbl, _ in district_tasks]
        for label, task in district_tasks:
            prev = existing.get(label)
            if prev is None:
                pending.append((label, task)); print(f"  {label}: new, will submit")
            elif prev[0] == "COMPLETED":
                finished[label] = "COMPLETED"; print(f"  {label}: already COMPLETED, skipping")
            elif prev[0] in ("RUNNING", "READY"):
                running[label] = prev[1]; print(f"  {label}: already {prev[0]}, resuming poll")
            else:
                pending.append((label, task)); print(f"  {label}: previously {prev[0]}, retrying")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n{len(finished)} done, {len(running)} active, {len(pending)} to submit "
          f"(manifest saved to {MANIFEST}).\n")

    while pending or running:
        while len(running) < MAX_CONCURRENT and pending:
            label, task = pending.pop(0)
            try:
                task.start(); running[label] = task
                print(f"[STARTED] {label} (id={task.id})")
            except Exception as e:
                print(f"[FAILED TO START] {label}: {e}"); finished[label] = "FAILED_TO_START"
        for label in list(running):
            try:
                st = running[label].status(); state = st.get("state", "UNKNOWN")
            except Exception as e:
                print(f"[{label}] status check failed: {e}"); continue
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                print(f"[{state}] {label}")
                if state == "FAILED":
                    print(f"    error: {st.get('error_message', 'no message')}")
                finished[label] = state; del running[label]
        if pending or running:
            print(f"...{len(running)} running, {len(pending)} queued, {len(finished)} finished. "
                  f"Next check in {POLL_INTERVAL_SEC}s.")
            time.sleep(POLL_INTERVAL_SEC)

    bad = {k: v for k, v in finished.items() if v != "COMPLETED"}
    print(f"\nEXTRACTION DONE: {len(finished) - len(bad)} completed, {len(bad)} failed/other.")
    if bad:
        print("Failed:", bad, "\nRe-run the same command; completed tasks are skipped.")
    print(f"Exports are in Google Drive folder '{DRIVE_FOLDER}'.")
    return not bad


# ======================================================================
# PART 2 -- DAILY GRANARY TABLE
# ======================================================================
def _pixel_id(df):
    return (df["Longitude"].round(5).map("{:.5f}".format) + "_" +
            df["Latitude"].round(5).map("{:.5f}".format))


def add_keys(df):
    d = pd.to_datetime(df["date"])
    df["Year"] = d.dt.year
    df["Decade_N"] = np.minimum((d.dt.dayofyear - 1) // 10 + 1, 36)  # 36 blocks from 1 Jan
    df["Pixel_ID"] = _pixel_id(df)
    return df


def load_optical(export_dir, prefix):
    files = sorted(glob.glob(os.path.join(export_dir, f"{prefix}_optical*{OPTICAL_SUFFIX}*.csv")))
    if not files:
        raise FileNotFoundError(f"No '{prefix}_optical*{OPTICAL_SUFFIX}*.csv' in {export_dir}")
    parts = []
    for f in files:
        d = add_keys(pd.read_csv(f, usecols=["date", "Latitude", "Longitude"] + OPTICAL_COLS))
        parts.append(d.groupby(["Pixel_ID", "Year", "Decade_N"]).agg(
            **{c: (c, "median") for c in OPTICAL_COLS}, N_Optical_Obs=("NDVI", "count")).reset_index())
        print(f"    optical: {os.path.basename(f)} -> {len(d):,} raw rows")
    return pd.concat(parts).groupby(["Pixel_ID", "Year", "Decade_N"]).agg(
        **{c: (c, "mean") for c in OPTICAL_COLS}, N_Optical_Obs=("N_Optical_Obs", "sum")).reset_index()


def load_sar(export_dir, prefix):
    d = add_keys(pd.read_csv(os.path.join(export_dir, f"{prefix}_sar.csv"),
                             usecols=["date", "Latitude", "Longitude", "SAR_VV_dB"]))
    print(f"    sar: {len(d):,} raw rows")
    return d.groupby(["Pixel_ID", "Year", "Decade_N"], as_index=False)["SAR_VV_dB"].mean()


def load_soil(export_dir, prefix):
    d = pd.read_csv(os.path.join(export_dir, f"{prefix}_soil.csv"),
                    usecols=["Latitude", "Longitude", "Soil_pH"])
    d["Pixel_ID"] = _pixel_id(d)
    return d.groupby("Pixel_ID", as_index=False)["Soil_pH"].mean()


def load_weather(export_dir, prefix):
    return weather_to_decades(pd.read_csv(os.path.join(export_dir, f"{prefix}_weather.csv")))


def weather_to_decades(w):
    """Daily ERA5 rows (date + K/m/J units) -> decade-level district weather."""
    w = w.drop_duplicates("date").copy()
    d = pd.to_datetime(w["date"])
    w["Year"] = d.dt.year
    w["Decade_N"] = np.minimum((d.dt.dayofyear - 1) // 10 + 1, 36)
    for src, dst in [("Mean_Max_Temp_K", "Mean_Max_Temp_C"), ("Mean_Min_Temp_K", "Mean_Min_Temp_C"),
                     ("Mean_Temp_K", "Mean_Temp_C")]:
        w[dst] = w[src] - 273.15
    w["Rain_mm"] = w["Rain_m"] * 1000.0
    w["Solar_Radiation_MJ"] = w["Solar_J_m2"] / 1e6
    return w.groupby(["Year", "Decade_N"], as_index=False).agg(
        Mean_Max_Temp_C=("Mean_Max_Temp_C", "mean"), Mean_Min_Temp_C=("Mean_Min_Temp_C", "mean"),
        Mean_Temp_C=("Mean_Temp_C", "mean"),
        Decade_Rain_mm=("Rain_mm", "sum"),            # decade TOTAL, not daily
        Solar_Radiation_MJ=("Solar_Radiation_MJ", "mean"))


def load_yield(path):
    y = pd.read_csv(path, usecols=["Year", "District", "Target_Yield"])
    if (y.groupby(["District", "Year"])["Target_Yield"].nunique() > 1).any():
        print("WARNING: several Target_Yield values per District/Year; using the first.")
    return y.groupby(["District", "Year"], as_index=False)["Target_Yield"].first()


def season_and_dap(mid_date):
    cands = []
    for y in (mid_date.year - 1, mid_date.year):
        for name, (m, d) in SEASON_STARTS.items():
            s = pd.Timestamp(y, m, d)
            if s <= mid_date:
                cands.append((s, name))
    start, name = max(cands)
    return name, (mid_date - start).days


def stage_from_dap(dap):
    if dap <= CROP_DURATION_DAYS:
        for name, lo, hi, nlo, nhi, _, group in STAGES:
            if lo <= dap <= hi:
                return name, nlo, nhi, group
    return "Fallow", np.nan, np.nan, "Fallow"


def add_season_and_stage(df):
    lookup = {}
    for s in pd.Series(df["Start_Date"].unique()):
        season, dap = season_and_dap(s + pd.Timedelta(days=5))   # decade midpoint
        stage, nlo, nhi, group = stage_from_dap(dap)
        lookup[s] = dict(Cropping_Season=season, DAP=dap,
                         SubSeasonal_Period=f"Decade_{dap // 10 + 1}" if stage != "Fallow" else "Fallow",
                         Stage=stage, Stage_Group=group, _nlo=nlo, _nhi=nhi)
    df = df.join(pd.DataFrame.from_dict(lookup, orient="index"), on="Start_Date")

    dn = df["NDVI_Growth_Velocity"]
    df["Trend"] = np.select([dn > TREND_THRESHOLD, dn < -TREND_THRESHOLD],
                            ["Greening", "Senescing"], default="Stable")
    df.loc[dn.isna(), "Trend"] = np.nan

    in_range = (df["NDVI"] >= df["_nlo"] - NDVI_TOLERANCE) & (df["NDVI"] <= df["_nhi"] + NDVI_TOLERANCE)
    trend_ok = np.where(df["DAP"] <= 75, dn >= -NDVI_TOLERANCE,
                        np.where(df["DAP"] >= 91, dn <= NDVI_TOLERANCE, True))
    consistent = (in_range & trend_ok).astype(object)
    consistent[df["NDVI"].isna() | df["Stage"].eq("Fallow")] = np.nan
    df["Stage_Consistent"] = consistent
    return df.drop(columns=["_nlo", "_nhi"])


def build_district(export_dir, prefix, district, yields):
    print(f"\n[{district}]")
    return assemble_district(load_optical(export_dir, prefix), load_sar(export_dir, prefix),
                             load_soil(export_dir, prefix), load_weather(export_dir, prefix),
                             district, yields)


def assemble_district(opt, sar, soil, wx, district, yields):
    """opt/sar: decade-level per pixel; soil: per pixel; wx: decade-level district weather."""
    pixels = sorted(opt["Pixel_ID"].unique())    # present in optical = passed cropland mask
    if MAX_PIXELS_PER_DISTRICT and len(pixels) > MAX_PIXELS_PER_DISTRICT:
        pixels = sorted(np.random.RandomState(RANDOM_SEED).choice(
            pixels, MAX_PIXELS_PER_DISTRICT, replace=False))
    print(f"    {len(pixels)} cropland pixels kept")

    keys = ["Pixel_ID", "Year", "Decade_N"]
    df = pd.MultiIndex.from_product([pixels, range(START_YEAR, END_YEAR + 1), range(1, 37)],
                                    names=keys).to_frame(index=False)
    df = (df.merge(opt, on=keys, how="left").merge(sar, on=keys, how="left")
            .merge(wx, on=["Year", "Decade_N"], how="left").merge(soil, on="Pixel_ID", how="left"))
    df["N_Optical_Obs"] = df["N_Optical_Obs"].fillna(0).astype(int)
    df["Optical_Observed"] = df["N_Optical_Obs"] > 0

    df = df.sort_values(keys).reset_index(drop=True)
    for c in OPTICAL_COLS:   # short gaps interpolated; long gaps (e.g. early 2018) stay NaN
        df[c] = df.groupby("Pixel_ID")[c].transform(
            lambda s: s.interpolate(limit=MAX_GAP_DECADES, limit_area="inside"))
    df["NDVI_Growth_Velocity"] = df.groupby("Pixel_ID")["NDVI"].diff()
    df["Rolling_Rain_3_Decades"] = df.groupby("Pixel_ID")["Decade_Rain_mm"].transform(
        lambda s: s.rolling(3, min_periods=1).sum())

    df["Start_Date"] = (pd.to_datetime(df["Year"].astype(str) + "-01-01")
                        + pd.to_timedelta((df["Decade_N"] - 1) * 10, unit="D"))
    df["Month"] = df["Start_Date"].dt.month
    df["District"] = district
    df["Longitude"] = df["Pixel_ID"].str.split("_").str[0].astype(float)
    df["Latitude"] = df["Pixel_ID"].str.split("_").str[1].astype(float)
    df = df.merge(yields, on=["District", "Year"], how="left")
    df = add_season_and_stage(df)
    print(f"    pixels without soil match: {df['Soil_pH'].isna().groupby(df['Pixel_ID']).all().sum()}")
    return df


def missing_exports(export_dir, prefixes):
    """Expected export files not yet in export_dir (uses the manifest if present)."""
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}
    missing = []
    for p in prefixes:
        labels = manifest.get(p) or [f"{p}_sar", f"{p}_soil", f"{p}_weather"]
        for lbl in labels:
            if not glob.glob(os.path.join(export_dir, f"{lbl}*.csv")):
                missing.append(lbl)
        if not glob.glob(os.path.join(export_dir, f"{p}_optical*{OPTICAL_SUFFIX}*.csv")):
            missing.append(f"{p}_optical*{OPTICAL_SUFFIX}")
    return sorted(set(missing))


def _swap_in(tmp, out, tries=5):
    """Move tmp over out. On Windows this fails with 'Access is denied' if out is open in Excel
    or locked by OneDrive; retry briefly, then fall back to a new timestamped filename."""
    for i in range(tries):
        try:
            os.replace(tmp, out)
            return out
        except PermissionError:
            time.sleep(2)
    alt = f"{os.path.splitext(out)[0]}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    os.replace(tmp, alt)
    print(f"\n!!! Could not overwrite '{out}' (open in Excel, or locked by OneDrive sync?).\n"
          f"    Saved to '{alt}' instead. Close the old file to use the normal name next time.")
    return alt


def write_table(df, out, quiet=False):
    rain = "Decade_Rain_mm"
    if LEGACY_COLUMN_NAMES:
        df = df.rename(columns={rain: "Daily_Rain_mm"}); rain = "Daily_Rain_mm"
    cols = ["Year", "SubSeasonal_Period", "Stage", "District", "Longitude", "Latitude",
            "Target_Yield", "Start_Date", "NDVI", "EVI", "MNDWI", "LSWI", "SAR_VV_dB",
            "Mean_Max_Temp_C", "Mean_Min_Temp_C", "Mean_Temp_C", rain, "Solar_Radiation_MJ",
            "Soil_pH", "Pixel_ID", "NDVI_Growth_Velocity", "Month", "Cropping_Season",
            "Rolling_Rain_3_Decades",
            "DAP", "Stage_Group", "Trend", "Stage_Consistent", "Optical_Observed",
            "N_Optical_Obs", "Decade_N"]
    df = df.assign(Granary=df["District"].map(GRANARY_OF))   # mapped from District
    unmapped = sorted(df.loc[df["Granary"].isna(), "District"].unique())
    if unmapped:
        print(f"WARNING: no granary mapping for {unmapped}; add them to GRANARY_OF")
    cols.insert(cols.index("District") + 1, "Granary")       # Granary right after District
    tmp = out + ".tmp"                      # write to a temp file, then swap in:
    df[cols].to_csv(tmp, index=False)       # a half-written file can never replace a good one
    out = _swap_in(tmp, out)
    if quiet:
        return
    print(f"\nSaved {len(df):,} rows -> {out}")
    print("\nRows per district in the saved file:")
    print(df.groupby("District").agg(rows=("Year", "size"), pixels=("Pixel_ID", "nunique")).to_string())
    print("\nShare of rows WITH a real optical observation (%):")
    print((df.groupby(["District", "Year"])["Optical_Observed"].mean() * 100)
          .round(1).unstack().to_string())


def run_process(export_dir, yield_csv, out, prefixes, allow_partial=False):
    miss = missing_exports(export_dir, prefixes)
    if miss:
        print(f"\nMissing export files in '{export_dir}':")
        for m in miss:
            print(f"   - {m}.csv")
        if not allow_partial:
            print("\nDownload/sync them from Drive (or use --allow-partial to skip affected "
                  "districts) and run 'process' again.")
            return False
    yields = load_yield(yield_csv)
    frames = []
    for p in prefixes:
        try:
            frames.append(build_district(export_dir, p, DISTRICTS[p], yields))
        except FileNotFoundError as e:
            print(f"[{DISTRICTS[p]}] SKIPPED: {e}")
    if not frames:
        print("Nothing to process."); return False
    write_table(pd.concat(frames, ignore_index=True), out)
    return True


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["extract", "process", "all"])
    ap.add_argument("--export-dir", default=DRIVE_FOLDER,
                    help="local folder holding the CSVs downloaded/synced from Drive")
    ap.add_argument("--yield-csv", default="malaysia_granary_annual_ml_ready.csv")
    ap.add_argument("--out", default="DailyGranary_rebuilt.csv")
    ap.add_argument("--districts", nargs="*", default=list(DISTRICTS))
    ap.add_argument("--allow-partial", action="store_true",
                    help="process even if some districts/files are missing")
    ap.add_argument("--wait-minutes", type=int, default=0,
                    help="(all) after GEE finishes, wait this long for files to appear in --export-dir")
    a = ap.parse_args()

    if a.command in ("extract", "all"):
        ok = run_extraction()
        if a.command == "extract":
            return
        if not ok and not a.allow_partial:
            sys.exit("Some exports failed; fix/re-run, or use --allow-partial.")
        waited = 0
        while a.wait_minutes > waited and missing_exports(a.export_dir, a.districts):
            print(f"Waiting for files in '{a.export_dir}'... ({waited}/{a.wait_minutes} min)")
            time.sleep(60); waited += 1

    run_process(a.export_dir, a.yield_csv, a.out, a.districts, a.allow_partial)


if __name__ == "__main__":
    main()