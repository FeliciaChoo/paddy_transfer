"""paddy_mask.py  --  tag every sampled pixel against the PLANMalaysia paddy polygons. NON-DESTRUCTIVE.

    python pipeline/paddy_mask.py --paddy geo/paddy_planmalaysia.geojson --pixels DailyGranary_by_granary.csv --out data

Same philosophy as eda_clean.py: NO pixel and NO row is ever deleted. The polygons are cleaned (invalid geometry repaired,
zero-area slivers and fragments below --min-ha dropped FROM THE POLYGON LAYER only), then each pixel is tagged:

    In_Paddy            the pixel centre falls inside a cleaned paddy polygon
    Paddy_Class         scheme            inside a polygon that names a granary scheme (nmskim)
                        unnamed_granary   inside a polygon flagged 'Dalam' (in a granary) but with no scheme name
                        non_granary       inside a polygon flagged 'Luar' (outside every granary)
                        not_paddy         inside no polygon
    Paddy_Scheme        the scheme name of that polygon ('' when none)
    In_Granary_Scheme   Paddy_Class is scheme or unnamed_granary
    In_Own_Scheme       the polygon belongs to the scheme(s) that the pixel's own granary is mapped to in
                        geo/granary_scheme_map.csv. <NA> when the granary is not in that file.
    Poly_ha             area of that polygon (0 when none)
    Dist_to_paddy_m     approximate distance (metres) to the nearest paddy polygon; 0 when inside

Outputs in --out:
    paddy_pixel_tags.csv      one row per pixel (what eda_clean.py --paddy-tags reads)
    paddy_mask_summary.csv    per granary: share of pixels on paddy, in a granary scheme, in its own scheme
    paddy_scheme_area.csv     paddy area (ha) and polygon count per class / scheme
    paddy_cleaning_log.csv    what the polygon cleaning repaired or dropped

IMPORTANT: the scheme labels in the PLANMalaysia layer were found to be incomplete (for example KADA is labelled on only
about 9,400 ha, and 13 Pahang polygons in a granary have no name). Treat In_Own_Scheme as a FLAG to verify with the data owner,
not as the truth. geo/granary_scheme_map.csv is yours to edit.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

METRES_PER_DEG = 111_320.0
UNNAMED = "unnamed"          # token used in granary_scheme_map.csv for 'in a granary but no scheme name'
PRIORITY = {"scheme": 0, "unnamed_granary": 1, "non_granary": 2}


# ----------------------------------------------------------------------------------------------------------------
# polygon layer
# ----------------------------------------------------------------------------------------------------------------
def _polygonal(g):
    """Keep only the polygon parts of a geometry (make_valid can return a GeometryCollection)."""
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import unary_union
    if g is None or g.is_empty:
        return None
    if g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    parts = [p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
    if not parts:
        return None
    u = unary_union(parts)
    return u if isinstance(u, (Polygon, MultiPolygon)) else None


def _approx_ha(g):
    """Area in hectares from degrees (local scale at the polygon's latitude). Only used if the layer has no area column."""
    lat = np.radians(g.centroid.y)
    return g.area * (METRES_PER_DEG * np.cos(lat)) * METRES_PER_DEG / 1e4


def load_paddy(path, min_ha=1.0, area_col="luhek"):
    """Read the GeoJSON, repair and filter it. Returns (polys DataFrame, cleaning-log DataFrame)."""
    from shapely import make_valid
    from shapely.geometry import shape
    with open(path, encoding="utf-8") as fh:
        gj = json.load(fh)
    rows, n_invalid, n_lost = [], 0, 0
    for f in gj["features"]:
        p, geom = f.get("properties") or {}, f.get("geometry")
        if not geom:
            n_lost += 1
            continue
        g = shape(geom)
        if not g.is_valid:
            n_invalid += 1
            g = _polygonal(make_valid(g))
        if g is None:
            n_lost += 1
            continue
        area = p.get(area_col)
        area = float(area) if area is not None and not pd.isna(area) else _approx_ha(g)
        jel = str(p.get("jelapang") or "").strip()
        name = str(p.get("nmskim") or "").strip()
        if jel == "Luar":
            cls = "non_granary"
        elif name:
            cls = "scheme"
        else:
            cls = "unnamed_granary"
        rows.append({"geometry": g, "state": p.get("negeri", ""), "jelapang": jel, "scheme": name, "class": cls, "area_ha": area})
    allp = pd.DataFrame(rows)
    zero = allp.area_ha <= 0
    small = (~zero) & (allp.area_ha < min_ha)
    keep = ~zero & ~small
    log = pd.Series({
        "polygons_in_file": len(gj["features"]),
        "polygons_without_usable_geometry": n_lost,
        "invalid_geometry_repaired": n_invalid,
        "zero_area_dropped": int(zero.sum()),
        f"below_{min_ha:g}_ha_dropped": int(small.sum()),
        "polygons_kept": int(keep.sum()),
        "ha_in_file": round(float(allp.area_ha.sum()), 1),
        "ha_kept": round(float(allp.loc[keep, "area_ha"].sum()), 1),
        "pct_area_kept": round(100 * float(allp.loc[keep, "area_ha"].sum()) / max(float(allp.area_ha.sum()), 1e-9), 2),
    }, name="value").to_frame()
    return allp.loc[keep].reset_index(drop=True), log


def scheme_area(polys):
    t = polys.assign(group=np.where(polys["class"] == "scheme", polys.scheme, polys["class"]))
    return (t.groupby(["class", "group"]).agg(polygons=("area_ha", "size"), area_ha=("area_ha", "sum")).round(1)
             .reset_index().sort_values("area_ha", ascending=False))


def load_scheme_map(path):
    """granary -> set of scheme names (use 'unnamed' for granary polygons that carry no name)."""
    if not path or not os.path.exists(path):
        return {}
    m = pd.read_csv(path)
    out = {}
    for g, s in zip(m["Granary"], m["Scheme"]):
        out.setdefault(str(g).strip(), set()).add(str(s).strip())
    return out


# ----------------------------------------------------------------------------------------------------------------
# pixel tagging
# ----------------------------------------------------------------------------------------------------------------
def tag_pixels(pixels, polys, scheme_map=None):
    """pixels needs Pixel_ID, District (the granary), Longitude, Latitude. Returns one tag row per Pixel_ID."""
    from shapely.geometry import Point
    from shapely.strtree import STRtree
    scheme_map = scheme_map or {}
    px = pixels.drop_duplicates("Pixel_ID")[["Pixel_ID", "District", "Longitude", "Latitude"]].reset_index(drop=True)
    geoms = list(polys.geometry)
    tree = STRtree(geoms) if geoms else None
    cls, scheme, poly_ha, dist = [], [], [], []
    for lon, lat in zip(px.Longitude, px.Latitude):
        pt = Point(lon, lat)
        hit = tree.query(pt, predicate="intersects") if tree is not None else []
        if len(hit):
            # overlapping polygons: a named scheme beats an unnamed granary polygon beats a non-granary one; then the larger polygon
            best = min(hit, key=lambda i: (PRIORITY[polys["class"].iat[i]], -polys.area_ha.iat[i]))
            cls.append(polys["class"].iat[best])
            scheme.append(polys.scheme.iat[best])
            poly_ha.append(float(polys.area_ha.iat[best]))
            dist.append(0.0)
        else:
            cls.append("not_paddy")
            scheme.append("")
            poly_ha.append(0.0)
            dist.append(float(pt.distance(geoms[int(tree.nearest(pt))]) * METRES_PER_DEG) if tree is not None else np.nan)
    t = px.rename(columns={"District": "Granary"}).copy()
    t["Paddy_Class"], t["Paddy_Scheme"], t["Poly_ha"], t["Dist_to_paddy_m"] = cls, scheme, poly_ha, np.round(dist, 1)
    t["In_Paddy"] = t.Paddy_Class != "not_paddy"
    t["In_Granary_Scheme"] = t.Paddy_Class.isin(["scheme", "unnamed_granary"])

    def own(row):
        wanted = scheme_map.get(row.Granary)
        if wanted is None:
            return pd.NA
        if row.Paddy_Class == "unnamed_granary":
            return UNNAMED in wanted
        return row.Paddy_Class == "scheme" and row.Paddy_Scheme in wanted
    t["In_Own_Scheme"] = pd.array([own(r) for r in t.itertuples()], dtype="boolean")
    return t


def summarise(tags):
    g = tags.groupby("Granary")
    s = pd.DataFrame({
        "pixels": g.size(),
        "pct_on_paddy": (g.In_Paddy.mean() * 100).round(1),
        "pct_in_granary_scheme": (g.In_Granary_Scheme.mean() * 100).round(1),
        "pct_in_own_scheme": (g.In_Own_Scheme.apply(lambda x: x.astype("float").mean()) * 100).round(1),
        "pct_non_granary_paddy": (g.Paddy_Class.apply(lambda x: (x == "non_granary").mean()) * 100).round(1),
        "median_dist_m_when_off_paddy": tags.loc[~tags.In_Paddy].groupby("Granary").Dist_to_paddy_m.median().reindex(g.size().index).round(0),
    })
    allrow = pd.DataFrame({"pixels": [len(tags)], "pct_on_paddy": [round(tags.In_Paddy.mean() * 100, 1)],
                           "pct_in_granary_scheme": [round(tags.In_Granary_Scheme.mean() * 100, 1)],
                           "pct_in_own_scheme": [round(tags.In_Own_Scheme.astype("float").mean() * 100, 1)],
                           "pct_non_granary_paddy": [round((tags.Paddy_Class == "non_granary").mean() * 100, 1)],
                           "median_dist_m_when_off_paddy": [round(tags.loc[~tags.In_Paddy, "Dist_to_paddy_m"].median(), 0)]}, index=["ALL"])
    return pd.concat([s, allrow]).rename_axis("Granary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paddy", default="geo/paddy_planmalaysia.geojson", help="PLANMalaysia paddy polygons (GeoJSON)")
    ap.add_argument("--pixels", default="DailyGranary_by_granary.csv", help="CSV with Pixel_ID, District, Longitude, Latitude")
    ap.add_argument("--scheme-map", default="geo/granary_scheme_map.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--min-ha", type=float, default=1.0, help="drop polygon fragments below this area (they hold ~0.2%% of the area)")
    ap.add_argument("--area-col", default="luhek")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("1) cleaning the polygon layer")
    polys, log = load_paddy(a.paddy, a.min_ha, a.area_col)
    log.to_csv(os.path.join(a.out, "paddy_cleaning_log.csv"))
    print(log.to_string())
    scheme_area(polys).to_csv(os.path.join(a.out, "paddy_scheme_area.csv"), index=False)

    print("\n2) tagging pixels")
    cols = pd.read_csv(a.pixels, nrows=0).columns
    need = ["Pixel_ID", "District", "Longitude", "Latitude"]
    use = [c for c in need if c in cols]
    if "Pixel_ID" not in cols:
        use = [c for c in ["District", "Longitude", "Latitude"] if c in cols]
    px = pd.read_csv(a.pixels, usecols=use)
    if "Pixel_ID" not in px.columns:
        px["Pixel_ID"] = px.Longitude.astype(str) + "_" + px.Latitude.astype(str)
    n_px = px.Pixel_ID.nunique()
    tags = tag_pixels(px, polys, load_scheme_map(a.scheme_map))
    assert len(tags) == n_px, "a pixel was lost while tagging"
    tags.to_csv(os.path.join(a.out, "paddy_pixel_tags.csv"), index=False)
    summ = summarise(tags)
    summ.to_csv(os.path.join(a.out, "paddy_mask_summary.csv"))
    print(summ.to_string())
    print(f"\nWrote paddy_pixel_tags.csv ({len(tags)} pixels, none removed), paddy_mask_summary.csv, "
          f"paddy_scheme_area.csv, paddy_cleaning_log.csv in {a.out}/")
    print("Next: python pipeline/eda_clean.py ... --paddy-tags data/paddy_pixel_tags.csv   (adds flags, removes no rows)")
    print("      python pipeline/paddy_robustness.py --clean eda/DailyGranary_clean.csv --tags data/paddy_pixel_tags.csv --out data")


if __name__ == "__main__":
    main()
