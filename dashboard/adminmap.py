"""Administrative shapes for the maps: each granary is drawn as the district(s) it covers, over a state base map.

Reads geo/malaysia_districts.geojson, geo/malaysia_states.geojson and geo/granary_districts.csv (see geo/README.txt).
If those files are missing everything returns None and the maps fall back to the sampled-pixel outlines.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

GEO = Path(__file__).resolve().parent.parent / "geo"
BORNEO = {"Sabah", "Sarawak", "WP Labuan"}                 # no granary there; leaving them out keeps the map light and focused
DISTRICT_EPS, STATE_EPS = 0.002, 0.006                    # simplification tolerance in degrees (~200 m / ~650 m)


def _rdp(pts, eps):
    """Douglas-Peucker line simplification (iterative, numpy)."""
    pts = np.asarray(pts, float)
    if len(pts) < 5:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[[0, -1]] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        seg = q - p
        n = np.hypot(*seg)
        mid = pts[a + 1:b]
        dist = (np.hypot(*(mid - p).T) if n == 0 else np.abs(seg[0] * (mid[:, 1] - p[1]) - seg[1] * (mid[:, 0] - p[0])) / n)
        i = int(dist.argmax())
        if dist[i] > eps:
            keep[a + 1 + i] = True
            stack += [(a, a + 1 + i), (a + 1 + i, b)]
    return pts[keep]


def _area_centroid(ring):
    r = np.asarray(ring, float)
    x, y = r[:, 0], r[:, 1]
    cr = x[:-1] * y[1:] - x[1:] * y[:-1]
    a = cr.sum() / 2
    if abs(a) < 1e-12:
        return abs(a), r.mean(0)
    return abs(a), np.array([((x[:-1] + x[1:]) * cr).sum() / (6 * a), ((y[:-1] + y[1:]) * cr).sum() / (6 * a)])


def _polys(geom, eps):
    """GeoJSON Polygon/MultiPolygon -> list of polygons, each a list of rings [[lon, lat], ...] (holes kept)."""
    parts = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    out = []
    for poly in parts:
        rings = []
        for k, ring in enumerate(poly):
            s = _rdp(ring, eps)
            if len(s) >= 4 and (k == 0 or _area_centroid(s)[0] > 1e-6):
                rings.append(np.round(s, 4).tolist())
        if rings and _area_centroid(rings[0])[0] > 2e-6:         # drop specks smaller than ~0.03 km2
            out.append(rings)
    return out


@st.cache_data(show_spinner=False)
def load():
    """(granary_parts, granary_points, state_base) as DataFrames, or None when the geo files are not there."""
    try:
        dist = json.load(open(GEO / "malaysia_districts.geojson", encoding="utf-8"))
        stat = json.load(open(GEO / "malaysia_states.geojson", encoding="utf-8"))
        mp = pd.read_csv(GEO / "granary_districts.csv")
    except (OSError, ValueError):
        return None
    by_name = {f["properties"]["name"]: f for f in dist["features"]}
    parts, pts = [], []
    for g, sub in mp.groupby("Granary", sort=False):
        found = [r for r in sub.itertuples() if r.District in by_name]
        if not found:
            continue
        allp, wsum, c = [], 0.0, np.zeros(2)
        for r in found:
            for poly in _polys(by_name[r.District]["geometry"], DISTRICT_EPS):
                allp.append(poly)
                a, cc = _area_centroid(poly[0])
                wsum += a
                c += a * cc
        admin = ", ".join(r.District for r in found) + " (" + ", ".join(sorted(set(sub.State))) + ")"
        for poly in allp:
            parts.append(dict(District=g, polygon=poly, Admin=admin))
        pts.append(dict(District=g, Longitude=c[0] / wsum, Latitude=c[1] / wsum, Admin=admin))
    base = []
    for f in stat["features"]:
        nm = f["properties"].get("name")
        if nm in BORNEO:
            continue
        for poly in _polys(f["geometry"], STATE_EPS):
            base.append(dict(State=nm, polygon=poly))
    return pd.DataFrame(parts), pd.DataFrame(pts), pd.DataFrame(base)


def peninsula_view():
    """(lat, lon, zoom) that frames the whole of Peninsular Malaysia."""
    return 4.1, 102.1, 5.6


def bounds_of(granary=None):
    """(lon0, lat0, lon1, lat1) of one granary's shapes, or of all of them; None when there are no admin shapes"""
    g = load()
    if g is None or g[0].empty:
        return None
    p = g[0] if granary is None else g[0][g[0].District == granary]
    if p.empty:
        return None
    xy = np.array([pt for poly in p.polygon for pt in poly[0]])
    return (*xy.min(0), *xy.max(0))
