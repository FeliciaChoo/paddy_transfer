"""
build_granary_geojson.py -- one GeoJSON feature per granary for the Leaflet map (dashboard/tab_map.render_leaflet_map).

    python pipeline/build_granary_geojson.py            # writes data/malaysia_granaries.geojson

Each granary is the administrative district(s) it covers (geo/granary_districts.csv + geo/malaysia_districts.geojson, simplified), as one MultiPolygon with properties
{"District": <granary name, as in the data>, "Admin": <the districts and states it covers>}. A granary with no district mapping is drawn from its sampled-pixel outline
(data/granary_footprints.csv). The dashboard rebuilds the same file in memory when it is missing, so this script only saves that work.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build(root=None):
    import ast

    import pandas as pd
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    feats, seen = [], set()
    try:
        from dashboard import adminmap
        got = adminmap.load()
    except Exception:
        got = None
    if got is not None:
        parts = got[0]
        for g, sub in parts.groupby("District", sort=False):
            polys = [[[[float(x), float(y)] for x, y in ring] for ring in p] for p in sub.polygon]
            feats.append({"type": "Feature", "properties": {"District": g, "Admin": sub.Admin.iloc[0]}, "geometry": {"type": "MultiPolygon", "coordinates": polys}})
            seen.add(g)
    fp = os.path.join(root, "data", "granary_footprints.csv")
    if os.path.exists(fp):
        for r in pd.read_csv(fp).itertuples():
            if r.District in seen:
                continue
            ring = [[float(x), float(y)] for x, y in ast.literal_eval(r.polygon)]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
            feats.append({"type": "Feature", "properties": {"District": r.District, "Admin": "sampled-pixel outline"}, "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return {"type": "FeatureCollection", "features": feats}


if __name__ == "__main__":
    gj = build()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "malaysia_granaries.geojson")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(gj, f, separators=(",", ":"))
    print(f"{out}: {len(gj['features'])} granaries, {os.path.getsize(out) / 1e3:.0f} kB")
