"""The paddy mask (PLANMalaysia polygons): cleaning, pixel tagging, the optional eda_clean flags and the verdict gate.

Run:  python -m pytest tests/test_paddy_mask.py -q
All data here is synthetic (tiny squares), so the tests run in about a second and need no real files.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import eda_clean                                      # noqa: E402
import paddy_mask as pm                               # noqa: E402
import paddy_robustness as pr                         # noqa: E402

DEG = 0.01   # ~1.1 km. A 0.01 x 0.01 degree square is ~120 ha, 0.001 x 0.001 is ~1.2 ha.


def square(x, y, s):
    return [[[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]]]


def feat(coords, ha, jel="Dalam", name="", state="Kedah"):
    return {"type": "Feature", "properties": {"negeri": state, "jelapang": jel, "nmskim": name, "luhek": ha},
            "geometry": {"type": "Polygon", "coordinates": coords}}


@pytest.fixture
def layer(tmp_path):
    bowtie = [[[103.0, 5.0], [103.01, 5.01], [103.01, 5.0], [103.0, 5.01], [103.0, 5.0]]]            # self-intersecting = invalid
    feats = [
        feat(square(100.0, 6.0, DEG), 120, "Dalam", "MADA"),                                          # A  named scheme
        feat(square(101.0, 5.0, DEG), 120, "Luar", ""),                                               # B  non-granary
        feat(square(102.0, 4.0, DEG), 120, "Dalam", ""),                                              # C  granary, no name
        feat(bowtie, 60, "Dalam", "KADA"),                                                            # D  invalid, repairable
        feat(square(104.0, 3.0, 0.001), 0.2, "Luar", ""),                                             # E  fragment below 1 ha
        feat(square(105.0, 3.0, 0.001), 0, "Luar", ""),                                               # F  zero area
        feat(square(100.0, 6.0, DEG), 120, "Luar", ""),                                               # G  overlaps A completely
    ]
    p = tmp_path / "paddy.geojson"
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return p


def test_cleaning_repairs_drops_and_logs(layer):
    polys, log = pm.load_paddy(layer, min_ha=1.0)
    v = log["value"]
    assert v["polygons_in_file"] == 7 and v["invalid_geometry_repaired"] == 1
    assert v["zero_area_dropped"] == 1 and v["below_1_ha_dropped"] == 1 and v["polygons_kept"] == 5
    assert all(g.is_valid for g in polys.geometry), "an invalid geometry survived cleaning"
    assert (polys.area_ha >= 1).all()
    assert set(polys["class"]) == {"scheme", "non_granary", "unnamed_granary"}


def test_pixels_are_tagged_and_none_are_lost(layer):
    polys, _ = pm.load_paddy(layer)
    px = pd.DataFrame({"Pixel_ID": ["a", "b", "c", "d", "far"], "District": ["MADA", "MADA", "X", "X", "MADA"],
                       "Longitude": [100.005, 101.005, 102.005, 100.5, 99.0], "Latitude": [6.005, 5.005, 4.005, 6.5, 6.0]})
    t = pm.tag_pixels(px, polys, {"MADA": {"MADA"}})
    assert len(t) == len(px) and t.Pixel_ID.is_unique
    c = t.set_index("Pixel_ID")
    assert c.loc["a", "Paddy_Class"] == "scheme" and c.loc["a", "Paddy_Scheme"] == "MADA"      # named beats the overlapping Luar polygon
    assert c.loc["b", "Paddy_Class"] == "non_granary"
    assert c.loc["c", "Paddy_Class"] == "unnamed_granary"
    assert c.loc["far", "Paddy_Class"] == "not_paddy" and c.loc["far", "Dist_to_paddy_m"] > 0
    assert c.loc["a", "Dist_to_paddy_m"] == 0 and c.loc["a", "In_Paddy"]
    assert bool(c.loc["a", "In_Granary_Scheme"]) and not bool(c.loc["b", "In_Granary_Scheme"])


def test_own_scheme_mapping_including_unnamed_and_unknown(layer):
    polys, _ = pm.load_paddy(layer)
    px = pd.DataFrame({"Pixel_ID": ["m", "m_out", "u", "unk"], "District": ["MADA", "MADA", "PEK", "NOTMAPPED"],
                       "Longitude": [100.005, 101.005, 102.005, 100.005], "Latitude": [6.005, 5.005, 4.005, 6.005]})
    t = pm.tag_pixels(px, polys, {"MADA": {"MADA"}, "PEK": {pm.UNNAMED}}).set_index("Pixel_ID")
    assert bool(t.loc["m", "In_Own_Scheme"]) is True
    assert bool(t.loc["m_out", "In_Own_Scheme"]) is False                 # on paddy, but not in MADA's scheme
    assert bool(t.loc["u", "In_Own_Scheme"]) is True                      # 'unnamed' token matches an unnamed granary polygon
    assert pd.isna(t.loc["unk", "In_Own_Scheme"])                         # granary missing from the map is UNKNOWN, not False


def test_summary_has_one_row_per_granary_plus_all(layer):
    polys, _ = pm.load_paddy(layer)
    px = pd.DataFrame({"Pixel_ID": list("abcd"), "District": ["G1", "G1", "G2", "G2"],
                       "Longitude": [100.005, 99.0, 101.005, 102.005], "Latitude": [6.005, 6.0, 5.005, 4.005]})
    s = pm.summarise(pm.tag_pixels(px, polys, {"G1": {"MADA"}, "G2": {pm.UNNAMED}}))
    assert list(s.index) == ["G1", "G2", "ALL"] and s.loc["ALL", "pixels"] == 4
    assert s.loc["G1", "pct_on_paddy"] == 50.0


def test_eda_flags_are_additive_and_never_drop_rows(tmp_path):
    d = pd.DataFrame({"Pixel_ID": ["a"] * 3 + ["b"] * 3 + ["new"] * 2, "Year": 2021, "x": range(8)})
    tags = pd.DataFrame({"Pixel_ID": ["a", "b"], "Paddy_Class": ["scheme", "not_paddy"], "In_Own_Scheme": [True, pd.NA]})
    p = tmp_path / "tags.csv"
    tags.to_csv(p, index=False)
    before = d.copy()
    out = eda_clean.add_paddy_flags(d.copy(), p)
    assert len(out) == len(before) and out.x.tolist() == before.x.tolist()
    assert out.loc[out.Pixel_ID == "a", "flag_not_paddy"].eq(False).all()
    assert out.loc[out.Pixel_ID == "b", "flag_not_paddy"].all()
    assert out.loc[out.Pixel_ID == "new", "Paddy_Class"].eq("unknown").all()            # missing from the tag file: kept, not dropped
    assert not out.loc[out.Pixel_ID.isin(["b", "new"]), "flag_outside_own_scheme"].any()   # unknown mapping is never flagged
    bad = tags.assign(Pixel_ID=["a", "a"])
    bad.to_csv(p, index=False)
    with pytest.raises(AssertionError, match="duplicate"):
        eda_clean.add_paddy_flags(d.copy(), p)


def test_default_eda_flags_do_not_change():
    """Without --paddy-tags the existing flag set is exactly what it was before this feature."""
    src = (ROOT / "pipeline" / "eda_clean.py").read_text()
    assert "if a.paddy_tags:" in src and "flag_not_paddy" not in src.split("def add_flags")[1].split("def add_paddy_flags")[0]


def _df(ci_low, ci_high, diff, name="on_paddy"):
    return pd.DataFrame([{"pixel_set": "all_pixels", "diff_model_minus_baseline": 1.0, "ci_low": 0.0, "ci_high": 2.0},
                         {"pixel_set": name, "diff_model_minus_baseline": diff, "ci_low": ci_low, "ci_high": ci_high}])


def test_verdict_gate_needs_an_interval_entirely_below_zero():
    quiet = _df(-0.7, 2.6, 1.0)
    assert pr.verdict(quiet, quiet)[0] == "QUALITY_CONTROL_ONLY"
    assert pr.verdict(_df(-3.0, -0.5, -1.7), quiet)[0] == "ACCURACY_GAIN"          # clear win in test 2
    assert pr.verdict(quiet, _df(-3.0, -0.5, -1.7))[0] == "ACCURACY_GAIN"          # clear win in test 3
    lucky = _df(-1.5, 1.2, -0.2)                                                     # better on average, but the interval spans zero
    assert pr.verdict(lucky, lucky)[0] == "QUALITY_CONTROL_ONLY"
    all_only = pd.DataFrame([{"pixel_set": "all_pixels", "diff_model_minus_baseline": -2, "ci_low": -3, "ci_high": -1}])
    assert pr.verdict(all_only, all_only)[0] == "QUALITY_CONTROL_ONLY"             # the unmasked set cannot earn the mask a gain


def test_robustness_tests_run_and_are_deterministic():
    rng = np.random.default_rng(0)
    rows = [{"District": g, "Year": y, "Yield": 4000 + 300 * i + rng.normal(0, 150), "npx": 10,
             **{k: rng.normal() for k in pr.FEATS}} for i, g in enumerate("ABCDEF") for y in range(2018, 2025)]
    a = pd.DataFrame(rows)
    assert pr.growth_test(a) == pr.growth_test(a) and pr.cold_start_test(a) == pr.cold_start_test(a)
    g, c = pr.growth_test(a), pr.cold_start_test(a)
    assert g["n"] == 36 and c["n"] == 42 and c["granaries"] == 6
    assert g["ci_low"] <= g["diff_model_minus_baseline"] <= g["ci_high"]
    t = pr.correlation_table({"x": a})
    assert set(t.pixel_set) == {"x"} and len(t) == len(pr.GROWTH_FEATS)


def test_command_line_end_to_end(layer, tmp_path):
    px = pd.DataFrame({"Pixel_ID": ["a", "b"], "District": ["MADA", "MADA"], "Longitude": [100.005, 99.0], "Latitude": [6.005, 6.0], "Year": 2021})
    pxp = tmp_path / "px.csv"
    px.to_csv(pxp, index=False)
    smap = tmp_path / "map.csv"
    pd.DataFrame({"Granary": ["MADA"], "Scheme": ["MADA"]}).to_csv(smap, index=False)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, str(ROOT / "pipeline" / "paddy_mask.py"), "--paddy", str(layer), "--pixels", str(pxp),
                        "--scheme-map", str(smap), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    t = pd.read_csv(out / "paddy_pixel_tags.csv")
    assert len(t) == 2 and t.set_index("Pixel_ID").loc["a", "Paddy_Class"] == "scheme"
    for f in ["paddy_mask_summary.csv", "paddy_scheme_area.csv", "paddy_cleaning_log.csv"]:
        assert (out / f).exists()
