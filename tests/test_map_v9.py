"""PACE V9 Leaflet map: the granary GeoJSON, the colour classes and the data table behind the map. Run: python -m pytest tests/test_map_v9.py -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("streamlit")
from dashboard import tab_map            # noqa: E402

DATA = ROOT / "data"
needs_data = pytest.mark.skipif(not (DATA / "transfer_yield_dataset.csv").exists(), reason="needs data/transfer_yield_dataset.csv")


class _D:
    DATA_DIR = DATA
    districts = pd.read_csv(DATA / "districts.csv") if (DATA / "districts.csv").exists() else None


@needs_data
def test_every_granary_has_a_marker_position_a_state_and_two_planting_windows():
    p = tab_map.granary_points(_D)
    assert set(p.District) == set(pd.read_csv(DATA / "transfer_yield_dataset.csv").District)             # spelled as in the data (KETARA, not IADA KETARA)
    assert p.Latitude.between(1, 7).all() and p.Longitude.between(99, 105).all()                          # Peninsular Malaysia
    assert (p[["State", "Season_1_Window", "Season_2_Window"]] != "not recorded").all().all()


@needs_data
def test_a_click_is_matched_to_the_nearest_marker_within_tolerance_and_ignored_elsewhere():
    p = tab_map.granary_points(_D)
    r = p[p.District == "IADA Barat Laut Selangor"].iloc[0]
    assert tab_map.nearest_granary(p, r.Latitude + 0.01, r.Longitude - 0.01) == "IADA Barat Laut Selangor"
    assert tab_map.nearest_granary(p, 0.0, 110.0) is None


def test_the_tooltip_carries_the_v10_fields_and_escapes_text():
    row = {"District": "A<b>", "State": "Kedah & Perlis", "Year": 2024, "Yield_t_ha": 4.2, "Yield_kg_ha": 4200.0, "Optical_Valid_Fraction": 0.5,
           "Season_1_Window": "Mar – Jul", "Season_2_Window": "Sep – Jan"}
    h = tab_map.tooltip_html(row)
    assert "4.20 t/ha (4,200 kg/ha)" in h and "50.0%" in h and "Season 1:</b> Mar – Jul" in h and "Season 2:</b> Sep – Jan" in h and "Kedah &amp; Perlis" in h and "A&lt;b&gt;" in h
    row["Yield_t_ha"] = float("nan")
    assert "No reference" in tab_map.tooltip_html(row)


@needs_data
def test_marker_positions_come_from_the_metadata_file_and_fall_back_to_the_data_centre():
    p = tab_map.granary_points(_D).set_index("District")
    assert (p.loc["IADA Barat Laut Selangor", "Latitude"], p.loc["IADA Barat Laut Selangor", "Longitude"]) == (3.55, 101.10)        # the specified position, not the pixel centre
    assert p.loc["KETARA", "State"] == "Terengganu" and "3 staggered phases" in p.loc["MADA", "Season_2_Window"] and "4 irrigation blocks" in p.loc["IADA Barat Laut Selangor", "Season_1_Window"]


def test_the_granary_is_read_from_the_clicked_tooltip_text():
    names = ["MADA", "IADA Kerian", "IADA Barat Laut Selangor", "KADA"]
    assert tab_map.district_from_tooltip("IADA Barat Laut Selangor (Selangor)\n2024 Yield: 4.19 t/ha", names) == "IADA Barat Laut Selangor"
    assert tab_map.district_from_tooltip("  KADA   (Kelantan) 2024 Yield", names) == "KADA"
    assert tab_map.district_from_tooltip("something else", names) is None and tab_map.district_from_tooltip(None, names) is None


@needs_data
def test_the_leaflet_map_has_one_marker_per_granary_one_visible_base_layer_and_sticky_tooltips():
    import folium
    pts = tab_map.granary_points(_D)
    rows = pts.assign(Year=2024, Yield_kg_ha=4200.0, Yield_t_ha=4.2, Optical_Valid_Fraction=0.5)
    m = tab_map.build_folium_map(rows, "MADA", "Yield", tab_map.FIXED_YIELD_EDGES, 1.87, 4.81, 2024)
    kids = list(m._children.values())
    tiles = [k for k in kids if isinstance(k, folium.TileLayer)]
    assert [t.layer_name for t in tiles] == ["Clean Light Map (Default)", "OpenStreetMap Street View", "Satellite Imagery (Esri)"]
    assert [t.show for t in tiles] == [True, False, False]                                           # otherwise the satellite layer would cover the default map
    markers = [k for k in kids if isinstance(k, folium.CircleMarker)]
    assert len(markers) == len(pts) + 1                                                              # one per granary, plus the ring round the selected one
    tips = [t for k in markers for t in k._children.values() if isinstance(t, folium.Tooltip)]
    assert len(tips) == len(pts) and all(t.options.get("sticky") for t in tips)
    assert any(isinstance(k, folium.LayerControl) for k in kids)


def test_the_fixed_v10_legend_is_five_equal_bands_from_1_87_to_4_81():
    e = tab_map.FIXED_YIELD_EDGES
    assert np.allclose(np.diff([tab_map.FIXED_RANGE[0]] + e + [tab_map.FIXED_RANGE[1]]), (4.81 - 1.87) / 5, atol=0.01)


@needs_data
def test_the_reference_practice_file_has_who_what_window_and_a_source_for_every_row():
    r = pd.read_csv(DATA / "advisory_reference_practice.csv")
    assert len(r) >= 3 and list(r.columns) == ["Who", "Topic", "Window", "What", "Source"] and r.notna().all().all()


def test_yield_classes_are_five_equal_bands_over_the_data_range_and_missing_is_grey():
    t = pd.Series([2.37, 3.1, 4.0, 5.65])
    edges, lo, hi = tab_map._yield_edges(t)
    assert len(edges) == 4 and edges == sorted(edges) and lo <= 2.37 and hi >= 5.65
    assert np.allclose(np.diff([lo] + edges + [hi]), (hi - lo) / 5)
    cols = [tab_map._class_colour(v, edges, tab_map.YIELD_COLOURS) for v in (lo, edges[0] - 1e-9, edges[0], edges[1], edges[2], edges[3], hi)]
    assert cols[0] == cols[1] == tab_map.YIELD_COLOURS[0] and cols[2] == tab_map.YIELD_COLOURS[1] and cols[-1] == tab_map.YIELD_COLOURS[-1]
    for bad in ("No data", None, float("nan")):
        assert tab_map._class_colour(bad, edges, tab_map.YIELD_COLOURS) == tab_map.NO_DATA_COLOUR


def test_the_map_table_falls_back_to_the_yield_table_and_derives_the_optical_fraction(tmp_path):
    y = tmp_path / "y.csv"
    pd.DataFrame({"District": ["A", "B"], "Year": [2024, 2024], "Yield_kg_ha": [4000.0, 3500.0], "Optical_observed_pct": [50.0, 80.0]}).to_csv(y, index=False)
    d = tab_map._map_table(str(tmp_path / "missing_stage.csv"), str(y), 0.0)
    assert list(d.columns) == ["District", "Year", "Yield_kg_ha", "Optical_Valid_Fraction"] and d.Optical_Valid_Fraction.tolist() == [0.5, 0.8]


@needs_data
def test_the_map_table_prefers_the_growth_stage_table():
    d = tab_map._map_table(str(DATA / "transfer_stage_dataset.csv"), str(DATA / "transfer_yield_dataset.csv"), 0.0)
    assert len(d) == 70 and d.Optical_Valid_Fraction.between(0, 1).all()
