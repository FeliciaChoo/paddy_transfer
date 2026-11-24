"""Tests for the refactored PaddyTwinAI Leaflet map (dashboard/tab_map.py).

Run: python -m pytest tests/test_map.py -q
"""
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


def test_granary_metadata_covers_peninsular_malaysia_with_two_planting_windows():
    for dist, meta in tab_map.GRANARY_META.items():
        lat, lon = meta["coords"]
        assert 1.0 <= lat <= 7.0 and 99.0 <= lon <= 105.0, dist
        assert meta["state"] and meta["s1"] and meta["s2"]
    assert tab_map.GRANARY_META["IADA Barat Laut Selangor"]["coords"] == [3.55, 101.10]
    assert tab_map.GRANARY_META["KETARA"]["state"] == "Terengganu"


def test_derive_condition_prioritizes_optical_visibility_then_crop_status():
    assert tab_map._derive_condition({"Optical_Valid_Fraction": 0.30, "Recent_status": "behind"}) == "Limited satellite visibility"
    assert tab_map._derive_condition({"Optical_Valid_Fraction": 0.80, "Recent_status": "behind"}) == "Inspect declining vegetation"
    assert tab_map._derive_condition({"Optical_Valid_Fraction": 0.80, "Recent_status": "ahead"}) == "Maintain field surveillance"
    assert tab_map._derive_condition({"Optical_Valid_Fraction": 0.80, "Recent_status": "on track"}) == "Routine monitoring"
    assert tab_map._derive_condition({"Optical_Valid_Fraction": np.nan, "Yield_z": -1.2}) == "Inspect declining vegetation"


def test_yield_and_ndvi_colour_bands_handle_missing_values():
    assert tab_map._yield_colour(float("nan")) == tab_map.NO_DATA_COLOUR
    assert tab_map._yield_colour(2.0) == tab_map.YIELD_COLOURS[0]
    assert tab_map._yield_colour(5.0) == tab_map.YIELD_COLOURS[-1]
    assert len(tab_map.FIXED_YIELD_EDGES) == 4 and tab_map.FIXED_YIELD_EDGES == sorted(tab_map.FIXED_YIELD_EDGES)

    assert tab_map._ndvi_colour(float("nan")) == tab_map.NO_DATA_COLOUR
    assert tab_map._ndvi_colour(0.25) == "#EAB308"
    assert tab_map._ndvi_colour(0.45) == "#86EFAC"
    assert tab_map._ndvi_colour(0.60) == "#15803D"
    assert tab_map._ndvi_colour(0.75) == "#0E7490"


def test_tooltip_escapes_html_and_formats_yield_and_windows():
    row = {
        "District": "A<b>",
        "State": "Kedah & Perlis",
        "Condition": "Routine monitoring",
        "Yield_t_ha": 4.2,
        "Yield_kg_ha": 4200.0,
        "Optical_Valid_Fraction": 0.5,
        "Last_NDVI": 0.62,
        "Season_1_Window": "Mar – Jul",
        "Season_2_Window": "Sep – Jan",
    }
    h = tab_map._build_tooltip(row, 2024)
    assert "4.20 t/ha (4,200 kg/ha)" in h
    assert "50.0%" in h and "0.62" in h
    assert "S1 Mar – Jul | S2 Sep – Jan" in h
    assert "Kedah &amp; Perlis" in h and "A&lt;b&gt;" in h

    row["Yield_t_ha"] = float("nan")
    assert "No reference" in tab_map._build_tooltip(row, 2024)


@needs_data
def test_reference_practice_file_has_required_columns():
    r = pd.read_csv(DATA / "advisory_reference_practice.csv")
    assert len(r) >= 3 and list(r.columns) == ["Who", "Topic", "Window", "What", "Source"] and r.notna().all().all()


def test_map_table_falls_back_to_yield_table_and_derives_optical_fraction(tmp_path):
    y = tmp_path / "y.csv"
    pd.DataFrame({
        "District": ["A", "B"],
        "Year": [2024, 2024],
        "Yield_kg_ha": [4000.0, 3500.0],
        "Optical_observed_pct": [50.0, 80.0],
    }).to_csv(y, index=False)
    d = tab_map._map_table(str(tmp_path / "missing_stage.csv"), str(y), 0.0)
    assert list(d.columns) == ["District", "Year", "Yield_kg_ha", "Optical_Valid_Fraction"]
    assert d.Optical_Valid_Fraction.tolist() == [0.5, 0.8]


@needs_data
def test_map_table_prefers_growth_stage_table():
    d = tab_map._map_table(str(DATA / "transfer_stage_dataset.csv"), str(DATA / "transfer_yield_dataset.csv"), 0.0)
    assert len(d) == d.District.nunique() * d.Year.nunique() and len(d) >= 70
    assert d.Optical_Valid_Fraction.between(0, 1).all()

    def test_the_map_page_shows_a_field_summary_card_for_the_selected_granary(tmp_path):
        at = _app(ROLE_KPKM, tmp_path / "reg.json")
        _granary_box(at).set_value("IADA Kerian").run()
        assert not at.exception
        t = _text(at)
        assert "Review a paddy field · IADA Kerian" in t
        assert "Mapped paddy field groups" in [m.label for m in at.metric]
        assert "Pending review" not in t and any("Season 1" in m.value for m in at.markdown)