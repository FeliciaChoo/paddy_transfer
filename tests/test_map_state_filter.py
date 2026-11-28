"""The map follows the selected granary: its state is chosen automatically (also from 'All States' and when the page first opens) and the map zooms to it."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from dashboard import tab_map  # noqa: E402

needs_data = pytest.mark.skipif(not (ROOT / "data" / "transfer_stage_dataset.csv").exists(), reason="project data not present")

POINTS = pd.DataFrame({"District": ["A", "B", "C"], "Latitude": [3.5, 6.0, np.nan], "Longitude": [101.1, 102.3, 103.0]})


def test_the_map_zooms_to_the_fields_of_the_chosen_state_and_keeps_the_default_view_for_all_states():
    assert tab_map._view_points(POINTS, "All States", False) is None                          # nothing chosen: the default view
    assert tab_map._view_points(POINTS.iloc[:1], "Selangor", False) == [[3.5, 101.1]]          # a state: zoom to its fields
    assert tab_map._view_points(POINTS, "Kelantan", False) == [[3.5, 101.1], [6.0, 102.3]]     # a field without a position is skipped
    assert tab_map._view_points(POINTS, "All States", True) == [[3.5, 101.1], [6.0, 102.3]]    # after 'Show all mapped fields': fit them all
    assert tab_map._view_points(POINTS.iloc[0:0], "Selangor", False) is None                   # nothing to show
    assert tab_map._view_points(POINTS.iloc[2:], "Selangor", False) is None                    # only fields without a position


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)


def _select(at, granary):
    """change the selected granary the way the sync sees it. (The sidebar box has no key, so its identity changes with its index and AppTest drops every second simulated click;
    in a browser each click is delivered.)"""
    at.session_state["granary"] = granary
    return at.run()


@needs_data
def test_the_map_opens_on_the_state_of_the_selected_granary_and_follows_a_change_of_granary():
    at = _app().run()
    assert not at.exception
    first = at.session_state["granary"]
    assert at.session_state["map_state_filter"] == tab_map.GRANARY_META[first]["state"] != "All States"      # opened on that granary's state, not on All States
    for g, state in (("KADA", "Kelantan"), ("MADA", "Kedah & Perlis"), ("IADA Rompin", "Pahang"), ("IADA Kerian", "Perak")):
        _select(at, g)
        assert not at.exception
        assert at.session_state["map_state_filter"] == state == tab_map.GRANARY_META[g]["state"], g      # the app's table and the map's table agree


@needs_data
def test_a_user_who_widens_the_map_is_not_pulled_back_until_the_granary_changes():
    at = _app().run()
    at.session_state["map_state_filter"] = "All States"                                        # 'Show all mapped fields' / the State box
    at.run()
    assert at.session_state["map_state_filter"] == "All States"                                # same granary: left alone
    _select(at, "KADA")
    assert at.session_state["map_state_filter"] == "Kelantan"                                  # a new granary: the relevant state again


@needs_data
def test_the_map_page_shows_only_the_relevant_state_for_the_selected_granary():
    at = _app()
    at.session_state["nav_mode"] = "KPKM Programme Manager"
    at.session_state["kpkm_page_idx"] = 0
    at.run()
    assert not at.exception
    state_box = next(b for b in at.selectbox if b.label == "State")
    assert state_box.value == "Selangor" and "All States" in state_box.options
    _select(at, "KADA")
    assert not at.exception
    assert next(b for b in at.selectbox if b.label == "State").value == "Kelantan"
