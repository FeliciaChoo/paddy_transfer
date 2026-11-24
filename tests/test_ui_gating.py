"""Interface-level checks of the clean sidebar navigation, zero-click AutoML scoring, and governance gating.

Run:  python -m pytest tests/test_ui_gating.py -q
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest        # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "transfer_yield_dataset.csv").exists(), reason="needs data/transfer_yield_dataset.csv")

ROLE_VAL, ROLE_KPKM = "Model Validator & Data Steward", "KPKM Programme Manager"


def _app(role=ROLE_VAL, registry=None, step=None, page_idx=0):
    import tempfile
    from acts_transfer import governance
    governance.DEFAULT_PATH = registry if registry is not None else Path(tempfile.mkdtemp()) / "governance_registry.json"
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.session_state["nav_mode"] = role
    at.session_state["kpkm_page_idx"] = page_idx
    if step:
        at.session_state["validator_step"] = step
    return at.run()


def _kpkm_page(at, n):
    return _btn(at, f"nav_kpkm_{n - 1}").click().run()


def _btn(at, key):
    b = [x for x in at.button if x.key == key]
    return b[0] if b else None


def _text(at):
    return " ".join(x.value for blk in (at.markdown, at.warning, at.info, at.success, at.error, at.caption) for x in blk)


def _granary_box(at):
    return [x for x in at.sidebar.selectbox if x.label == "Selected Granary"][0]


def _ws_box(at):
    return [x for x in at.sidebar.selectbox if x.label == "Workspace"][0]


def test_sidebar_workspace_dropdown_and_clean_nav_buttons(tmp_path):
    at = _app("Overview · Summary & Model Status", tmp_path / "reg.json")
    assert not at.exception
    assert len(at.sidebar.radio) == 0
    assert [x.label for x in at.sidebar.selectbox] == ["Selected Granary", "Workspace"]
    # Overview is the top button and is excluded from the Workspace dropdown
    assert _ws_box(at).options == [ROLE_KPKM, ROLE_VAL]
    assert _ws_box(at).value is None
    assert [b.key for b in at.sidebar.button] == ["nav_ov"]

    # Switch workspace via dropdown to KPKM Programme Manager
    _ws_box(at).set_value(ROLE_KPKM).run()
    assert at.session_state["nav_mode"] == ROLE_KPKM
    for i in range(4):
        assert _btn(at, f"nav_kpkm_{i}") is not None
    _btn(at, "nav_kpkm_1").click().run()
    assert at.session_state["kpkm_page_idx"] == 1

    # Switch workspace via dropdown to Model Validator & Data Steward
    _ws_box(at).set_value(ROLE_VAL).run()
    assert at.session_state["nav_mode"] == ROLE_VAL
    for s in range(1, 5):
        assert _btn(at, f"nav_val_{s}") is not None
    _btn(at, "nav_val_2").click().run()
    assert at.session_state["validator_step"] == 2

    # Clicking top Overview button returns to Overview and clears Workspace dropdown selection
    _btn(at, "nav_ov").click().run()
    assert at.session_state["nav_mode"] == "Overview · Summary & Model Status"
    assert _ws_box(at).value is None


def test_every_page_of_both_workspaces_renders(tmp_path):
    for n in range(1, 5):
        at = _app(ROLE_KPKM, tmp_path / "reg.json", page_idx=n - 1)
        assert not at.exception, (n, [e.value for e in at.exception])
    for n in range(1, 5):
        at = _app(ROLE_VAL, tmp_path / "reg.json", step=n)
        assert not at.exception, (n, [e.value for e in at.exception])


def test_automl_runs_automatically_from_step_1_with_stage_scores(tmp_path):
    at = _app(ROLE_VAL, tmp_path / "reg.json", step=1)
    assert not at.exception
    kpi = {m.label: m.value for m in at.metric}
    assert "Step 1 · Data Quality Score" in kpi and kpi["Step 1 · Data Quality Score"].endswith("%")
    assert "Step 2 · Domain Similarity Score" in kpi and kpi["Step 2 · Domain Similarity Score"].endswith("%")
    assert "Step 3 · AutoML Model Winner" in kpi and kpi["Step 3 · AutoML Model Winner"].endswith("%")
    assert "Step 4 · Safeguard Readiness" in kpi and kpi["Step 4 · Safeguard Readiness"].endswith("%")
    assert "automl_summary" in at.session_state


def test_the_kpkm_workspace_has_no_model_signoff_controls(tmp_path):
    for n in range(1, 5):
        at = _app(ROLE_KPKM, tmp_path / "reg.json", page_idx=n - 1)
        assert not at.exception
        for key in ("so_approve", "so_reject"):
            assert _btn(at, key) is None, (n, key)


def test_the_overview_hub_shows_the_governance_status_and_jumps_to_a_workspace(tmp_path):
    at = _app("Overview · Summary & Model Status", tmp_path / "reg.json")
    assert not at.exception
    kpi = {m.label: m.value for m in at.metric}
    assert kpi["Verified Transfer Models"].startswith("0 / ") and "Mean Granary Yield" in kpi
    _btn(at, "jump_validator").click().run()
    assert at.session_state["nav_mode"] == ROLE_VAL and at.session_state["validator_step"] == 1 and not at.exception
    at = _app("Overview · Summary & Model Status", tmp_path / "reg.json")
    _btn(at, "jump_kpkm").click().run()
    assert at.session_state["nav_mode"] == ROLE_KPKM


def test_the_validator_steps_have_next_and_back_buttons(tmp_path):
    at = _app(ROLE_VAL, tmp_path / "reg.json", step=1)
    path = [("nx_1", 2), ("nx_2", 3), ("nx_3", 4), ("bk_4", 3), ("bk_3", 2), ("bk_2", 1)]
    for key, want in path:
        _btn(at, key).click().run()
        assert at.session_state["validator_step"] == want and not at.exception, (key, want)
    at = _app(ROLE_VAL, tmp_path / "reg.json", step=4)
    _btn(at, "nx_4").click().run()
    assert at.session_state["nav_mode"] == "Overview · Summary & Model Status"


def test_clicking_the_sidebar_granary_is_the_one_selected_granary_for_every_page(tmp_path):
    at = _app(ROLE_KPKM, tmp_path / "reg.json")
    _granary_box(at).set_value("IADA Kerian").run()
    assert at.session_state["granary"] == "IADA Kerian"
    at = _kpkm_page(at, 2)
    assert at.session_state["granary"] == "IADA Kerian" and not at.exception


def test_a_signed_off_granary_reaches_the_kpkm_brief_and_an_unsigned_one_does_not(tmp_path):
    reg = tmp_path / "reg.json"
    at = _app(ROLE_VAL, reg, step=4)
    at.text_input[[t.key for t in at.text_input].index("so_name")].set_value("Test Validator").run()
    _btn(at, "so_reject").click().run()
    assert not at.exception
    from acts_transfer import governance
    e = governance.load(reg)["granaries"]["IADA Barat Laut Selangor"]
    assert e["status"] == governance.REJECTED and e["validator"] == "Test Validator" and e["published"]["baseline_model"]
    at = _app(ROLE_KPKM, reg, page_idx=3)
    _granary_box(at).set_value("IADA Barat Laut Selangor").run()
    _btn(at, "get_advisory").click().run()
    t = _text(at)
    assert "validator-reviewed yield forecast" in t and "Local Baseline Active" in t
    _granary_box(at).set_value("IADA Kerian").run()
    _btn(at, "get_advisory").click().run()
    assert "no validator-reviewed yield forecast for this granary" in _text(at)


def test_the_decision_brief_shows_the_cited_reference_practice(tmp_path):
    at = _app(ROLE_KPKM, tmp_path / "reg.json", page_idx=3)
    _granary_box(at).set_value("IADA Barat Laut Selangor").run()
    _btn(at, "get_advisory").click().run()
    t = _text(at)
    assert not at.exception and "Source Resource" in t and "has not been checked by this project" in t and "Kaedah Pengurusan Tanaman Padi" in t


def test_the_decision_brief_renders_counterfactual_cards_when_present(tmp_path, monkeypatch):
    res_dir = tmp_path / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    reg_file = res_dir / "governance_registry.json"
    reg_file.write_text(json.dumps({
        "IADA Barat Laut Selangor": {
            "updated_at": "2026-11-24T12:00:00",
            "approved_method": "HGB + RULE",
            "delta_nt": -142.5,
            "prescriptions": [{
                "lever": "Planting Schedule Synchronization",
                "window": "Decades 6–10",
                "delta_y_kg_ha": 240.0,
                "ci_low": 180.0,
                "ci_high": 300.0,
                "rule_id": "R01_PLANT_SLIP",
                "schedule_bound": "JPS/IADA gazetted water-release window",
            }],
        }
    }), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    at = _app(ROLE_KPKM, tmp_path / "reg.json", page_idx=3)
    _granary_box(at).set_value("IADA Barat Laut Selangor").run()
    _btn(at, "get_advisory").click().run()
    t = _text(at)
    assert not at.exception
    assert "Causal counterfactual prescriptions" in t and "HGB + RULE" in t


def test_the_sidebar_always_keeps_the_selected_granary_and_a_role_card(tmp_path):
    for mode in ("Overview · Summary & Model Status", ROLE_KPKM, ROLE_VAL):
        at = _app(mode, tmp_path / "reg.json")
        assert [x.label for x in at.sidebar.selectbox] == ["Selected Granary", "Workspace"] and not at.exception
        assert any("sb-role-box" in m.value for m in at.sidebar.markdown)


def test_the_map_page_shows_a_field_summary_card_for_the_selected_granary(tmp_path):
    at = _app(ROLE_KPKM, tmp_path / "reg.json", page_idx=0)
    _granary_box(at).set_value("IADA Kerian").run()
    assert not at.exception
    t = _text(at)
    assert "Review a paddy field · IADA Kerian" in t
    assert "Mapped paddy field groups" in [m.label for m in at.metric]
    assert "Pending review" not in t and any("Season 1" in m.value for m in at.markdown)