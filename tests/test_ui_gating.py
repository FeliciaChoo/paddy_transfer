"""Interface-level checks of the tab order, the locks between tabs, and recalculation, through the real app.

Run:  python -m pytest tests/test_ui_gating.py -q      (each test runs the real assessment, so it takes a few seconds per step)
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest        # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "transfer_yield_dataset.csv").exists(), reason="needs data/transfer_yield_dataset.csv")

ROLE_VAL, ROLE_KPKM = "Model Validator & Data Steward", "KPKM Programme Manager"      # also the values of the sidebar workspace radio (nav_mode)
KPKM_PAGES = ["Granary Field Condition Map", "Historical Replay & Yield Fit", "In-Season Nowcast & 2025–2028 Outlook", "Programme Manager Decision Brief"]
VALIDATOR_STEPS = ["Step 1 · Spatial & Optical Map", "Step 2 · Domain Divergence Audit", "Step 3 · Transfer Testing & Audit", "Step 4 · Validator Sign-Off Gate"]
STEP_OF = {"tr_confirm": 2, "tr_assess": 2, "tr_harm": 2, "tr_suit": 2, "tr_strategy": 3, "tr_run": 3}      # the validator step each button lives on


def _app(role=ROLE_VAL, registry=None, step=None):
    """the real app in a workspace (the validator's by default: that is where the transfer steps live). `registry`: a path so a test never touches the real governance registry"""
    import tempfile
    from acts_transfer import governance
    governance.DEFAULT_PATH = registry if registry is not None else Path(tempfile.mkdtemp()) / "governance_registry.json"      # never the real registry
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=600)
    at.session_state["nav_mode"] = role
    if step:
        at.session_state["validator_step"] = step
    return at.run()


def _goto(at, step):
    """open a validator step (the sidebar radio, as a click would)"""
    at.session_state["validator_step"] = step
    return at.run()


def _kpkm_page(at, n):
    r = [x for x in at.sidebar.radio if x.label == "Pages"][0]
    return r.set_value(r.options[n - 1]).run()


def _btn(at, key):
    b = [x for x in at.button if x.key == key]
    return b[0] if b else None


def _has(at, key, kind):
    return any(getattr(x, "key", None) == key for x in getattr(at, kind))


def _text(at):
    return " ".join(x.value for blk in (at.markdown, at.warning, at.info, at.success, at.error, at.caption) for x in blk)


def _through(at, last):
    seq = ("tr_confirm", "tr_assess", "tr_harm", "tr_suit", "tr_strategy", "tr_run")
    for k in seq[: seq.index(last) + 1]:
        if at.session_state["validator_step"] != STEP_OF[k]:
            _goto(at, STEP_OF[k])
        _btn(at, k).click().run()


def _radio(at, label):
    r = [x for x in at.sidebar.radio if x.label == label]
    return r[0] if r else None


def test_each_workspace_has_its_own_pages(tmp_path):
    at = _app("Overview", tmp_path / "reg.json")
    assert not at.exception and [x.label for x in at.sidebar.radio] == ["Workspace"]
    assert at.sidebar.radio[0].options == ["Overview", ROLE_KPKM, ROLE_VAL]
    at = _app(ROLE_KPKM, tmp_path / "reg.json")
    assert not at.exception and _radio(at, "Pages").options == KPKM_PAGES and _radio(at, "Steps") is None
    at = _app(ROLE_VAL, tmp_path / "reg.json")
    assert not at.exception and _radio(at, "Steps").options == VALIDATOR_STEPS and _radio(at, "Pages") is None


def test_every_page_of_both_workspaces_renders(tmp_path):
    for n in range(1, 5):
        at = _kpkm_page(_app(ROLE_KPKM, tmp_path / "reg.json"), n)
        assert not at.exception, (n, [e.value for e in at.exception])
    for n in range(1, 5):
        at = _app(ROLE_VAL, tmp_path / "reg.json", step=n)
        assert not at.exception, (n, [e.value for e in at.exception])


def test_the_kpkm_workspace_has_no_model_controls(tmp_path):
    at = _app(ROLE_KPKM, tmp_path / "reg.json")
    for n in range(1, 5):
        at = _kpkm_page(at, n)
        assert not at.exception
        for key in ("tr_confirm", "tr_assess", "tr_run", "so_approve", "so_reject"):
            assert _btn(at, key) is None, (n, key)


def test_the_overview_hub_shows_the_governance_status_and_jumps_to_a_workspace(tmp_path):
    at = _app("Overview", tmp_path / "reg.json")
    assert not at.exception
    kpi = {m.label: m.value for m in at.metric}
    assert kpi["Verified Transfer Models"].startswith("0 / ") and "Mean Granary Yield" in kpi          # nothing is approved until a validator signs off
    _btn(at, "jump_validator").click().run()
    assert at.session_state["nav_mode"] == ROLE_VAL and at.session_state["validator_step"] == 1 and not at.exception
    at = _app("Overview", tmp_path / "reg.json")
    _btn(at, "jump_kpkm").click().run()
    assert at.session_state["nav_mode"] == ROLE_KPKM and _radio(at, "Pages") is not None


def test_the_validator_steps_have_next_and_back_buttons(tmp_path):
    at = _app(ROLE_VAL, tmp_path / "reg.json")
    path = [("nx_1", 2), ("nx_2", 3), ("nx_3", 4), ("bk_4", 3), ("bk_3", 2), ("bk_2", 1)]
    for key, want in path:
        _btn(at, key).click().run()
        assert at.session_state["validator_step"] == want and not at.exception, (key, want)
    at = _app(ROLE_VAL, tmp_path / "reg.json", step=4)
    _btn(at, "nx_4").click().run()
    assert at.session_state["nav_mode"] == "Overview"                                                  # Finish returns to the overview hub


def test_clicking_the_sidebar_granary_is_the_one_selected_granary_for_every_page(tmp_path):
    at = _app(ROLE_KPKM, tmp_path / "reg.json")
    at.sidebar.selectbox[0].set_value("IADA Kerian").run()
    assert at.session_state["granary"] == "IADA Kerian"
    at = _kpkm_page(at, 2)
    assert at.session_state["granary"] == "IADA Kerian" and not at.exception


def test_downstream_steps_are_locked_with_no_action_widgets_at_the_start():
    at = _app(step=2)
    assert _btn(at, "tr_confirm") is not None
    for key in ("tr_assess", "tr_harm", "tr_suit"):
        assert _btn(at, key) is None, key
    at = _goto(at, 3)
    for key in ("tr_strategy", "tr_run"):
        assert _btn(at, key) is None, key
    assert not _has(at, "pr_season", "number_input") and not _has(at, "out_n", "slider")      # the forecast preview offers nothing yet
    locked = [w.value for w in at.warning if w.value.startswith("Locked.")]                     # Streamlit moves the leading emoji into the icon
    assert len(locked) == 4, locked                                                              # strategy, training, prediction and outlook each explain why they are locked


def test_each_stage_unlocks_only_the_next_one():
    at = _app()
    _through(at, "tr_assess")
    assert _btn(at, "tr_harm") is not None and _btn(at, "tr_suit") is None
    _btn(at, "tr_harm").click().run()
    assert _btn(at, "tr_suit") is not None
    _btn(at, "tr_suit").click().run()
    at = _goto(at, 3)
    assert _btn(at, "tr_strategy") is not None and _btn(at, "tr_run") is None
    _btn(at, "tr_strategy").click().run()
    assert _btn(at, "tr_run") is not None
    assert not _has(at, "pr_season", "number_input") and not _has(at, "out_n", "slider")          # still no model: prediction and outlook stay locked
    _btn(at, "tr_run").click().run()
    assert not at.exception
    assert _has(at, "pr_season", "number_input") and _has(at, "out_n", "slider")                   # now they unlock
    assert "Model generating these predictions" in _text(at)                                      # and name the model that makes them


def test_backend_refuses_even_if_the_page_is_bypassed():
    at = _app(step=2)
    _btn(at, "tr_confirm").click().run()
    wf = at.session_state["acts_wf"]
    from acts_transfer import WorkflowError
    with pytest.raises(WorkflowError):
        wf.run(methods=["kmm"])
    with pytest.raises(WorkflowError):
        wf.predict_for(None)
    _btn(at, "tr_assess").click().run()
    with pytest.raises(WorkflowError):
        wf.run(methods=["kmm"])


def test_changing_the_strategy_clears_and_recalculates_downstream_and_tells_the_user():
    at = _app()
    _through(at, "tr_run")
    wf = at.session_state["acts_wf"]
    assert wf.prediction is not None and wf.outlook is not None            # tabs 6 and 7 computed from the evaluated model
    at.multiselect[[m.key for m in at.multiselect].index("tr_methods")].set_value(["coral"]).run()
    _btn(at, "tr_strategy").click().run()
    wf = at.session_state["acts_wf"]
    assert wf.result is None and wf.prediction is None and wf.outlook is None
    assert "were cleared and are recalculated" in _text(at)
    assert not _has(at, "pr_season", "number_input")                        # locked again until the model is retrained
    _btn(at, "tr_run").click().run()
    wf = at.session_state["acts_wf"]
    assert wf.result is not None and wf.prediction is not None and wf.outlook is not None


def test_a_signed_off_granary_reaches_the_kpkm_brief_and_an_unsigned_one_does_not(tmp_path):
    reg = tmp_path / "reg.json"
    at = _app(ROLE_VAL, reg)
    _through(at, "tr_run")
    _goto(at, 4)
    at.text_input[[t.key for t in at.text_input].index("so_name")].set_value("Test Validator").run()
    _btn(at, "so_reject").click().run()                                                    # a recorded decision: the target is now REJECTED_LOCAL_FALLBACK
    assert not at.exception
    from acts_transfer import governance
    e = governance.load(reg)["granaries"]["IADA Barat Laut Selangor"]
    assert e["status"] == governance.REJECTED and e["validator"] == "Test Validator" and e["published"]["baseline_model"]
    at = _kpkm_page(_app(ROLE_KPKM, reg), 4)
    at.sidebar.selectbox[0].set_value("IADA Barat Laut Selangor").run()
    _btn(at, "get_advisory").click().run()
    t = _text(at)
    assert "validator-reviewed yield forecast" in t and "Local Baseline Active" in t
    at.sidebar.selectbox[0].set_value("IADA Kerian").run()
    _btn(at, "get_advisory").click().run()
    assert "no validator-reviewed yield forecast for this granary" in _text(at)


def test_approval_is_impossible_when_the_gate_has_not_passed(tmp_path):
    reg = tmp_path / "reg.json"
    at = _app(ROLE_VAL, reg)
    _through(at, "tr_run")
    _goto(at, 4)
    b = _btn(at, "so_approve")
    assert b is None or b.disabled                                                         # no deployable hybrid was run, or its gate failed
    from acts_transfer import governance
    with pytest.raises(ValueError):
        governance.record_signoff("X", "APPROVE", "v", "", {"gate_passed": False}, path=reg)


def test_the_full_validator_to_kpkm_path_with_the_hybrid_approved(tmp_path):
    """Discover -> Verify -> Deploy through the real pages: train the Detrended Residual Transfer hybrid, approve it, and read it as the KPKM Programme Manager"""
    reg = tmp_path / "reg.json"
    at = _app(ROLE_VAL, reg)
    _through(at, "tr_suit")
    _goto(at, 3)
    ms = at.multiselect[[m.key for m in at.multiselect].index("tr_methods")]
    ms.set_value(list(dict.fromkeys(list(ms.value) + ["detrended_residual_transfer"]))).run()
    _btn(at, "tr_strategy").click().run()
    _btn(at, "tr_run").click().run()
    assert not at.exception
    c = at.session_state["acts_wf"].choice
    row = c["table"][c["table"].method == "detrended_residual_transfer"].iloc[0]
    assert bool(row["deployable"]) and row["seasons compared"] >= 3
    _goto(at, 4)
    at.text_input[[t.key for t in at.text_input].index("so_name")].set_value("Validator A").run()
    ap = _btn(at, "so_approve")
    assert ap is not None and ap.disabled == (not bool(row["gate passed"]))                  # the button is enabled exactly when the gate passed
    if not row["gate passed"]:
        pytest.skip("the hybrid did not pass the gate on this data, so there is nothing to approve")
    ap.click().run()
    assert not at.exception
    from acts_transfer import governance
    e = governance.load(reg)["granaries"]["IADA Barat Laut Selangor"]
    assert e["status"] == governance.APPROVED and e["published"]["transfer_model"] == "detrended_residual_transfer"
    assert set(e["published"]["outlook"]) == {"Normal", "El Niño drought", "Wet monsoon"}
    at = _kpkm_page(_app(ROLE_KPKM, reg), 3)
    at.sidebar.selectbox[0].set_value("IADA Barat Laut Selangor").run()
    assert not at.exception and "Verified Transfer Model" in _text(at)
    assert _has(at, "kp_scen", "radio")                                                    # the scenario selector exists only for an approved model


def test_the_decision_brief_shows_the_cited_reference_practice(tmp_path):
    at = _kpkm_page(_app(ROLE_KPKM, tmp_path / "reg.json"), 4)
    at.sidebar.selectbox[0].set_value("IADA Barat Laut Selangor").run()
    _btn(at, "get_advisory").click().run()
    t = _text(at)
    assert not at.exception and "Source Resource" in t and "has not been checked by this project" in t and "Kaedah Pengurusan Tanaman Padi" in t


def test_the_sidebar_always_keeps_the_selected_granary_and_a_role_card(tmp_path):
    for mode in ("Overview", ROLE_KPKM, ROLE_VAL):
        at = _app(mode, tmp_path / "reg.json")
        assert len(at.sidebar.selectbox) == 1 and at.sidebar.selectbox[0].options[0] and not at.exception
        assert any("sb-role-box" in m.value for m in at.sidebar.markdown)


def test_the_map_page_shows_a_field_summary_card_for_the_selected_granary(tmp_path):
    at = _app(ROLE_KPKM, tmp_path / "reg.json")
    at.sidebar.selectbox[0].set_value("IADA Kerian").run()
    assert not at.exception
    t = _text(at)
    assert "Field summary: IADA Kerian" in t
    assert "Rank among granaries" in [m.label for m in at.metric] and "Optical valid coverage" in [m.label for m in at.metric]
    assert "Pending review" not in t and any("Season 1" in m.value for m in at.markdown)
