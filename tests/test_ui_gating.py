"""Interface-level gating: tab 7 stays locked, and no later-step button exists, until the earlier steps are complete (and the backend agrees).

Run:  python -m pytest tests/test_ui_gating.py -q      (a few seconds per step: it runs the real assessment)
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest        # noqa: E402

pytestmark = pytest.mark.skipif(not (ROOT / "data" / "transfer_yield_dataset.csv").exists(), reason="needs data/transfer_yield_dataset.csv")


def _app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=600).run()


def _btn(at, key):
    b = [x for x in at.button if x.key == key]
    return b[0] if b else None


def _text(at):
    return " ".join(x.value for blk in (at.markdown, at.warning, at.info, at.success, at.error) for x in blk)


def test_transfer_tab_is_locked_and_later_buttons_do_not_exist_at_the_start():
    at = _app()
    assert not at.exception
    assert "Transfer learning is locked" in _text(at)
    for key in ("tr_assess", "tr_harm", "tr_suit", "tr_strategy", "tr_run"):
        assert _btn(at, key) is None, key
    assert _btn(at, "tr_confirm") is not None


def test_each_step_unlocks_only_the_next_one():
    at = _app()
    _btn(at, "tr_confirm").click().run()
    assert _btn(at, "tr_assess") is not None and _btn(at, "tr_harm") is None and _btn(at, "tr_run") is None
    _btn(at, "tr_assess").click().run()
    assert _btn(at, "tr_harm") is not None and _btn(at, "tr_suit") is None
    assert "Transfer learning is locked" in _text(at)                 # assessment alone does not unlock transfer
    _btn(at, "tr_harm").click().run()
    assert _btn(at, "tr_suit") is not None
    assert "Transfer learning is locked" in _text(at)                 # harmonisation alone does not unlock transfer
    _btn(at, "tr_suit").click().run()
    assert "Transfer learning is locked" not in _text(at)
    assert _btn(at, "tr_run") is None                                 # still needs a confirmed strategy
    _btn(at, "tr_strategy").click().run()
    assert _btn(at, "tr_run") is not None


def test_backend_refuses_even_if_the_page_is_bypassed():
    at = _app()
    _btn(at, "tr_confirm").click().run()
    wf = at.session_state["acts_wf"]
    from acts_transfer import WorkflowError
    with pytest.raises(WorkflowError):
        wf.run(methods=["kmm"])                                       # no assessment yet: the backend says no
    _btn(at, "tr_assess").click().run()
    with pytest.raises(WorkflowError):
        wf.run(methods=["kmm"])                                       # assessment but no harmonisation: still no
