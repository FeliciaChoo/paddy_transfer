"""The 'Refresh this granary' button: lock files, the background process, and the reload of the in-memory results."""
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard import tab_transfer as tt


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """lock directory in tmp, and a stand-in for precompute_automl.py that records its arguments and takes ~1.5 s"""
    script = tmp_path / "fake_precompute.py"
    script.write_text("import os, sys, time, pathlib\npathlib.Path(sys.argv[0]).with_suffix('.args').write_text(' '.join(sys.argv[1:]))\ntime.sleep(1.5)\nprint('fake refresh log line')\nsys.exit(int(os.environ.get('FAKE_RC', '0')))\n", encoding="utf-8")
    monkeypatch.setattr(tt, "_REFRESH_LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(tt, "_REFRESH_SCRIPT", script)
    monkeypatch.setattr(tt, "_ROOT", tmp_path)
    monkeypatch.setattr(tt, "_REFRESH_PROCS", {})
    monkeypatch.setattr(tt, "_REFRESH_DONE", {})
    monkeypatch.setattr(tt, "_REFRESH_CLEARED", {})
    monkeypatch.delenv("FAKE_RC", raising=False)
    return script


def test_lock_file_name_is_filesystem_safe():
    assert tt._granary_lock_file("IADA Barat Laut/Selangor: 1").name == "IADA_Barat_Laut_Selangor__1.pid"
    assert tt._granary_lock_file("KADA").parent == tt._REFRESH_LOCK_DIR


def test_the_real_script_and_the_granary_option_exist():
    src = (ROOT / "pipeline" / "precompute_automl.py").read_text(encoding="utf-8")
    assert tt._REFRESH_SCRIPT == ROOT / "pipeline" / "precompute_automl.py" and tt._REFRESH_SCRIPT.exists()
    assert '"--only"' in src and '"--force"' in src                                          # the two options the button passes


def test_a_dead_pid_lock_is_not_refreshing_and_is_removed(sandbox):
    lock = tt._granary_lock_file("KADA")
    lock.parent.mkdir(parents=True)
    lock.write_text("999999999", encoding="utf-8")                                           # no such process
    assert tt._is_granary_refreshing("KADA") is False and not lock.exists()
    lock.write_text("not a pid", encoding="utf-8")
    assert tt._is_granary_refreshing("KADA") is False and not lock.exists()
    assert tt._is_granary_refreshing("never started") is False


def test_start_runs_the_script_for_this_granary_only_and_a_finished_run_is_not_a_zombie(sandbox):
    g = "IADA Rompin"
    tt._start_background_granary_refresh(g)
    assert tt._is_granary_refreshing(g) is True
    assert int(tt._granary_lock_file(g).read_text()) == tt._REFRESH_PROCS[g].pid
    pid_before = tt._REFRESH_PROCS[g].pid
    tt._start_background_granary_refresh(g)                                                  # a second click while running starts nothing
    assert tt._REFRESH_PROCS[g].pid == pid_before
    deadline = time.time() + 20
    while tt._is_granary_refreshing(g) and time.time() < deadline:
        time.sleep(0.2)
    assert tt._is_granary_refreshing(g) is False                                             # the child exited and was reaped: the button comes back
    assert not tt._granary_lock_file(g).exists()
    assert sandbox.with_suffix(".args").read_text() == "--only IADA Rompin --force"            # this granary only, recomputed even if current
    assert not tt._is_granary_refreshing("KADA")                                             # another granary is unaffected


@pytest.mark.skipif(os.name == "nt", reason="uses /proc")
def test_a_zombie_from_an_earlier_server_process_does_not_count_as_running(sandbox):
    import subprocess
    p = subprocess.Popen([sys.executable, "-c", "pass"])                                     # exits at once; not waited for, so it becomes a zombie
    deadline = time.time() + 10
    while time.time() < deadline and Path(f"/proc/{p.pid}/status").exists() and "Z" not in Path(f"/proc/{p.pid}/status").read_text().split("State:")[1].split("\n")[0]:
        time.sleep(0.05)
    lock = tt._granary_lock_file("KADA")
    lock.parent.mkdir(parents=True)
    lock.write_text(str(p.pid), encoding="utf-8")                                            # a lock left by a process this module no longer holds a handle for
    assert tt._is_granary_refreshing("KADA") is False
    p.wait()


def test_reload_clears_every_in_memory_copy_of_the_results():
    from streamlit.testing.v1 import AppTest
    src = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import streamlit as st
from dashboard import tab_transfer as tt
if "seeded" not in st.session_state:                      # seed once: a second run must show whether the click removed them
    st.session_state["seeded"] = True
    for k in tt._AUTOML_SESSION_KEYS:
        st.session_state[k] = "old"
tt.render_refresh_controls("IADA Rompin")
'''
    at = AppTest.from_string(src, default_timeout=60).run()
    assert not at.exception
    assert [b.label for b in at.button] == ["Refresh this granary", "Reload cached results"]
    assert all(at.session_state[k] == "old" for k in ("automl_summary", "tl_bundle"))
    at.button(key="btn_reload_cache_IADA Rompin").click().run()
    assert not at.exception
    assert not any(k in at.session_state for k in tt._AUTOML_SESSION_KEYS)                  # the in-memory copies are gone, so the next run reads the refreshed file


# ---------------------------------------------------------------------------------------------------- the page notices a finished refresh by itself
G = "IADA Rompin"


def _page(tick: bool):
    """a page that renders the controls; tick=True first calls the polling fragment exactly as one timer tick would"""
    from streamlit.testing.v1 import AppTest
    src = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import streamlit as st
from dashboard import tab_transfer as tt
if "seeded" not in st.session_state:
    st.session_state["seeded"] = True
    for k in tt._AUTOML_SESSION_KEYS:
        st.session_state[k] = "old"
if {tick!r} and "ticked" not in st.session_state:         # one timer tick per session: the real page renders the fragment only while a refresh runs, so a rerun does not re-enter it
    st.session_state["ticked"] = True
    tt._refresh_progress({G!r})
tt.render_refresh_controls({G!r})
"""
    return AppTest.from_string(src, default_timeout=60)


def _wait_until_finished():
    deadline = time.time() + 20
    while tt._REFRESH_PROCS.get(G) is not None and tt._REFRESH_PROCS[G].poll() is None and time.time() < deadline:
        time.sleep(0.1)


@pytest.fixture
def reloads(monkeypatch):
    calls = []
    orig = tt._reload_cached_results
    monkeypatch.setattr(tt, "_reload_cached_results", lambda: (calls.append(1), orig())[1])
    return calls


def test_the_exit_code_of_a_finished_run_is_recorded(sandbox, monkeypatch):
    tt._start_background_granary_refresh(G)
    _wait_until_finished()
    assert tt._is_granary_refreshing(G) is False and tt._REFRESH_DONE[G][1] == 0
    monkeypatch.setenv("FAKE_RC", "3")
    tt._start_background_granary_refresh(G)
    _wait_until_finished()
    assert tt._is_granary_refreshing(G) is False and tt._REFRESH_DONE[G][1] == 3


def test_while_a_refresh_runs_the_page_shows_progress_and_a_disabled_button(sandbox, reloads):
    tt._start_background_granary_refresh(G)
    at = _page(tick=True).run()
    assert not at.exception
    assert any("Background refresh running" in i.value and "updates by itself" in i.value for i in at.info)
    assert [(b.label, b.disabled) for b in at.button] == [("Refreshing...", True), ("Reload cached results", True)]      # both buttons stay; neither can be used while running
    assert reloads == [] and at.session_state["automl_summary"] == "old"                      # nothing is reloaded before the run ends
    _wait_until_finished()


def test_when_the_refresh_ends_the_page_reloads_the_results_by_itself_once(sandbox, reloads):
    tt._start_background_granary_refresh(G)
    at = _page(tick=False).run()
    assert any("Background refresh running" in i.value for i in at.info)
    _wait_until_finished()
    at = _page(tick=True).run()                                                              # one timer tick after the process ended
    assert not at.exception
    assert any("Refresh finished" in x.value for x in at.success)
    assert not any(k in at.session_state for k in tt._AUTOML_SESSION_KEYS)                  # old payload dropped: the next run reads the new file
    assert [(b.label, b.disabled) for b in at.button] == [("Refresh this granary", False), ("Reload cached results", False)]
    assert len(reloads) == 1
    at.run()                                                                                 # any later interaction: no second reload, no second message
    assert len(reloads) == 1 and not at.success


def test_a_busy_page_still_reloads_when_it_comes_back_after_the_run_ended(sandbox, reloads):
    at = _page(tick=False).run()                                                             # the session exists before the run starts
    tt._start_background_granary_refresh(G)
    _wait_until_finished()
    tt._is_granary_refreshing(G)
    at.run()                                                                                 # no timer tick in between: the main script notices
    assert any("Refresh finished" in x.value for x in at.success) and len(reloads) == 1


def test_a_failed_refresh_keeps_the_old_results_and_shows_the_log(sandbox, reloads, monkeypatch):
    monkeypatch.setenv("FAKE_RC", "3")
    tt._start_background_granary_refresh(G)
    at = _page(tick=False).run()
    _wait_until_finished()
    at = _page(tick=True).run()
    assert not at.exception
    assert any("failed (exit code 3)" in e.value and "fake refresh log line" in e.value for e in at.error)
    assert reloads == [] and at.session_state["automl_summary"] == "old"                      # the old results are still what is shown


def test_the_shared_cache_is_reloaded_once_per_run_even_when_nobody_was_watching(sandbox, reloads):
    tt._start_background_granary_refresh(G)
    _wait_until_finished()
    tt._is_granary_refreshing(G)                                                             # the run ended while no browser had the page open
    a = _page(tick=False).run()                                                              # the first visitor afterwards triggers the one reload ...
    assert len(reloads) == 1 and not a.success                                               # ... without being told about a run they did not start
    b = _page(tick=False).run()                                                              # a second visitor: nothing more to do
    assert len(reloads) == 1 and not b.success


def test_the_poll_interval_is_a_few_seconds():
    assert 1 <= tt._REFRESH_POLL_SECONDS <= 10


def test_the_polling_fragment_reruns_the_whole_page_only_when_the_refresh_has_ended(sandbox):
    """in the browser a fragment updates only itself: without an app rerun the 'Refreshing...' button would stay disabled"""
    from streamlit.testing.v1 import AppTest
    src = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import streamlit as st
from dashboard import tab_transfer as tt
def _count(*a, **k):
    st.session_state["reruns"] = st.session_state.get("reruns", 0) + 1
st.rerun = _count
tt._refresh_progress({G!r})
"""
    tt._start_background_granary_refresh(G)
    at = AppTest.from_string(src, default_timeout=60).run()
    assert not at.exception and "reruns" not in at.session_state                              # still running: no page rerun
    _wait_until_finished()
    at = AppTest.from_string(src, default_timeout=60).run()
    assert not at.exception and at.session_state["reruns"] == 1                              # ended: exactly one rerun of the whole page


# ---------------------------------------------------------------------------------------------------- two buttons that stay, a message box beside them, no other text
def test_when_idle_there_are_two_buttons_and_no_message_at_all(sandbox):
    at = _page(tick=False).run()
    assert not at.exception
    assert [(b.label, b.disabled) for b in at.button] == [("Refresh this granary", False), ("Reload cached results", False)]
    assert not at.info and not at.success and not at.error and not at.warning                  # no standing text: nothing until a button is used


def test_clicking_reload_shows_its_message_beside_the_buttons_once(sandbox):
    at = _page(tick=False).run()
    at.button(key=f"btn_reload_cache_{G}").click().run()
    assert not at.exception
    assert any("Cached results reloaded" in x.value for x in at.success)
    assert [b.label for b in at.button] == ["Refresh this granary", "Reload cached results"]   # both buttons are still there
    at.run()                                                                                   # the next interaction: the message is gone again
    assert not at.success and not at.info


def test_the_message_box_is_the_third_column_next_to_the_two_buttons(sandbox):
    """the message is rendered in the column to the right of the buttons, not above or below them"""
    at = _page(tick=False).run()
    at.button(key=f"btn_reload_cache_{G}").click().run()
    cols = [c for c in at.columns]
    assert len(cols) == 3
    assert [b.label for b in cols[0].button] == ["Refresh this granary"] and [b.label for b in cols[1].button] == ["Reload cached results"]
    assert any("Cached results reloaded" in x.value for x in cols[2].success) and not cols[0].success and not cols[1].success


def test_results_from_older_code_are_signalled_only_by_the_hover_help_not_by_text_on_the_page():
    from streamlit.testing.v1 import AppTest
    src = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import streamlit as st
from dashboard import tab_transfer as tt
st.session_state["automl_summary"] = {{"stale_code": True}}
tt.render_refresh_controls({G!r})
"""
    at = AppTest.from_string(src, default_timeout=60).run()
    assert not at.exception and not at.info and not at.warning
    assert "earlier version" in at.button(key=f"btn_refresh_{G}").help
