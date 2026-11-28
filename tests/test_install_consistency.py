"""pipeline/check_install.py: it must catch the 'one file is newer than its neighbour' failures and stay quiet on a consistent tree."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import check_install as ci


def tree(tmp_path, files: dict) -> Path:
    for rel, src in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    return tmp_path


def test_the_project_itself_has_no_new_mismatch_only_the_known_legacy_ones():
    new, known = ci.split_known(ci.check(ROOT))
    assert new == [], new
    assert ci.main([str(ROOT)]) == 0
    assert len(known) <= 5 and all(k.startswith(ci.KNOWN_LEGACY) for k in known)            # nothing hides behind the allow-list


def test_a_name_imported_from_an_older_module_is_reported_with_both_files(tmp_path):
    t = tree(tmp_path, {"acts_transfer/diagnostics.py": "def compute_a():\n    pass\n",
                        "dashboard/tab_transfer.py": "from acts_transfer.diagnostics import compute_a, gate_evidence\n"})
    msgs = ci.check(t)
    assert len(msgs) == 1 and "dashboard/tab_transfer.py imports 'gate_evidence' from acts_transfer/diagnostics.py" in msgs[0]
    (t / "acts_transfer" / "diagnostics.py").write_text("def compute_a():\n    pass\n\ndef gate_evidence():\n    pass\n", encoding="utf-8")
    assert ci.check(t) == []                                                                # replacing the older file fixes it


def test_an_attribute_used_on_a_module_that_lacks_it_is_reported(tmp_path):
    t = tree(tmp_path, {"dashboard/tab_transfer.py": "def render_validator_signoff(D):\n    pass\n",
                        "app.py": "from dashboard import tab_transfer\n\ntab_transfer.render_validator_signoff(1)\ntab_transfer.render_refresh_controls('x')\n"})
    msgs = ci.check(t)
    assert len(msgs) == 1 and "app.py uses 'tab_transfer.render_refresh_controls'" in msgs[0] and "dashboard/tab_transfer.py" in msgs[0]


def test_aliases_relative_imports_and_conditional_definitions_are_followed(tmp_path):
    t = tree(tmp_path, {"dashboard/bridge.py": "try:\n    import x\nexcept ImportError:\n    x = None\n\nif True:\n    def late():\n        pass\n\nVALUE = 1\n",
                        "dashboard/tab_overview.py": "from . import bridge as B\nfrom .bridge import VALUE, late\n\nB.late()\nB.VALUE\nB.missing()\n"})
    msgs = ci.check(t)
    assert len(msgs) == 1 and "'B.missing'" in msgs[0]


def test_star_imports_and_unknown_modules_are_not_guessed_about(tmp_path):
    t = tree(tmp_path, {"acts_transfer/wide.py": "from os.path import *\n",
                        "dashboard/a.py": "from acts_transfer.wide import anything\n",
                        "dashboard/b.py": "import numpy as np\nfrom acts_transfer.nope import thing\n"})
    msgs = ci.check(t)
    assert not any("anything" in m for m in msgs)                                          # a star import: cannot tell, so no claim
    assert any("'acts_transfer.nope', which does not exist" in m for m in msgs)


def test_a_file_that_does_not_parse_is_reported(tmp_path):
    t = tree(tmp_path, {"dashboard/broken.py": "def (:\n"})
    assert any("cannot be parsed" in m for m in ci.check(t))


def test_the_exit_code_follows_the_new_mismatches_only(tmp_path, capsys):
    t = tree(tmp_path, {"acts_transfer/d.py": "A = 1\n", "dashboard/t.py": "from acts_transfer.d import B\n"})
    assert ci.main([str(t)]) == 1 and "one of the two files is older" in capsys.readouterr().out
    (t / "dashboard" / "t.py").write_text("from acts_transfer.d import A\n", encoding="utf-8")
    assert ci.main([str(t)]) == 0
