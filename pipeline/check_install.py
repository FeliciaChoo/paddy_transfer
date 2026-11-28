"""Static check that the project files belong together. Run it after copying a bundle in:  python pipeline/check_install.py

It reads the source of app.py, dashboard/, acts_transfer/ and pipeline/ WITHOUT importing them (no TensorFlow, no Streamlit, about a second) and reports
  * `from acts_transfer.x import name` / `from dashboard.x import name` / `from .x import name` where x does not define that name, and
  * `module.attr` uses (for example `tab_transfer.render_refresh_controls(...)` in app.py) where the module does not define attr.
Both are what breaks when one file is newer than its neighbour, for example:
    ImportError: cannot import name 'gate_evidence' from 'acts_transfer.diagnostics'
The message names the file that uses the missing name and the file that should define it: replace the OLDER one of the two with the newer copy.
Exit code 0 = consistent, 1 = at least one mismatch. Names that are built dynamically are not seen, so a clean result is necessary, not sufficient.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("acts_transfer", "dashboard")                         # importable project packages
SCAN = ["app.py"] + [f"{p}/*.py" for p in PACKAGES] + ["pipeline/*.py"]
# Mismatches that are already in the ORIGINAL zip, before any bundle: tab_governance.py calls helpers (_wf, _data, _lbl, primary_label) that tab_transfer.py never had, and
# an old script imports TransferWorkflow, which acts_transfer no longer exports. They are listed as notes and do not fail the check.
KNOWN_LEGACY = ("dashboard/tab_governance.py uses 'T.", "pipeline/validate_transfer_assessment.py imports 'TransferWorkflow'")


def _defined(path: Path) -> set | None:
    """names a module defines at its top level (functions, classes, assignments, imports); None when it has a star import or cannot be parsed"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    names: set = set()
    for node in tree.body:
        for n in ast.walk(node) if isinstance(node, (ast.If, ast.Try)) else [node]:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            names.add(x.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    if a.name == "*":
                        return None
                    names.add((a.asname or a.name).split(".")[0])
    return names


def _module_path(root: Path, dotted: str) -> Path | None:
    p = root.joinpath(*dotted.split("."))
    for cand in (p.with_suffix(".py"), p / "__init__.py"):
        if cand.exists():
            return cand
    return None


def check(root: Path = ROOT) -> list[str]:
    """returns one message per mismatch"""
    root = Path(root)
    problems: list[str] = []
    cache: dict = {}

    def defs(path: Path):
        if path not in cache:
            cache[path] = _defined(path)
        return cache[path]

    files = sorted({f for pat in SCAN for f in root.glob(pat)})
    for f in files:
        rel = f.relative_to(root).as_posix()
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            problems.append(f"{rel}: cannot be parsed ({exc})")
            continue
        pkg = f.parent.name if f.parent != root else ""
        aliases: dict = {}                                                         # local name -> module file, for `alias.attr` checks
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            base = node.module or ""
            if node.level:                                                         # from . import x  /  from .x import y
                base = (pkg + ("." + base if base else "")) if pkg else base
            top = base.split(".")[0]
            if top not in PACKAGES:
                continue
            target = _module_path(root, base)
            for a in node.names:
                if a.name == "*":
                    continue
                sub = _module_path(root, f"{base}.{a.name}")                      # `from dashboard import tab_transfer` imports a module
                if sub is not None:
                    aliases[a.asname or a.name] = sub
                    continue
                d = defs(target) if target else None
                if target is None:
                    problems.append(f"{rel}: imports from '{base}', which does not exist")
                elif d is not None and a.name not in d:
                    problems.append(f"{rel} imports '{a.name}' from {target.relative_to(root).as_posix()}, which does not define it "
                                    f"(one of the two files is older: replace the older one)")
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
                target = aliases[node.value.id]
                d = defs(target)
                if d is not None and node.attr not in d:
                    msg = (f"{rel} uses '{node.value.id}.{node.attr}', which {target.relative_to(root).as_posix()} does not define "
                           f"(one of the two files is older: replace the older one)")
                    if msg not in problems:
                        problems.append(msg)
    return problems


def split_known(problems: list[str]) -> tuple[list[str], list[str]]:
    """(new mismatches, known legacy ones)"""
    known = [p for p in problems if p.startswith(KNOWN_LEGACY)]
    return [p for p in problems if p not in known], known


def main(argv=None) -> int:
    root = Path(argv[0]) if argv else ROOT
    new, known = split_known(check(root))
    for p in new:
        print(" -", p)
    if known:
        print(f"note: {len(known)} older mismatch(es) were already in the original zip and are unrelated to any bundle (tab_governance.py / validate_transfer_assessment.py)")
    if not new:
        print(f"OK: no new mismatched imports or module attributes under {root}")
        return 0
    print(f"{len(new)} mismatch(es) found under {root}: in each line one of the two files is older: replace it with the newer copy")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
