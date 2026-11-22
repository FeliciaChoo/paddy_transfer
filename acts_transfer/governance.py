"""Governance registry (PACE V8): the bridge between the Model Validator & Data Steward (writer, Tab 5) and the KPKM Programme Manager (reader, Tabs 1, 4, 5).

results/governance_registry.json  {"version": 1, "granaries": {<granary>: entry}}

An entry is written ONLY by a recorded sign-off. States:
  APPROVED                 the validator approved the transfer model after both gate criteria passed. The KPKM views serve the Hybrid Trend + S-TrAdaBoost.R2 forecast
                           with the green 'Verified Transfer Model' badge.
  REJECTED_LOCAL_FALLBACK  the gate failed (automatic) or the validator rejected the model. The KPKM views serve the validated local baseline (season trend or
                           persistence) with the amber 'Local Baseline Active' badge.
  PENDING_REVIEW           the gate passed but nobody has signed off yet, or there is no entry at all. Defaults safely to the local baseline.
A granary with no entry is PENDING_REVIEW. Approval is refused unless the stored gate evidence shows both criteria passed, so it cannot be recorded by hand
for a model that failed. Each entry keeps the evidence, the validator's name and note, and a timestamp; the 'published' block holds the numbers the KPKM views display
(so they never run or configure a model).
"""
import datetime as _dt
import json
import os
import tempfile
from pathlib import Path

APPROVED, REJECTED, PENDING = "APPROVED", "REJECTED_LOCAL_FALLBACK", "PENDING_REVIEW"
BADGE = {APPROVED: ("🟢", "Verified Transfer Model", "success"), REJECTED: ("🟠", "Local Baseline Active", "warning"), PENDING: ("🟠", "Local Baseline Active (pending validator review)", "warning")}
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "results" / "governance_registry.json"


def load(path=None):
    p = Path(path or DEFAULT_PATH)
    if not p.exists():
        return {"version": 1, "granaries": {}}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        d.setdefault("granaries", {})
        return d
    except Exception:
        return {"version": 1, "granaries": {}, "error": "the registry file could not be read; every granary is treated as PENDING_REVIEW"}


def _save(reg, path=None):
    p = Path(path or DEFAULT_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, default=str)
    os.replace(tmp, p)                                                       # atomic: a reader never sees a half-written file


def gate_status(choice):
    """the automatic status the gate alone implies: PENDING_REVIEW if both criteria passed (awaiting sign-off), else REJECTED_LOCAL_FALLBACK"""
    return PENDING if choice["safeguard"].get("gate_passed") and choice.get("transfer_supported") else REJECTED


def record_signoff(granary, decision, validator, note, evidence, published=None, path=None):
    """decision: 'APPROVE' or 'REJECT'. evidence: dict with gate_passed (bool), model_key, model_label, delta_nt, baseline, domain_ok, token, source, validation.
    APPROVE is refused unless evidence['gate_passed']."""
    decision = decision.upper()
    if decision not in ("APPROVE", "REJECT"):
        raise ValueError("decision must be APPROVE or REJECT")
    if not str(validator or "").strip():
        raise ValueError("the validator's name is required for a sign-off")
    if decision == "APPROVE" and not evidence.get("gate_passed"):
        raise ValueError("cannot approve: the dual-criteria gate did not pass (domain relevance and Delta_NT < 0 against all local baselines)")
    reg = load(path)
    entry = dict(status=APPROVED if decision == "APPROVE" else REJECTED, decided=decision, validator=validator.strip(), note=(note or "").strip(),
                 timestamp=_dt.datetime.now().isoformat(timespec="seconds"), evidence=evidence, published=published or {})
    reg["granaries"][granary] = entry
    _save(reg, path)
    return entry


def clear(granary, path=None):
    reg = load(path)
    reg["granaries"].pop(granary, None)
    _save(reg, path)


def effective(granary, reg=None, path=None):
    """what the KPKM views serve for a granary: status, whether the transfer forecast is served, the badge, and the published numbers"""
    reg = reg if reg is not None else load(path)
    e = reg.get("granaries", {}).get(granary)
    status = e["status"] if e else PENDING
    icon, text, kind = BADGE[status]
    return dict(status=status, serves_transfer=status == APPROVED, badge=f"{icon} {text}", kind=kind, entry=e, published=(e or {}).get("published", {}))


def counts(granaries, reg=None, path=None):
    """(approved transfer, local baseline, of which pending) over the given granaries: the live count on the KPKM Overview"""
    reg = reg if reg is not None else load(path)
    st = [effective(g, reg)["status"] for g in granaries]
    return st.count(APPROVED), len(st) - st.count(APPROVED), st.count(PENDING)
