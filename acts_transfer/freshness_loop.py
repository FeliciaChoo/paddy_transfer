"""Section 5 · Closed-Loop Data Freshness, Meeting Coordination & Retraining Trigger.

Evaluates lag_years = current_system_year - max(Y['Year']) >= 1, persists Programme Manager
meeting coordination requests and Data Steward ingestion events in results/data_update_log.json,
and runs the incremental pipeline (ingest_polygon_daily -> eda_clean -> build_stage_features ->
build_transfer_dataset -> prepare_app_data).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = ROOT / "results" / "data_update_log.json"


def check_data_freshness(Y: pd.DataFrame, current_year: int | None = None) -> dict[str, Any]:
    """Evaluate lag_years = current_system_year - max(Y['Year']) and list missing harvest years."""
    sys_year = int(datetime.now().year) if current_year is None else int(current_year)
    if Y is None or Y.empty or "Year" not in Y.columns:
        return {
            "is_stale": True,
            "current_year": sys_year,
            "max_data_year": None,
            "lag_years": sys_year,
            "missing_years": [sys_year],
        }
    max_yr = int(pd.to_numeric(Y["Year"], errors="coerce").dropna().max())
    lag = sys_year - max_yr
    missing = list(range(max_yr + 1, sys_year + 1)) if lag >= 1 else []
    return {
        "is_stale": lag >= 1,
        "current_year": sys_year,
        "max_data_year": max_yr,
        "lag_years": lag,
        "missing_years": missing,
    }


def load_update_log(log_path: Path | str = DEFAULT_LOG_PATH) -> dict[str, Any]:
    p = Path(log_path)
    if not p.exists():
        return {"requests": [], "ingestion_runs": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        data.setdefault("requests", [])
        data.setdefault("ingestion_runs", [])
        return data
    except Exception:
        return {"requests": [], "ingestion_runs": []}


def append_coordination_request(
    scope: str,
    granary: str,
    items: list[str],
    meeting_notes: str,
    log_path: Path | str = DEFAULT_LOG_PATH,
) -> dict[str, Any]:
    """Save Programme Manager meeting coordination notes & requested items to results/data_update_log.json."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    log = load_update_log(p)
    entry = {
        "id": f"REQ-{len(log['requests']) + 1:03d}",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": "KPKM Programme Manager",
        "scope": scope,
        "granary": granary if scope == "Selected Granary" else "All 10 Granaries",
        "requested_items": list(items),
        "meeting_notes": meeting_notes.strip(),
        "status": "PENDING_STEWARD",
    }
    log["requests"].append(entry)
    p.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return entry


def merge_new_yield_csv(uploaded_df: pd.DataFrame, data_dir: Path = ROOT / "data") -> int:
    """Merge and deduplicate new annual harvest yield records on ['District', 'Year']."""
    target_csv = data_dir / "transfer_yield_dataset.csv"
    if target_csv.exists():
        existing = pd.read_csv(target_csv)
        combined = pd.concat([existing, uploaded_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["District", "Year"], keep="last").sort_values(["District", "Year"])
    else:
        combined = uploaded_df.drop_duplicates(subset=["District", "Year"], keep="last")
    combined.to_csv(target_csv, index=False)
    return len(combined)


def run_retraining_pipeline(
    root_dir: Path = ROOT,
    log_path: Path | str = DEFAULT_LOG_PATH,
    automl_note: str = "",
) -> tuple[bool, list[str]]:
    """Execute ingest_polygon_daily -> eda_clean -> build_stage_features -> build_transfer_dataset -> prepare_app_data."""
    scripts = [
        "pipeline/ingest_polygon_daily.py",
        "pipeline/eda_clean.py",
        "pipeline/build_stage_features.py",
        "pipeline/build_transfer_dataset.py",
        "pipeline/prepare_app_data.py",
    ]
    logs: list[str] = []
    ok = True
    for rel in scripts:
        script_path = root_dir / rel
        if not script_path.exists():
            logs.append(f"SKIP (not found): {rel}")
            continue
        res = subprocess.run([sys.executable, str(script_path)], cwd=str(root_dir), capture_output=True, text=True)
        if res.returncode != 0:
            ok = False
            logs.append(f"FAIL {rel}: {res.stderr.strip()[-400:]}")
            break
        logs.append(f"OK {rel}")

    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    log = load_update_log(p)
    for req in log["requests"]:
        if req.get("status") == "PENDING_STEWARD" and ok:
            req["status"] = "RESOLVED"
    log["ingestion_runs"].append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "success": ok,
        "automl_config_applied": automl_note,
        "steps": logs,
    })
    p.write_text(json.dumps(log, indent=2), encoding="utf-8")
    return ok, logs