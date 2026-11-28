"""Steps 1–4 · Data Steward Ingestion, AutoML-Assisted Domain Divergence Audit, Transfer Benchmark & Validator Sign-Off.

Reads directly from acts_transfer.automl_runner.run_full_automl_for_granary so every step
renders immediately with 0-100% AutoML scores and zero sequential button locks.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from acts_transfer import governance as gov
from acts_transfer.automl_runner import run_full_automl_for_granary
from acts_transfer.causal_id import dml_estimate_and_refute, partition_causal_variables
from acts_transfer.diagnostics import (
    compute_few_shot_scarcity,
    compute_weather_ablation,
    compute_weather_driver_attribution,
    gate_evidence,
)
from acts_transfer.methods import LABELS


# ------------------------------------------------------------------------------------------------
# Background refresh helpers (kept intact for test compatibility; UI manual buttons removed)
# ------------------------------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
_REFRESH_SCRIPT = _ROOT / "pipeline" / "precompute_automl.py"
_REFRESH_LOCK_DIR = _ROOT / "results" / ".refresh_locks"
_REFRESH_PROCS: dict = {}
_REFRESH_DONE: dict = {}
_REFRESH_CLEARED: dict = {}
_REFRESH_POLL_SECONDS = 3
_AUTOML_SESSION_KEYS = (
    "automl_summary",
    "tl_bundle",
    "tl_assessment",
    "tl_eval_results",
    "tl_choice",
    "tl_all_donor_ranking",
)


def _granary_lock_file(granary: str) -> Path:
    safe_name = "".join(c if c.isalnum() else "_" for c in granary)
    return _REFRESH_LOCK_DIR / f"{safe_name}.pid"


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.check_output(["tasklist", "/FI", f"PID eq {pid}"], stderr=subprocess.DEVNULL, text=True)
        return re.search(rf"(?<!\d){pid}(?!\d)", out) is not None
    os.kill(pid, 0)
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            if any(line.startswith("State:") and "Z" in line.split(":", 1)[1] for line in f):
                return False
    except OSError:
        pass
    return True


def _is_granary_refreshing(granary: str) -> bool:
    proc = _REFRESH_PROCS.get(granary)
    if proc is not None:
        if proc.poll() is None:
            return True
        _REFRESH_PROCS.pop(granary, None)
        _REFRESH_DONE[granary] = (time.time(), proc.returncode)
        _granary_lock_file(granary).unlink(missing_ok=True)
        return False
    lock_file = _granary_lock_file(granary)
    if not lock_file.exists():
        return False
    try:
        alive = _pid_alive(int(lock_file.read_text(encoding="utf-8").strip()))
    except Exception:
        alive = False
    if not alive:
        lock_file.unlink(missing_ok=True)
    return alive


def _start_background_granary_refresh(granary: str) -> None:
    if _is_granary_refreshing(granary):
        return
    _REFRESH_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _REFRESH_LOCK_DIR / "last_refresh.log"
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            [sys.executable, str(_REFRESH_SCRIPT), "--only", granary, "--force"],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(_ROOT),
            creationflags=creationflags,
            start_new_session=(os.name != "nt"),
        )
    _REFRESH_PROCS[granary] = proc
    _granary_lock_file(granary).write_text(str(proc.pid), encoding="utf-8")


def _reload_cached_results() -> None:
    """Drop every in-memory copy of the AutoML payload so the next run reads the newly written file."""
    st.cache_data.clear()
    for k in list(st.session_state.keys()):
        if k in _AUTOML_SESSION_KEYS or k.startswith("tl_"):
            st.session_state.pop(k, None)


def _log_tail(n: int = 6) -> str:
    try:
        lines = (_REFRESH_LOCK_DIR / "last_refresh.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()
        return "\n".join(lines[-n:])
    except OSError:
        return ""


def _drop_stale_results(granary: str) -> bool:
    done = _REFRESH_DONE.get(granary)
    if done and done[1] == 0 and _REFRESH_CLEARED.get(granary) != done[0]:
        _REFRESH_CLEARED[granary] = done[0]
        _reload_cached_results()
        return True
    return False


def _finish_refresh(granary: str, ran_here: bool = False) -> bool:
    done = _REFRESH_DONE.get(granary)
    reloaded = _drop_stale_results(granary)
    if done is None:
        if not ran_here:
            return False
        _reload_cached_results()
        st.session_state["_refresh_msg"] = (granary, "success", f"Refresh finished for **{granary}**. The page now shows the new results.")
        return True
    seen = st.session_state.setdefault("_refresh_seen", {})
    if seen.get(granary) == done[0]:
        return reloaded
    seen[granary] = done[0]
    if done[1] == 0:
        st.session_state["_refresh_msg"] = (granary, "success", f"Refresh finished for **{granary}**. The page now shows the new results.")
    else:
        st.session_state["_refresh_msg"] = (
            granary,
            "error",
            f"The refresh for **{granary}** failed (exit code {done[1]}); the earlier results are still shown.\n\n```\n{_log_tail()}\n```",
        )
    return True


def render_refresh_controls(target_granary: str) -> None:
    """No-op UI hook: manual refresh buttons are retired because Data Steward ingestion runs full 9-granary AutoML automatically."""
    if not _is_granary_refreshing(target_granary) and _finish_refresh(target_granary):
        st.rerun()


# ------------------------------------------------------------------------------------------------
# DATA STEWARD ROLE · Hybrid Event-Triggered + Checkpointed CSV Ingestion & Continuous AutoML
# ------------------------------------------------------------------------------------------------
def _run_full_automl_with_progress(progress_container=None, total_granaries: int = 9) -> bool:
    """Run pipeline/precompute_automl.py --force across all granaries and stream live [DONE] progress."""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.Popen(
        [sys.executable, str(_REFRESH_SCRIPT), "--force"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(_ROOT),
        text=True,
        bufsize=1,
        creationflags=creationflags,
    )
    completed = 0
    pbar = progress_container.progress(0.0, text="Starting parallel AutoML sweep across all granaries...") if progress_container else None
    tail_lines: list[str] = []

    if proc.stdout is not None:
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            tail_lines.append(line)
            if len(tail_lines) > 15:
                tail_lines.pop(0)
            if "[DONE]" in line:
                completed = min(completed + 1, total_granaries)
                if progress_container is not None:
                    progress_container.write(f"`{line}`")
                if pbar is not None:
                    pbar.progress(
                        completed / max(total_granaries, 1),
                        text=f"Completed {completed} / {total_granaries} granaries...",
                    )

    proc.wait()
    if pbar is not None and proc.returncode == 0:
        pbar.progress(1.0, text=f"Completed {total_granaries} / {total_granaries} granaries.")
    if proc.returncode != 0 and progress_container is not None:
        progress_container.error("AutoML sweep error:\n```\n" + "\n".join(tail_lines[-8:]) + "\n```")
    return proc.returncode == 0


def _inspect_uploaded_yield_diff(up_file, existing_df: pd.DataFrame) -> dict:
    """Auto-detect new (District, Year) pairs (Δ_fetch), yield revisions, and duplicates without burning satellite API calls."""
    try:
        raw = pd.read_excel(up_file) if up_file.name.endswith(".xlsx") else pd.read_csv(up_file)
    except Exception as exc:
        return {
            "errors": [f"Could not parse uploaded file: {exc}"],
            "new_pairs": [],
            "revised_pairs": [],
            "duplicates": [],
            "clean_df": pd.DataFrame(),
            "preview_df": pd.DataFrame(),
        }

    required = {"District", "Year", "Yield_kg_ha"}
    missing = required - set(raw.columns)
    if missing:
        return {
            "errors": [f"Missing required columns: {', '.join(sorted(missing))}"],
            "new_pairs": [],
            "revised_pairs": [],
            "duplicates": [],
            "clean_df": pd.DataFrame(),
            "preview_df": pd.DataFrame(),
        }

    clean = raw[["District", "Year", "Yield_kg_ha"]].dropna().copy()
    clean["District"] = clean["District"].astype(str).str.strip()
    clean["Year"] = pd.to_numeric(clean["Year"], errors="coerce")
    clean["Yield_kg_ha"] = pd.to_numeric(clean["Yield_kg_ha"], errors="coerce")

    errors = []
    if clean["Year"].isna().any() or clean["Yield_kg_ha"].isna().any():
        errors.append("Year and Yield_kg_ha must be numeric.")
    else:
        clean["Year"] = clean["Year"].astype(int)
        if (clean["Yield_kg_ha"] <= 0).any() or (clean["Yield_kg_ha"] > 15000).any():
            errors.append("Yield_kg_ha contains out-of-range values (expected 1–15,000 kg/ha).")

    if errors:
        return {
            "errors": errors,
            "new_pairs": [],
            "revised_pairs": [],
            "duplicates": [],
            "clean_df": clean,
            "preview_df": pd.DataFrame(),
        }

    existing_map = {}
    if not existing_df.empty and required.issubset(existing_df.columns):
        for _, r in existing_df[["District", "Year", "Yield_kg_ha"]].dropna().iterrows():
            existing_map[(str(r["District"]).strip(), int(r["Year"]))] = float(r["Yield_kg_ha"])

    new_pairs, revised_pairs, duplicates, preview_rows = [], [], [], []
    for _, r in clean.iterrows():
        key = (r["District"], int(r["Year"]))
        y_val = float(r["Yield_kg_ha"])
        if key not in existing_map:
            new_pairs.append(key)
            preview_rows.append(
                {
                    "District": key[0],
                    "Year": key[1],
                    "Yield_kg_ha": round(y_val, 1),
                    "Pipeline Action": "NEW -> Extract Satellite (Days 0–70) & Retrain",
                }
            )
        elif abs(existing_map[key] - y_val) > 1e-2:
            revised_pairs.append(key)
            preview_rows.append(
                {
                    "District": key[0],
                    "Year": key[1],
                    "Yield_kg_ha": round(y_val, 1),
                    "Pipeline Action": f"REVISION (was {existing_map[key]:,.0f}) -> Reuse Cached Satellite",
                }
            )
        else:
            duplicates.append(key)

    return {
        "errors": [],
        "new_pairs": new_pairs,
        "revised_pairs": revised_pairs,
        "duplicates": duplicates,
        "clean_df": clean,
        "preview_df": pd.DataFrame(preview_rows),
    }


def render_steward_ingestion_panel(D, allow_upload: bool = False) -> None:
    """Data Steward Role Panel: CSV dropzones locked by default (allow_upload=False) until live ingestion is enabled."""
    from acts_transfer.freshness_loop import check_data_freshness, load_update_log, merge_new_yield_csv, run_retraining_pipeline

    y_df = getattr(D, "yh", pd.DataFrame())
    fresh = check_data_freshness(y_df)
    log = load_update_log()
    pending = [r for r in log.get("requests", []) if r.get("status") == "PENDING_STEWARD"]
    total_granaries = len(getattr(D, "ALL", [])) or 9

    header = "🔒 Data Steward Ingestion & Closed-Loop Retraining (Locked)" if not allow_upload else "📥 Data Steward Ingestion & Closed-Loop Retraining (Auto-Detect CSV)"
    if pending:
        header += f" · 🔔 {len(pending)} Pending Manager Request(s)"

    with st.expander(header, expanded=False):
        if not allow_upload:
            st.info(
                "**Dataset Upload Locked:** Live CSV ingestion and automated retraining are currently disabled. "
                "The workspace is running on the verified baseline dataset."
            )

        if fresh.get("is_stale"):
            missing_str = ", ".join(map(str, fresh["missing_years"]))
            st.warning(
                f"**Freshness Notice:** Active dataset runs through **{fresh['max_data_year']}** "
                f"(system year **{fresh['current_year']}**, missing: **{missing_str}**)."
            )

        if pending:
            st.markdown("**Pending Requests from KPKM Programme Manager (`results/data_update_log.json`):**")
            for req in pending:
                st.info(
                    f"**[{req['id']} · {req['granary']}]** {', '.join(req.get('requested_items', []))} — "
                    f"{req.get('meeting_notes', '')}"
                )

        c_u1, c_u2 = st.columns([2, 1.3])
        with c_u1:
            up_yield = st.file_uploader(
                "1. Upload Annual Yield Dataset (.csv / .xlsx) — Locked",
                type=["csv", "xlsx"],
                disabled=not allow_upload,
                key="steward_auto_yield_up",
            )
        with c_u2:
            up_daily = st.file_uploader(
                "2. Optional Local Satellite Daily CSVs (polygon_daily/*.csv) — Locked",
                type=["csv"],
                accept_multiple_files=True,
                disabled=not allow_upload,
                key="steward_auto_daily_up",
            )

        if allow_upload and up_yield is not None:
            diff = _inspect_uploaded_yield_diff(up_yield, y_df)
            m1, m2, m3 = st.columns(3)
            m1.metric("New Satellite Extractions (Δ_fetch)", len(diff["new_pairs"]))
            m2.metric("Yield-Only Revisions", len(diff["revised_pairs"]))
            m3.metric("Duplicates Skipped", len(diff["duplicates"]))

            if diff["errors"]:
                st.error("Validation blocked: " + " | ".join(diff["errors"]))
            elif not diff["new_pairs"] and not diff["revised_pairs"] and not up_daily:
                st.info("Uploaded file matches existing records — no satellite extraction or retraining required.")
            else:
                if not diff["preview_df"].empty:
                    st.dataframe(diff["preview_df"], width="stretch", hide_index=True)

                if st.button(
                    "🚀 Validate & Run Pipeline (Rebuild Dataset + Full 9-Granary AutoML)",
                    type="primary",
                    width="stretch",
                    disabled=not allow_upload,
                    key="steward_validate_run_btn",
                ):
                    with st.status("Executing checkpointed ingestion & full 9-granary AutoML...", expanded=True) as status:
                        st.write("Step 1/3 · Merging yield records & rebuilding feature datasets (updating mtime_token)...")
                        merge_new_yield_csv(diff["clean_df"], data_dir=D.DATA_DIR)
                        if up_daily:
                            pdir = _ROOT / "polygon_daily"
                            pdir.mkdir(exist_ok=True)
                            for f in up_daily:
                                (pdir / f.name).write_bytes(f.getvalue())

                        ok_data, step_logs = run_retraining_pipeline(automl_note="Rebuilt datasets from Data Steward upload")
                        if not ok_data:
                            status.update(label="Dataset rebuild failed.", state="error")
                            st.error("Dataset rebuild error:\n" + "\n".join(step_logs))
                            return

                        st.write("Step 2/3 · Running full parallel AutoML sweep across all 9 granaries (~90s)...")
                        ok_automl = _run_full_automl_with_progress(progress_container=status, total_granaries=total_granaries)
                        if not ok_automl:
                            status.update(label="AutoML sweep encountered an error.", state="error")
                            return

                        st.write("Step 3/3 · Clearing caches and loading fresh Step 4 verdicts...")
                        status.update(label="Pipeline complete! Review updated Step 4 verdicts below.", state="complete")

                    _reload_cached_results()
                    st.session_state["validator_step"] = 4
                    st.rerun()

def _get_automl_payload(D) -> dict:
    """Retrieve (or compute via @st.cache_data) the full AutoML & causal audit payload for the active granary."""
    target = st.session_state["granary"]
    if "automl_summary" in st.session_state and st.session_state["automl_summary"].get("target") == target:
        return st.session_state["automl_summary"]
    ds = D.DATA_DIR / "transfer_stage_dataset.csv"
    dy = D.DATA_DIR / "transfer_yield_dataset.csv"
    mtime = (ds if ds.exists() else dy).stat().st_mtime if (ds.exists() or dy.exists()) else 0.0
    auto = run_full_automl_for_granary(target, str(D.DATA_DIR), mtime, allow_stale=True)
    st.session_state["automl_summary"] = auto
    st.session_state["tl_bundle"] = auto["bundle"]
    st.session_state["tl_assessment"] = auto["assessment"]
    st.session_state["tl_eval_results"] = auto["eval_results"]
    st.session_state["tl_choice"] = auto["choice"]
    return auto


def _render_winner_cards(scores: dict):
    """Top 4 AutoML Winner KPI Cards (Donor Pool, Model, Feature Set, Scaling/Alignment)."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Donor Pool", scores["donor_winner"][0], f"Score: {scores['donor_winner'][1]:.1f}%")
    c2.metric("Best Model", scores["model_winner"][0], f"Score: {scores['model_winner'][1]:.1f}%")
    c3.metric("Best Feature Set", scores["feature_winner"][0], f"Score: {scores['feature_winner'][1]:.1f}%")
    c4.metric("Best Scaling / Alignment", scores["scaling_winner"][0], f"Score: {scores['scaling_winner'][1]:.1f}%")


# ------------------------------------------------------------------------------------------------
# STEP 2 · Causal Feature & Region Similarity Check (AutoML-Assisted)
# ------------------------------------------------------------------------------------------------
def render_domain_audit(D):
    target = st.session_state["granary"]
    auto = _get_automl_payload(D)
    scores = auto["scores"]
    donor_df = auto["donor_table"]
    asmt = auto["assessment"]

    c_tgt, c_pool = st.columns([1.2, 2.2])
    with c_tgt:
        st.selectbox(
            "Target Granary (locked to Selected Granary)",
            options=[target],
            index=0,
            disabled=True,
            key=f"locked_target_s2_{target}",
            help="Controlled by SELECTED GRANARY in the left navigation bar.",
        )
    with c_pool:
        st.info(
            f"**AutoML Domain & Causal Audit Active for {target}:** "
            f"All {len(donor_df)} other granaries were screened automatically. "
            f"Outliers with Domain AUC >= 0.90 are excluded from the transfer pool."
        )

    _render_winner_cards(scores)

    t_donors, t_causal, t_harm = st.tabs([
        "A. Donor Granary Similarity Scores (%)",
        "B. Causal Feature Identification & Refutation",
        "C. Harmonisation & Shift Diagnostics",
    ])

    with t_donors:
        st.markdown("**Pairwise Climate & Calendar Divergence Screen (0–100% Similarity Score)**")
        st.dataframe(donor_df, hide_index=True, width="stretch")

    with t_causal:
        ds = D.DATA_DIR / "transfer_stage_dataset.csv"
        dy = D.DATA_DIR / "transfer_yield_dataset.csv"
        y_df = pd.read_csv(ds if ds.exists() else dy)
        parts = partition_causal_variables(y_df)
        p1, p2, p3 = st.columns(3)
        p1.markdown(f"**Confounders ($X$):** `{', '.join(parts['X']) or 'n/a'}`")
        p2.markdown(f"**Actionable Practices ($T$):** `{', '.join(parts['T']) or 'n/a'}`")
        p3.markdown(f"**Mediators ($M$):** `{', '.join(parts['M']) or 'n/a'}`")

        t_choices = parts["T"] or ["C1_LSWI_rep"]
        chosen_t = st.selectbox("Inspect Double Machine Learning (DML) treatment lever", t_choices, key=f"dml_t_{target}")
        dml = dml_estimate_and_refute(y_df, treatment_col=chosen_t)
        m1, m2, m3 = st.columns(3)
        m1.metric("Deconfounded Effect (τ)", f"{dml['tau']:+.1f} kg/ha")
        m2.metric("Standard Error", f"±{dml['se']:.1f} kg/ha" if pd.notna(dml["se"]) else "n/a")
        m3.metric("Propensity Overlap Sample", f"{dml['n_overlap']} / {dml.get('n_total', dml['n_overlap'])}")
        if dml["refutations"]:
            ref_df = pd.DataFrame(dml["refutations"])
            ref_df["new_effect"] = ref_df["new_effect"].round(1)
            ref_df["passed"] = ref_df["passed"].map(lambda v: "PASS" if v else "CHECK")
            ref_df.columns = ["Refutation Check", "Refuted Effect (kg/ha)", "Status"]
            st.dataframe(ref_df, hide_index=True, width="stretch")

    with t_harm:
        st.markdown(f"**Overall Readiness Verdict:** `{getattr(asmt, 'verdict', 'READY')}`")
        diag = getattr(asmt, "summary", None)
        if isinstance(diag, pd.DataFrame) and len(diag):
            st.dataframe(diag, hide_index=True, width="stretch")
        else:
            st.write(
                f"Harmonised feature set ({len(auto['bundle'].features)} variables) validated with fold-safe RobustScaler "
                f"and optical quality weights $w_t = \\max(0.2, \\text{{Optical\\_Valid\\_Fraction}}_t)$."
            )

    st.markdown("### Pipeline Configuration (Pre-Selected by AutoML)")
    override = st.toggle("Unlock manual override (default uses AutoML top-scoring configuration)", value=False, key=f"ov_s2_{target}")
    st.checkbox(
        f"Exclude climate/calendar outliers (Domain AUC ≥ 0.90) — **Score: {scores['donor_winner'][1]:.1f}%**",
        value=True,
        disabled=not override,
        key=f"chk_outlier_{target}",
    )
    st.checkbox(
        f"Use {scores['feature_winner'][0]} — **Score: {scores['feature_winner'][1]:.1f}%**",
        value=True,
        disabled=not override,
        key=f"chk_feat_{target}",
    )
    st.checkbox(
        f"Apply {scores['scaling_winner'][0]} — **Score: {scores['scaling_winner'][1]:.1f}%**",
        value=True,
        disabled=not override,
        key=f"chk_scale_{target}",
    )


# ------------------------------------------------------------------------------------------------
# STEP 3 · AutoML Model Training & Backtesting (Construct & Analyze)
# ------------------------------------------------------------------------------------------------
def render_benchmark_audit(D):
    target = st.session_state["granary"]
    auto = _get_automl_payload(D)
    scores = auto["scores"]
    choice = auto["choice"]
    eval_res = auto["eval_results"]

    _render_winner_cards(scores)

    st.markdown("### AutoML Transfer & Causal Model Leaderboard (0–100% Score)")
    st.dataframe(auto["model_table"], hide_index=True, width="stretch")

    t_fwd, t_loso, t_imp = st.tabs([
        "Expanding-Window Forward Validation (Primary)",
        "Leave-One-Season-Out (LOSO) Check",
        "Paired Improvement & Bootstrap CI",
    ])

    with t_fwd:
        fwd = eval_res.get("forward")
        if fwd is not None and len(fwd.metrics):
            m_df = fwd.metrics.copy()
            m_df["Method"] = m_df["method"].map(lambda m: LABELS.get(m, m))
            cols = [c for c in ("Method", "MAE", "RMSE", "R2", "n_test") if c in m_df.columns]
            st.dataframe(m_df[cols].round(4), hide_index=True, width="stretch")

    with t_loso:
        loso = eval_res.get("loso")
        if loso is not None and len(loso.metrics):
            l_df = loso.metrics.copy()
            l_df["Method"] = l_df["method"].map(lambda m: LABELS.get(m, m))
            cols = [c for c in ("Method", "MAE", "RMSE", "R2", "n_test") if c in l_df.columns]
            st.dataframe(l_df[cols].round(4), hide_index=True, width="stretch")

    with t_imp:
        if isinstance(choice.get("table"), pd.DataFrame) and len(choice["table"]):
            st.dataframe(choice["table"], hide_index=True, width="stretch")

    st.info(choice["statement"])


# ------------------------------------------------------------------------------------------------
# STEP 4 · Model Approval & Recommendation Check (Execute)
# ------------------------------------------------------------------------------------------------
def _evidence(auto, choice, gate_passed, method):
    kg = float(getattr(auto["bundle"], "y_to_kg", 1.0))
    t = choice["table"]
    row = t[t["method"] == method]
    donors = auto["donor_table"]
    ev = dict(
        gate_passed=bool(gate_passed),
        model_key=method,
        model_label=LABELS.get(method, method) if method else "none",
        source=", ".join(donors[donors["Status"] == "Included"]["Donor Granary"]),
        validation="forward validation (expanding window)",
        token=choice.get("token"),
        automl_scores=auto["scores"],
    )
    if len(row):
        r = row.iloc[0]
        ev.update(
            delta_nt=float(r["Delta_NT (RMSE)"]),
            delta_nt_kg_ha=float(r["Delta_NT (RMSE)"]) * kg,
            baseline=r["Delta_NT baseline"],
            domain_ok=bool(r["Criterion 1 domain relevance"]),
            seasons_compared=int(r["seasons compared"]),
            statistically_supported=bool(r["statistically supported"]),
            rmse_common_kg_ha={k: float(v) * kg for k, v in r["RMSE vs all baselines (common seasons)"].items()},
        )
        ge = gate_evidence(auto, method)
        if ge.get("available"):
            ev["evidence_criteria"] = {
                k: ge[k]
                for k in (
                    "c0_delta_nt_negative",
                    "c1_lower_mae",
                    "c2_interval_excludes_zero",
                    "c3_loso_win",
                    "c4_enough_seasons",
                    "domain_ok",
                    "held_out_seasons",
                    "min_seasons",
                    "mae_change_kg_ha",
                    "ci95_low_kg_ha",
                    "ci95_high_kg_ha",
                    "best_baseline",
                    "strength",
                    "reason",
                    "nested_delta_nt_kg_ha",
                    "nested_method",
                    "nested_ok",
                    "nested_gate",
                )
            }
    return ev


def _record_decision(D, auto, choice, target, decision, who, notes, gate_passed):
    from acts_transfer.publish import build_published_from_automl
    from acts_transfer.sweep import clean_physical_ranges

    ds, dy = D.DATA_DIR / "transfer_stage_dataset.csv", D.DATA_DIR / "transfer_yield_dataset.csv"
    Y, _ = clean_physical_ranges(pd.read_csv(str(ds if ds.exists() else dy)))
    method = choice["selected"] if decision == "APPROVE" else (choice.get("best_transfer") or choice["selected"])
    try:
        published = build_published_from_automl(auto, Y, target)
        return gov.record_signoff(target, decision, who or "Data Steward", notes, _evidence(auto, choice, gate_passed, method), published)
    except Exception as ex:
        st.error(f"The sign-off was not recorded because the forecast tables could not be built: {ex}")
        return None


def render_preapproval_diagnostics(bundle, kg: float):
    st.markdown("#### Pre-approval diagnostics: does the weather signal earn its place?")

    ab = compute_weather_ablation(bundle, kg)
    st.markdown("**(a) Weather-feature ablation** (forward validation: real weather / satellite inputs vs. zeroed anomalies)")
    if ab["n_folds"] == 0 or pd.isna(ab["weather_gain"]):
        st.info("Not enough earlier seasons to run the ablation (at least 3 target seasons are needed).")
    else:
        gain = float(ab["weather_gain"])
        a1, a2, a3 = st.columns(3)
        a1.metric("Error with real weather", f"{ab['err_real']:,.0f} kg/ha")
        a2.metric("Error with zeroed anomalies", f"{ab['err_ablated']:,.0f} kg/ha")
        a3.metric("Saved by real weather", f"{gain:+,.0f} kg/ha", "weather helps" if gain > 0 else "weather does not help", delta_color="normal" if gain > 0 else "inverse")
        if gain <= 0:
            st.warning(
                f"Over {ab['n_folds']} held-out seasons the forecasts were no better with the real weather and satellite inputs than with an average season. "
                "A transfer model that leans on these inputs is not supported by this evidence, whatever the safeguard gate says."
            )
        else:
            st.caption(
                f"RMSE over {ab['n_folds']} held-out seasons; {len(ab['weather_features'])} weather / satellite features; "
                "only the held-out season's inputs differ between the two runs."
            )

    fs = compute_few_shot_scarcity(bundle, kg, shots=(2, 3, 4))
    st.markdown("**(b) Few-shot data scarcity** (forecast error when the target granary has only n seasons of history)")
    if fs.empty:
        st.info("Not enough target seasons for the few-shot comparison.")
    else:
        st.dataframe(
            fs,
            hide_index=True,
            width="stretch",
            column_config={
                "Local-Only Error (kg/ha)": st.column_config.NumberColumn(format="%.0f"),
                "Transfer Error (kg/ha)": st.column_config.NumberColumn(format="%.0f"),
                "Transfer Advantage": st.column_config.NumberColumn("Transfer Advantage (kg/ha, + = transfer better)", format="%+.0f"),
                "Test Seasons": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption(
            "All rows are scored on the same held-out seasons. Each forecast uses only the n most recent earlier target seasons "
            "and donor seasons that came before it."
        )

    dr = compute_weather_driver_attribution(bundle, kg, top_k=6)
    st.markdown("**(c) Top weather / satellite drivers** (yield change in kg/ha for a +1 SD shift of the driver)")
    if dr.empty:
        st.info("No weather or satellite drivers are available for attribution.")
    else:
        d = dr.iloc[::-1]
        fig = go.Figure(
            go.Bar(
                x=d["Impact (kg/ha per +1 SD)"],
                y=d["Driver"],
                orientation="h",
                marker_color=["#2ca02c" if v >= 0 else "#d62728" for v in d["Impact (kg/ha per +1 SD)"]],
                hovertemplate="%{y}: %{x:+,.0f} kg/ha per +1 SD<extra></extra>",
            )
        )
        fig.update_layout(xaxis_title="Yield change per +1 SD shift (kg/ha)", yaxis_title=None, height=60 + 42 * len(d), margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Descriptive, not causal: correlated drivers share their effect, and each slope comes from only a few seasons.")


def render_gate_evidence(ge: dict, gate_passed: bool) -> None:
    st.markdown("#### Evidence behind the gate")
    if not ge.get("available"):
        st.info("No transfer candidate was evaluated, so there is no evidence to show.")
        return
    mark = lambda ok: "✅" if ok else "❌"  # noqa: E731
    ci = (
        f"{ge['ci95_low_kg_ha']:+,.0f} to {ge['ci95_high_kg_ha']:+,.0f} kg/ha (mean {ge['mae_change_kg_ha']:+,.0f})"
        if pd.notna(ge["ci95_low_kg_ha"])
        else f"not testable: {ge['held_out_seasons']} paired season(s)"
    )
    rows = [
        ("0", "Safeguard gate: Δ_NT < 0 against the best local baseline", mark(ge["c0_delta_nt_negative"]), f"{ge['delta_nt_kg_ha']:+,.1f} kg/ha vs {ge['delta_nt_baseline']}"),
        ("1", f"Lower MAE than the best baseline ({ge['best_baseline']}; the best of all six, donor-trained ones included)", mark(ge["c1_lower_mae"]), ""),
        ("2", "95% interval of the error change excludes zero", mark(ge["c2_interval_excludes_zero"]), ci),
        ("3", "Also better under leave-one-season-out", mark(ge["c3_loso_win"]), ""),
        ("4", f"At least {ge['min_seasons']} held-out seasons", mark(ge["c4_enough_seasons"]), f"{ge['held_out_seasons']} paired ({ge['common_seasons']} common to all baselines)"),
        ("-", "Agro-climatic domain compatible with the donors", mark(ge["domain_ok"]), ""),
        (
            "-",
            "Nested check: method chosen on the earlier seasons, Δ_NT < 0 on the later ones" + (" (a gate condition)" if ge.get("nested_gate") else " (shown only)"),
            mark(ge["nested_ok"]) if ge.get("nested_available") else "—",
            (
                f"{ge['nested_delta_nt_kg_ha']:+,.1f} kg/ha with {LABELS.get(ge['nested_method'], ge['nested_method'])}; chosen on {ge['nested_select_seasons']} seasons, judged on {ge['nested_hold_seasons']}"
                if ge.get("nested_available")
                else f"too few seasons ({ge.get('nested_hold_seasons', 0)} later)"
            ),
        ),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["#", "Criterion", "Met", "Measured"]), hide_index=True, width="stretch")
    if gate_passed and ge["strength"] == "supported":
        st.success("The gate passed and the improvement is statistically supported.")
    elif gate_passed:
        st.warning(
            f"**The gate passed, but the evidence is thin:** {ge['reason']} Δ_NT < 0 is a safety check, not proof that transfer improves the forecast. "
            "Approve only with ground verification, or record the local baseline."
        )
    else:
        st.info("The safeguard gate triggered, so the local baseline is the model of record. The evidence above is shown for reference.")
    if gate_passed and ge.get("nested_available") and not ge["nested_ok"]:
        st.warning(
            f"**Selection check failed:** choosing the method on the earlier seasons only, it would have been {ge['nested_delta_nt_kg_ha']:+,.0f} kg/ha worse than the best local baseline on the later "
            "seasons. Part of the gain above may be luck from picking the best of several candidates on the same seasons it is judged on."
        )


def render_validator_signoff(D):
    target = st.session_state["granary"]
    auto = _get_automl_payload(D)
    scores = auto["scores"]
    choice = auto["choice"]
    sg = choice["safeguard"]
    gate_passed = bool(sg.get("gate_passed", False))
    kg = float(getattr(auto["bundle"], "y_to_kg", 1000.0))

    _render_winner_cards(scores)

    st.markdown("### Pre-Approval Validation: Donor Pool & Weather Outlook")

    if "donor_pool_table" in auto and not auto["donor_pool_table"].empty:
        st.markdown("**1. Donor Pool Size Competition (Single Closest Donor vs. Pooled)**")
        st.dataframe(auto["donor_pool_table"], width="stretch", hide_index=True)

    st.markdown("**2. Pre-Approval Yield Outlook vs. Same-as-Last-Year Baseline**")
    win_method_id = str(auto["model_table"].iloc[0]["Method ID"])
    pred_df = auto["eval_results"]["forward"].predictions
    sub_pred = pred_df[pred_df["method"].isin([win_method_id, "persistence"])].copy()

    if not sub_pred.empty:
        piv = sub_pred.pivot(index="group", columns="method", values="y_pred") * kg
        actuals = sub_pred[sub_pred["method"] == win_method_id].set_index("group")["y_true"] * kg

        rmse_model = float(((piv[win_method_id] - actuals) ** 2).mean() ** 0.5) if win_method_id in piv else float("nan")
        rmse_pers = float(((piv["persistence"] - actuals) ** 2).mean() ** 0.5) if "persistence" in piv else float("nan")
        delta_kg = rmse_model - rmse_pers

        c1, c2, c3 = st.columns(3)
        c1.metric("Pre-Approval Model RMSE", f"{rmse_model:,.0f} kg/ha", f"{delta_kg:+,.0f} kg/ha vs Persistence", delta_color="inverse")
        c2.metric("Same-as-Last-Year RMSE", f"{rmse_pers:,.0f} kg/ha")
        c3.metric("Safeguard Gate", "PASS" if gate_passed else "FALLBACK")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actuals.index, y=actuals.values, mode="lines+markers", name="Actual Yield (kg/ha)", line=dict(color="#1f77b4", width=3)))
        if win_method_id in piv.columns:
            fig.add_trace(go.Scatter(x=piv.index, y=piv[win_method_id], mode="lines+markers", name=f"Candidate Outlook ({LABELS.get(win_method_id, win_method_id)})", line=dict(color="#2ca02c", width=2, dash="dash")))
        if "persistence" in piv.columns:
            fig.add_trace(go.Scatter(x=piv.index, y=piv["persistence"], mode="lines+markers", name="Same as Last Year (Persistence)", line=dict(color="#7f7f7f", width=2, dash="dot")))
        fig.update_layout(xaxis_title="Season Year", yaxis_title="Yield (kg/ha)", height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Negative-Transfer Safeguard Gate & Governance Decision")
    g1, g2, g3 = st.columns(3)
    g1.metric("Candidate Transfer Winner", LABELS.get(choice.get("best_transfer", ""), str(choice.get("best_transfer", "n/a"))))
    d_nt = sg.get("delta_nt")
    g2.metric("Δ_NT (RMSE vs Local Baseline)", f"{d_nt:+.3f} t/ha" if d_nt is not None and pd.notna(d_nt) else "n/a")
    g3.metric("Safeguard Gate Status", "PASS (Δ_NT < 0)" if gate_passed else "TRIGGERED (Local Fallback)")

    st.write(choice["statement"])

    reg = gov.load()
    cur_entry = gov.effective(target, reg)
    st.caption(f"Current registry status for **{target}**: `{cur_entry['status']}`")

    ev_method = choice["selected"] if gate_passed and choice.get("transfer_supported") else choice.get("best_transfer")
    render_gate_evidence(gate_evidence(auto, ev_method), gate_passed)

    render_preapproval_diagnostics(auto["bundle"], kg)

    c_v1, c_v2 = st.columns([1, 2])
    validator_name = c_v1.text_input("Validator / Data Steward Name", value="Model Validator & Data Steward", key="so_name")
    notes = c_v2.text_input("Governance & Audit Notes", placeholder="Optional sign-off note for KPKM Programme Manager", key="so_notes")

    b_app, b_rej = st.columns(2)
    with b_app:
        if st.button(
            "✅ Approve Transfer Model for KPKM Workspace",
            key="so_approve",
            disabled=not gate_passed,
            width="stretch",
        ):
            if _record_decision(D, auto, choice, target, "APPROVE", validator_name, notes, True) is None:
                return
            st.success(f"Approved **{LABELS.get(choice['selected'], choice['selected'])}** for **{target}**.")
            st.rerun()

    with b_rej:
        if st.button(
            "🛡️ Record Local Baseline Fallback (Reject Transfer)",
            key="so_reject",
            width="stretch",
        ):
            if _record_decision(D, auto, choice, target, "REJECT", validator_name, notes, gate_passed) is None:
                return
            st.warning(f"Recorded Local Baseline Fallback (**{LABELS.get(choice['reference'], choice['reference'])}**) for **{target}**.")
            st.rerun()