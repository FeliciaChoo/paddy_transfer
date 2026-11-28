"""Pre-compute the AutoML donor-pool & feature-regime sweep so the dashboard opens instantly.

    python pipeline/precompute_automl.py                      # everything that is out of date (nothing is recomputed if it is all current)
    python pipeline/precompute_automl.py --only KETARA        # just one granary (repeat --only for several); the others are kept as they are
    python pipeline/precompute_automl.py --force              # recompute everything
    python pipeline/precompute_automl.py --workers 4          # number of processes (default: one per granary to do, at most the CPU count)

Fast because: (1) it is incremental, a granary whose data and code are unchanged is reused; (2) the file is saved after EVERY granary, so an interrupted run keeps what
it finished and the dashboard can already use it; (3) each worker is pinned to one BLAS thread, because these models fit thousands of tiny ridge regressions and several
processes each starting a many-threaded BLAS fight for the same cores and run slower than one thread each.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):      # before numpy is imported (also in the spawned worker processes)
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import json
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))      # run from any directory or IDE: the project root must be importable (also in the worker processes)

warnings.filterwarnings("ignore")


def _compute_one(granary: str, data_dir_str: str, mtime_token: float) -> tuple[str, dict]:
    from acts_transfer.automl_runner import _compute_automl_for_granary_uncached
    return granary, _compute_automl_for_granary_uncached(granary, data_dir_str, mtime_token)


def plan_work(blob, mtime_token: float, ctoken: str, granaries: list[str], only: list[str] | None = None, force: bool = False):
    """What to reuse and what to compute. Returns (kept payloads, their code stamps, granaries to compute).
    Payloads built from other data are dropped. Payloads built by other code are kept (the dashboard shows them flagged as older) unless they are being recomputed."""
    kept, stamps = {}, {}
    if isinstance(blob, dict) and abs(float(blob.get("mtime_token", -1.0)) - float(mtime_token)) < 1e-3:
        for g, payload in blob.get("granaries", {}).items():
            kept[g] = payload
            stamps[g] = blob.get("built_with", {}).get(g, blob.get("code_token"))
    wanted = list(only) if only else list(granaries)
    unknown = [g for g in wanted if g not in granaries]
    if unknown:
        raise ValueError(f"unknown granary: {unknown}; known: {granaries}")
    todo = [g for g in wanted if force or stamps.get(g) != ctoken]
    return kept, stamps, todo


def _summary_row(g: str, auto: dict) -> dict:
    top = auto["donor_pool_table"].iloc[0]
    sg = auto["choice"]["safeguard"]
    return {
        "Granary": g,
        "Verdict": "PASS" if bool(sg["gate_passed"]) else "FALLBACK",
        "Delta_NT (kg/ha)": round(float(sg["delta_nt"]) * 1000.0, 1),
        "Feature Regime": str(top["Feature Regime"]),
        "Winning Pool": str(top["Donor Pool Candidate"]),
        "Winning Model": str(top["Best Method"]),
        "Fwd RMSE (kg/ha)": float(top["Best Forward RMSE (kg/ha)"]),
    }


def _save(out_dir: Path, mtime_token: float, ctoken: str, payloads: dict, stamps: dict) -> None:
    """write the cache and the JSON summary atomically (a crash never leaves a half-written file)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    pkl, tmp = out_dir / "automl_precomputed.pkl", out_dir / "automl_precomputed.pkl.tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"mtime_token": mtime_token, "code_token": ctoken, "built_with": dict(stamps), "granaries": payloads}, f)
    os.replace(tmp, pkl)
    rows = pd.DataFrame([_summary_row(g, a) for g, a in payloads.items()])
    if len(rows):
        rows = rows.sort_values(["Verdict", "Delta_NT (kg/ha)"], ascending=[False, True])
    js, jtmp = out_dir / "automl_granary_summary.json", out_dir / "automl_granary_summary.json.tmp"
    jtmp.write_text(json.dumps({"dataset_mtime": mtime_token, "summary": rows.to_dict(orient="records")}, indent=2), encoding="utf-8")
    os.replace(jtmp, js)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=[], metavar="GRANARY", help="compute only this granary (repeatable)")
    ap.add_argument("--force", action="store_true", help="recompute even if the stored result is current")
    ap.add_argument("--workers", type=int, default=0, help="worker processes (default: as many as granaries to compute, at most the CPU count)")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    stage_p, yield_p = data_dir / "transfer_stage_dataset.csv", data_dir / "transfer_yield_dataset.csv"
    src_p = stage_p if stage_p.exists() else yield_p
    mtime_token = src_p.stat().st_mtime
    granaries = sorted(pd.read_csv(src_p)["District"].unique())

    from acts_transfer.automl_runner import PRECOMPUTED_PATH, code_token
    ctoken = code_token()
    blob = None
    if PRECOMPUTED_PATH.exists():
        try:
            with open(PRECOMPUTED_PATH, "rb") as f:
                blob = pickle.load(f)
        except Exception:
            blob = None
    payloads, stamps, todo = plan_work(blob, mtime_token, ctoken, granaries, args.only, args.force)
    out_dir = root / "results"
    print(f"{len(granaries)} granaries; {len(todo)} to compute, {len([g for g in granaries if g in payloads and g not in todo])} reused: {todo or 'everything is up to date'}", flush=True)

    if todo:
        workers = args.workers or min(len(todo), os.cpu_count() or 1)
        print(f"computing with {workers} worker process(es), one BLAS thread each...", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_compute_one, g, str(data_dir), mtime_token): g for g in todo}
            for fut in as_completed(futures):
                g, auto = fut.result()
                payloads[g], stamps[g] = auto, ctoken
                _save(out_dir, mtime_token, ctoken, payloads, stamps)                  # checkpoint: usable right now, even if a later granary fails
                row = _summary_row(g, auto)
                print(f"  [DONE] {g:26s} -> {row['Verdict']:8s} | Delta_NT: {row['Delta_NT (kg/ha)']:+6.1f} kg/ha | {row['Feature Regime']}   (elapsed {time.perf_counter() - t0:.0f}s)", flush=True)
    else:
        _save(out_dir, mtime_token, ctoken, payloads, stamps)

    print(f"\nDone in {time.perf_counter() - t0:.1f}s.")
    print(f"Dashboard cache saved : {out_dir / 'automl_precomputed.pkl'}")
    print(f"JSON summary saved    : {out_dir / 'automl_granary_summary.json'}\n")
    print(pd.DataFrame([_summary_row(g, a) for g, a in payloads.items()]).to_string(index=False))


if __name__ == "__main__":
    main()
