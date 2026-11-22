"""
run_benchmark_sweep.py -- PACE V4 Step 1 (Discover): transfer MADA to every other granary under identical out-of-fold splits and pool the paired error differences.

    python pipeline/run_benchmark_sweep.py --data data --out data

Writes data/benchmark_sweep_pairs.csv (one row per held-out granary-year and method), benchmark_sweep_pooled.csv (pooled Delta_NT / Delta_DS with bootstrap CI and Wilcoxon),
benchmark_sweep_granary.csv (per-granary RMSE and Delta_NT, with domain descriptors) and benchmark_sweep_suitability.csv. See acts_transfer/sweep.py for the protocol.
"""
import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from acts_transfer import Config                       # noqa: E402
from acts_transfer.sweep import run_sweep               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data")
    ap.add_argument("--source", default="MADA")
    a = ap.parse_args()
    Y = pd.read_csv(os.path.join(a.data, "transfer_stage_dataset.csv"))
    F = pd.read_csv(os.path.join(a.data, "transfer_decade_features.csv"))
    t = time.time()
    res = run_sweep(Y, F, a.source, Config(), progress=lambda i, n, g: print(f"[{i + 1}/{n}] {g}", flush=True))
    for k in ("pairs", "pooled", "granary", "suitability"):
        res[k].round(4).to_csv(os.path.join(a.out, f"benchmark_sweep_{k}.csv"), index=False)
    print(f"done in {time.time() - t:.0f}s: {len(res['pairs'])} method-pairs over {res['meta']['n_targets']} target granaries")
    p = res["pooled"]
    print(p[(p.strategy == "loso") & (p.comparison == "Delta_NT")][["method", "n_pairs", "mean_diff_kg_ha", "ci95_low", "ci95_high", "wilcoxon_p_less", "pct_pairs_transfer_better", "passed"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
