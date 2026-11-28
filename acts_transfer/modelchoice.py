"""Which model is the MODEL OF RECORD after evaluation? Transfer learning replaces the best baseline only when the evidence supports it.

This is the gate between evaluation and everything downstream (yield prediction, future outlook, the advisory's yield evidence). A transfer
method becomes the selected model only if ALL of these hold; otherwise the best baseline stays selected and the interface says so:
  1  its forward-validation MAE is lower than the best baseline's (the best baseline is chosen by forward MAE, so it is the hardest comparison)
  2  that improvement is statistically supported: the bootstrap interval over held-out seasons excludes zero
  3  it is also better than the best baseline under leave-one-season-out (the result does not depend on how the data were split)
  4  there are enough held-out seasons (configurable, default 5) for the interval to mean anything
  5  the run was not an override of a NOT RECOMMENDED verdict (an override is research, not a basis for predictions)
Negative-transfer safeguard (methodology step E2):  Delta_NT = RMSE_transfer - min(RMSE_target_only, RMSE_persistence), on the primary validation.
Delta_NT < 0 is criterion 0. With cfg.nt_gate = "delta_nt" (default, as in the methodology) criterion 0 and criterion 5 decide deployment and criteria 1 to 4
are still computed and shown as the strength of the evidence; with "delta_nt+stats" all of them must hold. Either way, when no transfer method passes,
the model of record is the best LOCAL baseline (target-only ridge, persistence, season trend or target mean): never a source-trained model.
Forward validation (earlier seasons only) is the primary check because it cannot interpolate a trend; if it produced no folds the leave-one-out
result is used and the statement says so.
"""
import hashlib

import numpy as np
import pandas as pd

from .config import Config
from .evaluation import BASELINES

LOCAL_BASELINES = ("target_only", "persistence", "season_trend", "target_mean")   # baselines that use the target granary's own history only
from .methods import LABELS


def choose(results: dict, cfg: Config, override_used: bool = False, sweep: dict = None, domain: dict = None):
    """sweep (optional, PACE V4 evidence-first qualification): {"required": bool, "passed": {"forward": {method: bool}, "loso": {method: bool}}, "detail": {...}}.
    When required, a transfer method is deployable only if the benchmark sweep passed for it under the primary validation, in addition to the safeguard (Delta_NT < 0).
    domain (PACE V8 criterion 1, Domain Relevance): {"ok": bool, "reason": str}. False when the source pool holds a granary that the divergence screen flags (KETARA).
    Criterion 2 (Baseline Pass): Delta_NT = RMSE_transfer - min(RMSE_season_trend, RMSE_target_only, RMSE_persistence) < 0, with every RMSE taken over the SEASONS COMMON to the
    transfer method and all three baselines (the season trend needs 3 training years and so is scored on fewer seasons than the others; comparing different seasons is not fair)."""
    fwd, loso = results.get("forward"), results.get("loso")
    primary_name = "forward"
    primary_key = "forward"
    primary = fwd
    if primary is None or primary.metrics is None or len(primary.metrics) == 0:
        primary, primary_name, primary_key = loso, "leave-one-season-out", "loso"
    sw_req = bool(sweep and sweep.get("required"))
    mp = primary.metrics.set_index("method").MAE
    _pm = primary.metrics.set_index("method")
    rm = _pm["RMSE"] if "RMSE" in _pm.columns else mp                  # hand-built metrics without RMSE fall back to MAE
    ml = loso.metrics.set_index("method").MAE if loso is not None and len(loso.metrics) else pd.Series(dtype=float)
    base_p = [b for b in BASELINES if b in mp.index]
    ref_all = min(base_p, key=lambda b: mp[b])                 # the hardest comparison for the statistical criteria (includes source-only and pooled)
    local_p = [b for b in LOCAL_BASELINES if b in mp.index] or base_p
    ref = min(local_p, key=lambda b: mp[b])                    # the fallback model of record: best LOCAL baseline
    nt_ref = [b for b in ("season_trend", "target_only", "persistence") if b in rm.index]
    pred = getattr(primary, "predictions", None)
    have_pred = isinstance(pred, pd.DataFrame) and len(pred) and {"method", "group", "abs_err"} <= set(pred.columns)
    dom_ok = True if domain is None else bool(domain.get("ok", True))

    def _common(m):
        """RMSE of the transfer method and of every Delta_NT baseline over the seasons all of them were scored on, and how many seasons that is"""
        if not have_pred or not nt_ref or m not in set(pred.method):
            return {x: float(rm[x]) for x in [m] + nt_ref}, int(primary.metrics.set_index("method").n_test.get(m, 0)) if "n_test" in primary.metrics.columns else 0
        gs = set(pred[pred.method == m].group)
        for b in nt_ref:
            gs &= set(pred[pred.method == b].group)
        sub = pred[pred.group.isin(gs)]
        out = {x: float((sub[sub.method == x].abs_err ** 2).mean() ** 0.5) for x in [m] + nt_ref}
        return out, len(gs)
    base_l = [b for b in BASELINES if b in ml.index]
    trans = [m for m in mp.index if m not in BASELINES]
    rows = []
    for m in trans:
        imp = primary.improvement[(primary.improvement.method == m) & (primary.improvement.baseline == ref_all)]
        verdict = imp.verdict.iloc[0] if len(imp) else "n/a"
        n_seasons = int(imp.n_seasons.iloc[0]) if len(imp) else 0
        cr, n_common = _common(m)
        nt_base = min(nt_ref, key=lambda b: cr[b]) if nt_ref else None
        delta = float(cr[m] - cr[nt_base]) if nt_base else float("nan")
        c0 = bool(nt_base is not None and delta < 0)
        c_ev = n_common >= cfg.min_test_seasons if have_pred else True
        c_dom = dom_ok
        c_dep = cfg.deployable is None or m in cfg.deployable
        c1 = bool(mp[m] < mp[ref_all])
        c2 = verdict == "improved"
        c3 = bool(m in ml.index and base_l and ml[m] < min(ml[b] for b in base_l))
        c4 = n_seasons >= cfg.adopt_min_seasons
        c5 = not override_used
        stats_ok = bool(c1 and c2 and c3 and c4)
        c_sw = bool(sweep["passed"].get(primary_key, {}).get(m, False)) if sw_req else True
        gate_ok = bool(c0 and c_dom and c_ev)                                 # PACE V8 dual criteria (domain relevance + baseline pass), on enough seasons
        ok = bool(gate_ok and c_dep and c5 and c_sw and (stats_ok if cfg.nt_gate == "delta_nt+stats" else True))
        d_ds = float(rm[m] - rm["source_only"]) if "source_only" in rm.index else float("nan")
        rows.append({"method": m, "MAE": float(mp[m]), "RMSE": float(rm[m]), f"best baseline ({ref_all}) MAE": float(mp[ref_all]),
                     "Delta_NT (RMSE)": delta, "Delta_NT baseline": nt_base or "n/a", "0 Delta_NT < 0": c0, "seasons compared": n_common,
                     "deployable": c_dep, "Criterion 1 domain relevance": c_dom, "Criterion 2 baseline pass": c0, f"at least {cfg.min_test_seasons} common seasons": c_ev, "gate passed": gate_ok,
                     "RMSE vs all baselines (common seasons)": {k: v for k, v in cr.items()},
                     "Delta_NT vs target-only (RMSE)": float(rm[m] - rm["target_only"]) if "target_only" in rm.index else float("nan"), "Delta_DS vs source-only (RMSE)": d_ds,
                     "sweep qualified": c_sw, "statistically supported": stats_ok,
                     "1 lower MAE than best baseline": c1, "2 interval excludes zero": c2, "3 also better leave-one-out": c3,
                     f"4 at least {cfg.adopt_min_seasons} held-out seasons": c4, "5 not an override run": c5, "supported": ok,
                     "verdict": verdict})
    table = pd.DataFrame(rows)
    supported = table[table.supported].sort_values("Delta_NT (RMSE)") if len(table) else table
    gate_only = table[table["gate passed"] & ~table.supported] if len(table) else table
    transfer_supported = len(supported) > 0
    selected = supported.method.iloc[0] if transfer_supported else ref
    pool_t = table[table["deployable"]] if len(table) and table["deployable"].any() else table
    # the candidate transfer winner is the lowest-MAE method among those that PASS the gate when any does, otherwise the lowest-MAE method overall. Without this the
    # displayed "best method" (lowest MAE) and the pool-level gate ("some method passes") could describe two different models, so a row could read PASS next to a positive Delta_NT.
    cand_t = pool_t[pool_t["gate passed"]] if len(pool_t) and pool_t["gate passed"].any() else pool_t
    best_t = cand_t.sort_values("MAE").method.iloc[0] if len(cand_t) else None
    if transfer_supported:
        r0 = table[table.method == selected].iloc[0]
        extra = ("The improvement is also statistically supported (bootstrap interval excludes zero, and it holds under leave-one-year-out)."
                 if r0["statistically supported"] else "The evidence is weak: with so few held-out seasons the improvement is not statistically supported, so treat this forecast with caution.")
        statement = (("Qualified by the 10-granary benchmark sweep and " if sw_req else "") + f"negative-transfer safeguard PASSED: {LABELS.get(selected, selected)} has Delta_NT = {r0['Delta_NT (RMSE)']:+.3f} (RMSE, t/ha units of the harmonised label) "
                     f"against {LABELS.get(r0['Delta_NT baseline'], r0['Delta_NT baseline'])} under {primary_name} validation, so it is deployed. {extra}")
    else:
        why = []
        if best_t is not None:
            r = table[table.method == best_t].iloc[0]
            if not r["Criterion 1 domain relevance"]:
                why.append("Criterion 1 (domain relevance) failed: " + (domain.get("reason", "the source pool contains a granary with an incompatible agro-climatic calendar") if domain else ""))
            elif not r[f"at least {cfg.min_test_seasons} common seasons"]:
                why.append(f"only {int(r['seasons compared'])} common held-out seasons (at least {cfg.min_test_seasons} are required)")
            elif not r["sweep qualified"]:
                why.append(f"the best transfer method ({LABELS.get(best_t, best_t)}) did not pass the 10-granary benchmark sweep (Discover), so the source is not qualified for transfer")
            elif not r["0 Delta_NT < 0"]:
                why.append(f"the best transfer method ({LABELS.get(best_t, best_t)}) has Delta_NT = {r['Delta_NT (RMSE)']:+.3f} (RMSE) against "
                           f"{LABELS.get(r['Delta_NT baseline'], r['Delta_NT baseline'])}: it is not better than the local baselines")
            elif not r["1 lower MAE than best baseline"]:
                why.append(f"the best transfer method ({LABELS.get(best_t, best_t)}) did not have a lower error than {LABELS.get(ref, ref)}")
            elif not r["2 interval excludes zero"]:
                why.append(f"{LABELS.get(best_t, best_t)} had a lower average error than {LABELS.get(ref, ref)} but the difference is not statistically supported ({r['verdict']})")
            elif not r["3 also better leave-one-out"]:
                why.append(f"{LABELS.get(best_t, best_t)} was not better under leave-one-season-out, so the result depends on how the data are split")
            elif not r[f"4 at least {cfg.adopt_min_seasons} held-out seasons"]:
                why.append("too few held-out seasons to call any difference an improvement")
            elif not r["5 not an override run"]:
                why.append("the run overrode a NOT RECOMMENDED verdict, so it cannot be a basis for predictions")
        if not len(table):
            why.append("no transfer method was run")
        statement = ("Negative-transfer safeguard TRIGGERED, automatic fallback to the best local baseline: " + "; ".join(why or ["no transfer method met every criterion"]) +
                     f". The selected model is {LABELS.get(ref, ref)}, and transfer learning is not used for any prediction.")
    token = hashlib.sha256((str(selected) + str(ref) + str(mp.round(6).to_dict()) + str(ml.round(6).to_dict())).encode()).hexdigest()[:16]
    best_row = (table[table["deployable"]] if len(table) and table["deployable"].any() else table).sort_values("Delta_NT (RMSE)").iloc[0] if len(table) else None
    safeguard = dict(status="transfer deployed" if transfer_supported else "fallback to local baseline", delta_nt=None if best_row is None else float(best_row["Delta_NT (RMSE)"]),
                     delta_nt_method=None if best_row is None else best_row["method"], delta_nt_baseline=None if best_row is None else best_row["Delta_NT baseline"],
                     gate=cfg.nt_gate, gate_passed=bool(len(table) and (table["gate passed"] & table["deployable"]).any()), domain_relevant=dom_ok,
                     domain_reason="" if domain is None else domain.get("reason", ""))
    return dict(selected=selected, reference=ref, reference_all=ref_all, transfer_supported=transfer_supported, best_transfer=best_t, table=table, statement=statement,
                primary=primary_name, token=token, mae=mp.to_dict(), rmse=rm.to_dict(), override_used=override_used, safeguard=safeguard, sweep_required=sw_req, primary_key=primary_key)
