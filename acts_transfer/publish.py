"""What the Model Validator publishes with a sign-off, for the KPKM Programme Manager to read (see governance.py). Plain numbers in kg/ha, no model objects.

  backtest   per season, under forward validation and leave-one-year-out: the actual yield, the transfer model's prediction and the local baseline's
  nowcast    the latest season with stage features, predicted from the heading-stage features (up to 70 days after sowing) plus the trend: the in-season forecast
  outlook    2025-2028 per scenario (Normal, El Nino drought, Wet monsoon): the transfer forecast and the local baseline, each with a band that widens with the horizon
The transfer rows are present only when the gate passed. Which of them the KPKM views show is decided at read time from the registry status, never here.
"""
import numpy as np
import pandas as pd

from . import predict as predmod
from .methods import LABELS
from .scenarios import SCENARIOS, scenario_inputs


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def build_published(wf, Y, target, seasons=(2025, 2026, 2027, 2028)):
    wf.require_choice()
    c, b = wf.choice, wf.bundle
    kg = float(getattr(b, "y_to_kg", 1.0))
    ref = c["reference"]
    tkey = c["selected"] if c["transfer_supported"] else (c["best_transfer"] if c.get("safeguard", {}).get("gate_passed") else None)
    out = dict(target=target, transfer_model=tkey, transfer_label=LABELS.get(tkey, tkey) if tkey else None, baseline_model=ref, baseline_label=LABELS.get(ref, ref),
               unit="kg/ha", features=list(b.features), backtest={}, nowcast={}, outlook={}, scenario_shifts={})
    for name, r in wf.results.items():
        p = r.predictions
        if p is None or not len(p):
            continue
        piv = p.pivot_table(index="group", columns="method", values="y_pred") * kg
        act = p.groupby("group").y_true.first() * kg
        rows = []
        for g in piv.index:
            rows.append(dict(season=int(g), actual=_f(act[g]), transfer=_f(piv.get(tkey, pd.Series(dtype=float)).get(g)) if tkey else None,
                             baseline=_f(piv.get(ref, pd.Series(dtype=float)).get(g)), persistence=_f(piv.get("persistence", pd.Series(dtype=float)).get(g)),
                             season_trend=_f(piv.get("season_trend", pd.Series(dtype=float)).get(g))))
        out["backtest"][name] = rows
    keys = [k for k in (tkey, ref) if k]

    def table_for(inputs):
        t = predmod.predict_table(b, wf.results, c, inputs, wf.base, wf.run_cfg, wf._finals, list(dict.fromkeys(list(wf.result.methods) + keys)))
        rows = {}
        for k in keys:
            sub = t[t.model_key == k]
            rows[k] = [dict(season=int(r.season), predicted=_f(r.predicted * kg), low=_f(r.low * kg), high=_f(r.high * kg), band_source=r.band_source) for r in sub.itertuples()]
        return rows
    last_row = b.target.dropna(subset=["y"]).sort_values("group").iloc[-1]
    now = pd.DataFrame([dict(season=int(last_row.group), **{f: float(last_row[f]) for f in b.features})])
    nt = table_for(now)
    out["nowcast"] = dict(season=int(last_row.group), actual=_f(last_row.y * kg), transfer=(nt.get(tkey) or [None])[0] if tkey else None, baseline=(nt.get(ref) or [None])[0],
                          stage="heading (about 70 days after sowing): vegetative and reproductive stage features only")
    for sc in SCENARIOS:
        inp, sh = scenario_inputs(Y, target, list(b.features), sc, list(seasons))
        out["outlook"][sc] = {k: v for k, v in table_for(inp).items()}
        out["scenario_shifts"][sc] = {k: float(v) for k, v in sh.items()}
    return out
