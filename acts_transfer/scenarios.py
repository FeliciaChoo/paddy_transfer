"""Climate / NDVI scenario inputs for the multi-year outlook (PACE V8 E3). The outlook is NEVER run on flat historical-average inputs only.

A scenario moves each stage feature of the target from its own typical value by a tail of the year-to-year ANOMALIES observed across all granaries in 2018-2024
(anomaly = value - that granary's own mean over the years). 'El Nino drought' takes the dry / hot / bright / low-canopy tail, 'Wet monsoon' the wet tail, 'Normal' no shift.
These are assumptions about what a drought or wet year looks like in this data, not forecasts of the weather; they can be adjusted by the user in the dashboard.
"""
import numpy as np
import pandas as pd

SCENARIOS = ["Normal", "El Niño drought", "Wet monsoon"]
# feature family -> (percentile of the anomaly used in a drought year, in a wet year); None = no shift
RULES = {"Rain_mm": (10, 90), "Tmax_C": (90, 10), "Solar_MJ": (90, 10), "EVI": (10, None), "LSWI": (10, 90), "MNDWI": (10, 90)}


def _family(col):
    import re
    m = re.match(r"^C[12]_(.+?)_(veg|rep|rip)$", col)
    return m.group(1) if m else None


def anomaly_table(Y: pd.DataFrame, feats):
    """within-granary anomalies of every feature (value minus the granary's own mean), all granaries pooled"""
    A = Y[["District", "Year"] + [f for f in feats if f in Y]].copy()
    for f in feats:
        if f in A:
            A[f] = A[f] - A.groupby("District")[f].transform("mean")
    return A


def scenario_shifts(Y: pd.DataFrame, feats, scenario: str):
    """{feature: shift in the feature's own units} for one scenario"""
    if scenario == "Normal":
        return {}
    A = anomaly_table(Y, feats)
    col = 0 if scenario == "El Niño drought" else 1
    out = {}
    for f in feats:
        fam = _family(f)
        if fam in RULES and RULES[fam][col] is not None and f in A and A[f].notna().sum() >= 10:
            out[f] = float(np.nanpercentile(A[f], RULES[fam][col]))
    return out


def scenario_inputs(Y: pd.DataFrame, target: str, feats, scenario: str, seasons, overrides: dict = None):
    """one row per season: the target's typical feature values plus the scenario shift. overrides: {feature: extra shift} chosen by the user"""
    T = Y[Y.District == target]
    base = {f: float(T[f].mean()) for f in feats}
    sh = scenario_shifts(Y, feats, scenario)
    row = {f: base[f] + sh.get(f, 0.0) + (overrides or {}).get(f, 0.0) for f in feats}
    return pd.DataFrame([dict(season=s, **row) for s in seasons]), sh
