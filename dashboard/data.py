"""Load the app-ready CSVs written by pipeline/prepare_app_data.py (one place, cached)."""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import streamlit as st

from .bridge import ensure_rules_file

DATA_DIR = Path(os.environ.get("PADDY_APP_DATA", Path(__file__).resolve().parent.parent / "data"))

REQUIRED = {
    "districts": "districts.csv",          # one row per granary: centre point, number of sampled pixels
    "fp": "granary_footprints.csv",        # outline of each granary's sampled paddy pixels (map shape)
    "yh": "yield_history.csv",             # past reported yield (+ change, z-score, jump flags)
    "outlook": "yield_outlook.csv",        # future yield outlook with a band  <- plug a trained model in here
    "backtest": "yield_backtest.csv",      # how simple yield baselines did on past years
    "relation": "ndvi_yield_relation.csv",
    "cond": "condition_summary.csv",       # current crop condition, 10/30-day NDVI forecast, status, stage
    "phen": "phenology.csv",               # estimated planting calendar
    "preds": "chosen_predictions.csv",     # NDVI forecasts vs observed, every test year
    "skill": "model_skill.csv",
}
OPTIONAL = {"series": "ndvi_district_decade.csv",
            "tl_summary": "transfer_summary.csv",       # pipeline/transfer_benchmark.py --out data (transfer vs normal training)
            "tl_verdict": "transfer_verdict.csv",
            "tl_paired": "transfer_paired.csv",
            "resil_fc": "resilience_forecast.csv",      # pipeline/resilience_from_forecast.py: resilience measured through the NDVI forecast
            "ctx": "context_signals.csv",               # same script: recent weather / water / soil against each granary's own normal
            "yield_dyn": "yield_dynamics_summary.csv",
            "yield_iw": "yield_instance_weighting_summary.csv",   # pipeline/yield_instance_weighting.py
            "yield_tab": "yield_transfer_summary.csv"}      # pipeline/yield_transfer_benchmark.py: TrAdaBoost on yield inputs  # pipeline/yield_dynamics_transfer.py: transfer vs normal training for yield


@st.cache_data
def _read(path, mtime):  # mtime is only here so that editing a CSV (e.g. the advice rules) refreshes the cache
    return pd.read_csv(path)


def load():
    missing = [f for f in REQUIRED.values() if not (DATA_DIR / f).exists()]
    if missing:
        st.error(f"Missing in {DATA_DIR}: {', '.join(missing)}. Run `python pipeline/prepare_app_data.py ...` first "
                 "(commands are in README.md).")
        st.stop()
    D = SimpleNamespace(**{k: _read(str(DATA_DIR / f), (DATA_DIR / f).stat().st_mtime) for k, f in REQUIRED.items()})
    for k, f in OPTIONAL.items():
        p = DATA_DIR / f
        setattr(D, k, _read(str(p), p.stat().st_mtime) if p.exists() else None)
    D.DATA_DIR = DATA_DIR
    p = ensure_rules_file(DATA_DIR)                            # administrator intervention rules (written once, then editable)
    D.irules = _read(str(p), p.stat().st_mtime)
    D.fp = D.fp.assign(polygon=D.fp.polygon.map(json.loads))
    D.ALL = sorted(D.districts.District)                       # every granary with yield records
    D.SAT = sorted(D.cond.District)                            # granaries that also have satellite forecasts
    D.LAST_YEAR = int(D.yh.Year.max())                         # last year with reported yield
    D.FINAL_TEST_YEAR = int(D.skill.test_year.max())           # last held-out year of the NDVI model
    D.AS_OF = str(D.cond.As_of.max()) if len(D.cond) else "n/a"
    D.MEAN_YIELD = D.yh.groupby("District").Yield.mean()
    D.WEAK = sorted(D.cond.loc[D.cond.Signal == "weak", "District"]) if len(D.cond) else []
    return D
