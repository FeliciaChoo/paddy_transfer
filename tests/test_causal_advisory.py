"""Unit tests for the 3-stage causal methodology (Identification, Transferability, Adaptation) and LLM guardrails.

Run: python -m pytest tests/test_causal_advisory.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from acts_transfer import causal_adapt      # noqa: E402
from dashboard import llm                 # noqa: E402


def test_stage1_confounded_vs_deconfounded_treatment_effect():
    """Stage 1 (Identification): Deconfounding weather/soil (X) recovers true causal effect of LSWI (T) on Yield (Y)."""
    rng = np.random.RandomState(7)
    n = 120
    # Confounder X (favorable rain/soil raises both LSWI and Yield)
    rain_confounder = rng.normal(0, 1, n)
    lswi_treatment = 0.25 + 0.08 * rain_confounder + rng.normal(0, 0.03, n)
    true_tau = 2.0  # true causal effect of +1 LSWI = +2.0 t/ha
    yield_y = 4.5 + 0.9 * rain_confounder + true_tau * lswi_treatment + rng.normal(0, 0.05, n)

    # Naive correlation is biased upward by rain_confounder
    naive_slope = np.polyfit(lswi_treatment, yield_y, 1)[0]

    # DML-style orthogonalization: residualize Y on X and T on X
    y_res = yield_y - np.polyval(np.polyfit(rain_confounder, yield_y, 1), rain_confounder)
    t_res = lswi_treatment - np.polyval(np.polyfit(rain_confounder, lswi_treatment, 1), rain_confounder)
    dml_tau = np.dot(t_res, y_res) / np.dot(t_res, t_res)

    assert naive_slope > true_tau + 1.5
    assert abs(dml_tau - true_tau) < 0.35


def test_stage3_optimize_target_prescriptions_respects_schedule_bounds_and_sop_rules(tmp_path):
    """Stage 3 (Adaptation): Counterfactual delta-T* is bounded by JPS/IADA canal release decades and maps to SOP rules."""
    sched = tmp_path / "sched.csv"
    pd.DataFrame([{
        "Granary": "MADA",
        "start_decade": 6,
        "end_decade": 11,
        "agency": "MADA / JPS",
    }]).to_csv(sched, index=False)

    rules = tmp_path / "rules.csv"
    pd.DataFrame([{
        "rule_id": "R01_PLANT_SLIP",
        "action": "Custom SOP: synchronize nursery sowing to MADA Phase 1 canal release",
    }]).to_csv(rules, index=False)

    rng = np.random.RandomState(42)
    slip = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, 2.0])
    lswi = np.linspace(0.20, 0.35, 8)
    y = 5.2 - 0.35 * slip + 1.5 * lswi + rng.normal(0, 0.02, 8)
    bundle = SimpleNamespace(
        X_target=np.column_stack([slip, lswi]),
        y_target=y,
        feature_names=["Schedule_Slip_Decades", "LSWI_rep"],
    )
    target = SimpleNamespace(name="MADA")
    presc = causal_adapt.optimize_target_prescriptions(
        bundle=bundle,
        target=target,
        choice={"method": "HGB + RULE"},
        schedule_csv=str(sched),
        rules_csv=str(rules),
    )
    assert len(presc) >= 1
    top = presc[0]
    assert top["delta_y_kg_ha"] > 0
    assert top["ci_low"] <= top["delta_y_kg_ha"] <= top["ci_high"]
    assert "Decades 6–11" in top["window"]
    assert "MADA / JPS" in top["schedule_bound"]
    assert top["rule_id"] == "R01_PLANT_SLIP"
    assert "Custom SOP" in top["action"]


def test_narrate_counterfactual_cards_formats_evidence_and_catches_hallucinated_numbers(monkeypatch):
    """Advisory narration formats Who/What/When/Why cards and blocks ungrounded numbers."""
    def fake_call_llm(system, user):
        assert "Who" in user and "What" in user and "When" in user and "Why" in user
        assert "245.5" in user
        return "**Who:** IADA officers [E1]. **What:** Align planting [E1]. **When:** Decades 6–11 [E1]. **Why:** +245.5 kg/ha gain (hallucinated 9999 kg/ha) [E1]."

    monkeypatch.setattr(llm, "api_key", lambda: "test-key")
    monkeypatch.setattr(llm, "call_llm", fake_call_llm)

    text, bad, err = llm.narrate_counterfactual_cards("MADA", [{
        "lever": "Planting Schedule Synchronization",
        "action": "Align planting",
        "window": "Decades 6–11",
        "delta_y_kg_ha": 245.5,
        "ci_low": 190.0,
        "ci_high": 301.0,
        "schedule_bound": "MADA / JPS",
        "rule_id": "R01_PLANT_SLIP",
    }])
    assert err is None and text is not None
    assert "9999" in bad and "245.5" not in bad