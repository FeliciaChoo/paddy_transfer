"""The new tab order and its dependencies: assessment -> strategy -> train and evaluate -> yield prediction -> future outlook -> advisory.

Run:  python -m pytest tests/test_pipeline_order.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_transfer_gating import FEATS, cfg, complete_through_suitability, make_domain, ready_workflow       # noqa: E402
from acts_transfer import Config, TransferWorkflow, WorkflowError                                             # noqa: E402
from acts_transfer import modelchoice, predict                                                                 # noqa: E402
from acts_transfer.evaluation import EvaluationResult                                                          # noqa: E402


def trained(methods=("kmm", "coral"), **kw):
    wf = complete_through_suitability(ready_workflow(**kw))
    wf.select_strategy(list(methods))
    wf.run()
    return wf


# ------------------------------------------------------------------------------------------------ downstream needs an evaluated model
def test_prediction_and_outlook_are_refused_before_training():
    wf = complete_through_suitability(ready_workflow())
    inp = predict.make_inputs(wf.bundle, "typical", [2030])
    with pytest.raises(WorkflowError, match="evaluate"):
        wf.predict_for(inp)
    with pytest.raises(WorkflowError, match="evaluate"):
        wf.outlook_for(inp)
    st = {s["key"]: s["status"] for s in wf.stages()}
    assert st["prediction"] == "locked" and st["outlook"] == "locked" and st["training"] in ("locked", "ready")


def test_stages_follow_the_new_order():
    wf = ready_workflow()
    order = [s["key"] for s in wf.stages()]
    assert order == ["assessment", "strategy", "training", "prediction", "outlook", "advisory"]
    st = {s["key"]: s["status"] for s in wf.stages()}
    assert st["strategy"] == "locked" and st["training"] == "locked"
    wf.assess()
    wf.harmonise()
    assert {s["key"]: s["status"] for s in wf.stages()}["strategy"] == "locked"      # suitability still missing
    wf.evaluate_suitability()
    st = {s["key"]: s["status"] for s in wf.stages()}
    assert st["assessment"] == "complete" and st["strategy"] == "ready" and st["training"] == "locked"
    wf.select_strategy(["kmm"])
    assert {s["key"]: s["status"] for s in wf.stages()}["training"] == "ready"
    wf.run()
    st = {s["key"]: s["status"] for s in wf.stages()}
    assert st["training"] == "complete" and st["prediction"] == "ready" and st["outlook"] == "ready"


def test_predictions_name_the_model_and_its_role():
    wf = trained()
    t = wf.predict_for(predict.make_inputs(wf.bundle, "typical", [2030]), "typical")
    assert {"model", "role", "predicted", "band"} <= set(t.columns)
    sel = t[t.role.str.startswith("SELECTED")]
    assert len(sel) == 1 and sel.model_key.iloc[0] == wf.choice["selected"]
    assert set(t.model_key) >= {wf.choice["selected"], wf.choice["reference"]}


def test_outlook_inputs_are_validated():
    wf = trained()
    with pytest.raises(WorkflowError, match="missing columns"):
        predict.make_inputs(wf.bundle, "upload", [2030], pd.DataFrame({"season": [2030], "f1": [0.1]}))
    bad = predict.make_inputs(wf.bundle, "typical", [2030]).copy()
    bad["f1"] = bad["f1"].astype(object)
    bad.loc[0, "f1"] = "abc"
    with pytest.raises(WorkflowError):
        predict.check_inputs(bad, wf.bundle.features)
    ok = predict.make_inputs(wf.bundle, "typical", [2030, 2031])
    assert len(wf.outlook_for(ok, "typical")) >= 2


# ------------------------------------------------------------------------------------------------ changing upstream recalculates downstream
def test_changing_the_strategy_clears_evaluation_prediction_and_outlook_and_says_so():
    wf = trained()
    inp = predict.make_inputs(wf.bundle, "typical", [2030, 2031])
    wf.predict_for(inp)
    wf.outlook_for(inp)
    wf.pop_notices()
    wf.select_strategy(["coral"])
    assert wf.result is None and wf.choice is None and wf.prediction is None and wf.outlook is None
    n = " ".join(wf.notices)
    assert "model evaluation" in n and "yield prediction" in n and "future yield outlook" in n
    assert {s["key"]: s["status"] for s in wf.stages()}["prediction"] == "locked"
    with pytest.raises(WorkflowError):
        wf.predict_for(inp)                                            # cannot run downstream without a validated model


def test_retraining_clears_the_old_prediction_so_nothing_stale_survives():
    wf = trained()
    wf.predict_for(predict.make_inputs(wf.bundle, "typical", [2030]))
    assert wf.prediction is not None
    wf.run()
    assert wf.prediction is None and wf.outlook is None


def test_changing_the_data_clears_everything_downstream():
    wf = trained()
    wf.predict_for(predict.make_inputs(wf.bundle, "typical", [2030]))
    wf.assess()                                                         # re-assessing is an upstream change
    assert wf.bundle is None and wf.suit is None and wf.result is None and wf.prediction is None


# ------------------------------------------------------------------------------------------------ TrAdaBoost is not privileged
def test_tradaboost_is_a_reference_not_a_default_and_never_when_transfer_is_not_recommended():
    wf = complete_through_suitability(ready_workflow())
    s = wf.suit
    assert "tradaboost_r2" not in s["defaults"] or any(sc["id"] == "B" and sc["met"] for sc in s["scenarios"])
    assert "tradaboost_r2" in s["reference"]                           # available to compare, but only on request
    from test_transfer_gating import _not_recommended_workflow
    nr = _not_recommended_workflow()
    assert nr.suit["defaults"] == []                                    # no transfer method is pre-selected when transfer is not recommended


# ------------------------------------------------------------------------------------------------ the model-of-record gate, one criterion at a time
def _res(mae, improvement, n_folds=6):
    metrics = pd.DataFrame({"method": list(mae), "MAE": list(mae.values()), "n_test": n_folds})
    imp = pd.DataFrame(improvement, columns=["method", "baseline", "n_seasons", "verdict"])
    return EvaluationResult(pd.DataFrame(), metrics, imp, pd.DataFrame(), [], [], "x", "ridge", n_folds, [], list(mae), {}, [])


def _gate(fwd_mae, loso_mae, verdict="improved", n=6, override=False):
    ref = min((m for m in fwd_mae if m in ("target_only", "season_trend", "target_mean", "pooled", "source_only")), key=fwd_mae.get)
    res = {"forward": _res(fwd_mae, [("coral", ref, n, verdict)], n), "loso": _res(loso_mae, [])}
    return modelchoice.choose(res, Config(nt_gate="delta_nt+stats"), override)      # the strict mode: every statistical criterion as well as Delta_NT < 0


BASE = {"target_only": 0.30, "season_trend": 0.25, "source_only": 0.7, "pooled": 0.35, "target_mean": 0.5}


def test_transfer_is_selected_only_when_every_criterion_holds():
    c = _gate({**BASE, "coral": 0.20}, {**BASE, "coral": 0.18})
    assert c["transfer_supported"] and c["selected"] == "coral"
    assert "safeguard PASSED" in c["statement"]


@pytest.mark.parametrize("name, kwargs, fwd, loso", [
    ("not lower than the best baseline", {}, {**BASE, "coral": 0.27}, {**BASE, "coral": 0.2}),
    ("difference not statistically supported", {"verdict": "no clear difference"}, {**BASE, "coral": 0.20}, {**BASE, "coral": 0.18}),
    ("worse under leave-one-out", {}, {**BASE, "coral": 0.20}, {**BASE, "coral": 0.40}),
    ("too few held-out seasons", {"n": 4}, {**BASE, "coral": 0.20}, {**BASE, "coral": 0.18}),
    ("an override run", {"override": True}, {**BASE, "coral": 0.20}, {**BASE, "coral": 0.18}),
])
def test_each_failed_criterion_keeps_the_baseline(name, kwargs, fwd, loso):
    c = _gate(fwd, loso, **kwargs)
    assert not c["transfer_supported"], name
    assert c["selected"] == "season_trend", name                       # the best baseline by forward error
    assert "safeguard TRIGGERED" in c["statement"], name
    assert "not used for any prediction" in c["statement"], name


def test_the_best_baseline_is_the_hardest_comparison():
    c = _gate({**BASE, "coral": 0.28}, {**BASE, "coral": 0.2})          # beats target_only (0.30) but not season_trend (0.25)
    assert c["reference"] == "season_trend" and not c["transfer_supported"]


# ------------------------------------------------------------------------------------------------ the negative-transfer safeguard (methodology step E2)
def _gate_nt(fwd_mae, loso_mae, mode="delta_nt", **kw):
    ref = min((m for m in fwd_mae if m in ("target_only", "season_trend", "target_mean", "pooled", "source_only", "persistence")), key=fwd_mae.get)
    res = {"forward": _res(fwd_mae, [("coral", ref, 6, kw.get("verdict", "improved"))], 6), "loso": _res(loso_mae, [])}
    return modelchoice.choose(res, Config(nt_gate=mode), kw.get("override", False))


def test_delta_nt_below_zero_deploys_the_transfer_model():
    c = _gate_nt({**BASE, "persistence": 0.40, "coral": 0.20}, {**BASE, "persistence": 0.4, "coral": 0.18})
    assert c["transfer_supported"] and c["selected"] == "coral"
    assert c["safeguard"]["delta_nt"] < 0 and c["safeguard"]["delta_nt_baseline"] == "season_trend"      # min(season_trend 0.25, target_only 0.30, persistence 0.40): V8 includes the trend


def test_delta_nt_at_or_above_zero_blocks_transfer_and_falls_back_to_a_local_baseline():
    c = _gate_nt({**BASE, "persistence": 0.22, "coral": 0.24}, {**BASE, "persistence": 0.22, "coral": 0.2})   # persistence (0.22) beats coral (0.24)
    assert not c["transfer_supported"]
    assert c["safeguard"]["delta_nt"] > 0 and c["safeguard"]["delta_nt_baseline"] == "persistence"
    assert c["selected"] == "persistence" and c["selected"] != "source_only" and c["selected"] != "pooled"   # never a model trained on the source


def test_the_fallback_is_always_a_local_baseline_even_when_source_only_is_best():
    c = _gate_nt({**BASE, "source_only": 0.01, "coral": 0.20}, {**BASE, "source_only": 0.01, "coral": 0.18}, mode="delta_nt+stats", verdict="no clear difference")
    assert not c["transfer_supported"] and c["selected"] in ("target_only", "season_trend", "persistence", "target_mean")


def test_an_override_run_is_never_deployed_even_when_delta_nt_is_negative():
    c = _gate_nt({**BASE, "coral": 0.10}, {**BASE, "coral": 0.10}, override=True)
    assert not c["transfer_supported"]


# ------------------------------------------------------------------------------------------------ causal counterfactual prescriptions & governance registry
def test_generate_prescriptions_requires_evaluated_model_and_writes_registry(tmp_path):
    wf = complete_through_suitability(ready_workflow())
    reg = tmp_path / "governance_registry.json"
    with pytest.raises(WorkflowError, match="evaluate"):
        wf.generate_prescriptions(registry_path=str(reg))

    wf.select_strategy(["kmm"])
    wf.run()
    wf.bundle.X_target = wf.bundle.target[FEATS].to_numpy()
    wf.bundle.y_target = wf.bundle.target["y"].to_numpy()
    wf.bundle.feature_names = ["Schedule_Slip_Decades", "LSWI_rep", "StemBorer_Weather_Risk", "f4"]
    wf.bundle.X_target[:, 0] = np.linspace(3.0, 0.0, len(wf.bundle.target))

    presc = wf.generate_prescriptions(
        schedule_csv=str(tmp_path / "missing_sched.csv"),
        rules_csv=str(tmp_path / "missing_rules.csv"),
        registry_path=str(reg),
    )
    assert reg.exists() and wf.prescriptions is not None
    saved = pd.read_json(reg)
    assert "T" in saved.columns and isinstance(presc, list)

    # Changing upstream strategy clears stale prescriptions
    wf.select_strategy(["coral"])
    assert wf.prescriptions is None