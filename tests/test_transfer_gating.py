"""Tests that transfer learning cannot bypass the mandatory assessment and harmonisation checks, and that the evaluation cannot leak.

Run:  python -m pytest tests -q
Synthetic data only, so the tests do not depend on any project file.
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from acts_transfer import Config, Domain, TransferWorkflow, WorkflowError            # noqa: E402
from acts_transfer import assessment, evaluation, harmonise, methods                  # noqa: E402
from acts_transfer.harmonise import AssessmentBlocked, StaleAssessment               # noqa: E402

FEATS = ["f1", "f2", "f3", "f4"]
UNITS = {"f1": "unitless", "f2": "unitless", "f3": "dB", "f4": "deg C", "Yield_kg_ha": "kg/ha"}


def make_domain(name, n, seed, shift=0.0, label_shift=0.0, start=2010):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, 4)) + shift
    y = 4.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.1, size=n) + label_shift
    df = pd.DataFrame(X, columns=FEATS)
    df.insert(0, "Year", np.arange(start, start + n))
    df["Yield_kg_ha"] = y * 1000.0
    # keep f1..f4 inside plausible ranges for the configured names by using their own (unconfigured) names: no range check applies
    return Domain(name, df, list(FEATS), "Yield_kg_ha", "Year", dict(UNITS))


def cfg(**kw):
    return Config(mmd_permutations=60, bootstrap_n=200, **kw)


def ready_workflow(n_s=12, n_t=8, **kw):
    wf = TransferWorkflow(cfg(**kw))
    wf.select_domains(make_domain("S", n_s, 1), make_domain("T", n_t, 2, shift=0.3), "ridge")
    return wf


def complete_through_suitability(wf):
    wf.assess()
    wf.harmonise()
    wf.evaluate_suitability()
    return wf


# ------------------------------------------------------------------------------------------------ mandatory order
def test_run_without_anything_raises():
    wf = ready_workflow()
    with pytest.raises(WorkflowError, match="assess"):
        wf.run(methods=["kmm"])


def test_harmonise_without_assessment_raises():
    wf = ready_workflow()
    with pytest.raises(WorkflowError, match="assess"):
        wf.harmonise()


def test_suitability_without_harmonisation_raises():
    wf = ready_workflow()
    wf.assess()
    with pytest.raises(WorkflowError, match="harmonise"):
        wf.evaluate_suitability()


def test_run_with_assessment_but_without_harmonisation_raises():
    wf = ready_workflow()
    wf.assess()
    with pytest.raises(WorkflowError, match="harmonise"):
        wf.run(methods=["kmm"])


def test_run_without_suitability_raises():
    wf = ready_workflow()
    wf.assess()
    wf.harmonise()
    with pytest.raises(WorkflowError, match="suitability"):
        wf.run(methods=["kmm"])


def test_run_without_strategy_raises():
    wf = complete_through_suitability(ready_workflow())
    with pytest.raises(WorkflowError, match="strategy"):
        wf.run()


def test_status_locks_later_steps():
    wf = ready_workflow()
    st = {s["key"]: s["status"] for s in wf.status()}
    assert st["select"] == "complete" and st["assess"] == "current"
    assert all(st[k] == "locked" for k in ("harmonise", "suitability", "strategy", "run", "compare"))


def test_full_path_runs_and_marks_everything_complete():
    wf = complete_through_suitability(ready_workflow())
    wf.select_strategy(["kmm", "coral"])
    r = wf.run()
    assert r.n_folds > 0 and {"kmm", "coral", "source_only", "target_only"} <= set(r.metrics.method)
    assert all(s["status"] == "complete" for s in wf.status())


# ------------------------------------------------------------------------------------------------ stale / tampered data
def test_changed_data_after_assessment_is_refused():
    wf = ready_workflow()
    wf.assess()
    wf.source.df.loc[0, "f1"] += 5.0                      # the data change after it was assessed
    with pytest.raises(StaleAssessment):
        wf.harmonise()


def test_changed_data_after_harmonisation_blocks_run():
    wf = complete_through_suitability(ready_workflow())
    wf.select_strategy(["kmm"])
    wf.target.df.loc[0, "f2"] += 5.0
    with pytest.raises(StaleAssessment):
        wf.run()


def test_changing_domains_clears_every_later_step():
    wf = complete_through_suitability(ready_workflow())
    wf.select_strategy(["kmm"])
    wf.run()
    wf.select_domains(make_domain("S2", 12, 3), make_domain("T2", 8, 4), "ridge")
    assert wf.report is None and wf.bundle is None and wf.suit is None and wf.result is None and wf.strategy is None
    with pytest.raises(WorkflowError):
        wf.run(methods=["kmm"])


def test_assessment_from_other_data_cannot_harmonise():
    a, b = ready_workflow(), ready_workflow()
    b.source = make_domain("S", 12, 99)                   # different data than the report was built on
    rep = a.assess()
    with pytest.raises(StaleAssessment):
        harmonise.harmonise(rep, b.source, b.target, harmonise.default_spec(rep, b.source, b.target), b.cfg)


def test_evaluate_rejects_raw_data():
    wf = ready_workflow()
    with pytest.raises(TypeError, match="HarmonisedBundle"):
        evaluation.evaluate(wf.target.df, ["kmm"], "ridge", wf.cfg)


# ------------------------------------------------------------------------------------------------ readiness and suitability gates
def test_not_ready_data_blocks_harmonisation_and_everything_after():
    wf = TransferWorkflow(cfg())
    wf.select_domains(make_domain("S", 12, 1), make_domain("T", 2, 2), "ridge")   # 2 target rows: below the minimum
    rep = wf.assess()
    assert rep.readiness == "not_ready"
    with pytest.raises(AssessmentBlocked):
        wf.harmonise()
    assert {s["key"]: s["status"] for s in wf.status()}["harmonise"] == "blocked"
    with pytest.raises(WorkflowError):
        wf.run(methods=["source_only"])


def test_not_recommended_needs_an_explicit_override():
    wf = TransferWorkflow(cfg(min_obs_block=2))
    wf.select_domains(make_domain("S", 12, 1), make_domain("T", 2, 2), "ridge")   # one target training row per fold
    complete_through_suitability(wf)
    assert wf.suit["suitability"] == "not_recommended"
    wf.select_strategy(["source_only"])
    with pytest.raises(WorkflowError, match="override"):
        wf.run()
    r = wf.run(override_reason="research comparison, accepting the risk")
    assert wf.override and any("override" in n.lower() for n in r.notes)


def test_ineligible_method_is_rejected():
    wf = TransferWorkflow(cfg(min_obs_block=2))
    wf.select_domains(make_domain("S", 12, 1), make_domain("T", 3, 2), "ridge")  # 2 target rows per training fold: TrAdaBoost needs 3
    complete_through_suitability(wf)
    with pytest.raises(WorkflowError, match="not eligible"):
        wf.select_strategy(["tradaboost_r2"])


def test_methods_not_implemented_are_not_offered():
    wf = complete_through_suitability(ready_workflow())
    tab = wf.suit["methods"].set_index("key")
    for k in ("fine_tune_nn", "adversarial_da", "tree_fine_tune"):
        assert not bool(tab.loc[k, "eligible"])
    with pytest.raises(WorkflowError):
        wf.select_strategy(["fine_tune_nn"])


def test_parameter_transfer_needs_a_ridge_base():
    wf = TransferWorkflow(cfg())
    wf.select_domains(make_domain("S", 12, 1), make_domain("T", 8, 2), "random_forest")
    complete_through_suitability(wf)
    assert not bool(wf.suit["methods"].set_index("key").loc["ridge_prior", "eligible"])


# ------------------------------------------------------------------------------------------------ harmonisation behaviour
def test_harmonisation_converts_units_keeps_originals_and_records_everything():
    wf = ready_workflow()
    before = wf.target.df.copy()
    wf.assess()
    b = wf.harmonise()
    pd.testing.assert_frame_equal(wf.target.df, before)                   # the original is untouched
    assert 3.0 < b.target.y.mean() < 5.5                                   # kg/ha -> t/ha
    assert any(l["step"] == "convert units" for l in b.log) and any(l["step"] == "deferred to training folds" for l in b.log)
    assert b.ready and all(v["passed"] for v in b.validation)


def test_invalid_values_become_missing_not_kept():
    s = make_domain("S", 12, 1)
    s.df["NDVI_mean"] = 0.5
    s.features.append("NDVI_mean")
    s.df.loc[2, "NDVI_mean"] = 7.0                                         # impossible NDVI
    t = make_domain("T", 8, 2)
    t.df["NDVI_mean"] = np.linspace(0.3, 0.7, 8)
    t.features.append("NDVI_mean")
    wf = TransferWorkflow(cfg())
    wf.select_domains(s, t, "ridge")
    wf.assess()
    b = wf.harmonise()
    assert np.isnan(b.source.loc[2, "NDVI_mean"])
    assert any(l["step"] == "invalid values to missing" and l["rows_affected"] >= 1 for l in b.log)


def test_unavailable_features_are_listed_as_proposed_not_invented():
    wf = ready_workflow()
    rep = wf.assess()
    prop = rep.compat[rep.compat.status == "PROPOSED, not available"].feature.tolist()
    assert "humidity" in prop and "land surface temperature" in prop
    assert "humidity" not in wf.source.df.columns


# ------------------------------------------------------------------------------------------------ metrics are real
def test_smd_matches_the_formula_and_detects_a_shift():
    rng = np.random.RandomState(0)
    a, b = rng.normal(0, 1, 400), rng.normal(2, 1, 400)
    assert assessment.smd_of(a, b) == pytest.approx(2.0, abs=0.2)
    same = assessment.smd_of(rng.normal(0, 1, 400), rng.normal(0, 1, 400))
    assert abs(same) < 0.2


def test_mmd_detects_shift_and_not_identical_distributions():
    rng = np.random.RandomState(1)
    cfg_ = cfg()
    A = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc"))
    same = assessment.domain_differences(A, pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc")), cfg_)
    shifted = assessment.domain_differences(A, pd.DataFrame(rng.normal(size=(60, 3)) + 1.5, columns=list("abc")), cfg_)
    assert same["mmd_p"] > 0.05 and shifted["mmd_p"] < 0.05


# ------------------------------------------------------------------------------------------------ no leakage
def _bundle():
    wf = complete_through_suitability(ready_workflow())
    return wf.bundle, wf.cfg


def test_heldout_labels_never_change_their_own_prediction():
    bundle, c = _bundle()
    keys = ["source_only", "target_only", "pooled", "tradaboost_r2", "kmm", "coral", "tca", "ridge_prior"]
    base = evaluation.evaluate(bundle, keys, "ridge", c).predictions
    g = sorted(bundle.target.group.unique())[3]
    tgt = bundle.target.copy()
    tgt.loc[tgt.group == g, "y"] = tgt.loc[tgt.group == g, "y"] + 50.0               # corrupt ONE season's label
    changed = evaluation.evaluate(dataclasses.replace(bundle, target=tgt), keys, "ridge", c).predictions
    a = base[base.group == g].set_index("method").y_pred
    b = changed[changed.group == g].set_index("method").y_pred
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())                   # its own prediction cannot see its own label
    other = base[base.group != g].groupby("method").y_pred.sum()
    other2 = changed[changed.group != g].groupby("method").y_pred.sum()
    assert not np.allclose(other.drop(["source_only"]).values, other2.drop(["source_only"]).values)   # sanity: it IS used when it is training data


def test_training_arrays_never_contain_the_heldout_season(monkeypatch):
    bundle, c = _bundle()
    seen = []

    class Spy(methods.SourceOnly):
        def fit(self, Xs, ys, Xt, yt):
            seen.append(np.array(Xt))
            return super().fit(Xs, ys, Xt, yt)

    monkeypatch.setitem(methods.REGISTRY, "source_only", Spy)
    evaluation.evaluate(bundle, ["source_only"], "ridge", dataclasses.replace(c, cv_strategy="loso"))
    T = bundle.target.sort_values("group")
    feats = bundle.features
    assert len(seen) == len(T)
    for (_, row), xt in zip(T.iterrows(), seen):
        assert len(xt) == len(T) - 1
        assert not any(np.allclose(row[feats].values.astype(float), r) for r in xt)


def test_forward_validation_uses_only_earlier_seasons(monkeypatch):
    bundle, c = _bundle()
    seen = []

    class Spy(methods.SourceOnly):
        def fit(self, Xs, ys, Xt, yt):
            seen.append(len(Xt))
            return super().fit(Xs, ys, Xt, yt)

    monkeypatch.setitem(methods.REGISTRY, "source_only", Spy)
    evaluation.evaluate(bundle, ["source_only"], "ridge", dataclasses.replace(c, cv_strategy="forward"))
    assert seen == list(range(2, len(bundle.target)))                               # 2, 3, ..., n-1 earlier seasons, never later ones


def test_scarcity_setting_limits_target_training_seasons():
    bundle, c = _bundle()
    r = evaluation.evaluate(bundle, ["target_only"], "ridge", dataclasses.replace(c, max_target_train_groups=3))
    assert max(r.n_target_train) == 3
