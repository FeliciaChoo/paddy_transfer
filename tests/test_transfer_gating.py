"""Tests that transfer learning cannot bypass the mandatory assessment and harmonisation checks, and that the evaluation cannot leak.

Run:  python -m pytest tests -q
Synthetic data only, so the tests do not depend on any project file.
"""
import copy
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


def _not_recommended_workflow():
    wf = TransferWorkflow(cfg())
    wf.select_domains(make_domain("S", 12, 1), make_domain("T", 8, 2, shift=3.0, label_shift=2.0), "ridge")   # features AND yield differ strongly
    complete_through_suitability(wf)
    assert wf.suit["suitability"] == "not_recommended"
    return wf


def test_not_recommended_needs_an_override_only_to_run_a_transfer_method():
    wf = _not_recommended_workflow()
    wf.select_strategy(["kmm"])
    with pytest.raises(WorkflowError, match="override"):
        wf.run()                                                      # a transfer method needs a stated reason
    r = wf.run(override_reason="research comparison, accepting the risk")
    assert wf.override and any("override" in n.lower() for n in r.notes)
    assert wf.choice["transfer_supported"] is False                    # an override run can never become the model used for predictions


def test_baselines_only_run_needs_no_override_even_when_not_recommended():
    wf = _not_recommended_workflow()
    wf.select_strategy(["target_only"])
    wf.run()                                                           # direct model training involves no transfer, so nothing to override
    assert wf.choice is not None and wf.choice["selected"] in ("source_only", "target_only", "target_mean", "season_trend", "pooled")


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


# ---- regression: four impossible EVI readings (1e9) once turned single forecast folds into nonsense for KADA, Kemasin Semerak and Pulau Pinang
def test_the_shipped_tables_hold_no_impossible_vegetation_index_values():
    import pandas as pd
    from acts_transfer.sweep import clean_physical_ranges
    for f in ("transfer_stage_dataset.csv", "transfer_decade_features.csv", "transfer_yield_dataset.csv"):
        _, n = clean_physical_ranges(pd.read_csv(f"data/{f}"))
        assert n == 0, (f, n)
    c = pd.read_csv("eda/DailyGranary_clean.csv", usecols=["EVI_clean", "LSWI_clean", "MNDWI_clean", "NDVI_clean"])
    assert (c.abs().max() <= 1.0).all()


def test_the_range_guard_turns_an_impossible_index_into_a_gap_and_leaves_the_rest_alone():
    import pandas as pd
    from acts_transfer.sweep import clean_physical_ranges
    Y = pd.DataFrame({"District": ["A", "B", "C"], "C1_EVI_veg": [0.5, 3.3e8, -0.2], "C1_Rain_mm_rep": [900.0, 120.0, 5000.0], "EVI_pix_mean": [0.4, 0.5, 6.8e7]})
    out, n = clean_physical_ranges(Y)
    assert n == 2 and out["C1_EVI_veg"].isna().tolist() == [False, True, False] and out["EVI_pix_mean"].isna().tolist() == [False, False, True]
    assert out["C1_Rain_mm_rep"].tolist() == [900.0, 120.0, 5000.0] and Y["C1_EVI_veg"].iloc[1] == 3.3e8      # rain is not an index; the input is untouched


def test_no_granary_gets_an_absurd_forecast_error_in_the_automl_step():
    import os
    import pandas as pd
    from acts_transfer.automl_runner import run_full_automl_for_granary
    Y = pd.read_csv("data/transfer_stage_dataset.csv")
    mt = os.path.getmtime("data/transfer_stage_dataset.csv")
    for g in sorted(Y.District.unique()):
        a = run_full_automl_for_granary(g, "data", mt)
        assert a["model_table"]["Forward RMSE (t/ha)"].max() < 5.0, (g, a["model_table"][["Method ID", "Forward RMSE (t/ha)"]].to_dict("records"))
        for r in a["eval_results"].values():
            assert (r.predictions.y_pred.abs() < 50).all(), g          # yields are in t/ha: a prediction of tens of t/ha is a numerical blow-up


# ================================================================================================================================================
# Step 5/6 regression: donor-pool competition, pre-approval diagnostics (weather ablation, few-shot scarcity, driver attribution), safeguard + registry
# ================================================================================================================================================
ROOT = Path(__file__).resolve().parent.parent
# Which granary passes the gate depends on the sweep, so these are pinned to the current data and sweep (Meteo-Only vs Meteo + Optical, 5 Two-Stage rounds):
VOLATILE, STABLE = "IADA Rompin", "IADA Barat Laut Selangor"            # both pass the gate (Delta_NT < 0): Rompin -387 kg/ha, Barat Laut Selangor -22 kg/ha
POOL_COLS = ["Donor Pool Candidate", "Num Donors", "Best Method", "Best Forward RMSE (kg/ha)", "Delta_NT vs Local (kg/ha)", "Safeguard Gate"]
DRIVER = "C1_LSWI_rep"
FEATS_DIAG = [DRIVER, "C1_Rain_mm_rep", "C1_Tmax_C_rep", "C1_EVI_rep", "Optical_Valid_Fraction"]


@pytest.fixture(scope="module")
def automl():
    """the AutoML payload of the volatile and of the stable granary, computed once for the whole module"""
    from acts_transfer.automl_runner import run_full_automl_for_granary
    mt = (ROOT / "data" / "transfer_stage_dataset.csv").stat().st_mtime
    return {g: run_full_automl_for_granary(g, str(ROOT / "data"), mt) for g in (VOLATILE, STABLE)}


def diag_bundle(slope=600.0, n_target=7, n_donor_seasons=12, seed=0, noise=40.0):
    """a harmonised bundle (yield in t/ha, y_to_kg = 1000): the target granary and two donors, yield = granary level + slope * (standardised LSWI) + small noise.
    Only C1_LSWI_rep drives yield; the other weather features are pure noise."""
    from acts_transfer.sweep import build_bundle
    r = np.random.RandomState(seed)
    rows = []
    for name, years, level in (("TGT", range(2014, 2014 + n_target), 4200.0), ("D1", range(2010, 2010 + n_donor_seasons), 5200.0), ("D2", range(2010, 2010 + n_donor_seasons), 4800.0)):
        for yr in years:
            f = {DRIVER: r.normal(0.30, 0.08), "C1_Rain_mm_rep": r.normal(200, 60), "C1_Tmax_C_rep": r.normal(30, 1.2), "C1_EVI_rep": r.normal(0.5, 0.07)}
            rows.append(dict(District=name, Year=yr, Yield_kg_ha=level + slope * (f[DRIVER] - 0.30) / 0.08 + r.normal(0, noise), Optical_Valid_Fraction=0.9, **f))
    bundle, kept = build_bundle(pd.DataFrame(rows), ["D1", "D2"], "TGT", FEATS_DIAG)
    assert DRIVER in kept
    return bundle


# ---------------------------------------------------------------------------------------------------- donor-pool competition
@pytest.mark.parametrize("granary", [VOLATILE, STABLE])
def test_donor_pool_competition_schema_and_selection(automl, granary):
    a = automl[granary]
    assert "donor_pool_table" in a
    t = a["donor_pool_table"]
    assert len(t) >= 1 and set(POOL_COLS) <= set(t.columns), t.columns.tolist()
    rmse = t["Best Forward RMSE (kg/ha)"].astype(float)
    assert np.isfinite(rmse).all() and (rmse >= 0).all()
    assert rmse.is_monotonic_increasing, rmse.tolist()                                       # sorted ascending by forward RMSE
    assert a["scores"]["donor_winner"][0] == t.iloc[0]["Donor Pool Candidate"]               # the winner is the first row
    assert set(t["Safeguard Gate"]) <= {"PASS", "FALLBACK"}
    delta = t["Delta_NT vs Local (kg/ha)"].astype(float)
    away_from_zero = delta.abs() >= 0.05                                                      # the table rounds to 0.1 kg/ha: a delta that rounds to 0.0 has no visible sign
    assert ((t["Safeguard Gate"] == "PASS") == (delta < 0))[away_from_zero].all(), t[POOL_COLS].to_dict("records")
    assert (t["Num Donors"].astype(int) >= 1).all() and not t.duplicated(["Donor Pool Candidate", "Feature Regime"]).any()     # one row per donor pool and feature regime
    assert set(t["Feature Regime"]) <= {"Meteo-Only (Climate)", "Meteo + Optical Fusion"}
    # the delivered payload belongs to the winning pool: its best transfer method has the RMSE shown in row 0
    ch = a["choice"]
    row = ch["table"][ch["table"]["method"] == ch["best_transfer"]].iloc[0]
    assert float(row["RMSE"]) * float(a["bundle"].y_to_kg) == pytest.approx(float(t.iloc[0]["Best Forward RMSE (kg/ha)"]), abs=0.1)


# ---------------------------------------------------------------------------------------------------- weather ablation and attribution
def test_weather_ablation_and_attribution_diagnostics():
    from acts_transfer.diagnostics import compute_weather_ablation, compute_weather_driver_attribution
    bundle = diag_bundle()
    ab = compute_weather_ablation(bundle, kg=1000.0)
    for k in ("err_real", "err_ablated", "weather_gain"):
        assert isinstance(ab[k], float) and np.isfinite(ab[k]), (k, ab[k])
    assert ab["weather_gain"] > 0 and ab["err_real"] < ab["err_ablated"]                      # real weather lowers the error
    assert ab["weather_gain"] == pytest.approx(ab["err_ablated"] - ab["err_real"]) and ab["n_folds"] >= 3
    assert "Optical_Valid_Fraction" not in ab["weather_features"] and DRIVER in ab["weather_features"]
    # control: when yield does NOT respond to any weather feature the real weather must not earn a comparable gain
    flat = compute_weather_ablation(diag_bundle(slope=0.0), kg=1000.0)
    assert flat["weather_gain"] < ab["weather_gain"] / 5

    dr = compute_weather_driver_attribution(bundle, kg=1000.0, top_k=6)
    assert dr["Abs Impact"].is_monotonic_decreasing and len(dr) <= 6 and len(dr) >= 1
    assert dr.iloc[0]["Driver"] == DRIVER and dr.iloc[0]["Impact (kg/ha per +1 SD)"] > 0       # the true driver gets the highest positive sensitivity
    assert dr.iloc[0]["Impact (kg/ha per +1 SD)"] == pytest.approx(600.0, rel=0.25)             # about 600 kg/ha per SD was built in
    assert dr["Impact (kg/ha per +1 SD)"].iloc[1:].abs().max() < 0.2 * dr.iloc[0]["Abs Impact"]  # the noise features are small by comparison
    assert len(compute_weather_driver_attribution(bundle, kg=1000.0, top_k=2)) == 2
    assert "Optical_Valid_Fraction" not in set(dr["Driver"])                                    # a static covariate is not a weather driver


# ---------------------------------------------------------------------------------------------------- few-shot scarcity
def _record_fits(monkeypatch):
    from acts_transfer import diagnostics as dg
    fits = []

    class SpyTransfer(dg.AnomalyRidge):
        def fit(self, *a, **k):
            r = super().fit(*a, **k)
            fits.append(("transfer", self))
            return r

    class SpyLocal(dg.LocalRidge):
        def fit(self, *a, **k):
            r = super().fit(*a, **k)
            fits.append(("local", self))
            return r
    monkeypatch.setattr(dg, "AnomalyRidge", SpyTransfer)
    monkeypatch.setattr(dg, "LocalRidge", SpyLocal)
    return fits


def test_few_shot_scarcity_benchmark_outputs(monkeypatch):
    import dataclasses as dc
    from acts_transfer.diagnostics import compute_few_shot_scarcity
    bundle = diag_bundle(n_target=6, n_donor_seasons=12)
    T = bundle.target.dropna(subset=["y"]).sort_values("group")
    assert len(T) >= 5 and bundle.source.group.nunique() >= 10
    fits = _record_fits(monkeypatch)
    out = compute_few_shot_scarcity(bundle, kg=1000.0, shots=(2, 3, 4))
    assert list(out.columns[:4]) == ["Target History", "Local-Only Error (kg/ha)", "Transfer Error (kg/ha)", "Transfer Advantage"]
    assert out["Target History"].tolist() == ["n = 2 seasons", "n = 3 seasons", "n = 4 seasons"]
    errs = out[["Local-Only Error (kg/ha)", "Transfer Error (kg/ha)"]].to_numpy(float)
    assert np.isfinite(errs).all() and (errs >= 0).all()
    assert np.allclose(out["Transfer Advantage"], out["Local-Only Error (kg/ha)"] - out["Transfer Error (kg/ha)"])
    assert out["Test Seasons"].nunique() == 1                                                  # every n is scored on the same held-out seasons
    assert (out["Transfer Advantage"] > 0).all()                                               # donors carry the weather slope the 2-4 local rows cannot learn

    feats, last_g = bundle.features, T.group.iloc[-1]
    assert fits and any(m.meta_["test_season"] == last_g for _, m in fits)                     # the final season is held out in at least one fold
    for kind, m in fits:
        g, n = m.meta_["test_season"], m.meta_["n"]
        x_test = T.loc[T.group == g, feats].values[0]
        y_test = float(T.loc[T.group == g, "y"].iloc[0])
        assert len(m.train_yt_) == n and m.meta_["max_target_group"] < g                      # the n seasons directly before the test season, nothing later
        assert not any(np.allclose(x_test, r) for r in m.train_Xt_) and not np.isclose(m.train_yt_, y_test).any()      # the held-out row is not a training row
        if kind == "local":                                                                    # scaler statistics come from the training slice alone
            np.testing.assert_allclose(m.mean_, m.train_Xt_.mean(axis=0))
            np.testing.assert_allclose(m.scale_, m.train_Xt_.std(axis=0))
        else:
            np.testing.assert_allclose(m.mu_x_, np.nanmean(m.train_Xt_, axis=0))
            assert m.mu_y_ == pytest.approx(m.train_yt_.mean())
            assert m.meta_["max_source_group"] is None or m.meta_["max_source_group"] < g     # donor seasons strictly before the test season

    # sentinel: replace the LAST season (features and yield) with absurd values. No training slice, scaler statistic or fitted slope may change.
    before = [(k, m.coef_.copy() if k == "transfer" else m.m_.coef_.copy(), (m.mu_x_ if k == "transfer" else m.mean_).copy()) for k, m in fits]
    fits.clear()
    tgt = bundle.target.copy()
    last = tgt.group == last_g
    tgt.loc[last, feats] = 1e6
    tgt.loc[last, "y"] = 99.0
    compute_few_shot_scarcity(dc.replace(bundle, target=tgt), kg=1000.0, shots=(2, 3, 4))
    after = [(k, m.coef_.copy() if k == "transfer" else m.m_.coef_.copy(), (m.mu_x_ if k == "transfer" else m.mean_).copy()) for k, m in fits]
    assert len(before) == len(after)
    for (k1, c1, mu1), (k2, c2, mu2) in zip(before, after):
        assert k1 == k2
        np.testing.assert_array_equal(c1, c2)
        np.testing.assert_array_equal(mu1, mu2)


def test_few_shot_scarcity_skips_shots_the_history_cannot_support():
    from acts_transfer.diagnostics import compute_few_shot_scarcity
    out = compute_few_shot_scarcity(diag_bundle(n_target=3, n_donor_seasons=10), kg=1000.0, shots=(2, 3, 4))
    assert out["Target History"].tolist() == ["n = 2 seasons"]                                  # n = 3 and 4 leave no season to hold out


# ---------------------------------------------------------------------------------------------------- safeguard gate and governance registry
def test_safeguard_gate_and_governance_registry_integrity(automl, tmp_path):
    from acts_transfer import governance as gov
    from acts_transfer.modelchoice import LOCAL_BASELINES
    from acts_transfer.publish import build_published_from_automl
    from acts_transfer.sweep import clean_physical_ranges
    from dashboard.tab_transfer import _evidence
    reg = tmp_path / "governance_registry.json"                                                # never the real registry

    # ---- Delta_NT >= 0: the gate triggers, only a local baseline can be served, and approval cannot be recorded.
    # The triggered case is built from a real payload instead of naming a granary that happens to fail today: whether a given granary sits on the right side of
    # the gate moves with the candidates it searches (Kemasin Semerak went from +25 to -2 kg/ha when the "Meteo Anomalies" regime was added: one more candidate was
    # enough for the best of them to dip 2 kg/ha under the line), and a test must not need a granary to fail.
    a_fb = copy.deepcopy(automl[STABLE])
    ch = a_fb["choice"]
    ch["safeguard"]["delta_nt"] = 0.05
    ch["safeguard"]["gate_passed"] = False
    ch["transfer_supported"] = False
    ch["selected"] = "persistence"
    sg = ch["safeguard"]
    assert sg["delta_nt"] >= 0 and sg["gate_passed"] is False
    assert ch["transfer_supported"] is False and ch["selected"] in LOCAL_BASELINES            # no donor-trained model is the model of record
    assert gov.gate_status(ch) == gov.REJECTED
    ev = _evidence(a_fb, ch, sg["gate_passed"], ch["best_transfer"])
    assert ev["gate_passed"] is False
    with pytest.raises(ValueError, match="cannot approve"):
        gov.record_signoff(STABLE, "APPROVE", "Validator", "try", ev, path=reg)
    assert not reg.exists()                                                                    # the refusal wrote nothing
    entry = gov.record_signoff(STABLE, "REJECT", "Validator", "gate triggered", ev, path=reg)
    assert entry["status"] == gov.REJECTED
    eff = gov.effective(STABLE, path=reg)
    assert eff["serves_transfer"] is False and "Local Baseline" in eff["badge"]

    # ---- Delta_NT < 0: the gate passes and the evidence stored with the sign-off is complete
    a = automl[VOLATILE]
    ch = a["choice"]
    sg = ch["safeguard"]
    assert sg["delta_nt"] < 0 and sg["gate_passed"] is True
    assert ch["transfer_supported"] is True and ch["selected"] not in LOCAL_BASELINES
    assert gov.gate_status(ch) == gov.PENDING
    ev = _evidence(a, ch, sg["gate_passed"], ch["selected"])
    kg = float(a["bundle"].y_to_kg)
    assert ev["gate_passed"] is True and ev["model_key"] == ch["selected"]
    assert ev["delta_nt_kg_ha"] < 0 and ev["delta_nt_kg_ha"] == pytest.approx(ev["delta_nt"] * kg)
    assert {"step1_data", "step2_domain", "step3_automl", "step4_gate", "donor_winner"} <= set(ev["automl_scores"])
    assert ev["seasons_compared"] >= 3 and ev["domain_ok"] is True and all(np.isfinite(v) for v in ev["rmse_common_kg_ha"].values())
    Y, _ = clean_physical_ranges(pd.read_csv(ROOT / "data" / "transfer_stage_dataset.csv"))
    pub = build_published_from_automl(a, Y, VOLATILE)
    bt = pd.DataFrame(pub["backtest"]["forward"])
    assert len(bt) >= 3 and np.isfinite(bt[["actual", "baseline"]].to_numpy(float)).all() and bt["transfer"].notna().any()
    assert pub["outlook"] and all(len(v.get(ch["selected"], [])) >= 1 for v in pub["outlook"].values())
    assert pub["nowcast"]["transfer"] is not None and pub["transfer_model"] == ch["selected"]
    entry = gov.record_signoff(VOLATILE, "APPROVE", "Validator", "gate passed", ev, pub, path=reg)
    assert entry["status"] == gov.APPROVED and entry["evidence"]["delta_nt_kg_ha"] == ev["delta_nt_kg_ha"]
    assert gov.effective(VOLATILE, path=reg)["serves_transfer"] is True
    assert gov.effective(STABLE, path=reg)["serves_transfer"] is False                         # the other granary is unaffected



# ---------------------------------------------------------------------------------------------------- a stale precomputed cache must not be served
def test_a_precomputed_payload_is_served_only_for_the_same_data_and_the_same_code(tmp_path, monkeypatch):
    import pickle
    from acts_transfer import automl_runner as R
    p = tmp_path / "automl_precomputed.pkl"
    monkeypatch.setattr(R, "PRECOMPUTED_PATH", p)
    assert R._load_precomputed("G", 1.0) is None                                               # no file
    blob = {"mtime_token": 1.0, "code_token": R.code_token(), "granaries": {"G": {"marker": "ok"}}}
    p.write_bytes(pickle.dumps(blob))
    assert R._load_precomputed("G", 1.0) == {"marker": "ok"}                                   # same data, same code
    assert R._load_precomputed("G", 2.0) is None                                               # other data
    assert R._load_precomputed("H", 1.0) is None                                               # granary not in the file
    p.write_bytes(pickle.dumps({**blob, "code_token": "built-by-older-code"}))
    assert R._load_precomputed("G", 1.0) is None                                               # built by other code: recompute instead
    p.write_bytes(pickle.dumps({"mtime_token": 1.0, "granaries": {"G": {"marker": "ok"}}}))
    assert R._load_precomputed("G", 1.0) is None                                               # an old file with no code stamp is never trusted


# ---------------------------------------------------------------------------------------------------- speed: the dashboard must not wait for a recomputation
def test_the_dashboard_may_show_a_payload_from_older_code_but_flagged_and_the_tests_may_not(tmp_path, monkeypatch):
    import pickle
    from acts_transfer import automl_runner as R
    p = tmp_path / "automl_precomputed.pkl"
    monkeypatch.setattr(R, "PRECOMPUTED_PATH", p)
    now = R.code_token()
    p.write_bytes(pickle.dumps({"mtime_token": 1.0, "code_token": now, "built_with": {"NEW": now, "OLD": "older-code"},
                                "granaries": {"NEW": {"marker": "new"}, "OLD": {"marker": "old"}}}))
    assert R._load_precomputed("NEW", 1.0) == {"marker": "new"}                                # each granary carries its own stamp
    assert R._load_precomputed("OLD", 1.0) is None                                             # strict (tests, recompute): older code is refused
    assert R._load_precomputed("OLD", 1.0, allow_stale=True) == {"marker": "old", "stale_code": True}      # dashboard: instant, flagged
    assert "stale_code" not in R._load_precomputed("NEW", 1.0, allow_stale=True)
    assert R._load_precomputed("OLD", 2.0, allow_stale=True) is None                           # other data is never served, however lenient


def test_the_precomputed_file_is_read_once_until_it_changes(tmp_path, monkeypatch):
    import pickle
    from acts_transfer import automl_runner as R
    p = tmp_path / "automl_precomputed.pkl"
    monkeypatch.setattr(R, "PRECOMPUTED_PATH", p)
    R._BLOB_MEMO.clear()
    p.write_bytes(pickle.dumps({"mtime_token": 1.0, "code_token": R.code_token(), "granaries": {"G": {"v": 1}}}))
    first = R._read_blob()
    assert R._read_blob() is first                                                             # the same object: not unpickled again
    p.write_bytes(pickle.dumps({"mtime_token": 1.0, "code_token": R.code_token(), "granaries": {"G": {"v": 2}, "H": {}}}))
    assert R._read_blob() is not first and R._load_precomputed("G", 1.0) == {"v": 2}           # a changed file is re-read


def _precompute_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("precompute_automl_under_test", ROOT / "pipeline" / "precompute_automl.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_the_precompute_plan_reuses_what_is_current_and_only_computes_what_is_asked_for():
    pc = _precompute_module()
    names = ["A", "B", "C"]
    blob = {"mtime_token": 5.0, "code_token": "v2", "built_with": {"A": "v2", "B": "v1"}, "granaries": {"A": {"a": 1}, "B": {"b": 1}}}
    kept, stamps, todo = pc.plan_work(blob, 5.0, "v2", names)
    assert todo == ["B", "C"] and set(kept) == {"A", "B"}                                      # A is current; B was built by older code; C is missing
    assert pc.plan_work(blob, 5.0, "v2", names, only=["A"])[2] == []                           # nothing to do for a current granary
    assert pc.plan_work(blob, 5.0, "v2", names, only=["C"])[2] == ["C"] and set(pc.plan_work(blob, 5.0, "v2", names, only=["C"])[0]) == {"A", "B"}      # --only keeps the rest
    assert pc.plan_work(blob, 5.0, "v2", names, only=["A"], force=True)[2] == ["A"]
    assert pc.plan_work(blob, 6.0, "v2", names)[0] == {} and pc.plan_work(blob, 6.0, "v2", names)[2] == names      # other data: everything is recomputed
    assert pc.plan_work(None, 5.0, "v2", names)[2] == names
    with pytest.raises(ValueError, match="unknown granary"):
        pc.plan_work(blob, 5.0, "v2", names, only=["Z"])


def test_importing_the_evaluation_module_does_not_load_tensorflow():
    import subprocess
    code = "import sys; import acts_transfer.evaluation, acts_transfer.automl_runner; print('tensorflow' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    assert r.stdout.strip().splitlines()[-1] == "False"                                        # TensorFlow (about 14 s to import) is only loaded by the one method that needs it
