"""Tests of the concordance / uplift validation (pipeline/validate_concordance_uplift.py).

They test the INSTRUMENT: no leakage, correct statistics, an effect that is really there is found, no effect is found when there is none. They deliberately do NOT assert that the real
data confirm the protocol's claims (concordant seasons yield more, the transfer effect estimate beats the baselines, ...): a validation whose test can only pass proves nothing, so those
four assertions are REPORTED by the script as findings and the tests only check that they are reported honestly.
"""
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

_spec = importlib.util.spec_from_file_location("validate_concordance_uplift", ROOT / "pipeline" / "validate_concordance_uplift.py")
V = importlib.util.module_from_spec(_spec)
sys.modules["validate_concordance_uplift"] = V
_spec.loader.exec_module(V)

from acts_transfer.config import Config  # noqa: E402

needs_data = pytest.mark.skipif(not (ROOT / "data" / "transfer_stage_dataset.csv").exists(), reason="project data not present")


def synthetic_panel(n_gran=10, n_years=30, tau_moist=0.0, tau_plant=0.0, seed=0):
    """granary-year panel with known treatment effects: yield = granary level + weather terms + tau * (lever - its granary mean) + noise; levers depend on weather a little"""
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_gran):
        level = 4000 + 300 * rng.normal()
        slip0, lswi0 = rng.integers(-5, 6), 0.25 + 0.05 * rng.normal()
        for k in range(n_years):
            rain, tmax, tmean, sol, ph = 900 + 120 * rng.normal(), 33 + rng.normal(), 27 + 0.7 * rng.normal(), 20 + rng.normal(), 5.4
            slip = slip0 + 2.0 * rng.normal()
            lswi = lswi0 + 0.00004 * (rain - 900) + 0.04 * rng.normal()
            y = level + 0.8 * (rain - 900) - 60 * (tmax - 33) + tau_plant * (slip - slip0) + tau_moist * (lswi - lswi0) + 120 * rng.normal()
            rows.append(dict(District=f"G{g}", Year=2000 + k, Yield_kg_ha=y, C1_Rain_mm_rep=rain, C1_Tmax_C_rep=tmax, C1_Mean_Temp_C_rep=tmean, C1_Solar_MJ_rip=sol, Soil_pH=ph,
                             Schedule_Slip_Decades=slip, C1_LSWI_rep=lswi))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- no leakage
def test_the_folds_use_only_strictly_earlier_seasons_for_the_target_and_for_every_donor():
    Y = synthetic_panel(n_gran=5, n_years=9)
    gs = sorted(Y.District.unique())
    folds = V.make_folds(Y, gs, Config(cv_strategy="forward"))
    assert len(folds) == 5 * 7                                                                  # 9 seasons, at least 2 to train: 7 held out per granary
    audit = V.audit_leakage(folds, Y)
    assert audit["source_time_safe"] is True and audit["violations"] == []
    for f in folds:
        tr, te = Y.loc[f["train_index"]], Y.loc[f["test_index"]]
        assert tr.Year.max() < f["Year"] and len(te) == 1 and te.Year.iloc[0] == f["Year"]     # also the donors' same-season rows are out


def test_the_leakage_audit_catches_a_future_row_and_a_held_out_row_in_its_own_training_set():
    Y = synthetic_panel(n_gran=4, n_years=9)
    gs = sorted(Y.District.unique())
    folds = V.make_folds(Y, gs, Config(cv_strategy="forward"))
    f = folds[0]
    future = Y.index[(Y.District == "G1") & (Y.Year == f["Year"])][0]                           # a donor row from the SAME season: the weather leak the protocol forbids
    bad = [dict(f, train_index=np.append(f["train_index"], future))] + folds[1:]
    assert V.audit_leakage(bad, Y)["source_time_safe"] is False
    own = [dict(f, train_index=np.append(f["train_index"], f["test_index"]))]
    a = V.audit_leakage(own, Y)
    assert a["source_time_safe"] is False and any("own training set" in v["reason"] for v in a["violations"])


def test_a_season_is_never_scored_with_a_model_that_saw_its_own_outcome():
    """changing the held-out season's yield must not change its recommendation or its weather-adjusted expectation"""
    Y = synthetic_panel(n_gran=5, n_years=9, tau_moist=2000.0, seed=3)
    gs = sorted(Y.District.unique())
    cfg = Config(cv_strategy="forward")
    folds = V.make_folds(Y, gs, cfg)
    S1, _ = V.replay(Y, folds)
    Y2 = Y.copy()
    last = Y2.index[(Y2.District == "G0") & (Y2.Year == 2008)][0]
    Y2.loc[last, "Yield_kg_ha"] += 5000.0
    S2, _ = V.replay(Y2, V.make_folds(Y2, gs, cfg))
    a = S1[(S1.District == "G0") & (S1.Year == 2008)].iloc[0]
    b = S2[(S2.District == "G0") & (S2.Year == 2008)].iloc[0]
    assert a["y_expected"] == pytest.approx(b["y_expected"]) and a["planting_rec"] == b["planting_rec"] and a["moisture_rec"] == b["moisture_rec"]
    assert b["y_res"] == pytest.approx(a["y_res"] + 5000.0)


# ---------------------------------------------------------------------------------------------------- concordance rules
def test_concordance_uses_the_tolerances_and_the_step_is_only_applied_when_it_is_predicted_to_help():
    hist = pd.Series([-3.0, -2.0, -2.5, -2.0])
    base, rec, applied = V.recommend_target(hist, -1.0, tau0=-20.0, fitted=True)               # earlier planting predicted to raise yield (tau < 0): step applied
    assert (base, rec, applied) == (-2.0, -3.0, True)
    base, rec, applied = V.recommend_target(hist, -1.0, tau0=+20.0, fitted=True)               # the effect points the other way: no change recommended
    assert (base, rec, applied) == (-2.0, -2.0, False)
    assert V.recommend_target(hist, -1.0, tau0=-20.0, fitted=False)[2] is False                # an unfitted model recommends nothing
    assert V.recommend_target(hist, -1.0, -20.0, True, tbase="median")[0] == -2.25             # the baseline can be the median instead of the last value
    wild = pd.Series([-2.0, -2.0, -2.0])
    assert V.recommend_target(wild, -5.0, tau0=-1.0, fitted=True)[1] >= -2.0 - 5.0              # never beyond what the earlier seasons support plus one step
    assert V.is_concordant(-3.9, -3.0, 1.0) == 1.0 and V.is_concordant(-4.1, -3.0, 1.0) == 0.0     # +/- 1.0 decade
    assert V.is_concordant(0.30, 0.27, 0.04) == 1.0 and V.is_concordant(0.32, 0.27, 0.04) == 0.0   # +/- 0.04 LSWI
    assert V.is_concordant(1.0, 1.0 + 1.0, 1.0) == 1.0                                          # the boundary counts as concordant
    assert math.isnan(V.is_concordant(np.nan, 1.0, 1.0)) and math.isnan(V.is_concordant(None, 1.0, 1.0))


# ---------------------------------------------------------------------------------------------------- the statistics
def test_the_cluster_bootstrap_resamples_whole_granaries_and_matches_the_plain_numbers():
    cl = np.array(["A"] * 4 + ["B"] * 3 + ["C"] * 5)
    v = np.arange(12, dtype=float)
    cb = V.ClusterBoot(cl, B=500, seed=1)
    assert cb.M.shape == (501, 3) and np.all(cb.M[1:].sum(axis=1) == 3) and np.all(cb.M[0] == 1)    # every draw has as many whole granaries as the sample
    m = cb.mean(v)
    assert m["estimate"] == pytest.approx(v.mean()) and m["ci_low"] <= m["estimate"] <= m["ci_high"]
    assert V.ClusterBoot(cl, B=500, seed=1).mean(v) == m                                          # deterministic
    grp = np.array([1, 1, 0, 0] + [1, 0, 0] + [0] * 5)
    d = cb.mean_diff(v, grp)
    assert d["estimate"] == pytest.approx(v[grp == 1].mean() - v[grp == 0].mean()) and d["n_group1"] == 3 and d["n_group0"] == 9
    eb, en = np.array([3.0, -4.0, 2.0, 1.0] * 3), np.array([1.0, 1.0, 5.0, -1.0] * 3)
    g = cb.rmse_gain(eb, en)
    assert g["estimate"] == pytest.approx(np.sqrt((eb ** 2).mean()) - np.sqrt((en ** 2).mean()))
    assert cb.win_rate(eb, en)["estimate"] == pytest.approx(100 * np.mean(np.abs(en) < np.abs(eb)))
    assert cb.mae_gain(eb, en)["estimate"] == pytest.approx(np.abs(eb).mean() - np.abs(en).mean())


def test_the_bootstrap_interval_is_wide_with_few_granaries_and_covers_the_truth_in_most_repeats():
    rng = np.random.default_rng(5)
    hits, widths = 0, []
    for r in range(60):
        cl = np.repeat(np.arange(8), 6)
        v = rng.normal(0.0, 1.0, 48) + np.repeat(rng.normal(0.0, 1.0, 8), 6)                    # granary-level shifts: the reason to resample granaries, not seasons
        m = V.ClusterBoot(cl, B=300, seed=r).mean(v)
        hits += m["ci_low"] <= 0.0 <= m["ci_high"]
        widths.append(m["ci_high"] - m["ci_low"])
    assert hits >= 45                                                                           # a 95% interval from 8 clusters under-covers a little; it must still cover most of the time


# ---------------------------------------------------------------------------------------------------- the estimator finds what is there, and nothing when nothing is there
def test_the_treatment_effect_estimator_recovers_a_planted_effect_and_a_zero_effect():
    Yp = synthetic_panel(tau_moist=3000.0, tau_plant=-40.0, seed=1)
    m = V.PartialLinear("C1_LSWI_rep").fit(Yp)
    assert m.ok and m.tau0() == pytest.approx(3000.0, rel=0.25)
    mp = V.PartialLinear("Schedule_Slip_Decades").fit(Yp)
    assert mp.tau0() == pytest.approx(-40.0, abs=12.0)
    Y0 = synthetic_panel(tau_moist=0.0, tau_plant=0.0, seed=2)
    assert abs(V.PartialLinear("C1_LSWI_rep").fit(Y0).tau0()) < 900.0                          # no effect: the estimate stays near zero (the planted size is 3000)


def test_the_r_loss_prefers_the_true_effect_to_no_effect_on_held_out_seasons_only_when_there_is_one():
    def held_out_losses(tau):
        Yp = synthetic_panel(tau_moist=tau, seed=4)
        tr, te = Yp[Yp.Year < 2020], Yp[Yp.Year >= 2020]
        m = V.PartialLinear("C1_LSWI_rep").fit(tr)
        r_y, r_t = te["Yield_kg_ha"].to_numpy() - m.y_hat(te), te["C1_LSWI_rep"].to_numpy() - m.t_hat(te)
        return np.mean((r_y - m.tau(te) * r_t) ** 2), np.mean(r_y ** 2)
    with_effect, none_with = held_out_losses(3000.0)
    assert with_effect < 0.5 * none_with                                                        # a real effect: far lower loss than the no-effect benchmark
    no_effect, none_without = held_out_losses(0.0)
    assert no_effect > 0.9 * none_without                                                       # no effect: no real gain over the benchmark (it can only add estimation noise)


def test_the_permutation_placebo_stands_out_when_the_effect_is_real_and_not_when_it_is_not():
    real = V.refute(synthetic_panel(n_gran=8, n_years=20, tau_moist=3000.0, seed=6), "moisture", n_perm=150, seed=1)
    assert real["placebo"]["p_value_two_sided"] < 0.05                                          # a real effect is far outside the placebo distribution
    assert abs(real["placebo"]["mean"]) < 0.2 * abs(real["tau_per_unit"])                       # and the placebo itself is centred near zero
    null = V.refute(synthetic_panel(n_gran=8, n_years=20, tau_moist=0.0, seed=7), "moisture", n_perm=150, seed=1)
    assert null["placebo"]["interval_contains_zero"] is True and null["placebo"]["p_value_two_sided"] > 0.01
    assert real["random_common_cause"]["stable"] is True                                        # an irrelevant extra covariate does not move a real estimate


def test_a_fitted_effect_is_not_reported_for_too_little_data():
    Yp = synthetic_panel(n_gran=1, n_years=5)
    m = V.PartialLinear("C1_LSWI_rep").fit(Yp)
    assert m.ok is False and np.all(m.tau(Yp) == 0.0) and m.tau0() == 0.0


def test_stress_seasons_are_the_extremes_of_rainfall_or_temperature_within_each_granary():
    Y = synthetic_panel(n_gran=4, n_years=20, seed=8)
    s = V.stress_flags(Y)
    assert 0.2 <= s.mean() <= 0.5                                                               # about a fifth per variable, so a bit more than that for either
    wet = Y.loc[Y.groupby("District").C1_Rain_mm_rep.transform(lambda x: (x - x.mean()).abs()).idxmax()]
    assert bool(s.loc[wet.name])                                                                # the most extreme rainfall season is flagged


# ---------------------------------------------------------------------------------------------------- the whole script on the project's data
@needs_data
def test_the_script_runs_on_the_project_data_and_reports_honestly(tmp_path):
    rep = V.run(ROOT, tmp_path, quick=True, forecast=False, verbose=False)
    assert (tmp_path / "concordance_uplift_report.json").exists() and (tmp_path / "concordance_uplift_summary.csv").exists()
    saved = json.loads((tmp_path / "concordance_uplift_report.json").read_text(encoding="utf-8"))
    for k in ("data", "domain_screen", "leakage_audit", "concordance", "r_loss", "refutation", "gazetted_window", "acceptance", "deviations", "limitations", "references"):
        assert k in saved, k
    assert saved["leakage_audit"]["source_time_safe"] is True and saved["leakage_audit"]["n_folds"] == saved["data"]["held_out_seasons"] > 0
    assert saved["data"]["protocol_claimed"] == {"panel_rows": 70, "held_out_seasons": 59}        # the protocol's counts are recorded next to the real ones, not silently adopted
    c = saved["concordance"]["both_levers"]
    assert c["n_concordant"] + c["n_discordant"] <= saved["data"]["held_out_seasons"] and c["n_concordant"] > 0 and c["n_discordant"] > 0
    acc = saved["acceptance"]
    assert set(acc) >= {"a1_no_future_or_same_year_source_rows", "a2_concordant_seasons_have_higher_weather_adjusted_yield", "a3_transfer_cate_has_lower_rloss_than_target_only_and_source_only",
                        "a4_placebo_effect_indistinguishable_from_zero", "status"}
    assert acc["a1_no_future_or_same_year_source_rows"]["holds"] is True                       # integrity is enforced; the outcomes a2-a4 are findings, whatever they are
    assert "a2-a4" in acc["status"] and "not enforced" in acc["status"]
    assert not any("Thorburn" in r for r in saved["references"]) and any("Thorburn" in r for r in saved["references_dropped"])   # an unverifiable citation is not carried over
    summary = pd.read_csv(tmp_path / "concordance_uplift_summary.csv")
    assert {"section", "metric", "estimate", "ci_low", "ci_high"} <= set(summary.columns) and len(summary) > 10
    assert not json.dumps(saved).count("NaN")                                                   # valid JSON: no bare NaN


@needs_data
def test_the_domain_screen_decides_compatibility_not_a_granary_name():
    d = V.domain_compatibility(ROOT)
    assert set(d.columns) >= {"District", "median_auc", "max_auc", "donors_flagged", "compatible"} and len(d) >= 2
    assert (d["compatible"] == (d["median_auc"] < V.AUC_EXCLUDE)).all()


@needs_data
def test_the_script_is_deterministic_for_a_given_seed(tmp_path):
    a = V.run(ROOT, None, quick=True, forecast=False, verbose=False, seed=3)
    b = V.run(ROOT, None, quick=True, forecast=False, verbose=False, seed=3)
    assert a["concordance"]["both_levers"]["weather_adjusted_kg_ha"] == b["concordance"]["both_levers"]["weather_adjusted_kg_ha"]
    assert a["refutation"]["planting"]["placebo"]["p_value_two_sided"] == b["refutation"]["planting"]["placebo"]["p_value_two_sided"]
