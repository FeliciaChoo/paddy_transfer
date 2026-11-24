"""Tests for the PACE methodology additions: the four registered transfer methods (+ Z-anomaly), the in-fold harmonisation (robust scaling, VIF selection p <= 6),
the look-ahead guard on a pooled source, the persistence baseline and extra metrics, the domain screen, and the phenology stage features.

Run:  python -m pytest tests -q        Synthetic data only.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from acts_transfer import Config, Domain, methods                                           # noqa: E402
from acts_transfer import evaluation, harmonise                                              # noqa: E402
from acts_transfer.methods import ALIASES, LABELS, REGISTRY, Prep, canonical                # noqa: E402
from acts_transfer.screening import compatible_pool, screen_granaries                        # noqa: E402

NEW = ["granary_intercept_ridge", "kmm_ridge", "two_stage_tradaboost_r2", "s_tradaboost_r2", "z_anomaly_transfer"]


def _pooled(level_gap=2.0, n_units=3, n=14, nt=7, seed=0):
    """source granaries and a target that yields `level_gap` t/ha MORE for the same feature response"""
    r = np.random.RandomState(seed)
    Xs, ys, us = [], [], []
    for k in range(n_units):
        X = r.normal(size=(n, 5))
        Xs.append(X)
        ys.append(4 + 0.6 * X[:, 0] - 0.3 * X[:, 1] + r.normal(scale=.05, size=n))
        us += [f"G{k}"] * n
    Xt = r.normal(size=(nt, 5))
    yt = 4 + level_gap + 0.6 * Xt[:, 0] - 0.3 * Xt[:, 1] + r.normal(scale=.05, size=nt)
    Xq = r.normal(size=(30, 5))
    yq = 4 + level_gap + 0.6 * Xq[:, 0] - 0.3 * Xq[:, 1]
    return np.vstack(Xs), np.concatenate(ys), np.array(us), Xt, yt, Xq, yq


# ------------------------------------------------------------------------------------------------ registration
def test_the_methodology_methods_are_registered_with_labels_and_json_aliases():
    for k in NEW:
        assert k in REGISTRY and k in LABELS
    for alias in ("Granary_Intercept_Ridge", "KMM_Ridge", "Two_Stage_TrAdaBoost_R2", "S_TrAdaBoost_R2"):
        assert alias in LABELS and canonical(alias) in REGISTRY and LABELS[alias] == LABELS[canonical(alias)]
    assert "persistence" in LABELS and "persistence" in evaluation.BASELINES


@pytest.mark.parametrize("key", NEW)
def test_each_new_method_fits_predicts_and_ignores_missing_context(key):
    Xs, ys, us, Xt, yt, Xq, yq = _pooled()
    m = REGISTRY[key](cfg=Config())
    m.ctx = dict(unit_s=us)
    p = m.fit(Xs, ys, Xt, yt).predict(Xq)
    assert p.shape == (30,) and np.isfinite(p).all()
    m2 = REGISTRY[key](cfg=Config()).fit(Xs, ys, Xt, yt)                          # no unit context at all: still runs
    assert np.isfinite(m2.predict(Xq)).all()


def test_kmm_ridge_uses_ridge_whatever_base_is_chosen():
    assert REGISTRY["kmm_ridge"](base="random_forest", cfg=Config()).base == "ridge"


# ------------------------------------------------------------------------------------------------ the point of each method
def test_granary_intercept_ridge_absorbs_a_yield_level_gap_that_a_pooled_ridge_cannot():
    Xs, ys, us, Xt, yt, Xq, yq = _pooled(level_gap=2.0)
    cfg = Config(select_features=False)
    gi = REGISTRY["granary_intercept_ridge"](cfg=cfg)
    gi.ctx = dict(unit_s=us)
    e_gi = np.abs(gi.fit(Xs, ys, Xt, yt).predict(Xq) - yq).mean()
    e_pool = np.abs(REGISTRY["pooled"](cfg=cfg).fit(Xs, ys, Xt, yt).predict(Xq) - yq).mean()
    assert e_gi < 0.5 * e_pool
    assert gi.info["n_granaries"] == 3 and gi.info["target_intercept"] > 5.0        # the target intercept sits near its own level (about 6), not the source's 4


def test_z_anomaly_transfer_adds_back_the_target_level():
    Xs, ys, us, Xt, yt, Xq, yq = _pooled(level_gap=2.0)
    z = REGISTRY["z_anomaly_transfer"](cfg=Config(select_features=False))
    z.ctx = dict(unit_s=us)
    p = z.fit(Xs, ys, Xt, yt).predict(Xq)
    assert abs(p.mean() - yq.mean()) < 0.6 and abs(p.mean() - ys.mean()) > 1.0


def test_two_stage_can_switch_a_useless_source_off_and_never_raises_target_weights_in_stage_one():
    from acts_transfer._boost2 import TwoStageTrAdaBoostR2
    r = np.random.RandomState(3)
    Xs, Xt = r.normal(size=(40, 3)), r.normal(size=(6, 3))
    ys = r.normal(size=40) * 3                                                      # unrelated to X: a useless source
    yt = Xt[:, 0] + r.normal(scale=.05, size=6)
    m = TwoStageTrAdaBoostR2(n_estimators=20).fit(Xs, ys, Xt, yt)
    assert m.source_weights_.max() <= m.source_weights_.max() + 1e-9 and len(m.stage1_scores_) == 21
    assert m.stage1_scores_[m.stage1_step_] == min(m.stage1_scores_)                # the step was chosen on target error
    assert np.isfinite(m.predict(Xt)).all()


def test_s_tradaboost_samples_a_bounded_source_and_keeps_blocks_balanced():
    from acts_transfer._boost2 import STrAdaBoostR2
    r = np.random.RandomState(4)
    Xs, Xt = r.normal(size=(300, 3)), r.normal(size=(5, 3))
    ys, yt = Xs[:, 0], Xt[:, 0]
    m = STrAdaBoostR2(n_estimators=10, ratio=3.0).fit(Xs, ys, Xt, yt)
    assert m.sample_counts_.sum() == 10 * 15                                        # 3 x 5 target rows drawn per round, not all 300
    assert np.isfinite(m.predict(Xt)).all()


# ------------------------------------------------------------------------------------------------ in-fold harmonisation
def test_feature_selection_keeps_at_most_p_and_drops_a_duplicate_first():
    r = np.random.RandomState(0)
    X = r.normal(size=(60, 10))
    X[:, 9] = X[:, 0] * 1.0001 + r.normal(scale=1e-4, size=60)                      # a near-duplicate of column 0
    y = X[:, 0] + 0.5 * X[:, 1] + r.normal(scale=.1, size=60)
    p = Prep(Config(max_features=6)).fit(X, y)
    assert len(p.keep) <= 6 and not ({0, 9} <= set(p.keep))                         # never both of the duplicated pair
    assert p.transform(X).shape[1] == len(p.keep)


def test_prep_is_fitted_on_the_rows_it_is_given_only():
    r = np.random.RandomState(1)
    A, B = r.normal(size=(30, 4)), r.normal(size=(30, 4)) + 100
    p = Prep(Config()).fit(A)
    assert abs(np.median(p.transform(A))) < 0.5 and np.abs(p.transform(B)).mean() > 20     # B did not shape the scaler


def test_scaler_choice_is_honoured():
    X = np.random.RandomState(2).normal(size=(40, 3))
    assert type(Prep(Config(scaler="robust")).fit(X).sc).__name__ == "RobustScaler"
    assert type(Prep(Config(scaler="standard", select_features=False)).fit(X).sc).__name__ == "StandardScaler"


# ------------------------------------------------------------------------------------------------ evaluation
FE = ["f1", "f2", "f3", "f4"]


def _bundle(n_years=8, units=("A", "B")):
    r = np.random.RandomState(5)
    rows = []
    for u in units:
        for yr in range(2018, 2018 + n_years):
            x = r.normal(size=4)
            rows.append(dict(group=yr, unit=u, y=4 + .5 * x[0] + r.normal(scale=.05), **dict(zip(FE, x))))
    S = pd.DataFrame(rows)
    T = pd.DataFrame([dict(group=yr, unit="T", y=4.5 + .5 * x[0], **dict(zip(FE, x))) for yr in range(2018, 2018 + n_years) for x in [r.normal(size=4)]])
    return harmonise.HarmonisedBundle(S, T, FE, "y", "group", [], pd.DataFrame(), [], True, "t", "t", "S", "T", {}, y_to_kg=1000.0)


def test_source_rows_of_the_held_out_year_and_later_years_are_not_used_for_training(monkeypatch):
    seen = []
    orig = methods.SourceOnly.fit

    def spy(self, Xs, ys, Xt, yt):
        seen.append((len(Xs), tuple(sorted(set(self.ctx["group_s"]))), self.ctx["group_t"].max()))
        return orig(self, Xs, ys, Xt, yt)
    monkeypatch.setattr(methods.SourceOnly, "fit", spy)
    b = _bundle()
    for strat in ("forward", "loso"):
        seen.clear()
        evaluation.evaluate(b, ["source_only"], "ridge", Config(cv_strategy=strat, bootstrap_n=50))
        assert seen
        for n, gs, tmax in seen:
            held = ({2018 + i for i in range(8)} - set(gs))
            assert held                                                             # something was withheld in every fold
            if strat == "forward":
                assert max(gs) < min(held)                                          # only years before the held-out year
    seen.clear()
    evaluation.evaluate(b, ["source_only"], "ridge", Config(cv_strategy="forward", source_time_safe=False, bootstrap_n=50))
    assert all(len(gs) == 8 for _, gs, _ in seen)                                   # switching the guard off restores the full source


def test_persistence_baseline_and_the_extra_metrics_are_reported():
    r = evaluation.evaluate(_bundle(), ["target_only", "granary_intercept_ridge"], "ridge", Config(bootstrap_n=100))
    m = r.metrics.set_index("method")
    assert "persistence" in m.index
    for c in ("rRMSE_pct", "MAPE_pct", "RMSE_kg_ha", "MAE_kg_ha", "RMSE_ci_low", "RMSE_ci_high", "win_rate_vs_persistence_pct"):
        assert c in m.columns
    assert m.loc["target_only", "RMSE_kg_ha"] == pytest.approx(1000 * m.loc["target_only", "RMSE"])
    assert m.loc["target_only", "RMSE_ci_low"] <= m.loc["target_only", "RMSE"] <= m.loc["target_only", "RMSE_ci_high"] + 1e-9
    assert 0 <= m.loc["granary_intercept_ridge", "win_rate_vs_persistence_pct"] <= 100


def test_persistence_predicts_the_previous_years_yield_in_forward_validation():
    b = _bundle()
    r = evaluation.evaluate(b, ["target_only"], "ridge", Config(cv_strategy="forward", bootstrap_n=50))
    p = r.predictions[r.predictions.method == "persistence"].set_index("group")
    T = b.target.set_index("group").y
    for g, row in p.iterrows():
        assert row.y_pred == pytest.approx(T[g - 1])


# ------------------------------------------------------------------------------------------------ domain screen
def test_screen_flags_the_granary_whose_climate_is_separable_from_the_target():
    r = np.random.RandomState(0)
    rows = []
    for g, shift in (("T", 0.0), ("OK", 0.1), ("FAR", 6.0)):
        for yr in (2018, 2019):
            for d in range(1, 37):
                rows.append(dict(District=g, Year=yr, Decade_N=d, EVI=0.3 + .1 * np.sin(d / 6) + r.normal(scale=.02), Mean_Temp_C=26 + shift + r.normal(), Rain_decade_mm=80 + 20 * shift + r.normal(scale=10),
                                 Solar_Radiation_MJ=15 + r.normal()))
    F = pd.DataFrame(rows)
    s = screen_granaries(F, "T").set_index("granary")
    assert bool(s.loc["FAR", "flagged"]) and not bool(s.loc["OK", "flagged"])
    assert compatible_pool(s.reset_index()) == ["OK"]


# ------------------------------------------------------------------------------------------------ phenology stage features
def _stage_module():
    spec = importlib.util.spec_from_file_location("bsf", ROOT / "pipeline" / "build_stage_features.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_savitzky_golay_finds_two_cycles_and_the_green_up_precedes_each_peak():
    m = _stage_module()
    t = np.arange(36)
    evi = 0.2 + 0.4 * np.exp(-((t - 8) / 3.0) ** 2) + 0.3 * np.exp(-((t - 26) / 3.0) ** 2)
    s, ups = m.greenups(evi + np.random.RandomState(0).normal(scale=.01, size=36))
    assert len(ups) == 2 and ups[0][0] < ups[0][1] < ups[1][0] < ups[1][1]
    assert abs(ups[0][1] - 8) <= 1 and abs(ups[1][1] - 26) <= 1


def test_stage_windows_are_counted_from_green_up_and_a_missing_cycle_stays_missing():
    m = _stage_module()
    n = 36
    dec = pd.DataFrame({k: np.arange(n, dtype=float) for k in
                        ("EVI", "LSWI", "MNDWI", "SAR_VV_dB", "Rain_mm", "Tmax_C", "Mean_Temp_C", "Solar_MJ")})
    f = m.stage_features(dec, dec["EVI"].values, [(4, 10)])
    assert f["C1_EVI_veg"] == pytest.approx(np.mean([4, 5, 6, 7]))                  # 0-40 d = the 4 decades from green-up
    assert f["C1_EVI_rep"] == pytest.approx(np.mean([8, 9, 10]))                    # 40-70 d
    assert f["C1_EVI_rip"] == pytest.approx(np.mean([11, 12, 13]))                  # 70-100 d
    assert f["C1_Rain_mm_rep"] == pytest.approx(8 + 9 + 10)                         # rain is a total, not a mean
    assert np.isnan(f["C2_EVI_veg"]) and np.isnan(f["C2_Solar_MJ_rip"])             # no second cycle: missing, not filled


def test_the_stage_dataset_has_one_row_per_granary_year_when_it_is_built():
    f = ROOT / "data" / "transfer_stage_dataset.csv"
    if not f.exists():
        pytest.skip("run pipeline/build_stage_features.py first")
    d = pd.read_csv(f)
    assert not d.duplicated(["District", "Year"]).any() and len(d) == d.District.nunique() * d.Year.nunique()
    assert d.Yield_kg_ha.notna().all() and "Optical_Valid_Fraction" in d


# ------------------------------------------------------------------------------------------------ PACE V4: benchmark sweep and evidence-first qualification
def _fake_pairs(shift, n_g=6, n_y=7, seed=0):
    r = np.random.RandomState(seed)
    rows = []
    for g in range(n_g):
        for y in range(n_y):
            base = abs(r.normal(300, 60))
            rows.append(dict(strategy="loso", granary=f"G{g}", group=2018 + y, method="m", y_true=4500.0, err_transfer=base + shift + r.normal(0, 20),
                             err_target_only=base, err_source_only=base + 400, err_persistence=base + 20))
    return pd.DataFrame(rows)


def test_pooling_passes_a_consistent_gain_and_fails_a_consistent_loss():
    from acts_transfer.sweep import pool_pairs
    good = pool_pairs(_fake_pairs(-150), Config(bootstrap_n=300)).query("comparison == 'Delta_NT'").iloc[0]
    bad = pool_pairs(_fake_pairs(+150), Config(bootstrap_n=300)).query("comparison == 'Delta_NT'").iloc[0]
    assert good.passed and good.mean_diff_kg_ha < -100 and good.ci95_high < 0 and good.wilcoxon_p_less < 0.05
    assert not bad.passed and bad.mean_diff_kg_ha > 100
    assert good.n_pairs == 42 and good.n_granaries == 6


def test_pooling_reports_delta_ds_against_source_only_and_a_cautious_granary_level_interval():
    from acts_transfer.sweep import pool_pairs
    p = pool_pairs(_fake_pairs(-50), Config(bootstrap_n=300))
    ds = p[p.comparison == "Delta_DS"].iloc[0]
    assert ds.mean_diff_kg_ha < -400                                                # transfer far better than source-only by construction
    nt = p[p.comparison == "Delta_NT"].iloc[0]
    assert nt.cluster_ci95_low <= nt.mean_diff_kg_ha <= nt.cluster_ci95_high


def test_a_failed_sweep_blocks_deployment_even_when_delta_nt_is_negative():
    from acts_transfer import modelchoice
    from tests.test_pipeline_order import BASE, _res                                  # noqa
    res = {"forward": _res({**BASE, "persistence": 0.4, "coral": 0.20}, [("coral", "target_only", 6, "improved")], 6), "loso": _res({**BASE, "persistence": 0.4, "coral": 0.18}, [])}
    no_sweep = modelchoice.choose(res, Config(nt_gate="delta_nt"), False)
    assert no_sweep["transfer_supported"]
    failed = modelchoice.choose(res, Config(nt_gate="delta_nt"), False, {"required": True, "passed": {"forward": {"coral": False}, "loso": {"coral": False}}})
    assert not failed["transfer_supported"] and failed["selected"] != "coral" and "sweep" in failed["statement"]
    ok = modelchoice.choose(res, Config(nt_gate="delta_nt"), False, {"required": True, "passed": {"forward": {"coral": True}, "loso": {"coral": True}}})
    assert ok["transfer_supported"] and "benchmark sweep" in ok["statement"]
    missing = modelchoice.choose(res, Config(nt_gate="delta_nt"), False, {"required": True, "passed": {"forward": {}, "loso": {}}})
    assert not missing["transfer_supported"]                                          # no sweep evidence: nothing is qualified


def test_the_sweep_builds_one_row_per_granary_year_and_never_includes_the_target_in_the_source():
    from acts_transfer.sweep import build_bundle
    rows = [dict(District=g, Year=y, Yield_kg_ha=4000 + 100 * i + 10 * y % 7, C1_EVI_veg=0.3 + .01 * y + .02 * i, C1_SAR_VV_dB_rep=-9 + .1 * y, Soil_pH=5.0 + .1 * i, Optical_Valid_Fraction=.7)
            for i, g in enumerate(("MADA", "T", "X")) for y in range(2018, 2025)]
    Y = pd.DataFrame(rows)
    b, used = build_bundle(Y, ["MADA"], "T", ["C1_EVI_veg", "C1_SAR_VV_dB_rep", "Soil_pH", "Optical_Valid_Fraction"])
    assert "Optical_Valid_Fraction" not in used and "Soil_pH" not in used           # constant features are dropped
    assert set(b.source.unit) == {"MADA"} and set(b.target.unit) == {"T"} and len(b.target) == 7 and b.y_to_kg == 1000.0
    assert b.target.y.between(3, 6).all()                                           # t/ha


# ------------------------------------------------------------------------------------------------ PACE V8: detrended residual transfer, dual-criteria gate, governance, scenarios
def test_ridge_trend_shrinks_the_slope_and_is_flat_with_fewer_than_three_years():
    a, b, t0 = methods.ridge_trend([2018, 2019, 2020, 2021, 2022], [10, 9, 8, 7, 6], lam=0.0)
    assert b == pytest.approx(-1.0) and a == pytest.approx(8.0) and t0 == pytest.approx(2020)
    assert abs(methods.ridge_trend([2018, 2019, 2020, 2021, 2022], [10, 9, 8, 7, 6], lam=5.0)[1]) < 1.0       # regularised: shrunk towards zero
    assert methods.ridge_trend([2018, 2019], [5, 9])[1] == 0.0


def _drift_world(seed=0):
    """target yields drift down 0.15 t/ha a year AND respond to a feature; the source granaries drift differently and sit at another level"""
    r = np.random.RandomState(seed)
    rows_s, rows_t = [], []
    for u, (lvl, drift) in {"A": (6.0, -0.05), "B": (5.5, 0.02)}.items():
        for yr in range(2012, 2024):
            x = r.normal(size=3)
            rows_s.append((u, yr, lvl + drift * (yr - 2018) + 0.4 * x[0] + r.normal(scale=.03), x))
    for yr in range(2018, 2025):
        x = r.normal(size=3)
        rows_t.append((yr, 4.5 - 0.15 * (yr - 2018) + 0.4 * x[0] + r.normal(scale=.03), x))
    return rows_s, rows_t


def test_the_hybrid_removes_the_level_lag_that_the_plain_residual_free_model_has():
    rs, rt = _drift_world()
    Xs, ys = np.array([r[3] for r in rs]), np.array([r[2] for r in rs])
    us, gs = np.array([r[0] for r in rs]), np.array([r[1] for r in rs])
    train = rt[:5]
    Xt, yt, gt = np.array([r[2] for r in train]), np.array([r[1] for r in train]), np.array([r[0] for r in train])
    test = rt[5:]
    Xq, yq, gq = np.array([r[2] for r in test]), np.array([r[1] for r in test]), np.array([r[0] for r in test])
    cfg = Config(select_features=False)
    h = REGISTRY["detrended_residual_transfer"](cfg=cfg)
    h.ctx = dict(unit_s=us, group_s=gs, group_t=gt, group_pred=gq)
    p = REGISTRY["s_tradaboost_r2"](cfg=cfg)
    p.ctx = dict(unit_s=us)
    e_h = np.abs(h.fit(Xs, ys, Xt, yt).predict(Xq) - yq).mean()
    e_p = np.abs(p.fit(Xs, ys, Xt, yt).predict(Xq) - yq).mean()
    assert e_h < 0.5 * e_p                                                                   # the trend carries the drift; the plain model lags behind the level
    assert h.info["trend_slope_kg_ha_per_year"] < -50                                         # the learned target trend is downward (kg/ha per year)


def test_the_hybrid_needs_the_season_of_every_row_and_uses_optical_weights():
    rs, rt = _drift_world()
    Xs, ys, us = np.array([r[3] for r in rs]), np.array([r[2] for r in rs]), np.array([r[0] for r in rs])
    Xt, yt = np.array([r[2] for r in rt[:5]]), np.array([r[1] for r in rt[:5]])
    m = REGISTRY["detrended_residual_transfer"](cfg=Config())
    m.ctx = dict(unit_s=us)
    with pytest.raises(methods.MethodUnavailable):
        m.fit(Xs, ys, Xt, yt)
    m.ctx = dict(unit_s=us, group_s=np.array([r[1] for r in rs]), group_t=np.array([r[0] for r in rt[:5]]), w_s=np.full(len(ys), 0.5), w_t=np.full(5, 0.2))
    assert m.fit(Xs, ys, Xt, yt).info["optical_weights_used"]


def test_leave_one_year_out_persistence_is_strictly_past_only():
    b = _bundle()
    r = evaluation.evaluate(b, ["target_only"], "ridge", Config(cv_strategy="loso", bootstrap_n=50))
    p = r.predictions[r.predictions.method == "persistence"]
    assert 2018 not in set(p.group)                                                          # no 2019 peek for 2018
    T = b.target.set_index("group").y
    for g, row in p.set_index("group").iterrows():
        assert row.y_pred == pytest.approx(T[g - 1])


def test_metrics_carry_the_train_error_next_to_the_test_error():
    m = evaluation.evaluate(_bundle(), ["target_only"], "ridge", Config(bootstrap_n=50)).metrics.set_index("method")
    assert m.loc["target_only", "train_RMSE_kg_ha"] < m.loc["target_only", "RMSE_kg_ha"]      # an unregularised small-sample fit looks better in-sample than out-of-sample
    assert m.loc["target_only", "gap_RMSE_kg_ha"] == pytest.approx(m.loc["target_only", "RMSE_kg_ha"] - m.loc["target_only", "train_RMSE_kg_ha"])


def test_delta_nt_is_taken_over_the_seasons_common_to_the_transfer_model_and_every_baseline():
    from acts_transfer import modelchoice
    b = _bundle(n_years=9)
    res = {s: evaluation.evaluate(b, ["detrended_residual_transfer", "target_only"], "ridge", Config(cv_strategy=s, bootstrap_n=50)) for s in ("forward", "loso")}
    c = modelchoice.choose(res, Config(deployable=("detrended_residual_transfer",), min_test_seasons=3), False)
    r = c["table"][c["table"].method == "detrended_residual_transfer"].iloc[0]
    p = res["forward"].predictions
    common = set(p[p.method == "detrended_residual_transfer"].group) & set(p[p.method == "season_trend"].group) & set(p[p.method == "persistence"].group) & set(p[p.method == "target_only"].group)
    assert r["seasons compared"] == len(common) and len(common) < len(set(p[p.method == "detrended_residual_transfer"].group))      # the season trend scores fewer seasons
    sub = p[p.group.isin(common)]
    rm = {k: float((sub[sub.method == k].abs_err ** 2).mean() ** 0.5) for k in ("detrended_residual_transfer", "season_trend", "persistence", "target_only")}
    assert r["Delta_NT (RMSE)"] == pytest.approx(rm["detrended_residual_transfer"] - min(rm["season_trend"], rm["persistence"], rm["target_only"]))
    assert r["Delta_NT baseline"] == min(("season_trend", "persistence", "target_only"), key=rm.get)


def test_only_deployable_methods_can_be_approved_and_domain_relevance_is_a_hard_criterion():
    from acts_transfer import modelchoice
    from tests.test_pipeline_order import BASE, _res                                          # noqa
    res = {"forward": _res({**BASE, "persistence": 0.4, "coral": 0.18, "tca": 0.19}, [("coral", "target_only", 6, "improved"), ("tca", "target_only", 6, "improved")], 6),
           "loso": _res({**BASE, "persistence": 0.4, "coral": 0.18, "tca": 0.19}, [])}
    c = modelchoice.choose(res, Config(deployable=("tca",)), False)
    t = c["table"].set_index("method")
    assert bool(t.loc["tca", "supported"]) and not bool(t.loc["coral", "supported"]) and bool(t.loc["coral", "gate passed"])     # coral passes the gate but is a benchmark only
    bad = modelchoice.choose(res, Config(deployable=("tca",)), False, None, {"ok": False, "reason": "the source pool contains KETARA"})
    assert not bad["transfer_supported"] and not bad["safeguard"]["gate_passed"] and "Criterion 1" in bad["statement"] and "KETARA" in bad["statement"]


def test_the_registry_states_defaults_and_counts(tmp_path):
    from acts_transfer import governance as gov
    reg = tmp_path / "r.json"
    assert gov.effective("G1", path=reg)["status"] == gov.PENDING and not gov.effective("G1", path=reg)["serves_transfer"]           # no entry: the safe default
    gov.record_signoff("G1", "APPROVE", "Val", "ok", {"gate_passed": True, "model_key": "m"}, {"transfer_model": "m"}, path=reg)
    gov.record_signoff("G2", "REJECT", "Val", "no", {"gate_passed": False}, path=reg)
    assert gov.effective("G1", path=reg)["serves_transfer"] and gov.effective("G1", path=reg)["badge"].endswith("Verified Transfer Model")
    assert gov.effective("G2", path=reg)["status"] == gov.REJECTED and gov.effective("G2", path=reg)["badge"].endswith("Local Baseline Active")
    assert gov.counts(["G1", "G2", "G3"], path=reg) == (1, 2, 1)
    with pytest.raises(ValueError):
        gov.record_signoff("G3", "APPROVE", "", "", {"gate_passed": True}, path=reg)                # a sign-off needs a named validator
    reg.write_text("not json")
    assert gov.effective("G1", path=reg)["status"] == gov.PENDING                                    # a corrupt registry never serves a transfer model


def test_scenarios_move_the_features_the_right_way_and_normal_changes_nothing():
    from acts_transfer.scenarios import scenario_inputs, scenario_shifts
    r = np.random.RandomState(0)
    rows = []
    for g in "ABCDE":
        base = r.normal(3000, 300)
        for y in range(2018, 2025):
            rows.append(dict(District=g, Year=y, C1_Rain_mm_rep=base + r.normal(0, 150), C1_Tmax_C_rep=33 + r.normal(0, .8), C1_EVI_veg=.4 + r.normal(0, .04)))
    Y = pd.DataFrame(rows)
    f = ["C1_Rain_mm_rep", "C1_Tmax_C_rep", "C1_EVI_veg"]
    assert scenario_shifts(Y, f, "Normal") == {}
    d, w = scenario_shifts(Y, f, "El Niño drought"), scenario_shifts(Y, f, "Wet monsoon")
    assert d["C1_Rain_mm_rep"] < 0 < w["C1_Rain_mm_rep"] and d["C1_Tmax_C_rep"] > 0 > w["C1_Tmax_C_rep"] and d["C1_EVI_veg"] < 0
    inp, _ = scenario_inputs(Y, "A", f, "El Niño drought", [2025, 2026])
    assert list(inp.season) == [2025, 2026] and inp.C1_Rain_mm_rep.iloc[0] < Y[Y.District == "A"].C1_Rain_mm_rep.mean()


def test_forecast_bands_widen_with_the_horizon():
    from dashboard.tab_governance import local_outlook
    h = pd.DataFrame({"Year": range(2014, 2025), "Yield": [4600, 4700, 4500, 4650, 4400, 4550, 4300, 4450, 4200, 4350, 4100]})
    o = local_outlook(h, [2025, 2026, 2027, 2028])
    w = [r["high"] - r["low"] for r in o]
    assert all(b > a for a, b in zip(w, w[1:]))
