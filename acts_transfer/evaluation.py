"""Stage 6/7 · Run the methods and compare them, with grouped validation that cannot leak.

Validation   one fold per held-out target season (group).
  loso       leave-one-season-out: train on every other target season. Uses LATER seasons for training, which is acceptable only when seasons are
             few and roughly exchangeable; neighbouring seasons can be correlated, so the result may be optimistic.
  forward    expanding window: train only on EARLIER target seasons (the stricter, forecasting-style check; needs at least 2 earlier seasons).
Inside every fold, ALL fitting uses training rows only: imputation, standardisation, CORAL/TCA alignment, KMM weights, boosting, the ridge
penalty. The held-out season's features and labels are never seen. The source domain is always fully available (it is the knowledge source).
Scarcity   cfg.max_target_train_groups limits how many labelled target seasons a fold may use.
Every method sees the SAME held-out rows, so the errors are comparable.
Look-ahead guard (cfg.source_time_safe, default on): a pooled source contains OTHER granaries in the same years, which share weather and market effects with the
held-out year. Source rows from the held-out year are therefore not used for training, and in forward validation neither is any later year.
Baselines: persistence (the last observed yield of the target, a naive forecast) is scored alongside the season-trend and target-mean baselines.
Metrics: MAE, RMSE, rRMSE (%), MAPE (%), R2, the win rate against persistence, and a bootstrap 95% interval for RMSE and MAE over held-out seasons.
"""
from dataclasses import dataclass, field
from .config import Config
from .harmonise import HarmonisedBundle
from .methods import REGISTRY, MethodUnavailable
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler


def fit_predict_delta_tradaboost(
    S_train: pd.DataFrame,
    T_train: pd.DataFrame,
    T_test: pd.DataFrame,
    feats: list[str],
    alpha: float = 10.0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Fits TwoStageTrAdaBoostR2 on first-differenced yield (Δy = y_t - y_{t-1})
    and feature deltas (ΔX = X_t - X_{t-1}), returning predicted yield in t/ha.
    """
    def _make_deltas(df: pd.DataFrame) -> pd.DataFrame:
        d = df.sort_values(["unit", "group"]).copy()
        d["prev_y"] = d.groupby("unit")["y"].shift(1)
        d["delta_y"] = d["y"] - d["prev_y"]
        for f in feats:
            d[f"{f}_delta"] = d[f] - d.groupby("unit")[f].shift(1)
        return d

    # Combine T_train and T_test briefly just to get strictly backward-looking lag (t-1)
    # without ever touching future labels (y in T_test is masked out)
    T_all = pd.concat([T_train.assign(_is_test=False), T_test.assign(_is_test=True)], ignore_index=True)
    T_all = T_all.sort_values("group").reset_index(drop=True)
    T_all["prev_y"] = T_all["y"].shift(1)
    T_all["delta_y"] = T_all["y"] - T_all["prev_y"]
    for f in feats:
        T_all[f"{f}_delta"] = T_all[f] - T_all[f].shift(1)

    T_tr_d = T_all[~T_all["_is_test"]].dropna(subset=["prev_y", "delta_y"])
    T_te_d = T_all[T_all["_is_test"]].copy()

    # Fallback to persistence (prev_y) if target has < 2 delta seasons in this fold
    fallback_pred = T_te_d["prev_y"].fillna(T_train["y"].mean()).values
    if len(T_tr_d) < 2 or T_te_d["prev_y"].isna().any():
        return np.clip(fallback_pred, 0.0, 12.0)

    S_tr_d = _make_deltas(S_train).dropna(subset=["prev_y", "delta_y"])
    if len(S_tr_d) < 5:
        return np.clip(fallback_pred, 0.0, 12.0)

    delta_cols = [f"{f}_delta" for f in feats]

    # Fold-safe imputation and scaling fit ONLY on source + target training deltas
    imp = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    Xs_raw = imp.fit_transform(S_tr_d[delta_cols])
    Xt_raw = imp.transform(T_tr_d[delta_cols])
    Xte_raw = imp.transform(T_te_d[delta_cols])

    Xs = scaler.fit_transform(Xs_raw)
    Xt = scaler.transform(Xt_raw)
    Xte = scaler.transform(Xte_raw)

    ys = S_tr_d["delta_y"].values
    yt = T_tr_d["delta_y"].values

    cv_folds = min(5, max(2, len(Xt)))
    from adapt.instance_based import TwoStageTrAdaBoostR2        # imported here, not at the top: it pulls in TensorFlow (about 14 s), which only this method needs
    model = TwoStageTrAdaBoostR2(
        estimator=Ridge(alpha=alpha),
        Xt=Xt,
        yt=yt,
        n_estimators=10,
        n_estimators_fs=10,
        cv=cv_folds,
        random_state=random_state,
        verbose=0,
    )
    model.fit(Xs, ys)

    pred_delta = model.predict(Xte)
    pred_y_tha = T_te_d["prev_y"].values + pred_delta
    return np.clip(pred_y_tha, 0.0, 12.0)

BASELINES = ("source_only", "target_only", "target_mean", "season_trend", "persistence", "pooled")

YIELD_MAX_KG_HA = 12000.0        # biological upper bound of a paddy yield (the same bound the Delta learner uses); the lower bound is 0


def plausible_label_bounds(bundle):
    """(low, high) in the units of the harmonised label, or None when the label does not look like a yield.
    A yield cannot be negative or above 12,000 kg/ha. The bound is applied only when every observed label (source and target) already lies inside it, so a
    bundle whose label is something else (a dB value, an index) is never clipped. It is a constant, so it cannot leak anything from a held-out season."""
    kg = float(getattr(bundle, "y_to_kg", 1.0)) or 1.0
    lab = np.concatenate([np.asarray(bundle.source["y"], float), np.asarray(bundle.target["y"], float)])
    lab = lab[np.isfinite(lab)]
    lo, hi = 0.0, YIELD_MAX_KG_HA / kg
    return (lo, hi) if len(lab) and lab.min() >= lo and lab.max() <= hi else None


@dataclass
class EvaluationResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    improvement: pd.DataFrame
    per_group: pd.DataFrame
    fold_info: list
    failures: list
    strategy: str
    base: str
    n_folds: int
    n_target_train: list
    methods: list
    config: dict
    notes: list = field(default_factory=list)


def _folds(groups, cfg):
    out = []
    for i, g in enumerate(groups):
        if cfg.cv_strategy == "forward":
            train = groups[:i]
            if len(train) < 2:
                continue
        else:
            train = [x for x in groups if x != g]
        if cfg.max_target_train_groups:
            k = cfg.max_target_train_groups
            train = sorted(train, key=lambda x: abs(x - g))[:k] if all(isinstance(x, (int, float, np.integer)) for x in train) else train[-k:]
        out.append((g, train))
    return out


def _boot_ci(d, n, seed=0):
    r = np.random.RandomState(seed)
    d = np.asarray(d, float)
    m = [d[r.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def _persistence(T, g, train_groups):
    """the target's last observed yield BEFORE g among the training seasons. Strictly past-only: for the earliest season there is no persistence forecast (NaN),
    so it is not scored. (An earlier version borrowed the next year for the first season; that peeked into the future.)"""
    tr = T[T.group.isin(train_groups)].sort_values("group")
    before = tr[tr.group < g]
    return float(before.iloc[-1].y) if len(before) else np.nan


def _boot_stat(err, fn, n, seed=0):
    r = np.random.RandomState(seed)
    err = np.asarray(err, float)
    v = [fn(err[r.randint(0, len(err), len(err))]) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


def evaluate(bundle: HarmonisedBundle, methods, base, cfg: Config):
    if not isinstance(bundle, HarmonisedBundle):
        raise TypeError("evaluate() accepts only a HarmonisedBundle, which only harmonise() can create from a completed, non-blocked assessment")
    feats = bundle.features
    S = bundle.source
    Xs_all, ys_all = S[feats].values, S.y.values
    us_all = S["unit"].astype(str).values if "unit" in S else np.array(["source"] * len(S))
    gs_all = S["group"].values
    ws_all = S["w"].values if "w" in S else np.ones(len(S))                       # optical-quality sample weights w = max(0.2, Optical_Valid_Fraction)
    T = bundle.target.dropna(subset=["y"])
    groups = sorted(T.group.unique().tolist())
    rows, info, fails, ntr = [], [], [], []
    bounds = plausible_label_bounds(bundle)
    n_clipped = 0
    for g, train_groups in _folds(groups, cfg):
        tr, te = T[T.group.isin(train_groups)], T[T.group == g]
        Xt, yt = tr[feats].values, tr.y.values
        ntr.append(len(tr))
        if cfg.source_time_safe:
            keep = (gs_all < g) if cfg.cv_strategy == "forward" else (gs_all != g)
        else:
            keep = np.ones(len(gs_all), bool)
        Xs, ys, us, gs, wsx = Xs_all[keep], ys_all[keep], us_all[keep], gs_all[keep], ws_all[keep]
        wt = tr["w"].values if "w" in tr else np.ones(len(tr))
        for key in methods:
            try:
                m = REGISTRY[key](base=base, seed=cfg.random_state, cfg=cfg)
                m.ctx = dict(unit_s=us, group_s=gs, group_t=tr.group.values, w_s=wsx, w_t=wt)
                m.fit(Xs, ys, Xt, yt)
                m.ctx["group_pred"] = te.group.values
                p = np.asarray(m.predict(te[feats].values), float)
                m.ctx["group_pred"] = tr.group.values
                p_tr = np.asarray(m.predict(Xt), float)
                if bounds is not None:       # a ridge fitted on 2-3 target rows can extrapolate to 13-18 t/ha: bound every prediction to the biological range
                    n_clipped += int(((p < bounds[0]) | (p > bounds[1])).sum())
                    p, p_tr = np.clip(p, *bounds), np.clip(p_tr, *bounds)
            except MethodUnavailable as e:
                fails.append(dict(method=key, group=g, reason=str(e)))
                continue
            except Exception as e:
                fails.append(dict(method=key, group=g, reason=f"{type(e).__name__}: {e}"))
                continue
            for yi, pi in zip(te.y.values, p):
                rows.append(dict(method=key, group=g, y_true=yi, y_pred=pi, abs_err=abs(yi - pi)))
            info.append(dict(method=key, group=g, n_target_train=len(tr), train_mae_target=float(np.mean(np.abs(yt - p_tr))), train_mse_target=float(np.mean((yt - p_tr) ** 2)), **m.info))
    # the pure time-trend baseline: y ~ season fitted on the training seasons only. If the features cannot beat this, they may only be proxying time.
    numeric_groups = all(isinstance(x, (int, float, np.integer)) for x in groups)
    if numeric_groups:
        for g, train_groups in _folds(groups, cfg):
            tr, te = T[T.group.isin(train_groups)], T[T.group == g]
            if len(tr) >= 3:
                c = np.polyfit(tr.group.astype(float), tr.y.values, 1)
                for yi, gi in zip(te.y.values, te.group.values):
                    rows.append(dict(method="season_trend", group=g, y_true=yi, y_pred=float(np.polyval(c, gi)), abs_err=abs(yi - np.polyval(c, gi))))
    # persistence: last observed yield of the target (needs no features, no source)
    for g, train_groups in _folds(groups, cfg):
        te = T[T.group == g]
        pv = _persistence(T, g, train_groups)
        if not np.isnan(pv):
            for yi in te.y.values:
                rows.append(dict(method="persistence", group=g, y_true=yi, y_pred=pv, abs_err=abs(yi - pv)))
    pred = pd.DataFrame(rows)
    notes = []
    if n_clipped:
        notes.append(f"{n_clipped} prediction(s) fell outside the biological yield range (0 to {YIELD_MAX_KG_HA:,.0f} kg/ha) and were bounded to it; this happens when a model is fitted on very few target seasons.")
    if numeric_groups and len(T) >= 4:
        from scipy.stats import spearmanr
        rho = float(spearmanr(T.group, T.y)[0])
        if abs(rho) >= cfg.trend_flag_rho:
            notes.append(f"The target yield trends strongly with season (Spearman {rho:+.2f}). Leave-one-season-out can interpolate between neighbouring seasons, "
                         "and features that drift with time can look predictive. Compare with the season-trend baseline and with forward validation.")
    if pred.empty:
        return EvaluationResult(pred, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), info, fails, cfg.cv_strategy, base, 0, ntr, list(methods), cfg.to_dict(),
                                ["no fold produced a prediction"])
    mets = []
    wide = pred.pivot_table(index="group", columns="method", values="abs_err")
    kg = float(getattr(bundle, "y_to_kg", 1.0))
    for k, d in pred.groupby("method"):
        n = len(d)
        mae = d.abs_err.mean()
        rmse = float(np.sqrt((d.abs_err ** 2).mean()))
        var = ((d.y_true - d.y_true.mean()) ** 2).sum()
        r2_all = float(1 - ((d.y_true - d.y_pred) ** 2).sum() / var) if var > 0 else np.nan
        r2 = r2_all if n >= cfg.r2_min_test_n else np.nan
        mean_y = float(d.y_true.mean())
        mape = float(100 * (d.abs_err / d.y_true.abs().replace(0, np.nan)).mean())
        win = np.nan
        if "persistence" in wide.columns and k != "persistence":
            both = wide[[k, "persistence"]].dropna()
            win = float(100 * (both[k] < both["persistence"]).mean()) if len(both) else np.nan
        e = d.abs_err.values
        if n >= 3:
            rlo, rhi = _boot_stat(e, lambda v: float(np.sqrt((v ** 2).mean())), cfg.bootstrap_n, cfg.random_state)
            mlo, mhi = _boot_stat(e, lambda v: float(v.mean()), cfg.bootstrap_n, cfg.random_state)
        else:
            rlo = rhi = mlo = mhi = np.nan
        tr_rmse = tr_mae = np.nan
        if info:
            fi = [i for i in info if i["method"] == k]
            if fi:
                tr_rmse = float(np.sqrt(np.mean([i["train_mse_target"] for i in fi])))
                tr_mae = float(np.mean([i["train_mae_target"] for i in fi]))
        mets.append(dict(method=k, n_test=n, MAE=mae, RMSE=rmse, R2=r2, train_RMSE=tr_rmse, train_MAE=tr_mae, train_RMSE_kg_ha=tr_rmse * kg, train_MAE_kg_ha=tr_mae * kg,
                         gap_RMSE_kg_ha=(rmse - tr_rmse) * kg if tr_rmse == tr_rmse else np.nan, R2_note="" if not np.isnan(r2) else
                         f"not reported: fewer than {cfg.r2_min_test_n} held-out points" + (f" (indicative value {r2_all:.2f})" if not np.isnan(r2_all) else ""),
                         R2_indicative=r2_all, rRMSE_pct=100 * rmse / mean_y if mean_y else np.nan, MAPE_pct=mape, win_rate_vs_persistence_pct=win,
                         RMSE_ci_low=rlo, RMSE_ci_high=rhi, MAE_ci_low=mlo, MAE_ci_high=mhi,
                         RMSE_kg_ha=rmse * kg, MAE_kg_ha=mae * kg))
    metrics = pd.DataFrame(mets).sort_values("MAE").reset_index(drop=True)
    imp = []
    for k in wide.columns:
        for b in BASELINES:
            if b in wide.columns and k != b:
                both = wide[[k, b]].dropna()
                d = (both[k] - both[b]).values
                if len(d) == 0:
                    continue
                rel = 100 * d.mean() / both[b].mean() if both[b].mean() else np.nan
                if len(d) >= 5:
                    lo, hi = _boot_ci(d, cfg.bootstrap_n, cfg.random_state)
                    verdict = "improved" if hi < 0 else ("worse" if lo > 0 else "no clear difference")
                else:
                    lo = hi = np.nan
                    verdict = f"not testable (only {len(d)} held-out seasons)"
                imp.append(dict(method=k, baseline=b, n_seasons=len(d), mean_abs_err_change=d.mean(), relative_change_pct=rel, ci95_low=lo, ci95_high=hi,
                                better_in=int((d < 0).sum()), verdict=verdict))
    improvement = pd.DataFrame(imp)
    if len(wide) < cfg.r2_min_test_n:
        notes.append(f"Only {len(wide)} held-out seasons: every metric is unstable, and R2 is not reported below {cfg.r2_min_test_n} points.")
    if cfg.cv_strategy == "loso":
        notes.append("Leave-one-season-out trains on later seasons too; neighbouring seasons can be correlated, so results may be optimistic.")
    return EvaluationResult(pred, metrics, improvement, wide.reset_index(), info, fails, cfg.cv_strategy, base, len(wide), ntr, list(methods),
                            cfg.to_dict(), notes)
