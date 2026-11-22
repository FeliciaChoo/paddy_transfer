"""Yield prediction (tab 6) and future outlook (tab 7): predictions from FULLY TRAINED models, each labelled with the model that made it.

Inputs   a table with a 'season' column and every harmonised feature. Four ways to get it, and the choice is always shown with the result:
  typical   the mean of each feature over the target's observed seasons (a 'normal season' assumption, not a forecast of the weather)
  last      the features of the last observed season repeated (a 'same as last season' assumption)
  custom    values typed in by the user
  upload    a table of future features (for example after a new satellite extraction)
Nothing is invented: there is no weather forecast in this project, so a multi-year outlook from features rests on the input assumption chosen.
Uncertainty   plus or minus the 80th percentile of the absolute errors that model made on HELD-OUT seasons (forward validation, else leave-one-out).
              It is a typical-error band measured with known inputs; with assumed inputs the true uncertainty is larger, and the interface says so.
The season-trend baseline needs only the season; the other models use the features.
"""
import numpy as np
import pandas as pd

from .config import Config
from .errors import WorkflowError
from .evaluation import BASELINES
from .harmonise import HarmonisedBundle
from .methods import LABELS, REGISTRY


def check_inputs(df, features):
    if df is None or len(df) == 0:
        raise WorkflowError("the input table is empty")
    miss = [c for c in ["season"] + list(features) if c not in df.columns]
    if miss:
        raise WorkflowError("the input table is missing columns: " + ", ".join(miss))
    raw = df[["season"] + list(features)].copy()
    out = raw.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    text = [c for c in features if int(out[c].isna().sum()) > int(raw[c].isna().sum())]
    if text:
        raise WorkflowError("non-numeric values in: " + ", ".join(text) + " (they are not converted to missing or filled in)")
    if out["season"].isna().any():
        raise WorkflowError("every row needs a numeric season")
    gaps = [c for c in features if out[c].isna().any()]
    if gaps:
        raise WorkflowError("missing values in: " + ", ".join(gaps) + ". A prediction needs a value for every feature; nothing is filled in silently")
    if np.isinf(out[features].values).any():
        raise WorkflowError("the input table contains infinite values")
    return out.reset_index(drop=True)


def make_inputs(bundle: HarmonisedBundle, scenario, seasons, custom=None):
    feats = bundle.features
    if scenario in ("custom", "upload"):
        return check_inputs(custom, feats)
    T = bundle.target.dropna(subset=["y"]).sort_values("group")
    if scenario == "typical":
        row = T[feats].mean()
    elif scenario == "last":
        row = T.iloc[-1][feats]
    else:
        raise WorkflowError(f"unknown input scenario: {scenario}")
    df = pd.DataFrame([row.values.astype(float)] * len(seasons), columns=feats)
    df.insert(0, "season", [float(s) for s in seasons])
    return df


def band_for(results: dict, key: str):
    """(plus or minus band, where it came from): the 80th percentile of that model's held-out absolute errors"""
    for name in ("forward", "loso"):
        r = results.get(name)
        if r is not None and len(r.predictions):
            e = r.predictions[r.predictions.method == key].abs_err
            if len(e) >= 2:
                return float(np.percentile(e, 80)), ("forward validation" if name == "forward" else "leave-one-season-out")
    return np.nan, "no held-out errors available"


def horizon_errors(bundle: HarmonisedBundle, key, max_h=4, min_train=3):
    """for models that need only the season (season trend, target mean): the error made h seasons AHEAD, measured by refitting on the seasons known at the time.
    {h: (80th percentile absolute error, number of tests)}. Models that need features cannot be tested this way (the future features are not known)."""
    if key not in ("season_trend", "target_mean", "persistence"):
        return {}
    T = bundle.target.dropna(subset=["y"]).sort_values("group")
    g, y = T.group.astype(float).values, T.y.values
    out = {}
    for h in range(1, max_h + 1):
        errs = []
        for end in range(min_train, len(T)):
            if end - 1 + h < len(T):
                k = end
                pred = (np.polyval(np.polyfit(g[:k], y[:k], 1), g[k - 1 + h]) if key == "season_trend" else (y[k - 1] if key == "persistence" else y[:k].mean()))
                errs.append(abs(pred - y[k - 1 + h]))
        if len(errs) >= 2:
            out[h] = (float(np.percentile(errs, 80)), len(errs))
    return out


def role_of(key, choice):
    if key == choice["selected"]:
        return "SELECTED model" + (" (transfer learning)" if key not in BASELINES else " (baseline)")
    if key == choice["reference"]:
        return "best baseline"
    return "baseline" if key in BASELINES else "transfer learning (not selected)"


def fit_final(bundle: HarmonisedBundle, key, base, cfg: Config):
    T = bundle.target.dropna(subset=["y"])
    if key == "season_trend":
        return ("trend", np.polyfit(T.group.astype(float), T.y.values, 1))
    if key == "persistence":
        return ("const", float(T.sort_values("group").y.iloc[-1]))
    feats = bundle.features
    S = bundle.source
    # the final model is for FUTURE seasons, so it may use every observed source year (the look-ahead guard applies only inside validation folds)
    m = REGISTRY[key](base=base, seed=cfg.random_state, cfg=cfg)
    m.ctx = dict(unit_s=S["unit"].astype(str).values if "unit" in S else None, group_s=S.group.values, group_t=T.group.values,
                 w_s=S["w"].values if "w" in S else None, w_t=T["w"].values if "w" in T else None)
    m.fit(S[feats].values, S.y.values, T[feats].values, T.y.values)
    return ("model", m)


def predict_table(bundle, results, choice, inputs, base, cfg, cache, run_methods):
    """one row per (model, season): the prediction, its band, and which model made it"""
    inputs = check_inputs(inputs, bundle.features)
    keys = list(dict.fromkeys([choice["selected"], choice["reference"], "target_only"] + [m for m in run_methods if m not in BASELINES]))
    rows = []
    for k in keys:
        if k not in run_methods and k not in (choice["selected"], choice["reference"]):
            continue
        ck = (choice["token"], k)
        if ck not in cache:
            cache[ck] = fit_final(bundle, k, base, cfg)
        kind, obj = cache[ck]
        if kind == "trend":
            pred = np.polyval(obj, inputs["season"].values)
        elif kind == "const":
            pred = np.full(len(inputs), obj)
        else:
            obj.ctx["group_pred"] = inputs["season"].values if "season" in inputs else None
            pred = np.asarray(obj.predict(inputs[bundle.features].values), float)
        band0, src0 = band_for(results, k)
        hz = horizon_errors(bundle, k)
        last = float(bundle.target.dropna(subset=["y"]).group.max())
        for s, p in zip(inputs["season"].values, pred):
            band, src = band0, src0
            h = int(round(float(s) - last))
            if h >= 1 and hz:                                   # the error actually measured h seasons ahead, when it can be measured
                hh = min(h, max(hz))
                band, src = hz[hh][0], f"measured {hh} season(s) ahead on {hz[hh][1]} tests" + ("" if hh == h else f"; widened by sqrt({h}/{hh}) for {h} ahead (not measured)")
                if hh != h:
                    band = band * float(np.sqrt(h / hh))
                if 1 in hz and h > 1:                           # a measured error can dip by chance at a longer horizon; the band never narrows below sqrt(h) times the one-season error
                    band = max(band, hz[1][0] * float(np.sqrt(h)))
            elif h >= 2 and not np.isnan(band0):                # a feature model: its forward-validation error, widened with the horizon (never a flat band)
                band, src = band0 * float(np.sqrt(h)), f"{src0}; widened by sqrt({h}) for {h} seasons ahead"
            rows.append(dict(model_key=k, model=LABELS.get(k, k), role=role_of(k, choice), season=int(s) if float(s).is_integer() else s, predicted=float(p),
                             band=band, low=float(p - band) if not np.isnan(band) else np.nan, high=float(p + band) if not np.isnan(band) else np.nan, band_source=src))
    return pd.DataFrame(rows)
