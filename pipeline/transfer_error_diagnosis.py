"""
transfer_error_diagnosis.py -- WHY did some transfer methods make the NDVI forecast worse than normal training?

    python pipeline/transfer_error_diagnosis.py --clean eda/DailyGranary_clean.csv --pheno eda/eda_phenology.csv --out data

Same task and split as transfer_methods.py (30 days ahead by default, 1 year of the target's own labels). Each hypothesis is tested by changing ONE thing:

  H1  std_pool is worse because the target's mean/SD are estimated from too few rows.
        test: std_pool_stable  = the target is standardised with its mean/SD from ALL its training-period feature rows (features need no labels),
              the source side unchanged.
  H2  residual is worse because the correction model is too flexible for ~36 target rows.
        test: residual_bias    = only a constant (bias) correction;  residual_strong = the same ridge with a 100x stronger penalty.
  H3  CORAL is worse because the covariance of the target is estimated from too few rows for 20+ features.
        test: coral_eps10 / coral_eps100 = a much stronger regulariser added to both covariances.
A hypothesis is SUPPORTED when the changed version recovers to about pooled_ridge or better, and REJECTED when it stays worse.
"""
import argparse, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ndvi_transfer as nt
import transfer_methods as tm
warnings.filterwarnings("ignore")
FEATS = nt.FEATS


def cell(rows, y0, target, sources, test_years, k, min_year):
    t0 = min(test_years); cut = (t0 - y0) * 36
    test = rows[(rows.District == target) & rows.year.isin(test_years)]
    train = rows[(rows.year >= min_year) & (rows.year < t0) & (rows.t_target < cut)]
    src = train[train.District.isin(sources)]
    tgt_all = train[train.District == target]
    tgt = tgt_all[tgt_all.year >= t0 - k]
    if len(test) < 20 or len(src) < 50 or len(tgt) < 10:
        return None
    return test, src, tgt, tgt_all


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--clean", default="eda/DailyGranary_clean.csv")
    ap.add_argument("--pheno", default="eda/eda_phenology.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--k", type=int, default=1)
    a = ap.parse_args()
    ts = nt.load_series(a.clean, 5); ph = pd.read_csv(a.pheno).set_index("District")
    gr = sorted(set(ts.District) & set(ph.index)); rows, y0 = nt.make_rows(ts, ph, a.horizon)
    out = []
    for tg in gr:
        c = cell(rows, y0, tg, [g for g in gr if g != tg], [2022, 2023, 2024], a.k, 2019)
        if c is None:
            continue
        test, src, tgt, tgt_all = c
        Xte, yte = test[FEATS].values, test["delta"].values
        Xs, ys, Xt, yt = src[FEATS].values, src["delta"].values, tgt[FEATS].values, tgt["delta"].values
        mae = lambda p: float(np.abs(p - yte).mean())
        res = {"pooled_ridge": mae(tm.ridge(np.vstack([Xs, Xt]), np.r_[ys, yt]).predict(Xte))}
        # H1: standardisation statistics
        parts, yy = [], []
        for g, d in src.groupby("District"):
            z, _, _ = tm.zscore_by(d); parts.append(z); yy.append(d["delta"].values)
        for name, stat_df in (("std_pool", tgt), ("std_pool_stable", tgt_all)):
            _, mu, sd = tm.zscore_by(stat_df)
            zt = (Xt - mu) / np.where(sd < 1e-9, 1.0, sd); zte = (Xte - mu) / np.where(sd < 1e-9, 1.0, sd)
            res[name] = mae(Ridge(alpha=1.0).fit(np.vstack(parts + [zt]), np.r_[np.concatenate(yy), yt]).predict(zte))
        # H2: residual correction
        ms = tm.ridge(Xs, ys); r = yt - ms.predict(Xt)
        res["residual"] = mae(ms.predict(Xte) + tm.ridge(Xt, r, alpha=10.0).predict(Xte))
        res["residual_strong"] = mae(ms.predict(Xte) + tm.ridge(Xt, r, alpha=1000.0).predict(Xte))
        res["residual_bias"] = mae(ms.predict(Xte) + r.mean())
        # H3: CORAL regularisation
        sc = tm.StandardScaler().fit(np.vstack([Xs, Xt])); Zs, Zt, Zte = sc.transform(Xs), sc.transform(Xt), sc.transform(Xte)
        for name, eps in (("coral", 1.0), ("coral_eps10", 10.0), ("coral_eps100", 100.0)):
            Za = tm.coral_align(Zs, Zt, eps=eps)
            res[name] = mae(Ridge(alpha=1.0).fit(np.vstack([Za, Zt]), np.r_[ys, yt]).predict(Zte))
        out.append(dict(target=tg, **res))
    r = pd.DataFrame(out).set_index("target")
    pct = (r.div(r["pooled_ridge"], axis=0) - 1) * 100
    summ = pd.DataFrame({"mean_MAE": r.mean().round(4), "change_vs_pooled_%": pct.mean().round(1),
                         "granaries_better_than_pooled": (r.lt(r["pooled_ridge"], axis=0)).sum()})
    os.makedirs(a.out, exist_ok=True)
    summ.to_csv(os.path.join(a.out, "transfer_error_diagnosis.csv"))
    pd.set_option("display.width", 200)
    print(f"{a.horizon * 10} days ahead, {a.k} year of own labels, {len(r)} granaries:"); print(summ.to_string())


if __name__ == "__main__":
    main()
