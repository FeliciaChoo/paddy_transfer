"""Stage 5/6 · The transfer methods that this backend can actually run, and an honest list of those it cannot.

Every method has the same interface: fit(Xs, ys, Xt, yt) on RAW training arrays (source rows, labelled target rows of the TRAINING folds only),
then predict(X). All preprocessing (median imputation, standardisation) is fitted inside fit() on training data only.

  family            method          what it does                                                         label use
  baseline          source_only     train on the source, apply to the target                              source labels
  baseline          target_only     train on the labelled target rows only                                target labels
  baseline          pooled          train on source + target rows with equal weight (naive)               both
  instance-based    tradaboost_r2   boosting that shrinks the weight of source rows that fit target rows poorly   both (needs target labels)
  instance-based    kmm             Kernel Mean Matching: weights source rows so their feature mean matches the
                                    target's in an RBF kernel space; then a weighted model on source + target    NO labels needed for the weights
  feature-based     coral           re-colours source features to the target's mean and covariance, then trains  both
  feature-based     tca             Transfer Component Analysis (linear kernel): a shared low-dimensional space
                                    that minimises the MMD between domains, then trains in it                    both
  parameter-based   ridge_prior     a source Ridge model plus a strongly regularised Ridge correction fitted on
                                    the target rows (the penalty is chosen by leave-one-out on those rows)       both; Ridge only
  hierarchical      granary_intercept_ridge   one Ridge slope shared by all granaries + a partially-pooled intercept per granary (source granaries
                                    and the target), so a yield-LEVEL difference between granaries is absorbed by the intercepts, not the slopes   both
  instance-based    kmm_ridge       Kernel Mean Matching weights + a weighted Ridge (Ridge forced, whatever the base model)                       NO labels for the weights
  instance-based    two_stage_tradaboost_r2   Pardoe & Stone (2010): stage 1 down-weights source rows (step chosen on target error), stage 2 boosts target rows  both
  instance-based    s_tradaboost_r2           importance-sampled, balanced-weight TrAdaBoost.R2 (arXiv:2204.12044)                                  both
  anomaly           z_anomaly_transfer        within-granary Z-scores of features and yield: the model learns anomalies, the target level is added back both
NOT implemented (so not offered): fine-tuning a neural network, adversarial domain adaptation (DANN), and fine-tuning of Random Forest or
XGBoost models: trees have no weights to initialise from and continue training, so "fine-tuning" them is not the same operation.
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import RobustScaler

from ._boost2 import STrAdaBoostR2, TwoStageTrAdaBoostR2
from ._tradaboost import TrAdaBoostR2
from .assessment import _rbf


class MethodUnavailable(Exception):
    pass


def make_base(base, seed=0):
    if base == "ridge":
        return Ridge(alpha=1.0)
    if base == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=seed)
    if base == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as e:
            raise MethodUnavailable("xgboost is not installed in this environment") from e
        return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed, verbosity=0)
    raise ValueError(f"unknown base model: {base}")


def available_bases():
    out = ["ridge", "random_forest"]
    try:
        import xgboost  # noqa: F401
        out.append("xgboost")
    except ImportError:
        pass
    return out


def _vif_select(Z, y, max_features, vif_max, corr_max):
    """C3: drop collinear features, then the least label-related ones, until at most max_features remain. Fitted on the rows given (training rows only).
    Returns the kept column indices. VIF = diagonal of the inverse correlation matrix (a small ridge keeps it finite when rows <= columns)."""
    keep = list(range(Z.shape[1]))
    ymask = None if y is None else ~np.isnan(y)

    def rel(j):
        if y is None or ymask.sum() < 3 or np.nanstd(Z[ymask, j]) == 0:
            return 0.0
        return abs(float(np.corrcoef(Z[ymask, j], y[ymask])[0, 1]))
    while len(keep) > 1:
        R = np.corrcoef(Z[:, keep], rowvar=False)
        R = np.nan_to_num(R, nan=0.0)
        np.fill_diagonal(R, 1.0)
        vif = np.diag(np.linalg.inv(R + 1e-6 * np.eye(len(keep))))
        worst = int(np.argmax(vif))
        offd = np.abs(R - np.eye(len(keep)))
        if vif[worst] > vif_max or offd.max() > corr_max:
            cand = [i for i in range(len(keep)) if vif[i] >= vif[worst] * 0.999] if vif[worst] > vif_max else [int(np.unravel_index(np.argmax(offd), offd.shape)[0]), int(np.unravel_index(np.argmax(offd), offd.shape)[1])]
            drop = min(cand, key=lambda i: rel(keep[i]))          # of the collinear candidates, drop the one least related to the label
            keep.pop(drop)
        elif len(keep) > max_features:
            keep.pop(min(range(len(keep)), key=lambda i: rel(keep[i])))
        else:
            break
    return keep


class Prep:
    """median imputation, then robust (median/IQR) or standard scaling, then in-fold VIF/correlation feature selection (p <= max_features).
    Everything is fitted on the rows given to fit() and nothing else. With cfg=None it is the plain imputer + standardiser."""
    def __init__(self, cfg=None):
        self.cfg = cfg

    def fit(self, X, y=None):
        c = self.cfg
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True)
        Xi = self.imp.fit_transform(X)
        self.sc = (RobustScaler() if (c is not None and c.scaler == "robust") else StandardScaler()).fit(Xi)
        Z = self.sc.transform(Xi)
        self.keep = list(range(Z.shape[1]))
        if c is not None and c.select_features and Z.shape[1] > 1 and len(Z) >= 3:
            self.keep = _vif_select(Z, None if y is None else np.asarray(y, float), c.max_features, c.vif_max, c.corr_max)
        return self

    def transform(self, X):
        return self.sc.transform(self.imp.transform(X))[:, self.keep]


def _labelled(X, y):
    m = ~np.isnan(y)
    return X[m], y[m]


def _fit_model(base, seed, Z, y, w=None):
    m = make_base(base, seed)
    if w is None:
        return m.fit(Z, y)
    return m.fit(Z, y, sample_weight=w)


def kmm_weights(Zs, Zt, B=10.0):
    """Kernel Mean Matching (Huang et al. 2006): minimise ||mean_s(beta phi(x)) - mean_t(phi(x))||^2 subject to 0 <= beta <= B and the weights
    averaging about one. RBF kernel, median-heuristic bandwidth. Uses features only: no labels."""
    ns, nt = len(Zs), len(Zt)
    pooled = np.vstack([Zs, Zt])
    d = np.sqrt(((pooled[:, None, :] - pooled[None, :, :]) ** 2).sum(-1))
    sigma = float(np.median(d[np.triu_indices_from(d, 1)])) or 1.0
    K = _rbf(Zs, Zs, sigma) + 1e-6 * np.eye(ns)
    kappa = (ns / nt) * _rbf(Zs, Zt, sigma).sum(1)
    eps = (np.sqrt(ns) - 1) / np.sqrt(ns)
    res = minimize(lambda b: 0.5 * b @ K @ b - kappa @ b, np.ones(ns), jac=lambda b: K @ b - kappa, bounds=[(0, B)] * ns, method="SLSQP",
                   constraints=[{"type": "ineq", "fun": lambda b: ns * (1 + eps) - b.sum()}, {"type": "ineq", "fun": lambda b: b.sum() - ns * (1 - eps)}],
                   options={"maxiter": 300})
    return np.clip(res.x, 0, B), dict(converged=bool(res.success), bandwidth=sigma, message=str(res.message))


def _mpow(C, p):
    vals, vecs = np.linalg.eigh(C)
    return (vecs * np.clip(vals, 1e-6, None) ** p) @ vecs.T


def coral_align(Zs, Zt, eps=1.0):
    """CORAL (Sun et al. 2016): whiten the source with its covariance, re-colour with the target's, and shift to the target mean"""
    Cs = np.cov(Zs, rowvar=False) + eps * np.eye(Zs.shape[1])
    Ct = np.cov(Zt, rowvar=False) + eps * np.eye(Zs.shape[1])
    return (Zs - Zs.mean(0)) @ _mpow(Cs, -0.5) @ _mpow(Ct, 0.5) + Zt.mean(0)


class _TCA:
    """linear-kernel Transfer Component Analysis (Pan et al. 2011): the directions that keep variance while minimising the MMD between domains"""
    def __init__(self, dim=4, mu=1.0):
        self.dim, self.mu = dim, mu

    def fit(self, Zs, Zt):
        X = np.vstack([Zs, Zt]); n, ns, nt = len(X), len(Zs), len(Zt)
        K = X @ X.T
        e = np.r_[np.ones(ns) / ns, -np.ones(nt) / nt]
        L = np.outer(e, e)
        H = np.eye(n) - np.ones((n, n)) / n
        A = np.linalg.solve(K @ L @ K + self.mu * np.eye(n), K @ H @ K)
        w, V = np.linalg.eig(A)
        idx = np.argsort(-w.real)[: max(1, min(self.dim, n - 1))]
        self.X, self.W = X, V[:, idx].real
        return self

    def transform(self, Z):
        return (Z @ self.X.T) @ self.W


class _Base:
    key, family = "", ""

    def __init__(self, base="ridge", seed=0, cfg=None):
        self.base, self.seed, self.cfg, self.info = base, seed, cfg, {}
        self.ctx = {}                       # set by evaluation / predict: unit_s (granary of each source row), group_s, group_t. Methods that do not need it ignore it

    def prep(self, X, y=None):
        return Prep(self.cfg).fit(X, y)

    def source_units(self, mask):
        u = self.ctx.get("unit_s")
        return np.asarray(u)[mask] if u is not None and len(u) == len(mask) else np.array(["source"] * int(mask.sum()))


class SourceOnly(_Base):
    key, family = "source_only", "baseline"

    def fit(self, Xs, ys, Xt, yt):
        Xl, yl = _labelled(Xs, ys)
        self.p = self.prep(Xl, yl)
        self.m = _fit_model(self.base, self.seed, self.p.transform(Xl), yl)
        return self

    def predict(self, X):
        return self.m.predict(self.p.transform(X))


class TargetOnly(_Base):
    key, family = "target_only", "baseline"

    def fit(self, Xs, ys, Xt, yt):
        Xl, yl = _labelled(Xt, yt)
        self.const = float(np.mean(yl)) if len(yl) < 2 else None
        if self.const is None:
            self.p = self.prep(Xl, yl)
            self.m = _fit_model(self.base, self.seed, self.p.transform(Xl), yl)
        return self

    def predict(self, X):
        return np.full(len(X), self.const) if self.const is not None else self.m.predict(self.p.transform(X))


class TargetMean(_Base):
    """the no-transfer, no-feature alternative baseline: always predict the mean of the labelled target training rows.
    If a model cannot beat this, its features add nothing for the target."""
    key, family = "target_mean", "baseline"

    def fit(self, Xs, ys, Xt, yt):
        self.mu = float(np.nanmean(yt))
        return self

    def predict(self, X):
        return np.full(len(X), self.mu)


class Pooled(_Base):
    key, family = "pooled", "baseline"

    def fit(self, Xs, ys, Xt, yt):
        Xa, ya = _labelled(np.vstack([Xs, Xt]), np.r_[ys, yt])
        self.p = self.prep(Xa, ya)
        self.m = _fit_model(self.base, self.seed, self.p.transform(Xa), ya)
        return self

    def predict(self, X):
        return self.m.predict(self.p.transform(X))


class TrAdaBoost(_Base):
    key, family = "tradaboost_r2", "instance-based"

    def fit(self, Xs, ys, Xt, yt):
        Xsl, ysl = _labelled(Xs, ys)
        Xtl, ytl = _labelled(Xt, yt)
        self.p = self.prep(np.vstack([Xsl, Xtl]), np.r_[ysl, ytl])
        weak = Ridge(alpha=1.0) if self.base == "ridge" else DecisionTreeRegressor(max_depth=3, random_state=self.seed)
        self.m = TrAdaBoostR2(base_estimator=weak, n_estimators=30, random_state=self.seed).fit(self.p.transform(Xsl), ysl, self.p.transform(Xtl), ytl)
        w = self.m.final_weights_[: self.m.n_source_]
        self.info = dict(source_weight_share=float(w.sum() / self.m.final_weights_.sum()), ess_ratio=float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
                         n_source=len(w), n_target=len(Xtl))
        return self

    def predict(self, X):
        return self.m.predict(self.p.transform(X))


class KMM(_Base):
    key, family = "kmm", "instance-based"

    def fit(self, Xs, ys, Xt, yt):
        Xsl, ysl = _labelled(Xs, ys)
        Xtl, ytl = _labelled(Xt, yt)
        self.p = self.prep(np.vstack([Xsl, Xt]), np.r_[ysl, np.full(len(Xt), np.nan)])
        Zs, Zt = self.p.transform(Xsl), self.p.transform(Xt)
        beta, kinfo = kmm_weights(Zs, Zt)
        self.m = _fit_model(self.base, self.seed, np.vstack([Zs, self.p.transform(Xtl)]), np.r_[ysl, ytl], np.r_[beta + 1e-8, np.ones(len(ytl))])
        self.info = dict(ess_ratio=float(beta.sum() ** 2 / (beta ** 2).sum() / len(beta)), beta_min=float(beta.min()), beta_max=float(beta.max()),
                         n_source=len(beta), **kinfo)
        return self

    def predict(self, X):
        return self.m.predict(self.p.transform(X))


class CORAL(_Base):
    key, family = "coral", "feature-based"

    def fit(self, Xs, ys, Xt, yt):
        Xsl, ysl = _labelled(Xs, ys)
        Xtl, ytl = _labelled(Xt, yt)
        self.p = self.prep(np.vstack([Xsl, Xtl]), np.r_[ysl, ytl])
        Zs, Zt = self.p.transform(Xsl), self.p.transform(Xtl)
        eps = self.cfg.coral_eps if self.cfg else 1.0
        Za = coral_align(Zs, Zt, eps)
        self.m = _fit_model(self.base, self.seed, np.vstack([Za, Zt]), np.r_[ysl, ytl])
        self.info = dict(eps=eps, n_target_rows_for_covariance=len(Zt), n_features=Zs.shape[1])
        return self

    def predict(self, X):
        return self.m.predict(self.p.transform(X))


class TCA(_Base):
    key, family = "tca", "feature-based"

    def fit(self, Xs, ys, Xt, yt):
        Xsl, ysl = _labelled(Xs, ys)
        Xtl, ytl = _labelled(Xt, yt)
        self.p = self.prep(np.vstack([Xsl, Xtl]), np.r_[ysl, ytl])
        Zs, Zt = self.p.transform(Xsl), self.p.transform(Xtl)
        self.t = _TCA(dim=min(4, Zs.shape[1])).fit(Zs, Zt)
        self.m = _fit_model(self.base, self.seed, np.vstack([self.t.transform(Zs), self.t.transform(Zt)]), np.r_[ysl, ytl])
        self.info = dict(components=int(self.t.W.shape[1]), n_rows=len(Zs) + len(Zt))
        return self

    def predict(self, X):
        return self.m.predict(self.t.transform(self.p.transform(X)))


class RidgePrior(_Base):
    key, family = "ridge_prior", "parameter-based"
    GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)

    def fit(self, Xs, ys, Xt, yt):
        if self.base != "ridge":
            raise MethodUnavailable("ridge_prior needs a Ridge base model: coefficients cannot be carried over from a forest or boosted trees")
        Xsl, ysl = _labelled(Xs, ys)
        Xtl, ytl = _labelled(Xt, yt)
        self.p = self.prep(Xsl, ysl)
        Zs, Zt = self.p.transform(Xsl), self.p.transform(Xtl)
        self.src = Ridge(alpha=1.0).fit(Zs, ysl)
        r = ytl - self.src.predict(Zt)
        lam = 100.0
        if len(r) >= 3:
            errs = []
            for a in self.GRID:
                e = 0.0
                for i in range(len(r)):
                    keep = np.arange(len(r)) != i
                    e += abs(Ridge(alpha=a).fit(Zt[keep], r[keep]).predict(Zt[i:i + 1])[0] - r[i])
                errs.append(e)
            lam = self.GRID[int(np.argmin(errs))]
        self.res = Ridge(alpha=lam).fit(Zt, r)
        self.info = dict(penalty=lam, tuned_by_loo=len(r) >= 3, n_target_rows=len(r))
        return self

    def predict(self, X):
        Z = self.p.transform(X)
        return self.src.predict(Z) + self.res.predict(Z)


def _weak(base, seed):
    return Ridge(alpha=1.0) if base == "ridge" else DecisionTreeRegressor(max_depth=3, random_state=seed)


class GranaryInterceptRidge(_Base):
    """Hierarchical granary-intercept Ridge: y = a_granary + Z.b. One slope vector b for every granary, one intercept per granary, partially pooled
    towards the common level: a_g = m + n_g/(n_g + k) (mean residual_g - m), k = 1. The target granary's intercept comes from its TRAINING rows only.
    This is how the yield-LEVEL difference between granaries (label shift) is resolved without letting it distort the slopes."""
    key, family = "granary_intercept_ridge", "hierarchical"
    K = 1.0

    def fit(self, Xs, ys, Xt, yt):
        ms = ~np.isnan(ys)
        Xsl, ysl, us = Xs[ms], ys[ms], self.source_units(ms)
        Xtl, ytl = _labelled(Xt, yt)
        self.p = self.prep(np.vstack([Xsl, Xtl]), np.r_[ysl, ytl])
        Z = np.vstack([self.p.transform(Xsl), self.p.transform(Xtl)])
        y = np.r_[ysl, ytl]
        u = np.r_[us.astype(object), np.array(["__target__"] * len(ytl), dtype=object)]
        names = list(dict.fromkeys(u.tolist()))
        a = {g: float(y.mean()) for g in names}
        coef = np.zeros(Z.shape[1])
        for _ in range(15):                                              # back-fitting: slopes given intercepts, intercepts given slopes
            r = y - np.array([a[g] for g in u])
            coef = Ridge(alpha=1.0, fit_intercept=False).fit(Z, r).coef_
            res = y - Z @ coef
            m = float(res.mean())
            for g in names:
                k = u == g
                a[g] = m + k.sum() / (k.sum() + self.K) * (float(res[k].mean()) - m)
        self.coef, self.a_t = coef, a.get("__target__", float(y.mean()))
        self.info = dict(n_granaries=len(names) - ("__target__" in names), n_target_rows=len(ytl), target_intercept=self.a_t,
                         intercept_spread=float(np.std([v for g, v in a.items()])), n_features_used=int(Z.shape[1]))
        return self

    def predict(self, X):
        return self.a_t + self.p.transform(X) @ self.coef


class KMMRidge(KMM):
    """Kernel Mean Matching weights with a weighted RIDGE model, whatever base model is selected (the 'KMM + Weighted Ridge' of the methodology)"""
    key, family = "kmm_ridge", "instance-based"

    def __init__(self, base="ridge", seed=0, cfg=None):
        super().__init__("ridge", seed, cfg)


class TwoStageTrAdaBoost(_Base):
    key, family = "two_stage_tradaboost_r2", "instance-based"

    def fit(self, Xs, ys, Xt, yt):
        Xsl, ysl = _labelled(Xs, ys)
        Xtl, ytl = _labelled(Xt, yt)
        self.p = self.prep(np.vstack([Xsl, Xtl]), np.r_[ysl, ytl])
        n = self.cfg.boost_rounds if self.cfg else 30
        self.m = TwoStageTrAdaBoostR2(_weak(self.base, self.seed), n_estimators=n, random_state=self.seed).fit(
            self.p.transform(Xsl), ysl, self.p.transform(Xtl), ytl)
        w = self.m.source_weights_
        self.info = dict(stage1_step=self.m.stage1_step_, source_weight_ess_ratio=float(w.sum() ** 2 / (w ** 2).sum() / len(w)), n_source=len(w), n_target=len(ytl))
        return self

    def predict(self, X):
        return self.m.predict(self.p.transform(X))


class STrAdaBoost(_Base):
    key, family = "s_tradaboost_r2", "instance-based"

    def fit(self, Xs, ys, Xt, yt):
        Xsl, ysl = _labelled(Xs, ys)
        Xtl, ytl = _labelled(Xt, yt)
        self.p = self.prep(np.vstack([Xsl, Xtl]), np.r_[ysl, ytl])
        n = self.cfg.boost_rounds if self.cfg else 30
        self.m = STrAdaBoostR2(_weak(self.base, self.seed), n_estimators=n, random_state=self.seed).fit(self.p.transform(Xsl), ysl, self.p.transform(Xtl), ytl)
        w = self.m.final_weights_[: self.m.n_source_]
        self.info = dict(source_weight_share=float(w.sum() / self.m.final_weights_.sum()), ess_ratio=float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
                         n_source=len(w), n_target=len(ytl), source_rows_ever_sampled=int((self.m.sample_counts_ > 0).sum()))
        return self

    def predict(self, X):
        return self.m.predict(self.p.transform(X))


class ZAnomalyTransfer(_Base):
    """Z-score anomaly transfer: features and yield are Z-scored WITHIN each granary (the target's statistics come from its training rows only), a Ridge
    model learns anomaly -> anomaly across all granaries, and the target's own yield level and spread are added back. Level and spread are therefore the
    target's, and only the response of yield to a feature anomaly is transferred. The target spread is shrunk towards the source spread (k = 2) because it
    is estimated from a handful of years."""
    key, family = "z_anomaly_transfer", "anomaly"
    K = 2.0

    @staticmethod
    def _z(Z, mu, sd):
        return (Z - mu) / np.where(sd > 1e-9, sd, 1.0)

    def fit(self, Xs, ys, Xt, yt):
        ms = ~np.isnan(ys)
        Xsl, ysl, us = Xs[ms], ys[ms], self.source_units(ms)
        Xtl, ytl = _labelled(Xt, yt)
        self.p = self.prep(np.vstack([Xsl, Xtl]), np.r_[ysl, ytl])
        Zs, Zt = self.p.transform(Xsl), self.p.transform(Xtl)
        A, B, sds = [], [], []
        for g in dict.fromkeys(us.tolist()):
            k = us == g
            if k.sum() < 3:
                continue
            mu, sd = Zs[k].mean(0), Zs[k].std(0, ddof=1)
            ysd = ysl[k].std(ddof=1)
            if ysd <= 1e-9:
                continue
            A.append(self._z(Zs[k], mu, sd))
            B.append((ysl[k] - ysl[k].mean()) / ysd)
            sds.append(ysd)
        self.t_mu_x, self.t_sd_x = Zt.mean(0), (Zt.std(0, ddof=1) if len(Zt) > 1 else np.ones(Zt.shape[1]))
        self.t_mu_y = float(ytl.mean())
        pool_sd = float(np.mean(sds)) if sds else float(np.std(ysl, ddof=1))
        n = len(ytl)
        t_sd = float(ytl.std(ddof=1)) if n > 1 else pool_sd
        self.t_sd_y = float(np.sqrt(((n - 1) * t_sd ** 2 + self.K * pool_sd ** 2) / (n - 1 + self.K)))
        if len(Zt) > 1:
            A.append(self._z(Zt, self.t_mu_x, self.t_sd_x))
            B.append((ytl - self.t_mu_y) / max(t_sd, 1e-9))
        self.m = Ridge(alpha=1.0).fit(np.vstack(A), np.concatenate(B))
        self.info = dict(n_granaries=len(sds), target_level=self.t_mu_y, target_spread=self.t_sd_y, source_spread=pool_sd)
        return self

    def predict(self, X):
        z = self._z(self.p.transform(X), self.t_mu_x, self.t_sd_x)
        return self.t_mu_y + self.t_sd_y * self.m.predict(z)


def ridge_trend(years, y, lam=1.0):
    """regularised local linear time trend: slope = sum(t (y - ybar)) / (sum(t^2) + lam) on centred years. Fewer than 3 years: a flat line at the mean.
    Returns (intercept at the mean year, slope per year, mean year)."""
    years, y = np.asarray(years, float), np.asarray(y, float)
    if len(y) < 3 or np.ptp(years) == 0:
        return float(np.mean(y)), 0.0, float(np.mean(years))
    t = years - years.mean()
    b = float(np.sum(t * (y - y.mean())) / (np.sum(t ** 2) + lam))
    return float(y.mean()), b, float(years.mean())


class DetrendedResidualTransfer(_Base):
    """Detrended Residual Transfer (Hybrid Season Trend + S-TrAdaBoost.R2).
      1. trend     a regularised local linear time trend per granary, fitted on that granary's TRAINING years only (the source granaries and the target)
      2. residual  r = y - trend, across the source and the target training years; S-TrAdaBoost.R2, weighted by the optical-quality weights
                   w = max(0.2, Optical_Valid_Fraction), learns r from the stage satellite/climate features
      3. rebuild   y_hat = trend_target(season) + r_hat
    The trend carries the secular drift (a downward drift in this target); the transferred model carries only the inter-annual stress anomaly. Needs the season of
    every row to predict: ctx['group_s'] / ctx['group_t'] when fitting and ctx['group_pred'] when predicting."""
    key, family = "detrended_residual_transfer", "hybrid (trend + instance-based)"

    def fit(self, Xs, ys, Xt, yt):
        ms = ~np.isnan(ys)
        Xsl, ysl, us = Xs[ms], ys[ms], self.source_units(ms)
        gs = np.asarray(self.ctx.get("group_s"))[ms] if self.ctx.get("group_s") is not None else None
        gt = self.ctx.get("group_t")
        Xtl, ytl = _labelled(Xt, yt)
        if gs is None or gt is None or len(gt) != len(Xt):
            raise MethodUnavailable("needs the season of every source and target row")
        gt = np.asarray(gt)[~np.isnan(yt)]
        rs = np.empty(len(ysl))
        for u in dict.fromkeys(us.tolist()):                              # each source granary is detrended with its own trend
            k = us == u
            a, b, t0 = ridge_trend(gs[k], ysl[k])
            rs[k] = ysl[k] - (a + b * (gs[k].astype(float) - t0))
        self.ta, self.tb, self.tt0 = ridge_trend(gt, ytl)
        rt = ytl - (self.ta + self.tb * (np.asarray(gt, float) - self.tt0))
        self.p = self.prep(np.vstack([Xsl, Xtl]), np.r_[rs, rt])
        ws, wt = self.ctx.get("w_s"), self.ctx.get("w_t")
        ws = np.asarray(ws)[ms] if ws is not None and len(ws) == len(ms) else None
        wt = np.asarray(wt)[~np.isnan(yt)] if wt is not None and len(wt) == len(yt) else None
        n = self.cfg.boost_rounds if self.cfg else 30
        self.m = STrAdaBoostR2(_weak(self.base, self.seed), n_estimators=n, random_state=self.seed).fit(self.p.transform(Xsl), rs, self.p.transform(Xtl), rt, sw_s=ws, sw_t=wt)
        self.last_season = float(np.max(gt))
        self.info = dict(trend_slope_kg_ha_per_year=1000.0 * self.tb, n_source=len(rs), n_target=len(rt), residual_sd_source=float(np.std(rs)), residual_sd_target=float(np.std(rt)),
                         optical_weights_used=bool(ws is not None))
        return self

    def trend_at(self, seasons):
        return self.ta + self.tb * (np.asarray(seasons, float) - self.tt0)

    def predict(self, X, seasons=None):
        seasons = self.ctx.get("group_pred") if seasons is None else seasons
        if seasons is None:
            seasons = np.full(len(X), self.last_season + 1)
        return self.trend_at(seasons) + self.m.predict(self.p.transform(X))


REGISTRY = {c.key: c for c in (SourceOnly, TargetOnly, TargetMean, Pooled, TrAdaBoost, KMM, CORAL, TCA, RidgePrior,
                                GranaryInterceptRidge, KMMRidge, TwoStageTrAdaBoost, STrAdaBoost, ZAnomalyTransfer, DetrendedResidualTransfer)}
LABELS = {"source_only": "B2 Source-Only (pretrain and test, no adaptation)", "target_only": "B1 Target-Only (normal training, no transfer)", "target_mean": "B4 Target mean floor (no features)", "season_trend": "Season trend (no features)", "pooled": "B3 Naive pooled training", "tradaboost_r2": "Standard TrAdaBoost.R2 (diagnostic reference)",
          "kmm": "Kernel Mean Matching + weighted model", "coral": "CORAL feature alignment", "tca": "Transfer Component Analysis",
          "ridge_prior": "Parameter transfer (source Ridge + regularised target correction)", "persistence": "B4 Persistence floor (last observed yield)",
          "granary_intercept_ridge": "Granary-Intercept Ridge (hierarchical)", "kmm_ridge": "KMM + Weighted Ridge",
          "two_stage_tradaboost_r2": "Two-Stage TrAdaBoost.R2", "s_tradaboost_r2": "S-TrAdaBoost.R2", "z_anomaly_transfer": "Z-Score Anomaly Transfer",
          "detrended_residual_transfer": "Detrended Residual Transfer (Hybrid Trend + S-TrAdaBoost.R2)"}
# The methodology file names the methods Granary_Intercept_Ridge, KMM_Ridge, Two_Stage_TrAdaBoost_R2, S_TrAdaBoost_R2 (and Z-Score Anomaly Transfer).
# Those names resolve to the registry keys, and each is also registered in LABELS so a lookup by either name gives the display label.
ALIASES = {"Granary_Intercept_Ridge": "granary_intercept_ridge", "KMM_Ridge": "kmm_ridge", "Two_Stage_TrAdaBoost_R2": "two_stage_tradaboost_r2",
           "S_TrAdaBoost_R2": "s_tradaboost_r2", "Detrended_Residual_Transfer": "detrended_residual_transfer", "Z_Score_Anomaly_Transfer": "z_anomaly_transfer", "Persistence": "persistence"}
for _alias, _key in ALIASES.items():
    LABELS[_alias] = LABELS[_key]


def canonical(key):
    """the registry key for a method name written either way (kmm_ridge or KMM_Ridge)"""
    return ALIASES.get(key, key)
NOT_IMPLEMENTED = {
    "fine_tune_nn": "Fine-tuning a neural network: no neural model exists in this system.",
    "adversarial_da": "Adversarial domain adaptation (DANN): not implemented; it needs a neural feature extractor and far more rows.",
    "tree_fine_tune": "Fine-tuning Random Forest or XGBoost: trees have no weights to initialise from, so this is not the same operation.",
}


def eligibility(key, n_s, n_t, n_feat, base, cfg):
    """(eligible, reason) given the data actually available for training"""
    key = canonical(key)
    if key not in REGISTRY:
        return False, NOT_IMPLEMENTED.get(key, "not implemented")
    if base == "xgboost" and "xgboost" not in available_bases():
        return False, "xgboost is not installed"
    if n_s < 2:
        return False, "fewer than 2 labelled source rows"
    if key == "source_only":
        return True, "needs only source labels"
    if key == "target_only":
        return (n_t >= 2, "needs at least 2 labelled target rows" if n_t < 2 else f"{n_t} labelled target rows (a weak baseline when this is small)")
    if key == "pooled":
        return (n_t >= 1, "needs at least 1 labelled target row")
    if key == "target_mean":
        return (n_t >= 1, "uses no features: the check on whether the features help at all")
    if key == "tradaboost_r2":
        ok = n_t >= cfg.min_target_tradaboost
        return ok, (f"{n_t} labelled target rows" if ok else f"needs at least {cfg.min_target_tradaboost} labelled target rows to judge the source rows; has {n_t}")
    if key == "kmm":
        ok = n_s <= cfg.kmm_max_source and n_t >= 2
        return ok, ("uses features only for the weights" if ok else f"needs at least 2 target rows and at most {cfg.kmm_max_source} source rows")
    if key == "coral":
        return (n_t >= 2, f"covariance from {n_t} target rows for {n_feat} features: regularised, interpret with care" if n_t >= 2 else "needs at least 2 target rows")
    if key == "tca":
        ok = n_t >= 2 and n_s + n_t <= cfg.tca_max_rows
        return ok, ("linear kernel, 4 components" if ok else f"needs at least 2 target rows and at most {cfg.tca_max_rows} rows in total")
    if key == "ridge_prior":
        if base != "ridge":
            return False, "needs a Ridge base model (coefficients cannot be transferred to trees)"
        return (n_t >= 2, "penalty tuned by leave-one-out" if n_t >= 3 else "needs at least 2 target rows (a fixed penalty is used below 3)")
    if key == "granary_intercept_ridge":
        return (n_t >= 2, "partial pooling of one intercept per granary; the target intercept comes from its training rows" if n_t >= 2 else "needs at least 2 target rows")
    if key == "kmm_ridge":
        ok = n_s <= cfg.kmm_max_source and n_t >= 2
        return ok, ("kernel-mean-matching weights (features only) with a weighted Ridge" if ok else f"needs at least 2 target rows and at most {cfg.kmm_max_source} source rows")
    if key in ("two_stage_tradaboost_r2", "s_tradaboost_r2"):
        ok = n_t >= cfg.min_target_tradaboost
        return ok, (f"{n_t} labelled target rows" if ok else f"needs at least {cfg.min_target_tradaboost} labelled target rows; has {n_t}")
    if key == "detrended_residual_transfer":
        ok = n_t >= max(3, cfg.min_target_tradaboost)
        return ok, ("trend per granary + residual transfer; needs the season of every row" if ok else f"needs at least {max(3, cfg.min_target_tradaboost)} labelled target years for a trend and a residual model; has {n_t}")
    if key == "z_anomaly_transfer":
        return (n_t >= 3, "needs at least 3 target rows to estimate the target's own mean and spread" if n_t < 3 else f"target level and spread from {n_t} rows (spread shrunk to the source)")
    return False, "unknown"
