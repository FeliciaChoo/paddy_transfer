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


class Prep:
    """median imputation then standardisation, fitted on the rows given to fit() and nothing else"""
    def fit(self, X):
        self.imp = SimpleImputer(strategy="median", keep_empty_features=True)
        self.sc = StandardScaler().fit(self.imp.fit_transform(X))
        return self

    def transform(self, X):
        return self.sc.transform(self.imp.transform(X))


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


class SourceOnly(_Base):
    key, family = "source_only", "baseline"

    def fit(self, Xs, ys, Xt, yt):
        Xl, yl = _labelled(Xs, ys)
        self.p = Prep().fit(Xl)
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
            self.p = Prep().fit(Xl)
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
        self.p = Prep().fit(Xa)
        self.m = _fit_model(self.base, self.seed, self.p.transform(Xa), ya)
        return self

    def predict(self, X):
        return self.m.predict(self.p.transform(X))


class TrAdaBoost(_Base):
    key, family = "tradaboost_r2", "instance-based"

    def fit(self, Xs, ys, Xt, yt):
        Xsl, ysl = _labelled(Xs, ys)
        Xtl, ytl = _labelled(Xt, yt)
        self.p = Prep().fit(np.vstack([Xsl, Xtl]))
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
        self.p = Prep().fit(np.vstack([Xsl, Xt]))
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
        self.p = Prep().fit(np.vstack([Xsl, Xtl]))
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
        self.p = Prep().fit(np.vstack([Xsl, Xtl]))
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
        self.p = Prep().fit(Xsl)
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


REGISTRY = {c.key: c for c in (SourceOnly, TargetOnly, TargetMean, Pooled, TrAdaBoost, KMM, CORAL, TCA, RidgePrior)}
LABELS = {"source_only": "Source-only model", "target_only": "Target-only model", "target_mean": "Target mean (no features)", "season_trend": "Season trend (no features)", "pooled": "Pooled (naive)", "tradaboost_r2": "TrAdaBoost.R2",
          "kmm": "Kernel Mean Matching + weighted model", "coral": "CORAL feature alignment", "tca": "Transfer Component Analysis",
          "ridge_prior": "Parameter transfer (source Ridge + regularised target correction)"}
NOT_IMPLEMENTED = {
    "fine_tune_nn": "Fine-tuning a neural network: no neural model exists in this system.",
    "adversarial_da": "Adversarial domain adaptation (DANN): not implemented; it needs a neural feature extractor and far more rows.",
    "tree_fine_tune": "Fine-tuning Random Forest or XGBoost: trees have no weights to initialise from, so this is not the same operation.",
}


def eligibility(key, n_s, n_t, n_feat, base, cfg):
    """(eligible, reason) given the data actually available for training"""
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
    return False, "unknown"
