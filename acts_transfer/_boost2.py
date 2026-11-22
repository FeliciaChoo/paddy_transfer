"""Two-Stage TrAdaBoost.R2 (Pardoe & Stone 2010) and S-TrAdaBoost.R2 (Gupta et al. 2022/2023, arXiv:2204.12044), written for small target samples.

Both take (X_source, y_source, X_target, y_target) of already-prepared (scaled) arrays and expose predict(X) and the final source weights.

Two-Stage TrAdaBoost.R2
  Stage 1  boost with the TARGET weights held fixed: only source-row weights shrink (beta ** loss) when the weak learner fits them badly.
           The step that gives the lowest leave-one-out error on the labelled TARGET rows is kept (the paper picks it by cross-validation).
  Stage 2  freeze the source weights of that step and run AdaBoost.R2 updates on the TARGET rows only.
  Prediction: weighted median of the stage-2 learners.
  Why it suits nt = 7: the source can never out-vote the target through the target weight update, and the number of source-weight steps is chosen
  on target error, so a source that does not help is switched off (step 0 = uniform weights is allowed to win).

S-TrAdaBoost.R2 (importance-sampled, balanced weights)
  Round by round: (1) draw an importance sample of the source rows with probability proportional to their current weight, of at most
  `ratio` x the number of target rows, so a large source cannot skew the fit; (2) fit the weak learner on that sample plus all target rows;
  (3) update BOTH blocks: source rows that the learner fits badly are decayed by b**loss, target rows get the AdaBoost.R2 update (no cap on how far a source weight may fall), then (4) renormalise each block to
  its own mass, so source and target keep a balanced total weight. Prediction: weighted median of the second half of the rounds.
  This is an implementation of the published idea (importance sampling + balanced, unconstrained weight update) for this project's small tables.
  It is not a line-by-line port of the authors' code.
"""
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge


def _wmedian(preds, wts):
    """weighted median across rounds (rows of preds), per sample"""
    wts = np.asarray(wts, float)
    if wts.sum() <= 0:
        return preds.mean(axis=0)
    order = np.argsort(preds, axis=0)
    sp = np.take_along_axis(preds, order, axis=0)
    sw = np.take_along_axis(np.tile(wts.reshape(-1, 1), (1, preds.shape[1])), order, axis=0)
    cum = np.cumsum(sw, axis=0)
    idx = (cum >= cum[-1] / 2.0).argmax(axis=0)
    return sp[idx, np.arange(preds.shape[1])]


def _fit(base, X, y, w):
    m = clone(base)
    try:
        return m.fit(X, y, sample_weight=w)
    except TypeError:                                   # a learner without sample_weight: weighted bootstrap instead
        rng = np.random.RandomState(0)
        idx = rng.choice(len(X), size=len(X), replace=True, p=w / w.sum())
        return clone(base).fit(X[idx], y[idx])


def _loss(pred, y):
    e = np.abs(pred - y)
    m = e.max()
    return e / m if m > 0 else np.zeros_like(e)


class TwoStageTrAdaBoostR2:
    def __init__(self, base_estimator=None, n_estimators=30, random_state=0):
        self.base_estimator = base_estimator if base_estimator is not None else Ridge(alpha=1.0)
        self.n_estimators, self.random_state = n_estimators, random_state

    def _loo_target_error(self, Xs, ys, Xt, yt, ws):
        """leave-one-out absolute error on the labelled target rows of a model trained on weighted source + the other target rows"""
        n_t, errs = len(Xt), []
        for i in range(n_t):
            k = np.arange(n_t) != i
            X = np.vstack([Xs, Xt[k]])
            y = np.r_[ys, yt[k]]
            w = np.r_[ws, np.ones(k.sum())]
            errs.append(abs(_fit(self.base_estimator, X, y, w + 1e-12).predict(Xt[i:i + 1])[0] - yt[i]))
        return float(np.mean(errs))

    def fit(self, Xs, ys, Xt, yt):
        n_s, n_t = len(Xs), len(Xt)
        N = max(int(self.n_estimators), 2)
        beta = 1.0 / (1.0 + np.sqrt(2.0 * np.log(max(n_s / N, 1.0 + 1e-6))))
        # ---- stage 1: only source weights move; target weights stay at 1
        ws = np.ones(n_s)
        cand = [ws.copy()]
        for _ in range(N):
            m = _fit(self.base_estimator, np.vstack([Xs, Xt]), np.r_[ys, yt], np.r_[ws, np.ones(n_t)] + 1e-12)
            loss_s = _loss(m.predict(np.vstack([Xs, Xt])), np.r_[ys, yt])[:n_s]
            ws = ws * np.power(beta, loss_s)
            cand.append(ws * n_s / ws.sum())                       # keep the source block at its original total mass, so only the relative weights differ
        scores = [self._loo_target_error(Xs, ys, Xt, yt, w) if n_t >= 3 else 0.0 for w in cand]
        self.stage1_step_ = int(np.argmin(scores)) if n_t >= 3 else N
        self.source_weights_ = cand[self.stage1_step_]
        self.stage1_scores_ = scores
        # ---- stage 2: freeze source weights, AdaBoost.R2 on the target rows only
        X, y = np.vstack([Xs, Xt]), np.r_[ys, yt]
        wt = np.ones(n_t) / n_t
        self.estimators_, self.est_w_ = [], []
        for _ in range(N):
            w = np.r_[self.source_weights_ / max(self.source_weights_.sum(), 1e-12) * (n_s / (n_s + n_t)), wt * (n_t / (n_s + n_t))]
            m = _fit(self.base_estimator, X, y, w + 1e-12)
            lt = _loss(m.predict(Xt), yt)
            err = float(np.clip(np.sum(wt * lt), 1e-6, 1 - 1e-6))
            if err >= 0.5 and self.estimators_:
                break
            b = err / (1 - err)
            wt = wt * np.power(b, 1.0 - lt)
            wt = wt / wt.sum()
            self.estimators_.append(m)
            self.est_w_.append(np.log(1.0 / b))
        self.n_source_ = n_s
        self.final_weights_ = np.r_[self.source_weights_, wt * n_t]
        return self

    def predict(self, X):
        P = np.array([e.predict(X) for e in self.estimators_])
        return _wmedian(P, self.est_w_)


class STrAdaBoostR2:
    def __init__(self, base_estimator=None, n_estimators=30, ratio=3.0, random_state=0):
        self.base_estimator = base_estimator if base_estimator is not None else Ridge(alpha=1.0)
        self.n_estimators, self.ratio, self.random_state = n_estimators, ratio, random_state

    def fit(self, Xs, ys, Xt, yt, sw_s=None, sw_t=None):
        """sw_s, sw_t: optional initial row weights (here the optical-quality weights w = max(0.2, Optical_Valid_Fraction)); a low-quality row starts with less say"""
        rng = np.random.RandomState(self.random_state)
        n_s, n_t = len(Xs), len(Xt)
        N = max(int(self.n_estimators), 2)
        ws = np.ones(n_s) / n_s if sw_s is None else np.asarray(sw_s, float) / np.sum(sw_s)
        wt = np.ones(n_t) / n_t if sw_t is None else np.asarray(sw_t, float) / np.sum(sw_t)
        m_s = int(min(n_s, max(2, round(self.ratio * n_t))))        # importance-sample size: a big source cannot swamp a small target
        self.estimators_, self.est_w_, self.sample_counts_ = [], [], np.zeros(n_s)
        for _ in range(N):
            idx = rng.choice(n_s, size=m_s, replace=False, p=ws / ws.sum()) if m_s < n_s else np.arange(n_s)
            self.sample_counts_[idx] += 1
            X = np.vstack([Xs[idx], Xt])
            y = np.r_[ys[idx], yt]
            w = np.r_[ws[idx] / ws[idx].sum() * 0.5, wt * 0.5]       # balanced: source block and target block carry half the mass each
            m = _fit(self.base_estimator, X, y, w + 1e-12)
            ls, lt = _loss(m.predict(Xs), ys), _loss(m.predict(Xt), yt)
            err = float(np.clip(np.sum(wt * lt), 1e-6, 1 - 1e-6))
            err = min(err, 0.5 - 1e-6)                                # AdaBoost.R2 stops at 0.5; here the round is kept with a floor weight instead
            b = err / (1 - err)
            ws = ws * np.power(b, ls)                                # source rows that fit badly decay (b < 1); no floor, so a bad source row can fall without limit
            wt = wt * np.power(b, 1.0 - lt)                          # target rows: the AdaBoost.R2 update (hard rows gain relative weight)
            ws, wt = ws / ws.sum(), wt / wt.sum()                    # each block renormalised to its own mass: balanced
            self.estimators_.append(m)
            self.est_w_.append(max(np.log(1.0 / b), 1e-3))
        self.n_source_ = n_s
        self.final_weights_ = np.r_[ws * n_s, wt * n_t]
        return self

    def predict(self, X):
        half = len(self.estimators_) // 2
        P = np.array([e.predict(X) for e in self.estimators_[half:]])
        return _wmedian(P, self.est_w_[half:])
