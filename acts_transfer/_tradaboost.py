"""
tradaboost_r2.py
----------------
Implementation of TrAdaBoost.R2 (Pardoe, D., & Stone, P. (2010).
"Boosting for regression transfer." Proceedings of the 27th International
Conference on Machine Learning (ICML 2010), 863-870.)

TrAdaBoost.R2 is the regression extension of TrAdaBoost (Dai et al., 2007).
It boosts a weak regressor over N boosting rounds, where the training set is
the union of SOURCE (data-rich) and TARGET (data-scarce) instances. Each
round:
  1. Fits the weak learner on the current weighted distribution.
  2. Computes per-instance absolute relative error.
  3. Decreases the weight of SOURCE instances that have high error (i.e. the
     algorithm "forgets" source examples that the model disagrees with --
     evidence that they don't transfer well to the target task).
  4. Increases the relative influence of TARGET instances using an AdaBoost.R2
     -style multiplicative update.
The final prediction is a weighted median over the second half of boosting
rounds (as specified in the original paper), since early rounds are
considered less reliable.

This file is used for Kedah (target, data-scarce) <- Sekinchan (source,
data-rich) season-level yield transfer, replacing the simpler "pooled +
z-scored + up-weighted" approach with the formally cited TrAdaBoost.R2
algorithm.
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone


class TrAdaBoostR2(BaseEstimator, RegressorMixin):
    """TrAdaBoost.R2 (Pardoe & Stone, 2010).

    Parameters
    ----------
    base_estimator : sklearn regressor, default=DecisionTreeRegressor(max_depth=3)
        Weak learner fit at each boosting round.
    n_estimators : int, default=30
        Number of boosting rounds (N in the paper).
    learning_rate : float, default=1.0
        Shrinkage applied to the AdaBoost.R2-style weight update on target
        instances.
    random_state : int, optional
    """

    def __init__(self, base_estimator=None, n_estimators=30, learning_rate=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X_source, y_source, X_target, y_target):
        rng = np.random.RandomState(self.random_state)
        n_s, n_t = len(X_source), len(X_target)
        X = np.vstack([X_source, X_target])
        y = np.concatenate([y_source, y_target])
        n = n_s + n_t

        # initial uniform weights, split evenly across source/target blocks
        weights = np.ones(n) / n

        self.estimators_ = []
        self.estimator_weights_ = []  # beta_t per round, used for weighted median
        self.estimator_errors_ = []

        # Beta value for source-instance decay, fixed per Dai et al. (2007)/Pardoe & Stone (2010):
        # beta = 1 / (1 + sqrt(2 * ln(n_s / N)))
        N = self.n_estimators
        log_arg = n_s / max(N, 1)
        if log_arg <= 1.0:
            log_arg = 1.0 + 1e-6  # safety guard if ever called with N >= n_source
        beta = 1.0 / (1.0 + np.sqrt(2.0 * np.log(log_arg)))

        for t in range(N):
            base = clone(self.base_estimator) if self.base_estimator is not None else DecisionTreeRegressor(max_depth=3, random_state=rng.randint(1e6))

            # weighted bootstrap sample to fit the weak learner on this round's distribution
            p = weights / weights.sum()
            idx = rng.choice(n, size=n, replace=True, p=p)
            base.fit(X[idx], y[idx])

            preds = base.predict(X)
            abs_err = np.abs(preds - y)
            max_err = abs_err.max() if abs_err.max() > 0 else 1e-12
            loss = abs_err / max_err  # normalised loss in [0, 1], as in AdaBoost.R2

            # target-domain weighted average loss for this round
            target_loss = np.average(loss[n_s:], weights=weights[n_s:])
            target_loss = np.clip(target_loss, 1e-6, 1 - 1e-6)
            beta_t = target_loss / (1.0 - target_loss)

            # weight update:
            #   source instances: decay by beta^loss  (forget poorly-transferring source rows)
            #   target instances: AdaBoost.R2-style update by beta_t^-loss (boost hard target rows)
            weights[:n_s] *= np.power(beta, loss[:n_s])
            weights[n_s:] *= np.power(beta_t, -self.learning_rate * loss[n_s:])
            weights /= weights.sum()

            self.estimators_.append(base)
            self.estimator_weights_.append(np.log(1.0 / beta_t) if beta_t > 0 else 0.0)
            self.estimator_errors_.append(target_loss)

        self.final_weights_ = weights.copy()      # added for diagnostics only; the algorithm is unchanged
        self.n_source_ = n_s
        return self

    def predict(self, X):
        # Per Pardoe & Stone (2010): use only the SECOND HALF of boosting rounds
        # (N/2 .. N), weighted-median combination, since early rounds are
        # dominated by source-only fitting before target weights stabilise.
        half = len(self.estimators_) // 2
        preds = np.array([e.predict(X) for e in self.estimators_[half:]])  # (n_rounds_used, n_samples)
        wts = np.array(self.estimator_weights_[half:])
        if wts.sum() == 0:
            return preds.mean(axis=0)

        # weighted median across rounds, per sample
        order = np.argsort(preds, axis=0)
        sorted_preds = np.take_along_axis(preds, order, axis=0)
        sorted_wts = wts[order.reshape(-1)].reshape(order.shape) if wts.ndim else wts[order]
        # simpler: weights don't depend on sample, so just sort weight vector aligned to each column
        sorted_wts = np.tile(wts.reshape(-1, 1), (1, preds.shape[1]))
        sorted_wts = np.take_along_axis(sorted_wts, order, axis=0)

        cum_wts = np.cumsum(sorted_wts, axis=0)
        cutoff = cum_wts[-1] / 2.0
        median_idx = (cum_wts >= cutoff).argmax(axis=0)
        return sorted_preds[median_idx, np.arange(preds.shape[1])]