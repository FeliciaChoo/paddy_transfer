# ACTS transfer-learning workflow: guide

Everything below describes code that exists and was tested. Where something is only proposed, it says so.

## 1. Navigation: the required order
**Overview → Dynamic maps → Data Assessment → Transfer Strategy → Train and Evaluate Model → Yield Prediction → Future Yield Outlook → Intervention Advisory.**
The first two tabs are unchanged. Tabs 3 to 8 share a status strip (six stages, each complete, ready, locked, blocked or partial, with the reason) and show a notice when a change
upstream cleared downstream results. Older features were kept and moved: the reported-yield and baseline-outlook views (all granaries) sit inside tabs 6 and 7; the
decision tracker is the second sub-tab of tab 8; the earlier model checks (transfer learning on the yield-outlook method, stored TrAdaBoost results) sit inside tab 5.

## 2. What each tab needs, produces, and does when something upstream changes
| Tab | Needs | Produces | If an upstream step changes |
|---|---|---|---|
| 3 Data Assessment | the selected source and target data | assessment, harmonised data, suitability verdict | everything below is cleared |
| 4 Transfer Strategy | a suitability verdict | the scenario A to F reading, a recommended method list, a confirmed plan | evaluation, predictions, outlook are cleared |
| 5 Train and Evaluate Model | a confirmed plan | metrics for every method and baseline under two validations, the verdict on whether transfer helps, the model of record | predictions and outlook are cleared |
| 6 Yield Prediction | an evaluated model | validated (held-out) predictions and predictions for chosen inputs, each naming its model | recalculated from the new model on the next view |
| 7 Future Yield Outlook | an evaluated model | the outlook for 1 to 4 years under a stated input assumption, with ranges, against the baseline outlook | recalculated the same way |
| 8 Intervention Advisory | crop condition (always); the validated prediction when the model was built for that granary | advice, report, decision tracker | the prediction line is dropped or refreshed |
Tab 8 is not locked as a whole, on purpose: the advisory covers all ten granaries and works from crop condition alone, while the validated yield prediction is used only for
the granary the model was built for. The status line at the top of tab 8 says which inputs the advice rests on.

## 3. Files in this change
Modified: `app.py` (tab order), `dashboard/tab_transfer.py` (split into the assessment, strategy, training, prediction and outlook tabs and a status strip), `dashboard/tab_yield.py`
(two wrappers), `dashboard/tab_advisory.py` and `dashboard/bridge.py` (the validated prediction as evidence), `dashboard/tab_overview.py` (three tab labels and one line),
`acts_transfer/workflow.py`, `selection.py`, `config.py`, `report.py`.
New: `acts_transfer/modelchoice.py`, `acts_transfer/predict.py`, `tests/test_pipeline_order.py`; rewritten: `tests/test_ui_gating.py`.

## 4. Assessment metrics and formulas
- **Standardised mean difference** (one feature): d = (mean_t - mean_s) / s_p with s_p = sqrt(((n_s-1) s_s^2 + (n_t-1) s_t^2) / (n_s+n_t-2)). Size labels use configurable cut-offs (default 0.2 / 0.5 / 0.8).
- **Kolmogorov-Smirnov** (one feature): D = max |F_s(x) - F_t(x)|, two-sample p-value, Benjamini-Hochberg adjusted across features.
- **Wasserstein-1** (one feature): the integral of |F_s - F_t|, in the feature's units, and divided by s_p for a unitless version.
- **MMD^2, unbiased** (all shared features together, pooled-standardised): sum over i != j of k(x_i, x_j) / (n(n-1)) + sum over i != j of k(y_i, y_j) / (m(m-1)) - 2 sum of k(x_i, y_j) / (nm), with k = exp(-|x-y|^2 / 2 sigma^2) and sigma the median pairwise distance. Permutation p-value (domain labels shuffled).
- **CORAL distance**: |C_s - C_t|_F^2 / (4 d^2) on pooled-standardised features; permutation p-value; reported as not interpretable when there are fewer rows than features.
- **Domain classifier AUC**: 5-fold cross-validated ROC AUC of a logistic regression separating source rows from target rows (climate features; vegetation and radar features), at ten-day level. 0.5 is chance. A domain is called far at or above the configurable level (default 0.85, a convention).
- **Calendar offset**: the circular shift k (in ten-day periods, converted to days) that maximises the correlation of the two mean seasonal NDVI curves. Informational only.
- **Label shift**: SMD, KS and Wasserstein on yield, only when both domains have labels.
Which metric for what: SMD and Wasserstein describe one feature in comparable units; KS detects any shape difference in one feature; MMD and CORAL look at all features jointly; AUC is an effect size that does not saturate when n is large. They are never added into one score.
Interpretation uses significance tests (after correction) and configurable effect-size cut-offs; there is no hidden universal threshold. All thresholds are in `Config`.

## 5. Scenarios and selection logic
| Scenario | Conditions (from the assessment) | Recommended | Runnable here |
|---|---|---|---|
| A Similar distributions | no joint, per-feature or label difference | source-only, pooled, target-only comparison | yes |
| B Moderate shift, compatible features | a shift, units agree, no covariance or label shift | KMM, TrAdaBoost.R2 | yes |
| C Significant feature differences | covariance differs, or at least half the features differ, or at least two large SMDs | CORAL, TCA | yes |
| D A useful source model exists | Ridge on the source beats the source mean in leave-one-out | parameter transfer (source Ridge plus a regularised target correction) | Ridge only |
| E Domain shift needing adaptation | joint MMD significant; type follows the target labels actually available (supervised, semi-supervised, unsupervised) | TrAdaBoost.R2 / parameter transfer / CORAL, or KMM / CORAL / TCA without labels | yes |
| F Severe mismatch or negative-transfer risk | not ready; or features and yield both differ significantly; or climate AUC at the far level; or fewer than 2 target labels; or a large yield shift with a feature shift and few labels | collect data, improve harmonisation, change source, keep a target-only baseline | baselines only |
Overall verdict: not recommended if F applies; recommended if B, C or E applies, readiness is ready and there are enough labelled target rows; otherwise conditionally recommended, with the reasons listed.
Each scenario shows observed conditions, evidence, recommended methods, why, assumptions, benefits and risks, alternatives, and whether this system can run it. The user can choose any eligible alternative.

## 6. Implemented versus proposed
Implemented and runnable: source-only, target-only, target-mean (no features), season-trend (no features), naive pooled, TrAdaBoost.R2, Kernel Mean Matching with a weighted model, CORAL, Transfer Component Analysis (linear kernel), parameter transfer for Ridge (source model plus regularised target correction). Base models: Ridge, Random Forest, and XGBoost only if installed.
Not implemented, and never offered as runnable: fine-tuning of a neural network (no neural model exists), adversarial domain adaptation (DANN), and fine-tuning of Random Forest or XGBoost (trees have no weights to initialise from, so it is not the same operation). Parameter transfer is refused for non-Ridge base models.

## 7. Validation and comparison
One fold per held-out target season. **Leave-one-season-out** trains on all other target seasons (it uses later seasons too, so it can be optimistic); **forward** trains only on earlier seasons (needs at least two). Both are always run. Inside every fold all fitting (imputation, standardisation, CORAL and TCA alignment, KMM weights, boosting, penalties) uses training rows only; the held-out season's features and labels are never seen. A scarcity setting limits how many target training seasons a fold may use. Every method is scored on the same held-out rows.
Reported: MAE, RMSE, R^2 (only with at least 10 held-out points, otherwise a stated reason), predicted versus actual, absolute errors, error distribution, per-season errors, and the change in MAE against each baseline with a bootstrap interval over seasons (only with at least 5 seasons; otherwise "not testable"). Baselines: source-only, target-only, target mean, season trend, naive pooled. A domain distance is never presented as a prediction error.

## 8. How the interface explains increased or decreased error
Step 7 has a diagnostics table per method and baseline. Each row has a measured quantity, a status and a limitation. **Verified finding** means the measured condition is true in this run (it does not prove it caused the error); **Possible explanation** means consistent with the data but unproven; **Not supported** means the measurements point against it. Hypotheses: distribution difference, irrelevant or badly weighted source rows (effective sample size), too few target rows, a different feature-yield relationship (coefficient agreement), feature or unit mismatch, one season driving the difference, overfitting (training versus held-out error), unstable evaluation from few seasons, a time trend that features may only proxy, and violated method assumptions.

## 9. Commands
```
pip install -r requirements.txt                       # dashboard and the workflow
python pipeline/build_transfer_dataset.py --clean eda/DailyGranary_clean.csv --out data   # only if you rebuild the datasets
streamlit run app.py                                  # open tab 6, then tab 7
python -m pytest tests -q                             # 49 tests
python pipeline/validate_transfer_assessment.py --out data   # does the assessment predict transfer harm? (all targets)
```

## 10. Tests that transfer cannot bypass the checks
`tests/test_transfer_gating.py` (27 tests): running with nothing, with assessment only, with assessment and harmonisation but no suitability, or with no strategy all raise; changed data after the assessment or after harmonisation is refused (fingerprint mismatch); changing the domains clears every later step; a report from other data cannot harmonise; `evaluate()` rejects raw data; not-ready data blocks harmonisation; a NOT RECOMMENDED verdict needs an override reason; ineligible and not-implemented methods are refused; parameter transfer needs a Ridge base; harmonisation keeps the originals untouched; invalid values become missing; unavailable features are listed as proposed and never invented; SMD and MMD behave correctly on known distributions; a held-out season's label can never change its own prediction; the training arrays never contain the held-out season; forward validation uses only earlier seasons; the scarcity setting is respected.
`tests/test_ui_gating.py` (3 tests): tab 7 is locked and no later-step button exists at the start; each step unlocks only the next one; the backend still refuses when the page is bypassed.


## 11. Real results on the project data

### 11.1 IADA Barat Laut Selangor (Sekinchan area) to MADA, yield in t/ha, 7 held-out seasons
| method | Leave-one-season-out | Forward |
|---|---|---|
| season_trend | 0.151 | 0.255 |
| tradaboost_r2 | 0.129 | 0.271 |
| coral | 0.223 | 0.286 |
| tca | 0.304 | 0.307 |
| ridge_prior | 0.289 | 0.308 |
| target_only | 0.179 | 0.345 |
| pooled | 0.296 | 0.374 |
| target_mean | 0.511 | 0.570 |
| source_only | 0.768 | 0.619 |

Both validations are shown because MADA's yield falls steadily over the seven years (Spearman -0.93 with season). Leave-one-season-out can interpolate between neighbouring
seasons, so it flatters every model. Under forward validation the pure season-trend baseline is the best, and no transfer method beats it.

### 11.2 Does the assessment predict where transfer helps? All nine targets, one source, forward validation
| Target (source: IADA Barat Laut Selangor) | Suitability verdict | Climate AUC | Median change in error vs target-only (%) | Median change vs season trend (%) |
|---|---|---|---|---|
| IADA Kemasin Semerak | not_recommended | 0.800 | -21.070 | -39.840 |
| IADA Kerian | not_recommended | 0.570 | -2.540 | 22.230 |
| IADA Pekan | not_recommended | 0.570 | 211.240 | 124.300 |
| IADA Pulau Pinang | not_recommended | 0.600 | 24.780 | -26.880 |
| IADA Rompin | not_recommended | 0.690 | -26.810 | -55.510 |
| IADA Seberang Perak | not_recommended | 0.610 | 12.380 | -1.120 |
| KADA | conditionally_recommended | 0.690 | 44.000 | 16.510 |
| KETARA | not_recommended | 0.960 | 109.390 | 43.710 |
| MADA | conditionally_recommended | 0.470 | -11.090 | 20.190 |

Spearman correlations of the median change in error against target-only with: climate AUC -0.10 (p = 0.80), vegetation/radar AUC -0.25 (p = 0.52), ten-day MMD^2 -0.25 (p = 0.52), calendar offset +0.09 (p = 0.81), label SMD +0.07 (p = 0.86), over 9 targets. None is significant.

What the assessment does catch: **KETARA**, the only target above the far-domain level (climate AUC 0.96), where every transfer method was worse than target-only. What it misses: **KADA** (rated conditionally recommended, climate AUC 0.69), where every method was also worse. Of the 7 NOT RECOMMENDED targets, 3 improved on target-only; of the 2 conditionally recommended, 1 improved. So the verdict is a caution that calls for a baseline comparison, not a prediction. With 7 yearly yields per target, every row is noisy.

## 12. The model of record, predictions and the outlook
After evaluation `acts_transfer/modelchoice.py` decides which model feeds tabs 6 to 8. The best baseline is chosen by forward-validation error. A transfer method replaces it only
if all five hold: (1) lower forward MAE than that baseline; (2) a bootstrap interval over held-out seasons that excludes zero; (3) also better under leave-one-season-out;
(4) at least `adopt_min_seasons` (5) paired held-out seasons; (5) the run was not an override of a NOT RECOMMENDED verdict. Otherwise transfer learning is not used for any prediction,
and the interface says why. Predictions and the outlook come from `acts_transfer/predict.py`: every row names its model and role; ranges are the 80th percentile of the
model's held-out errors (for the season-trend model, measured at the same number of seasons ahead where it could be tested, on only 2 to 4 tests, so indicative); inputs are an
explicit assumption (typical season, last observed season, custom values, or an uploaded table) because this project has no weather forecast; non-numeric or missing inputs are rejected, never filled in.
A prediction of higher yield is not a realised improvement: transfer learning changes the prediction, not the field.


## PACE methodology additions (MADA → IADA Barat Laut Selangor)

**Safeguard (E2).** Delta_NT = RMSE_transfer − min(RMSE_target-only Ridge, RMSE_persistence), on the primary validation. Delta_NT < 0 deploys the best transfer method; Delta_NT ≥ 0 blocks transfer and the model of record becomes
the best *local* baseline (target-only, persistence, season trend or target mean). Override runs are never deployed. `Config.nt_gate`: `delta_nt` (default) or `delta_nt+stats` (also needs the bootstrap interval to
exclude zero, no worse under LOYO, and enough held-out years). Criteria 1 to 4 are always computed and displayed as the strength of the evidence.

**Look-ahead guard.** Pooled source rows from the held-out year (and later years in forward validation) are excluded from training in every fold (`Config.source_time_safe`).

**Methods.**
- *Granary-Intercept Ridge*: y = a_g + Zb. One slope vector; per-granary intercepts a_g = m + n_g/(n_g + 1)(mean residual_g − m), fitted by back-fitting. The target's intercept uses its training years only.
- *KMM + Weighted Ridge*: kernel-mean-matching weights on features only (RBF, median-heuristic bandwidth), weighted Ridge.
- *Two-Stage TrAdaBoost.R2* (Pardoe & Stone 2010): stage 1 shrinks source weights only, the step is chosen by leave-one-out error on the target rows (step 0 = no re-weighting is allowed to win); stage 2 freezes them and boosts the target rows.
- *S-TrAdaBoost.R2* (arXiv:2204.12044): each round importance-samples at most 3 × n_target source rows by weight, fits, decays badly fitted source rows, applies the AdaBoost.R2 update to the target rows and renormalises each block. Written from the published idea; not the authors' code.
- *Z-Score Anomaly Transfer*: within-granary Z-scores of features and yield (target statistics from its training rows; target spread shrunk toward the source spread), Ridge on anomalies, target level and spread added back.

**In-fold harmonisation (`methods.Prep`).** Median imputation → RobustScaler (or StandardScaler) → VIF/correlation selection (VIF > 10 or |r| > 0.95 dropped first, then the least label-related, until p ≤ 6). Fitted on the rows a method is given, nothing else.

**Metrics.** MAE and RMSE (also in kg/ha), rRMSE %, MAPE %, R² (reported at ≥ 10 held-out points; an indicative value otherwise), win rate against persistence, bootstrap 95% intervals for RMSE and MAE resampling the held-out seasons.

**Stage features (`pipeline/build_stage_features.py`).** See the module docstring. Cycle 2 is detected in 32 of 70 granary-years, so C2_* features are opt-in.
