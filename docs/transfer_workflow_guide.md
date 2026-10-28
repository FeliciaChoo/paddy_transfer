# ACTS transfer-learning workflow: guide

Everything below describes code that exists and was tested. Where something is only proposed, it says so.

## 1. Where Pre-Transfer Assessment is
Main navigation, tab **6 · Pre-Transfer Assessment** (steps 1 to 4). It comes before tab **7 · Transfer Learning** (steps 5 to 7). Both tabs show the same progress bar
(seven steps, each marked complete, current, locked or blocked), and tab 7 explains why it is locked and sends the user back to tab 6. The Overview page lists both tabs.

## 2. The user journey
1. **Select domains** (source, target, base model, how many target seasons a model may train on, thresholds). Clicking *Confirm domains* loads the real data.
2. **Assess data**: quality, feature compatibility (available versus only proposed), distribution differences at the yearly and the ten-day level, domain classifier, risks, next steps, downloadable report.
3. **Harmonise data**: unit conversion, renaming, de-duplication, invalid values set to missing, features retained or removed, validation checks. Nothing that is fitted is fitted here.
4. **Evaluate suitability**: recommended, conditionally recommended or not recommended, with scenarios A to F and their evidence.
5. **Select strategy** (tab 7): eligible methods, how each uses features and distance; a NOT RECOMMENDED verdict needs a written override reason.
6. **Run**: all methods plus baselines, with both validation strategies.
7. **Compare**: metrics, validation sensitivity, improvement against a chosen baseline, predicted versus actual, error distribution, seasonal errors, diagnostics, downloads.

## 3. Files
Created: `acts_transfer/` (`config.py`, `assessment.py`, `harmonise.py`, `selection.py`, `methods.py`, `_tradaboost.py`, `evaluation.py`, `diagnostics.py`, `workflow.py`, `errors.py`, `report.py`, `_util.py`),
`dashboard/tab_transfer.py`, `pipeline/build_transfer_dataset.py`, `pipeline/validate_transfer_assessment.py`, `pipeline/transfer_error_diagnosis.py`, `tests/test_transfer_gating.py`, `tests/test_ui_gating.py`, `data/transfer_yield_dataset.csv`, `data/transfer_decade_features.csv`.
Modified: `app.py` (three tabs), `dashboard/tab_overview.py` (one line), `requirements.txt` (scikit-learn, scipy), `README.md`.
Nothing existing was replaced; the other tabs are unchanged.

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
python -m pytest tests -q                             # 30 tests
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
