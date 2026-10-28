# Transfer-learning assessment and evaluation: IADA Barat Laut Selangor to MADA

## Workflow status
| step | status | detail |
|---|---|---|
| 1 · Select domains | complete |  |
| 2 · Assess data | complete |  |
| 3 · Harmonise data | complete |  |
| 4 · Evaluate suitability | complete |  |
| 5 · Select strategy | complete |  |
| 6 · Run transfer | complete |  |
| 7 · Compare results | complete |  |

## Pre-transfer assessment: IADA Barat Laut Selangor to MADA

Computed 2026-10-26 02:13:20; fingerprint `c945db21c1e25e0b`.

**Data readiness: CONDITIONAL**

## Readiness reasons
- source: Optical_observed_pct below 20% in Year [2018]: those rows rest mostly on gap-filled inputs
- target: Optical_observed_pct below 20% in Year [2018]: those rows rest mostly on gap-filled inputs
- 7 labelled target rows for 8 features (the configured rule of thumb is 3 rows per feature)
- CORAL at the modelling level is not interpretable: the covariance of 8 features is estimated from 7 rows (fewer rows than features).

## Source data: IADA Barat Laut Selangor
- 7 observations, 7 groups ([2018, 2019, 2020, 2021, 2022, 2023, 2024]), 7 labelled, 8 features
- [warn] Optical_observed_pct below 20% in Year [2018]: those rows rest mostly on gap-filled inputs

## Target data: MADA
- 7 observations, 7 groups ([2018, 2019, 2020, 2021, 2022, 2023, 2024]), 7 labelled, 8 features
- [info] SAR_VV_dB_mean: 1 extreme outlier(s) beyond 3.0 x IQR (kept; check them)
- [warn] Optical_observed_pct below 20% in Year [2018]: those rows rest mostly on gap-filled inputs

## Feature compatibility
| feature | status | available_in_source | available_in_target | unit_source | unit_target | note |
|---|---|---|---|---|---|---|
| Annual_Rain_mm | shared | True | True | mm per year | mm per year |  |
| EVI_mean | shared | True | True | unitless | unitless |  |
| Mean_Temp_C_mean | shared | True | True | deg C | deg C |  |
| NDVI_amplitude | shared | True | True | unitless | unitless |  |
| NDVI_max | shared | True | True | unitless | unitless |  |
| NDVI_mean | shared | True | True | unitless | unitless |  |
| SAR_VV_dB_mean | shared | True | True | dB | dB |  |
| Solar_Radiation_MJ_mean | shared | True | True | MJ/m2 | MJ/m2 |  |
| humidity | PROPOSED, not available | False | False | - | - | not in either dataset; it is not used and is never filled in |
| land surface temperature | PROPOSED, not available | False | False | - | - | not in either dataset; it is not used and is never filled in |
| irrigation / water supply | PROPOSED, not available | False | False | - | - | not in either dataset; it is not used and is never filled in |
| planting date | PROPOSED, not available | False | False | - | - | not in either dataset; it is not used and is never filled in |
| variety | PROPOSED, not available | False | False | - | - | not in either dataset; it is not used and is never filled in |
| fertiliser | PROPOSED, not available | False | False | - | - | not in either dataset; it is not used and is never filled in |

## Domain differences: Modelling level
| feature | n_source | n_target | mean_source | mean_target | sd_source | sd_target | SMD | KS | KS_p | Wasserstein | Wasserstein_per_SD | KS_p_adj | SMD_size | KS_significant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Annual_Rain_mm | 7 | 7 | 2507.982 | 2480.751 | 500.131 | 326.401 | -0.064 | 0.286 | 0.963 | 188.978 | 0.448 | 0.963 | negligible | False |
| EVI_mean | 7 | 7 | 0.314 | 0.301 | 0.029 | 0.018 | -0.554 | 0.571 | 0.212 | 0.021 | 0.872 | 0.424 | medium | False |
| Mean_Temp_C_mean | 7 | 7 | 26.857 | 26.852 | 0.201 | 0.283 | -0.019 | 0.286 | 0.963 | 0.075 | 0.303 | 0.963 | negligible | False |
| NDVI_amplitude | 7 | 7 | 0.239 | 0.626 | 0.111 | 0.065 | 4.260 | 1.000 | 0.001 | 0.387 | 4.260 | 0.002 | large | True |
| NDVI_max | 7 | 7 | 0.565 | 0.795 | 0.061 | 0.008 | 5.273 | 1.000 | 0.001 | 0.230 | 5.273 | 0.002 | large | True |
| NDVI_mean | 7 | 7 | 0.444 | 0.445 | 0.012 | 0.043 | 0.038 | 0.429 | 0.575 | 0.025 | 0.797 | 0.767 | negligible | False |
| SAR_VV_dB_mean | 7 | 7 | -9.634 | -10.377 | 0.164 | 0.227 | -3.755 | 1.000 | 0.001 | 0.744 | 3.755 | 0.002 | large | True |
| Solar_Radiation_MJ_mean | 7 | 7 | 18.510 | 18.245 | 0.637 | 0.592 | -0.430 | 0.429 | 0.575 | 0.275 | 0.447 | 0.767 | small | False |
MMD^2 = 0.3196 (permutation p = 0.002; the smallest possible p is 1/(permutations+1)); CORAL = 0.0555 (p = 0.968; NOT interpretable: fewer rows than features)

## Domain differences: Ten-day level (finer observations)
| feature | n_source | n_target | mean_source | mean_target | sd_source | sd_target | SMD | KS | KS_p | Wasserstein | Wasserstein_per_SD | KS_p_adj | SMD_size | KS_significant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NDVI | 221 | 229 | 0.443 | 0.437 | 0.072 | 0.208 | -0.034 | 0.383 | 0.000 | 0.130 | 0.827 | 0.000 | negligible | True |
| EVI | 221 | 229 | 0.323 | 0.299 | 0.058 | 0.171 | -0.184 | 0.431 | 0.000 | 0.111 | 0.862 | 0.000 | negligible | True |
| SAR_VV_dB | 205 | 205 | -9.633 | -10.374 | 0.681 | 1.567 | -0.614 | 0.263 | 0.000 | 0.788 | 0.652 | 0.000 | medium | True |
| Mean_Temp_C | 252 | 252 | 26.857 | 26.852 | 0.689 | 1.028 | -0.005 | 0.123 | 0.044 | 0.279 | 0.319 | 0.066 | negligible | False |
| Rain_decade_mm | 252 | 252 | 69.666 | 68.910 | 45.780 | 47.428 | -0.016 | 0.099 | 0.168 | 5.968 | 0.128 | 0.201 | negligible | False |
| Solar_Radiation_MJ | 252 | 252 | 18.510 | 18.245 | 2.366 | 2.657 | -0.105 | 0.091 | 0.245 | 0.386 | 0.154 | 0.245 | negligible | False |
MMD^2 = 0.0801 (permutation p = 0.002; the smallest possible p is 1/(permutations+1)); CORAL = 0.1128 (p = 0.002)

## Domain classifier and crop calendar (ten-day level)
- Climate AUC 0.47
- Vegetation and radar AUC 0.73
- Calendar offset 120 days (curve correlation -0.59 unshifted, +0.85 aligned); informational

## Label (yield) shift
| n_source | n_target | mean_source | mean_target | SMD | KS | KS_p | Wasserstein |
|---|---|---|---|---|---|---|---|
| 7 | 7 | 4476.571 | 4448.000 | -0.074 | 0.429 | 0.575 | 260.857 |

## Negative-transfer risks
- Too few labelled target rows for a flexible model (7 rows, 8 features): estimates will be unstable and the evaluation noisy.
- 3 of 8 features differ in distribution at the modelling level (adjusted KS p < 0.05): ['NDVI_amplitude', 'NDVI_max', 'SAR_VV_dB_mean'].
- 3 feature(s) have a large standardised mean difference (|SMD| >= 0.8) at the modelling level.
- The joint feature distributions differ at the modelling level (MMD permutation p = 0.002).
- 3 of 6 features differ in distribution at the ten-day level (adjusted KS p < 0.05): ['NDVI', 'EVI', 'SAR_VV_dB'].
- The joint feature distributions differ at the ten-day level (MMD permutation p = 0.002).
- The feature covariance differs at the ten-day level (CORAL permutation p = 0.002).

## Recommended next steps
- Harmonise names and units (Step 3); invalid values will be set to missing, never kept.
- Review each warning; the workflow can continue, and the warnings stay in the report.
- Keep a target-only baseline and a source-only baseline in every comparison, because the risks above mean transfer may not help.

## What each metric means
- **SMD**: Mean difference in pooled-SD units for ONE feature. Unitless. Mean only; unstable with few rows.
- **KS**: Largest gap between the two cumulative distributions for ONE feature; detects shape and spread. Needs rows to have power; p adjusted (BH).
- **Wasserstein**: Average movement needed to turn one distribution into the other for ONE feature, in its own units (and per pooled SD).
- **MMD**: Joint difference over ALL shared features (RBF kernel, median bandwidth, standardised). Permutation p-value.
- **CORAL**: Difference in covariance (how features move together), standardised. Permutation p-value.
- **Label shift**: SMD / KS / Wasserstein on the yield itself (needs labels in both domains).
- **Domain AUC**: Cross-validated AUC of a classifier that tries to tell source rows from target rows (climate features; vegetation/radar features). 0.5 = indistinguishable, near 1 = very different. An effect size, so unlike a permutation p-value it does not saturate when n is large. Needs many rows.
- **Calendar offset**: Shift in days that best aligns the two mean seasonal NDVI curves (circular). Informational: models that use annual aggregates or phase features are much less sensitive to it than models that use calendar dates.

## Data harmonisation
Fingerprint `8b35fbf60d81baa8`; originals are unchanged.

### Transformations applied
| step | domain | detail | rows_affected |
|---|---|---|---|
| convert units | source | Yield_kg_ha x 0.001 -> Yield_t_ha (t/ha) | 7 |
| remove duplicates | source | exact duplicate rows | 0 |
| convert units | target | Yield_kg_ha x 0.001 -> Yield_t_ha (t/ha) | 7 |
| remove duplicates | target | exact duplicate rows | 0 |
| invalid values to missing | source | values outside the plausible range | 0 |
| invalid values to missing | target | values outside the plausible range | 0 |
| deferred to training folds | both | median (fitted inside each training fold); standardise (fitted inside each training fold) | 0 |

### Features
| feature | retained | reason | unit |
|---|---|---|---|
| Annual_Rain_mm | True | shared by both domains | mm per year |
| EVI_mean | True | shared by both domains | unitless |
| Mean_Temp_C_mean | True | shared by both domains | deg C |
| NDVI_amplitude | True | shared by both domains | unitless |
| NDVI_max | True | shared by both domains | unitless |
| NDVI_mean | True | shared by both domains | unitless |
| SAR_VV_dB_mean | True | shared by both domains | dB |
| Solar_Radiation_MJ_mean | True | shared by both domains | MJ/m2 |

### Validation
| check | passed | detail |
|---|---|---|
| at least one shared feature retained | True | 8 retained |
| same feature names in both domains | True | identical columns |
| all retained features numeric | True | numeric dtype |
| no infinite values | True | checked after conversion |
| labels present | True | source 7, target 7 labelled rows |
| no duplicate rows left | True | after de-duplication |
| one row per group | True | group = season/year |
| units agree for every retained feature | True | declared units compared after conversion |

## Transfer suitability
**CONDITIONALLY RECOMMENDED**

- the data readiness status carries warnings (see the assessment)
- only 6 labelled target rows per training fold for 8 features (rule of thumb: 3 per feature), so any difference found will be uncertain

### Scenarios A to F
**A. Similar source and target distributions**: not met
- Observed: A difference was detected, so this scenario does not apply.
- Evidence: modelling-level MMD p = 0.002; ten-day-level MMD p = 0.002; 3/8 features differ (adjusted KS); 3 with |SMD| >= 0.8
- Recommended: Source-only model, Pooled (naive), Target-only model; runnable here: source_only, pooled, target_only
- Why: If the domains look alike, transfer may be unnecessary: compare source-only and pooled models with a target-only model before adding complexity.
- Assumptions: Absence of a detected difference is not proof of similarity when there are few rows.
- Benefits and risks: Simplest option; risk is hidden differences that low power cannot detect.
- Alternatives: TrAdaBoost.R2 or KMM as a check

**B. Moderate distribution shift with compatible features**: not met
- Observed: Not met (covariance or label shift is present, or no shift was found).
- Evidence: modelling-level MMD p = 0.002; ten-day-level MMD p = 0.002; 3/8 features differ (adjusted KS); 3 with |SMD| >= 0.8; modelling-level covariance not interpretable (fewer rows than features); ten-day-level CORAL p = 0.002
- Recommended: Kernel Mean Matching + weighted model, TrAdaBoost.R2; runnable here: kmm, tradaboost_r2
- Why: Instance weighting: KMM weights source rows so their feature mean matches the target (no labels needed); TrAdaBoost.R2 lowers the weight of source rows that the model fits badly on target rows (needs labelled target rows).
- Assumptions: The feature-yield relationship is shared (KMM); enough target labels to judge source rows (TrAdaBoost).
- Benefits and risks: Keeps useful source rows and drops misleading ones; unstable weights with few rows.
- Alternatives: CORAL, TCA

**C. Significant feature-distribution differences**: CONDITIONS MET
- Observed: Several features, or the feature covariance, differ clearly.
- Evidence: modelling-level MMD p = 0.002; ten-day-level MMD p = 0.002; 3/8 features differ (adjusted KS); 3 with |SMD| >= 0.8; modelling-level covariance not interpretable (fewer rows than features); ten-day-level CORAL p = 0.002
- Recommended: CORAL feature alignment, Transfer Component Analysis; runnable here: coral, tca
- Why: Feature-based: CORAL aligns the source's mean and covariance to the target; TCA finds a shared space that cuts the domain gap.
- Assumptions: The relationship between the aligned features and yield is shared.
- Benefits and risks: Can remove a systematic offset; can also remove real signal, and the covariance is noisy with few rows.
- Alternatives: KMM, TrAdaBoost.R2

**D. A useful source model exists**: not met
- Observed: A Ridge model on the source has leave-one-out error 1.81 x that of the source mean (7 labelled rows).
- Evidence: source skill ratio = 1.81
- Recommended: Parameter transfer (source Ridge + regularised target correction); runnable here: ridge_prior
- Why: Parameter-based: keep the source Ridge coefficients and learn a strongly regularised correction from the target rows.
- Assumptions: Ridge base model; the source coefficients are a useful prior. Tree models cannot be fine-tuned this way.
- Benefits and risks: Needs very few target rows; inherits any source bias.
- Alternatives: TrAdaBoost.R2

**E. Domain shift requiring adaptation**: CONDITIONS MET
- Observed: A joint feature difference was detected; target labels available per training fold: 6 -> supervised.
- Evidence: modelling-level MMD p = 0.002; ten-day-level MMD p = 0.002; 3/8 features differ (adjusted KS); 3 with |SMD| >= 0.8
- Recommended: TrAdaBoost.R2, Parameter transfer (source Ridge + regularised target correction), CORAL feature alignment; runnable here: tradaboost_r2, ridge_prior, coral
- Why: Adaptation type follows the labels actually available: supervised.
- Assumptions: Adaptation helps only if the shift is in the features and not in the feature-yield relationship.
- Benefits and risks: May reduce the source-target gap; a feature gap does not by itself mean a worse prediction.
- Alternatives: source-only and target-only baselines

**F. Severe mismatch or negative-transfer risk**: not met
- Observed: None of the severe-mismatch conditions holds.
- Evidence: modelling-level MMD p = 0.002; ten-day-level MMD p = 0.002; 3/8 features differ (adjusted KS); 3 with |SMD| >= 0.8; label KS p = 0.575
- Recommended: none; runnable here: none
- Why: Do not force transfer: collect more target data, improve harmonisation, try a different source, or keep a target-only baseline.
- Assumptions: -
- Benefits and risks: Avoids a model that is worse than not transferring.
- Alternatives: a different source domain

### Method table
| method | family | eligible | reason | recommended_by | features_compared | scaling | metric | distance_influence | limitations |
|---|---|---|---|---|---|---|---|---|---|
| Source-only model | baseline | True | needs only source labels | - | none | source mean and SD | none | none | assumes the target shares the source's feature distribution and feature-yield relationship |
| Target-only model | baseline | True | 6 labelled target rows (a weak baseline when this is small) | - | none | target training-row mean and SD | none | none | unstable when the target has few labelled rows |
| Target mean (no features) | baseline | True | uses no features: the check on whether the features help at all | - | none | none | none | none | ignores every input, so it cannot respond to a poor or good season |
| Pooled (naive) | baseline | True | needs at least 1 labelled target row | - | none | pooled mean and SD of source and target training rows | none | none | treats every source row as equally relevant |
| TrAdaBoost.R2 | instance-based | True | 6 labelled target rows | E | none directly: relevance is judged by prediction error, not feature distance | pooled mean and SD of source and target training rows | normalised absolute error of the weak learner on each row (uses labels) | each of 30 rounds multiplies a source row's weight by beta^loss; the final weights are reported | needs enough labelled target rows to judge the source rows; with few the weights are unstable |
| Kernel Mean Matching + weighted model | instance-based | True | uses features only for the weights | - | all retained shared features | pooled mean and SD of source rows and target training rows | MMD with an RBF kernel (median-heuristic bandwidth); KMM minimises that discrepancy | weights in [0, 10] make the weighted source feature mean match the target's; the effective sample size is reported | assumes the feature-yield relationship is shared; weights are unstable where the domains barely overlap |
| CORAL feature alignment | feature-based | True | covariance from 6 target rows for 8 features: regularised, interpret with care | C, E | all retained shared features | pooled mean and SD, then covariance alignment | covariance difference (CORAL distance) and mean | no weights: the source features are re-coloured to the target covariance and mean | second-order only; the target covariance is noisy with few rows (a regulariser is added) |
| Transfer Component Analysis | feature-based | True | linear kernel, 4 components | C | all retained shared features | pooled mean and SD | MMD in a linear kernel space | learns 4 components that keep variance and cut the MMD | needs enough rows; ignores labels when choosing the components |
| Parameter transfer (source Ridge + regularised target correction) | parameter-based | True | penalty tuned by leave-one-out | E | none (corrects a source model with target labels) | source mean and SD | none; penalty by leave-one-out | the penalty chosen by leave-one-out sets how far the correction may move from the source model | Ridge only; assumes the source coefficients are a sensible starting point |
| fine_tune_nn | not implemented | False | Fine-tuning a neural network: no neural model exists in this system. | not offered | - | - | - | - | - |
| adversarial_da | not implemented | False | Adversarial domain adaptation (DANN): not implemented; it needs a neural feature extractor and far more rows. | not offered | - | - | - | - | - |
| tree_fine_tune | not implemented | False | Fine-tuning Random Forest or XGBoost: trees have no weights to initialise from, so this is not the same operation. | not offered | - | - | - | - | - |

## Transfer performance comparison (yield in the harmonised unit, held-out seasons only)
### Validation: loso (7 held-out seasons)
| method | n_test | MAE | RMSE | R2 | R2_note |
|---|---|---|---|---|---|
| tradaboost_r2 | 7 | 0.129 | 0.175 |  | not reported: fewer than 10 held-out points |
| season_trend | 7 | 0.151 | 0.192 |  | not reported: fewer than 10 held-out points |
| target_only | 7 | 0.179 | 0.195 |  | not reported: fewer than 10 held-out points |
| coral | 7 | 0.223 | 0.265 |  | not reported: fewer than 10 held-out points |
| ridge_prior | 7 | 0.289 | 0.311 |  | not reported: fewer than 10 held-out points |
| pooled | 7 | 0.296 | 0.383 |  | not reported: fewer than 10 held-out points |
| tca | 7 | 0.304 | 0.344 |  | not reported: fewer than 10 held-out points |
| target_mean | 7 | 0.511 | 0.541 |  | not reported: fewer than 10 held-out points |
| source_only | 7 | 0.768 | 0.845 |  | not reported: fewer than 10 held-out points |

Change in mean absolute error against each baseline (negative is better):
| method | baseline | n_seasons | mean_abs_err_change | relative_change_pct | ci95_low | ci95_high | better_in | verdict |
|---|---|---|---|---|---|---|---|---|
| coral | source_only | 7 | -0.545 | -70.924 | -0.795 | -0.305 | 7 | improved |
| coral | target_only | 7 | 0.045 | 25.065 | -0.068 | 0.146 | 2 | no clear difference |
| coral | target_mean | 7 | -0.288 | -56.280 | -0.428 | -0.166 | 7 | improved |
| coral | season_trend | 7 | 0.072 | 47.551 | -0.084 | 0.214 | 2 | no clear difference |
| coral | pooled | 7 | -0.072 | -24.432 | -0.169 | 0.032 | 5 | no clear difference |
| pooled | source_only | 7 | -0.473 | -61.523 | -0.784 | -0.207 | 6 | improved |
| pooled | target_only | 7 | 0.117 | 65.501 | -0.033 | 0.290 | 3 | no clear difference |
| pooled | target_mean | 7 | -0.215 | -42.144 | -0.438 | -0.029 | 5 | improved |
| pooled | season_trend | 7 | 0.144 | 95.258 | -0.023 | 0.355 | 3 | no clear difference |
| ridge_prior | source_only | 7 | -0.480 | -62.447 | -0.727 | -0.249 | 7 | improved |
| ridge_prior | target_only | 7 | 0.110 | 61.526 | 0.004 | 0.192 | 1 | worse |
| ridge_prior | target_mean | 7 | -0.222 | -43.534 | -0.331 | -0.116 | 7 | improved |
| ridge_prior | season_trend | 7 | 0.137 | 90.567 | -0.018 | 0.289 | 2 | no clear difference |
| ridge_prior | pooled | 7 | -0.007 | -2.402 | -0.140 | 0.153 | 4 | no clear difference |
| season_trend | source_only | 7 | -0.617 | -80.294 | -0.940 | -0.310 | 7 | improved |
| season_trend | target_only | 7 | -0.027 | -15.239 | -0.121 | 0.067 | 4 | no clear difference |
| season_trend | target_mean | 7 | -0.360 | -70.370 | -0.542 | -0.184 | 6 | improved |
| season_trend | pooled | 7 | -0.144 | -48.786 | -0.355 | 0.023 | 4 | no clear difference |
| source_only | target_only | 7 | 0.590 | 330.129 | 0.317 | 0.869 | 0 | worse |
| source_only | target_mean | 7 | 0.257 | 50.363 | -0.001 | 0.503 | 2 | no clear difference |
| source_only | season_trend | 7 | 0.617 | 407.464 | 0.310 | 0.940 | 0 | worse |
| source_only | pooled | 7 | 0.473 | 159.894 | 0.207 | 0.784 | 1 | worse |
| target_mean | source_only | 7 | -0.257 | -33.494 | -0.503 | 0.001 | 5 | no clear difference |
| target_mean | target_only | 7 | 0.332 | 186.060 | 0.212 | 0.451 | 0 | worse |
| target_mean | season_trend | 7 | 0.360 | 237.492 | 0.184 | 0.542 | 1 | worse |
| target_mean | pooled | 7 | 0.215 | 72.844 | 0.029 | 0.438 | 2 | worse |
| target_only | source_only | 7 | -0.590 | -76.751 | -0.869 | -0.317 | 7 | improved |
| target_only | target_mean | 7 | -0.332 | -65.042 | -0.451 | -0.212 | 7 | improved |
| target_only | season_trend | 7 | 0.027 | 17.979 | -0.067 | 0.121 | 3 | no clear difference |
| target_only | pooled | 7 | -0.117 | -39.578 | -0.290 | 0.033 | 4 | no clear difference |
| tca | source_only | 7 | -0.465 | -60.460 | -0.703 | -0.238 | 6 | improved |
| tca | target_only | 7 | 0.125 | 70.073 | 0.021 | 0.243 | 2 | worse |
| tca | target_mean | 7 | -0.207 | -40.546 | -0.363 | -0.058 | 6 | improved |
| tca | season_trend | 7 | 0.152 | 100.651 | -0.009 | 0.329 | 2 | no clear difference |
| tca | pooled | 7 | 0.008 | 2.762 | -0.088 | 0.119 | 4 | no clear difference |
| tradaboost_r2 | source_only | 7 | -0.639 | -83.198 | -0.901 | -0.405 | 7 | improved |
| tradaboost_r2 | target_only | 7 | -0.050 | -27.730 | -0.141 | 0.052 | 5 | no clear difference |
| tradaboost_r2 | target_mean | 7 | -0.382 | -74.736 | -0.545 | -0.223 | 7 | improved |
| tradaboost_r2 | season_trend | 7 | -0.022 | -14.737 | -0.175 | 0.139 | 4 | no clear difference |
| tradaboost_r2 | pooled | 7 | -0.167 | -56.333 | -0.299 | -0.010 | 5 | improved |

Absolute error by season:
| group | coral | pooled | ridge_prior | season_trend | source_only | target_mean | target_only | tca | tradaboost_r2 |
|---|---|---|---|---|---|---|---|---|---|
| 2018 | 0.134 | 0.016 | 0.320 | 0.000 | 1.266 | 0.773 | 0.163 | 0.247 | 0.052 |
| 2019 | 0.455 | 0.741 | 0.499 | 0.060 | 1.017 | 0.566 | 0.242 | 0.640 | 0.375 |
| 2020 | 0.343 | 0.467 | 0.272 | 0.200 | 1.188 | 0.449 | 0.084 | 0.341 | 0.082 |
| 2021 | 0.014 | 0.189 | 0.077 | 0.299 | 0.542 | 0.299 | 0.233 | 0.224 | 0.036 |
| 2022 | 0.108 | 0.011 | 0.241 | 0.016 | 0.609 | 0.243 | 0.066 | 0.203 | 0.201 |
| 2023 | 0.320 | 0.416 | 0.313 | 0.179 | 0.336 | 0.665 | 0.300 | 0.372 | 0.147 |
| 2024 | 0.189 | 0.230 | 0.298 | 0.306 | 0.422 | 0.582 | 0.162 | 0.100 | 0.011 |

- The target yield trends strongly with season (Spearman -0.93). Leave-one-season-out can interpolate between neighbouring seasons, and features that drift with time can look predictive. Compare with the season-trend baseline and with forward validation.
- Only 7 held-out seasons: every metric is unstable, and R2 is not reported below 10 points.
- Leave-one-season-out trains on later seasons too; neighbouring seasons can be correlated, so results may be optimistic.
- Baselines (source-only, target-only, target mean, pooled) and a season-trend baseline are always included, so every method is judged against them.

### Validation: forward (5 held-out seasons)
| method | n_test | MAE | RMSE | R2 | R2_note |
|---|---|---|---|---|---|
| season_trend | 4 | 0.255 | 0.304 |  | not reported: fewer than 10 held-out points |
| tradaboost_r2 | 5 | 0.271 | 0.393 |  | not reported: fewer than 10 held-out points |
| coral | 5 | 0.286 | 0.341 |  | not reported: fewer than 10 held-out points |
| tca | 5 | 0.307 | 0.341 |  | not reported: fewer than 10 held-out points |
| ridge_prior | 5 | 0.308 | 0.340 |  | not reported: fewer than 10 held-out points |
| target_only | 5 | 0.345 | 0.361 |  | not reported: fewer than 10 held-out points |
| pooled | 5 | 0.374 | 0.453 |  | not reported: fewer than 10 held-out points |
| target_mean | 5 | 0.570 | 0.609 |  | not reported: fewer than 10 held-out points |
| source_only | 5 | 0.619 | 0.688 |  | not reported: fewer than 10 held-out points |

Change in mean absolute error against each baseline (negative is better):
| method | baseline | n_seasons | mean_abs_err_change | relative_change_pct | ci95_low | ci95_high | better_in | verdict |
|---|---|---|---|---|---|---|---|---|
| coral | source_only | 5 | -0.333 | -53.806 | -0.510 | -0.114 | 4 | improved |
| coral | target_only | 5 | -0.059 | -17.110 | -0.247 | 0.129 | 3 | no clear difference |
| coral | target_mean | 5 | -0.284 | -49.800 | -0.518 | 0.075 | 4 | no clear difference |
| coral | season_trend | 4 | -0.046 | -17.910 |  |  | 3 | not testable (only 4 held-out seasons) |
| coral | pooled | 5 | -0.088 | -23.527 | -0.404 | 0.222 | 4 | no clear difference |
| pooled | source_only | 5 | -0.245 | -39.594 | -0.672 | 0.150 | 3 | no clear difference |
| pooled | target_only | 5 | 0.029 | 8.392 | -0.154 | 0.229 | 2 | no clear difference |
| pooled | target_mean | 5 | -0.196 | -34.355 | -0.336 | -0.055 | 4 | improved |
| pooled | season_trend | 4 | 0.176 | 68.981 |  |  | 2 | not testable (only 4 held-out seasons) |
| ridge_prior | source_only | 5 | -0.312 | -50.334 | -0.730 | -0.053 | 4 | improved |
| ridge_prior | target_only | 5 | -0.038 | -10.880 | -0.142 | 0.070 | 3 | no clear difference |
| ridge_prior | target_mean | 5 | -0.262 | -46.027 | -0.357 | -0.171 | 5 | improved |
| ridge_prior | season_trend | 4 | 0.119 | 46.494 |  |  | 2 | not testable (only 4 held-out seasons) |
| ridge_prior | pooled | 5 | -0.067 | -17.780 | -0.237 | 0.104 | 3 | no clear difference |
| season_trend | source_only | 4 | -0.222 | -46.495 |  |  | 4 | not testable (only 4 held-out seasons) |
| season_trend | target_only | 4 | -0.104 | -28.890 |  |  | 2 | not testable (only 4 held-out seasons) |
| season_trend | target_mean | 4 | -0.410 | -61.611 |  |  | 4 | not testable (only 4 held-out seasons) |
| season_trend | pooled | 4 | -0.176 | -40.822 |  |  | 2 | not testable (only 4 held-out seasons) |
| source_only | target_only | 5 | 0.274 | 79.438 | 0.029 | 0.597 | 1 | worse |
| source_only | target_mean | 5 | 0.049 | 8.672 | -0.301 | 0.526 | 3 | no clear difference |
| source_only | season_trend | 4 | 0.222 | 86.897 |  |  | 0 | not testable (only 4 held-out seasons) |
| source_only | pooled | 5 | 0.245 | 65.546 | -0.150 | 0.672 | 2 | no clear difference |
| target_mean | source_only | 5 | -0.049 | -7.980 | -0.526 | 0.301 | 2 | no clear difference |
| target_mean | target_only | 5 | 0.225 | 65.118 | 0.045 | 0.382 | 1 | worse |
| target_mean | season_trend | 4 | 0.410 | 160.491 |  |  | 0 | not testable (only 4 held-out seasons) |
| target_mean | pooled | 5 | 0.196 | 52.335 | 0.055 | 0.336 | 1 | worse |
| target_only | source_only | 5 | -0.274 | -44.270 | -0.597 | -0.029 | 4 | improved |
| target_only | target_mean | 5 | -0.225 | -39.437 | -0.382 | -0.045 | 4 | improved |
| target_only | season_trend | 4 | 0.104 | 40.628 |  |  | 2 | not testable (only 4 held-out seasons) |
| target_only | pooled | 5 | -0.029 | -7.742 | -0.229 | 0.154 | 3 | no clear difference |
| tca | source_only | 5 | -0.312 | -50.450 | -0.523 | -0.099 | 4 | improved |
| tca | target_only | 5 | -0.038 | -11.088 | -0.166 | 0.110 | 4 | no clear difference |
| tca | target_mean | 5 | -0.263 | -46.152 | -0.439 | 0.035 | 4 | no clear difference |
| tca | season_trend | 4 | -0.001 | -0.508 |  |  | 3 | not testable (only 4 held-out seasons) |
| tca | pooled | 5 | -0.067 | -17.971 | -0.284 | 0.168 | 3 | no clear difference |
| tradaboost_r2 | source_only | 5 | -0.349 | -56.286 | -0.743 | 0.041 | 3 | no clear difference |
| tradaboost_r2 | target_only | 5 | -0.074 | -21.560 | -0.251 | 0.139 | 4 | no clear difference |
| tradaboost_r2 | target_mean | 5 | -0.299 | -52.495 | -0.492 | -0.106 | 4 | improved |
| tradaboost_r2 | season_trend | 4 | 0.055 | 21.639 |  |  | 2 | not testable (only 4 held-out seasons) |
| tradaboost_r2 | pooled | 5 | -0.103 | -27.633 | -0.172 | -0.035 | 5 | improved |

Absolute error by season:
| group | coral | pooled | ridge_prior | season_trend | source_only | target_mean | target_only | tca | tradaboost_r2 |
|---|---|---|---|---|---|---|---|---|---|
| 2020 | 0.592 | 0.145 | 0.042 |  | 1.188 | 0.189 | 0.289 | 0.518 | 0.111 |
| 2021 | 0.166 | 0.783 | 0.467 | 0.489 | 0.542 | 0.767 | 0.440 | 0.364 | 0.772 |
| 2022 | 0.084 | 0.148 | 0.391 | 0.187 | 0.609 | 0.527 | 0.421 | 0.179 | 0.060 |
| 2023 | 0.399 | 0.566 | 0.341 | 0.039 | 0.336 | 0.784 | 0.413 | 0.373 | 0.399 |
| 2024 | 0.189 | 0.230 | 0.298 | 0.306 | 0.422 | 0.582 | 0.162 | 0.100 | 0.011 |

- The target yield trends strongly with season (Spearman -0.93). Leave-one-season-out can interpolate between neighbouring seasons, and features that drift with time can look predictive. Compare with the season-trend baseline and with forward validation.
- Only 5 held-out seasons: every metric is unstable, and R2 is not reported below 10 points.

## Diagnostics
### CORAL feature alignment versus Target-only model
coral was worse than target_only on average (+0.045, 2 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample, Inappropriate model assumption. Possible explanations (not proven): Large source-target distribution difference, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 3/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.67; same sign for 75% of features | Not supported | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | - |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2020 accounts for 46% of the extra error; per-season change range -0.219 to +0.259 | Not supported | With few seasons one unusual season dominates any average. | - |
| Overfitting | mean absolute error on the target training rows = 0.152 vs 0.223 on held-out seasons (ratio 0.68) | Not supported | A gap is normal for any model; a large one signals memorising few rows. | - |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.068, +0.146] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.93; MAE of this method 0.223 vs the pure season-trend baseline 0.151 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | covariance estimated from 6.0 target rows for 8 features | Verified finding | The regulariser keeps it defined but pulls the alignment toward the identity. | Use fewer features or stronger regularisation. |

### CORAL feature alignment versus Season trend (no features)
coral was worse than season_trend on average (+0.072, 2 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample, Inappropriate model assumption. Possible explanations (not proven): Large source-target distribution difference, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 3/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.67; same sign for 75% of features | Not supported | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | - |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2019 accounts for 44% of the extra error; per-season change range -0.284 to +0.395 | Not supported | With few seasons one unusual season dominates any average. | - |
| Overfitting | mean absolute error on the target training rows = 0.152 vs 0.223 on held-out seasons (ratio 0.68) | Not supported | A gap is normal for any model; a large one signals memorising few rows. | - |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.084, +0.214] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.93; MAE of this method 0.223 vs the pure season-trend baseline 0.151 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | covariance estimated from 6.0 target rows for 8 features | Verified finding | The regulariser keeps it defined but pulls the alignment toward the identity. | Use fewer features or stronger regularisation. |

### Transfer Component Analysis versus Target-only model
tca was worse than target_only on average (+0.125, 2 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 3/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.67; same sign for 75% of features | Not supported | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | - |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2019 accounts for 42% of the extra error; per-season change range -0.062 to +0.397 | Not supported | With few seasons one unusual season dominates any average. | - |
| Overfitting | mean absolute error on the target training rows = 0.204 vs 0.304 on held-out seasons (ratio 0.67) | Not supported | A gap is normal for any model; a large one signals memorising few rows. | - |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [+0.021, +0.243] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.93; MAE of this method 0.304 vs the pure season-trend baseline 0.151 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | no specific assumption was violated that the data can check | Not applicable | - | - |

### Transfer Component Analysis versus Season trend (no features)
tca was worse than season_trend on average (+0.152, 2 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 3/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.67; same sign for 75% of features | Not supported | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | - |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2019 accounts for 43% of the extra error; per-season change range -0.206 to +0.580 | Not supported | With few seasons one unusual season dominates any average. | - |
| Overfitting | mean absolute error on the target training rows = 0.204 vs 0.304 on held-out seasons (ratio 0.67) | Not supported | A gap is normal for any model; a large one signals memorising few rows. | - |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.009, +0.329] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.93; MAE of this method 0.304 vs the pure season-trend baseline 0.151 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | no specific assumption was violated that the data can check | Not applicable | - | - |

### TrAdaBoost.R2 versus Target-only model
tradaboost_r2 was better than target_only on average (-0.050, 5 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, Overfitting, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 3/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | effective sample size = 1.00 of the source rows; source weight share 0.00 | Not supported | The weights show where the method put its trust, not whether that trust was right. | The weights are spread, so concentration is not the problem. |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.67; same sign for 75% of features | Not supported | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | - |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2022 accounts for 50% of the extra error; per-season change range -0.197 to +0.135 | Not supported | With few seasons one unusual season dominates any average. | Check season 2022 (weather, planting date, data quality). |
| Overfitting | mean absolute error on the target training rows = 0.030 vs 0.129 on held-out seasons (ratio 0.24) | Possible explanation | A gap is normal for any model; a large one signals memorising few rows. | Increase regularisation or use fewer features. |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.141, +0.052] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.93; MAE of this method 0.129 vs the pure season-trend baseline 0.151 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | needs about 3+ target rows to judge source rows; has 6.0 per fold | Not supported | The rule is a minimum for the algorithm to run, not a guarantee of stability. | - |

### TrAdaBoost.R2 versus Season trend (no features)
tradaboost_r2 was better than season_trend on average (-0.022, 4 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, Overfitting, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 3/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | effective sample size = 1.00 of the source rows; source weight share 0.00 | Not supported | The weights show where the method put its trust, not whether that trust was right. | The weights are spread, so concentration is not the problem. |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.67; same sign for 75% of features | Not supported | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | - |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2019 accounts for 57% of the extra error; per-season change range -0.295 to +0.315 | Not supported | With few seasons one unusual season dominates any average. | Check season 2019 (weather, planting date, data quality). |
| Overfitting | mean absolute error on the target training rows = 0.030 vs 0.129 on held-out seasons (ratio 0.24) | Possible explanation | A gap is normal for any model; a large one signals memorising few rows. | Increase regularisation or use fewer features. |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.175, +0.139] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.93; MAE of this method 0.129 vs the pure season-trend baseline 0.151 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | needs about 3+ target rows to judge source rows; has 6.0 per fold | Not supported | The rule is a minimum for the algorithm to run, not a guarantee of stability. | - |

### Parameter transfer (source Ridge + regularised target correction) versus Target-only model
ridge_prior was worse than target_only on average (+0.110, 1 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, Overfitting, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 3/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.67; same sign for 75% of features | Not supported | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | - |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2019 accounts for 28% of the extra error; per-season change range -0.156 to +0.256 | Not supported | With few seasons one unusual season dominates any average. | - |
| Overfitting | mean absolute error on the target training rows = 0.084 vs 0.289 on held-out seasons (ratio 0.29) | Possible explanation | A gap is normal for any model; a large one signals memorising few rows. | Increase regularisation or use fewer features. |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [+0.004, +0.192] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.93; MAE of this method 0.289 vs the pure season-trend baseline 0.151 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | no specific assumption was violated that the data can check | Not applicable | - | - |

### Parameter transfer (source Ridge + regularised target correction) versus Season trend (no features)
ridge_prior was worse than season_trend on average (+0.137, 2 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, Overfitting, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 3/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.67; same sign for 75% of features | Not supported | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | - |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2019 accounts for 37% of the extra error; per-season change range -0.222 to +0.439 | Not supported | With few seasons one unusual season dominates any average. | - |
| Overfitting | mean absolute error on the target training rows = 0.084 vs 0.289 on held-out seasons (ratio 0.29) | Possible explanation | A gap is normal for any model; a large one signals memorising few rows. | Increase regularisation or use fewer features. |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.018, +0.289] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.93; MAE of this method 0.289 vs the pure season-trend baseline 0.151 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | no specific assumption was violated that the data can check | Not applicable | - | - |

## Settings
```
{
 "min_obs_block": 3,
 "missing_warn_pct": 10.0,
 "missing_block_pct": 40.0,
 "outlier_iqr_k": 3.0,
 "quality_min_pct": 20.0,
 "min_obs_per_feature": 3.0,
 "alpha": 0.05,
 "smd_small": 0.2,
 "smd_medium": 0.5,
 "smd_large": 0.8,
 "mmd_permutations": 500,
 "random_state": 0,
 "cv_strategy": "loso",
 "max_target_train_groups": null,
 "r2_min_test_n": 10,
 "bootstrap_n": 2000,
 "min_target_tradaboost": 3,
 "far_auc": 0.85,
 "auc_min_rows": 30,
 "trend_flag_rho": 0.7,
 "kmm_max_source": 600,
 "tca_max_rows": 800,
 "coral_eps": 1.0,
 "valid_ranges": {
  "NDVI": [
   -1.0,
   1.0
  ],
  "NDVI_amplitude": [
   0.0,
   2.0
  ],
  "EVI": [
   -1.0,
   3.0
  ],
  "SAR_VV_dB": [
   -35.0,
   5.0
  ],
  "Mean_Temp_C": [
   10.0,
   40.0
  ],
  "Annual_Rain_mm": [
   0.0,
   8000.0
  ],
  "Rain_decade_mm": [
   0.0,
   1500.0
  ],
  "Solar_Radiation_MJ": [
   0.0,
   35.0
  ],
  "Optical_observed_pct": [
   0.0,
   100.0
  ],
  "Yield_t_ha": [
   0.0,
   15.0
  ],
  "Yield_kg_ha": [
   0.0,
   15000.0
  ]
 },
 "proposed_features": [
  "humidity",
  "land surface temperature",
  "irrigation / water supply",
  "planting date",
  "variety",
  "fertiliser"
 ]
}
```
Base model: ridge

## Event log
- 02:13:17: selected source 'IADA Barat Laut Selangor' and target 'MADA', base model ridge
- 02:13:20: assessment: readiness conditional
- 02:13:20: harmonisation: validated
- 02:13:20: suitability: conditionally_recommended
- 02:13:20: strategy selected: ['coral', 'tca', 'tradaboost_r2', 'ridge_prior']
- 02:13:24: ran ['source_only', 'target_only', 'target_mean', 'pooled', 'coral', 'tca', 'tradaboost_r2', 'ridge_prior']