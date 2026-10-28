# Transfer-learning assessment and evaluation: IADA Barat Laut Selangor to KETARA

## Workflow status
| step | status | detail |
|---|---|---|
| 1 · Select domains | complete |  |
| 2 · Assess data | complete |  |
| 3 · Harmonise data | complete |  |
| 4 · Evaluate suitability | complete | transfer is NOT RECOMMENDED |
| 5 · Select strategy | complete |  |
| 6 · Run transfer | complete |  |
| 7 · Compare results | complete |  |

## Pre-transfer assessment: IADA Barat Laut Selangor to KETARA

Computed 2026-10-26 02:13:28; fingerprint `a8240a2e8b1533c6`.

**Data readiness: CONDITIONAL**

## Readiness reasons
- source: Optical_observed_pct below 20% in Year [2018]: those rows rest mostly on gap-filled inputs
- target: Optical_observed_pct below 20% in Year [2018]: those rows rest mostly on gap-filled inputs
- 7 labelled target rows for 8 features (the configured rule of thumb is 3 rows per feature)
- CORAL at the modelling level is not interpretable: the covariance of 8 features is estimated from 7 rows (fewer rows than features).

## Source data: IADA Barat Laut Selangor
- 7 observations, 7 groups ([2018, 2019, 2020, 2021, 2022, 2023, 2024]), 7 labelled, 8 features
- [warn] Optical_observed_pct below 20% in Year [2018]: those rows rest mostly on gap-filled inputs

## Target data: KETARA
- 7 observations, 7 groups ([2018, 2019, 2020, 2021, 2022, 2023, 2024]), 7 labelled, 8 features
- [info] NDVI_mean: 1 extreme outlier(s) beyond 3.0 x IQR (kept; check them)
- [info] NDVI_max: 1 extreme outlier(s) beyond 3.0 x IQR (kept; check them)
- [info] NDVI_amplitude: 1 extreme outlier(s) beyond 3.0 x IQR (kept; check them)
- [info] EVI_mean: 1 extreme outlier(s) beyond 3.0 x IQR (kept; check them)
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
| Annual_Rain_mm | 7 | 7 | 2507.982 | 3570.159 | 500.131 | 689.770 | 1.763 | 0.714 | 0.053 | 1062.177 | 1.763 | 0.071 | large | False |
| EVI_mean | 7 | 7 | 0.314 | 0.321 | 0.029 | 0.037 | 0.191 | 0.286 | 0.963 | 0.010 | 0.292 | 0.963 | negligible | False |
| Mean_Temp_C_mean | 7 | 7 | 26.857 | 25.144 | 0.201 | 0.277 | -7.072 | 1.000 | 0.001 | 1.713 | 7.072 | 0.005 | large | True |
| NDVI_amplitude | 7 | 7 | 0.239 | 0.449 | 0.111 | 0.155 | 1.561 | 0.857 | 0.008 | 0.211 | 1.561 | 0.016 | large | True |
| NDVI_max | 7 | 7 | 0.565 | 0.687 | 0.061 | 0.130 | 1.205 | 0.857 | 0.008 | 0.142 | 1.396 | 0.016 | large | True |
| NDVI_mean | 7 | 7 | 0.444 | 0.471 | 0.012 | 0.045 | 0.814 | 0.857 | 0.008 | 0.041 | 1.250 | 0.016 | large | True |
| SAR_VV_dB_mean | 7 | 7 | -9.634 | -9.974 | 0.164 | 0.247 | -1.623 | 0.714 | 0.053 | 0.340 | 1.623 | 0.071 | large | False |
| Solar_Radiation_MJ_mean | 7 | 7 | 18.510 | 18.381 | 0.637 | 0.647 | -0.200 | 0.286 | 0.963 | 0.162 | 0.252 | 0.963 | small | False |
MMD^2 = 0.3972 (permutation p = 0.002; the smallest possible p is 1/(permutations+1)); CORAL = 0.1089 (p = 0.315; NOT interpretable: fewer rows than features)

## Domain differences: Ten-day level (finer observations)
| feature | n_source | n_target | mean_source | mean_target | sd_source | sd_target | SMD | KS | KS_p | Wasserstein | Wasserstein_per_SD | KS_p_adj | SMD_size | KS_significant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NDVI | 221 | 219 | 0.443 | 0.484 | 0.072 | 0.163 | 0.328 | 0.329 | 0.000 | 0.089 | 0.704 | 0.000 | small | True |
| EVI | 221 | 219 | 0.323 | 0.332 | 0.058 | 0.134 | 0.085 | 0.297 | 0.000 | 0.072 | 0.694 | 0.000 | negligible | True |
| SAR_VV_dB | 205 | 205 | -9.633 | -9.977 | 0.681 | 1.397 | -0.313 | 0.224 | 0.000 | 0.570 | 0.518 | 0.000 | small | True |
| Mean_Temp_C | 252 | 252 | 26.857 | 25.144 | 0.689 | 0.991 | -2.008 | 0.714 | 0.000 | 1.713 | 2.008 | 0.000 | large | True |
| Rain_decade_mm | 252 | 252 | 69.666 | 99.171 | 45.780 | 75.813 | 0.471 | 0.262 | 0.000 | 29.674 | 0.474 | 0.000 | small | True |
| Solar_Radiation_MJ | 252 | 252 | 18.510 | 18.381 | 2.366 | 3.299 | -0.045 | 0.099 | 0.168 | 0.730 | 0.254 | 0.168 | negligible | False |
MMD^2 = 0.1965 (permutation p = 0.002; the smallest possible p is 1/(permutations+1)); CORAL = 0.1211 (p = 0.002)

## Domain classifier and crop calendar (ten-day level)
- Climate AUC 0.97
- Vegetation and radar AUC 0.75
- Calendar offset 60 days (curve correlation -0.25 unshifted, +0.85 aligned); informational

## Label (yield) shift
| n_source | n_target | mean_source | mean_target | SMD | KS | KS_p | Wasserstein |
|---|---|---|---|---|---|---|---|
| 7 | 7 | 4476.571 | 4784.143 | 0.637 | 0.571 | 0.212 | 467.857 |

## Negative-transfer risks
- Too few labelled target rows for a flexible model (7 rows, 8 features): estimates will be unstable and the evaluation noisy.
- 4 of 8 features differ in distribution at the modelling level (adjusted KS p < 0.05): ['Mean_Temp_C_mean', 'NDVI_amplitude', 'NDVI_max', 'NDVI_mean'].
- 6 feature(s) have a large standardised mean difference (|SMD| >= 0.8) at the modelling level.
- The joint feature distributions differ at the modelling level (MMD permutation p = 0.002).
- 5 of 6 features differ in distribution at the ten-day level (adjusted KS p < 0.05): ['NDVI', 'EVI', 'SAR_VV_dB', 'Mean_Temp_C', 'Rain_decade_mm'].
- 1 feature(s) have a large standardised mean difference (|SMD| >= 0.8) at the ten-day level.
- The joint feature distributions differ at the ten-day level (MMD permutation p = 0.002).
- The feature covariance differs at the ten-day level (CORAL permutation p = 0.002).
- FAR DOMAIN: a classifier tells the two climates apart with AUC = 0.97 (>= 0.85; 0.5 is chance). Transfer between very different climates often hurts.

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
Fingerprint `02a615fb9a357380`; originals are unchanged.

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
**NOT RECOMMENDED**

- the climates are very different (domain-classifier AUC 0.97 >= 0.85)

### Scenarios A to F
**A. Similar source and target distributions**: not met
- Observed: A difference was detected, so this scenario does not apply.
- Evidence: modelling-level MMD p = 0.002; ten-day-level MMD p = 0.002; 4/8 features differ (adjusted KS); 6 with |SMD| >= 0.8
- Recommended: Source-only model, Pooled (naive), Target-only model; runnable here: source_only, pooled, target_only
- Why: If the domains look alike, transfer may be unnecessary: compare source-only and pooled models with a target-only model before adding complexity.
- Assumptions: Absence of a detected difference is not proof of similarity when there are few rows.
- Benefits and risks: Simplest option; risk is hidden differences that low power cannot detect.
- Alternatives: TrAdaBoost.R2 or KMM as a check

**B. Moderate distribution shift with compatible features**: not met
- Observed: Not met (covariance or label shift is present, or no shift was found).
- Evidence: modelling-level MMD p = 0.002; ten-day-level MMD p = 0.002; 4/8 features differ (adjusted KS); 6 with |SMD| >= 0.8; modelling-level covariance not interpretable (fewer rows than features); ten-day-level CORAL p = 0.002
- Recommended: Kernel Mean Matching + weighted model, TrAdaBoost.R2; runnable here: kmm, tradaboost_r2
- Why: Instance weighting: KMM weights source rows so their feature mean matches the target (no labels needed); TrAdaBoost.R2 lowers the weight of source rows that the model fits badly on target rows (needs labelled target rows).
- Assumptions: The feature-yield relationship is shared (KMM); enough target labels to judge source rows (TrAdaBoost).
- Benefits and risks: Keeps useful source rows and drops misleading ones; unstable weights with few rows.
- Alternatives: CORAL, TCA

**C. Significant feature-distribution differences**: CONDITIONS MET
- Observed: Several features, or the feature covariance, differ clearly.
- Evidence: modelling-level MMD p = 0.002; ten-day-level MMD p = 0.002; 4/8 features differ (adjusted KS); 6 with |SMD| >= 0.8; modelling-level covariance not interpretable (fewer rows than features); ten-day-level CORAL p = 0.002
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
- Evidence: modelling-level MMD p = 0.002; ten-day-level MMD p = 0.002; 4/8 features differ (adjusted KS); 6 with |SMD| >= 0.8
- Recommended: TrAdaBoost.R2, Parameter transfer (source Ridge + regularised target correction), CORAL feature alignment; runnable here: tradaboost_r2, ridge_prior, coral
- Why: Adaptation type follows the labels actually available: supervised.
- Assumptions: Adaptation helps only if the shift is in the features and not in the feature-yield relationship.
- Benefits and risks: May reduce the source-target gap; a feature gap does not by itself mean a worse prediction.
- Alternatives: source-only and target-only baselines

**F. Severe mismatch or negative-transfer risk**: CONDITIONS MET
- Observed: the climates are very different (domain-classifier AUC 0.97 >= 0.85)
- Evidence: modelling-level MMD p = 0.002; ten-day-level MMD p = 0.002; 4/8 features differ (adjusted KS); 6 with |SMD| >= 0.8; label KS p = 0.212
- Recommended: Target-only model; runnable here: target_only
- Why: Do not force transfer: collect more target data, improve harmonisation, try a different source, or keep a target-only baseline.
- Assumptions: -
- Benefits and risks: Avoids a model that is worse than not transferring.
- Alternatives: a different source domain

### Method table
| method | family | eligible | reason | recommended_by | features_compared | scaling | metric | distance_influence | limitations |
|---|---|---|---|---|---|---|---|---|---|
| Source-only model | baseline | True | needs only source labels | - | none | source mean and SD | none | none | assumes the target shares the source's feature distribution and feature-yield relationship |
| Target-only model | baseline | True | 6 labelled target rows (a weak baseline when this is small) | F | none | target training-row mean and SD | none | none | unstable when the target has few labelled rows |
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

**Override of a NOT RECOMMENDED verdict. Reason recorded:** documented example run

## Transfer performance comparison (yield in the harmonised unit, held-out seasons only)
### Validation: loso (7 held-out seasons)
| method | n_test | MAE | RMSE | R2 | R2_note |
|---|---|---|---|---|---|
| season_trend | 7 | 0.338 | 0.374 |  | not reported: fewer than 10 held-out points |
| ridge_prior | 7 | 0.351 | 0.441 |  | not reported: fewer than 10 held-out points |
| coral | 7 | 0.412 | 0.519 |  | not reported: fewer than 10 held-out points |
| tca | 7 | 0.460 | 0.524 |  | not reported: fewer than 10 held-out points |
| target_only | 7 | 0.501 | 0.617 |  | not reported: fewer than 10 held-out points |
| tradaboost_r2 | 7 | 0.508 | 0.542 |  | not reported: fewer than 10 held-out points |
| pooled | 7 | 0.509 | 0.577 |  | not reported: fewer than 10 held-out points |
| source_only | 7 | 0.571 | 0.736 |  | not reported: fewer than 10 held-out points |
| target_mean | 7 | 0.666 | 0.699 |  | not reported: fewer than 10 held-out points |

Change in mean absolute error against each baseline (negative is better):
| method | baseline | n_seasons | mean_abs_err_change | relative_change_pct | ci95_low | ci95_high | better_in | verdict |
|---|---|---|---|---|---|---|---|---|
| coral | source_only | 7 | -0.160 | -27.921 | -0.447 | 0.149 | 5 | no clear difference |
| coral | target_only | 7 | -0.089 | -17.849 | -0.574 | 0.353 | 3 | no clear difference |
| coral | target_mean | 7 | -0.255 | -38.211 | -0.469 | -0.017 | 5 | improved |
| coral | season_trend | 7 | 0.074 | 21.795 | -0.137 | 0.286 | 4 | no clear difference |
| coral | pooled | 7 | -0.097 | -19.080 | -0.223 | 0.027 | 5 | no clear difference |
| pooled | source_only | 7 | -0.062 | -10.926 | -0.429 | 0.324 | 4 | no clear difference |
| pooled | target_only | 7 | 0.008 | 1.521 | -0.394 | 0.398 | 3 | no clear difference |
| pooled | target_mean | 7 | -0.158 | -23.642 | -0.285 | -0.026 | 6 | improved |
| pooled | season_trend | 7 | 0.171 | 50.513 | 0.005 | 0.353 | 1 | worse |
| ridge_prior | source_only | 7 | -0.221 | -38.646 | -0.421 | -0.001 | 6 | improved |
| ridge_prior | target_only | 7 | -0.151 | -30.072 | -0.597 | 0.210 | 5 | no clear difference |
| ridge_prior | target_mean | 7 | -0.316 | -47.404 | -0.574 | -0.040 | 5 | improved |
| ridge_prior | season_trend | 7 | 0.012 | 3.673 | -0.146 | 0.155 | 4 | no clear difference |
| ridge_prior | pooled | 7 | -0.158 | -31.120 | -0.413 | 0.095 | 5 | no clear difference |
| season_trend | source_only | 7 | -0.233 | -40.820 | -0.543 | 0.079 | 4 | no clear difference |
| season_trend | target_only | 7 | -0.163 | -32.550 | -0.459 | 0.093 | 3 | no clear difference |
| season_trend | target_mean | 7 | -0.328 | -49.268 | -0.489 | -0.174 | 7 | improved |
| season_trend | pooled | 7 | -0.171 | -33.560 | -0.353 | -0.005 | 6 | improved |
| source_only | target_only | 7 | 0.070 | 13.975 | -0.403 | 0.577 | 4 | no clear difference |
| source_only | target_mean | 7 | -0.095 | -14.275 | -0.523 | 0.327 | 4 | no clear difference |
| source_only | season_trend | 7 | 0.233 | 68.975 | -0.079 | 0.543 | 3 | no clear difference |
| source_only | pooled | 7 | 0.062 | 12.266 | -0.324 | 0.429 | 3 | no clear difference |
| target_mean | source_only | 7 | 0.095 | 16.652 | -0.327 | 0.523 | 3 | no clear difference |
| target_mean | target_only | 7 | 0.165 | 32.954 | -0.182 | 0.482 | 3 | no clear difference |
| target_mean | season_trend | 7 | 0.328 | 97.113 | 0.174 | 0.489 | 0 | worse |
| target_mean | pooled | 7 | 0.158 | 30.961 | 0.026 | 0.285 | 1 | worse |
| target_only | source_only | 7 | -0.070 | -12.261 | -0.577 | 0.403 | 3 | no clear difference |
| target_only | target_mean | 7 | -0.165 | -24.786 | -0.482 | 0.182 | 4 | no clear difference |
| target_only | season_trend | 7 | 0.163 | 48.257 | -0.093 | 0.459 | 4 | no clear difference |
| target_only | pooled | 7 | -0.008 | -1.499 | -0.398 | 0.394 | 4 | no clear difference |
| tca | source_only | 7 | -0.111 | -19.500 | -0.372 | 0.154 | 5 | no clear difference |
| tca | target_only | 7 | -0.041 | -8.250 | -0.489 | 0.358 | 4 | no clear difference |
| tca | target_mean | 7 | -0.207 | -30.991 | -0.423 | 0.018 | 5 | no clear difference |
| tca | season_trend | 7 | 0.122 | 36.025 | -0.055 | 0.293 | 3 | no clear difference |
| tca | pooled | 7 | -0.049 | -9.625 | -0.192 | 0.084 | 3 | no clear difference |
| tradaboost_r2 | source_only | 7 | -0.064 | -11.128 | -0.469 | 0.326 | 3 | no clear difference |
| tradaboost_r2 | target_only | 7 | 0.006 | 1.292 | -0.345 | 0.328 | 3 | no clear difference |
| tradaboost_r2 | target_mean | 7 | -0.159 | -23.814 | -0.368 | 0.009 | 5 | no clear difference |
| tradaboost_r2 | season_trend | 7 | 0.170 | 50.172 | -0.042 | 0.413 | 2 | no clear difference |
| tradaboost_r2 | pooled | 7 | -0.001 | -0.226 | -0.221 | 0.237 | 3 | no clear difference |

Absolute error by season:
| group | coral | pooled | ridge_prior | season_trend | source_only | target_mean | target_only | tca | tradaboost_r2 |
|---|---|---|---|---|---|---|---|---|---|
| 2018 | 0.104 | 0.447 | 0.120 | 0.430 | 0.370 | 0.659 | 1.281 | 0.238 | 0.496 |
| 2019 | 0.009 | 0.053 | 0.467 | 0.213 | 0.608 | 0.441 | 0.493 | 0.127 | 0.566 |
| 2020 | 0.781 | 0.711 | 0.651 | 0.436 | 1.322 | 0.727 | 0.329 | 0.779 | 0.474 |
| 2021 | 0.810 | 0.646 | 0.743 | 0.506 | 1.184 | 0.506 | 0.230 | 0.826 | 0.329 |
| 2022 | 0.090 | 0.172 | 0.072 | 0.133 | 0.343 | 0.436 | 0.654 | 0.281 | 0.401 |
| 2023 | 0.483 | 0.755 | 0.384 | 0.518 | 0.083 | 1.050 | 0.427 | 0.411 | 0.355 |
| 2024 | 0.604 | 0.779 | 0.016 | 0.131 | 0.090 | 0.846 | 0.095 | 0.557 | 0.933 |

- The target yield trends strongly with season (Spearman -0.79). Leave-one-season-out can interpolate between neighbouring seasons, and features that drift with time can look predictive. Compare with the season-trend baseline and with forward validation.
- Only 7 held-out seasons: every metric is unstable, and R2 is not reported below 10 points.
- Leave-one-season-out trains on later seasons too; neighbouring seasons can be correlated, so results may be optimistic.
- Baselines (source-only, target-only, target mean, pooled) and a season-trend baseline are always included, so every method is judged against them.
- Run with an explicit override of a NOT RECOMMENDED verdict. Reason given: documented example run

### Validation: forward (5 held-out seasons)
| method | n_test | MAE | RMSE | R2 | R2_note |
|---|---|---|---|---|---|
| target_only | 5 | 0.308 | 0.405 |  | not reported: fewer than 10 held-out points |
| season_trend | 4 | 0.448 | 0.548 |  | not reported: fewer than 10 held-out points |
| ridge_prior | 5 | 0.572 | 0.673 |  | not reported: fewer than 10 held-out points |
| source_only | 5 | 0.604 | 0.810 |  | not reported: fewer than 10 held-out points |
| pooled | 5 | 0.610 | 0.679 |  | not reported: fewer than 10 held-out points |
| tca | 5 | 0.616 | 0.724 |  | not reported: fewer than 10 held-out points |
| target_mean | 5 | 0.637 | 0.776 |  | not reported: fewer than 10 held-out points |
| tradaboost_r2 | 5 | 0.644 | 0.711 |  | not reported: fewer than 10 held-out points |
| coral | 5 | 0.676 | 0.785 |  | not reported: fewer than 10 held-out points |

Change in mean absolute error against each baseline (negative is better):
| method | baseline | n_seasons | mean_abs_err_change | relative_change_pct | ci95_low | ci95_high | better_in | verdict |
|---|---|---|---|---|---|---|---|---|
| coral | source_only | 5 | 0.071 | 11.776 | -0.311 | 0.439 | 2 | no clear difference |
| coral | target_only | 5 | 0.368 | 119.646 | -0.037 | 0.867 | 2 | no clear difference |
| coral | target_mean | 5 | 0.039 | 6.066 | -0.558 | 0.725 | 3 | no clear difference |
| coral | season_trend | 4 | 0.049 | 11.022 |  |  | 2 | not testable (only 4 held-out seasons) |
| coral | pooled | 5 | 0.065 | 10.695 | -0.326 | 0.489 | 3 | no clear difference |
| pooled | source_only | 5 | 0.006 | 0.977 | -0.720 | 0.686 | 2 | no clear difference |
| pooled | target_only | 5 | 0.303 | 98.424 | 0.023 | 0.554 | 1 | worse |
| pooled | target_mean | 5 | -0.027 | -4.182 | -0.232 | 0.237 | 3 | no clear difference |
| pooled | season_trend | 4 | 0.162 | 36.253 |  |  | 2 | not testable (only 4 held-out seasons) |
| ridge_prior | source_only | 5 | -0.032 | -5.316 | -0.651 | 0.519 | 3 | no clear difference |
| ridge_prior | target_only | 5 | 0.265 | 86.059 | -0.008 | 0.538 | 2 | no clear difference |
| ridge_prior | target_mean | 5 | -0.065 | -10.153 | -0.510 | 0.339 | 3 | no clear difference |
| ridge_prior | season_trend | 4 | 0.072 | 15.993 |  |  | 2 | not testable (only 4 held-out seasons) |
| ridge_prior | pooled | 5 | -0.038 | -6.232 | -0.416 | 0.220 | 2 | no clear difference |
| season_trend | source_only | 4 | 0.023 | 5.429 |  |  | 1 | not testable (only 4 held-out seasons) |
| season_trend | target_only | 4 | 0.085 | 23.545 |  |  | 2 | not testable (only 4 held-out seasons) |
| season_trend | target_mean | 4 | -0.310 | -40.901 |  |  | 3 | not testable (only 4 held-out seasons) |
| season_trend | pooled | 4 | -0.162 | -26.607 |  |  | 2 | not testable (only 4 held-out seasons) |
| source_only | target_only | 5 | 0.297 | 96.505 | -0.269 | 0.913 | 2 | no clear difference |
| source_only | target_mean | 5 | -0.033 | -5.109 | -0.866 | 0.800 | 3 | no clear difference |
| source_only | season_trend | 4 | -0.023 | -5.149 |  |  | 3 | not testable (only 4 held-out seasons) |
| source_only | pooled | 5 | -0.006 | -0.967 | -0.686 | 0.720 | 3 | no clear difference |
| target_mean | source_only | 5 | 0.033 | 5.384 | -0.800 | 0.866 | 2 | no clear difference |
| target_mean | target_only | 5 | 0.329 | 107.085 | 0.004 | 0.640 | 1 | worse |
| target_mean | season_trend | 4 | 0.310 | 69.207 |  |  | 1 | not testable (only 4 held-out seasons) |
| target_mean | pooled | 5 | 0.027 | 4.365 | -0.237 | 0.232 | 2 | no clear difference |
| target_only | source_only | 5 | -0.297 | -49.111 | -0.913 | 0.269 | 3 | no clear difference |
| target_only | target_mean | 5 | -0.329 | -51.711 | -0.640 | -0.004 | 4 | improved |
| target_only | season_trend | 4 | -0.085 | -19.058 |  |  | 2 | not testable (only 4 held-out seasons) |
| target_only | pooled | 5 | -0.303 | -49.603 | -0.554 | -0.023 | 4 | improved |
| tca | source_only | 5 | 0.011 | 1.890 | -0.352 | 0.342 | 2 | no clear difference |
| tca | target_only | 5 | 0.308 | 100.220 | -0.119 | 0.831 | 2 | no clear difference |
| tca | target_mean | 5 | -0.021 | -3.315 | -0.631 | 0.664 | 3 | no clear difference |
| tca | season_trend | 4 | -0.013 | -2.850 |  |  | 2 | not testable (only 4 held-out seasons) |
| tca | pooled | 5 | 0.006 | 0.905 | -0.398 | 0.452 | 3 | no clear difference |
| tradaboost_r2 | source_only | 5 | 0.040 | 6.556 | -0.660 | 0.737 | 2 | no clear difference |
| tradaboost_r2 | target_only | 5 | 0.336 | 109.387 | 0.011 | 0.661 | 2 | worse |
| tradaboost_r2 | target_mean | 5 | 0.007 | 1.112 | -0.219 | 0.150 | 1 | no clear difference |
| tradaboost_r2 | season_trend | 4 | 0.271 | 60.438 |  |  | 0 | not testable (only 4 held-out seasons) |
| tradaboost_r2 | pooled | 5 | 0.034 | 5.525 | -0.169 | 0.236 | 2 | no clear difference |

Absolute error by season:
| group | coral | pooled | ridge_prior | season_trend | source_only | target_mean | target_only | tca | tradaboost_r2 |
|---|---|---|---|---|---|---|---|---|---|
| 2020 | 1.388 | 0.609 | 0.782 |  | 1.322 | 0.152 | 0.087 | 1.338 | 0.344 |
| 2021 | 0.561 | 0.112 | 0.304 | 0.146 | 1.184 | 0.088 | 0.317 | 0.518 | 0.223 |
| 2022 | 0.156 | 0.540 | 0.811 | 0.837 | 0.343 | 0.874 | 0.234 | 0.204 | 0.921 |
| 2023 | 0.670 | 1.012 | 0.948 | 0.679 | 0.083 | 1.225 | 0.806 | 0.463 | 0.799 |
| 2024 | 0.604 | 0.779 | 0.016 | 0.131 | 0.090 | 0.846 | 0.095 | 0.557 | 0.933 |

- The target yield trends strongly with season (Spearman -0.79). Leave-one-season-out can interpolate between neighbouring seasons, and features that drift with time can look predictive. Compare with the season-trend baseline and with forward validation.
- Only 5 held-out seasons: every metric is unstable, and R2 is not reported below 10 points.

## Diagnostics
### CORAL feature alignment versus Target-only model
coral was better than target_only on average (-0.089, 3 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample, Inappropriate model assumption. Possible explanations (not proven): Large source-target distribution difference, Different feature-yield relationship, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 4/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.09; same sign for 50% of features | Possible explanation | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | Prefer parameter-based correction or target-only learning if the relationships differ. |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2021 accounts for 36% of the extra error; per-season change range -1.178 to +0.581 | Not supported | With few seasons one unusual season dominates any average. | - |
| Overfitting | mean absolute error on the target training rows = 0.314 vs 0.412 on held-out seasons (ratio 0.76) | Not supported | A gap is normal for any model; a large one signals memorising few rows. | - |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.574, +0.353] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.79; MAE of this method 0.412 vs the pure season-trend baseline 0.338 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | covariance estimated from 6.0 target rows for 8 features | Verified finding | The regulariser keeps it defined but pulls the alignment toward the identity. | Use fewer features or stronger regularisation. |

### CORAL feature alignment versus Season trend (no features)
coral was worse than season_trend on average (+0.074, 4 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample, Inappropriate model assumption. Possible explanations (not proven): Large source-target distribution difference, Different feature-yield relationship, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 4/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.09; same sign for 50% of features | Possible explanation | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | Prefer parameter-based correction or target-only learning if the relationships differ. |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2024 accounts for 42% of the extra error; per-season change range -0.326 to +0.473 | Not supported | With few seasons one unusual season dominates any average. | - |
| Overfitting | mean absolute error on the target training rows = 0.314 vs 0.412 on held-out seasons (ratio 0.76) | Not supported | A gap is normal for any model; a large one signals memorising few rows. | - |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.137, +0.286] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.79; MAE of this method 0.412 vs the pure season-trend baseline 0.338 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | covariance estimated from 6.0 target rows for 8 features | Verified finding | The regulariser keeps it defined but pulls the alignment toward the identity. | Use fewer features or stronger regularisation. |

### Transfer Component Analysis versus Target-only model
tca was better than target_only on average (-0.041, 4 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, Different feature-yield relationship, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 4/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.09; same sign for 50% of features | Possible explanation | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | Prefer parameter-based correction or target-only learning if the relationships differ. |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2021 accounts for 40% of the extra error; per-season change range -1.043 to +0.597 | Not supported | With few seasons one unusual season dominates any average. | - |
| Overfitting | mean absolute error on the target training rows = 0.346 vs 0.460 on held-out seasons (ratio 0.75) | Not supported | A gap is normal for any model; a large one signals memorising few rows. | - |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.489, +0.358] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.79; MAE of this method 0.460 vs the pure season-trend baseline 0.338 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | no specific assumption was violated that the data can check | Not applicable | - | - |

### Transfer Component Analysis versus Season trend (no features)
tca was worse than season_trend on average (+0.122, 3 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, Different feature-yield relationship, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 4/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.09; same sign for 50% of features | Possible explanation | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | Prefer parameter-based correction or target-only learning if the relationships differ. |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2024 accounts for 34% of the extra error; per-season change range -0.192 to +0.426 | Not supported | With few seasons one unusual season dominates any average. | - |
| Overfitting | mean absolute error on the target training rows = 0.346 vs 0.460 on held-out seasons (ratio 0.75) | Not supported | A gap is normal for any model; a large one signals memorising few rows. | - |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.055, +0.293] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.79; MAE of this method 0.460 vs the pure season-trend baseline 0.338 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | no specific assumption was violated that the data can check | Not applicable | - | - |

### TrAdaBoost.R2 versus Target-only model
tradaboost_r2 was worse than target_only on average (+0.006, 3 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, Different feature-yield relationship, Seasonal variability: one season drives the difference, Overfitting, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 4/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | effective sample size = 1.00 of the source rows; source weight share 0.21 | Not supported | The weights show where the method put its trust, not whether that trust was right. | The weights are spread, so concentration is not the problem. |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.09; same sign for 50% of features | Possible explanation | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | Prefer parameter-based correction or target-only learning if the relationships differ. |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2024 accounts for 72% of the extra error; per-season change range -0.786 to +0.838 | Possible explanation | With few seasons one unusual season dominates any average. | Check season 2024 (weather, planting date, data quality). |
| Overfitting | mean absolute error on the target training rows = 0.206 vs 0.508 on held-out seasons (ratio 0.41) | Possible explanation | A gap is normal for any model; a large one signals memorising few rows. | Increase regularisation or use fewer features. |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.345, +0.328] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.79; MAE of this method 0.508 vs the pure season-trend baseline 0.338 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | needs about 3+ target rows to judge source rows; has 6.0 per fold | Not supported | The rule is a minimum for the algorithm to run, not a guarantee of stability. | - |

### TrAdaBoost.R2 versus Season trend (no features)
tradaboost_r2 was worse than season_trend on average (+0.170, 2 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, Different feature-yield relationship, Seasonal variability: one season drives the difference, Overfitting, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 4/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | effective sample size = 1.00 of the source rows; source weight share 0.21 | Not supported | The weights show where the method put its trust, not whether that trust was right. | The weights are spread, so concentration is not the problem. |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.09; same sign for 50% of features | Possible explanation | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | Prefer parameter-based correction or target-only learning if the relationships differ. |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2024 accounts for 53% of the extra error; per-season change range -0.177 to +0.802 | Possible explanation | With few seasons one unusual season dominates any average. | Check season 2024 (weather, planting date, data quality). |
| Overfitting | mean absolute error on the target training rows = 0.206 vs 0.508 on held-out seasons (ratio 0.41) | Possible explanation | A gap is normal for any model; a large one signals memorising few rows. | Increase regularisation or use fewer features. |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.042, +0.413] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.79; MAE of this method 0.508 vs the pure season-trend baseline 0.338 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | needs about 3+ target rows to judge source rows; has 6.0 per fold | Not supported | The rule is a minimum for the algorithm to run, not a guarantee of stability. | - |

### Parameter transfer (source Ridge + regularised target correction) versus Target-only model
ridge_prior was better than target_only on average (-0.151, 5 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, Different feature-yield relationship, Overfitting, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 4/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.09; same sign for 50% of features | Possible explanation | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | Prefer parameter-based correction or target-only learning if the relationships differ. |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2021 accounts for 61% of the extra error; per-season change range -1.162 to +0.513 | Not supported | With few seasons one unusual season dominates any average. | Check season 2021 (weather, planting date, data quality). |
| Overfitting | mean absolute error on the target training rows = 0.086 vs 0.351 on held-out seasons (ratio 0.25) | Possible explanation | A gap is normal for any model; a large one signals memorising few rows. | Increase regularisation or use fewer features. |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.597, +0.210] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.79; MAE of this method 0.351 vs the pure season-trend baseline 0.338 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
| Inappropriate model assumption | no specific assumption was violated that the data can check | Not applicable | - | - |

### Parameter transfer (source Ridge + regularised target correction) versus Season trend (no features)
ridge_prior was worse than season_trend on average (+0.012, 4 of 7 seasons better). Verified findings: Too few target observations, Unstable evaluation from a very small target sample. Possible explanations (not proven): Large source-target distribution difference, Different feature-yield relationship, Overfitting, The features may only proxy a time trend.

| Hypothesis | Measured | Status | Limitation | Recommended action |
|---|---|---|---|---|
| Large source-target distribution difference | modelling-level MMD p = 0.002; 4/8 features differ (adjusted KS); ten-day-level MMD p = 0.002 | Possible explanation | A distribution gap is not a prediction error; it can coexist with a good transfer. With 7 yearly rows the tests have little power. | Align or weight the features (KMM, CORAL) and compare with the target-only baseline. |
| Irrelevant or incorrectly weighted source rows | this method does not weight source rows | Not applicable | - | - |
| Too few target observations | 6.0 labelled target rows per training fold for 8 features (configured: 3 per feature) | Verified finding | That the condition holds does not prove it caused the larger error. | Collect more labelled target seasons or reduce the number of features. |
| Different feature-yield relationship | cosine of standardised Ridge coefficients (source vs target) = +0.09; same sign for 50% of features | Possible explanation | The target coefficients come from only 7 rows and are unstable. This diagnostic uses target labels, so it explains the result but is not used to choose a method. | Prefer parameter-based correction or target-only learning if the relationships differ. |
| Feature mismatch or inconsistent units | 0 shared feature(s) with differing units; 0 not shared; unit check after harmonisation: passed | Not supported | Declared units can still hide a different definition (for example a different aggregation window). | - |
| Seasonal variability: one season drives the difference | worst season 2019 accounts for 36% of the extra error; per-season change range -0.310 to +0.254 | Not supported | With few seasons one unusual season dominates any average. | - |
| Overfitting | mean absolute error on the target training rows = 0.086 vs 0.351 on held-out seasons (ratio 0.25) | Possible explanation | A gap is normal for any model; a large one signals memorising few rows. | Increase regularisation or use fewer features. |
| Unstable evaluation from a very small target sample | 7 held-out seasons; 95% range of the change [-0.146, +0.155] | Verified finding | With this few held-out points a difference of the size seen can arise by chance. | Treat the comparison as indicative only; add seasons before drawing conclusions. |
| The features may only proxy a time trend | Spearman(season, target yield) = -0.79; MAE of this method 0.351 vs the pure season-trend baseline 0.338 | Possible explanation | A strong trend lets leave-one-season-out interpolate between neighbouring seasons; forward validation does not. | Judge the method by forward validation and against the season-trend baseline. |
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
- 02:13:24: selected source 'IADA Barat Laut Selangor' and target 'KETARA', base model ridge
- 02:13:28: assessment: readiness conditional
- 02:13:28: harmonisation: validated
- 02:13:28: suitability: not_recommended
- 02:13:28: strategy selected: ['coral', 'tca', 'tradaboost_r2', 'ridge_prior']
- 02:13:32: ran ['source_only', 'target_only', 'target_mean', 'pooled', 'coral', 'tca', 'tradaboost_r2', 'ridge_prior']