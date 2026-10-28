# Transfer learning assessment: ACTS granaries

Built from the project's own result files (data/pre_transfer_*.csv, transfer_methods_*.csv, yield_methods_*.csv, transfer_error_diagnosis.csv) and the earlier benchmarks.
Scripts: `pre_transfer_assessment.py`, `transfer_methods.py`, `yield_dynamics_methods.py`, `transfer_error_diagnosis.py`, plus the earlier `transfer_benchmark.py` and `yield_dynamics_transfer.py`.

## 1. Transfer learning scenarios: criteria, techniques and distance features

### 1.1 The questions that decide the technique
1. **Does the target have labels, and how many instances?** (none, a few dozen rows, hundreds or more)
2. **What differs between source and target?** the inputs only (covariate shift), the level or calibration of the inputs (harmonisation problem), or the relationship between inputs and the label (conditional shift)?
3. **How many sources, and how different are they?** (several close ones, one far one)
4. **Is there any signal to transfer?** If the inputs do not explain the label even inside one domain, no technique can help.

### 1.2 Scenario to technique
| # | Scenario | How to recognise it (criteria) | Technique family and examples | Main risk |
|---|---|---|---|---|
| S0 | Cold start: no target labels, similar domains | domain AUC near 0.5; small distance | Pool the sources; weight or select sources by distance | none if domains are close |
| S1 | Covariate shift: inputs differ, relationship the same | domain AUC clearly above 0.5, but a source model still fits the target | **Instance weighting**: importance weights from a domain classifier or density ratio, kernel mean matching; granary-level distance weights | weights become unstable with few rows |
| S2 | Level or calibration shift: same shape, different baseline (sensor, data quality, definitions) | per-feature mean or spread ratios differ; data-quality shares differ | **Feature-based harmonisation**: per-domain standardisation, quantile mapping, CORAL; or a per-domain intercept | removing level can remove real signal |
| S3 | Conditional shift with a few target labels | source model has biased residuals on the target; calendars or management differ | **Parameter-based**: fine-tune, residual correction, hierarchical model with shrinkage; align the calendar first | overfitting the correction on few rows |
| S4 | Many target labels, many sources of unequal relevance | at least dozens to hundreds of labelled target rows | **TrAdaBoost-type instance re-weighting** driven by target error | needs enough target rows to judge the sources |
| S5 | Far domain | domain AUC above about 0.85, calendar offset above about 30 days | **Domain adaptation** (CORAL with strong regularisation, MMD or adversarial alignment) after alignment, or do not transfer | negative transfer |
| S6 | Few labels but a long related series | the dynamics (carry-over, seasonality) look alike across series | **Parameter transfer of dynamics**: learn rates on the long or many series, apply to the short one; pretrain then fine-tune | the rates differ by place |
| S7 | No signal in the inputs | inputs uncorrelated with the label inside one domain | Do not use transfer to predict from inputs; use the label's own history | wasted effort |
| S8 | Labels defined differently | levels disagree with another source for the same place | **Harmonise definitions first** (units, season or year, area basis) | silent bias |

### 1.3 What the pre-transfer assessment uses to measure distance (covariates only, never yield)
- **Distribution distance** (standardised Wasserstein, averaged): NDVI, EVI, radar VV, temperature, rainfall, solar radiation at ten-day level. Soil pH is left out because it is one constant per granary.
- **Domain classifier AUC**, separately for climate and for optical plus radar: 0.5 means the classifier cannot tell the granary from the rest, near 1.0 means very different.
- **Crop calendar**: day of year of the first NDVI peak and the offset to the nearest source.
- **Data quality**: share of periods with a real optical observation, and the share interpolated.
- **Label behaviour (only where labels exist)**: mean, variation and large jumps of yield. Used to flag unstable labels, not to compute the distance.

### 1.4 Which scenario each ACTS task is
| Task | Scenario | What was tried | Outcome |
|---|---|---|---|
| NDVI forecast for a granary with 1 to 3 years of labels | S0 / S1, with S5 for KETARA and S3 for Rompin | pooled (normal), TrAdaBoost, feature-based, parameter-based, source selection and weighting | modest gains from distance weighting and per-granary intercepts; section 3 |
| Yield outlook, 7 yearly values | S6 | parameter transfer of the carry-over rate; nearest sources; shrinkage; instance weighting | matches "same as last year"; section 3 |
| Yield from satellite and weather inputs | S7 | pooled, TrAdaBoost | no signal in the inputs, so nothing helps |
| Detailed yield for one district (data awaited) | S4 becomes possible | not yet | plan in section 3.5 |

## 2. Pre-transfer assessment

**Have I done data assessment and harmonisation?** Partly, and the formal assessment is new:
- Done earlier: cleaning, range checks, missingness flags, interpolation with flags, planting-phase alignment of the vegetation cycle, one common ten-day grid, one common source for weather and soil.
- Done now: the pre-transfer assessment below (distance, domain AUC, calendar, data quality, label stability).
- **Not done, and needs the data owner:** harmonising the yield definition. A comparison with the 2023 Agriculture Census found agreement for Kerian (ratio 1.01) but differences of -29% to +19% for the other granaries I could read reliably, and two districts share one yield series.

### 2.1 Results
| Granary | Mean distance | Climate AUC | Optical/SAR AUC | Calendar offset (days) | Optical observed % | Interpolated % | Yield CV | Yield jumps | Risk label |
|---|---|---|---|---|---|---|---|---|---|
| IADA Barat Laut Selangor | 0.389 | 0.518 | 0.748 | 20.0 | 45.2 | 33.8 | 0.049 | 0 | ok to transfer from nearest sources |
| IADA Kemasin Semerak | 0.409 | 0.627 | 0.711 | 20.0 | 45.0 | 34.1 | 0.041 | 0 | ok to transfer from nearest sources |
| IADA Kerian | 0.364 | 0.555 | 0.721 | 30.0 | 56.6 | 27.3 | 0.074 | 0 | ok to transfer from nearest sources |
| IADA Pekan | 0.335 | 0.58 | 0.662 | 0.0 | 43.8 | 34.5 | 0.077 | 0 | ok to transfer from nearest sources |
| IADA Pulau Pinang | 0.399 | 0.569 | 0.699 | 20.0 | 71.5 | 18.6 | 0.052 | 0 | ok to transfer from nearest sources |
| IADA Rompin | 0.351 | 0.606 | 0.575 | 90.0 | 44.0 | 33.6 | 0.23 | 2 | calendar differs: align stage first |
| IADA Seberang Perak | 0.319 | 0.578 | 0.647 | 0.0 | 55.5 | 28.0 | 0.2 | 1 | ok to transfer from nearest sources |
| KADA | 0.359 | 0.619 | 0.66 | 20.0 | 45.5 | 32.6 | 0.098 | 0 | ok to transfer from nearest sources |
| KETARA | 0.58 | 0.93 | 0.676 | 160.0 | 47.4 | 32.6 | 0.135 | 0 | far domain: do not pool blindly |
| MADA | 0.463 | 0.602 | 0.777 | 20.0 | 57.2 | 28.3 | 0.113 | 0 | ok to transfer from nearest sources |

### 2.2 Findings
- **Nine granaries are close in climate** (AUC 0.52 to 0.63). Pooling them is reasonable.
- **KETARA is a far domain**: mean distance 0.58, climate AUC 0.93, first NDVI peak 160 days from its nearest source. Transfer into KETARA has the highest risk.
- **Rompin's calendar differs by 90 days** and its yield is unstable (CV 0.23, two jumps). Seberang Perak is unstable too (CV 0.20).
- **Data quality differs**: real optical observations are 44% to 57% of periods for most granaries but 72% for Pulau Pinang. This is a hidden difference between domains.
- **Nearest sources follow covariates, not geography**: Pekan's nearest is Seberang Perak, not Rompin.

### 2.3 Harmonisation checklist
| Item | Status |
|---|---|
| One spatial unit (granary = its districts, sampled cropland pixels) | done |
| One ten-day time grid for all granaries | done |
| Gap filling of optical data, with flags kept | done; share of observed data differs by granary (flagged above) |
| Radar and optical on the same grid | done |
| Planting calendar alignment (phase features) | done for the model; Rompin and KETARA are still far from their nearest sources |
| Weather and soil from one source for all granaries | done (coarse: one weather value per district) |
| Column naming and units (the rainfall column is a ten-day total) | consistent across granaries; the name is misleading |
| Feature scaling across granaries | global only; per-granary scaling was tested and rejected (section 3) |
| **Yield definition and basis** | **not harmonised; needs the data owner** |
| Yield time basis (annual versus season) | annual only; season-level data would help |
| Label noise (jumps) | flagged for Rompin and Seberang Perak |

### 2.4 Decision gate before any transfer
1. Compute the assessment on covariates only. 2. If climate AUC is above about 0.85 or the calendar offset above about 30 days, align or adapt first, or do not transfer. 3. Otherwise pool the sources, weighted by distance. 4. Check that the label definitions match before pooling any labels.

## 3. Expected result and actual result

### 3.1 Expectations, written before the transfer methods were run
See `data/pre_transfer_expectations.md` for the full text and the time it was written.

### 3.2 NDVI forecast: expected versus actual (1 year of own labels; change in error against normal pooled training; negative is better)
| Method | Family | Expected (before) | Actual, 10 days | Actual, 30 days |
|---|---|---|---|---|
| std_pool | Feature-based | 0 to +5% worse | +2.5% (worse; 3/10 granaries better) | +13.2% (worse; 0/10 granaries better) |
| coral | Feature-based / domain adaptation | +5% to +20% worse (k=1) | -0.2% (no clear difference; 6/10 granaries better) | +10.0% (worse; 3/10 granaries better) |
| residual | Parameter-based | -3% to +5% | +9.0% (worse; 3/10 granaries better) | +7.2% (no clear difference; 4/10 granaries better) |
| granary_intercept | Parameter-based (hierarchical) | -5% to 0% | -1.4% (no clear difference; 7/10 granaries better) | -2.7% (no clear difference; 8/10 granaries better) |
| nearest3 | Source selection | +5% to +15% worse | -1.5% (no clear difference; 5/10 granaries better) | -4.6% (no clear difference; 7/10 granaries better) |
| dist_weight | Source weighting (granary-level instance weighting) | -2% to +3% | -2.3% (better; 10/10 granaries better) | -3.9% (better; 9/10 granaries better) |

TrAdaBoost (earlier run, same task): 0.0740 vs pooled 0.0581 at 10 days (+27%), 0.1013 vs 0.0747 at 30 days (+36%).

### 3.3 Yield outlook: expected versus actual (mean absolute % error)
| Ahead | Cases | Last year | Normal (own) | Transfer (all others) | Nearest 3 sources | Shrunk own rate |
|---|---|---|---|---|---|---|
| 1 year | 40 | 8.14 | 10.22 | 8.45 | 8.65 | 8.85 |
| 2 years | 30 | 11.09 | 11.89 | 10.66 | 10.71 | 11.14 |
| 3 years | 20 | 15.88 | 14.21 | 14.28 | 14.28 | 14.28 |

Expected: nearest sources and shrinkage within about one point of the existing transfer method, and none clearly better than last year. Actual: nearest sources +0.2 points and shrinkage +0.4 points worse than the existing transfer method at one year, and both clearly worse than last year at one year. Two and three years ahead are equal within the noise (only 30 and 20 cases).

### 3.4 Why some methods had more error
| Variant (30 days, 1 year of own labels) | Mean MAE | Change vs pooled (%) | Granaries better than pooled |
|---|---|---|---|
| pooled_ridge | 0.0747 | 0.0 | 0 |
| std_pool | 0.0845 | 13.7 | 0 |
| std_pool_stable | 0.0834 | 12.0 | 0 |
| residual | 0.08 | 8.7 | 4 |
| residual_strong | 0.0733 | -1.1 | 8 |
| residual_bias | 0.074 | -0.1 | 6 |
| coral | 0.0821 | 10.7 | 3 |
| coral_eps10 | 0.0744 | 0.3 | 7 |
| coral_eps100 | 0.0734 | -1.0 | 7 |

- **Per-granary standardisation (std_pool, +13% at 30 days):** the cause is not noisy statistics, because using the target's full feature history still gives +12%. The likely cause is lost information: the granary's own baseline and amplitude carry signal for forecasts that look further ahead. It is worse at 30 days than at 10.
- **Residual correction (+9% at 10 days):** too flexible for about 36 target rows. A constant bias correction or a 100 times stronger penalty brings it back to normal (-0.1% and -1.1% at 30 days).
- **CORAL (+10% at 30 days):** the target covariance is estimated from few rows for more than 20 features. A much stronger regulariser brings it back to normal or slightly better (+0.3% and -1.0%). It helps only where the domains really differ: KETARA improved by 7% at 10 days and 17% at 30 days, while close granaries such as Pekan and Seberang Perak got worse. The correlation between distance and gain is positive (0.65 at 10 days, 0.81 at 30 days).
- **TrAdaBoost:** needs enough target rows to judge which source rows to trust. With 36 rows or fewer its weights are unstable, and its error lands between training on the target alone and normal pooled training.
- **Hard source selection (nearest 3):** keeps closer sources but throws away data. It helped some granaries (Kemasin Semerak -21%) and hurt others (Rompin +7%), so the average is not clearly different.
- **Distance weighting works** because it keeps all sources and only tilts toward the closer ones: better in 10 of 10 granaries at 10 days.
- **Yield outlook:** the carry-over rate does not depend on weather or vegetation similarity, so choosing sources by covariate distance cannot help, and shrinking toward the pooled rate only moves it back to the existing method.

### 3.5 When the detailed yield data arrives
Many target rows (seasons by sections) make S4 possible, so instance weighting and TrAdaBoost can be tested fairly. Use this order:
1. Compute the assessment on the covariates of the new target. 2. Harmonise the yield definition with the data owner. 3. Test normal training first. 4. Then distance-weighted pooling, per-granary intercepts, strongly regularised adaptation, and TrAdaBoost-type instance weighting. 5. Validate by leaving out whole sections and whole seasons, and report the paired confidence interval over units, not only the average.

## 4. Limits of this evidence
- Ten granaries and seven years: small differences of two to four percent are only just distinguishable from noise.
- The regularisation variants in 3.4 were chosen after seeing the failures. They explain the failures, but they have not been validated as new methods.
- Using KETARA's result to choose an adaptation for it is a hypothesis to test on new data, not a proven rule.
