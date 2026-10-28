# Expected results BEFORE running the transfer methods
Written 2026-10-20 20:42 UTC, after pre_transfer_assessment.py and before transfer_methods.py / yield_dynamics_methods.py were run.

## What the assessment says (the evidence the expectations rest on)
- Nine granaries are close in climate: domain AUC (climate) 0.52 to 0.63, near the 0.5 of "indistinguishable". Their optical/radar AUC is 0.58 to 0.78,
  moderate. So pooling them should work, and the differences between them are small and mostly in level, not in shape.
- KETARA is the outlier: mean distance 0.58 (others 0.32 to 0.46), climate AUC 0.93, first NDVI peak 160 days away from its nearest source.
  Transfer INTO KETARA has the highest negative-transfer risk.
- IADA Rompin: planting calendar 90 days from its nearest source; yield CV 0.23 with 2 jumps (noisy labels).
- About half the ten-day periods have no real optical observation (44% to 57%; Pulau Pinang 72%): a hidden difference between granaries.
- Yield: 7 values each; Seberang Perak CV 0.20 and Rompin 0.23 are unstable.

## Expectations: NDVI forecast (mean MAE over granaries, relative to pooled_ridge = normal training)
| Method (family)                         | Expected change | Why I expect it |
|-----------------------------------------|-----------------|-----------------|
| std_pool (feature-based, z-score each granary) | 0 to +5% worse  | granaries differ little in level, so removing level discards little signal but adds noise from estimating each granary's mean and spread from 1 to 3 years |
| coral (feature-based alignment)          | +5% to +20% worse at k=1, closer at k=3 | the target covariance must be estimated from about 36 rows and 20+ features: noisy at k=1 |
| residual (parameter-based correction)    | -3% to +5%      | a small ridge on a few target rows can fix an offset but can also overfit; little to fix for 9 of 10 granaries |
| granary_intercept (parameter-based, per-granary offset) | -5% to 0 (slightly better) | absorbs level differences and the optical-data-quality difference without discarding any source rows |
| nearest3 (source selection)              | +5% to +15% worse | with AUC near 0.5 the other six granaries are just as informative; throwing them away costs data and raises variance |
| dist_weight (source weighting by granary distance) | -2% to +3% | mild re-weighting of nearly equal sources changes little |
| tradaboost (instance weighting, already run) | about +25% worse | known result; too few target rows to judge sources |
Per granary: transfer into KETARA should have the largest error and be the one case where methods that handle shift (intercept, residual, std_pool) have the best chance to help.
Distance vs gain: I expect a negative Spearman correlation (granaries far from the rest gain less from pooling).

## Expectations: yield dynamics (mean % error, 1 to 3 years ahead)
| Method | Expected | Why |
|--------|----------|-----|
| ar1_transfer (existing)  | about equal to last_year (8.5% vs 8.1% at 1 year) | the carry-over rate is about 0.94 for every granary |
| ar1_nearest (source selection on covariate distance) | same or up to +1 point worse than ar1_transfer | the carry-over rate does not depend on weather or vegetation similarity, and 3 sources estimate it less well than 9 |
| ar1_shrunk (shrink own rate to the pooled rate) | within +-0.5 point of ar1_transfer | with 3 to 6 yearly pairs the own rate is mostly noise, so shrinkage pulls it back to the pooled rate |
Overall: no method beats last_year by a statistically clear margin; the ceiling is the behaviour of yield, not the transfer method.
