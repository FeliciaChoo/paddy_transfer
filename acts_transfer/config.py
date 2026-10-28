"""Configuration and the Domain description. Every threshold lives here so it can be changed and reported; none is hidden in the code.

Basis of the defaults (all configurable):
  alpha 0.05                    conventional significance level, applied AFTER Benjamini-Hochberg correction across features.
  smd_small/medium/large        Cohen's benchmarks (0.2 / 0.5 / 0.8) for a standardised mean difference. They describe size, not harm.
  outlier_iqr_k 3.0             Tukey's 'far out' fence (1.5 would be 'outside'); extreme values only, because yearly data are few.
  min_obs_per_feature 3         rule of thumb for how many labelled target rows per input a regression needs before it is stable; change it
                                to suit your model. It only produces a WARNING, never a block.
  min_obs_block 3               fewer labelled rows than this cannot even be cross-validated.
  r2_min_test_n 10              R2 from fewer held-out points is not reported (it is dominated by one point).
  valid_ranges                  physical plausibility, not statistical: values outside are treated as invalid, never silently kept.
"""
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd


def _ranges():
    return {"NDVI": (-1.0, 1.0), "NDVI_amplitude": (0.0, 2.0), "EVI": (-1.0, 3.0), "SAR_VV_dB": (-35.0, 5.0), "Mean_Temp_C": (10.0, 40.0),
            "Annual_Rain_mm": (0.0, 8000.0), "Rain_decade_mm": (0.0, 1500.0), "Solar_Radiation_MJ": (0.0, 35.0),
            "Optical_observed_pct": (0.0, 100.0), "Yield_t_ha": (0.0, 15.0), "Yield_kg_ha": (0.0, 15000.0)}


@dataclass
class Config:
    # data quality
    min_obs_block: int = 3
    missing_warn_pct: float = 10.0
    missing_block_pct: float = 40.0
    outlier_iqr_k: float = 3.0
    quality_min_pct: float = 20.0               # a per-row data-quality column (for example share of real optical observations) below this is flagged
    min_obs_per_feature: float = 3.0
    # domain differences
    alpha: float = 0.05
    smd_small: float = 0.2
    smd_medium: float = 0.5
    smd_large: float = 0.8
    mmd_permutations: int = 500
    random_state: int = 0
    # evaluation
    cv_strategy: str = "loso"                   # 'loso' leave-one-season-out, or 'forward' expanding window (train only on earlier seasons)
    max_target_train_groups: Optional[int] = None   # simulate scarcity: use at most this many target training seasons
    r2_min_test_n: int = 10
    bootstrap_n: int = 2000
    min_target_tradaboost: int = 3
    far_auc: float = 0.85                       # domain-classifier AUC (climate) at or above this = a far domain. A convention (0.5 is chance), not a law
    auc_min_rows: int = 30                      # rows per domain needed before a domain classifier is trusted
    trend_flag_rho: float = 0.7                 # |Spearman(season, yield)| above this: leave-one-season-out can interpolate a trend, so flag it
    kmm_max_source: int = 600
    tca_max_rows: int = 800
    coral_eps: float = 1.0
    valid_ranges: dict = field(default_factory=_ranges)
    proposed_features: tuple = ("humidity", "land surface temperature", "irrigation / water supply", "planting date", "variety", "fertiliser")

    def range_for(self, col):
        """the plausibility range of a column: exact name first, then the longest matching prefix"""
        if col in self.valid_ranges:
            return self.valid_ranges[col]
        best = max((k for k in self.valid_ranges if col.startswith(k)), key=len, default=None)
        return self.valid_ranges[best] if best else None

    def to_dict(self):
        return asdict(self)


@dataclass
class Domain:
    """One side of the transfer. df has one row per observation (here a granary-year; a farm-season elsewhere)."""
    name: str
    df: pd.DataFrame
    features: list
    target: str
    group: str                                  # the season / year column used for grouped validation
    units: dict = field(default_factory=dict)   # declared units, column -> text; used to detect mismatches
    quality_col: Optional[str] = None           # a column describing how trustworthy a row's inputs are (not a model input)
    fine_df: Optional[pd.DataFrame] = None      # optional finer observations of the same drivers (for distribution comparison only)
    fine_features: Optional[list] = None
