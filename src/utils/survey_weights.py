"""Complex survey design adjustment utilities."""

import pandas as pd
from typing import Optional


def apply_nhanes_weights(data: pd.DataFrame, weight_col: str = 'WTMEC2YR',
                         psu_col: str = 'SDMVPSU', strata_col: str = 'SDMVSTRA'):
    """Apply NHANES complex survey weights for population-level inference."""
    raise NotImplementedError("Utility implementation")


def apply_brfss_weights(data: pd.DataFrame, weight_col: str = '_LLCPWT',
                        strata_col: str = '_STSTR'):
    """Apply BRFSS survey weights."""
    raise NotImplementedError("Utility implementation")
