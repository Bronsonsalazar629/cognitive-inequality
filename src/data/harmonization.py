"""
Cross-Dataset Harmonization

Z-score standardizes cognitive and SES measures within each dataset
for cross-dataset comparison. Creates unified DataFrame with dataset labels.
"""

import pandas as pd
import numpy as np
from typing import Dict


def z_standardize(series: pd.Series) -> pd.Series:
    """Z-score standardize a series (mean=0, SD=1)."""
    mean = series.mean()
    std = series.std()
    if std > 0:
        return (series - mean) / std
    return series - mean


def harmonize_datasets(datasets: Dict[str, pd.DataFrame],
                       cognitive_col: str = 'cognitive_score',
                       ses_col: str = 'ses_index') -> pd.DataFrame:
    """
    Harmonize multiple datasets by z-standardizing key variables within each.

    Args:
        datasets: Dict mapping dataset name to DataFrame
        cognitive_col: Name of cognitive outcome column
        ses_col: Name of SES exposure column

    Returns:
        Combined DataFrame with z-scored variables and dataset labels
    """
    harmonized = []

    for name, df in datasets.items():
        h = df.copy()
        h['dataset'] = name

        if cognitive_col in h.columns:
            h['cognitive_z'] = z_standardize(h[cognitive_col])
        if ses_col in h.columns:
            h['ses_z'] = z_standardize(h[ses_col])

        harmonized.append(h)

    return pd.concat(harmonized, ignore_index=True)
