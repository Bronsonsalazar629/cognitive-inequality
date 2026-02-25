# Cognitive Inequality Research System - Pivot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pivot the Clinical Fairness Intervention System into a Cognitive Inequality Research platform that analyzes causal pathways from socioeconomic status to cognitive decline using NHANES, BRFSS, and GSS datasets.

**Architecture:** In-place modification of existing codebase. Delete clinical-domain modules, rename/adapt reusable ones (LLM clients, PC algorithm, graph refiner), build new data loaders and analysis stubs. The 4-tier LLM architecture is preserved with updated prompts for social epidemiology.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy, statsmodels, scikit-learn, xgboost, shap, causal-learn, networkx, plotly, matplotlib, seaborn, pyreadstat, pydantic, pyyaml

---

### Task 1: Rename project directory and initialize git

**Files:**
- Rename: `Clinical-fairness-intervention-system/` -> `cognitive-inequality-research/`

**Step 1: Rename the project directory**

```bash
mv /home/bronson/2res/Clinical-fairness-intervention-system /home/bronson/2res/cognitive-inequality-research
```

**Step 2: Initialize git repo**

```bash
cd /home/bronson/2res/cognitive-inequality-research
git init
echo "__pycache__/
*.pyc
.env
data/raw/
data/processed/
*.csv
*.xpt
*.dta
*.zip
llm_cache/
llm_logs/
results/
*.pkl
.DS_Store
" > .gitignore
```

**Step 3: Commit initial state**

```bash
git add -A
git commit -m "chore: initialize cognitive inequality research project from clinical fairness codebase"
```

---

### Task 2: Delete clinical-domain files

**Files:**
- Delete: `src/bias_detection.py`
- Delete: `src/clinical_fairness_evaluator.py`
- Delete: `src/clinical_fairness_report.py`
- Delete: `src/visualize_bias_pathways.py`
- Delete: `src/code_generator.py`
- Delete: `src/systematic_experiments.py`
- Delete: `src/validate_integration.py`
- Delete: `src/llm_refinement_assistant.py`
- Delete: `176541_DE1_0_2008_Beneficiary_Summary_File_Sample_1/` (entire directory)
- Delete: `clinical-fairness-hack/` (entire directory)
- Delete: `results/` contents (keep directory)
- Delete: `scripts/` contents (keep directory)

**Step 1: Delete old domain files**

```bash
cd /home/bronson/2res/cognitive-inequality-research
rm -f src/bias_detection.py src/clinical_fairness_evaluator.py src/clinical_fairness_report.py
rm -f src/visualize_bias_pathways.py src/code_generator.py src/systematic_experiments.py
rm -f src/validate_integration.py src/llm_refinement_assistant.py
rm -rf "176541_DE1_0_2008_Beneficiary_Summary_File_Sample_1"
rm -rf clinical-fairness-hack
rm -rf results/*
rm -rf scripts/*
```

**Step 2: Commit deletion**

```bash
git add -A
git commit -m "chore: remove clinical fairness domain-specific modules"
```

---

### Task 3: Create new directory structure

**Files:**
- Create directories for the new module layout

**Step 1: Create directory tree**

```bash
cd /home/bronson/2res/cognitive-inequality-research
mkdir -p src/data src/analysis src/llm src/simulation src/visualization src/utils
mkdir -p data/raw data/processed data/cache
mkdir -p tests notebooks results/figures results/tables results/models results/reports
touch src/data/__init__.py src/analysis/__init__.py src/llm/__init__.py
touch src/simulation/__init__.py src/visualization/__init__.py src/utils/__init__.py
```

**Step 2: Move existing files to new locations**

```bash
# LLM modules -> src/llm/
mv src/llm_client_base.py src/llm/llm_client_base.py
mv src/deepseek_client.py src/llm/deepseek_client.py
mv src/gemini_client.py src/llm/gemini_client.py
mv src/causal_graph_refiner.py src/llm/causal_graph_refiner.py
mv src/bias_interpreter.py src/llm/impact_interpreter.py
mv src/intervention_recommender.py src/llm/intervention_recommender.py

# Analysis modules -> src/analysis/
mv src/pc_algorithm_clinical.py src/analysis/pc_algorithm_social.py
mv src/causal_analysis.py src/analysis/causal_analysis.py

# Utils
mv src/confidence_intervals.py src/utils/bootstrap.py

# Visualization
mv src/graphs.py src/visualization/graphs.py

# Keep intervention_engine for reference during adaptation, then delete
mv src/intervention_engine.py src/simulation/intervention_engine_old.py
```

**Step 3: Remove leftover empty files from src/ root (except main.py and __init__.py)**

```bash
# Remove any research-grade causal analysis duplicate
rm -f src/causal_analysis_research_grade.py
```

**Step 4: Commit restructure**

```bash
git add -A
git commit -m "refactor: restructure into modular layout (data/analysis/llm/simulation/visualization/utils)"
```

---

### Task 4: Update requirements.txt

**Files:**
- Modify: `requirements.txt`

**Step 1: Write updated requirements**

```
# Core data science
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
statsmodels>=0.14.0

# Data loading
pyreadstat>=1.2.0

# Causal inference
causal-learn>=0.1.3
networkx>=3.1

# ML prediction model
xgboost>=2.0.0
shap>=0.43.0

# LLM integration
requests>=2.31.0
pydantic>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.17.0
pyvis>=0.3.2

# Web app
streamlit>=1.28.0

# Config
pyyaml>=6.0
```

**Step 2: Install new dependencies**

Run: `pip install pyreadstat xgboost shap statsmodels`

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: update dependencies for cognitive inequality research"
```

---

### Task 5: Fix import paths in moved LLM modules

**Files:**
- Modify: `src/llm/causal_graph_refiner.py` (line 24: `from src.llm_client_base import` -> `from src.llm.llm_client_base import`)
- Modify: `src/llm/impact_interpreter.py` (line 17: same fix)
- Modify: `src/llm/intervention_recommender.py` (line 18: same fix)
- Modify: `src/llm/gemini_client.py` (check for imports of `llm_client_base`)
- Modify: `src/llm/deepseek_client.py` (check for imports of `llm_client_base`)

**Step 1: Update all import paths**

In each file under `src/llm/`, replace:
- `from src.llm_client_base import` -> `from src.llm.llm_client_base import`
- `from llm_client_base import` -> `from src.llm.llm_client_base import`

In `src/analysis/causal_analysis.py`, update any imports of moved modules.

In `src/analysis/pc_algorithm_social.py`, update any imports.

**Step 2: Verify imports work**

Run: `cd /home/bronson/2res/cognitive-inequality-research && python -c "from src.llm.gemini_client import create_smart_llm_client; print('OK')"`
Expected: `OK` (or import error to fix)

**Step 3: Commit**

```bash
git add -A
git commit -m "fix: update import paths after module reorganization"
```

---

### Task 6: Build NHANES data loader

**Files:**
- Create: `src/data/data_loader_nhanes.py`
- Test: `tests/test_data_loader_nhanes.py`

**Step 1: Write the test**

```python
# tests/test_data_loader_nhanes.py
"""Tests for NHANES data loader."""
import pytest
import pandas as pd
import numpy as np


def test_create_cognitive_composite():
    """Cognitive composite should be mean of z-scored tests."""
    from src.data.data_loader_nhanes import create_cognitive_composite

    df = pd.DataFrame({
        'CFDDS': [50.0, 60.0, 70.0, 80.0],
        'CFDCST': [15.0, 20.0, 25.0, 30.0],
        'CFDCSR': [3.0, 5.0, 7.0, 9.0],
    })
    result = create_cognitive_composite(df)
    assert 'cognitive_score' in result.columns
    assert abs(result['cognitive_score'].mean()) < 0.01  # z-scored mean ~ 0
    assert abs(result['cognitive_score'].std() - 1.0) < 0.3  # SD ~ 1


def test_create_ses_index():
    """SES index should be 0-1 normalized weighted composite."""
    from src.data.data_loader_nhanes import create_ses_index

    df = pd.DataFrame({
        'INDFMPIR': [0.0, 2.5, 5.0],
        'DMDEDUC2': [1.0, 3.0, 5.0],
    })
    result = create_ses_index(df)
    assert 'ses_index' in result.columns
    assert result['ses_index'].min() >= 0.0
    assert result['ses_index'].max() <= 1.0
    # Lowest income + lowest education = lowest SES
    assert result['ses_index'].iloc[0] == result['ses_index'].min()


def test_create_depression_score():
    """Depression score should sum PHQ-9 items."""
    from src.data.data_loader_nhanes import create_depression_score

    df = pd.DataFrame({
        'DPQ020': [0, 1, 2],
        'DPQ030': [0, 1, 2],
        'DPQ040': [0, 1, 2],
        'DPQ050': [0, 1, 2],
        'DPQ060': [0, 1, 2],
        'DPQ070': [0, 1, 2],
        'DPQ080': [0, 1, 2],
        'DPQ090': [0, 1, 2],
        'DPQ100': [0, 0, 0],
    })
    result = create_depression_score(df)
    assert 'depression_score' in result.columns
    assert result['depression_score'].iloc[0] == 0
    assert result['depression_score'].iloc[1] == 8
    assert result['depression_score'].iloc[2] == 16


def test_create_screen_time():
    """Screen time should sum computer and TV hours."""
    from src.data.data_loader_nhanes import create_screen_time

    df = pd.DataFrame({
        'PAQ710': [2.0, 4.0, float('nan')],
        'PAQ715': [3.0, 2.0, 1.0],
    })
    result = create_screen_time(df)
    assert 'screen_time_hours' in result.columns
    assert result['screen_time_hours'].iloc[0] == 5.0
    assert result['screen_time_hours'].iloc[1] == 6.0
    assert pd.isna(result['screen_time_hours'].iloc[2])


def test_filter_age_range():
    """Should filter to ages 25-45."""
    from src.data.data_loader_nhanes import filter_age_range

    df = pd.DataFrame({
        'RIDAGEYR': [20, 25, 35, 45, 50, 60],
        'val': [1, 2, 3, 4, 5, 6],
    })
    result = filter_age_range(df)
    assert len(result) == 3
    assert result['RIDAGEYR'].min() >= 25
    assert result['RIDAGEYR'].max() <= 45
```

**Step 2: Run test to verify it fails**

Run: `cd /home/bronson/2res/cognitive-inequality-research && python -m pytest tests/test_data_loader_nhanes.py -v`
Expected: FAIL (module not found)

**Step 3: Write the implementation**

```python
# src/data/data_loader_nhanes.py
"""
NHANES 2013-2014 Data Loader

Loads and preprocesses National Health and Nutrition Examination Survey data
for cognitive inequality analysis. Fetches demographic, cognitive, depression,
screen time, sleep, healthcare access, and confounder variables.

Target population: US adults ages 25-45.
Output: Processed DataFrame with composite scores and survey weights.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# NHANES 2013-2014 table URLs (XPT format)
NHANES_TABLES = {
    'DEMO_H': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT',
    'CFQ_H': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/CFQ_H.XPT',
    'DPQ_H': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DPQ_H.XPT',
    'PAQ_H': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/PAQ_H.XPT',
    'SLQ_H': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/SLQ_H.XPT',
    'HIQ_H': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/HIQ_H.XPT',
    'SMQ_H': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/SMQ_H.XPT',
    'ALQ_H': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/ALQ_H.XPT',
    'BMX_H': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BMX_H.XPT',
    'BPQ_H': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BPQ_H.XPT',
    'DIQ_H': 'https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DIQ_H.XPT',
}

# Columns to keep from each table
TABLE_COLUMNS = {
    'DEMO_H': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'INDFMPIR',
               'DMDEDUC2', 'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA'],
    'CFQ_H': ['SEQN', 'CFDDS', 'CFDCST', 'CFDCSR'],
    'DPQ_H': ['SEQN', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060',
              'DPQ070', 'DPQ080', 'DPQ090', 'DPQ100'],
    'PAQ_H': ['SEQN', 'PAQ710', 'PAQ715'],
    'SLQ_H': ['SEQN', 'SLQ120'],
    'HIQ_H': ['SEQN', 'HIQ011'],
    'SMQ_H': ['SEQN', 'SMQ020'],
    'ALQ_H': ['SEQN', 'ALQ130'],
    'BMX_H': ['SEQN', 'BMXBMI'],
    'BPQ_H': ['SEQN', 'BPQ020'],
    'DIQ_H': ['SEQN', 'DIQ010'],
}


def fetch_nhanes_table(table_name: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Fetch a single NHANES table from CDC website or local cache.

    Args:
        table_name: NHANES table identifier (e.g., 'DEMO_H')
        cache_dir: Directory to cache downloaded files

    Returns:
        DataFrame with requested columns
    """
    if cache_dir:
        cache_path = cache_dir / f'{table_name}.csv'
        if cache_path.exists():
            logger.info(f"Loading {table_name} from cache")
            return pd.read_csv(cache_path)

    url = NHANES_TABLES[table_name]
    logger.info(f"Downloading {table_name} from {url}")
    df = pd.read_sas(url)

    # Keep only needed columns (some may be missing)
    wanted = TABLE_COLUMNS.get(table_name, [])
    available = [c for c in wanted if c in df.columns]
    df = df[available]

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_dir / f'{table_name}.csv', index=False)

    return df


def filter_age_range(df: pd.DataFrame, min_age: int = 25, max_age: int = 45) -> pd.DataFrame:
    """Filter to target age range."""
    return df[(df['RIDAGEYR'] >= min_age) & (df['RIDAGEYR'] <= max_age)].copy()


def create_cognitive_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create z-scored cognitive composite from CFDDS, CFDCST, CFDCSR.

    Requires at least 2 of 3 tests present for a valid composite.
    """
    df = df.copy()
    tests = ['CFDDS', 'CFDCST', 'CFDCSR']

    for test in tests:
        if test in df.columns:
            mean = df[test].mean()
            std = df[test].std()
            if std > 0:
                df[f'{test}_z'] = (df[test] - mean) / std
            else:
                df[f'{test}_z'] = 0.0
        else:
            df[f'{test}_z'] = np.nan

    z_cols = [f'{t}_z' for t in tests]
    valid_count = df[z_cols].notna().sum(axis=1)
    df['cognitive_score'] = df[z_cols].mean(axis=1)
    df.loc[valid_count < 2, 'cognitive_score'] = np.nan

    return df


def create_ses_index(df: pd.DataFrame, income_weight: float = 0.6, education_weight: float = 0.4) -> pd.DataFrame:
    """
    Create SES index from poverty income ratio and education level.

    Normalized to 0-1 scale with configurable weights.
    """
    df = df.copy()

    if 'INDFMPIR' in df.columns:
        pir_min, pir_max = df['INDFMPIR'].min(), df['INDFMPIR'].max()
        if pir_max > pir_min:
            df['income_norm'] = (df['INDFMPIR'] - pir_min) / (pir_max - pir_min)
        else:
            df['income_norm'] = 0.5
    else:
        df['income_norm'] = np.nan

    if 'DMDEDUC2' in df.columns:
        # DMDEDUC2: 1=<9th grade, 2=9-11, 3=HS/GED, 4=Some college, 5=College+
        df['education_norm'] = (df['DMDEDUC2'] - 1) / 4
    else:
        df['education_norm'] = np.nan

    df['ses_index'] = income_weight * df['income_norm'] + education_weight * df['education_norm']

    return df


def create_depression_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create PHQ-9 depression score from DPQ items."""
    df = df.copy()
    phq9_cols = [f'DPQ0{i}0' for i in range(2, 10)] + ['DPQ100']
    available = [c for c in phq9_cols if c in df.columns]
    df['depression_score'] = df[available].sum(axis=1)
    # Set to NaN if more than 2 items missing
    missing_count = df[available].isna().sum(axis=1)
    df.loc[missing_count > 2, 'depression_score'] = np.nan
    return df


def create_screen_time(df: pd.DataFrame) -> pd.DataFrame:
    """Create screen time composite from computer + TV hours."""
    df = df.copy()
    if 'PAQ710' in df.columns and 'PAQ715' in df.columns:
        df['screen_time_hours'] = df['PAQ710'] + df['PAQ715']
    elif 'PAQ710' in df.columns:
        df['screen_time_hours'] = df['PAQ710']
    elif 'PAQ715' in df.columns:
        df['screen_time_hours'] = df['PAQ715']
    else:
        df['screen_time_hours'] = np.nan
    return df


def create_healthcare_access(df: pd.DataFrame) -> pd.DataFrame:
    """Create healthcare access indicator from insurance status."""
    df = df.copy()
    if 'HIQ011' in df.columns:
        # HIQ011: 1=Yes, 2=No, 7=Refused, 9=Don't know
        df['has_insurance'] = (df['HIQ011'] == 1).astype(float)
        df.loc[df['HIQ011'].isin([7, 9]), 'has_insurance'] = np.nan
    else:
        df['has_insurance'] = np.nan
    return df


def create_sleep_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Extract sleep hours and create deprivation indicator."""
    df = df.copy()
    if 'SLQ120' in df.columns:
        df['sleep_hours'] = df['SLQ120']
        df['sleep_deprived'] = ((df['SLQ120'] < 6) | (df['SLQ120'] > 9)).astype(float)
    else:
        df['sleep_hours'] = np.nan
        df['sleep_deprived'] = np.nan
    return df


def create_confounders(df: pd.DataFrame) -> pd.DataFrame:
    """Encode confounder variables."""
    df = df.copy()
    if 'RIAGENDR' in df.columns:
        df['female'] = (df['RIAGENDR'] == 2).astype(float)
    if 'SMQ020' in df.columns:
        df['smoker'] = (df['SMQ020'] == 1).astype(float)
        df.loc[df['SMQ020'].isin([7, 9]), 'smoker'] = np.nan
    if 'BPQ020' in df.columns:
        df['hypertension'] = (df['BPQ020'] == 1).astype(float)
        df.loc[df['BPQ020'].isin([7, 9]), 'hypertension'] = np.nan
    if 'DIQ010' in df.columns:
        df['diabetes'] = (df['DIQ010'] == 1).astype(float)
        df.loc[df['DIQ010'].isin([7, 9]), 'diabetes'] = np.nan
    return df


def load_nhanes(cache_dir: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess complete NHANES 2013-2014 dataset for cognitive analysis.

    Args:
        cache_dir: Directory for caching raw downloads
        output_path: Path to save processed CSV

    Returns:
        Processed DataFrame with all variables, filtered to ages 25-45
    """
    cache = Path(cache_dir) if cache_dir else None

    # Fetch all tables
    tables = {}
    for name in NHANES_TABLES:
        tables[name] = fetch_nhanes_table(name, cache)

    # Merge all on SEQN
    df = tables['DEMO_H']
    for name, table in tables.items():
        if name != 'DEMO_H':
            df = df.merge(table, on='SEQN', how='left')

    logger.info(f"Merged dataset: {len(df)} rows, {len(df.columns)} columns")

    # Filter age range
    df = filter_age_range(df)
    logger.info(f"After age filter (25-45): {len(df)} rows")

    # Create composite variables
    df = create_cognitive_composite(df)
    df = create_ses_index(df)
    df = create_depression_score(df)
    df = create_screen_time(df)
    df = create_healthcare_access(df)
    df = create_sleep_variable(df)
    df = create_confounders(df)

    # Create SES quartiles
    df['ses_quartile'] = pd.qcut(
        df['ses_index'], q=4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High']
    )

    # Select final columns
    final_cols = [
        'SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'female',
        'INDFMPIR', 'DMDEDUC2', 'ses_index', 'ses_quartile',
        'CFDDS', 'CFDCST', 'CFDCSR', 'cognitive_score',
        'depression_score', 'screen_time_hours', 'sleep_hours', 'sleep_deprived',
        'has_insurance', 'smoker', 'ALQ130', 'BMXBMI', 'hypertension', 'diabetes',
        'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA',
    ]
    available_final = [c for c in final_cols if c in df.columns]
    df = df[available_final]

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed NHANES data to {output_path}")

    return df
```

**Step 4: Run tests**

Run: `cd /home/bronson/2res/cognitive-inequality-research && python -m pytest tests/test_data_loader_nhanes.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/data/data_loader_nhanes.py tests/test_data_loader_nhanes.py
git commit -m "feat: add NHANES 2013-2014 data loader with cognitive composite, SES index, and variable construction"
```

---

### Task 7: Build BRFSS data loader

**Files:**
- Create: `src/data/data_loader_brfss.py`
- Test: `tests/test_data_loader_brfss.py`

**Step 1: Write the test**

```python
# tests/test_data_loader_brfss.py
"""Tests for BRFSS data loader."""
import pytest
import pandas as pd
import numpy as np


def test_create_brfss_ses_index():
    """SES index should use 70/30 income/education weighting."""
    from src.data.data_loader_brfss import create_ses_index

    df = pd.DataFrame({
        'INCOME2': [1, 4, 8],   # 1=<$10K, 8=≥$75K
        'EDUCA': [1, 3, 6],     # 1=Never, 6=College grad
    })
    result = create_ses_index(df)
    assert 'ses_index' in result.columns
    assert result['ses_index'].iloc[0] < result['ses_index'].iloc[2]
    assert result['ses_index'].min() >= 0.0
    assert result['ses_index'].max() <= 1.0


def test_create_cognitive_impairment():
    """Binary cognitive impairment from DECIDE variable."""
    from src.data.data_loader_brfss import create_cognitive_impairment

    df = pd.DataFrame({
        'DECIDE': [1, 2, 7, 9],  # 1=Yes, 2=No, 7=Refused, 9=DK
    })
    result = create_cognitive_impairment(df)
    assert result['cognitive_impairment'].iloc[0] == 1.0
    assert result['cognitive_impairment'].iloc[1] == 0.0
    assert pd.isna(result['cognitive_impairment'].iloc[2])
    assert pd.isna(result['cognitive_impairment'].iloc[3])


def test_filter_brfss_age():
    """Should filter to ages 25-44."""
    from src.data.data_loader_brfss import filter_age_range

    df = pd.DataFrame({
        '_AGE_G': [1, 2, 3, 4, 5],  # age groups
        'val': range(5),
    })
    result = filter_age_range(df)
    assert all(result['_AGE_G'].isin([2, 3]))
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_loader_brfss.py -v`
Expected: FAIL

**Step 3: Write the implementation**

```python
# src/data/data_loader_brfss.py
"""
BRFSS 2022 Data Loader

Loads Behavioral Risk Factor Surveillance System data for validation
of the SES-cognition gradient. Uses self-reported cognitive difficulty
as outcome (binary), with stress, sleep, and insurance as mediators.

Target population: US adults ages 25-44.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BRFSS_URL = 'https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip'

BRFSS_COLUMNS = [
    'DECIDE', 'INCOME2', 'EDUCA', 'EMPLOY1',
    'MENTHLTH', 'SLEPTIM1', 'HLTHPLN1', 'GENHLTH',
    '_AGE_G', 'SEX1', '_RACE', 'SMOKDAY2', '_BMI5',
    '_LLCPWT', '_STSTR',
]


def filter_age_range(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to ages 25-44 using _AGE_G (2=25-34, 3=35-44)."""
    return df[df['_AGE_G'].isin([2, 3])].copy()


def create_cognitive_impairment(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary cognitive impairment from DECIDE variable."""
    df = df.copy()
    df['cognitive_impairment'] = np.nan
    df.loc[df['DECIDE'] == 1, 'cognitive_impairment'] = 1.0
    df.loc[df['DECIDE'] == 2, 'cognitive_impairment'] = 0.0
    return df


def create_ses_index(df: pd.DataFrame, income_weight: float = 0.7, education_weight: float = 0.3) -> pd.DataFrame:
    """Create SES index from income and education (70/30 weighting)."""
    df = df.copy()

    if 'INCOME2' in df.columns:
        valid_income = df['INCOME2'].isin(range(1, 9))
        df['income_norm'] = np.nan
        df.loc[valid_income, 'income_norm'] = (df.loc[valid_income, 'INCOME2'] - 1) / 7
    else:
        df['income_norm'] = np.nan

    if 'EDUCA' in df.columns:
        valid_educ = df['EDUCA'].isin(range(1, 7))
        df['education_norm'] = np.nan
        df.loc[valid_educ, 'education_norm'] = (df.loc[valid_educ, 'EDUCA'] - 1) / 5
    else:
        df['education_norm'] = np.nan

    df['ses_index'] = income_weight * df['income_norm'] + education_weight * df['education_norm']
    return df


def create_mediators(df: pd.DataFrame) -> pd.DataFrame:
    """Create mediator variables from BRFSS fields."""
    df = df.copy()

    # Mental health days (stress proxy)
    if 'MENTHLTH' in df.columns:
        df['mental_health_days'] = df['MENTHLTH'].copy()
        df.loc[df['MENTHLTH'] == 88, 'mental_health_days'] = 0  # 88 = None
        df.loc[df['MENTHLTH'].isin([77, 99]), 'mental_health_days'] = np.nan

    # Sleep hours
    if 'SLEPTIM1' in df.columns:
        df['sleep_hours'] = df['SLEPTIM1'].copy()
        df.loc[df['SLEPTIM1'].isin([77, 99]), 'sleep_hours'] = np.nan

    # Insurance
    if 'HLTHPLN1' in df.columns:
        df['has_insurance'] = np.nan
        df.loc[df['HLTHPLN1'] == 1, 'has_insurance'] = 1.0
        df.loc[df['HLTHPLN1'] == 2, 'has_insurance'] = 0.0

    return df


def load_brfss(raw_path: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess BRFSS 2022 dataset.

    Args:
        raw_path: Path to already-downloaded XPT file (skip download if provided)
        output_path: Path to save processed CSV

    Returns:
        Processed DataFrame filtered to ages 25-44
    """
    if raw_path and Path(raw_path).exists():
        logger.info(f"Loading BRFSS from {raw_path}")
        df = pd.read_sas(raw_path)
    else:
        logger.info(f"Downloading BRFSS from CDC (this is ~500MB)...")
        logger.info("Download URL: " + BRFSS_URL)
        logger.info("Please download manually and pass raw_path parameter.")
        raise FileNotFoundError(
            f"BRFSS data not found. Download from {BRFSS_URL}, "
            f"extract the XPT file, and pass its path as raw_path."
        )

    # Keep only needed columns
    available = [c for c in BRFSS_COLUMNS if c in df.columns]
    df = df[available]

    df = filter_age_range(df)
    logger.info(f"After age filter (25-44): {len(df)} rows")

    df = create_cognitive_impairment(df)
    df = create_ses_index(df)
    df = create_mediators(df)

    # SES quartiles
    df['ses_quartile'] = pd.qcut(
        df['ses_index'], q=4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High']
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed BRFSS data to {output_path}")

    return df
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_data_loader_brfss.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/data/data_loader_brfss.py tests/test_data_loader_brfss.py
git commit -m "feat: add BRFSS 2022 data loader with cognitive impairment, SES index, and mediators"
```

---

### Task 8: Build GSS data loader

**Files:**
- Create: `src/data/data_loader_gss.py`
- Test: `tests/test_data_loader_gss.py`

**Step 1: Write the test**

```python
# tests/test_data_loader_gss.py
"""Tests for GSS data loader."""
import pytest
import pandas as pd
import numpy as np


def test_create_gss_ses_index():
    """SES index should use 50/30/20 weighting."""
    from src.data.data_loader_gss import create_ses_index

    df = pd.DataFrame({
        'REALINC': [10000, 30000, 60000],
        'EDUC': [8, 12, 18],
        'PRESTG80': [20, 40, 70],
    })
    result = create_ses_index(df)
    assert 'ses_index' in result.columns
    assert result['ses_index'].iloc[0] < result['ses_index'].iloc[2]


def test_create_cognitive_score():
    """Cognitive score should scale WORDSUM to 0-100."""
    from src.data.data_loader_gss import create_cognitive_score

    df = pd.DataFrame({'WORDSUM': [0, 5, 10]})
    result = create_cognitive_score(df)
    assert result['cognitive_score'].iloc[0] == 0.0
    assert result['cognitive_score'].iloc[1] == 50.0
    assert result['cognitive_score'].iloc[2] == 100.0


def test_create_screen_time():
    """Screen time should convert weekly to daily hours."""
    from src.data.data_loader_gss import create_screen_time

    df = pd.DataFrame({'WWWHRS': [7, 14, 0]})
    result = create_screen_time(df)
    assert abs(result['screen_hours_daily'].iloc[0] - 1.0) < 0.01
    assert abs(result['screen_hours_daily'].iloc[1] - 2.0) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_loader_gss.py -v`
Expected: FAIL

**Step 3: Write the implementation**

```python
# src/data/data_loader_gss.py
"""
GSS 2010-2022 Data Loader

Loads General Social Survey cumulative data for longitudinal trend analysis
of the SES-cognition gradient. Uses vocabulary test (WORDSUM) as cognitive
measure and internet hours (WWWHRS) as screen time proxy.

Target population: US adults ages 25-45, years 2010-2022.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GSS_COLUMNS = [
    'YEAR', 'AGE', 'SEX', 'RACE', 'HEALTH',
    'WORDSUM', 'REALINC', 'EDUC', 'DEGREE', 'PRESTG80',
    'WWWHRS', 'EMAILHRS',
]


def filter_gss(df: pd.DataFrame, min_year: int = 2010, max_year: int = 2022,
               min_age: int = 25, max_age: int = 45) -> pd.DataFrame:
    """Filter to target years and age range."""
    mask = (
        (df['YEAR'] >= min_year) & (df['YEAR'] <= max_year) &
        (df['AGE'] >= min_age) & (df['AGE'] <= max_age)
    )
    return df[mask].copy()


def create_cognitive_score(df: pd.DataFrame) -> pd.DataFrame:
    """Scale WORDSUM (0-10) to 0-100."""
    df = df.copy()
    if 'WORDSUM' in df.columns:
        df['cognitive_score'] = (df['WORDSUM'] / 10) * 100
        # Treat out-of-range as missing
        df.loc[~df['WORDSUM'].between(0, 10), 'cognitive_score'] = np.nan
    else:
        df['cognitive_score'] = np.nan
    return df


def create_ses_index(df: pd.DataFrame, income_w: float = 0.5,
                     educ_w: float = 0.3, prestige_w: float = 0.2) -> pd.DataFrame:
    """Create 3-component SES index (income 50%, education 30%, prestige 20%)."""
    df = df.copy()

    for col, weight_name in [('REALINC', 'income_norm'), ('EDUC', 'educ_norm'), ('PRESTG80', 'prestige_norm')]:
        if col in df.columns:
            col_min, col_max = df[col].min(), df[col].max()
            if col_max > col_min:
                df[weight_name] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[weight_name] = 0.5
        else:
            df[weight_name] = np.nan

    df['ses_index'] = (income_w * df['income_norm'] +
                       educ_w * df['educ_norm'] +
                       prestige_w * df['prestige_norm'])
    return df


def create_screen_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert weekly internet hours to daily."""
    df = df.copy()
    if 'WWWHRS' in df.columns:
        df['screen_hours_daily'] = df['WWWHRS'] / 7
        # Cap at reasonable maximum
        df.loc[df['screen_hours_daily'] > 18, 'screen_hours_daily'] = np.nan
    else:
        df['screen_hours_daily'] = np.nan
    return df


def load_gss(raw_path: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess GSS cumulative dataset.

    Args:
        raw_path: Path to GSS .dta Stata file
        output_path: Path to save processed CSV

    Returns:
        Processed DataFrame filtered to 2010-2022, ages 25-45
    """
    if not raw_path or not Path(raw_path).exists():
        raise FileNotFoundError(
            "GSS data not found. Download the cumulative Stata file from "
            "https://gss.norc.org/get-the-data/stata and pass its path."
        )

    import pyreadstat
    logger.info(f"Loading GSS from {raw_path}")
    df, meta = pyreadstat.read_dta(raw_path, usecols=GSS_COLUMNS)

    df = filter_gss(df)
    logger.info(f"After filtering (2010-2022, ages 25-45): {len(df)} rows")

    df = create_cognitive_score(df)
    df = create_ses_index(df)
    df = create_screen_time(df)

    # SES quartiles
    df['ses_quartile'] = pd.qcut(
        df['ses_index'], q=4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High']
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed GSS data to {output_path}")

    return df
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_data_loader_gss.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/data/data_loader_gss.py tests/test_data_loader_gss.py
git commit -m "feat: add GSS 2010-2022 data loader with vocabulary score, SES index, and screen time"
```

---

### Task 9: Build harmonization module

**Files:**
- Create: `src/data/harmonization.py`
- Test: `tests/test_harmonization.py`

**Step 1: Write the test**

```python
# tests/test_harmonization.py
"""Tests for cross-dataset harmonization."""
import pytest
import pandas as pd
import numpy as np


def test_z_standardize():
    """Z-standardization should produce mean~0, sd~1."""
    from src.data.harmonization import z_standardize

    s = pd.Series([10, 20, 30, 40, 50])
    result = z_standardize(s)
    assert abs(result.mean()) < 0.01
    assert abs(result.std() - 1.0) < 0.01


def test_harmonize_datasets():
    """Harmonization should produce combined DataFrame with dataset labels."""
    from src.data.harmonization import harmonize_datasets

    nhanes = pd.DataFrame({
        'cognitive_score': [0.5, -0.3, 0.1],
        'ses_index': [0.2, 0.5, 0.8],
    })
    gss = pd.DataFrame({
        'cognitive_score': [60, 70, 80],
        'ses_index': [0.3, 0.6, 0.9],
    })

    result = harmonize_datasets({'nhanes': nhanes, 'gss': gss})
    assert 'dataset' in result.columns
    assert 'cognitive_z' in result.columns
    assert 'ses_z' in result.columns
    assert len(result) == 6
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_harmonization.py -v`

**Step 3: Write the implementation**

```python
# src/data/harmonization.py
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
```

**Step 4: Run tests**

Run: `python -m pytest tests/test_harmonization.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/harmonization.py tests/test_harmonization.py
git commit -m "feat: add cross-dataset harmonization with z-score standardization"
```

---

### Task 10: Build download orchestrator

**Files:**
- Create: `src/data/download_all_datasets.py`

**Step 1: Write the orchestrator**

```python
# src/data/download_all_datasets.py
"""
Download and preprocess all three datasets.

Usage:
    python -m src.data.download_all_datasets [--cache-dir data/raw] [--output-dir data/processed]
"""

import argparse
import logging
import pandas as pd
from pathlib import Path

from src.data.data_loader_nhanes import load_nhanes
from src.data.harmonization import harmonize_datasets

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_codebook(datasets: dict, output_path: str):
    """Auto-generate data dictionary from processed datasets."""
    rows = []
    for name, df in datasets.items():
        for col in df.columns:
            rows.append({
                'dataset': name,
                'variable': col,
                'dtype': str(df[col].dtype),
                'n_valid': int(df[col].count()),
                'n_missing': int(df[col].isna().sum()),
                'pct_missing': round(df[col].isna().mean() * 100, 1),
                'n_unique': int(df[col].nunique()),
                'min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else None,
                'mean': round(df[col].mean(), 3) if pd.api.types.is_numeric_dtype(df[col]) else None,
            })

    codebook = pd.DataFrame(rows)
    codebook.to_csv(output_path, index=False)
    logger.info(f"Codebook saved to {output_path}")


def print_quality_report(name: str, df: pd.DataFrame):
    """Print data quality summary."""
    logger.info(f"\n{'='*60}")
    logger.info(f"{name.upper()} Quality Report")
    logger.info(f"{'='*60}")
    logger.info(f"  N = {len(df)}")
    logger.info(f"  Columns = {len(df.columns)}")

    missing = df.isnull().sum()
    high_missing = missing[missing > 0].sort_values(ascending=False)
    if len(high_missing) > 0:
        logger.info(f"  Variables with missing data:")
        for var, count in high_missing.head(10).items():
            pct = count / len(df) * 100
            logger.info(f"    {var}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Download all datasets")
    parser.add_argument('--cache-dir', default='data/raw', help='Cache for raw downloads')
    parser.add_argument('--output-dir', default='data/processed', help='Output directory')
    parser.add_argument('--brfss-path', default=None, help='Path to BRFSS XPT file (manual download)')
    parser.add_argument('--gss-path', default=None, help='Path to GSS .dta file (manual download)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {}

    # NHANES (can auto-download)
    logger.info("Loading NHANES 2013-2014...")
    nhanes = load_nhanes(
        cache_dir=args.cache_dir,
        output_path=str(output_dir / 'nhanes_cognitive.csv')
    )
    datasets['nhanes'] = nhanes
    print_quality_report('NHANES', nhanes)

    # BRFSS (requires manual download)
    if args.brfss_path:
        from src.data.data_loader_brfss import load_brfss
        logger.info("Loading BRFSS 2022...")
        brfss = load_brfss(
            raw_path=args.brfss_path,
            output_path=str(output_dir / 'brfss_cognitive.csv')
        )
        datasets['brfss'] = brfss
        print_quality_report('BRFSS', brfss)
    else:
        logger.warning("BRFSS skipped (provide --brfss-path to include)")

    # GSS (requires manual download)
    if args.gss_path:
        from src.data.data_loader_gss import load_gss
        logger.info("Loading GSS 2010-2022...")
        gss = load_gss(
            raw_path=args.gss_path,
            output_path=str(output_dir / 'gss_cognitive.csv')
        )
        datasets['gss'] = gss
        print_quality_report('GSS', gss)
    else:
        logger.warning("GSS skipped (provide --gss-path to include)")

    # Harmonize
    if len(datasets) > 1:
        logger.info("Harmonizing datasets...")
        combined = harmonize_datasets(datasets)
        combined.to_csv(output_dir / 'combined_harmonized.csv', index=False)
        logger.info(f"Combined dataset: {len(combined)} rows")

    # Codebook
    generate_codebook(datasets, str(output_dir.parent / 'codebook.csv'))

    logger.info("\nDone! Datasets saved to " + str(output_dir))


if __name__ == '__main__':
    main()
```

**Step 2: Commit**

```bash
git add src/data/download_all_datasets.py
git commit -m "feat: add dataset download orchestrator with codebook generation"
```

---

### Task 11: Create stub modules for future phases

**Files:**
- Create: `src/analysis/mediation_analysis.py` (stub)
- Create: `src/analysis/prediction_model.py` (stub)
- Create: `src/analysis/longitudinal_trends.py` (stub)
- Create: `src/analysis/sensitivity_analysis.py` (stub)
- Create: `src/simulation/counterfactual_simulator.py` (stub)
- Create: `src/simulation/cost_effectiveness.py` (stub)
- Create: `src/visualization/mediation_diagrams.py` (stub)
- Create: `src/visualization/causal_network_viz.py` (stub)
- Create: `src/utils/survey_weights.py` (stub)
- Create: `src/utils/evalue.py` (stub)

**Step 1: Write all stubs**

Each stub has the class/function signatures from the proposal with `raise NotImplementedError`. See individual files below.

**src/analysis/mediation_analysis.py:**
```python
"""
Baron & Kenny Mediation Analysis with Bootstrap CIs

Implements multi-mediator mediation analysis for decomposing the
SES -> Cognitive Function pathway into direct and indirect effects
through screen time, depression, healthcare access, and sleep.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_mediation_effects(
    data: pd.DataFrame,
    exposure: str,
    mediators: List[str],
    outcome: str,
    confounders: List[str],
) -> Dict:
    """
    Multi-mediator Baron & Kenny mediation analysis.

    Returns:
        Dict with total_effect, direct_effect, indirect_effects,
        proportion_mediated, and model objects.
    """
    raise NotImplementedError("Phase 4 implementation")


def bootstrap_mediation(
    data: pd.DataFrame,
    exposure: str,
    mediators: List[str],
    outcome: str,
    confounders: List[str],
    n_boot: int = 5000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Bias-corrected bootstrap CIs for indirect effects.

    Returns:
        Dict mapping mediator name to {point_estimate, ci_lower, ci_upper, p_value}.
    """
    raise NotImplementedError("Phase 4 implementation")


def sobel_test(a: float, b: float, se_a: float, se_b: float) -> Tuple[float, float]:
    """
    Sobel test for indirect effect significance.

    Returns:
        (z_score, p_value)
    """
    raise NotImplementedError("Phase 4 implementation")
```

**src/analysis/prediction_model.py:**
```python
"""
XGBoost + SHAP Prediction Model

Trains a non-linear model to predict cognitive scores and uses SHAP
to decompose predictions into feature contributions. Validates
Baron-Kenny mediation results with data-driven feature importance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class CognitivePredictionModel:
    """XGBoost model for cognitive score prediction with SHAP explanations."""

    def __init__(self):
        self.model = None
        self.shap_values = None

    def train(
        self,
        data: pd.DataFrame,
        target: str,
        features: List[str],
        confounders: List[str],
    ) -> Dict:
        """Train XGBoost and return performance metrics."""
        raise NotImplementedError("Phase 4b implementation")

    def compute_shap_values(self, data: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for all predictions."""
        raise NotImplementedError("Phase 4b implementation")

    def partial_dependence(
        self, feature: str, grid: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute partial dependence for dose-response curve."""
        raise NotImplementedError("Phase 4b implementation")

    def cross_validate(self, data: pd.DataFrame, k: int = 5) -> Dict:
        """K-fold cross-validation returning R², MAE, RMSE."""
        raise NotImplementedError("Phase 4b implementation")

    def predict_counterfactual(
        self, data: pd.DataFrame, intervention: Dict
    ) -> np.ndarray:
        """Predict outcomes under counterfactual intervention."""
        raise NotImplementedError("Phase 4b implementation")
```

**src/analysis/longitudinal_trends.py:**
```python
"""
GSS Longitudinal Trend Analysis

Tests whether the SES-cognition gradient has worsened over time
(2010 vs 2022) using interaction models.
"""

import pandas as pd
from typing import Dict, List


def test_gradient_change(data: pd.DataFrame, years: List[int] = None) -> Dict:
    """Test SES x Year interaction for gradient change over time."""
    raise NotImplementedError("Phase 4 implementation")


def plot_temporal_trend(data: pd.DataFrame, variable: str = 'screen_hours_daily') -> None:
    """Plot variable over time stratified by SES quartile."""
    raise NotImplementedError("Phase 4 implementation")
```

**src/analysis/sensitivity_analysis.py:**
```python
"""
Sensitivity Analysis for Unmeasured Confounding

E-value computation, alternative SES specifications, missing data
comparison, and cognitive composite robustness checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def compute_evalue(estimate: float, se: float, ci_lower: float, ci_upper: float) -> Dict:
    """Compute E-value for point estimate and CI bound."""
    raise NotImplementedError("Phase 6 implementation")


def run_alternative_specifications(data: pd.DataFrame, specs: dict) -> Dict:
    """Rerun analysis with alternative variable specifications."""
    raise NotImplementedError("Phase 6 implementation")


def compare_missing_data_strategies(data: pd.DataFrame) -> Dict:
    """Compare complete-case vs MICE vs FIML results."""
    raise NotImplementedError("Phase 6 implementation")
```

**src/simulation/counterfactual_simulator.py:**
```python
"""
Counterfactual Intervention Simulator

Simulates policy interventions (screen time caps, stress reduction,
universal insurance) by modifying mediator values and predicting
cognitive outcomes using the trained XGBoost model.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class InterventionSimulator:
    """Simulate counterfactual outcomes under health interventions."""

    def __init__(self, prediction_model, data: pd.DataFrame):
        self.model = prediction_model
        self.data = data.copy()

    def simulate_intervention(self, intervention_type: str, parameters: Dict) -> Dict:
        """
        Apply intervention and predict outcomes.

        Intervention types: screen_cap, stress_reduction, universal_insurance, combined.
        """
        raise NotImplementedError("Phase 5 implementation")

    def calculate_cost_effectiveness(
        self, intervention_results: Dict, cost_per_capita: float
    ) -> Dict:
        """Compute cases prevented, cost per case, cost per QALY."""
        raise NotImplementedError("Phase 5 implementation")
```

**src/simulation/cost_effectiveness.py:**
```python
"""Cost-effectiveness analysis utilities."""


def compute_cost_per_qaly(total_cost: float, cases_prevented: int, qalys_per_case: float = 10.0) -> float:
    """Compute cost per quality-adjusted life year."""
    raise NotImplementedError("Phase 5 implementation")


def pareto_frontier(interventions: list) -> list:
    """Identify Pareto-optimal interventions (cost vs cases prevented)."""
    raise NotImplementedError("Phase 5 implementation")
```

**src/visualization/mediation_diagrams.py:**
```python
"""Mediation path diagram visualization."""


def plot_mediation_diagram(mediation_results: dict, output_path: str = None):
    """Generate publication-ready mediation path diagram."""
    raise NotImplementedError("Phase 5 visualization")
```

**src/visualization/causal_network_viz.py:**
```python
"""Interactive causal DAG visualization."""


def plot_causal_dag(edges: list, bootstrap_probs: dict = None, output_path: str = None):
    """Generate interactive causal network graph."""
    raise NotImplementedError("Phase 3 visualization")
```

**src/utils/survey_weights.py:**
```python
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
```

**src/utils/evalue.py:**
```python
"""E-value computation for sensitivity to unmeasured confounding."""

from typing import Dict


def evalue_ols(estimate: float, se: float, ci_lower: float, ci_upper: float) -> Dict:
    """E-value for OLS regression estimate (continuous outcome)."""
    raise NotImplementedError("Sensitivity analysis implementation")


def evalue_rr(rr: float, ci_lower: float, ci_upper: float) -> Dict:
    """E-value for risk ratio (binary outcome)."""
    raise NotImplementedError("Sensitivity analysis implementation")
```

**Step 2: Commit all stubs**

```bash
git add src/analysis/ src/simulation/ src/visualization/ src/utils/
git commit -m "feat: add stub modules for mediation, prediction, simulation, and sensitivity analysis"
```

---

### Task 12: Update prompts.yaml for social epidemiology

**Files:**
- Modify: `config/prompts_v1.yaml` -> rewrite as `config/prompts.yaml`

**Step 1: Write new prompts config**

```yaml
# Prompt Configuration for Cognitive Inequality Research
# 4-Tier LLM Architecture adapted for social epidemiology
# Version: 2.0

# ==============================================================================
# TIER 1: Causal Edge Validation (Social Epidemiology)
# ==============================================================================

causal_refinement:
  version: "2.0"
  temperature: 0.3
  system_instruction: |
    You are a social epidemiologist with expertise in causal inference and
    health disparities. Evaluate the plausibility of causal relationships
    between socioeconomic factors and cognitive health in young adults.

  task_description: |
    Validate causal edges discovered by PC algorithm using social epidemiology
    domain knowledge and data correlation evidence. Focus on SES -> mediator ->
    cognitive function pathways.


# ==============================================================================
# TIER 2: Impact Interpretation (Cognitive Decline Narratives)
# ==============================================================================

impact_interpretation:
  version: "2.0"
  temperature: 0.7
  system_instruction: |
    You are a science communicator specializing in translating statistical
    findings into concrete public health impacts. Make research about
    cognitive inequality accessible to policymakers and the general public.

  translation_requirements:
    focus: "Real-world cognitive impact, not statistics"
    required_elements:
      - specific_impact: "IQ points lost, cases per 100K, years of cognitive aging"
      - affected_population: "Income quartile, age group, demographic"
      - modifiable_factor: "Which behavior or policy could change this"
    constraints:
      - "No statistical jargon"
      - "Under 100 words"
      - "Concrete outcomes only"


# ==============================================================================
# TIER 3: Intervention Feasibility (Policy Assessment)
# ==============================================================================

intervention_feasibility:
  version: "2.0"
  temperature: 0.3
  system_instruction: |
    You are a public health policy analyst evaluating intervention proposals.
    Consider political feasibility, implementation logistics, equity impacts,
    and unintended consequences.

  evaluation_criteria:
    political_feasibility:
      description: "Likelihood of policy adoption given current political climate"
      score_range: [0.0, 1.0]

    implementation_barriers:
      description: "Logistical and technical challenges"
      max_items: 3

    equity_analysis:
      description: "Who benefits, who might be harmed"

    unintended_consequences:
      description: "Behavioral substitution, perverse incentives"

  deployment_recommendations:
    recommended: "Strong evidence, feasible, equitable"
    conditional: "Promising but needs pilot study or conditions"
    not_recommended: "Insufficient evidence, high cost, equity concerns"


# ==============================================================================
# General Configuration
# ==============================================================================

general:
  model: "gemini-2.0-flash"
  max_retries: 3
  cache_enabled: true
  logging_enabled: true

  output_format:
    json_schema_enforcement: true
    markdown_cleaning: true

  safety_constraints:
    no_pii_exposure: true
    evidence_based_claims_only: true
    acknowledge_limitations: true
```

**Step 2: Remove old prompts file**

```bash
rm -f config/prompts_v1.yaml
```

**Step 3: Commit**

```bash
git add config/prompts.yaml
git rm -f config/prompts_v1.yaml 2>/dev/null; true
git commit -m "feat: rewrite LLM prompts for social epidemiology domain"
```

---

### Task 13: Rewrite main.py pipeline orchestrator

**Files:**
- Modify: `src/main.py`

**Step 1: Rewrite main.py**

```python
"""
Cognitive Inequality Research System - Main Entry Point

Pipeline for analyzing causal pathways from socioeconomic inequality
to cognitive decline in young adults (ages 25-45).

Usage:
    python -m src.main download [--cache-dir data/raw]
    python -m src.main analyze --dataset nhanes
    python -m src.main pipeline --config config/api_keys.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CognitiveInequalityPipeline:
    """Orchestrates the cognitive inequality analysis pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path) if config_path else {}
        self.datasets = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def load_data(self, dataset_name: str, path: Optional[str] = None) -> pd.DataFrame:
        """Load a processed dataset."""
        if path and Path(path).exists():
            df = pd.read_csv(path)
        else:
            processed = Path('data/processed') / f'{dataset_name}_cognitive.csv'
            if processed.exists():
                df = pd.read_csv(processed)
            else:
                raise FileNotFoundError(
                    f"Dataset {dataset_name} not found. Run 'download' first."
                )

        self.datasets[dataset_name] = df
        logger.info(f"Loaded {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
        return df

    def run_descriptive(self, dataset_name: str = 'nhanes') -> Dict:
        """Run descriptive statistics (Phase 2)."""
        df = self.datasets.get(dataset_name)
        if df is None:
            df = self.load_data(dataset_name)

        logger.info("=" * 70)
        logger.info(f"DESCRIPTIVE STATISTICS - {dataset_name.upper()}")
        logger.info("=" * 70)

        stats = {
            'n': len(df),
            'age_mean': df['RIDAGEYR'].mean() if 'RIDAGEYR' in df.columns else None,
            'age_sd': df['RIDAGEYR'].std() if 'RIDAGEYR' in df.columns else None,
            'cognitive_mean': df['cognitive_score'].mean() if 'cognitive_score' in df.columns else None,
            'cognitive_sd': df['cognitive_score'].std() if 'cognitive_score' in df.columns else None,
            'ses_mean': df['ses_index'].mean() if 'ses_index' in df.columns else None,
            'missing_pct': df.isnull().mean().to_dict(),
        }

        for key, val in stats.items():
            if key != 'missing_pct':
                logger.info(f"  {key}: {val}")

        return stats

    def run_causal_discovery(self, dataset_name: str = 'nhanes') -> Dict:
        """Run PC algorithm causal discovery (Phase 3)."""
        logger.info("=" * 70)
        logger.info("CAUSAL DISCOVERY")
        logger.info("=" * 70)
        logger.info("Not yet implemented - use src.analysis.pc_algorithm_social")
        return {}

    def run_mediation(self, dataset_name: str = 'nhanes') -> Dict:
        """Run Baron-Kenny mediation analysis (Phase 4)."""
        logger.info("=" * 70)
        logger.info("MEDIATION ANALYSIS")
        logger.info("=" * 70)
        logger.info("Not yet implemented - use src.analysis.mediation_analysis")
        return {}

    def run_full_pipeline(self):
        """Run the complete analysis pipeline."""
        logger.info("=" * 70)
        logger.info("COGNITIVE INEQUALITY RESEARCH - FULL PIPELINE")
        logger.info("=" * 70)

        # Phase 1-2: Data + Descriptive
        self.load_data('nhanes')
        self.run_descriptive('nhanes')

        # Phase 3: Causal Discovery (stub)
        self.run_causal_discovery('nhanes')

        # Phase 4: Mediation (stub)
        self.run_mediation('nhanes')

        logger.info("\nPIPELINE COMPLETE (stubs for phases 3-5)")


def main():
    parser = argparse.ArgumentParser(
        description="Cognitive Inequality Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Download command
    dl_parser = subparsers.add_parser('download', help='Download and preprocess datasets')
    dl_parser.add_argument('--cache-dir', default='data/raw')
    dl_parser.add_argument('--output-dir', default='data/processed')
    dl_parser.add_argument('--brfss-path', default=None)
    dl_parser.add_argument('--gss-path', default=None)

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run analysis on processed data')
    analyze_parser.add_argument('--dataset', default='nhanes', choices=['nhanes', 'brfss', 'gss'])
    analyze_parser.add_argument('--config', help='Path to config YAML')

    # Pipeline command
    pipe_parser = subparsers.add_parser('pipeline', help='Run full analysis pipeline')
    pipe_parser.add_argument('--config', help='Path to config YAML')

    args = parser.parse_args()

    if args.command == 'download':
        from src.data.download_all_datasets import main as download_main
        sys.argv = ['download', '--cache-dir', args.cache_dir, '--output-dir', args.output_dir]
        if args.brfss_path:
            sys.argv += ['--brfss-path', args.brfss_path]
        if args.gss_path:
            sys.argv += ['--gss-path', args.gss_path]
        download_main()

    elif args.command == 'analyze':
        pipeline = CognitiveInequalityPipeline(args.config)
        pipeline.load_data(args.dataset)
        pipeline.run_descriptive(args.dataset)

    elif args.command == 'pipeline':
        pipeline = CognitiveInequalityPipeline(args.config)
        pipeline.run_full_pipeline()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add src/main.py
git commit -m "feat: rewrite main.py as cognitive inequality pipeline orchestrator"
```

---

### Task 14: Clean up old intervention engine reference

**Files:**
- Delete: `src/simulation/intervention_engine_old.py`
- Delete: `src/analysis/causal_analysis.py` (old LLM-based causal analysis, replaced by pc_algorithm_social)

**Step 1: Remove old files**

```bash
rm -f src/simulation/intervention_engine_old.py
rm -f src/analysis/causal_analysis.py
```

**Step 2: Commit**

```bash
git add -A
git commit -m "chore: remove old clinical fairness modules from new locations"
```

---

### Task 15: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Write new README**

```markdown
# Cognitive Inequality Research System

Investigates causal pathways from socioeconomic inequality to cognitive decline
in young adults (25-45) using NHANES, BRFSS, and GSS public health datasets.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download and preprocess NHANES (auto-downloads from CDC)
python -m src.data.download_all_datasets

# Run descriptive analysis
python -m src.main analyze --dataset nhanes

# Run full pipeline
python -m src.main pipeline
```

## Datasets

| Dataset | N (ages 25-45) | Cognitive Measure | Primary Use |
|---------|---------------|-------------------|-------------|
| NHANES 2013-2014 | ~2,500 | DSST + CERAD (validated) | Mediation analysis |
| BRFSS 2022 | ~100,000 | Self-reported difficulty | Validation |
| GSS 2010-2022 | ~4,000 | Vocabulary test | Longitudinal trends |

## Architecture

```
src/
├── data/        # Data loaders (NHANES, BRFSS, GSS) + harmonization
├── analysis/    # Mediation, causal discovery, prediction model
├── llm/         # 4-tier LLM validation (DeepSeek/Gemini)
├── simulation/  # Counterfactual intervention simulator
├── visualization/  # Publication-ready figures
└── utils/       # Bootstrap, survey weights, E-values
```

## License

MIT
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for cognitive inequality research project"
```

---

### Task 16: Run all tests and verify clean state

**Step 1: Run full test suite**

Run: `cd /home/bronson/2res/cognitive-inequality-research && python -m pytest tests/ -v`
Expected: All tests PASS (11 tests across 3 test files)

**Step 2: Verify imports work**

Run: `python -c "from src.data.data_loader_nhanes import load_nhanes; from src.data.data_loader_brfss import load_brfss; from src.data.data_loader_gss import load_gss; from src.data.harmonization import harmonize_datasets; print('All imports OK')"`
Expected: `All imports OK`

**Step 3: Verify no leftover clinical references**

Run: `grep -r "clinical" src/ --include="*.py" -l` (should find 0 or only in comments about provenance)
Run: `grep -r "fairness" src/ --include="*.py" -l` (should find 0)

**Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: verify clean pivot state, fix any remaining issues"
```
