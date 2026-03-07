# Pipeline Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add covariate adjustment, longitudinal validation (MIDUS 3 follow-up), sex/age-cohort
moderation, and publication-quality figures to the cognitive inequality pipeline.

**Architecture:** Layered additions — each phase is an independent module. Existing mediation code
already supports covariates. MIDUS 3 links to MR2 via `M2ID`. All figures call a single
`generate_all_figures(results)` entry point at the end of the pipeline.

**Tech Stack:** Python 3.11, statsmodels (OLS/WLS), matplotlib + seaborn (figures), pyreadstat
(SAV loading), pandas, numpy. Tests use pytest. No new dependencies needed.

---

### Task 1: Wire covariate adjustment into run_mediation

**Files:**
- Modify: `src/main.py:214-236`

**Step 1: Write the failing test**

Add to `tests/test_mediation_analysis.py`:

```python
def test_baron_kenny_with_covariates():
    """Covariates are passed through and reduce confounding."""
    import pandas as pd, numpy as np
    from src.analysis.mediation_analysis import baron_kenny_mediation
    rng = np.random.default_rng(0)
    n = 200
    age = rng.normal(44, 5, n)
    df = pd.DataFrame({
        'ses': rng.normal(0, 1, n),
        'mediator': rng.normal(0, 1, n),
        'outcome': rng.normal(0, 1, n),
        'age': age,
        'female': rng.integers(0, 2, n).astype(float),
    })
    result = baron_kenny_mediation(df, x='ses', m='mediator', y='outcome',
                                   covariates=['age', 'female'])
    assert hasattr(result, 'indirect')
```

**Step 2: Run test to verify it passes already** (covariates param exists)

```bash
cd /home/bronson/2res/cognitive-inequality-research
pytest tests/test_mediation_analysis.py::test_baron_kenny_with_covariates -v
```

Expected: PASS (covariates already supported in mediation_analysis.py)

**Step 3: Modify run_mediation in main.py**

In `src/main.py`, find the line:
```python
bk = baron_kenny_mediation(df, x='ses_index', m=m, y='cognitive_score',
                           weights=weights)
```
Change to:
```python
bk = baron_kenny_mediation(df, x='ses_index', m=m, y='cognitive_score',
                           covariates=['RB1PRAGE', 'female'], weights=weights)
```

Find the `analyze_all_mediators` call:
```python
boot_results = analyze_all_mediators(
    df, x='ses_index', y='cognitive_score',
    mediators=mediators, weights=weights, n_boot=n_bootstrap,
)
```
Change to:
```python
boot_results = analyze_all_mediators(
    df, x='ses_index', y='cognitive_score',
    mediators=mediators, covariates=['RB1PRAGE', 'female'],
    weights=weights, n_boot=n_bootstrap,
)
```

**Step 4: Verify pipeline loads without error**

```bash
python -c "
from src.main import CognitiveInequalityPipeline
p = CognitiveInequalityPipeline()
df = p.load_data('midus_mr2')
print('female' in df.columns, 'RB1PRAGE' in df.columns)
"
```

Expected: `True True`

**Step 5: Commit**

```bash
git add src/main.py
git commit -m "feat: add age and sex covariates to mediation analysis"
```

---

### Task 2: Build MIDUS 3 data loader

**Files:**
- Create: `src/data/data_loader_midus_m3.py`
- Create: `tests/test_data_loader_midus_m3.py`

**Context:** MIDUS 3 uses prefix `C1` for survey variables and `C3` for BTACT.
- Survey SAV: `M3_P1_SURVEY_N3294_20251029.sav` — variables: `M2ID`, `C1PRAGE`, `C1SRINC`, `C1STINC`
- BTACT SAV: `M3_P3_BTACT_N3291_20210922.sav` — variables: `M2ID`, `C3TCOMP`
- Both files are in the project root (not in `data/`).

**Step 1: Write the failing test**

```python
# tests/test_data_loader_midus_m3.py
import pytest
from pathlib import Path

SURVEY_SAV = Path('M3_P1_SURVEY_N3294_20251029.sav')
BTACT_SAV  = Path('M3_P3_BTACT_N3291_20210922.sav')

@pytest.mark.skipif(not SURVEY_SAV.exists(), reason="M3 SAV not present")
def test_load_midus_m3_shape():
    from src.data.data_loader_midus_m3 import load_midus_m3
    df = load_midus_m3()
    assert 'M2ID' in df.columns
    assert 'C1PRAGE' in df.columns
    assert 'ses_index_m3' in df.columns
    assert 'cognitive_score_m3' in df.columns
    assert len(df) > 500

@pytest.mark.skipif(not SURVEY_SAV.exists(), reason="M3 SAV not present")
def test_load_midus_m3_no_allnan():
    from src.data.data_loader_midus_m3 import load_midus_m3
    df = load_midus_m3()
    assert df['cognitive_score_m3'].notna().sum() > 400
    assert df['ses_index_m3'].notna().sum() > 400
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_data_loader_midus_m3.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the loader**

```python
# src/data/data_loader_midus_m3.py
"""MIDUS 3 data loader — survey + BTACT cognitive battery."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

logger = logging.getLogger(__name__)

SURVEY_SAV = 'M3_P1_SURVEY_N3294_20251029.sav'
BTACT_SAV  = 'M3_P3_BTACT_N3291_20210922.sav'

SURVEY_COLS = ['M2ID', 'C1PRAGE', 'C1SRINC', 'C1STINC', 'C1SEX']
BTACT_COLS  = ['M2ID', 'C3TCOMP']


def _find_sav(filename: str) -> Path:
    """Search project root and data/raw for SAV file."""
    for base in [Path('.'), Path('data/raw')]:
        p = base / filename
        if p.exists():
            return p
    raise FileNotFoundError(f"{filename} not found in . or data/raw/")


def load_midus_m3(survey_path: str = None, btact_path: str = None) -> pd.DataFrame:
    """
    Load MIDUS 3 survey + BTACT, return merged DataFrame.

    Columns returned:
        M2ID, C1PRAGE, female, ses_index_m3, cognitive_score_m3
    """
    survey_path = Path(survey_path) if survey_path else _find_sav(SURVEY_SAV)
    btact_path  = Path(btact_path)  if btact_path  else _find_sav(BTACT_SAV)

    logger.info(f"Loading M3 survey from {survey_path}")
    survey, _ = pyreadstat.read_sav(str(survey_path), usecols=SURVEY_COLS)
    survey = pd.DataFrame(survey)

    logger.info(f"Loading M3 BTACT from {btact_path}")
    btact, _ = pyreadstat.read_sav(str(btact_path), usecols=BTACT_COLS)
    btact = pd.DataFrame(btact)

    df = survey.merge(btact, on='M2ID', how='inner')
    logger.info(f"Merged M3 N={len(df)}")

    # Sex: 1=male 2=female in MIDUS coding
    df['female'] = (df['C1SEX'] == 2).astype(float)

    # SES index: log income (use total household income C1STINC, fallback to C1SRINC)
    income = df['C1STINC'].where(df['C1STINC'] > 0, df['C1SRINC'])
    income = income.where(income > 0)
    log_income = np.log1p(income)
    min_i, max_i = log_income.min(), log_income.max()
    df['ses_index_m3'] = (log_income - min_i) / (max_i - min_i) if max_i > min_i else 0.5

    # Cognitive score: z-score C3TCOMP
    cog = df['C3TCOMP'].where(df['C3TCOMP'] > 0)
    df['cognitive_score_m3'] = (cog - cog.mean()) / cog.std()

    final_cols = ['M2ID', 'C1PRAGE', 'female', 'ses_index_m3', 'cognitive_score_m3']
    return df[final_cols].copy()
```

**Step 4: Run tests**

```bash
pytest tests/test_data_loader_midus_m3.py -v
```

Expected: both tests PASS

**Step 5: Commit**

```bash
git add src/data/data_loader_midus_m3.py tests/test_data_loader_midus_m3.py
git commit -m "feat: add MIDUS 3 data loader for longitudinal follow-up"
```

---

### Task 3: Build longitudinal analysis module

**Files:**
- Create: `src/analysis/longitudinal_analysis.py`
- Create: `tests/test_longitudinal_analysis.py`

**Context:** Merges MR2 baseline (already loaded by pipeline) with M3 follow-up on `M2ID`.
Computes `cognitive_change = cognitive_score_m3 - cognitive_score` (both already z-scored).
Runs OLS: `cognitive_change ~ ses_index + RB1PRAGE + female`.

**Step 1: Write the failing tests**

```python
# tests/test_longitudinal_analysis.py
import numpy as np
import pandas as pd
import pytest
from src.analysis.longitudinal_analysis import (
    merge_baseline_followup,
    run_longitudinal_regression,
)


def _make_panel(n=200, ses_effect=0.3):
    rng = np.random.default_rng(42)
    m2id = np.arange(n)
    ses = rng.uniform(0, 1, n)
    baseline_cog = ses * ses_effect + rng.normal(0, 0.8, n)
    followup_cog = baseline_cog + ses * ses_effect + rng.normal(0, 0.5, n)
    mr2 = pd.DataFrame({
        'M2ID': m2id, 'ses_index': ses,
        'cognitive_score': baseline_cog,
        'RB1PRAGE': rng.uniform(34, 55, n),
        'female': rng.integers(0, 2, n).astype(float),
    })
    m3 = pd.DataFrame({
        'M2ID': m2id, 'cognitive_score_m3': followup_cog,
        'ses_index_m3': ses + rng.normal(0, 0.05, n),
    })
    return mr2, m3


def test_merge_baseline_followup():
    mr2, m3 = _make_panel()
    panel = merge_baseline_followup(mr2, m3)
    assert 'cognitive_change' in panel.columns
    assert len(panel) == len(mr2)
    assert panel['cognitive_change'].notna().sum() > 150


def test_longitudinal_regression_ses_positive():
    mr2, m3 = _make_panel(ses_effect=0.5)
    panel = merge_baseline_followup(mr2, m3)
    result = run_longitudinal_regression(panel)
    assert result['ses_coef'] > 0
    assert 'ses_pvalue' in result
    assert 'n' in result
    assert result['n'] > 150
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_longitudinal_analysis.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the module**

```python
# src/analysis/longitudinal_analysis.py
"""Longitudinal analysis: MR2 baseline → MIDUS 3 follow-up cognitive change."""

import logging
from typing import Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def merge_baseline_followup(mr2_df: pd.DataFrame, m3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge MR2 baseline with M3 follow-up on M2ID.

    Computes cognitive_change = cognitive_score_m3 - cognitive_score (both z-scored).
    Returns merged panel with baseline SES, covariates, and change score.
    """
    panel = mr2_df.merge(m3_df[['M2ID', 'cognitive_score_m3']], on='M2ID', how='inner')
    panel['cognitive_change'] = panel['cognitive_score_m3'] - panel['cognitive_score']
    logger.info(f"Panel N={len(panel)}, cognitive_change mean={panel['cognitive_change'].mean():.3f}")
    return panel


def run_longitudinal_regression(panel: pd.DataFrame,
                                 x: str = 'ses_index',
                                 y: str = 'cognitive_change',
                                 covariates: list = None) -> Dict:
    """
    OLS regression: cognitive_change ~ ses_index + covariates.

    Returns dict with ses_coef, ses_pvalue, ses_ci_lower, ses_ci_upper, r2, n.
    """
    if covariates is None:
        covariates = ['RB1PRAGE', 'female']

    cols = [x, y] + [c for c in covariates if c in panel.columns]
    data = panel[cols].dropna()

    X = sm.add_constant(data[[x] + [c for c in covariates if c in data.columns]])
    model = sm.OLS(data[y], X).fit()

    return {
        'ses_coef':     model.params[x],
        'ses_pvalue':   model.pvalues[x],
        'ses_ci_lower': model.conf_int().loc[x, 0],
        'ses_ci_upper': model.conf_int().loc[x, 1],
        'r2':           model.rsquared,
        'n':            int(len(data)),
        'model':        model,
    }
```

**Step 4: Run tests**

```bash
pytest tests/test_longitudinal_analysis.py -v
```

Expected: both PASS

**Step 5: Commit**

```bash
git add src/analysis/longitudinal_analysis.py tests/test_longitudinal_analysis.py
git commit -m "feat: add longitudinal analysis module for MR2→M3 cognitive change"
```

---

### Task 4: Wire run_longitudinal into the pipeline

**Files:**
- Modify: `src/main.py`

**Step 1: Add run_longitudinal method**

After `run_sensitivity` in `src/main.py`, add:

```python
# ------------------------------------------------------------------
# Phase 7: Longitudinal validation (MR2 baseline → MIDUS 3)
# ------------------------------------------------------------------

def run_longitudinal(self) -> Dict:
    """Test whether baseline SES predicts cognitive change to MIDUS 3."""
    from src.data.data_loader_midus_m3 import load_midus_m3
    from src.analysis.longitudinal_analysis import (
        merge_baseline_followup, run_longitudinal_regression,
    )

    mr2_df = self.datasets.get('midus_mr2')
    if mr2_df is None:
        mr2_df = self.load_data('midus_mr2')

    logger.info("=" * 70)
    logger.info("LONGITUDINAL VALIDATION — MR2 BASELINE → MIDUS 3 FOLLOW-UP")
    logger.info("=" * 70)

    try:
        m3_df = load_midus_m3()
    except FileNotFoundError as e:
        logger.warning(f"  MIDUS 3 files not found — skipping longitudinal: {e}")
        return {}

    panel = merge_baseline_followup(mr2_df, m3_df)
    result = run_longitudinal_regression(panel)

    logger.info(f"  N panel: {result['n']}")
    logger.info(f"  SES → cognitive_change: β={result['ses_coef']:.4f} "
                f"[{result['ses_ci_lower']:.4f}, {result['ses_ci_upper']:.4f}] "
                f"p={result['ses_pvalue']:.4f}")
    logger.info(f"  R²={result['r2']:.3f}")

    self.results['longitudinal'] = result
    return result
```

**Step 2: Call run_longitudinal in run_full_pipeline**

Find the line `self.run_sensitivity(dataset_name)` and add after it:

```python
self.run_longitudinal()
```

**Step 3: Verify pipeline runs end-to-end**

```bash
python -c "
from src.main import CognitiveInequalityPipeline
p = CognitiveInequalityPipeline()
p.load_data('midus_mr2')
r = p.run_longitudinal()
print('longitudinal keys:', list(r.keys()))
print('SES coef:', r.get('ses_coef'))
"
```

Expected: prints ses_coef value (positive expected)

**Step 4: Commit**

```bash
git add src/main.py
git commit -m "feat: add longitudinal validation phase to pipeline"
```

---

### Task 5: Build moderation analysis module

**Files:**
- Create: `src/analysis/moderation_analysis.py`
- Create: `tests/test_moderation_analysis.py`

**Step 1: Write the failing tests**

```python
# tests/test_moderation_analysis.py
import numpy as np
import pandas as pd
import pytest
from src.analysis.moderation_analysis import (
    run_interaction_model,
    run_subgroup_mediation,
)


def _make_df(n=400):
    rng = np.random.default_rng(0)
    ses = rng.uniform(0, 1, n)
    age = rng.uniform(34, 55, n)
    female = rng.integers(0, 2, n).astype(float)
    mediator = 0.3 * ses + rng.normal(0, 0.5, n)
    cog = 0.5 * ses + 0.2 * mediator - 0.1 * female + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        'ses_index': ses, 'RB1PRAGE': age, 'female': female,
        'mediator': mediator, 'cognitive_score': cog,
    })


def test_interaction_model_sex():
    df = _make_df()
    result = run_interaction_model(df, moderator='female')
    assert 'interaction_coef' in result
    assert 'interaction_pvalue' in result
    assert 'n' in result


def test_interaction_model_age_cohort():
    df = _make_df()
    result = run_interaction_model(df, moderator='age_cohort')
    assert 'interaction_coef' in result


def test_subgroup_mediation_sex():
    df = _make_df()
    result = run_subgroup_mediation(df, moderator='female',
                                    mediators=['mediator'])
    assert 0 in result and 1 in result
    assert 'mediator' in result[0]


def test_subgroup_mediation_age_cohort():
    df = _make_df()
    result = run_subgroup_mediation(df, moderator='age_cohort',
                                    mediators=['mediator'])
    assert 'younger' in result and 'older' in result
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_moderation_analysis.py -v
```

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the module**

```python
# src/analysis/moderation_analysis.py
"""Moderation analysis: does the SES→cognition gap vary by sex or age cohort?"""

import logging
from typing import Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.analysis.mediation_analysis import bootstrap_mediation

logger = logging.getLogger(__name__)


def run_interaction_model(df: pd.DataFrame,
                           moderator: str,
                           x: str = 'ses_index',
                           y: str = 'cognitive_score') -> Dict:
    """
    Fit OLS with interaction term: y ~ x * moderator + covariates.

    moderator='female'      → uses female column directly
    moderator='age_cohort'  → creates binary 0/1 split at median age
    """
    data = df.copy()

    if moderator == 'age_cohort':
        median_age = data['RB1PRAGE'].median()
        data['age_cohort'] = (data['RB1PRAGE'] > median_age).astype(float)
        mod_col = 'age_cohort'
        covariates = ['female']
    else:
        mod_col = moderator
        covariates = ['RB1PRAGE']

    cols = [x, y, mod_col] + covariates
    data = data[cols].dropna()

    data['interaction'] = data[x] * data[mod_col]
    predictors = [x, mod_col, 'interaction'] + covariates
    X = sm.add_constant(data[predictors])
    model = sm.OLS(data[y], X).fit()

    return {
        'interaction_coef':   model.params['interaction'],
        'interaction_pvalue': model.pvalues['interaction'],
        'interaction_ci':     tuple(model.conf_int().loc['interaction']),
        'n':                  int(len(data)),
        'model':              model,
    }


def run_subgroup_mediation(df: pd.DataFrame,
                            moderator: str,
                            mediators: List[str],
                            x: str = 'ses_index',
                            y: str = 'cognitive_score',
                            n_boot: int = 500) -> Dict:
    """
    Run bootstrap mediation separately per subgroup.

    moderator='female'     → groups: {0: male, 1: female}
    moderator='age_cohort' → groups: {'younger': age<=median, 'older': age>median}

    Returns dict of {group_label: {mediator: BootstrapResult}}
    """
    data = df.copy()

    if moderator == 'age_cohort':
        median_age = data['RB1PRAGE'].median()
        data['_group'] = np.where(data['RB1PRAGE'] <= median_age, 'younger', 'older')
        group_labels = ['younger', 'older']
    else:
        data['_group'] = data[moderator]
        group_labels = sorted(data['_group'].dropna().unique())

    results = {}
    for label in group_labels:
        subset = data[data['_group'] == label].copy()
        logger.info(f"  Subgroup {moderator}={label}, N={len(subset)}")
        group_results = {}
        for m in mediators:
            if m not in subset.columns:
                continue
            try:
                boot = bootstrap_mediation(subset, x=x, m=m, y=y, n_boot=n_boot)
                group_results[m] = boot
            except Exception as e:
                logger.warning(f"  bootstrap failed for {m} in group {label}: {e}")
        results[label] = group_results

    return results
```

**Step 4: Run tests**

```bash
pytest tests/test_moderation_analysis.py -v
```

Expected: all 4 PASS

**Step 5: Commit**

```bash
git add src/analysis/moderation_analysis.py tests/test_moderation_analysis.py
git commit -m "feat: add sex and age-cohort moderation analysis module"
```

---

### Task 6: Wire run_moderation into the pipeline

**Files:**
- Modify: `src/main.py`

**Step 1: Add run_moderation method**

After `run_longitudinal` in `src/main.py`, add:

```python
# ------------------------------------------------------------------
# Phase 8: Moderation / subgroup analysis
# ------------------------------------------------------------------

def run_moderation(self, dataset_name: str = 'midus_mr2') -> Dict:
    """Sex and age-cohort moderation of the SES→cognition indirect effects."""
    from src.analysis.moderation_analysis import (
        run_interaction_model, run_subgroup_mediation,
    )

    df = self.datasets.get(dataset_name)
    if df is None:
        df = self.load_data(dataset_name)
    mediators = [m for m in MR2_MEDIATORS if m in df.columns]

    logger.info("=" * 70)
    logger.info("MODERATION ANALYSIS — SEX AND AGE COHORT")
    logger.info("=" * 70)

    results = {}
    for moderator in ['female', 'age_cohort']:
        logger.info(f"\n  Moderator: {moderator}")

        interaction = run_interaction_model(df, moderator=moderator)
        logger.info(f"    Interaction β={interaction['interaction_coef']:.4f} "
                    f"p={interaction['interaction_pvalue']:.4f}")

        subgroup = run_subgroup_mediation(df, moderator=moderator,
                                          mediators=mediators, n_boot=500)
        results[moderator] = {
            'interaction': interaction,
            'subgroup': subgroup,
        }

    self.results['moderation'] = results
    return results
```

**Step 2: Call run_moderation in run_full_pipeline**

Add after `self.run_longitudinal()`:

```python
self.run_moderation(dataset_name)
```

**Step 3: Quick smoke test**

```bash
python -c "
from src.main import CognitiveInequalityPipeline
p = CognitiveInequalityPipeline()
p.load_data('midus_mr2')
r = p.run_moderation()
for mod, res in r.items():
    print(mod, 'interaction p=', round(res['interaction']['interaction_pvalue'], 4))
"
```

**Step 4: Commit**

```bash
git add src/main.py
git commit -m "feat: add moderation phase to pipeline"
```

---

### Task 7: Build visualization module — mediation forest plot

**Files:**
- Create: `src/visualization/__init__.py`
- Create: `src/visualization/figure_mediation.py`
- Create: `tests/test_visualization.py`

**Step 1: Write the failing test**

```python
# tests/test_visualization.py
import numpy as np
import pytest
from unittest.mock import MagicMock
from src.analysis.mediation_analysis import BootstrapResult


def _make_boot_results(n=7):
    names = [f'mediator_{i}' for i in range(n)]
    results = {}
    for i, name in enumerate(names):
        lo = -0.1 + i * 0.05
        hi = lo + 0.15
        results[name] = BootstrapResult(
            point_estimate=(lo + hi) / 2,
            ci_lower=lo, ci_upper=hi, n_boot=1000,
        )
    return results


def test_plot_mediation_forest_returns_figure():
    import matplotlib
    matplotlib.use('Agg')
    from src.visualization.figure_mediation import plot_mediation_forest
    boot_results = _make_boot_results()
    significant = ['mediator_3', 'mediator_4', 'mediator_5']
    fig = plot_mediation_forest(boot_results, significant)
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close('all')
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_visualization.py::test_plot_mediation_forest_returns_figure -v
```

**Step 3: Implement figure_mediation.py**

```python
# src/visualization/figure_mediation.py
"""Forest plot of mediation indirect effects with bootstrap CIs."""

from typing import Dict, List
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_mediation_forest(boot_results: Dict,
                          significant: List[str],
                          title: str = 'Indirect Effects: SES → Mediator → Cognition') -> plt.Figure:
    """
    Horizontal forest plot of bootstrap indirect effects.

    Significant mediators shown in dark gray; non-significant in light gray.
    Returns Figure (does not save or show).
    """
    import seaborn as sns
    sns.set_theme(style='whitegrid', font_scale=1.2)

    names = list(boot_results.keys())
    points = [boot_results[n].point_estimate for n in names]
    lowers = [boot_results[n].ci_lower for n in names]
    uppers = [boot_results[n].ci_upper for n in names]
    errors_lo = [p - l for p, l in zip(points, lowers)]
    errors_hi = [u - p for p, u in zip(points, uppers)]

    colors = ['#222222' if n in significant else '#aaaaaa' for n in names]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.7)))
    y_pos = np.arange(len(names))

    ax.barh(y_pos, points, xerr=[errors_lo, errors_hi],
            color=colors, edgecolor='white', height=0.5,
            error_kw=dict(ecolor='#555555', capsize=4, linewidth=1.5))

    ax.axvline(0, color='black', linewidth=1.0, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n.replace('_', ' ').title() for n in names])
    ax.set_xlabel('Indirect Effect (a × b)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    sig_patch = mpatches.Patch(color='#222222', label='Significant (95% CI ≠ 0)')
    ns_patch  = mpatches.Patch(color='#aaaaaa', label='Non-significant')
    ax.legend(handles=[sig_patch, ns_patch], loc='lower right', fontsize=10)

    n_boot = list(boot_results.values())[0].n_boot
    ax.text(0.01, -0.12, f'Bootstrap N={n_boot:,}', transform=ax.transAxes,
            fontsize=9, color='#666666')

    fig.tight_layout()
    return fig
```

**Step 4: Create `src/visualization/__init__.py`**

```python
# src/visualization/__init__.py
from .figure_mediation import plot_mediation_forest
```

**Step 5: Run test**

```bash
pytest tests/test_visualization.py::test_plot_mediation_forest_returns_figure -v
```

Expected: PASS

**Step 6: Commit**

```bash
git add src/visualization/ tests/test_visualization.py
git commit -m "feat: add mediation forest plot visualization"
```

---

### Task 8: Add longitudinal and moderation figures

**Files:**
- Create: `src/visualization/figure_longitudinal.py`
- Create: `src/visualization/figure_moderation.py`
- Modify: `src/visualization/__init__.py`

**Step 1: Implement figure_longitudinal.py**

```python
# src/visualization/figure_longitudinal.py
"""Scatter plot: baseline SES vs cognitive change, by sex."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_longitudinal_scatter(panel: pd.DataFrame,
                               result: dict,
                               x: str = 'ses_index',
                               y: str = 'cognitive_change') -> plt.Figure:
    """
    Scatter of baseline SES vs cognitive change.
    Points colored by sex. OLS line + 95% CI band.
    """
    import seaborn as sns
    sns.set_theme(style='whitegrid', font_scale=1.2)

    data = panel[[x, y, 'female']].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))

    for sex_val, label, color in [(0, 'Male', '#888888'), (1, 'Female', '#222222')]:
        sub = data[data['female'] == sex_val]
        ax.scatter(sub[x], sub[y], c=color, alpha=0.4, s=18, label=label)

    # Regression line with CI
    x_range = np.linspace(data[x].min(), data[x].max(), 100)
    coef = result['ses_coef']
    model = result['model']
    y_pred = model.predict(
        __import__('statsmodels').api.add_constant(
            pd.DataFrame({x: x_range, 'RB1PRAGE': data['RB1PRAGE'].mean(),
                          'female': 0.5})
        )
    )
    ax.plot(x_range, y_pred, color='black', linewidth=2)

    ax.set_xlabel('Baseline SES Index', fontsize=12)
    ax.set_ylabel('Cognitive Change (MR2 → M3, SD units)', fontsize=12)
    ax.set_title('Baseline SES Predicts Cognitive Change', fontsize=13, fontweight='bold')

    n = result['n']
    r2 = result['r2']
    p = result['ses_pvalue']
    ax.text(0.03, 0.95,
            f'β={coef:.3f}, p={p:.3f}, R²={r2:.3f}, N={n}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig
```

**Step 2: Implement figure_moderation.py**

```python
# src/visualization/figure_moderation.py
"""Two-panel figure: indirect effects by sex and age cohort subgroups."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict


def plot_moderation_comparison(moderation_results: Dict,
                                mediators: list) -> plt.Figure:
    """
    Two-panel figure: left=sex subgroups, right=age cohort subgroups.
    Shows indirect effect point estimates + CIs for each mediator per group.
    """
    import seaborn as sns
    sns.set_theme(style='whitegrid', font_scale=1.1)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(mediators) * 0.8)),
                              sharey=True)
    panel_configs = [
        ('female',     'Sex Moderation',      {0: 'Male', 1: 'Female'},     ['#aaaaaa', '#222222']),
        ('age_cohort', 'Age Cohort Moderation', {'younger': '34–44', 'older': '45–55'}, ['#aaaaaa', '#222222']),
    ]

    y_pos = np.arange(len(mediators))

    for ax, (moderator, title, label_map, colors) in zip(axes, panel_configs):
        if moderator not in moderation_results:
            ax.set_title(f'{title}\n(not run)', fontsize=11)
            continue

        subgroup = moderation_results[moderator]['subgroup']
        groups = list(subgroup.keys())
        width = 0.35
        offsets = [-width / 2, width / 2]

        for g_idx, (group, offset, color) in enumerate(zip(groups, offsets, colors)):
            group_results = subgroup[group]
            points, lo_err, hi_err = [], [], []
            for m in mediators:
                if m in group_results:
                    r = group_results[m]
                    points.append(r.point_estimate)
                    lo_err.append(r.point_estimate - r.ci_lower)
                    hi_err.append(r.ci_upper - r.point_estimate)
                else:
                    points.append(0); lo_err.append(0); hi_err.append(0)

            ax.barh(y_pos + offset, points, xerr=[lo_err, hi_err],
                    height=width, color=color, alpha=0.85,
                    error_kw=dict(ecolor='#444444', capsize=3),
                    label=label_map.get(group, str(group)))

        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Indirect Effect (a × b)', fontsize=11)
        ax.legend(fontsize=9)

    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([m.replace('_', ' ').title() for m in mediators], fontsize=10)
    fig.suptitle('Moderation of SES→Cognition Indirect Effects', fontsize=13,
                 fontweight='bold', y=1.02)
    fig.tight_layout()
    return fig
```

**Step 3: Update `src/visualization/__init__.py`**

```python
# src/visualization/__init__.py
from .figure_mediation import plot_mediation_forest
from .figure_longitudinal import plot_longitudinal_scatter
from .figure_moderation import plot_moderation_comparison
```

**Step 4: Commit**

```bash
git add src/visualization/
git commit -m "feat: add longitudinal scatter and moderation comparison figures"
```

---

### Task 9: Build generate_all_figures and wire into pipeline

**Files:**
- Create: `src/visualization/figure_summary.py`
- Modify: `src/visualization/__init__.py`
- Modify: `src/main.py`

**Step 1: Implement figure_summary.py**

```python
# src/visualization/figure_summary.py
"""Master function to generate and save all publication figures."""

import logging
from pathlib import Path
from typing import Dict
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

FIGURES_DIR = Path('results/figures')


def generate_all_figures(pipeline_results: Dict,
                          mediators: list,
                          out_dir: Path = FIGURES_DIR) -> Dict[str, Path]:
    """
    Generate all figures from pipeline results dict.

    Expects keys: 'mediation', 'longitudinal', 'moderation'.
    Saves 300 DPI PNGs to out_dir. Returns dict of {name: path}.
    """
    import matplotlib
    matplotlib.use('Agg')
    from src.visualization.figure_mediation import plot_mediation_forest
    from src.visualization.figure_longitudinal import plot_longitudinal_scatter
    from src.visualization.figure_moderation import plot_moderation_comparison

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = {}

    # Figure 1: Mediation forest plot
    med = pipeline_results.get('mediation', {})
    if med.get('bootstrap'):
        fig = plot_mediation_forest(med['bootstrap'], med.get('significant', []))
        path = out_dir / 'fig1_mediation_forest.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved['mediation'] = path
        logger.info(f"  Saved {path}")

    # Figure 2: Longitudinal scatter
    lon = pipeline_results.get('longitudinal', {})
    datasets = pipeline_results.get('_datasets', {})
    if lon.get('model') and 'midus_mr2' in datasets:
        # Need panel — reconstruct from stored data
        from src.data.data_loader_midus_m3 import load_midus_m3
        from src.analysis.longitudinal_analysis import merge_baseline_followup
        try:
            m3 = load_midus_m3()
            panel = merge_baseline_followup(datasets['midus_mr2'], m3)
            fig = plot_longitudinal_scatter(panel, lon)
            path = out_dir / 'fig2_longitudinal_scatter.png'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            saved['longitudinal'] = path
            logger.info(f"  Saved {path}")
        except Exception as e:
            logger.warning(f"  Longitudinal figure skipped: {e}")

    # Figure 3: Moderation comparison
    mod = pipeline_results.get('moderation', {})
    if mod:
        fig = plot_moderation_comparison(mod, mediators)
        path = out_dir / 'fig3_moderation.png'
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        saved['moderation'] = path
        logger.info(f"  Saved {path}")

    return saved
```

**Step 2: Update `src/visualization/__init__.py`**

```python
# src/visualization/__init__.py
from .figure_mediation import plot_mediation_forest
from .figure_longitudinal import plot_longitudinal_scatter
from .figure_moderation import plot_moderation_comparison
from .figure_summary import generate_all_figures
```

**Step 3: Call generate_all_figures at end of run_full_pipeline**

In `src/main.py`, in the `run_full_pipeline` method, after `self.run_moderation(dataset_name)`:

```python
# Generate publication figures
logger.info("=" * 70)
logger.info("GENERATING PUBLICATION FIGURES")
logger.info("=" * 70)
from src.visualization import generate_all_figures
self.results['_datasets'] = self.datasets
figure_paths = generate_all_figures(
    self.results,
    mediators=[m for m in MR2_MEDIATORS if m in self.datasets.get(dataset_name, {}).columns
               if hasattr(self.datasets.get(dataset_name, pd.DataFrame()), 'columns')],
    out_dir=Path('results/figures'),
)
logger.info(f"  Figures saved: {list(figure_paths.values())}")
```

**Step 4: End-to-end smoke test**

```bash
python -c "
import logging; logging.basicConfig(level=logging.WARNING)
from src.main import CognitiveInequalityPipeline
p = CognitiveInequalityPipeline()
p.load_data('midus_mr2')
p.run_mediation('midus_mr2', n_bootstrap=100)
p.run_moderation('midus_mr2')
from src.visualization import generate_all_figures
from src.main import MR2_MEDIATORS
from pathlib import Path
paths = generate_all_figures(p.results, MR2_MEDIATORS, out_dir=Path('/tmp/test_figs'))
print('Saved:', list(paths.keys()))
"
```

Expected: `Saved: ['mediation', 'moderation']` (longitudinal needs M3 files)

**Step 5: Run full test suite**

```bash
pytest tests/ -v --tb=short -q
```

Expected: all existing tests PASS, new tests PASS

**Step 6: Commit**

```bash
git add src/visualization/ src/main.py
git commit -m "feat: add generate_all_figures and wire visualization into pipeline"
```

---

### Task 10: Full pipeline run and final commit

**Step 1: Run full pipeline**

```bash
python -m src.main pipeline 2>&1 | tail -40
```

Expected: completes without errors, logs mediation CIs with covariates, logs longitudinal β, logs moderation interaction p-values, logs figure paths.

**Step 2: Verify figures exist**

```bash
ls results/figures/
```

Expected: `fig1_mediation_forest.png`, `fig2_longitudinal_scatter.png` (if M3 linked), `fig3_moderation.png`

**Step 3: Push to remote**

```bash
git push origin main
```
