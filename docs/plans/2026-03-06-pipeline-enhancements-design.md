# Pipeline Enhancements Design — 2026-03-06

**Goal:** Add covariate adjustment, longitudinal validation, moderation analysis, and
publication-quality figures to the cognitive inequality pipeline.

**Architecture:** Option A — layered additions to the existing pipeline. Each enhancement is an
independent phase with its own module, testable in isolation, and mapped to a methods subsection.

---

## Phase 1: Covariate Adjustment

**What:** Wire `covariates=['RB1PRAGE', 'female']` into `run_mediation` in `main.py`.

**How:** `baron_kenny_mediation` and `analyze_all_mediators` already accept a `covariates`
parameter. No changes needed to `mediation_analysis.py`. All 7 mediator analyses become
age- and sex-adjusted automatically.

**Files changed:** `src/main.py` only.

---

## Phase 2: Longitudinal Validation

**What:** Test whether baseline SES predicts cognitive change from MR2 to MIDUS 3 follow-up.

**New files:**
- `src/data/data_loader_midus_m3.py` — loads M3 Survey (`M3_P1_SURVEY_N3294_20251029.sav`)
  and BTACT (`M3_P3_BTACT_N3291_20210922.sav`), extracts `M2ID`, `C1PRAGE`, income vars
  (`C1SRINC`, `C1STINC`), and `C3TCOMP` (cognitive composite), merges on `M2ID`
- `src/analysis/longitudinal_analysis.py` — merges MR2 baseline + M3 follow-up on `M2ID`,
  computes `cognitive_change = M3_cognitive - MR2_cognitive` (z-scored), runs OLS:
  `cognitive_change ~ ses_index_baseline + age + sex`

**Pipeline change:** New `run_longitudinal` phase added to `CognitiveInequalityPipeline`,
called in `run_full_pipeline` after mediation.

**Key variable mapping:**

| Concept         | MR2 variable   | M3 variable  |
|-----------------|----------------|--------------|
| Participant ID  | M2ID           | M2ID         |
| Age             | RB1PRAGE       | C1PRAGE      |
| Income          | RB1PB1/RB1PB16 | C1SRINC      |
| Cognitive score | RB3TCOMPZ      | C3TCOMP      |

---

## Phase 3: Moderation / Subgroup Analysis

**What:** Test whether the SES→cognition gap differs by sex and age cohort.

**New file:** `src/analysis/moderation_analysis.py`

**Sex moderation:**
- Split sample into `female=0` and `female=1`
- Run full bootstrap mediation on each subgroup for all 7 mediators
- Interaction model: `cognitive_score ~ ses_index * female + age`
                                               Crafting Interpreters by Robert Nystrom
**Age cohort moderation:**
- Split at median age (~44): younger (34–44) vs older (45–55)
- Run full bootstrap mediation on each cohort
- Interaction model: `cognitive_score ~ ses_index * age_cohort + female`

**Pipeline change:** New `run_moderation` phase in `main.py`, called after mediation.
Results stored as `self.results['moderation']`.

---

## Phase 4: Publication-Quality Figures

**New module:** `src/visualization/`

| File | Contents |
|------|----------|
| `figure_mediation.py` | Horizontal forest plot, 7 mediators, bootstrap CIs, significant mediators highlighted |
| `figure_longitudinal.py` | Scatter of baseline SES vs cognitive change, OLS line + 95% CI band, colored by sex |
| `figure_moderation.py` | Two-panel: indirect effects by sex subgroup / by age cohort, CIs overlaid |
| `figure_summary.py` | 4-panel composite + path diagram with standardized coefficients |
| `__init__.py` | Exports `generate_all_figures(pipeline_results)` |

**Style:** `seaborn` whitegrid, `font_scale=1.2`, grayscale-compatible palette, 300 DPI,
saved to `results/figures/`.

**Pipeline change:** `generate_all_figures` called at end of `run_full_pipeline`.

---

## Implementation Order

1. Covariate adjustment (`main.py` — 5 min)
2. MIDUS 3 loader (`data_loader_midus_m3.py`)
3. Longitudinal analysis (`longitudinal_analysis.py` + `run_longitudinal`)
4. Moderation analysis (`moderation_analysis.py` + `run_moderation`)
5. Visualization module (`src/visualization/`)
6. Update Colab notebook to include new phases
7. Commit all changes

---

## Success Criteria

- Mediation results include age + sex as covariates in all reported models
- Longitudinal regression shows significant `ses_index` → `cognitive_change` coefficient
- Moderation plots show visually distinct indirect effect sizes across subgroups
- All figures export at 300 DPI with labeled axes, legend, and sample size annotations
- `python -m src.main pipeline` runs end-to-end without errors
