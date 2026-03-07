"""
Cognitive Inequality Research System - Main Entry Point

Pipeline for analyzing causal pathways from socioeconomic inequality
to cognitive decline in young adults (ages 34-55).

Primary dataset: MIDUS Refresher 2 (MR2) — BTACT + MoCA cognitive battery,
                 full mediator coverage, survey weights, N=421 complete cases.
Secondary:       Add Health Wave V, NHANES 2013-2014

Usage:
    python -m src.main pipeline                         # full analysis on MR2
    python -m src.main analyze --dataset midus_mr2      # descriptive only
    python -m src.main download --cache-dir data/raw    # fetch NHANES
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mediators available in MIDUS MR2 (all 100% complete in merged dataset)
MR2_MEDIATORS = [
    # Original behavioral/access mediators
    'depression_score',
    'screen_time_change',
    'sleep_change',
    'has_insurance',
    # New psychosocial / structural mediators (Phase 2)
    'purpose_in_life',
    'sense_of_control',
    'neighborhood_quality',
]

# Age variable per dataset
AGE_COL = {
    'midus_mr2': 'RB1PRAGE',
    'addhealth':  'age',
    'nhanes':     'RIDAGEYR',
}


class CognitiveInequalityPipeline:
    """Orchestrates the full cognitive inequality analysis pipeline."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path) if config_path else {}
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.results: Dict[str, Any] = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, dataset_name: str, path: Optional[str] = None) -> pd.DataFrame:
        """Load a processed dataset from CSV."""
        if path and Path(path).exists():
            df = pd.read_csv(path)
        else:
            processed = Path('data/processed') / f'{dataset_name}_cognitive.csv'
            if processed.exists():
                df = pd.read_csv(processed)
            else:
                raise FileNotFoundError(
                    f"Dataset '{dataset_name}' not found at {processed}. "
                    f"Run the appropriate loader first."
                )
        self.datasets[dataset_name] = df
        logger.info(f"Loaded {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
        return df

    # ------------------------------------------------------------------
    # Phase 1: Descriptive statistics
    # ------------------------------------------------------------------

    def run_descriptive(self, dataset_name: str = 'midus_mr2') -> Dict:
        """Descriptive statistics — Table 1 for the paper."""
        df = self.datasets.get(dataset_name)
        if df is None:
            df = self.load_data(dataset_name)

        logger.info("=" * 70)
        logger.info(f"DESCRIPTIVE STATISTICS — {dataset_name.upper()}")
        logger.info("=" * 70)

        age_col = AGE_COL.get(dataset_name, 'RB1PRAGE')

        stats: Dict[str, Any] = {'dataset': dataset_name, 'n': len(df)}

        # Age
        if age_col in df.columns:
            stats['age_mean'] = round(df[age_col].mean(), 2)
            stats['age_sd']   = round(df[age_col].std(),  2)
            logger.info(f"  Age:             {stats['age_mean']:.1f} ± {stats['age_sd']:.1f}")

        # Cognitive outcome
        if 'cognitive_score' in df.columns:
            stats['cognitive_mean'] = round(df['cognitive_score'].mean(), 3)
            stats['cognitive_sd']   = round(df['cognitive_score'].std(),  3)
            logger.info(f"  Cognitive score: {stats['cognitive_mean']:.3f} ± {stats['cognitive_sd']:.3f}")

        # SES
        if 'ses_index' in df.columns:
            stats['ses_mean'] = round(df['ses_index'].mean(), 3)
            stats['ses_sd']   = round(df['ses_index'].std(),  3)
            logger.info(f"  SES index:       {stats['ses_mean']:.3f} ± {stats['ses_sd']:.3f}")

        # Mediators
        for med in MR2_MEDIATORS:
            if med in df.columns:
                valid = df[med].notna()
                stats[f'{med}_mean'] = round(df.loc[valid, med].mean(), 3)
                stats[f'{med}_pct_valid'] = round(valid.mean() * 100, 1)
                logger.info(f"  {med}: mean={stats[f'{med}_mean']:.3f}, "
                            f"valid={stats[f'{med}_pct_valid']:.1f}%")

        # SES quartile breakdown of cognitive score
        if 'ses_quartile' in df.columns and 'cognitive_score' in df.columns:
            logger.info("\n  Cognitive score by SES quartile:")
            q_stats = df.groupby('ses_quartile', observed=True)['cognitive_score'].agg(['mean','std','count'])
            for q, row in q_stats.iterrows():
                logger.info(f"    {q}: {row['mean']:.3f} ± {row['std']:.3f}  (N={int(row['count'])})")
            stats['quartile_means'] = q_stats['mean'].round(3).to_dict()

        # Complete cases
        key_vars = ['cognitive_score', 'ses_index'] + [m for m in MR2_MEDIATORS if m in df.columns]
        cc = df[key_vars].dropna()
        stats['complete_cases'] = len(cc)
        logger.info(f"\n  Complete cases (all key vars): {len(cc)}")

        self.results['descriptive'] = stats
        return stats

    # ------------------------------------------------------------------
    # Phase 2: PC algorithm causal discovery
    # ------------------------------------------------------------------

    def run_causal_discovery(self, dataset_name: str = 'midus_mr2',
                             alpha: float = 0.05) -> Dict:
        """PC algorithm causal discovery on complete cases."""
        from src.analysis.pc_algorithm import discover_ses_cognition_paths

        df = self.datasets.get(dataset_name)
        if df is None:
            df = self.load_data(dataset_name)
        mediators = [m for m in MR2_MEDIATORS if m in df.columns]

        logger.info("=" * 70)
        logger.info("CAUSAL DISCOVERY — PC ALGORITHM")
        logger.info("=" * 70)
        logger.info(f"  Dataset: {dataset_name}, mediators: {mediators}")

        from src.llm.ollama_client import OllamaClient
        llm_available = OllamaClient().is_available()
        logger.info(f"  Ollama {'available' if llm_available else 'not available'} — "
                    f"{'enabling' if llm_available else 'skipping'} LLM graph validation")
        pc_result = discover_ses_cognition_paths(
            df, dataset_name=dataset_name,
            mediators=mediators, alpha=alpha, use_llm=llm_available,
        )
        logger.info(f"  Adjacency matrix:\n{pc_result.get('adjacency')}")
        self.results['pc_discovery'] = pc_result
        return pc_result

    # ------------------------------------------------------------------
    # Phase 3: Baron-Kenny mediation with bootstrap CIs
    # ------------------------------------------------------------------

    def run_mediation(self, dataset_name: str = 'midus_mr2',
                      n_bootstrap: int = 1000) -> Dict:
        """Baron-Kenny mediation analysis for each mediator."""
        from src.analysis.mediation_analysis import (
            analyze_all_mediators, baron_kenny_mediation,
            get_significant_mediators,
        )

        df = self.datasets.get(dataset_name)
        if df is None:
            df = self.load_data(dataset_name)
        mediators = [m for m in MR2_MEDIATORS if m in df.columns]

        # Survey weights (use MR2 stratification weight if available)
        weights = None
        if 'RB1PWGHT6' in df.columns:
            weights = df['RB1PWGHT6'].values

        logger.info("=" * 70)
        logger.info("MEDIATION ANALYSIS — BARON-KENNY + BOOTSTRAP CIs")
        logger.info("=" * 70)
        logger.info(f"  Dataset: {dataset_name}, N bootstrap: {n_bootstrap}")
        logger.info(f"  Exposure: ses_index → Outcome: cognitive_score")
        logger.info(f"  Mediators: {mediators}")

        # Full Baron-Kenny for each mediator (point estimates + paths)
        bk_results = {}
        for m in mediators:
            try:
                bk = baron_kenny_mediation(df, x='ses_index', m=m, y='cognitive_score',
                                           covariates=['RB1PRAGE', 'female'], weights=weights)
                bk_results[m] = bk
                logger.info(f"\n  [{m}]")
                logger.info(f"    a (SES→mediator):       {bk.a:.4f}")
                logger.info(f"    b (mediator→cognitive): {bk.b:.4f}")
                logger.info(f"    c  (total effect):      {bk.c:.4f}")
                logger.info(f"    c' (direct effect):     {bk.c_prime:.4f}")
                logger.info(f"    indirect (a×b):         {bk.indirect:.4f}")
                logger.info(f"    proportion mediated:    {bk.proportion_mediated:.3f}")
            except Exception as e:
                logger.warning(f"  BK failed for {m}: {e}")

        # Bootstrap CIs
        logger.info(f"\n  Bootstrap CIs ({n_bootstrap} iterations):")
        boot_results = analyze_all_mediators(
            df, x='ses_index', y='cognitive_score',
            mediators=mediators, covariates=['RB1PRAGE', 'female'],
            weights=weights, n_boot=n_bootstrap,
        )
        significant = get_significant_mediators(boot_results)

        for m, res in boot_results.items():
            sig = ' *' if m in significant else ''
            logger.info(f"    {m}: {res.point_estimate:.4f} "
                        f"[{res.ci_lower:.4f}, {res.ci_upper:.4f}]{sig}")

        logger.info(f"\n  Significant mediators (CI excludes 0): {significant}")

        mediation_results = {
            'baron_kenny': bk_results,
            'bootstrap': boot_results,
            'significant': significant,
        }
        self.results['mediation'] = mediation_results
        return mediation_results

    # ------------------------------------------------------------------
    # Phase 4: XGBoost + SHAP
    # ------------------------------------------------------------------

    def run_prediction(self, dataset_name: str = 'midus_mr2') -> Dict:
        """Train XGBoost and compute SHAP feature importance."""
        from src.analysis.prediction_model import CognitivePredictionModel

        df = self.datasets.get(dataset_name)
        if df is None:
            df = self.load_data(dataset_name)
        mediators = [m for m in MR2_MEDIATORS if m in df.columns]
        feature_cols = ['ses_index'] + mediators

        model_df = df[feature_cols + ['cognitive_score']].dropna()
        X = model_df[feature_cols]
        y = model_df['cognitive_score']

        logger.info("=" * 70)
        logger.info("PREDICTION MODEL — XGBoost + SHAP")
        logger.info("=" * 70)
        logger.info(f"  Training on N={len(X)}, features: {feature_cols}")

        model = CognitivePredictionModel()
        model.train(X, y)
        cv = model.cross_validate(X, y, cv=5)
        shap_vals = model.compute_shap_values(X)
        importance = model.feature_importance()

        r2_mean = float(np.mean(cv['r2_scores']))
        r2_std  = float(np.std(cv['r2_scores']))
        logger.info(f"  CV R²:   {r2_mean:.3f} ± {r2_std:.3f}")
        logger.info(f"  CV RMSE: {float(np.mean(cv['rmse_scores'])):.3f} ± "
                    f"{float(np.std(cv['rmse_scores'])):.3f}")
        logger.info("  SHAP feature importance (mean |SHAP|):")
        for feat, imp in importance.items():
            logger.info(f"    {feat}: {imp:.4f}")

        prediction_results = {
            'model': model,
            'X': X,
            'cv_results': cv,
            'feature_importance': importance,
            'r2_mean': r2_mean,
        }
        self.results['prediction'] = prediction_results
        return prediction_results

    # ------------------------------------------------------------------
    # Phase 5: Counterfactual intervention simulation
    # ------------------------------------------------------------------

    def run_counterfactual(self, dataset_name: str = 'midus_mr2') -> list:
        """Simulate +1 SD policy interventions on significant mediators."""
        from src.simulation.counterfactual_simulator import generate_interventions

        pred = self.results.get('prediction', {})
        med  = self.results.get('mediation', {})
        model = pred.get('model')
        X     = pred.get('X')
        boot  = med.get('bootstrap', {})

        if model is None or X is None or not boot:
            logger.warning("Run prediction and mediation before counterfactual.")
            return []

        logger.info("=" * 70)
        logger.info("COUNTERFACTUAL SIMULATION — SIGNIFICANT MEDIATORS")
        logger.info("=" * 70)

        interventions = generate_interventions(boot, model, X)

        if not interventions:
            logger.info("  No significant mediators — no interventions generated.")
        else:
            for r in interventions:
                logger.info(
                    f"  {r.variable}: baseline={r.baseline_mean:.3f} → "
                    f"counterfactual={r.counterfactual_mean:.3f}  "
                    f"Δ={r.effect_size:+.3f}  (N affected={r.affected_n})"
                )

        self.results['counterfactual'] = interventions
        return interventions

    # ------------------------------------------------------------------
    # Phase 6: Sensitivity analysis (E-values)
    # ------------------------------------------------------------------

    def run_sensitivity(self, dataset_name: str = 'midus_mr2') -> Dict:
        """E-value sensitivity analysis for the total and direct SES effects."""
        from src.analysis.sensitivity_analysis import run_ses_cognition_sensitivity

        mediation = self.results.get('mediation', {})
        bk_results = mediation.get('baron_kenny', {})
        if not bk_results:
            logger.warning("No Baron-Kenny results found — run mediation first.")
            return {}

        df = self.datasets.get(dataset_name)
        if df is None:
            df = self.load_data(dataset_name)

        sd_outcome = float(df['cognitive_score'].std()) if 'cognitive_score' in df.columns else 0.821

        sensitivity = run_ses_cognition_sensitivity(bk_results, sd_outcome=sd_outcome)
        self.results['sensitivity'] = sensitivity
        return sensitivity

    def run_full_pipeline(self, dataset_name: str = 'midus_mr2',
                          n_bootstrap: int = 1000) -> Dict:
        """Run all five analysis stages in sequence."""
        logger.info("=" * 70)
        logger.info("COGNITIVE INEQUALITY RESEARCH — FULL PIPELINE")
        logger.info(f"Primary dataset: {dataset_name.upper()}")
        logger.info("=" * 70)

        self.load_data(dataset_name)
        self.run_descriptive(dataset_name)
        self.run_causal_discovery(dataset_name)
        self.run_mediation(dataset_name, n_bootstrap=n_bootstrap)
        self.run_prediction(dataset_name)
        self.run_counterfactual(dataset_name)
        self.run_sensitivity(dataset_name)

        # Save summary results
        self._save_results()
        logger.info("\nPIPELINE COMPLETE — results saved to results/")
        return self.results

    def _save_results(self):
        """Persist key numeric results to results/."""
        out = Path('results')
        out.mkdir(exist_ok=True)

        # Descriptive stats
        if 'descriptive' in self.results:
            desc = {k: v for k, v in self.results['descriptive'].items()
                    if not isinstance(v, dict) or k == 'quartile_means'}
            with open(out / 'descriptive_stats.json', 'w') as f:
                json.dump(desc, f, indent=2, default=str)

        # Mediation results
        if 'mediation' in self.results:
            med = self.results['mediation']
            rows = []
            for name, res in med['bootstrap'].items():
                bk = med['baron_kenny'].get(name)
                rows.append({
                    'mediator':             name,
                    'a_path':               round(bk.a, 4) if bk else None,
                    'b_path':               round(bk.b, 4) if bk else None,
                    'total_effect':         round(bk.c, 4) if bk else None,
                    'direct_effect':        round(bk.c_prime, 4) if bk else None,
                    'indirect_effect':      round(res.point_estimate, 4),
                    'ci_lower':             round(res.ci_lower, 4),
                    'ci_upper':             round(res.ci_upper, 4),
                    'proportion_mediated':  round(bk.proportion_mediated, 3) if bk else None,
                    'significant':          name in med['significant'],
                })
            pd.DataFrame(rows).to_csv(out / 'mediation_results.csv', index=False)

        # Sensitivity / E-values
        if 'sensitivity' in self.results:
            sens = self.results['sensitivity']
            rows = []
            te = sens.get('total_effect', {})
            rows.append({
                'path': 'total_effect',
                'mediator': None,
                'estimate_type': 'c',
                'evalue_point': round(te.get('evalue_point', float('nan')), 3),
                'evalue_ci': round(te.get('evalue_ci', float('nan')), 3),
                'rr_approx': round(te.get('rr_approx', float('nan')), 3),
            })
            for med, ev in sens.get('direct_effects', {}).items():
                rows.append({
                    'path': 'direct_effect',
                    'mediator': med,
                    'estimate_type': "c'",
                    'evalue_point': round(ev.get('evalue_point', float('nan')), 3),
                    'evalue_ci': round(ev.get('evalue_ci', float('nan')), 3),
                    'rr_approx': round(ev.get('rr_approx', float('nan')), 3),
                })
            pd.DataFrame(rows).to_csv(out / 'sensitivity_evalues.csv', index=False)

        # SHAP importance
        if 'prediction' in self.results:
            imp = self.results['prediction']['feature_importance']
            r2  = self.results['prediction']['r2_mean']
            pd.DataFrame(
                [{'feature': k, 'mean_shap': v} for k, v in imp.items()]
            ).to_csv(out / 'shap_importance.csv', index=False)
            with open(out / 'model_performance.json', 'w') as f:
                json.dump({'r2_cv_mean': round(r2, 4)}, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cognitive Inequality Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest='command')

    # download
    dl = subparsers.add_parser('download', help='Download/preprocess NHANES and other datasets')
    dl.add_argument('--cache-dir', default='data/raw')
    dl.add_argument('--output-dir', default='data/processed')
    dl.add_argument('--brfss-path', default=None)
    dl.add_argument('--gss-path', default=None)

    # analyze (descriptive only)
    an = subparsers.add_parser('analyze', help='Descriptive statistics for one dataset')
    an.add_argument('--dataset', default='midus_mr2',
                    choices=['midus_mr2', 'addhealth', 'nhanes', 'brfss', 'gss'])
    an.add_argument('--config', default=None)

    # pipeline (full)
    pp = subparsers.add_parser('pipeline', help='Run full analysis pipeline')
    pp.add_argument('--dataset', default='midus_mr2',
                    choices=['midus_mr2', 'addhealth', 'nhanes'])
    pp.add_argument('--n-bootstrap', type=int, default=1000)
    pp.add_argument('--config', default=None)

    args = parser.parse_args()

    if args.command == 'download':
        from src.data.download_all_datasets import main as dl_main
        sys.argv = ['download',
                    '--cache-dir', args.cache_dir,
                    '--output-dir', args.output_dir]
        if args.brfss_path:
            sys.argv += ['--brfss-path', args.brfss_path]
        if args.gss_path:
            sys.argv += ['--gss-path', args.gss_path]
        dl_main()

    elif args.command == 'analyze':
        p = CognitiveInequalityPipeline(args.config)
        p.load_data(args.dataset)
        p.run_descriptive(args.dataset)

    elif args.command == 'pipeline':
        p = CognitiveInequalityPipeline(args.config)
        p.run_full_pipeline(dataset_name=args.dataset,
                            n_bootstrap=args.n_bootstrap)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
