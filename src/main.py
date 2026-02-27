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
