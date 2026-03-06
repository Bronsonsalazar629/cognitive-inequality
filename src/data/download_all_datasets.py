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

DATASET_REGISTRY = {
    'nhanes': {
        'status': 'auto',
        'instructions': 'Auto-downloaded from CDC NHANES API.',
    },
    'addhealth': {
        'status': 'manual',
        'instructions': (
            'Download from ICPSR Study 21600 (Add Health Wave V). '
            'Requires ICPSR login. Download DS0032 (main survey) and DS0042 (weights).'
        ),
    },
    'brfss': {
        'status': 'manual',
        'instructions': 'Download BRFSS XPT file from CDC website.',
    },
    'gss': {
        'status': 'manual',
        'instructions': 'Download GSS .dta file from NORC website.',
    },
    'piaac': {
        'status': 'manual',
        'instructions': (
            'Download PIAAC Cycle 2 (2023) public-use CSV from OECD/NCES. '
            'Semicolon-delimited. US file: prgusap2.csv.'
        ),
    },
    'nsduh': {
        'status': 'manual',
        'instructions': (
            'Download NSDUH 2024 public-use RData from SAMHSA. '
            'File: NSDUH_2024.RData.'
        ),
    },
}


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
    parser.add_argument('--addhealth-path', default=None, help='Path to Add Health DS0032 .rda file')
    parser.add_argument('--addhealth-weights', default=None, help='Path to Add Health DS0042 .rda file')
    parser.add_argument('--piaac-path', default=None, help='Path to PIAAC semicolon-delimited CSV')
    parser.add_argument('--nsduh-path', default=None, help='Path to NSDUH .RData file')
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

    # Add Health (requires manual download from ICPSR)
    if args.addhealth_path:
        from src.data.data_loader_addhealth import load_addhealth
        logger.info("Loading Add Health Wave V...")
        addhealth = load_addhealth(
            data_path=args.addhealth_path,
            weights_path=args.addhealth_weights,
            output_path=str(output_dir / 'addhealth_cognitive.csv')
        )
        datasets['addhealth'] = addhealth
        print_quality_report('Add Health', addhealth)
    else:
        logger.warning("Add Health skipped (provide --addhealth-path to include)")

    # PIAAC (requires manual download)
    if args.piaac_path:
        from src.data.data_loader_piaac import load_piaac
        logger.info("Loading PIAAC Cycle 2...")
        piaac = load_piaac(
            data_path=args.piaac_path,
            output_path=str(output_dir / 'piaac_cognitive.csv')
        )
        datasets['piaac'] = piaac
        print_quality_report('PIAAC', piaac)
    else:
        logger.warning("PIAAC skipped (provide --piaac-path to include)")

    # NSDUH (requires manual download)
    if args.nsduh_path:
        from src.data.data_loader_nsduh import load_nsduh
        logger.info("Loading NSDUH 2024...")
        nsduh = load_nsduh(
            data_path=args.nsduh_path,
            output_path=str(output_dir / 'nsduh_processed.csv')
        )
        datasets['nsduh'] = nsduh
        print_quality_report('NSDUH', nsduh)
    else:
        logger.warning("NSDUH skipped (provide --nsduh-path to include)")

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
