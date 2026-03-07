"""Master function to generate and save all publication figures."""

import logging
from pathlib import Path
from typing import Dict
import matplotlib
matplotlib.use('Agg')
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
    from src.visualization.figure_mediation import plot_mediation_forest
    from src.visualization.figure_longitudinal import plot_longitudinal_scatter
    from src.visualization.figure_moderation import plot_moderation_comparison

    out_dir = Path(out_dir)
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

    # Figure 2: Longitudinal scatter (only if panel data available)
    lon = pipeline_results.get('longitudinal', {})
    datasets = pipeline_results.get('_datasets', {})
    if lon.get('model') is not None and 'midus_mr2' in datasets:
        from src.data.data_loader_midus_m3 import load_midus_m3
        from src.analysis.longitudinal_analysis import merge_baseline_followup
        try:
            m3 = load_midus_m3()
            panel = merge_baseline_followup(datasets['midus_mr2'], m3)
            if panel['cognitive_change'].notna().sum() > 10:
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
