import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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
    from src.visualization.figure_mediation import plot_mediation_forest
    boot_results = _make_boot_results()
    significant = ['mediator_3', 'mediator_4', 'mediator_5']
    fig = plot_mediation_forest(boot_results, significant)
    assert fig is not None
    plt.close('all')
