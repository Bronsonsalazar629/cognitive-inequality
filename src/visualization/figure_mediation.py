"""Forest plot of mediation indirect effects with bootstrap CIs."""

from typing import Dict, List
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
