"""Two-panel figure: indirect effects by sex and age cohort subgroups."""

import matplotlib.pyplot as plt
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
        ('female',     'Sex Moderation',       {0: 'Male', 1: 'Female'},          ['#aaaaaa', '#222222']),
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

        for group, offset, color in zip(groups, offsets, colors):
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
