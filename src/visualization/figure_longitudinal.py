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
    Points colored by sex. OLS regression line overlaid.
    """
    import seaborn as sns
    sns.set_theme(style='whitegrid', font_scale=1.2)

    data = panel[[x, y, 'female']].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))

    for sex_val, label, color in [(0, 'Male', '#888888'), (1, 'Female', '#222222')]:
        sub = data[data['female'] == sex_val]
        ax.scatter(sub[x], sub[y], c=color, alpha=0.4, s=18, label=label)

    # Regression line
    import statsmodels.api as sm
    x_range = np.linspace(data[x].min(), data[x].max(), 100)
    model = result['model']
    # Build prediction frame matching model's exog columns
    exog_cols = model.model.exog_names  # e.g. ['const', 'ses_index', 'RB1PRAGE', 'female']
    pred_data = {col: 0.0 for col in exog_cols}
    pred_data['const'] = 1.0
    pred_data[x] = x_range
    if 'RB1PRAGE' in pred_data:
        pred_data['RB1PRAGE'] = float(data['RB1PRAGE'].mean()) if 'RB1PRAGE' in data.columns else 44.0
    if 'female' in pred_data:
        pred_data['female'] = 0.5
    pred_df = pd.DataFrame(pred_data)
    y_pred = model.predict(pred_df)
    ax.plot(x_range, y_pred, color='black', linewidth=2)

    ax.set_xlabel('Baseline SES Index', fontsize=12)
    ax.set_ylabel('Cognitive Change (MR2 → M3, SD units)', fontsize=12)
    ax.set_title('Baseline SES Predicts Cognitive Change', fontsize=13, fontweight='bold')

    coef = result['ses_coef']
    r2   = result['r2']
    p    = result['ses_pvalue']
    n    = result['n']
    ax.text(0.03, 0.95,
            f'β={coef:.3f}, p={p:.3f}, R²={r2:.3f}, N={n}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig
