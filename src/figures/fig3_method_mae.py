"""
fig3_method_mae.py
==================
Rolling MAE comparison across reconstruction methods (Fig 3).

Shows 12-month rolling mean absolute error for Ridge (MGSR), OLS,
MME, and Naive Mean relative to the CMIP6 truth.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_mae(
    csv_path: str,
    start_year: int = 1950,
    end_year: int = 2014,
) -> None:
    """Plot 12-month rolling MAE for each method."""
    if not os.path.exists(csv_path):
        print(f"[error] File not found: {csv_path}")
        return

    # Load and slice
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    df = df.loc[f"{start_year}-01-01":f"{end_year}-12-31"]

    # Column mapping
    mapping = {
        'Truth': 'anom_cmip6_truth',
        'Ridge': 'anom_recon_roll_model_src',
        'OLS':   'OLS_Recon',
        'MME':   'MME',
        'Naive': 'Naive',
    }

    # Compute 12-month rolling MAE
    mae_df = pd.DataFrame(index=df.index)
    methods = ['Ridge', 'OLS', 'MME', 'Naive']

    for m in methods:
        col = mapping[m]
        truth_col = mapping['Truth']
        if col in df.columns and truth_col in df.columns:
            abs_err = np.abs(df[col] - df[truth_col])
            mae_df[m] = abs_err.rolling(window=12, center=True).mean()
            mae_df[m] = mae_df[m] * 1000000  # mol/m³ → μmol/m³

    plt.rcParams['font.sans-serif'] = [
        'DejaVu Sans', 'Liberation Sans', 'sans-serif'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

    colors = {
        'Ridge': '#D7191C',
        'OLS':   '#2B83BA',
        'MME':   '#FDAE61',
        'Naive': '#ABDDA4',
    }

    # Baseline methods (dashed, lower z-order)
    for m in ['Naive', 'MME', 'OLS']:
        if m in mae_df.columns:
            ax.plot(mae_df.index, mae_df[m], color=colors[m],
                    lw=1.5, alpha=0.6, label=f'Benchmark: {m}',
                    linestyle='--')

    # Ridge (solid, top z-order)
    if 'Ridge' in mae_df.columns:
        ax.plot(mae_df.index, mae_df['Ridge'], color=colors['Ridge'],
                lw=2.5, alpha=1.0, label='MGSR (Ours)', zorder=10)
        ax.fill_between(mae_df.index, 0, mae_df['Ridge'],
                        color=colors['Ridge'], alpha=0.08)

    ax.set_ylabel("Mean Absolute Error (μmol m$^{-3}$)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylim(0, mae_df.max().max() * 1.2)
    ax.set_xlim(
        pd.Timestamp(f"{start_year}-01-01"),
        pd.Timestamp(f"{end_year}-12-31"),
    )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='gray', linestyle=':', alpha=0.2)

    ax.legend(loc='upper left', frameon=False, fontsize=9, ncol=2)
    plt.tight_layout()

    save_dir = "/g12338011ghq/project/hbb/OxygenDiffusion/" \
               "Oxy_regression_Surface/results/FigureResults"
    plt.savefig(
        os.path.join(save_dir, "fig3_MAE_Comparison.png"),
        bbox_inches='tight',
    )
    plt.show()


if __name__ == "__main__":
    csv = "/g12338011ghq/project/hbb/OxygenDiffusion/" \
          "Oxy_regression_Surface/results/Results_CSV/" \
          "Ship_bias_LR_2000_2014_monthly.csv"
    plot_mae(csv)
