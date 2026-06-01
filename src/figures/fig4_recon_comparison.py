"""
fig4_recon_comparison.py
========================
Reconstructed DO anomaly compared with CMIP6 truth across methods (Fig 4).

Two panels: monthly (top) and annual mean (bottom) time-series showing
MGSR (Ours), OLS, MME, and Naive Mean versus the CMIP6 truth.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import AutoMinorLocator


def plot_comparison(
    csv_path: str,
    start_year: int = 1950,
    end_year: int = 2014,
) -> None:
    """Generate the two-panel reconstruction comparison figure."""
    if not os.path.exists(csv_path):
        print(f"[error] File not found: {csv_path}")
        return

    # Load and preprocess
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    df = df.loc[f"{start_year}-01-01":f"{end_year}-12-31"]

    # Scale: mol/m³ → μmol/m³
    SCALE = 1_000_000
    df = df * SCALE
    df_annual = df.resample('YE').mean()

    # Column mapping
    mapping = {
        'Truth': 'anom_cmip6_truth',
        'Ridge': 'anom_recon_roll_model_src',
        'OLS':   'OLS_Recon',
        'MME':   'MME',
        'Naive': 'Naive',
    }

    style = {
        'Truth': {'color': '#000000', 'lw': 1.6, 'alpha': 1.0, 'ls': '-',
                   'zorder': 10, 'label': 'MPI-ESM1-2-LR'},
        'Ridge': {'color': '#D7191C', 'lw': 1.8, 'alpha': 1.0, 'ls': '-',
                   'zorder': 15, 'label': 'MGSR (Ours)'},
        'OLS':   {'color': '#2B83BA', 'lw': 1.0, 'alpha': 0.8, 'ls': '--',
                   'zorder': 5, 'label': 'OLS'},
        'MME':   {'color': '#FDAE61', 'lw': 1.0, 'alpha': 0.8, 'ls': '--',
                   'zorder': 4, 'label': 'MME'},
        'Naive': {'color': '#ABDDA4', 'lw': 1.0, 'alpha': 0.7, 'ls': '--',
                   'zorder': 3, 'label': 'Naive'},
    }

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), dpi=300)

    def draw_panel(ax, data, is_monthly=True):
        order = ['Truth', 'OLS', 'MME', 'Naive', 'Ridge']
        for key in order:
            x = data.index if is_monthly else data.index.year
            ax.plot(x, data[mapping[key]], **style[key])

        ax.set_ylabel("DO Anomaly ($\mu$mol m$^{-3}$)", fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(direction='out', length=5, width=1.2, labelsize=11)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(axis='y', linestyle=':', color='gray', alpha=0.2, zorder=0)

        if is_monthly:
            ax.set_xlim(data.index.min(), data.index.max())
        else:
            ax.set_xlim(data.index.year.min(), data.index.year.max())
            ax.set_xlabel("Year", fontsize=12)

        ax.legend(loc='upper right', frameon=False, ncol=2,
                  fontsize=10, columnspacing=1.0, handletextpad=0.4)

    draw_panel(ax1, df, is_monthly=True)
    draw_panel(ax2, df_annual, is_monthly=False)

    plt.tight_layout(h_pad=2.0)

    save_dir = "/g12338011ghq/project/hbb/OxygenDiffusion/" \
               "Oxy_regression_Surface/results/FigureResults"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "fig4_Final_Comparison.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"[save] {save_path}")
    plt.show()


if __name__ == "__main__":
    csv = "/g12338011ghq/project/hbb/OxygenDiffusion/" \
          "Oxy_regression_Surface/results/Results_CSV/" \
          "Ship_bias_LR_2000_2014_monthly.csv"
    plot_comparison(csv)
