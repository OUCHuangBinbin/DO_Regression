"""
fig1_obs_distribution.py
========================
Observational data distribution and spatial coverage (Fig 1).

Upper panel: yearly number of valid grid cells by platform (OSD, CTD, Argo).
Lower panel: monthly ocean spatial coverage (%) with 12-month moving average.
"""

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Configuration
FILES = {
    'OSD':  "/g12338011ghq/dataset/WOD/Subsets_MultiLevel/WOD_O2_Type_OSD.nc",
    'CTD':  "/g12338011ghq/dataset/WOD/Subsets_MultiLevel/WOD_O2_Type_CTD.nc",
    'Argo': "/g12338011ghq/dataset/WOD/Subsets_MultiLevel/WOD_O2_Type_PFL.nc",
}
GRIDDED_FILE = "/g12338011ghq/dataset/WOD/WOD_Gridded_O2_Lev0.nc"
START_YEAR = 1950
END_YEAR = 2014


def get_yearly_counts(file_path: str) -> pd.Series:
    """Count valid surface grid cells per year."""
    if not os.path.exists(file_path):
        print(f"[skip] Missing file: {file_path}")
        return pd.Series(dtype='float64')

    try:
        with xr.open_dataset(file_path) as ds:
            var_name = 'o2' if 'o2' in ds else list(ds.data_vars)[0]
            da = ds[var_name].sel(
                time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
            )
            if 'lev' in da.dims:
                da_surface = da.sel(lev=slice(0, 10)).notnull().any(dim='lev')
            else:
                da_surface = da.notnull()

            monthly = da_surface.sum(dim=['lat', 'lon'])
            yearly = monthly.resample(time='YS').sum()
            return pd.Series(
                yearly.squeeze().values,
                index=yearly['time.year'].values,
                dtype='float64',
            )
    except Exception as e:
        print(f"[error] Processing {file_path}: {e}")
        return pd.Series(dtype='float64')


def get_monthly_coverage(file_path: str) -> pd.Series:
    """Compute monthly ocean spatial coverage (%)."""
    print("Computing monthly coverage...")
    try:
        with xr.open_dataset(file_path) as ds:
            var_name = 'o2' if 'o2' in ds else list(ds.data_vars)[0]
            da = ds[var_name].sel(
                time=slice(f"{START_YEAR}-01-01", f"{END_YEAR}-12-31")
            )
            ocean_mask = da.notnull().any(dim='time')
            total_ocean = int(ocean_mask.sum().values)
            monthly_valid = da.notnull().sum(dim=['lat', 'lon'])
            coverage = (monthly_valid / total_ocean) * 100

            time_idx = pd.date_range(
                start=f"{START_YEAR}-01-01",
                end=f"{END_YEAR}-12-31",
                freq='MS',
            )
            return pd.Series(
                coverage.squeeze().values, index=time_idx, dtype='float64'
            )
    except Exception as e:
        print(f"[error] Coverage computation: {e}")
        return pd.Series(dtype='float64')


def plot() -> None:
    """Generate the two-panel figure."""
    print(f"Processing {START_YEAR}-{END_YEAR}...")

    s_osd = get_yearly_counts(FILES['OSD'])
    s_ctd = get_yearly_counts(FILES['CTD'])
    s_argo = get_yearly_counts(FILES['Argo'])
    s_coverage = get_monthly_coverage(GRIDDED_FILE)

    # Align yearly
    df = pd.DataFrame(
        {'OSD': s_osd, 'CTD': s_ctd, 'Argo': s_argo}
    ).fillna(0)
    df = df.reindex(np.arange(START_YEAR, END_YEAR + 1), fill_value=0)

    # Plot
    colors = {
        'OSD':  '#5D9CD3',
        'CTD':  '#F2B94A',
        'Argo': '#F05C66',
    }

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(13, 9), dpi=300
    )

    # Upper panel: stacked bar
    ax1.bar(df.index, df['OSD'],  color=colors['OSD'],  label='OSD',
            width=0.8, alpha=0.8)
    ax1.bar(df.index, df['CTD'],  bottom=df['OSD'],
            color=colors['CTD'],  label='CTD',  width=0.8, alpha=0.8)
    ax1.bar(df.index, df['Argo'], bottom=df['OSD'] + df['CTD'],
            color=colors['Argo'], label='Argo', width=0.8, alpha=0.8)

    ax1.set_ylabel('Number of Valid Grid Cells', fontsize=13)
    ax1.set_ylim(0, df.sum(axis=1).max() * 1.1)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend(loc='upper left', frameon=False, fontsize=12)

    # Lower panel: monthly coverage
    if s_coverage is not None and not s_coverage.empty:
        ax2.plot(s_coverage.index, s_coverage.values,
                 color='#377eb8', linewidth=0.8, alpha=0.5,
                 label='Monthly Coverage')
        rolling = s_coverage.rolling(window=12, center=True).mean()
        ax2.plot(s_coverage.index, rolling,
                 color='#e41a1c', linewidth=2.0,
                 label='12-Month Moving Avg')

    ax2.set_ylabel('Ocean Spatial Coverage (%)', fontsize=13)
    ax2.set_xlabel('Time', fontsize=13)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    ax2.legend(loc='upper left', frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig(
        "/g12338011ghq/project/hbb/OxygenDiffusion/"
        "Oxy_regression_Surface/results/FigureResults/fig1.png"
    )
    plt.show()


if __name__ == "__main__":
    plot()
