"""
GODIP-O2.py
===========
Compare global mean surface dissolved oxygen from multiple external
products (GDOIP corrected/uncorrected, O2-Map SJTU) against each other.

Produces monthly and annual mean comparison plots.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: global weighted mean
# ---------------------------------------------------------------------------

def _calculate_global_mean(data_array: xr.DataArray) -> xr.DataArray:
    """Compute area-weighted global mean."""
    lat_name = None
    for candidate in ('lat', 'latitude'):
        if candidate in data_array.coords:
            lat_name = candidate
            break
    if lat_name is None:
        raise ValueError("Latitude coordinate not found.")

    weights = np.cos(np.deg2rad(data_array[lat_name]))
    weights.name = "weights"

    spatial_dims = [
        d for d in data_array.dims
        if d not in ('time', 'bnds', 'time_bnds')
    ]
    return data_array.weighted(weights).mean(spatial_dims)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Configuration ---
    DATA_SOURCES = {
        "GDOIP Corrected": {
            "path": "/home/zju/hbb/Data/GDOIP-O2/"
                    "GT_ML_RF_ShipArgo_corrected.nc",
            "var_name": "o2",
            "unit": "μmol/kg",
        },
        "GDOIP Uncorrected": {
            "path": "/home/zju/hbb/Data/GDOIP-O2/"
                    "GT_ML_RF_ShipArgo_uncorrected.nc",
            "var_name": "o2",
            "unit": "μmol/kg",
        },
        "O2-Map SJTU": {
            "path": "/g12338011ghq/project/hbb/Dataset/GODIP-O2/"
                    "o2map_v1.2_5500m_sjtu_jingwei.nc",
            "var_name": "o2",
            "unit": "μmol·kg^-1",
        },
    }

    TIME_RANGE = [1970, 2014]  # set to None for full range
    TARGET_UNIT = "mol m-3"
    OUTPUT_DIR = "./results/external_data_comparison"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load, process, and convert units ---
    timeseries_data = {}
    print("--- Loading and processing external products ---")

    for label, info in DATA_SOURCES.items():
        print(f"  -> Processing: {label}")
        try:
            with xr.open_dataset(
                info['path'], decode_times=True, use_cftime=True
            ) as ds:
                if 'time' not in ds.coords or ds.time.size == 0:
                    print(f"     ... WARNING: No time dimension. Skip.")
                    continue

                data_var = ds[info['var_name']]

                # Time slice
                if TIME_RANGE:
                    ts = slice(str(TIME_RANGE[0]), str(TIME_RANGE[1]))
                    data_var = data_var.sel(time=ts)
                    if data_var.time.size == 0:
                        print(f"     ... WARNING: No data in range. Skip.")
                        continue

                # Surface level
                depth_dim = next(
                    (d for d in ('depth', 'lev', 'olevel')
                     if d in data_var.dims),
                    None
                )
                if depth_dim:
                    surface_data = data_var.isel({depth_dim: 0})
                else:
                    surface_data = data_var

                # Unit conversion: μmol/kg → mol/m³
                unit = info.get('unit', '').lower()
                if 'μmol' in unit and ('kg' in unit or 'l' in unit):
                    factor = 1025.0 / 1_000_000.0
                    surface_data = surface_data * factor
                    print(f"     ... Converted to {TARGET_UNIT}")

                # Global mean
                mean_ts = _calculate_global_mean(surface_data)
                timeseries_data[label] = mean_ts.load()

        except Exception as e:
            print(f"     ... ERROR: {e}")

    if not timeseries_data:
        print("No data loaded. Exiting.")
        return

    # --- Build aligned DataFrame ---
    series_list = []
    for label, ts in timeseries_data.items():
        try:
            index = ts.time.to_index().to_datetimeindex(unsafe=True)
            series_list.append(
                pd.Series(ts.values, index=index, name=label)
            )
        except Exception as e:
            print(f"Warning: time conversion failed for '{label}': {e}")

    df = pd.concat(series_list, axis=1)
    print(f"\nDataFrame with {len(df)} monthly steps.")

    # --- Plot ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(20, 14)
    )

    # (A) Monthly
    ax1.set_title(
        f'Monthly Mean Global Surface DO '
        f'({TIME_RANGE[0]}-{TIME_RANGE[1]})',
        fontsize=18,
    )
    if not df.empty:
        df.plot(ax=ax1, linewidth=1.2)
    ax1.set_ylabel(f'Mean DO ({TARGET_UNIT})', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.7)

    # (B) Annual
    ax2.set_title(
        f'Annual Mean Global Surface DO '
        f'({TIME_RANGE[0]}-{TIME_RANGE[1]})',
        fontsize=18,
    )
    if not df.empty:
        annual_df = df.resample('YE').mean()
        annual_df.plot(ax=ax2, style='-o', markersize=4, linewidth=2)
    ax2.set_xlabel('Year', fontsize=14)
    ax2.set_ylabel(f'Mean DO ({TARGET_UNIT})', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.7)

    plt.tight_layout()
    save_path = os.path.join(
        OUTPUT_DIR, "external_products_comparison.png"
    )
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
