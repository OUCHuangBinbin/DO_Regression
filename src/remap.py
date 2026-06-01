"""
remap.py — Resample NetCDF files to a coarser target resolution.

Supports:
  - Individual WOA climatology files (monthly)
  - CMIP6 model directories
  - Single NetCDF files (e.g. gridded observation products)

Uses xarray's coarsen().mean() with boundary='trim'.
"""

import os
import glob
import xarray as xr
import numpy as np


def resample_woa_files_individually(
    source_pattern: str,
    output_dir: str,
    target_resolution: int,
    var_name: str = 'o_an',
) -> None:
    """
    Resample individual WOA monthly climatology files to target resolution.
    """
    print(f"--- Resampling WOA to {target_resolution}-degree ---")

    files = sorted(glob.glob(source_pattern))
    if not files:
        print(f"[ERROR] No files matching: {source_pattern}")
        return

    os.makedirs(output_dir, exist_ok=True)

    for fpath in files:
        fname = os.path.basename(fpath)
        out_path = os.path.join(output_dir, fname)
        print(f"  -> {fname}")

        try:
            with xr.open_dataset(fpath, decode_times=False) as ds:
                da = ds[var_name]
                depth_dim = next(
                    (d for d in ('lev', 'olevel', 'depth')
                     if d in da.dims),
                    None
                )
                if depth_dim and da.sizes[depth_dim] > 1:
                    da = da.isel({depth_dim: 0})

                da_resampled = da.coarsen(
                    lon=target_resolution,
                    lat=target_resolution,
                    boundary='trim',
                ).mean()

                ds_out = da_resampled.to_dataset(name=var_name)
                ds_out.attrs = ds.attrs

                encoding = {
                    var_name: {
                        'dtype': 'float32',
                        '_FillValue': -9999.0,
                        'zlib': True,
                        'complevel': 4,
                    }
                }
                ds_out.to_netcdf(out_path, encoding=encoding)
                print(f"     ... saved to {out_path}")

        except Exception as e:
            print(f"     ... [ERROR] {e}")

    print("--- WOA resampling complete ---")


def resample_dataset(
    source_dir: str,
    output_dir: str,
    target_resolution: int,
    file_pattern: str = "*.nc",
) -> None:
    """
    Resample all NetCDF files in a directory to the target resolution.
    """
    if not os.path.exists(source_dir):
        print(f"[ERROR] Source dir not found: {source_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Resampling {source_dir} to {target_resolution}-degree ---")

    files = glob.glob(os.path.join(source_dir, file_pattern))
    for fpath in files:
        fname = os.path.basename(fpath)
        out_path = os.path.join(output_dir, fname)
        print(f"  -> {fname}")

        try:
            with xr.open_dataset(fpath) as ds:
                ds_resampled = ds.coarsen(
                    lon=target_resolution,
                    lat=target_resolution,
                    boundary='trim',
                ).mean()
                ds_resampled.to_netcdf(out_path)
                print(f"     ... saved")
        except Exception as e:
            print(f"     ... [ERROR] {e}")


def resample_netcdf_file(
    source_path: str,
    output_path: str,
    target_resolution: int,
    var_name: str = None,
    treat_zero_as_nan: bool = False,
) -> None:
    """
    Resample a single NetCDF file to target resolution.

    If treat_zero_as_nan is True, values <= 0 are set to NaN before
    averaging (useful for products that use 0 as a fill value).
    """
    print(f"  -> Processing: {os.path.basename(source_path)}")

    try:
        with xr.open_dataset(source_path, decode_times=False) as ds:
            if var_name is None:
                data_vars = [v for v in ds.data_vars if 'bnds' not in v]
                var_name = data_vars[0]
                print(f"     ... auto-detected variable: '{var_name}'")

            da = ds[var_name]

            if treat_zero_as_nan:
                print("     ... treating values <= 0 as NaN")
                da = da.where(da > 0)

            ds_resampled = da.coarsen(
                lon=target_resolution,
                lat=target_resolution,
                boundary='trim',
            ).mean()

            ds_out = ds_resampled.to_dataset(name=var_name)
            ds_out.attrs = ds.attrs

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ds_out.to_netcdf(output_path)
            print(f"     ... saved to {output_path}")

    except Exception as e:
        print(f"     ... [ERROR] {e}")


if __name__ == "__main__":
    # Example: resample a WOD file to 5 degrees
    resample_netcdf_file(
        source_path=(
            "/g12338011ghq/dataset/WOD/Type_5deg/"
            "WOD_O2_Type_Ship.nc"
        ),
        output_path=(
            "/g12338011ghq/dataset/WOD/Type_5deg/"
            "WOD_O2_Type_Ship_5deg.nc"
        ),
        target_resolution=5,
        var_name='o2',
    )
