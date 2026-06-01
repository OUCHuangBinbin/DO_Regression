"""
dataset.py — Data loading and preprocessing pipeline.

This module provides functions to:
  1. Load raw NetCDF files from CMIP6 models and observational sources
  2. Compute climatology and anomaly fields independently for each source
  3. Align all datasets to a common time axis
  4. Return stacked numpy arrays ready for reconstruction training

Key design: each model computes its own monthly climatology during a
reference period, and anomalies are computed relative to that climatology.
For true observations, WOA18 climatology is used.
"""

import os
import glob
import xarray as xr
import numpy as np
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# 1.  Weighted mean helpers
# ---------------------------------------------------------------------------

def calculate_weighted_mean(data_array: xr.DataArray) -> xr.DataArray:
    """
    Compute global area-weighted mean of a DataArray.

    Uses cosine(latitude) weights for physical correctness.
    Falls back to unweighted mean if latitude coordinate is missing.
    """
    if not hasattr(data_array, 'coords'):
        print("[WARN] Input to calculate_weighted_mean is a Numpy array. "
              "Returning unweighted mean.")
        return np.nanmean(data_array, axis=tuple(range(1, data_array.ndim)))

    if 'lat' not in data_array.coords:
        raise ValueError(
            "Latitude coordinate 'lat' not found for weighted mean calculation."
        )

    weights = np.cos(np.deg2rad(data_array.lat))
    weights.name = "weights"

    spatial_dims = [d for d in ['lon', 'lat'] if d in data_array.dims]
    if not spatial_dims:
        return data_array  # already a time series

    weighted_mean = data_array.weighted(weights).mean(spatial_dims)
    return weighted_mean


def _calculate_regional_mean(data_array: xr.DataArray) -> xr.DataArray:
    """
    Compute simple spatial arithmetic mean of a DataArray.
    """
    spatial_dims = [d for d in ['lon', 'lat'] if d in data_array.dims]
    if not spatial_dims:
        return data_array
    return data_array.mean(dim=spatial_dims)


# ---------------------------------------------------------------------------
# 2.  Single-file load & process
# ---------------------------------------------------------------------------

def _load_and_process_file(
    file_path: str,
    var_name: str,
    depth_level: int,
    time_slice: slice,
    is_obs_source: bool = False,
) -> Optional[xr.DataArray]:
    """
    Open, preprocess, and return a clean 2D (time × space) DataArray.

    Steps:
      - Decode CF-compliant times (cftime → datetimeindex)
      - Subset to the specified time range
      - Select surface level (depth=0)
      - Apply quality control for true observation sources
    """
    print(f"  -> Opening file: {os.path.basename(file_path)}")
    try:
        with xr.open_dataset(file_path, chunks=None, decode_times=False) as ds:
            ds = xr.decode_cf(ds, use_cftime=True)
            if isinstance(ds.time.to_index(),
                          xr.coding.cftimeindex.CFTimeIndex):
                ds['time'] = ds.indexes['time'].to_datetimeindex(
                    unsafe=True)

            ds_sliced = ds.sel(time=time_slice)
            if ds_sliced.time.size == 0:
                print(
                    f"     ... WARNING: No data in time range for "
                    f"{os.path.basename(file_path)}. Skipping."
                )
                return None

            data_var = ds_sliced[var_name]
            data_var = data_var.where(data_var >= 0)

            depth_dim_name = next(
                (d for d in ['lev', 'olevel', 'depth']
                 if d in data_var.dims),
                None
            )

            if depth_dim_name and data_var.sizes[depth_dim_name] > 1:
                surface_data = data_var.isel({depth_dim_name: depth_level})
            else:
                surface_data = data_var

            # Remove singleton dimensions
            dims_to_drop = [
                d for d in ['lev', 'olevel', 'depth']
                if d in surface_data.dims and surface_data.sizes[d] == 1
            ]
            if dims_to_drop:
                surface_data = surface_data.squeeze(dim=dims_to_drop,
                                                     drop=True)

            # Quality control for true observations (outlier removal)
            if is_obs_source:
                surface_data = _apply_obs_qc(surface_data)

            return surface_data

    except Exception as e:
        print(f"     ... WARNING: Failed to process "
              f"{os.path.basename(file_path)}. Error: {e}. Skipping.")
        return None


def _apply_obs_qc(da: xr.DataArray) -> xr.DataArray:
    """
    Apply outlier filtering to observational data.

    Steps:
      1. Remove non-positive values
      2. Z-score filter (±4σ over time)
      3. Quantile clipping (0.1% – 99.9%)
    """
    print("     ... Applying quality control for observation source.")

    # 1. Remove non-physical values
    da = da.where(da > 0)

    # 2. Z-score outlier removal (±4σ)
    mean_over_time = da.mean(dim='time')
    std_over_time = da.std(dim='time')
    z_threshold = 4.0
    upper_limit = mean_over_time + z_threshold * std_over_time
    lower_limit = mean_over_time - z_threshold * std_over_time

    original_count = da.count().item()
    da = da.where((da >= lower_limit) & (da <= upper_limit))
    filtered = da.count().item()
    print(f"     ... Z-score QC removed "
          f"{original_count - filtered} data points.")

    # 3. Quantile clipping
    lower_q = da.quantile(0.001, skipna=True)
    upper_q = da.quantile(0.999, skipna=True)
    print(f"     ... Quantile range: [{float(lower_q):.4f}, "
          f"{float(upper_q):.4f}]")

    da = da.where((da >= lower_q) & (da <= upper_q))

    return da


# ---------------------------------------------------------------------------
# 3.  WOA18 climatology loader
# ---------------------------------------------------------------------------

def _load_woa_climatology(woa_cfg: Dict) -> xr.DataArray:
    """
    Load WOA18 monthly climatology (12 months × lat × lon).

    Each monthly file is opened, the surface field is extracted,
    and all 12 months are concatenated along a 'month' dimension.
    """
    print("     -> Loading WOA18 monthly climatology...")
    woa_monthly = []

    for month_num in range(1, 13):
        month_str = f"{month_num:02d}"
        file_pattern = woa_cfg['path_pattern'].replace('*', month_str)
        found_files = glob.glob(file_pattern)

        if not found_files:
            raise FileNotFoundError(
                f"No WOA file for month {month_num}: {file_pattern}"
            )

        with xr.open_dataset(found_files[0], decode_times=False) as ds:
            raw = ds[woa_cfg['var_name']]
            if 'depth' in raw.dims:
                raw = raw.isel(depth=0, drop=True)
            raw = raw.squeeze(drop=True)
            woa_monthly.append(raw)

    clim = xr.concat(woa_monthly, dim='month')
    clim = clim.assign_coords(month=np.arange(1, 13))

    factor = woa_cfg.get('unit_conversion_factor', 1.0)
    if factor != 1.0:
        print(f"     -> Applying unit conversion ({factor}) to WOA.")
        clim = clim * factor

    print(f"     -> WOA climatology loaded. Shape: {clim.shape}")
    return clim


# ---------------------------------------------------------------------------
# 4.  Main orchestration
# ---------------------------------------------------------------------------

def load_and_prepare_data(config: Dict) -> Dict:
    """
    Load all data sources and prepare arrays for training and reconstruction.

    The pipeline follows these steps:
      1. Load raw training model fields, observation source fields, and
         true observation fields
      2. Compute independent climatologies and anomaly fields for each source
      3. Match variance of true observation anomalies to training anomalies
      4. Align all anomaly fields to a common time axis via reindex
      5. Stack and return as numpy arrays

    Returns a dictionary containing all arrays needed by the reconstructor.
    """
    data_cfg = config['data']
    inject_uncertainty = data_cfg.get('inject_uncertainty', False)

    time_slice = slice(
        f"{data_cfg['time_range'][0]}-01-01",
        f"{data_cfg['time_range'][1]}-12-31"
    )
    ref_period_slice = slice(
        f"{data_cfg['anomaly_reference_period'][0]}-01-01",
        f"{data_cfg['anomaly_reference_period'][1]}-12-31"
    )

    print(f"--- Using time range: "
          f"{data_cfg['time_range'][0]}-{data_cfg['time_range'][1]} ---")

    # ---- Step 1: Load all raw data sources ----
    print("\n--- [Step 1] Loading all raw data sources ---")
    training_dirs = data_cfg.get('training_model_dirs', [])
    all_nc_files = []
    for directory in training_dirs:
        all_nc_files.extend(
            glob.glob(os.path.join(directory, "**", "*.nc"), recursive=True)
        )
    all_nc_files = sorted(set(all_nc_files))

    # Exclude the observation files from the training file set
    paths_to_exclude = {
        os.path.abspath(data_cfg['observation_source_model_path']),
        os.path.abspath(data_cfg['observation_source_true_path']),
    }
    training_model_files = [
        f for f in all_nc_files
        if os.path.abspath(f) not in paths_to_exclude
    ]

    var_name = data_cfg.get('variable_name', 'o2')
    depth_level = data_cfg['depth_level']

    training_raw = [
        _load_and_process_file(f, var_name, depth_level, time_slice,
                               is_obs_source=False)
        for f in training_model_files
    ]
    training_raw = [ds for ds in training_raw if ds is not None]
    print(f"--- {len(training_raw)} training files loaded ---")

    # Observation source (model-based — for OSSE validation)
    obs_model_raw = _load_and_process_file(
        data_cfg['observation_source_model_path'],
        var_name, depth_level, time_slice,
        is_obs_source=False
    )

    # True observation source
    true_obs_raw = _load_and_process_file(
        data_cfg['observation_source_true_path'],
        data_cfg.get('true_obs_variable_name', 'o2'),
        depth_level, time_slice,
        is_obs_source=True
    )

    # Build dynamic observation mask from true obs
    print("\n--- [Step 1.5] Creating dynamic observation mask ---")
    obs_mask = xr.where(np.isnan(true_obs_raw), False, True)
    obs_mask.name = "dynamic_mask"
    print(f"     -> Mask shape: {obs_mask.shape}")
    base_time_coord = obs_mask.time
    print(f"     -> Using {len(base_time_coord)} timesteps from mask.")

    # Load bias ensemble (optional, for uncertainty injection)
    bias_ensemble_da = None
    if inject_uncertainty:
        print("--- UNCERTAINTY INJECTION: ENABLED ---")
        bias_path = data_cfg.get('bias_ensemble_path')
        if bias_path and os.path.exists(bias_path):
            print("     -> Loading bias ensemble...")
            bias_raw = _load_and_process_file(
                bias_path, 'o2_bias', -1, time_slice
            )
            if bias_raw is not None:
                bias_ensemble_da = bias_raw.reindex(
                    time=base_time_coord, method='nearest'
                )
                print(f"     -> Bias ensemble aligned. "
                      f"Shape: {bias_ensemble_da.shape}")
    else:
        print("--- UNCERTAINTY INJECTION: DISABLED ---")

    # Load WOA18 climatology
    print("\n--- [Step 1.8] Loading WOA18 climatology ---")
    woa_cfg = data_cfg.get('woa18_climatology')
    if not woa_cfg:
        raise ValueError(
            "Config 'woa18_climatology' missing in YAML."
        )
    woa_clim = _load_woa_climatology(woa_cfg)

    if not training_raw or obs_model_raw is None or true_obs_raw is None:
        raise ValueError("Essential data components failed to load.")

    # ---- Step 2: Compute independent anomalies ----
    print("\n--- [Step 2] Computing anomalies per source ---")

    # 2a. Training models: each model's monthly climatology
    #     from reference period → anomaly = raw − climatology
    training_anom_list = []
    for ds_raw in training_raw:
        clim = (
            ds_raw.sel(time=ref_period_slice)
            .groupby('time.month')
            .mean('time')
        )
        anom = ds_raw.groupby('time.month') - clim
        training_anom_list.append(anom)
    print("     -> Training data anomalies computed.")

    # 2b. Model-based observation source
    obs_model_clim = (
        obs_model_raw
        .sel(time=ref_period_slice)
        .groupby('time.month')
        .mean('time')
    )
    obs_model_anom = obs_model_raw.groupby('time.month') - obs_model_clim

    obs_model_clim_mean = calculate_weighted_mean(
        obs_model_clim.mean('month')
    )
    print(f"     -> Obs model climatology mean: "
          f"{float(obs_model_clim_mean.compute()):.6f}")

    # 2c. True observations: subtract WOA18 climatology
    true_obs_processed = true_obs_raw.where(true_obs_raw > 0)
    true_obs_converted = true_obs_processed * data_cfg.get(
        'unit_conversion_factor', 1.0
    )

    def _subtract_woa_clim(monthly_block, climatology=woa_clim):
        month = monthly_block['time.month'][0].item()
        clim = climatology.sel(month=month)
        return monthly_block - clim

    print("     -> Computing true obs anomalies via WOA subtraction...")
    true_obs_anom = true_obs_converted.groupby('time.month').apply(
        _subtract_woa_clim
    )

    true_obs_clim_mean = calculate_weighted_mean(woa_clim.mean('month'))
    print(f"     -> WOA climatology mean: "
          f"{float(true_obs_clim_mean.compute()):.6f}")

    # ---- Step 3: Match variance of true obs anomaly to training ----
    print("\n--- [Step 3] Matching variance ---")
    temp_train = xr.concat(training_anom_list, dim='model')
    train_anom_std = float(temp_train.std().compute())
    true_anom_std = float(true_obs_anom.std().compute())

    if true_anom_std > 1e-9:
        true_obs_anom_rescaled = (
            true_obs_anom / true_anom_std * train_anom_std
        )
        print(f"     -> Rescaled true obs std: {true_anom_std:.6f} → "
              f"{train_anom_std:.6f}")
    else:
        print("     -> WARNING: True obs std ≈ 0. Skipping rescale.")
        true_obs_anom_rescaled = true_obs_anom

    # ---- Step 4: Align all anomaly fields ----
    print("\n--- [Step 4] Aligning to common time axis ---")
    aligned_training = [
        ds.reindex(time=base_time_coord, method='nearest')
        for ds in training_anom_list
    ]
    aligned_obs_model = obs_model_anom.reindex(
        time=base_time_coord, method='nearest'
    )
    aligned_true_obs = true_obs_anom_rescaled.reindex(
        time=base_time_coord, method='nearest'
    )
    aligned_mask = obs_mask

    if aligned_training:
        print(f"     -> All datasets aligned. Length: "
              f"{len(aligned_training[0].time)} timesteps.")

    # ---- Step 5: Prepare training arrays ----
    print("\n--- [Step 5] Preparing arrays ---")

    # 5a. Training target Y (area-weighted mean anomaly)
    training_da_aligned = xr.concat(aligned_training, dim='model')
    Y_train_truth = calculate_weighted_mean(training_da_aligned)

    # 5b. Training features X (with optional uncertainty injection)
    training_da_for_X = training_da_aligned

    # 5c. Groups and training mask
    groups_list = [
        np.full(len(ds.time), i)
        for i, ds in enumerate(aligned_training)
    ]
    training_groups = np.concatenate(groups_list)
    training_mask_broadcasted, _ = xr.broadcast(
        aligned_mask, training_da_for_X
    )

    if groups_list:
        unique_groups = np.unique(training_groups)
        print(f"     -> {len(unique_groups)} unique groups: "
              f"{unique_groups}")

    # ---- Step 6: Normalization statistics ----
    print("\n--- [Step 6] Computing normalization stats ---")
    stats_base = xr.concat(training_raw, dim='sample')
    mean_stat = float(stats_base.mean().compute())
    std_stat = float(stats_base.std().compute())
    print(f"     -> Mean: {mean_stat:.4f}, Std: {std_stat:.4f}")

    # ---- Step 7: Stack and return ----
    print("\n--- [Step 7] Returning arrays ---")

    n_samples = (
        training_da_for_X.sizes['model']
        * training_da_for_X.sizes['time']
    )

    return {
        "training_da_values": training_da_for_X
            .stack(sample=('model', 'time'))
            .transpose('sample', 'lat', 'lon')
            .compute().values,

        "training_groups": training_groups,

        "Y_train_truth_values": Y_train_truth
            .stack(sample=('model', 'time'))
            .transpose()
            .compute().values,

        "training_mask_da_values": (
            training_mask_broadcasted
            .stack(sample=('model', 'time'))
            .transpose('sample', 'lat', 'lon')
            .compute().values > 0
        ),

        "observation_source_model_da_anom_values":
            aligned_obs_model.compute().values,

        "Y_obs_source_truth_anom_values":
            calculate_weighted_mean(aligned_obs_model).compute().values,

        "observation_source_true_da_anom_values":
            aligned_true_obs.compute().values,

        "true_obs_climatology_regional_mean":
            float(true_obs_clim_mean.compute()),

        "obs_model_climatology_regional_mean":
            float(obs_model_clim_mean.compute()),

        "obs_mask_da_values":
            (aligned_mask.compute().values > 0),

        "time_coord": aligned_mask.time.values,
        "coords": {
            "lat": aligned_mask.lat.values,
            "lon": aligned_mask.lon.values,
        },
        "stats": {"mean": mean_stat, "std": std_stat},
        "train_anom_std": train_anom_std,

        "inject_uncertainty": inject_uncertainty,
        "bias_ensemble_values": (
            bias_ensemble_da.values
            if bias_ensemble_da is not None else None
        ),
        "woa_clim": woa_clim,
    }
