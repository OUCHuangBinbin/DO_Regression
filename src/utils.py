"""
utils.py — Shared utilities for the oxygen reconstruction pipeline.

Provides:
  - Mask handling (standardize, group, flatten)
  - Training matrix construction (with optional uncertainty injection)
  - Time-series reconstruction from trained models
  - Evaluation metrics (RMSE, correlation, R²)
  - Visualization (time-series comparison, uncertainty bands, beta maps)
  - Signal processing (Butterworth filter)
  - I/O (Parquet / CSV save/load)
"""

import os
import pickle
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import yaml
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


# ======================================================================
# 1.  Mask helpers
# ======================================================================

def standardize_mask(mask: np.ndarray) -> np.ndarray:
    """Normalise mask to contiguous boolean array (mask > 0)."""
    return np.ascontiguousarray(mask > 0, dtype=bool)


def flatten_field_with_mask(
    field_2d: np.ndarray,
    mask_2d: np.ndarray,
) -> np.ndarray:
    """
    Extract 1-D vector from 2-D field at positions where mask > 0.

    Auto-squeezes leading singleton dimensions for robustness.
    """
    field = np.asarray(field_2d)
    mask = np.asarray(mask_2d)

    for arr in (field, mask):
        if arr.ndim == 3 and 1 in arr.shape:
            arr = np.squeeze(arr)

    if field.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            f"Expected 2-D arrays after squeeze, got "
            f"field {field.shape}, mask {mask.shape}"
        )
    if field.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: {field.shape} vs {mask.shape}"
        )

    return field[mask > 0]


def build_mask_groups(
    mask_da: xr.DataArray,
) -> Tuple[Dict, Dict]:
    """
    Group time steps by their unique observation pattern.

    Returns:
        hash_to_mask_map:
            {hash: {"mask_2d": (lat, lon) bool array,
                    "time_indices": list[int]}}
        groups_grouped:
            {hash: np.ndarray of time indices}
    """
    if "time" not in mask_da.dims:
        mask_2d = (mask_da > 0)
        h = "single"
        return (
            {h: {"mask_2d": mask_2d, "time_indices": None}},
            {h: np.array([0])},
        )

    da = mask_da
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)

    n_time = da.sizes["time"]
    hash_to_mask_map = {}
    groups_grouped = {}

    for t in range(n_time):
        m_t = (da.isel(time=t) > 0)
        pattern = m_t.values.astype(np.uint8).flatten()
        h = hash(pattern.tobytes())

        if h not in hash_to_mask_map:
            hash_to_mask_map[h] = {
                "mask_2d": m_t,
                "time_indices": [t],
            }
            groups_grouped[h] = np.array([t])
        else:
            hash_to_mask_map[h]["time_indices"].append(t)
            groups_grouped[h] = np.append(groups_grouped[h], t)

    return hash_to_mask_map, groups_grouped


# ======================================================================
# 2.  Global weighted mean
# ======================================================================

def _calculate_global_mean(data_array: xr.DataArray) -> xr.DataArray:
    """
    Compute area-weighted global mean.

    Handles both 'lat' and 'latitude' coordinate names.
    """
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


# ======================================================================
# 3.  Training matrix assembly (with optional uncertainty injection)
# ======================================================================

def build_training_matrices(
    data: dict,
    min_features: int = 5,
) -> tuple:
    """
    Build per-mask training matrices from the stacked numpy arrays
    produced by dataset.load_and_prepare_data().

    Each sample is assigned to a group based on its observation mask hash.
    Optionally, noise from a bias ensemble is injected into the feature
    vectors to represent observational uncertainty.

    Returns:
        X_grouped, Y_grouped, groups_grouped, hash_to_mask_map
    """
    print("\n--- [build_training_matrices] ---")

    X_stacked = data["training_da_values"]       # (n_samples, lat, lon)
    Y_stacked = data["Y_train_truth_values"]      # (n_samples,)
    groups_stacked = data["training_groups"]      # (n_samples,)
    mask_stacked = data["training_mask_da_values"]  # (n_samples, lat, lon)

    inject_uncertainty = data.get("inject_uncertainty", False)
    bias_ensemble = data.get("bias_ensemble_values", None)

    if inject_uncertainty and bias_ensemble is None:
        print("[WARN] Uncertainty requested but no bias ensemble "
              "available. Proceeding clean.")
        inject_uncertainty = False

    n_bias_members = (
        bias_ensemble.shape[0] if inject_uncertainty else 0
    )
    time_len = bias_ensemble.shape[1] if inject_uncertainty else 0
    n_samples = X_stacked.shape[0]

    print(f"     n_samples={n_samples}, "
          f"inject_uncertainty={inject_uncertainty}")

    X_grouped = defaultdict(list)
    Y_grouped = defaultdict(list)
    groups_grouped = defaultdict(list)
    hash_to_mask_map = {}

    for i in tqdm(range(n_samples), desc="Grouping samples"):
        mask_2d = standardize_mask(mask_stacked[i])

        if np.sum(mask_2d) < min_features:
            continue

        mask_hash = hash(mask_2d.tobytes())

        if mask_hash not in hash_to_mask_map:
            hash_to_mask_map[mask_hash] = mask_2d

        # Feature vector
        field_2d = X_stacked[i]
        x_vec = np.nan_to_num(field_2d[mask_2d])

        # Optional uncertainty injection
        if inject_uncertainty:
            member_idx = np.random.randint(0, n_bias_members)
            t_idx = i % time_len
            bias_field = bias_ensemble[member_idx, t_idx, :, :]
            bias_vec = np.nan_to_num(bias_field[mask_2d])
            x_vec = x_vec + bias_vec

        X_grouped[mask_hash].append(x_vec)
        Y_grouped[mask_hash].append(Y_stacked[i])
        groups_grouped[mask_hash].append(groups_stacked[i])

    # Second pass: convert lists to arrays and apply validity checks
    print("\n     -> Final filtering of grouped data...")
    final_X, final_Y, final_G, final_map = {}, {}, {}, {}

    for h, x_list in tqdm(X_grouped.items(), desc="Filtering groups"):
        X_arr = np.array(x_list)
        if X_arr.ndim != 2 or X_arr.shape[1] == 0:
            print(f"[WARN] Skipping mask {h}: bad feature dims.")
            continue

        unique_g = np.unique(groups_grouped[h])
        if len(unique_g) < 2:
            print(f"[WARN] Skipping mask {h}: "
                  f"only {len(unique_g)} group(s) for CV.")
            continue

        final_X[h] = X_arr
        final_Y[h] = np.array(Y_grouped[h])
        final_G[h] = np.array(groups_grouped[h])
        final_map[h] = hash_to_mask_map[h]

    print(f"     -> {len(final_X)} valid mask(s).")
    return final_X, final_Y, final_G, final_map


def ensure_training_anomaly(data: dict) -> xr.DataArray:
    """
    Return anomaly-field training data,
    preferring 'training_da_anom' or falling back to 'training_da'.
    """
    if "training_da_anom" in data:
        print("[INFO] Using 'training_da_anom'.")
        return data["training_da_anom"]

    if "training_da" in data:
        print("[INFO] Building anomaly from 'training_da' "
              "(subtract time mean).")
        clim = data["training_da"].mean(dim="time")
        return data["training_da"] - clim

    raise KeyError(
        "Neither 'training_da_anom' nor 'training_da' in data dict."
    )


# ======================================================================
# 4.  Reconstruction
# ======================================================================

def reconstruct_timeseries(
    model_library: dict,
    hash_to_mask_map: dict,
    source_values_anom: np.ndarray,
    mask_values: np.ndarray,
    source_name: str = "Unknown Source",
) -> np.ndarray:
    """
    Reconstruct a global-mean time-series from gridded anomaly fields
    using a pre-trained model library.

    For each time step the observation mask is hashed, the matching
    pre-trained model is looked up, and a prediction is made from the
    observed feature vector.
    """
    n_time = source_values_anom.shape[0]
    recon = np.full(n_time, np.nan)

    print(f"\n--- Reconstructing '{source_name}' ---")

    # Pre-compute hashes for every time step
    time_hashes = []
    for t in range(n_time):
        mask_2d = standardize_mask(mask_values[t])
        if np.any(mask_2d):
            time_hashes.append(hash(mask_2d.tobytes()))
        else:
            time_hashes.append(None)

    processed = 0
    for mask_hash, pipeline in tqdm(
        model_library.items(), desc=f"Reconstructing {source_name}"
    ):
        idx = [i for i, h in enumerate(time_hashes) if h == mask_hash]
        if not idx:
            continue

        batch = source_values_anom[idx]
        mask_2d = hash_to_mask_map[mask_hash]
        X_batch = np.array([
            np.nan_to_num(field[mask_2d]) for field in batch
        ])

        if X_batch.shape[0] > 0:
            pred = pipeline.predict(X_batch)
            recon[idx] = pred
            processed += len(idx)

    print(f"     -> {processed} / {n_time} steps reconstructed.")
    return recon


# ======================================================================
# 5.  Evaluation metrics
# ======================================================================

def evaluate_reconstruction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    series_name: str,
) -> dict:
    """
    Compute RMSE, Pearson correlation, and R² between two time-series.

    Returns dict with keys: rmse, correlation, r_squared.
    If fewer than 2 valid points, returns NaN metrics.
    """
    print(f"\n--- Evaluating: {series_name} ---")
    valid = ~np.isnan(y_true) & ~np.isnan(y_pred)

    if np.sum(valid) < 2:
        print("     -> Too few valid points.")
        return {"rmse": np.nan, "correlation": np.nan, "r_squared": np.nan}

    yt, yp = y_true[valid], y_pred[valid]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r = float(np.corrcoef(yt, yp)[0, 1])

    print(f"     -> RMSE = {rmse:.8f},  r = {r:.4f}")
    return {
        "rmse": rmse,
        "correlation": r,
        "r_squared": r ** 2,
    }


# ======================================================================
# 6.  Regional-mean helper
# ======================================================================

def compute_regional_mean_timeseries(
    da: xr.DataArray,
    mask_da: Optional[xr.DataArray] = None,
) -> np.ndarray:
    """
    Compute area-weighted regional-mean time-series.

    Handles dimensions (time, lat, lon) → (time,),
    (model, time, lat, lon) → (model, time), or (lat, lon) → scalar.
    """
    rename = {}
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if rename:
        da = da.rename(rename)

    if mask_da is not None:
        m = mask_da
        if "time" in m.dims and "time" not in da.dims:
            m = m.mean("time") > 0
        if "time" in da.dims and "time" not in m.dims:
            m = m.broadcast_like(da.isel(time=0))
        da = da.where(m > 0)

    if "lat" in da.dims:
        lat_rad = np.deg2rad(da["lat"])
        weights = np.cos(lat_rad)
        weights = weights / weights.mean()
    else:
        weights = None

    if "lat" in da.dims and "lon" in da.dims:
        if "model" in da.dims and "time" in da.dims:
            return (
                da.weighted(weights)
                .mean(dim=("lat", "lon"))
                .values
            )
        elif "time" in da.dims:
            return (
                da.weighted(weights)
                .mean(dim=("lat", "lon"))
                .values
            )
        else:
            return float(
                da.weighted(weights).mean(dim=("lat", "lon")).values
            )

    return float(da.mean().values)


# ======================================================================
# 7.  Uncertainty weights
# ======================================================================

def calculate_uncertainty_weights(
    mask_da_values: np.ndarray,
    baseline_coverage: float = 0.10,
    smoothing_window: int = 60,
) -> np.ndarray:
    """
    Compute time-dependent uncertainty scaling weights based on
    observational coverage.

    Fewer observations → larger weight → wider uncertainty bands.
    Weights are smoothed in time and clipped to [0.1, 5.0].
    An era factor linearly decreasing from 1.0 to 0.1 is applied.
    """
    n_points_raw = np.sum(mask_da_values > 0, axis=(1, 2))
    n_points_smooth = pd.Series(n_points_raw).rolling(
        window=smoothing_window, center=True, min_periods=1
    ).mean().values

    total_ocean = np.sum(np.any(mask_da_values > 0, axis=0))
    total_ocean = max(total_ocean, 1)

    coverage = n_points_smooth / total_ocean
    eps = 1e-6
    weights = np.sqrt(baseline_coverage / (coverage + eps))

    era_factor = np.linspace(1.0, 0.1, len(weights))
    weights = weights * era_factor
    weights = np.clip(weights, 0.1, 5.0)

    return weights


# ======================================================================
# 8.  Signal processing
# ======================================================================

def apply_filter(
    series: np.ndarray,
    cutoff_period: float,
    filter_type: str,
    sample_rate: float = 1.0,
) -> np.ndarray:
    """
    Apply a Butterworth filter (low-pass or high-pass) to a time-series.

    Args:
        cutoff_period: Cutoff period in the same units as sample_rate.
        filter_type: 'low' or 'high'.
        sample_rate: Sampling frequency (1.0 for annual, 12.0 for monthly).

    Returns:
        Filtered series (NaN positions preserved).
    """
    if filter_type not in ('low', 'high'):
        raise ValueError("filter_type must be 'low' or 'high'")

    series_pd = pd.Series(series)
    series_interp = series_pd.interpolate(
        method='linear', limit_direction='both'
    ).values
    nan_mask = np.isnan(series)

    nyquist = 0.5 * sample_rate
    cutoff_norm = (1.0 / cutoff_period) / nyquist
    b, a = butter(N=4, Wn=cutoff_norm, btype=filter_type)
    filtered = filtfilt(b, a, series_interp)
    filtered[nan_mask] = np.nan

    return filtered


# ======================================================================
# 9.  I/O — Parquet
# ======================================================================

def save_results_to_parquet(
    filepath: str,
    time_coord: np.ndarray,
    results_dict: dict,
    config: dict,
) -> None:
    """
    Save experiment results to Parquet + companion YAML metadata.
    """
    print(f"\n--- Saving results to {os.path.basename(filepath)} ---")
    df = pd.DataFrame(results_dict, index=pd.to_datetime(time_coord))

    try:
        df.to_parquet(
            filepath, engine='pyarrow', compression='gzip'
        )
        meta_path = filepath.replace('.parquet', '_meta.yaml')
        with open(meta_path, 'w') as f:
            yaml.dump(config, f)
        print(f"     -> Data saved to {filepath}")
        print(f"     -> Metadata saved to {meta_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save Parquet: {e}")


def load_results_from_parquet(filepath: str) -> Tuple[pd.DataFrame, dict]:
    """Load results from Parquet + companion YAML metadata."""
    print(f"\n--- Loading results from {os.path.basename(filepath)} ---")
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    df = pd.read_parquet(filepath)
    config = {}
    meta_path = filepath.replace('.parquet', '_meta.yaml')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            config = yaml.safe_load(f)

    return df, config


# ======================================================================
# 10.  Visualisation — time-series
# ======================================================================

def visualize_reconstruction_timeseries(
    time_coord: np.ndarray,
    save_path: str,
    title: str,
    ylabel: str = "Mean DO Anomaly",
    timeseries_dict: Optional[Dict[str, np.ndarray]] = None,
    colors_dict: Optional[Dict[str, str]] = None,
    is_annual_data: bool = False,
) -> None:
    """
    Flexible time-series comparison plot (monthly + annual panels, or
    just annual if is_annual_data is True).
    """
    if timeseries_dict is None:
        timeseries_dict = {}

    valid = {
        k: v for k, v in timeseries_dict.items()
        if v is not None and not np.all(np.isnan(v))
    }
    if not valid:
        print(f"[WARN] No valid data for "
              f"'{os.path.basename(save_path)}'. Skipping.")
        return

    if colors_dict is None:
        colors_dict = {}

    # Auto-colour for series without a specified colour
    auto_colors = cm.viridis(np.linspace(0, 1, len(valid)))
    final_colors = {}
    ci = 0
    for name in valid:
        if name in colors_dict:
            final_colors[name] = colors_dict[name]
        else:
            if 'reconstruction' in name.lower() or 'recon' in name.lower():
                final_colors[name] = colors_dict.get(
                    "My Reconstruction", "red"
                )
            else:
                final_colors[name] = auto_colors[ci]
                ci += 1

    plt.style.use('seaborn-v0_8-whitegrid')

    if is_annual_data:
        fig, ax = plt.subplots(figsize=(20, 10))
        fig.suptitle(title, fontsize=22, y=0.95)

        for name, series in valid.items():
            s = pd.Series(series, index=time_coord).dropna()
            ax.plot(s.index, s.values, label=name,
                    color=final_colors[name],
                    linewidth=2.0, marker='o', markersize=4)

        ax.set_title('Annual Mean Comparison', fontsize=18)
        ax.set_xlabel('Year', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, which='both', linestyle='--')
        ax.ticklabel_format(style='plain', axis='y', useOffset=False)
    else:
        fig, axes = plt.subplots(nrows=2, ncols=1,
                                 figsize=(20, 15), sharex=False)
        fig.suptitle(title, fontsize=22, y=0.95)

        ax1 = axes[0]
        df = pd.DataFrame(valid, index=pd.to_datetime(time_coord))
        for name in df.columns:
            s = df[name].dropna()
            ax1.plot(s.index, s.values, label=name,
                     color=final_colors[name], linewidth=1.2)
        ax1.set_title('Monthly Mean Comparison', fontsize=18)
        ax1.set_ylabel(ylabel, fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, which='both', linestyle='--')
        ax1.ticklabel_format(style='plain', axis='y', useOffset=False)

        ax2 = axes[1]
        annual_df = df.resample('YE').mean()
        for name in annual_df.columns:
            s = annual_df[name].dropna()
            ax2.plot(s.index.year, s.values, label=name,
                     color=final_colors[name],
                     linewidth=2.0, marker='o', markersize=4)
        ax2.set_title('Annual Mean Comparison', fontsize=18)
        ax2.set_xlabel('Year', fontsize=14)
        ax2.set_ylabel(ylabel, fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, which='both', linestyle='--')
        ax2.ticklabel_format(style='plain', axis='y', useOffset=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    plt.show()
    plt.close(fig)


def visualize_with_uncertainty(
    time_coord: np.ndarray,
    save_path: str,
    title: str,
    ylabel: str,
    median_line: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    other_lines_dict: Optional[Dict[str, np.ndarray]] = None,
    colors_dict: Optional[Dict[str, str]] = None,
) -> None:
    """
    Visualise reconstruction with uncertainty shading (2 panels:
    monthly and annual) plus optional reference curves.
    """
    print(f"\n--- Generating uncertainty plot: "
          f"{os.path.basename(save_path)} ---")

    other = other_lines_dict or {}
    data = {
        'Median': median_line,
        'Lower': lower_bound,
        'Upper': upper_bound,
    }
    data.update(other)

    df = pd.DataFrame(data, index=pd.to_datetime(time_coord))
    annual_df = df.resample('YE').mean()

    if colors_dict is None:
        colors_dict = {}

    auto_colors = cm.viridis(np.linspace(0, 1, len(other)))
    final_colors = {}
    ci = 0
    for name in other:
        if name in colors_dict:
            final_colors[name] = colors_dict[name]
        else:
            final_colors[name] = auto_colors[ci]
            ci += 1
    final_colors['Median'] = colors_dict.get("My Reconstruction", "red")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(nrows=2, ncols=1,
                             figsize=(20, 15), sharex=False)
    fig.suptitle(title, fontsize=22, y=0.95)

    # Monthly panel
    ax1 = axes[0]
    ax1.fill_between(
        df.index, df['Lower'], df['Upper'],
        color=final_colors['Median'], alpha=0.2,
        label='95% Uncertainty Range',
    )
    m = df['Median'].dropna()
    ax1.plot(m.index, m.values, color=final_colors['Median'],
             label='Reconstruction (Median)', linewidth=1.5)
    for name in other:
        s = df[name].dropna()
        ax1.plot(s.index, s.values, label=name,
                 color=final_colors[name], linewidth=1.2)
    ax1.set_title('Monthly Mean', fontsize=18)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', linestyle='--')
    ax1.ticklabel_format(style='plain', axis='y', useOffset=False)

    # Annual panel
    ax2 = axes[1]
    ax2.fill_between(
        annual_df.index.year,
        annual_df['Lower'],
        annual_df['Upper'],
        color=final_colors['Median'], alpha=0.2,
        label='95% Uncertainty Range',
    )
    m_ann = annual_df['Median'].dropna()
    ax2.plot(m_ann.index.year, m_ann.values,
             color=final_colors['Median'],
             label='Reconstruction (Median)',
             linewidth=2.0, marker='o', markersize=4)
    for name in other:
        s = annual_df[name].dropna()
        ax2.plot(s.index.year, s.values, label=name,
                 color=final_colors[name],
                 linewidth=1.5, marker='.', markersize=4)
    ax2.set_title('Annual Mean', fontsize=18)
    ax2.set_xlabel('Year', fontsize=14)
    ax2.set_ylabel(ylabel, fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True, which='both', linestyle='--')
    ax2.ticklabel_format(style='plain', axis='y', useOffset=False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=300)
    print(f"Uncertainty plot saved to: {save_path}")
    plt.show()
    plt.close(fig)


# ======================================================================
# 11.  Visualisation — beta (regression coefficient) maps
# ======================================================================

def visualize_betas(
    model_library: Dict[int, Any],
    hash_to_mask_map: Dict[int, Dict[str, Any]],
    data: Dict[str, Any],
    output_dir: str,
    max_plots: int = 6,
) -> None:
    """
    Map regression coefficients (betas) back to a 2-D grid for the
    first *max_plots* mask groups and save as PNGs.
    """
    if not model_library:
        print("No models in library; skipping beta maps.")
        return

    ref = data.get("observation_source_model_da")
    if ref is None:
        print("No reference field in data; skipping beta maps.")
        return

    rename = {}
    if "longitude" in ref.dims:
        rename["longitude"] = "lon"
    if "latitude" in ref.dims:
        rename["latitude"] = "lat"
    if rename:
        ref = ref.rename(rename)

    lat = ref["lat"].values
    lon = ref["lon"].values
    ny, nx = len(lat), len(lon)

    out_dir = os.path.join(output_dir, "betas_maps")
    os.makedirs(out_dir, exist_ok=True)

    for i, (mask_hash, model_info) in enumerate(model_library.items()):
        if i >= max_plots:
            break
        coef = model_info.get("coef")
        if coef is None:
            continue

        mask_info = hash_to_mask_map.get(mask_hash)
        if mask_info is None:
            continue
        mask_2d = mask_info["mask_2d"]
        try:
            if mask_2d.shape != (ny, nx):
                mask_2d = mask_2d.reindex_like(
                    ref.isel(time=0), method="nearest"
                ).values
            else:
                mask_2d = mask_2d.values
        except Exception:
            print(f"Mask shape mismatch for hash {mask_hash}; skip.")
            continue

        beta_map = np.full((ny, nx), np.nan)
        beta_map[mask_2d] = coef

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.pcolormesh(lon, lat, beta_map, shading="auto")
        ax.set_title(f"Regression coefficients (mask {mask_hash})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(ax.collections[0], ax=ax, label="Beta")
        plt.tight_layout()

        fig.savefig(
            os.path.join(out_dir,
                         f"betas_mask_{i}_hash_{mask_hash}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)

    print(f"Beta maps saved to: {out_dir}")


# ======================================================================
# 12.  Helper — annual aggregation
# ======================================================================

def _monthly_to_annual(
    time: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate monthly time-series to annual means."""
    years = np.array([
        t.astype("datetime64[Y]").astype(int) + 1970 for t in time
    ])
    unique_years = np.unique(years)
    annual = [
        np.nanmean(values[years == y]) for y in unique_years
    ]
    return unique_years, np.array(annual)
