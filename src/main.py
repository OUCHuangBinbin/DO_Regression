"""
main.py — Main training and reconstruction pipeline.

Pipeline steps:
  1. Load and prepare data (dataset.load_and_prepare_data)
  2. Build training matrices grouped by observation mask
  3. Train a clean Ridge model (for the median / best estimate)
  4. Train a robust Ridge model (for uncertainty quantification)
  5. Reconstruct from true observations (with uncertainty ensemble)
  6. Reconstruct from model-based observations (OSSE validation)
  7. Load and process reference products for comparison
  8. Evaluate, visualise, and save results

Usage:
    python main.py

Configuration is read from a YAML file (edit the path in main()).
"""

import os
import yaml
import pickle
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from dataset import load_and_prepare_data, _load_and_process_file, \
    calculate_weighted_mean
from models import RidgeReconstruction
from utils import (
    build_training_matrices,
    standardize_mask,
    evaluate_reconstruction,
    visualize_reconstruction_timeseries,
    visualize_with_uncertainty,
    apply_filter,
    save_results_to_parquet,
    calculate_uncertainty_weights,
)


# ======================================================================
# Reconstruction helper
# ======================================================================

def reconstruct_timeseries(
    model_library: dict,
    hash_to_mask_map: dict,
    source_values_anom: np.ndarray,
    mask_values: np.ndarray,
    source_name: str = "Unknown Source",
) -> np.ndarray:
    """
    Reconstruct global-mean anomaly time-series from gridded fields
    using a pre-trained model library.
    """
    n_time = source_values_anom.shape[0]
    recon = np.full(n_time, np.nan)

    print(f"\n--- Reconstructing '{source_name}' ---")

    time_hashes = []
    for t in range(n_time):
        m2d = standardize_mask(mask_values[t])
        if np.any(m2d):
            time_hashes.append(hash(m2d.tobytes()))
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

    print(f"     -> {processed} / {n_time} steps done.")
    return recon


# ======================================================================
# CSV export helpers
# ======================================================================

def save_ensemble_annual_results(
    ensemble_recon_anom: np.ndarray,
    time_coord: np.ndarray,
    output_dir: str,
    filename: str = "Ensemble_Annual_Details.csv",
) -> pd.DataFrame:
    """
    Save all ensemble member reconstructions as a wide-format annual CSV
    (rows = years, columns = members).
    """
    n_members = ensemble_recon_anom.shape[0]
    dates = pd.to_datetime(time_coord)

    df_monthly = pd.DataFrame(
        ensemble_recon_anom.T,
        index=dates,
        columns=[f"Member_{i+1}" for i in range(n_members)],
    )
    df_annual = df_monthly.resample("YE").mean()
    df_annual.index = df_annual.index.year
    df_annual.index.name = "Year"

    path = os.path.join(output_dir, filename)
    df_annual.to_csv(path, float_format="%.8f")
    print(f"     -> Ensemble members CSV: {path}")
    return df_annual


def save_all_results_to_csv(
    output_dir: str,
    time_coord: np.ndarray,
    median_recon_anom: np.ndarray,
    median_recon_abs: np.ndarray,
    final_lower_anom: np.ndarray,
    final_upper_anom: np.ndarray,
    lower_bound_abs: np.ndarray,
    upper_bound_abs: np.ndarray,
    lower_anom_smooth: np.ndarray,
    upper_anom_smooth: np.ndarray,
    lower_bound_smooth: np.ndarray,
    upper_bound_smooth: np.ndarray,
    Y_obs_truth_anom: np.ndarray,
    cmip6_truth_abs: np.ndarray,
    recon_model_anom: np.ndarray,
    recon_model_abs: np.ndarray,
    reference_timeseries_anom: dict,
    reference_timeseries_abs: dict,
    run_tag: str = "result",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save all time-series to monthly and annual CSV files.

    Column naming convention:
        anom_*  — anomaly fields
        abs_*   — absolute-value fields
        *_lo / *_hi  — 95% confidence interval bounds
        *_smooth  — 12-month rolling mean of bounds
    """
    time_index = pd.to_datetime(time_coord)
    cols = {}

    # Reconstruction (anomaly)
    cols["anom_recon_median"] = median_recon_anom
    cols["anom_recon_lo_raw"] = final_lower_anom
    cols["anom_recon_hi_raw"] = final_upper_anom
    cols["anom_recon_lo_smooth"] = lower_anom_smooth
    cols["anom_recon_hi_smooth"] = upper_anom_smooth

    # Reconstruction (absolute)
    cols["abs_recon_median"] = median_recon_abs
    cols["abs_recon_lo_raw"] = lower_bound_abs
    cols["abs_recon_hi_raw"] = upper_bound_abs
    cols["abs_recon_lo_smooth"] = lower_bound_smooth
    cols["abs_recon_hi_smooth"] = upper_bound_smooth

    # CMIP6 truth
    cols["anom_cmip6_truth"] = Y_obs_truth_anom
    cols["abs_cmip6_truth"] = cmip6_truth_abs

    # Model-source reconstruction
    cols["anom_recon_model_src"] = recon_model_anom
    cols["abs_recon_model_src"] = recon_model_abs

    # Reference products
    for name, series in reference_timeseries_anom.items():
        safe = name.replace(" ", "_").replace("-", "_")
        safe = safe.replace("(", "").replace(")", "")
        cols[f"anom_ref_{safe}"] = series
    for name, series in reference_timeseries_abs.items():
        safe = name.replace(" ", "_").replace("-", "_")
        safe = safe.replace("(", "").replace(")", "")
        cols[f"abs_ref_{safe}"] = series

    # Length alignment
    n = len(time_index)
    for k, arr in cols.items():
        if arr is None:
            cols[k] = np.full(n, np.nan)
        elif len(arr) != n:
            print(f"[WARN] Column '{k}' length {len(arr)} != {n}. "
                  f"Truncating/padding.")
            padded = np.full(n, np.nan)
            m = min(len(arr), n)
            padded[:m] = arr[:m]
            cols[k] = padded

    # Monthly CSV
    df_monthly = pd.DataFrame(cols, index=time_index)
    df_monthly.index.name = "time"
    monthly_path = os.path.join(output_dir, f"{run_tag}_monthly.csv")
    df_monthly.to_csv(monthly_path, float_format="%.8f")
    print(f"     -> Monthly CSV: {monthly_path}  "
          f"({df_monthly.shape})")

    # Annual CSV
    df_annual = df_monthly.resample("YE").mean()
    df_annual.index = df_annual.index.year
    df_annual.index.name = "year"
    annual_path = os.path.join(output_dir, f"{run_tag}_annual.csv")
    df_annual.to_csv(annual_path, float_format="%.8f")
    print(f"     -> Annual CSV: {annual_path}  "
          f"({df_annual.shape})")

    return df_monthly, df_annual


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    # -------------------------------------------------------------------
    # 0.  Configuration
    # -------------------------------------------------------------------
    config_path = "/g12338011ghq/project/hbb/OxygenDiffusion/" \
                  "Oxy_regression_Surface/config/Final_config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp_config = config["experiment"]
    output_dir = exp_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------------
    # 1.  Load & prepare data
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 1: Loading data")
    print("=" * 70)
    data = load_and_prepare_data(config)

    # -------------------------------------------------------------------
    # 2.  Build training matrices (clean model)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 2: Building clean training matrices")
    print("=" * 70)
    data["inject_uncertainty"] = False
    X_clean, Y_clean, groups_clean, hash_map_clean = \
        build_training_matrices(data, min_features=1)

    data["inject_uncertainty"] = True
    X_robust, Y_robust, groups_robust, hash_map_robust = \
        build_training_matrices(data, min_features=1)

    # -------------------------------------------------------------------
    # 3.  Configure lambda candidates
    # -------------------------------------------------------------------
    if "lambda_config" in exp_config:
        cfg = exp_config["lambda_config"]
        lambda_candidates = np.logspace(
            cfg["log_start"], cfg["log_stop"], num=cfg["num_points"]
        )
    else:
        lambda_candidates = exp_config.get(
            "lambda_candidates", [0.01, 0.1, 1, 10, 100]
        )

    # -------------------------------------------------------------------
    # 4.  Train clean model (for median / best estimate)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3a: Training clean Ridge model")
    print("=" * 70)
    reconstructor_clean = RidgeReconstruction(
        lambda_candidates=lambda_candidates
    )
    model_lib_clean = reconstructor_clean.train(
        X_clean, Y_clean, groups_clean, hash_map_clean, output_dir
    )

    # -------------------------------------------------------------------
    # 5.  Train robust model (for uncertainty)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 3b: Training robust Ridge model")
    print("=" * 70)
    reconstructor_robust = RidgeReconstruction(
        lambda_candidates=lambda_candidates
    )
    model_lib_robust = reconstructor_robust.train(
        X_robust, Y_robust, groups_robust, hash_map_robust, output_dir
    )

    # -------------------------------------------------------------------
    # 6.  Median reconstruction from true observations (clean model)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 4: Median reconstruction (clean model)")
    print("=" * 70)
    median_recon_anom = reconstruct_timeseries(
        model_lib_clean, hash_map_clean,
        data["observation_source_true_da_anom_values"],
        data["obs_mask_da_values"],
        source_name="True Source (median)"
    )

    # -------------------------------------------------------------------
    # 7.  Ensemble reconstruction (robust model, uncertainty injection)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 5: Ensemble reconstruction (robust model)")
    print("=" * 70)

    inject = data.get("inject_uncertainty", False)
    bias_ensemble = data.get("bias_ensemble_values", None)

    if not inject or bias_ensemble is None:
        print("[WARN] No bias ensemble. Single deterministic run.")
        num_runs = 1
    else:
        num_runs = bias_ensemble.shape[0]

    print(f"     -> {num_runs} ensemble member(s)")

    # Compute coverage-based uncertainty weights
    obs_mask_aligned = data["obs_mask_da_values"]
    weights = calculate_uncertainty_weights(
        obs_mask_aligned, baseline_coverage=0.10, smoothing_window=120,
    )
    weight_factor = weights[np.newaxis, :, np.newaxis, np.newaxis]
    print(f"     -> Weights: max={weights.max():.2f}, "
          f"min={weights.min():.2f}")

    ensemble_list = []
    for i in tqdm(range(num_runs), desc="Ensemble"):
        true_obs_anom = data["observation_source_true_da_anom_values"]

        if inject and bias_ensemble is not None:
            bias = bias_ensemble[i, :, :, :]
            weighted_bias = bias * weight_factor[0, :, :, :]
            obs_mask_valid = ~np.isnan(true_obs_anom)
            perturbed = np.where(
                obs_mask_valid,
                np.nan_to_num(true_obs_anom) + weighted_bias,
                true_obs_anom,
            )
        else:
            perturbed = true_obs_anom

        member = reconstruct_timeseries(
            model_lib_robust, hash_map_robust,
            perturbed, obs_mask_aligned,
            source_name=f"Ensemble Member {i+1}",
        )
        ensemble_list.append(member)

    ensemble_recon = np.array(ensemble_list)

    # Outlier cleaning: clip 10 extremes per time step
    ensemble_recon = np.sort(ensemble_recon, axis=0)[10:-10, :]

    # Compute symmetric 95% CI from cleaned ensemble std
    q25 = np.nanpercentile(ensemble_recon, 25, axis=0)
    q75 = np.nanpercentile(ensemble_recon, 75, axis=0)
    iqr = q75 - q25
    ens_median = np.nanmedian(ensemble_recon, axis=0)
    cleaned = np.clip(
        ensemble_recon,
        ens_median - 1.5 * iqr,
        ens_median + 1.5 * iqr,
    )
    ens_std = np.nanstd(cleaned, axis=0)
    half_width = 1.96 * ens_std
    final_lower_anom = median_recon_anom - half_width
    final_upper_anom = median_recon_anom + half_width

    # Smooth the uncertainty bounds
    SMOOTH = 24
    half_width_smooth = pd.Series(half_width).rolling(
        window=SMOOTH, center=True, min_periods=1
    ).mean().values

    lower_anom_smooth = median_recon_anom - half_width_smooth
    upper_anom_smooth = median_recon_anom + half_width_smooth

    # Convert to absolute values
    clim_mean = data["true_obs_climatology_regional_mean"]
    median_recon_abs = median_recon_anom + clim_mean
    lower_bound_abs = final_lower_anom + clim_mean
    upper_bound_abs = final_upper_anom + clim_mean
    lower_bound_smooth = median_recon_abs - half_width_smooth
    upper_bound_smooth = median_recon_abs + half_width_smooth

    # -------------------------------------------------------------------
    # 8.  Reconstruction from model source (OSSE validation)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 6: OSSE reconstruction (model source)")
    print("=" * 70)
    recon_model_anom = reconstruct_timeseries(
        model_lib_clean, hash_map_clean,
        data["observation_source_model_da_anom_values"],
        data["obs_mask_da_values"],
        source_name="Model Source"
    )

    # -------------------------------------------------------------------
    # 9.  Load reference products
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 7: Loading reference products")
    print("=" * 70)
    ref_anom = {}
    ref_abs = {}

    if "reference_products" in config:
        time_slice = slice(
            f"{config['data']['time_range'][0]}-01-01",
            f"{config['data']['time_range'][1]}-12-31"
        )
        for name, info in config["reference_products"].items():
            print(f"\n     -> {name}")
            raw = _load_and_process_file(
                info["path"], info["var_name"],
                config["data"]["depth_level"], time_slice,
            )
            if raw is None:
                continue

            factor = info.get("unit_conversion_factor", 1.0)
            converted = raw * factor

            # Anomaly
            clim_self = converted.groupby("time.month").mean("time")
            anom_da = converted.groupby("time.month") - clim_self

            # Time-alignment via reindex
            overlap = xr.DataArray(
                True, coords=converted.coords, dims=converted.dims
            )
            overlap_aligned = (
                overlap.reindex(
                    time=data["time_coord"], method="nearest",
                    tolerance="20D"
                )
                > 0
            )
            anom_aligned = anom_da.reindex(
                time=data["time_coord"], method="nearest",
                tolerance="20D"
            )
            abs_aligned = converted.reindex(
                time=data["time_coord"], method="nearest",
                tolerance="20D"
            )
            anom_aligned = anom_aligned.where(overlap_aligned)
            abs_aligned = abs_aligned.where(overlap_aligned)

            if np.all(np.isnan(anom_aligned.values)):
                print(f"     ... WARNING: All NaN after reindex. Skip.")
                continue

            anom_ts = calculate_weighted_mean(anom_aligned).values
            abs_ts = calculate_weighted_mean(abs_aligned).values

            if anom_ts.ndim == 1 and abs_ts.ndim == 1:
                ref_anom[name] = anom_ts
                ref_abs[name] = abs_ts

    # -------------------------------------------------------------------
    # 10.  Evaluation
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 8: Quantitative evaluation")
    print("=" * 70)
    eval_mask = pd.to_datetime(data["time_coord"]) >= "1900-01-01"
    results_eval = {}

    results_eval["Recon_vs_ModelSource"] = evaluate_reconstruction(
        y_true=data["Y_obs_source_truth_anom_values"][eval_mask],
        y_pred=recon_model_anom[eval_mask],
        series_name="Recon (model source) vs CMIP6 truth",
    )

    # -------------------------------------------------------------------
    # 11.  Visualisation
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 9: Visualisation")
    print("=" * 70)

    my_colors = {
        "CMIP6 Source (Truth)": "black",
        "Recon (from True Obs)": "red",
        "Recon (from Model Source)": "darkorange",
        "GDOIP Corrected": "navy",
        "GDOIP Uncorrected": "lightskyblue",
        "GT_ML_RF_ShipArgo_corrected": "forestgreen",
        "GT_ML_RF_ShipArgo_uncorrected": "lightgreen",
        "O2-Map SJTU_v1": "seagreen",
        "O2-Map SJTU_ship_only_v1.2": "darkgreen",
        "O2-Map SJTU_v1.2": "mediumvioletred",
    }

    # Absolute value: CMIP6 truth + model-source reconstruction
    cmip6_truth_abs = (
        data["Y_obs_source_truth_anom_values"]
        + data["obs_model_climatology_regional_mean"]
    )
    recon_model_abs = (
        recon_model_anom
        + data["obs_model_climatology_regional_mean"]
    )

    # (A) Anomaly comparison with uncertainty
    other_anom = {
        "CMIP6 Source (Truth)": data["Y_obs_source_truth_anom_values"],
        "Recon (from Model Source)": recon_model_anom,
    }
    other_anom.update(ref_anom)

    visualize_with_uncertainty(
        time_coord=data["time_coord"],
        save_path=os.path.join(
            output_dir, "anomaly_comparison_with_uncertainty.png"
        ),
        title="Comprehensive Comparison of Anomaly Trends",
        ylabel="Mean DO Anomaly (mol m-3)",
        median_line=median_recon_anom,
        lower_bound=lower_anom_smooth,
        upper_bound=upper_anom_smooth,
        other_lines_dict=other_anom,
        colors_dict=my_colors,
    )

    # (B) Absolute comparison with uncertainty
    other_abs = {
        "CMIP6 Source (Truth)": cmip6_truth_abs,
        "Recon (from Model Source)": recon_model_abs,
    }
    other_abs.update(ref_abs)

    visualize_with_uncertainty(
        time_coord=data["time_coord"],
        save_path=os.path.join(
            output_dir, "final_absolute_comparison_with_uncertainty.png"
        ),
        title="Final Reconstruction vs. References (Absolute)",
        ylabel="Absolute Mean DO (mol m-3)",
        median_line=median_recon_abs,
        lower_bound=lower_bound_smooth,
        upper_bound=upper_bound_smooth,
        other_lines_dict=other_abs,
        colors_dict=my_colors,
    )

    # -------------------------------------------------------------------
    # 12.  Time-scale decomposition (low-pass / high-pass filtering)
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 10: Time-scale decomposition")
    print("=" * 70)

    anom_dict = {
        "CMIP6 Source (Truth)":
            data["Y_obs_source_truth_anom_values"],
        "Recon (from Model Source)": recon_model_anom,
        "Recon (from True Obs)": median_recon_anom,
    }
    df_anom = pd.DataFrame(
        anom_dict, index=pd.to_datetime(data["time_coord"])
    )
    annual_df = df_anom.resample("YE").mean()

    cutoff = 20.0  # years
    lowpass = {}
    highpass = {}
    for name in annual_df.columns:
        s = annual_df[name].values
        lowpass[name] = apply_filter(
            s, cutoff_period=cutoff, filter_type="low", sample_rate=1.0
        )
        highpass[name] = apply_filter(
            s, cutoff_period=cutoff, filter_type="high", sample_rate=1.0
        )

    visualize_reconstruction_timeseries(
        time_coord=annual_df.index.year,
        save_path=os.path.join(
            output_dir,
            f"low_pass_filtered_comparison.png"
        ),
        title=f"Low-Pass Filtered (> {cutoff} years) Anomaly Trends",
        ylabel="Mean DO Anomaly (mol m-3)",
        timeseries_dict=lowpass,
        colors_dict=my_colors,
        is_annual_data=True,
    )

    visualize_reconstruction_timeseries(
        time_coord=annual_df.index.year,
        save_path=os.path.join(
            output_dir,
            f"high_pass_filtered_comparison.png"
        ),
        title=f"High-Pass Filtered (< {cutoff} years) Anomaly Trends",
        ylabel="Mean DO Anomaly (mol m-3)",
        timeseries_dict=highpass,
        colors_dict=my_colors,
        is_annual_data=True,
    )

    # -------------------------------------------------------------------
    # 13.  Save results
    # -------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Step 11: Saving results")
    print("=" * 70)

    # Collect all time-series
    all_results = {}
    for name, series in other_anom.items():
        all_results[f"anom_{name}"] = series
    for name, series in other_abs.items():
        all_results[f"abs_{name}"] = series
    all_results["anom_My Reconstruction (Median)"] = median_recon_anom
    all_results["abs_My Reconstruction (Median)"] = median_recon_abs
    all_results["anom_95% Uncertainty Range_lower"] = final_lower_anom
    all_results["anom_95% Uncertainty Range_upper"] = final_upper_anom
    all_results["abs_95% Uncertainty Range_lower"] = lower_bound_abs
    all_results["abs_95% Uncertainty Range_upper"] = upper_bound_abs

    results_path = os.path.join(
        output_dir, "reconstruction_results.parquet"
    )
    save_results_to_parquet(
        filepath=results_path,
        time_coord=data["time_coord"],
        results_dict=all_results,
        config=config,
    )

    # CSV export
    run_tag = os.path.splitext(os.path.basename(results_path))[0]
    save_ensemble_annual_results(
        ensemble_recon_anom=ensemble_recon,
        time_coord=data["time_coord"],
        output_dir=output_dir,
        filename="Ensemble_All_Members_Annual_Anom.csv",
    )
    save_all_results_to_csv(
        output_dir=output_dir,
        time_coord=data["time_coord"],
        median_recon_anom=median_recon_anom,
        median_recon_abs=median_recon_abs,
        final_lower_anom=final_lower_anom,
        final_upper_anom=final_upper_anom,
        lower_bound_abs=lower_bound_abs,
        upper_bound_abs=upper_bound_abs,
        lower_anom_smooth=lower_anom_smooth,
        upper_anom_smooth=upper_anom_smooth,
        lower_bound_smooth=lower_bound_smooth,
        upper_bound_smooth=upper_bound_smooth,
        Y_obs_truth_anom=data["Y_obs_source_truth_anom_values"],
        cmip6_truth_abs=cmip6_truth_abs,
        recon_model_anom=recon_model_anom,
        recon_model_abs=recon_model_abs,
        reference_timeseries_anom=ref_anom,
        reference_timeseries_abs=ref_abs,
        run_tag=run_tag,
    )

    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    main()
