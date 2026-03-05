# src/run_experiment.py
# 这个版本只用于进行模拟实验

import os
import yaml
import numpy as np
from tqdm import tqdm
import xarray as xr  # 确保导入 xarray
import pandas as pd
import pickle
# 从你自己的模块中导入所有需要的函数
from dataset import load_and_prepare_data, _load_and_process_file,calculate_weighted_mean
from models import RidgeReconstruction
from utils import (build_training_matrices, standardize_mask,
                   visualize_reconstruction_timeseries,
                   _calculate_global_mean,evaluate_reconstruction,visualize_with_uncertainty,
                   apply_filter,save_results_to_parquet,calculate_uncertainty_weights)


def reconstruct_timeseries(
        model_library: dict,
        hash_to_mask_map: dict,
        source_values_anom: np.ndarray,
        mask_values: np.ndarray,
        source_name: str = "Unknown Source"
) -> np.ndarray:
    """
    使用训练好的模型库，对给定的源数据进行重建（高效批量处理版）。
    """
    n_time = source_values_anom.shape[0]
    recon_anom = np.full(n_time, np.nan)

    print(f"\n--- Starting reconstruction for '{source_name}' ---")

    time_hashes = []
    for t in range(n_time):
        mask_2d = standardize_mask(mask_values[t])
        if np.any(mask_2d):  # 只有在掩码不为空时才计算哈希
            time_hashes.append(hash(mask_2d.tobytes()))
        else:
            time_hashes.append(None)  # 用 None 标记空掩码

    processed_count = 0
    for mask_hash, pipeline in tqdm(model_library.items(), desc=f"Reconstructing {source_name}"):
        time_indices = [i for i, h in enumerate(time_hashes) if h == mask_hash]

        if not time_indices:
            continue

        batch_fields = source_values_anom[time_indices]
        mask_2d = hash_to_mask_map[mask_hash]

        X_batch = np.array([np.nan_to_num(field[mask_2d]) for field in batch_fields])

        if X_batch.shape[0] > 0:
            predictions = pipeline.predict(X_batch)
            recon_anom[time_indices] = predictions
            processed_count += len(time_indices)

    print(f"--- Reconstruction for '{source_name}' Finished ---")
    print(f"  -> Total timesteps: {n_time}")
    print(f"  -> Successfully reconstructed: {processed_count}")

    return recon_anom


def main():

    # config_path = '/g12338011ghq/project/hbb/OxygenDiffusion/Oxy_regression/config/config.yaml'
    config_path = "/g12338011ghq/project/hbb/OxygenDiffusion/Oxy_regression/config/congfig_exp_vertical_mean.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    exp_config = config['experiment']
    output_dir = exp_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 加载和预处理数据 ---
    data = load_and_prepare_data(config)

    # --- 2. 构建训练矩阵 ---
    print("\n--- Training Clean Model (for Median) ---")
    print("这里没有训练模型，下一个训练")
    print(f"   [DEBUG] inject_uncertainty set to: {data.get('inject_uncertainty')}")
    # X_clean, Y_clean, g_clean, hash_map_clean = build_training_matrices(data,min_features=1)
    X_grouped, Y_grouped, groups_grouped, hash_to_mask_map = build_training_matrices(data,min_features=1)
    # --- 3. 训练模型 ---
    if 'lambda_config' in exp_config:
        cfg = exp_config['lambda_config']
        lambda_candidates = np.logspace(cfg['log_start'], cfg['log_stop'], num=cfg['num_points'])
    else: lambda_candidates = exp_config.get('lambda_candidates', [0.01, 0.1, 1, 10, 100])


    # 3.2 训练鲁棒模型 (Robust Model)
    print("\n--- Training Robust Model (for Uncertainty) ---")
    print(f"   [DEBUG] inject_uncertainty set to: {data.get('inject_uncertainty')}")
    reconstructor = RidgeReconstruction(lambda_candidates=lambda_candidates)
    model_library = reconstructor.train(X_grouped, Y_grouped, groups_grouped,hash_to_mask_map,output_dir)
    print("\n--- Saving trained models and hash map ---")
    model_lib_path = os.path.join(output_dir, "WOD_ridge_model_library.pkl")
    hash_map_path = os.path.join(output_dir, "WOD_hash_to_mask_map.pkl")
    with open(model_lib_path, "wb") as f: pickle.dump(model_library, f)
    with open(hash_map_path, "wb") as f: pickle.dump(hash_to_mask_map, f)

    print(f"   -> Models saved to {model_lib_path}")
    print(f"   -> Hash map saved to {hash_map_path}")
    # 3.2 生成最佳估计曲线
    median_recon_anom = reconstruct_timeseries(
        model_library, hash_to_mask_map,
        data["observation_source_true_da_anom_values"],
        data["obs_mask_da_values"],
        source_name="True Source"
    )
    # --- Step 4: 执行重建 ---
    # --- 这里是注入不确定性的集合重建 (Ensemble Reconstruction)： ---
    ensemble_recon_anom_list = []    # 准备一个列表来存储每一次重建的结果
    bias_ensemble = data.get("bias_ensemble_values", None)
    if bias_ensemble is None:
        print("[WARN] No bias ensemble available for reconstruction. Performing a single deterministic run.")
        num_ensemble_runs = 1
    else:
        num_ensemble_runs = bias_ensemble.shape[0]  # 这里我用的是集合数量，也可以自己设置一个循环的值，Sppiel论文里面的不确定性集合是200

    print(f"\n--- [Part X] Starting Ensemble Reconstruction with {num_ensemble_runs} members ---")
    # 1. 获取观测掩码
    obs_mask_aligned_values = data["obs_mask_da_values"]
    weights_series = calculate_uncertainty_weights(
        obs_mask_aligned_values,
        baseline_coverage=0.10,
        smoothing_window=60
    )
    weight_factor = weights_series[np.newaxis, :, np.newaxis, np.newaxis]
    print(f"   -> Weights applied. Max: {weights_series.max():.2f}, Min: {weights_series.min():.2f}")
    print("\n--- [Step 9] Loading and processing reference products for comparison ---")

    reference_timeseries_anom = {}
    reference_timeseries_abs = {}
    #重建模型源
    recon_model_anom = reconstruct_timeseries(
        model_library, hash_to_mask_map,
        data["observation_source_model_da_anom_values"],
        data["obs_mask_da_values"],
        source_name="Model Source"
    )

    print("\n" + "=" * 80)
    print("--- QUANTITATIVE EVALUATION OF RECONSTRUCTION RESULTS ---")
    print("=" * 80)
    eval_mask = pd.to_datetime(data["time_coord"]) >= pd.to_datetime('1900-01-01')

    # 评估字典
    evaluation_results = {}
    # 评估模型源重建
    evaluation_results["Recon_vs_ModelSource"] = evaluate_reconstruction(
        y_true=data["Y_obs_source_truth_anom_values"][eval_mask],
        y_pred=recon_model_anom[eval_mask],
        series_name="Recon (from Model Source) vs. CMIP6 Truth"
    )
    print("\nExperiment finished successfully.")


if __name__ == "__main__":
    main()