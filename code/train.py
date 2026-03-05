# src/run_experiment.py

import os
import yaml
import numpy as np
from tqdm import tqdm
import xarray as xr
import pandas as pd
import pickle

from dataset import load_and_prepare_data, _load_and_process_file,calculate_weighted_mean
from models import RidgeReconstruction
from utils import (build_training_matrices, standardize_mask,
                   visualize_reconstruction_timeseries,
                   evaluate_reconstruction,visualize_with_uncertainty,
                   apply_filter,save_results_to_parquet,calculate_uncertainty_weights)


def reconstruct_timeseries(
        model_library: dict,
        hash_to_mask_map: dict,
        source_values_anom: np.ndarray,
        mask_values: np.ndarray,
        source_name: str = "Unknown Source"
) -> np.ndarray:
    """
    使用训练好的模型库，对给定的源数据进行重建。
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
        time_indices = [i for i, h in enumerate(time_hashes) if h == mask_hash] # 现在 h 不会是 None

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
    # ===================================================================
    #                            路径配置
    # ===================================================================
    # config_path = "/g12338011ghq/project/hbb/OxygenDiffusion/Oxy_regression/config/exp3_MultiModel_config.yaml"
    # config_path = '/g12338011ghq/project/hbb/OxygenDiffusion/Oxy_regression/config/config_test_CanESM5.yaml'
    # config_path = '/g12338011ghq/project/hbb/OxygenDiffusion/Oxy_regression/config/config_test_MPI.yaml'
    config_path = '/g12338011ghq/project/hbb/OxygenDiffusion/Oxy_regression/config/config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    exp_config = config['experiment']
    output_dir = exp_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 加载和预处理数据 ---
    data = load_and_prepare_data(config)

    # --- 2. 构建训练矩阵 ---
    X_grouped, Y_grouped, groups_grouped, hash_to_mask_map = build_training_matrices(data,min_features=10)

    # --- 3. 训练模型 ---
    # 配置参数λ
    if 'lambda_config' in exp_config:
        cfg = exp_config['lambda_config']
        lambda_candidates = np.logspace(cfg['log_start'], cfg['log_stop'], num=cfg['num_points'])
    else: lambda_candidates = exp_config.get('lambda_candidates', [0.01, 0.1, 1, 10, 100])

    reconstructor = RidgeReconstruction(lambda_candidates=lambda_candidates)
    model_library = reconstructor.train(X_grouped, Y_grouped, groups_grouped, hash_to_mask_map, output_dir)

    # model_lib_path = os.path.join(output_dir, "IAP_unbias_ridge_model_library.pkl")
    # hash_map_path = os.path.join(output_dir, "IAP_unbias_hash_to_mask_map.pkl")
    model_lib_path = os.path.join(output_dir, "WOD_unbias_ridge_model_library.pkl")
    hash_map_path = os.path.join(output_dir, "WOD_unbias_hash_to_mask_map.pkl")
    with open(model_lib_path, "wb") as f: pickle.dump(model_library, f)
    with open(hash_map_path, "wb") as f: pickle.dump(hash_to_mask_map, f)

    print(f"   -> Models saved to {model_lib_path}")
    print(f"   -> Hash map saved to {hash_map_path}")

    # ===================================================================
    #                         注入不确定集合（未完成）
    # ===================================================================
    ensemble_recon_anom_list = []    # 准备一个列表来存储每一次重建的结果
    inject_uncertainty = data.get("inject_uncertainty", False)
    bias_ensemble = data.get("bias_ensemble_values", None)
    if not inject_uncertainty or bias_ensemble is None:
        print("[WARN] No bias ensemble available for reconstruction. Performing a single deterministic run.")
        num_ensemble_runs = 1
    else:
        # 这里我用的是集合数量，也可以自己设置一个循环的值，Sppiel论文里面的不确定性集合是200
        num_ensemble_runs = bias_ensemble.shape[0]
    print(f"\n--- [Part X] Starting Ensemble Reconstruction with {num_ensemble_runs} members ---")

    obs_mask_aligned_values = data["obs_mask_da_values"]
    n_obs_series = obs_mask_aligned_values.sum(axis=(1, 2))

    weights_series = calculate_uncertainty_weights(
        obs_mask_aligned_values,
        baseline_coverage=0.10,
        smoothing_window=60
    )
    # 3. 调整维度以便广播: (time,) -> (time, 1, 1)
    # bias_ensemble 的形状是 (member, time, lat, lon)
    # 我们需要在 time 维度上乘以权重
    weight_factor = weights_series[np.newaxis, :, np.newaxis, np.newaxis]
    # 加权注入不确定性结合（还需要讨论，未完成）
    # 逻辑：观测越少 -> 1/sqrt(n) 越大 -> 权重越大 -> 注入的偏差越大
    print(f"   -> Weights applied. Max: {weights_series.max():.2f}, Min: {weights_series.min():.2f}")

    for i in tqdm(range(num_ensemble_runs), desc="Ensemble Reconstruction"):
        # 准备“被污染”的真实观测数据
        true_obs_anom_values = data["observation_source_true_da_anom_values"]  #真实观测的异常值
        if inject_uncertainty:
            bias_to_add = data["bias_ensemble_values"][i, :, :, :]
            weighted_bias = bias_to_add * weight_factor

            # --- 只在有观测的地方添加偏差 ---
            obs_exists_mask = ~np.isnan(true_obs_anom_values)

            # 只在这些地方添加偏差
            perturbed_obs_anom = np.copy(true_obs_anom_values)  # 创建一个副本
            # 使用 where 条件操作
            # np.where(condition, x, y) -> if condition is true, take from x, else from y
            # 在 obs_exists_mask 为 True 的地方，加上偏差
            bias_component = np.nan_to_num(true_obs_anom_values) + weighted_bias

            perturbed_obs_anom = np.where(
                obs_exists_mask,  # 条件
                bias_component,  # 如果为 True，执行加法
                true_obs_anom_values  # 如果为 False，保持原始值 (NaN)
            )

        else:
            perturbed_obs_anom = true_obs_anom_values

            # 使用加噪的数据进行重建
        recon_anom_member = reconstruct_timeseries(
            model_library, hash_to_mask_map,
            perturbed_obs_anom,
            data["obs_mask_da_values"],
            source_name=f"Ensemble Member {i + 1}"
        )

        ensemble_recon_anom_list.append(recon_anom_member)

        # 将结果列表转换为一个 (n_members, n_time) 的 Numpy 数组
    ensemble_recon_anom = np.array(ensemble_recon_anom_list)

    # --- 计算统计量 ---
    # 计算中位数作为最佳估计
    median_recon_anom = np.nanmedian(ensemble_recon_anom, axis=0)
    # 计算 2.5% 和 97.5% 分位数作为不确定性范围
    lower_bound_anom = np.nanpercentile(ensemble_recon_anom, 2.5, axis=0)
    upper_bound_anom = np.nanpercentile(ensemble_recon_anom, 97.5, axis=0)

    # 恢复绝对值
    climatology_mean = data["true_obs_climatology_regional_mean"]
    median_recon_abs = median_recon_anom + climatology_mean
    lower_bound_abs = lower_bound_anom + climatology_mean
    upper_bound_abs = upper_bound_anom + climatology_mean

    # --- 4. 执行所有重建任务 ---
    recon_model_anom = reconstruct_timeseries(
        model_library, hash_to_mask_map,
        data["observation_source_model_da_anom_values"],
        data["obs_mask_da_values"],
        source_name="Model Source"
    )
    recon_true_anom = reconstruct_timeseries(
        model_library, hash_to_mask_map,
        data["observation_source_true_da_anom_values"],
        data["obs_mask_da_values"],
        source_name="True Obs Source"
    )

    recon_true_abs = recon_true_anom + data["true_obs_climatology_regional_mean"]
    # ===================================================================
    #          加入其他产品（溶解氧国际小组的数据产品）进行同时期的比较
    # ===================================================================
    print("\n--- [Step 9] Loading and processing reference products for comparison ---")
    reference_timeseries_anom = {}
    reference_timeseries_abs = {}

    # 在这里绘制其他产品：
    # 把溶解氧国际小组的产品加入进去

    if 'reference_products' in config:
        # 获取用于缩放的目标统计量 (来自训练集)
        train_anom_std = data["train_anom_std"]
        time_slice = slice(f"{config['data']['time_range'][0]}-01-01", f"{config['data']['time_range'][1]}-12-31")

        for name, info in config['reference_products'].items():
            print(f"\n   -> Processing reference: {name}")

            ref_da_raw = _load_and_process_file(
                info['path'], info['var_name'], config['data']['depth_level'], time_slice
            )

            if ref_da_raw is not None:
                # # --- 诊断开始 ---
                # print("\n" + "*" * 50)
                # print("--- TIME COORDINATE DIAGNOSIS ---")
                # print(f"Name: {name}")
                # # 打印基准时间轴的前5个值
                # print("\nBase time coordinate (from data['time_coord']):")
                # print(data['time_coord'][:5])
                # print("dtype:", data['time_coord'].dtype)
                #
                # # 打印 GDOIP 时间轴的前5个值
                # print("\nGDOIP time coordinate (raw):")
                # print(ref_da_raw.time.values[:5])
                # print("dtype:", ref_da_raw.time.values.dtype)
                # print("Calendar (from xarray attrs):", ref_da_raw.time.attrs.get('calendar', 'standard'))
                # print("*" * 50 + "\n")
                # # --- 诊断结束 ---

                # 1. 单位转换
                conversion_factor = info.get('unit_conversion_factor', 1.0)
                ref_da_converted = ref_da_raw * conversion_factor

                # 2. 计算异常
                ref_climatology = ref_da_converted.groupby('time.month').mean('time')
                ref_anom_da = ref_da_converted.groupby('time.month') - ref_climatology

                # --- 使用 reindex(method='nearest') 进行时间对齐 ---
                # 它会自动将 GDOIP 的月初数据，匹配到我们基准的月中时间点上
                overlap_mask = xr.DataArray(True, coords=ref_da_converted.coords, dims=ref_da_converted.dims)
                overlap_mask_aligned = overlap_mask.reindex(time=data['time_coord'], method='nearest',
                                                            tolerance='20D') > 0

                ref_anom_aligned = ref_anom_da.reindex(time=data['time_coord'], method='nearest', tolerance='20D')
                ref_abs_aligned = ref_da_converted.reindex(time=data['time_coord'], method='nearest', tolerance='20D')

                # 3. 使用掩码，将 GDOIP 在没有数据的时段强制设为 NaN
                ref_anom_aligned = ref_anom_aligned.where(overlap_mask_aligned)
                ref_abs_aligned = ref_abs_aligned.where(overlap_mask_aligned)

                # 检查 reindex 后是否全是 NaN (作为一个安全检查)
                if np.all(np.isnan(ref_anom_aligned.values)):
                    print(f"     ... WARNING: Reindexing resulted in all NaNs for '{name}'. No time overlap. Skipping.")
                    continue

                # 4. 计算区域平均
                ref_anom_ts = calculate_weighted_mean(ref_anom_aligned).values
                ref_abs_ts = calculate_weighted_mean(ref_abs_aligned).values

                # 5. 检查并添加到字典
                if ref_anom_ts.ndim == 1 and ref_abs_ts.ndim == 1:
                    reference_timeseries_anom[name] = ref_anom_ts
                    reference_timeseries_abs[name] = ref_abs_ts
                else:
                    print(f"     ... WARNING: Processed reference '{name}' is not 1-dimensional. Skipping.")
    else:
        pass
    # ===================================================================
    #                     量化评估重建结果
    # ===================================================================
    print("\n" + "=" * 80)
    print("--- QUANTITATIVE EVALUATION OF RECONSTRUCTION RESULTS ---")
    print("=" * 80)
    eval_mask = pd.to_datetime(data["time_coord"]) >= pd.to_datetime('1900-01-01')

    # 评估字典
    evaluation_results = {}
    # 1. 评估模型源重建
    evaluation_results["Recon_vs_ModelSource"] = evaluate_reconstruction(
        y_true=data["Y_obs_source_truth_anom_values"][eval_mask],
        y_pred=recon_model_anom[eval_mask],
        series_name="Recon (from Model Source) vs. CMIP6 Truth"
    )

    # --- Step Y: 可视化所有结果 ---
    print("\n" + "=" * 80)
    print("--- VISUALIZING ALL RESULTS ---")
    print("=" * 80)
    # --- 6. 清晰的可视化流程 ---
    print("\n--- [Step 6] Visualizing all results ---")
    # ===================================================================
    #                        绘制结果图
    # ===================================================================
    # 定义颜色
    my_colors = {

        # ===== Reference / Truth =====
        "CMIP6 Source (Truth)": "black",

        # ===== Reconstruction (Red family) =====
        "Recon (from True Obs)": "red", # 主结果，最醒目
        "Recon (from Model Source)": "darkorange", # 模式驱动重建

        # ===== GDOIP (Blue family) =====
        "GDOIP Corrected": "navy", # 深蓝
        "GDOIP Uncorrected": "lightskyblue", # 浅蓝

        # ===== ML / O2-Map / SJTU 系列 =====
        "GT_ML_RF_ShipArgo_corrected": "forestgreen",
        "GT_ML_RF_ShipArgo_uncorrected": "lightgreen",
        "O2-Map SJTU_v1":"seagreen",
        "O2-Map SJTU_ship_only_v1.2":"darkgreen",
        "O2-Map SJTU_v1.2": "mediumvioletred",

        # "My Reconstruction": "red"  # 用于绝对值图
    }
    #动态生成标题
    corr_vs_truth = evaluation_results.get("MyRecon_vs_ModelSource", {}).get("correlation")
    title_anom = "Comprehensive Comparison of Anomaly Trends"
    if corr_vs_truth is not None:
        title_anom += f"\n(My Recon vs CMIP6 Truth: r = {corr_vs_truth:.2f})"

    # 1. 恢复CMIP6真值的绝对值
    cmip6_truth_abs = data["Y_obs_source_truth_anom_values"] + data["obs_model_climatology_regional_mean"]
    # 2. 恢复模型源重建的绝对值
    recon_model_abs = recon_model_anom + data["obs_model_climatology_regional_mean"]
    print("后续增加的内容，绘制不确定性的集合")
    # 准备其他要对比的曲线
    anomaly_plot_path = os.path.join(output_dir, "anomaly_comparison_with_uncertainty.png")
    other_lines_anom = {
        "CMIP6 Source (Truth)": data["Y_obs_source_truth_anom_values"],
        "Recon (from Model Source)": recon_model_anom,
    }
    other_lines_anom.update(reference_timeseries_anom)

    visualize_with_uncertainty(
        time_coord=data["time_coord"],
        save_path=anomaly_plot_path,
        title="Comprehensive Comparison of Anomaly Trends",
        ylabel="Mean DO Anomaly (mol m-3)",
        median_line=median_recon_anom,
        lower_bound=lower_bound_anom,
        upper_bound=upper_bound_anom,
        other_lines_dict=other_lines_anom,
        colors_dict=my_colors
    )

    # (B) 最终绝对值对比图 (带有不确定性)
    final_plot_path = os.path.join(output_dir, "final_absolute_comparison_with_uncertainty.png")

    other_lines_abs = {
        "CMIP6 Source (Truth)":cmip6_truth_abs,
        "Recon (from Model Source)": recon_model_abs,
    }
    other_lines_abs.update(reference_timeseries_abs)

    visualize_with_uncertainty(
        time_coord=data["time_coord"],
        save_path=final_plot_path,
        title="Final Reconstruction vs. References (Absolute)",
        ylabel="Absolute Mean DO (mol m-3)",
        median_line=median_recon_abs,
        lower_bound=lower_bound_abs,
        upper_bound=upper_bound_abs,
        other_lines_dict=other_lines_abs,
        colors_dict=my_colors
    )
    # ===================================================================
    #                           时间尺度分离分析与可视化
    # ===================================================================
    print("\n" + "=" * 80)
    print("--- VISUALIZING TIME-SCALE DECOMPOSITION ---")
    print("=" * 80)

    # --- 准备年度数据 ---
    # 我们将在年平均数据上进行滤波，因为20年的周期对于月度数据来说太长
    anomaly_plot_dict = {
        "CMIP6 Source (Truth)": data["Y_obs_source_truth_anom_values"],
        "Recon (from Model Source)": recon_model_anom,
        "Recon (from True Obs)": recon_true_anom,
    }
    # 1. 将所有异常序列转换为年度DataFrame
    df_anom = pd.DataFrame(anomaly_plot_dict, index=pd.to_datetime(data["time_coord"]))
    annual_df_anom = df_anom.resample('YE').mean()

    # 2. 对每一列（每个产品）都应用低通和高通滤波
    cutoff_period_years = 20.0

    low_pass_dict = {}
    high_pass_dict = {}

    for name in annual_df_anom.columns:
        series_annual = annual_df_anom[name].values

        # 应用低通滤波
        low_pass_dict[name] = apply_filter(
            series_annual,
            cutoff_period=cutoff_period_years,
            filter_type='low',
            sample_rate=1.0  # 年度数据
        )

        # 应用高通滤波
        high_pass_dict[name] = apply_filter(
            series_annual,
            cutoff_period=cutoff_period_years,
            filter_type='high',
            sample_rate=1.0
        )

    # --- 3. 可视化滤波结果 ---

    # (A) 绘制低通滤波对比图 (长期趋势)
    low_pass_plot_path = os.path.join(output_dir, "low_pass_filtered_comparison.png")
    visualize_reconstruction_timeseries(
        time_coord=annual_df_anom.index.year,
        save_path=low_pass_plot_path,
        title=f"Low-Pass Filtered (> {cutoff_period_years} years) Anomaly Trends",
        ylabel="Mean DO Anomaly (mol m-3)",
        timeseries_dict=low_pass_dict,
        colors_dict=my_colors,
        is_annual_data=True
    )
    print(f"Low-pass comparison plot saved to {low_pass_plot_path}")

    # (B) 绘制高通滤波对比图 (年际变率)
    high_pass_plot_path = os.path.join(output_dir, "high_pass_filtered_comparison.png")
    visualize_reconstruction_timeseries(
        time_coord=annual_df_anom.index.year,
        save_path=high_pass_plot_path,
        title=f"High-Pass Filtered (< {cutoff_period_years} years) Anomaly Trends",
        ylabel="Mean DO Anomaly (mol m-3)",
        timeseries_dict=high_pass_dict,
        colors_dict=my_colors,
        is_annual_data=True
    )
    print(f"High-pass comparison plot saved to {high_pass_plot_path}")

    # ===================================================================
    #                           保存实验结果
    # ===================================================================
    print("\n保存实验结果：")

    # --- Step Z - 保存所有结果 ---
    # 1. 合并所有要保存的时间序列到一个大字典中
    # 我们用前缀来区分异常和绝对值
    all_results_to_save = {}
    for name, series in other_lines_anom.items():
        all_results_to_save[f"anom_{name}"] = series

    for name, series in other_lines_abs.items():
        all_results_to_save[f"abs_{name}"] = series

    # 使用 "My Reconstruction (Median)" 作为标准键名
    all_results_to_save["anom_My Reconstruction (Median)"] = median_recon_anom
    all_results_to_save["abs_My Reconstruction (Median)"] = median_recon_abs

    # 保存不确定性范围
    all_results_to_save["anom_95% Uncertainty Range_lower"] = lower_bound_anom
    all_results_to_save["anom_95% Uncertainty Range_upper"] = upper_bound_anom
    all_results_to_save["abs_95% Uncertainty Range_lower"] = lower_bound_abs
    all_results_to_save["abs_95% Uncertainty Range_upper"] = upper_bound_abs
    # 2. 调用保存函数
    # results_filepath = os.path.join(output_dir, "7Model_1_1_WOD_1970_2014.parquet")
    # results_filepath = os.path.join(output_dir, "IAP_unbias_LR_1940_2014.parquet")
    results_filepath = os.path.join(output_dir, "WOD_unbias_LR_1940_2014.parquet")

    save_results_to_parquet(filepath=results_filepath,time_coord=data["time_coord"],results_dict=all_results_to_save,config=config)

    print("\nExperiment finished successfully.")


if __name__ == "__main__":
    main()