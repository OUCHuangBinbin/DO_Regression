# src/run_experiment.py (最终修正版)

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
                   comprehensive_evaluation,calculate_vertical_mean,
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
    config_path = "/g12338011ghq/project/hbb/OxygenDiffusion/Oxy_regression/config/congfig_exp_vertical_mean.yaml"
    # config_path = '/g12338011ghq/project/hbb/OxygenDiffusion/Oxy_regression/config/config.yaml'
    # ExpName = 'MPI_1900_2014_1000m'
    # ExpName = 'CanESM5_1900_2014_1000m'
    # ExpName = "CNRM_1900_2014_1000m"
    # ExpName = "ACCESS_1900_2014_1000m"
    ExpName = 'Roll_12M_MPI_1900_2014_1000m'
    # ExpName = 'Roll_60M_CanESM5_1900_2014_1000m'
    # ExpName = 'Roll_60M_CanESM5_2000_2014_1000m'


    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    exp_config = config['experiment']
    output_dir = exp_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 加载和预处理数据 ---
    data = load_and_prepare_data(config)

    # --- 2. 构建训练矩阵 ---
    print("\n--- Training Clean Model (for Median) ---")
    data["inject_uncertainty"] = False
    print(f"   [DEBUG] inject_uncertainty set to: {data.get('inject_uncertainty')}")
    X_clean, Y_clean, g_clean, hash_map_clean = build_training_matrices(data,min_features=5)
    X_grouped, Y_grouped, groups_grouped, hash_to_mask_map = build_training_matrices(data,min_features=5)
    # --- 3. 训练模型 ---
    if 'lambda_config' in exp_config:
        cfg = exp_config['lambda_config']
        lambda_candidates = np.logspace(cfg['log_start'], cfg['log_stop'], num=cfg['num_points'])
    else: lambda_candidates = exp_config.get('lambda_candidates', [0.01, 0.1, 1, 10, 100])

    # 3.1 训练高精度模型 (Clean Model)
    reconstructor_clean = RidgeReconstruction(lambda_candidates=lambda_candidates)
    model_lib_clean = reconstructor_clean.train(X_clean, Y_clean, g_clean,hash_map_clean,output_dir)
    # 3.2 生成最佳估计曲线 (使用 Clean Model)
    median_recon_anom = reconstruct_timeseries(
        model_lib_clean, hash_map_clean,
        data["observation_source_true_da_anom_values"],
        data["obs_mask_da_values"],
        source_name="True Source"
    )
    # 3.2 训练鲁棒模型 (Robust Model)
    print("\n--- Training Robust Model (for Uncertainty) ---")
    data["inject_uncertainty"] = False
    print(f"   [DEBUG] inject_uncertainty set to: {data.get('inject_uncertainty')}")
    reconstructor = RidgeReconstruction(lambda_candidates=lambda_candidates)
    # model_library = reconstructor.train(X_grouped, Y_grouped, groups_grouped, hash_to_mask_map, output_dir)
    print("\n--- Saving trained models and hash map ---")
    model_lib_path = os.path.join(output_dir, f"WOD_{ExpName}.pkl")
    hash_map_path = os.path.join(output_dir, f"WOD_{ExpName}.pkl")
    # model_lib_path = os.path.join(output_dir, "IAP_ridge_model_library.pkl")
    # hash_map_path = os.path.join(output_dir, "IAP_hash_to_mask_map.pkl")
    # model_lib_path = os.path.join(output_dir, "OSD_ridge_model_library.pkl")
    # hash_map_path = os.path.join(output_dir, "OSD_hash_to_mask_map.pkl")
    # model_lib_path = os.path.join(output_dir, "CTD_ridge_model_library.pkl")
    # hash_map_path = os.path.join(output_dir, "CTD_hash_to_mask_map.pkl")
    # model_lib_path = os.path.join(output_dir, "PFL_ridge_model_library.pkl")
    # hash_map_path = os.path.join(output_dir, "PFL_hash_to_mask_map.pkl")
    # model_lib_path = os.path.join(output_dir, "Ship_ridge_model_library.pkl")
    # hash_map_path = os.path.join(output_dir, "Ship_hash_to_mask_map.pkl")
    # with open(model_lib_path, "wb") as f: pickle.dump(model_library, f)
    with open(model_lib_path, "wb") as f: pickle.dump(model_lib_clean, f)
    with open(hash_map_path, "wb") as f: pickle.dump(hash_to_mask_map, f)

    print(f"   -> Models saved to {model_lib_path}")
    print(f"   -> Hash map saved to {hash_map_path}")

    # --- Step 4: 执行重建 ---
    # --- 这里是注入不确定性的集合重建 (Ensemble Reconstruction)： ---
    ensemble_recon_anom_list = []    # 准备一个列表来存储每一次重建的结果
    inject_uncertainty = data.get("inject_uncertainty", False)
    # inject_uncertainty = data.get("inject_uncertainty", True)

    bias_ensemble = data.get("bias_ensemble_values", None)
    if not inject_uncertainty or bias_ensemble is None:
        print("[WARN] No bias ensemble available for reconstruction. Performing a single deterministic run.")
        num_ensemble_runs = 1
    else:
        num_ensemble_runs = bias_ensemble.shape[0]  # 这里我用的是集合数量，也可以自己设置一个循环的值，Sppiel论文里面的不确定性集合是200
    print(f"\n--- [Part X] Starting Ensemble Reconstruction with {num_ensemble_runs} members ---")
    # 1. 获取观测掩码
    obs_mask_aligned_values = data["obs_mask_da_values"]
    n_obs_series = obs_mask_aligned_values.sum(axis=(1, 2))
    # 2. 调用函数计算权重
    # smoothing_window=60 (5年) 是一个不错的起点
    # baseline_coverage=0.10 (10%) 假设这是现代的数据覆盖度
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
        true_obs_anom_values = data["observation_source_true_da_anom_values"]  # 真实观测的异常值
        if inject_uncertainty:
            bias_to_add = data["bias_ensemble_values"][i, :, :, :]
            current_weight_factor = weight_factor[0, :, :, :]

            weighted_bias = bias_to_add * current_weight_factor

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

        # 使用被污染的数据进行重建
        # 4.2 生成不确定性范围 (使用 Robust Model)
        # recon_anom_member = reconstruct_timeseries(
        #     model_library, hash_to_mask_map,
        #     perturbed_obs_anom,
        #     data["obs_mask_da_values"],
        #     source_name=f"Ensemble Member {i + 1}"
        # )
        recon_anom_member = reconstruct_timeseries(
            model_lib_clean, hash_map_clean,
            perturbed_obs_anom,
            data["obs_mask_da_values"],
            source_name=f"Ensemble Member {i + 1}"
        )

        ensemble_recon_anom_list.append(recon_anom_member)

        # 将结果列表转换为一个 (n_members, n_time) 的 Numpy 数组
    ensemble_recon_anom = np.array(ensemble_recon_anom_list)

    # --- 计算统计量 ---
    # 计算中位数作为最佳估计
    # median_recon_anom = np.nanmedian(ensemble_recon_anom, axis=0)
    # 计算 2.5% 和 97.5% 分位数作为不确定性范围
    # lower_bound_anom = np.nanpercentile(ensemble_recon_anom, 2.5, axis=0)
    # upper_bound_anom = np.nanpercentile(ensemble_recon_anom, 97.5, axis=0)

    median_robust = np.nanmedian(ensemble_recon_anom, axis=0)
    lower_robust = np.nanpercentile(ensemble_recon_anom, 2.5, axis=0)
    upper_robust = np.nanpercentile(ensemble_recon_anom, 97.5, axis=0)
    # --- 【核心修正: 策略 A】 ---
    # 计算 Robust Model 的不确定性宽度 (相对于它自己的中位数)
    width_lower = median_robust - lower_robust
    width_upper = upper_robust - median_robust

    # 将这个宽度“嫁接”到 Clean Model 的结果上
    # 这里的 median_recon_anom 是来自 Clean Model 的结果
    final_lower_anom = median_recon_anom - width_lower
    final_upper_anom = median_recon_anom + width_upper

    # 恢复绝对值
    climatology_mean = data["true_obs_climatology_regional_mean"]
    median_recon_abs = median_recon_anom + climatology_mean
    # lower_bound_abs = lower_bound_anom + climatology_mean
    # upper_bound_abs = upper_bound_anom + climatology_mean

    # --- 恢复绝对值 ---
    # 使用 final_lower_anom 和 final_upper_anom 来计算绝对值边界
    lower_bound_abs = final_lower_anom + climatology_mean
    upper_bound_abs = final_upper_anom + climatology_mean
    # median_recon_abs = median_recon_anom + climatology_mean  # 这里的 median_recon_anom 已经是 Clean 的了
    SMOOTH_WINDOW = 12
    lower_bound_smooth = pd.Series(lower_bound_abs).rolling(
        window=SMOOTH_WINDOW, center=True, min_periods=1
    ).mean().values
    upper_bound_smooth = pd.Series(upper_bound_abs).rolling(
        window=SMOOTH_WINDOW, center=True, min_periods=1
    ).mean().values
    lower_anom_smooth = pd.Series(final_lower_anom).rolling(window=SMOOTH_WINDOW, center=True,
                                                            min_periods=1).mean().values
    upper_anom_smooth = pd.Series(final_upper_anom).rolling(window=SMOOTH_WINDOW, center=True,
                                                            min_periods=1).mean().values
    # --- 4. 执行所有重建任务 ---
    recon_model_anom = reconstruct_timeseries(
        model_lib_clean, hash_map_clean,
        data["observation_source_model_da_anom_values"],
        data["obs_mask_da_values"],
        source_name="Model Source"
    )
    # recon_true_anom = reconstruct_timeseries(
    #     model_library, hash_to_mask_map,
    #     data["observation_source_true_da_anom_values"],
    #     data["obs_mask_da_values"],
    #     source_name="True Obs Source"
    # )
    recon_true_abs = median_recon_anom + data["true_obs_climatology_regional_mean"]

    print("\n--- [Step 9] Loading and processing reference products for comparison ---")

    reference_timeseries_anom = {}
    reference_timeseries_abs = {}

    # 在这里绘制其他产品：
    # 把溶解氧国际小组的产品加入进去

    if 'reference_products' in config:
        print("\n--- [Step 9] Loading and processing reference products for comparison (0-100m Integrated) ---")
        # 获取用于缩放的目标统计量 (来自训练集)
        # train_anom_std = data["train_anom_std"]
        time_slice = slice(f"{config['data']['time_range'][0]}-01-01", f"{config['data']['time_range'][1]}-12-31")
        y_cfg = config['data']['target_y_config']
        target_depth_range = y_cfg['depth_range']

        for name, info in config['reference_products'].items():
                print(f"\n   -> Processing reference: {name}")
                try:
                    # 手动加载以确保控制权
                    with xr.open_dataset(info['path'], decode_times=False) as ds:
                        ds = xr.decode_cf(ds, use_cftime=True)
                        if isinstance(ds.time.to_index(), xr.coding.cftimeindex.CFTimeIndex):
                            ds['time'] = ds.indexes['time'].to_datetimeindex(unsafe=True)

                        # 提取变量
                        da = ds[info['var_name']]

                        # 时间切片
                        da = da.sel(time=time_slice)
                        if da.time.size == 0:
                            print(f"     [WARN] No data in time range for {name}. Skipping.")
                            continue

                        depth_dim = next((d for d in ['lev', 'olevel', 'depth', 'pressure'] if d in da.dims), None)

                        if depth_dim:
                            print(f"     ... Performing vertical integration ({target_depth_range}m) on '{depth_dim}'")
                            ref_da_processed = calculate_vertical_mean(da, target_depth_range)
                        else:
                            print(
                                f"     [WARN] No depth dimension found for {name}. Assuming it is already depth-averaged or surface.")
                            ref_da_processed = da

                except Exception as e:
                    print(f"     [ERROR] Failed to load {name}: {e}")
                    continue

                if ref_da_processed is not None:
                    # 1. 单位转换
                    conversion_factor = info.get('unit_conversion_factor', 1.0)
                    ref_da_converted = ref_da_processed * conversion_factor

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

                    # 3. 计算区域平均
                    ref_anom_ts = calculate_weighted_mean(ref_anom_aligned).values
                    ref_abs_ts = calculate_weighted_mean(ref_abs_aligned).values

                    # 4. 检查并添加到字典
                    if ref_anom_ts.ndim == 1 and ref_abs_ts.ndim == 1:
                        reference_timeseries_anom[name] = ref_anom_ts
                        reference_timeseries_abs[name] = ref_abs_ts
                    else:
                        print(f"     ... WARNING: Processed reference '{name}' is not 1-dimensional. Skipping.")

        # --- Step X: 量化评估重建结果 ---
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
        metrics_df = comprehensive_evaluation(
            time_coord=data["time_coord"],
            y_true_raw=data["Y_obs_source_truth_anom_values"][eval_mask],
            y_pred_raw=recon_model_anom[eval_mask],
            output_dir=output_dir,
            var_name="Dissolved Oxygen"
        )
        # print(metrics_df)
    # --- 6. 清晰的可视化流程 ---
    print("\n--- [Step 6] Visualizing all results ---")

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

    # # (A) 综合异常对比图
    anomaly_plot_dict = {
        "CMIP6 Source (Truth)": data["Y_obs_source_truth_anom_values"],
        "Recon (from Model Source)": recon_model_anom,
        # "Recon (from True Obs)": recon_true_anom,
        "Recon(from True Obs": median_recon_anom,
    }
    anomaly_plot_dict.update(reference_timeseries_anom)

    # anomaly_plot_path = os.path.join(output_dir, "anomaly_comparison_all.png")
    # visualize_reconstruction_timeseries(
    #     time_coord=data["time_coord"],
    #     save_path=anomaly_plot_path,
    #     title=title_anom,
    #     ylabel="Mean DO Anomaly (mol m-3)",
    #     timeseries_dict=anomaly_plot_dict,
    #     colors_dict=my_colors
    # )
    # print(f"Comprehensive anomaly plot saved to {anomaly_plot_path}")
    #
    # # (B) 最终绝对值对比图
    #
    # 1. 恢复CMIP6真值的绝对值
    cmip6_truth_abs = data["Y_obs_source_truth_anom_values"] + data["obs_model_climatology_regional_mean"]
    # 2. 恢复模型源重建的绝对值
    recon_model_abs = recon_model_anom + data["obs_model_climatology_regional_mean"]
    # absolute_plot_dict = {
    #     "My Reconstruction": recon_true_abs,
    #     "CMIP6 Source (Truth)": cmip6_truth_abs,
    #     "Recon (from Model Source)": recon_model_abs,
    # }
    # # 将参考产品的绝对值序列添加进来
    # absolute_plot_dict.update(reference_timeseries_abs)
    #
    # final_plot_path = os.path.join(output_dir, "final_absolute_comparison.png")
    # visualize_reconstruction_timeseries(
    #     time_coord=data["time_coord"],
    #     save_path=final_plot_path,
    #     title="Final Reconstruction vs. Reference Products (Absolute)",
    #     ylabel="Absolute Mean DO (mol m-3)",
    #     timeseries_dict=absolute_plot_dict,
    #     colors_dict=my_colors
    # )
    # print(f"Final absolute plot saved to {final_plot_path}")
    # print("\nExperiment finished successfully.")

    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- ---  分割线  --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
    # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

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
        lower_bound=lower_anom_smooth,
        upper_bound=upper_anom_smooth,
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
        lower_bound=lower_bound_smooth,
        upper_bound=upper_bound_smooth,
        other_lines_dict=other_lines_abs,
        colors_dict=my_colors
    )

    # print("\nExperiment finished successfully.")


    # # --- : Step Z - 时间尺度分离分析与可视化 ---
    # print("\n" + "=" * 80)
    # print("--- VISUALIZING TIME-SCALE DECOMPOSITION ---")
    # print("=" * 80)
    #
    # # --- 准备年度数据 ---
    # # 我们将在年平均数据上进行滤波，因为20年的周期对于月度数据来说太长
    #
    # # 1. 将所有异常序列转换为年度DataFrame
    # df_anom = pd.DataFrame(anomaly_plot_dict, index=pd.to_datetime(data["time_coord"]))
    # annual_df_anom = df_anom.resample('YE').mean()
    #
    # # 2. 对每一列（每个产品）都应用低通和高通滤波
    # cutoff_period_years = 20.0
    #
    # low_pass_dict = {}
    # high_pass_dict = {}
    #
    # for name in annual_df_anom.columns:
    #     series_annual = annual_df_anom[name].values
    #
    #     # 应用低通滤波
    #     low_pass_dict[name] = apply_filter(
    #         series_annual,
    #         cutoff_period=cutoff_period_years,
    #         filter_type='low',
    #         sample_rate=1.0  # 年度数据
    #     )
    #
    #     # 应用高通滤波
    #     high_pass_dict[name] = apply_filter(
    #         series_annual,
    #         cutoff_period=cutoff_period_years,
    #         filter_type='high',
    #         sample_rate=1.0
    #     )
    #
    # # --- 3. 可视化滤波结果 ---
    #
    # # (A) 绘制低通滤波对比图 (长期趋势)
    # low_pass_plot_path = os.path.join(output_dir, "low_pass_filtered_comparison.png")
    # visualize_reconstruction_timeseries(
    #     time_coord=annual_df_anom.index.year,  # X轴现在是年份
    #     save_path=low_pass_plot_path,
    #     title=f"Low-Pass Filtered (> {cutoff_period_years} years) Anomaly Trends",
    #     ylabel="Mean DO Anomaly (mol m-3)",
    #     timeseries_dict=low_pass_dict,
    #     colors_dict=my_colors,
    #     is_annual_data=True  # 告诉绘图函数这是年度数据
    # )
    # print(f"Low-pass comparison plot saved to {low_pass_plot_path}")
    #
    # # (B) 绘制高通滤波对比图 (年际变率)
    # high_pass_plot_path = os.path.join(output_dir, "high_pass_filtered_comparison.png")
    # visualize_reconstruction_timeseries(
    #     time_coord=annual_df_anom.index.year,
    #     save_path=high_pass_plot_path,
    #     title=f"High-Pass Filtered (< {cutoff_period_years} years) Anomaly Trends",
    #     ylabel="Mean DO Anomaly (mol m-3)",
    #     timeseries_dict=high_pass_dict,
    #     colors_dict=my_colors,
    #     is_annual_data=True
    # )
    # print(f"High-pass comparison plot saved to {high_pass_plot_path}")

    print("\n保存实验结果：")

    # --- Step Z - 保存所有结果 ---
    # 1. 合并所有要保存的时间序列到一个大字典中
    # 我们用前缀来区分异常和绝对值
    all_results_to_save = {}

    for name, series in other_lines_anom.items():
        all_results_to_save[f"anom_{name}"] = series

    for name, series in other_lines_abs.items():
        all_results_to_save[f"abs_{name}"] = series

    # 我们使用 "My Reconstruction (Median)" 作为标准键名
    all_results_to_save["anom_My Reconstruction (Median)"] = median_recon_anom
    all_results_to_save["abs_My Reconstruction (Median)"] = median_recon_abs

    # 【强烈建议】: 也保存不确定性范围！
    # all_results_to_save["anom_95% Uncertainty Range_lower"] = lower_bound_anom
    # all_results_to_save["anom_95% Uncertainty Range_upper"] = upper_bound_anom
    all_results_to_save["anom_95% Uncertainty Range_lower"] = final_lower_anom
    all_results_to_save["anom_95% Uncertainty Range_upper"] = final_upper_anom
    all_results_to_save["abs_95% Uncertainty Range_lower"] = lower_bound_abs
    all_results_to_save["abs_95% Uncertainty Range_upper"] = upper_bound_abs
    # 2. 调用保存函数
    results_filepath = os.path.join(output_dir, f"WOD_{ExpName}.parquet")
    # results_filepath = os.path.join(output_dir, "7Model_1_1_WOD_1970_2014.parquet")
    # results_filepath = os.path.join(output_dir, "experiment_results_bias.parquet")
    # results_filepath = os.path.join(output_dir, "WOD_bias_LR_1900_2014.parquet")
    # results_filepath = os.path.join(output_dir, "IAP_bias_LR_1940_2014.parquet")
    # results_filepath = os.path.join(output_dir, "US_LR_1900_2014.parquet")
    # results_filepath = os.path.join(output_dir, "SU_LR_1900_2014.parquet")
    # results_filepath = os.path.join(output_dir, "JP_LR_1900_2014.parquet")
    # results_filepath = os.path.join(output_dir, "DE_LR_1900_2014.parquet")
    # results_filepath = os.path.join(output_dir, "UK_LR_1900_2014.parquet")
    # results_filepath = os.path.join(output_dir, "OSD_bias_LR_1900_2014.parquet")
    # results_filepath = os.path.join(output_dir, "CTD_bias_LR_1940_2014.parquet")
    # results_filepath = os.path.join(output_dir, "PFL_bias_LR_2000_2014.parquet")
    # results_filepath = os.path.join(output_dir, "Ship_bias_LR_2000_2014.parquet")




    save_results_to_parquet(
        filepath=results_filepath,
        time_coord=data["time_coord"],
        results_dict=all_results_to_save,
        config=config  # 传入完整的配置字典
    )

    print("\nExperiment finished successfully.")


if __name__ == "__main__":
    main()